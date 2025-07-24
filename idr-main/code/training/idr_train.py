import os
from datetime import datetime
from pyhocon import ConfigFactory
import torch
import utils.general as utils
import utils.plots as plt

class IDRTrainRunner():
    def __init__(self, **kwargs):
        torch.set_default_dtype(torch.float32)
        torch.set_num_threads(1)

        self.conf = ConfigFactory.parse_file(kwargs['conf'])
        
        # 保证 batch_size 为正整数
        self.batch_size = max(1, kwargs.get('batch_size', 1) // 2)
        self.nepochs = kwargs.get('nepochs', 2000)
        self.exps_folder_name = kwargs.get('exps_folder_name', 'exps')
        self.train_cameras = kwargs.get('train_cameras', False)

        # 选择设备，优先使用 MPS（Apple GPU），否则使用 CPU
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using MPS backend")
        else:
            self.device = torch.device("cpu")
            print("MPS not available, falling back to CPU")

        self.expname = self.conf.get_string('train.expname') + kwargs.get('expname', '')
        scan_id = kwargs.get('scan_id', -1)
        if scan_id != -1:
            self.expname = f"{self.expname}_{scan_id}"

        # 创建实验和检查点文件夹
        utils.mkdir_ifnotexists(os.path.join('../', self.exps_folder_name))
        self.expdir = os.path.join('../', self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        print('Loading data ...')
        dataset_conf = self.conf.get_config('dataset')
        if scan_id != -1:
            dataset_conf['scan_id'] = scan_id

        # 加载训练数据集
        self.train_dataset = utils.get_class(self.conf.get_string('train.dataset_class'))(self.train_cameras, **dataset_conf)
        print('Finish loading data ...')

        # 使用设置的 batch_size 加载 DataLoader
        self.train_dataloader = torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.train_dataset.collate_fn
        )

        self.model = utils.get_class(self.conf.get_string('train.model_class'))(conf=self.conf.get_config('model'))
        self.model.to(self.device)

        self.loss = utils.get_class(self.conf.get_string('train.loss_class'))(**self.conf.get_config('loss'))

        self.lr = self.conf.get_float('train.learning_rate')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            self.conf.get_list('train.sched_milestones', default=[]),
            gamma=self.conf.get_float('train.sched_factor', default=0.0)
        )

        self.start_epoch = 0

        # 检查是否有指定检查点进行恢复
        if kwargs.get('is_continue', False):
            checkpoint_files = sorted([f for f in os.listdir(self.checkpoints_path) if f.startswith("model_epoch_")])
            if checkpoint_files:
                checkpoint = kwargs.get('checkpoint', 'latest')
                if checkpoint == 'latest':
                    latest_checkpoint = checkpoint_files[-1]
                else:
                    latest_checkpoint = f"model_epoch_{checkpoint}.pth"

                checkpoint_path = os.path.join(self.checkpoints_path, latest_checkpoint)
                checkpoint_data = torch.load(checkpoint_path, map_location=self.device)
                self.model.load_state_dict(checkpoint_data['model_state_dict'])
                self.start_epoch = checkpoint_data['epoch'] + 1
                print(f"Resumed training from epoch {self.start_epoch}")

    def save_checkpoints(self, epoch):
        torch.save({"epoch": epoch, "model_state_dict": self.model.state_dict()},
                   os.path.join(self.checkpoints_path, f"model_epoch_{epoch}.pth"))

    def run(self):
        print("Starting training...")

        for epoch in range(self.start_epoch, self.nepochs + 1):
            # 每个 epoch 保存检查点
            self.save_checkpoints(epoch)

            for data_index, (indices, model_input, ground_truth) in enumerate(self.train_dataloader):
                # 分批处理输入数据，降低显存占用
                model_input["intrinsics"] = model_input["intrinsics"].to(self.device)
                model_input["uv"] = model_input["uv"].to(self.device)
                model_input["object_mask"] = model_input["object_mask"].to(self.device)

                # 分批次执行模型的前向传递
                outputs = []
                chunks = torch.split(model_input["uv"], 1000, dim=0)  # 按需调整分块大小
                for chunk in chunks:
                    output_chunk = self.model({"uv": chunk, **{k: v for k, v in model_input.items() if k != "uv"}})
                    outputs.append(output_chunk)
                model_outputs = torch.cat(outputs, dim=0)

                loss_output = self.loss(model_outputs, ground_truth)

                loss = loss_output['loss']
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.scheduler.step()
