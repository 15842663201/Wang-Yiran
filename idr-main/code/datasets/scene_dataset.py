import os
import torch
import numpy as np
import sys
import utils.general as utils
sys.path.append('/Users/vibo/Desktop/idr/code/utils')  # 使用实际的绝对路径
from utils import rend_util


class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 train_cameras,
                 data_dir,
                 img_res,
                 scan_id=0,
                 cam_file=None
                 ):
        # 修改路径为 scan65 目录
        self.instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))

        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res

        assert os.path.exists(self.instance_dir), "Data directory is empty"

        self.sampling_idx = None
        self.train_cameras = train_cameras

        # 从 image 和 mask 文件夹加载图像和遮罩
        image_dir = os.path.join(self.instance_dir, 'image')
        mask_dir = os.path.join(self.instance_dir, 'mask')

        # 使用 utils.glob_imgs 函数从正确的目录中加载图像和遮罩
        image_paths = sorted(utils.glob_imgs(image_dir))  # 从 image 文件夹加载图像
        mask_paths = sorted(utils.glob_imgs(mask_dir))   # 从 mask 文件夹加载遮罩

        # 输出调试信息
        print(f"Found {len(image_paths)} images and {len(mask_paths)} masks.")

        self.n_images = len(image_paths)

        self.cam_file = os.path.join(self.instance_dir, 'cameras.npz')
        if cam_file is not None:
            self.cam_file = os.path.join(self.instance_dir, cam_file)

        # 检查并加载相机参数
        print(f"Loading camera parameters from {self.cam_file}")
        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.intrinsics_all = []
        self.pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.intrinsics_all.append(torch.from_numpy(intrinsics).float())
            self.pose_all.append(torch.from_numpy(pose).float())

        # 加载图像
        self.rgb_images = []
        for path in image_paths:
            rgb = rend_util.load_rgb(path)
            rgb = rgb.reshape(3, -1).transpose(1, 0)
            self.rgb_images.append(torch.from_numpy(rgb).float())

        # 加载遮罩
        self.object_masks = []
        for path in mask_paths:
            object_mask = rend_util.load_mask(path)
            object_mask = object_mask.reshape(-1)
            self.object_masks.append(torch.from_numpy(object_mask).bool())

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        uv = uv.reshape(2, -1).transpose(1, 0)

        sample = {
            "object_mask": self.object_masks[idx],
            "uv": uv,
            "intrinsics": self.intrinsics_all[idx],
        }

        ground_truth = {
            "rgb": self.rgb_images[idx]
        }

        if self.sampling_idx is not None:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["uv"] = uv[self.sampling_idx, :]

        if not self.train_cameras:
            sample["pose"] = self.pose_all[idx]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):
        # 获取列表中的字典并返回 input，ground_truth 字典
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # 组合它们到一个新的字典
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def change_sampling_idx(self, sampling_size):
        if sampling_size == -1:
            self.sampling_idx = None
        else:
            self.sampling_idx = torch.randperm(self.total_pixels)[:sampling_size]

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

def get_pose_init(self):
    # 使用 MPS 设备（如果可用）
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 从文件中读取相机的线性初始化
    cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
    camera_dict = np.load(cam_file)
    scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
    world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

    init_pose = []
    for scale_mat, world_mat in zip(scale_mats, world_mats):
        P = world_mat @ scale_mat
        P = P[:3, :4]
        _, pose = rend_util.load_K_Rt_from_P(None, P)
        init_pose.append(pose)
    
    # 将初始化的姿态移动到 MPS 设备
    init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).to(device)
    
    # 将旋转矩阵转换为四元数
    init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
    init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

    return init_quat
