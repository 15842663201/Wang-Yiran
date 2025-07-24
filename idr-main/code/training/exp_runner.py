import sys
sys.path.append('../code')
import argparse
import torch

sys.path.append('/Users/vibo/Desktop/idr/code/utils')  # 确保这个路径是rend_util模块所在的路径
import rend_util

from training.idr_train import IDRTrainRunner

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='')
    parser.add_argument('--gpu', type=str, default='auto', help='Device to use [default: auto]')
    parser.add_argument('--is_continue', default=False, action="store_true", help='If set, indicates continuing from a previous run.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The timestamp of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The checkpoint epoch number of the run to be used in case of continuing from a previous run.')
    parser.add_argument('--train_cameras', default=False, action="store_true", help='If set, optimizing also camera location.')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')

    opt = parser.parse_args()

    # Automatically select MPS if available, otherwise CPU
    if opt.gpu == "auto":
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS backend")
        else:
            device = torch.device("cpu")
            print("MPS not available, using CPU")
    else:
        device = torch.device(opt.gpu)

    # Pass the selected device to the training runner
    trainrunner = IDRTrainRunner(conf=opt.conf,
                                 batch_size=opt.batch_size,
                                 nepochs=opt.nepoch,
                                 expname=opt.expname,
                                 gpu_index=device,  # Pass the selected device
                                 exps_folder_name='exps',
                                 is_continue=opt.is_continue,
                                 timestamp=opt.timestamp,
                                 checkpoint=opt.checkpoint,
                                 scan_id=opt.scan_id,
                                 train_cameras=opt.train_cameras
                                 )

    trainrunner.run()
