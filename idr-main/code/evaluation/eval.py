import sys
sys.path.append('../code')
import argparse
import os
from pyhocon import ConfigFactory
import torch
import numpy as np
import cvxpy as cp
from PIL import Image
import math

import utils.general as utils
import utils.plots as plt
from utils import rend_util

import numpy as np

def calculate_psnr(img1, img2, mask=None):
    """
    计算两个图像之间的 PSNR 值，img1 和 img2 的像素值应在 [0, 1] 范围内。
    mask 可选，用于指定图像的有效区域。
    """
    if mask is not None:
        # 如果 mask 是 2D (H, W)，扩展到 (H, W, 3) 以匹配图像的维度
        if mask.shape[-1] == 1:
            mask = np.repeat(mask, 3, axis=-1)
        mse = np.mean(((img1 - img2) ** 2)[mask > 0])
    else:
        mse = np.mean((img1 - img2) ** 2)
    
    if mse == 0:
        return float('inf')  # 当 MSE 为 0 时，图像完全一致，PSNR 返回无穷大
    
    max_value = 1.0  # 假设图像的像素值在 [0, 1] 范围内
    psnr = 10 * np.log10(max_value ** 2 / mse)
    
    return psnr
    
def evaluate(**kwargs):
    torch.set_default_dtype(torch.float32)

    conf = ConfigFactory.parse_file(kwargs['conf'])
    exps_folder_name = kwargs['exps_folder_name']
    evals_folder_name = kwargs['evals_folder_name']
    eval_cameras = kwargs['eval_cameras']
    eval_rendering = kwargs['eval_rendering']

    expname = conf.get_string('train.expname')
    if kwargs['expname']:
        expname += kwargs['expname']

    if kwargs['timestamp'] == 'latest':
        exp_path = os.path.join('../', kwargs['exps_folder_name'], expname)
        if os.path.exists(exp_path):
            timestamps = os.listdir(exp_path)
            print("Found folders:", timestamps)  # Print out the folders found
            if len(timestamps) == 0:
                print('WRONG EXP FOLDER: No timestamp folders found in', exp_path)
                exit()
            else:
                timestamp = sorted(timestamps)[-1]
                print("Using latest timestamp:", timestamp)  # Print the chosen timestamp
        else:
            print('WRONG EXP FOLDER: Experiment folder does not exist:', exp_path)
            exit()
    else:
        timestamp = kwargs['timestamp']

    utils.mkdir_ifnotexists(os.path.join('../', evals_folder_name))
    expdir = os.path.join('../', exps_folder_name, expname)
    evaldir = os.path.join('../', evals_folder_name, expname)
    utils.mkdir_ifnotexists(evaldir)

    # 选择设备：优先使用 MPS，如果不可用则回退到 CPU
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    model = utils.get_class(conf.get_string('train.model_class'))(conf=conf.get_config('model'))
    model.to(device)

    dataset_conf = conf.get_config('dataset')
    if kwargs['scan_id'] != -1:
        dataset_conf['scan_id'] = kwargs['scan_id']
    eval_dataset = utils.get_class(conf.get_string('train.dataset_class'))(eval_cameras, **dataset_conf)
    print(f"eval_dataset size: {len(eval_dataset)}")
    # settings for camera optimization
    scale_mat = eval_dataset.get_scale_mat()
    if eval_cameras:
        num_images = len(eval_dataset)
        pose_vecs = torch.nn.Embedding(num_images, 7, sparse=True).to(device)
        pose_vecs.weight.data.copy_(eval_dataset.get_pose_init())

        gt_pose = eval_dataset.get_gt_pose()

    if eval_rendering:
        print("Entering rendering evaluation...")
        
        eval_dataloader = torch.utils.data.DataLoader(eval_dataset,
                                                    batch_size=1,
                                                    shuffle=False)
        for batch in eval_dataloader:
            print(f"First batch: {batch}")
            break
        if len(eval_dataloader) == 0:
            print("No data loaded into eval_dataloader.")
        else:
            print(f"Loaded {len(eval_dataloader)} batches for evaluation.")
            
        total_pixels = eval_dataset.total_pixels
        img_res = eval_dataset.img_res

    old_checkpnts_dir = os.path.join(expdir, 'checkpoints')  # 正确的拼接方式
    print("Loading model from:", os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"))

    saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"), map_location=device)
    model.load_state_dict(saved_model_state["model_state_dict"])
    epoch = saved_model_state['epoch']

    if eval_cameras:
        data = torch.load(os.path.join(old_checkpnts_dir, 'CamParameters', str(kwargs['checkpoint']) + ".pth"), map_location=device)
        pose_vecs.load_state_dict(data["pose_vecs_state_dict"])

    ####################################################################################################################
    print("evaluating...")

    model.eval()
    if eval_cameras:
        pose_vecs.eval()

    with torch.no_grad():
        if eval_cameras:
            gt_Rs = gt_pose[:, :3, :3].double()
            gt_ts = gt_pose[:, :3, 3].double()

            pred_Rs = rend_util.quat_to_rot(pose_vecs.weight.data[:, :4]).cpu().double()
            pred_ts = pose_vecs.weight.data[:, 4:].cpu().double()

            R_opt, t_opt, c_opt, R_fixed, t_fixed = get_cameras_accuracy(pred_Rs, gt_Rs, pred_ts, gt_ts)

            cams_transformation = np.eye(4, dtype=np.double)
            cams_transformation[:3,:3] = c_opt * R_opt
            cams_transformation[:3,3] = t_opt

        mesh = plt.get_surface_trace(
            path=evaldir,  # 传入 evaldir 作为保存路径
            epoch=epoch,  # 当前的 epoch
            sdf=lambda x: model.implicit_network(x)[:, 0],  # 使用 SDF 函数生成表面
            resolution=kwargs['resolution'],  # 网格分辨率
            return_mesh=True  # 返回生成的 mesh 对象
        )
        # Transform to world coordinates
        if eval_cameras:
            mesh.apply_transform(cams_transformation)
        else:
            mesh.apply_transform(scale_mat)

        # Taking the biggest connected component
        components = mesh.split(only_watertight=False)
        areas = np.array([c.area for c in components], dtype=float)
        mesh_clean = components[areas.argmax()]
        mesh_clean.export('{0}/surface_world_coordinates_{1}.ply'.format(evaldir, epoch), 'ply')

    if eval_rendering:
        images_dir = '{0}/rendering'.format(evaldir)
        utils.mkdir_ifnotexists(images_dir)

        psnrs = []
        for data_index, (indices, model_input, ground_truth) in enumerate(eval_dataloader):
            model_input["intrinsics"] = model_input["intrinsics"].to(device)
            model_input["uv"] = model_input["uv"].to(device)
            model_input["object_mask"] = model_input["object_mask"].to(device)

            if eval_cameras:
                pose_input = pose_vecs(indices.to(device))
                model_input['pose'] = pose_input
            else:
                model_input['pose'] = model_input['pose'].to(device)

            split = utils.split_input(model_input, total_pixels, device)
            res = []
            for s in split:
                out = model(s)
                res.append({
                    'rgb_values': out['rgb_values'].detach(),
                })

            batch_size = ground_truth['rgb'].shape[0]
            model_outputs = utils.merge_output(res, total_pixels, batch_size, device=device)
            rgb_eval = model_outputs['rgb_values']
            rgb_eval = rgb_eval.reshape(batch_size, total_pixels, 3)

            rgb_eval = (rgb_eval + 1.) / 2.
            rgb_eval = plt.lin2img(rgb_eval, img_res).detach().cpu().numpy()[0]
            rgb_eval = rgb_eval.transpose(1, 2, 0)
            img = Image.fromarray((rgb_eval * 255).astype(np.uint8))
            img.save('{0}/eval_{1}.png'.format(images_dir,'%03d' % indices[0]))

            rgb_gt = ground_truth['rgb']
            rgb_gt = (rgb_gt + 1.) / 2.
            rgb_gt = plt.lin2img(rgb_gt, img_res).numpy()[0]
            rgb_gt = rgb_gt.transpose(1, 2, 0)

            mask = model_input['object_mask']
            mask = plt.lin2img(mask.unsqueeze(-1), img_res).cpu().numpy()[0]
            mask = mask.transpose(1, 2, 0)
            if np.sum(mask) > 0:  # 确保 mask 中有非 0 值
                print(f"Mask sum for batch {data_index}: {np.sum(mask)}")
                print(f"Mask shape: {mask.shape}")
                rgb_eval_masked = rgb_eval * mask
                rgb_gt_masked = rgb_gt * mask
                print(f"rgb_eval_masked min: {np.min(rgb_eval_masked)}, max: {np.max(rgb_eval_masked)}")
                print(f"rgb_gt_masked min: {np.min(rgb_gt_masked)}, max: {np.max(rgb_gt_masked)}")
    
            # 计算 PSNR
                psnr = calculate_psnr(rgb_eval_masked, rgb_gt_masked, mask)
                print(f"PSNR for batch {data_index}: {psnr}")
                psnrs.append(psnr)
            else:
                print(f"Mask is all zeros for batch {data_index}, skipping PSNR calculation.")


        psnrs = np.array(psnrs).astype(np.float64)
        print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))
        if psnrs.size == 0:
            print(f"PSNR array is empty for scan_id {scan_id}.")
        elif np.isnan(psnrs).all():
            print(f"PSNR array contains only NaN values for scan_id {scan_id}.")
        else:
            print("RENDERING EVALUATION {2}: psnr mean = {0} ; psnr std = {1}".format("%.2f" % psnrs.mean(), "%.2f" % psnrs.std(), scan_id))


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, default='./confs/dtu_fixed_cameras.conf')
    parser.add_argument('--expname', type=str, default='', help='The experiment name to be evaluated.')
    parser.add_argument('--exps_folder', type=str, default='exps', help='The experiments folder name.')
    parser.add_argument('--timestamp', default='latest', type=str, help='The experiment timestamp to test.')
    parser.add_argument('--checkpoint', default='latest', type=str, help='The trained model checkpoint to test')
    parser.add_argument('--scan_id', type=int, default=-1, help='If set, taken to be the scan id.')
    parser.add_argument('--resolution', default=512, type=int, help='Grid resolution for marching cube')
    parser.add_argument('--is_uniform_grid', default=False, action="store_true", help='If set, evaluate marching cube with uniform grid.')
    parser.add_argument('--eval_cameras', default=False, action="store_true", help='If set, evaluate camera accuracy of trained cameras.')
    parser.add_argument('--eval_rendering', default=False, action="store_true", help='If set, evaluate rendering quality.')

    opt = parser.parse_args()
    
    scan_id=65
    
    
    evaluate(conf=opt.conf,
             expname=opt.expname,
             exps_folder_name=opt.exps_folder,
             evals_folder_name='evals',
             timestamp=opt.timestamp,
             checkpoint=opt.checkpoint,
             scan_id = opt.scan_id if opt.scan_id != -1 else conf.get_int('dataset.scan_id', default=-1),
             resolution=opt.resolution,
             eval_cameras=opt.eval_cameras,
             eval_rendering=opt.eval_rendering
             )
