import os
from glob import glob
import torch
import plotly.graph_objs as go
import plotly.offline as offline
import numpy as np
from skimage import measure
import trimesh
import torchvision
from PIL import Image
from utils import rend_util

def plot(model, indices, model_outputs, pose, rgb_gt, path, epoch, img_res, plot_nimgs, max_depth, resolution, device):
    # arrange data to plot
    batch_size, num_samples, _ = rgb_gt.shape

    network_object_mask = model_outputs['network_object_mask']
    points = model_outputs['points'].reshape(batch_size, num_samples, 3)
    rgb_eval = model_outputs['rgb_values']
    rgb_eval = rgb_eval.reshape(batch_size, num_samples, 3)
    print(f"rgb_eval min: {np.min(rgb_eval)}, max: {np.max(rgb_eval)}")
    print(f"rgb_gt min: {np.min(rgb_gt)}, max: {np.max(rgb_gt)}")

    depth = torch.ones(batch_size * num_samples).to(device).float() * max_depth
    depth[network_object_mask] = rend_util.get_depth(points, pose).reshape(-1)[network_object_mask]
    depth = depth.reshape(batch_size, num_samples, 1)
    network_object_mask = network_object_mask.reshape(batch_size, -1)

    cam_loc, cam_dir = rend_util.get_camera_for_plot(pose)

    # plot rendered images
    plot_images(rgb_eval, rgb_gt, path, epoch, plot_nimgs, img_res)

    # plot depth maps
    plot_depth_maps(depth, path, epoch, plot_nimgs, img_res)

    data = []

    # plot surface
    surface_traces = get_surface_trace(path=path,
                                       epoch=epoch,
                                       sdf=lambda x: model.implicit_network(x)[:, 0],
                                       resolution=resolution
                                       )
    data.append(surface_traces[0])

    # plot cameras locations
    for i, loc, dir in zip(indices, cam_loc, cam_dir):
        data.append(get_3D_quiver_trace(loc.unsqueeze(0), dir.unsqueeze(0), name='camera_{0}'.format(i)))

    for i, p, m in zip(indices, points, network_object_mask):
        p = p[m]
        sampling_idx = torch.randperm(p.shape[0])[:2048]
        p = p[sampling_idx, :]

        val = model.implicit_network(p)
        caption = ["sdf: {0} ".format(v[0].item()) for v in val]

        data.append(get_3D_scatter_trace(p, name='intersection_points_{0}'.format(i), caption=caption))

    fig = go.Figure(data=data)
    scene_dict = dict(xaxis=dict(range=[-3, 3], autorange=False),
                      yaxis=dict(range=[-3, 3], autorange=False),
                      zaxis=dict(range=[-3, 3], autorange=False),
                      aspectratio=dict(x=1, y=1, z=1))
    fig.update_layout(scene=scene_dict, width=1400, height=1400, showlegend=True)
    filename = '{0}/surface_{1}.html'.format(path, epoch)
    offline.plot(fig, filename=filename, auto_open=False)

def get_3D_scatter_trace(points, name='', size=3, caption=None):
    assert points.shape[1] == 3, "3d scatter plot input points are not correctly shaped"
    assert len(points.shape) == 2, "3d scatter plot input points are not correctly shaped"

    trace = go.Scatter3d(
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        mode='markers',
        name=name,
        marker=dict(
            size=size,
            line=dict(
                width=2,
            ),
            opacity=1.0,
        ), text=caption)

    return trace

def get_3D_quiver_trace(points, directions, color='#bd1540', name=''):
    assert points.shape[1] == 3, "3d cone plot input points are not correctly shaped"
    assert len(points.shape) == 2, "3d cone plot input points are not correctly shaped"
    assert directions.shape[1] == 3, "3d cone plot input directions are not correctly shaped"
    assert len(directions.shape) == 2, "3d cone plot input directions are not correctly shaped"

    trace = go.Cone(
        name=name,
        x=points[:, 0].cpu(),
        y=points[:, 1].cpu(),
        z=points[:, 2].cpu(),
        u=directions[:, 0].cpu(),
        v=directions[:, 1].cpu(),
        w=directions[:, 2].cpu(),
        sizemode='absolute',
        sizeref=0.125,
        showscale=False,
        colorscale=[[0, color], [1, color]],
        anchor="tail"
    )

    return trace

def get_surface_trace(path, epoch, sdf, resolution=100, return_mesh=False):
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(sdf(pnts).detach().cpu().numpy())
        print("SDF values for chunk: ", z[-1][:10])  # 打印部分SDF值进行检查
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > 0 or np.max(z) < 0)):

        z = z.astype(np.float32)
        print("z shape: " + str(z.shape))
        print("grid['xyz'][0].shape: " + str(grid['xyz'][0].shape))
        print("grid['xyz'][1].shape: " + str(grid['xyz'][1].shape))
        print("grid['xyz'][2].shape: " + str(grid['xyz'][2].shape))
# Ensure that the shape of z matches the grid
        expected_size = grid['xyz'][0].shape[0] * grid['xyz'][1].shape[0] * grid['xyz'][2].shape[0]

        if z.size == expected_size:
            volume = z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0], grid['xyz'][2].shape[0]).transpose([1, 0, 2])
        else:
            raise ValueError("Shape mismatch detected: z size {}, expected size {}".format(z.size, expected_size))

        verts, faces, normals, values = measure.marching_cubes(

            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0], grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=0,
            spacing = (
                grid['xyz'][0][0, 0, 1] - grid['xyz'][0][0, 0, 0],  # X 轴的步长
                grid['xyz'][1][0, 1, 0] - grid['xyz'][1][0, 0, 0],  # Y 轴的步长
                grid['xyz'][2][1, 0, 0] - grid['xyz'][2][0, 0, 0]   # Z 轴的步长
                )
        )
        verts = verts + np.array([grid['xyz'][0][0, 0, 0], grid['xyz'][1][0, 0, 0], grid['xyz'][2][0, 0, 0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            opacity=1.0)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export('{0}/surface_{1}.ply'.format(path, epoch), 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def plot_depth_maps(depth_maps, path, epoch, plot_nrow, img_res):
    depth_maps_plot = lin2img(depth_maps, img_res)

    tensor = torchvision.utils.make_grid(depth_maps_plot.repeat(1, 3, 1, 1),
                                         scale_each=True,
                                         normalize=True,
                                         nrow=plot_nrow).cpu().detach().numpy()
    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/depth_{1}.png'.format(path, epoch))

def plot_images(rgb_points, ground_true, path, epoch, plot_nrow, img_res):
    ground_true = (ground_true.to(rgb_points.device) + 1.) / 2.
    rgb_points = (rgb_points + 1.) / 2.

    output_vs_gt = torch.cat((rgb_points, ground_true), dim=0)
    output_vs_gt_plot = lin2img(output_vs_gt, img_res)

    tensor = torchvision.utils.make_grid(output_vs_gt_plot,
                                         scale_each=False,
                                         normalize=False,
                                         nrow=plot_nrow).cpu().detach().numpy()

    tensor = tensor.transpose(1, 2, 0)
    scale_factor = 255
    tensor = (tensor * scale_factor).astype(np.uint8)

    img = Image.fromarray(tensor)
    img.save('{0}/rendering_{1}.png'.format(path, epoch))

def lin2img(tensor, img_res):
    batch_size, num_samples, channels = tensor.shape
    return tensor.permute(0, 2, 1).view(batch_size, channels, img_res[0], img_res[1])

def get_grid_uniform(resolution):
    # 定义网格的范围，这里假设生成的网格点在 [-1, 1] 的立方体内
    x = np.linspace(-1, 1, resolution)
    y = np.linspace(-1, 1, resolution)
    z = np.linspace(-1, 1, resolution)
    
    # 生成三维网格
    xyz = np.meshgrid(x, y, z, indexing='ij')  # 使用 'ij' 来确保正确的维度顺序
    
    # 将网格点展开为列表，使用 np.vstack 的标准方式将 xyz 展平
    grid_points = np.vstack([xyz[0].ravel(), xyz[1].ravel(), xyz[2].ravel()]).T

    # 将结果保存为字典，转换为 PyTorch Tensor
    grid = {
        'xyz': xyz,
        'grid_points': torch.tensor(grid_points, dtype=torch.float32)
    }
    print(f"grid['xyz'] shapes: {[arr.shape for arr in grid['xyz']]}")
    return grid
