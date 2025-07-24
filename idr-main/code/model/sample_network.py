import torch.nn as nn
import torch

class SampleNetwork(nn.Module):
    '''
    Represent the intersection (sample) point as differentiable function of the implicit geometry and camera parameters.
    See equation 3 in the paper for more details.
    '''

    def forward(self, surface_output, surface_sdf_values, surface_points_grad, surface_dists, surface_cam_loc, surface_ray_dirs):
        # Detect the device (MPS if available, else default to CPU)
        device = surface_output.device

        # Detach ray directions to prevent gradients from flowing back through them
        surface_ray_dirs_0 = surface_ray_dirs.detach()

        # Compute the dot product between the surface gradients and ray directions
        surface_points_dot = torch.bmm(surface_points_grad.view(-1, 1, 3),
                                       surface_ray_dirs_0.view(-1, 3, 1)).squeeze(-1)

        # Ensure the dot product does not contain zeros to avoid division by zero
        surface_points_dot = torch.where(surface_points_dot == 0, torch.tensor(1e-6).to(device), surface_points_dot)

        # Compute t(theta), i.e., distance to the intersection point adjusted by geometry
        surface_dists_theta = surface_dists - (surface_output - surface_sdf_values) / surface_points_dot

        # Compute the actual intersection point x(theta, c, v)
        surface_points_theta_c_v = surface_cam_loc + surface_dists_theta * surface_ray_dirs

        return surface_points_theta_c_v