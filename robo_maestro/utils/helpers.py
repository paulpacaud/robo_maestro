import os

import numpy as np
import torch
import pickle as pkl
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from torchvision.transforms import Resize
from scipy.spatial.transform import Rotation

from robo_maestro.utils.constants import CODE_DIR


def euler_to_quat(euler, degrees):
    rotation = Rotation.from_euler("xyz", euler, degrees=degrees)
    return rotation.as_quat()


def quat_to_euler(quat, degrees):
    rotation = Rotation.from_quat(quat)
    return rotation.as_euler("xyz", degrees=degrees)


def crop_center(im, crop_h, crop_w):
    h, w = im.shape[-2], im.shape[-1]
    start_x = w // 2 - (crop_w // 2)
    start_y = h // 2 - (crop_h // 2)
    return im[..., start_y:start_y + crop_w, start_x:start_x + crop_h], start_x, start_y


def resize(im, new_size, im_type="rgb"):
    if im_type == "rgb":
        interpolation = InterpolationMode.BILINEAR
    elif im_type == "depth":
        interpolation = InterpolationMode.NEAREST
    elif im_type == "gripper_attn":
        interpolation = InterpolationMode.NEAREST
    elif im_type == "pc":
        interpolation = InterpolationMode.NEAREST

    orig_h, orig_w = im.shape[-2], im.shape[-1]
    if orig_h < orig_w:
        ratio = (new_size / orig_h)
    else:
        ratio = (new_size / orig_w)

    h_resize = int(orig_h * ratio)
    w_resize = int(orig_w * ratio)

    resizer = Resize((h_resize, w_resize), interpolation=interpolation)

    new_im = resizer(im)
    return new_im, ratio


def process_keystep(obs, links_bbox, cam_list=["bravo_camera", "charlie_camera", "alpha_camera"], crop_size=None):
    rgb = []
    pc = []
    gripper_pos = obs["gripper_pos"]
    gripper_quat = obs["gripper_quat"]
    gripper_state = not obs["gripper_state"]
    gripper_pose = np.concatenate([gripper_pos, gripper_quat, np.expand_dims(gripper_state, 0)])
    for cam_name in cam_list:
        rgb.append(torch.from_numpy(obs[f"rgb_{cam_name}"]))
        pc.append(torch.from_numpy(obs[f"pcd_{cam_name}"]))

    rgb = torch.stack(rgb)  # (C, H, W, 3)
    pc = torch.stack(pc)  # (C, H, W, 3)

    if crop_size:
        rgb = rgb.permute(0, 3, 1, 2)
        pc = pc.permute(0, 3, 1, 2)

        rgb, ratio = resize(rgb, crop_size, im_type="rgb")
        pc, ratio = resize(pc, crop_size, im_type="pc")
        rgb, start_x, start_y = crop_center(rgb, crop_size, crop_size)
        pc, start_x, start_y = crop_center(pc, crop_size, crop_size)
        rgb = rgb.permute(0, 2, 3, 1)
        pc = pc.permute(0, 2, 3, 1)

    robot_info = obs["robot_info"]
    bbox_info = {}
    pose_info = {}
    for link_name, link_pose in robot_info.items():
        pose_info[f"{link_name}_pose"] = link_pose
        bbox_info[f"{link_name}_bbox"] = links_bbox[link_name]

    keystep = {
        "rgb": rgb.numpy().astype(np.uint8),
        "pc": pc.float().numpy(),
        "gripper": gripper_pose,
        "arm_links_info": (bbox_info, pose_info),
    }
    return keystep