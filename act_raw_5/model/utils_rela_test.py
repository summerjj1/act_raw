import os
import h5py
import torch
import numpy as np
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
# from transformers import pipeline
from PIL import Image
import requests
import cv2

# remove ACT. when
# from ACT.model.policy import ACTPolicy, CNNMLPPolicy
# from ACT.config.config import USE_DEPTH_ANYTHING, USE_LANG_SAM# must import first
from model.policy import ACTPolicy, CNNMLPPolicy
from config.config import USE_DEPTH_ANYTHING, USE_LANG_SAM# must import first

import IPython
e = IPython.embed

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#print(f"Using device: {device}")

# 假设在本文件同目录下有 relative_data_process_utils.py
from relative_data_process_utils import *

class EpisodicDataset(Dataset):
    def __init__(
        self,
        episode_ids,
        dataset_dir,
        camera_names,
        norm_stats,
        sequence_length=8,
        random_start=True,
        action_padding=True,
        use_relative_pose=True
    ):
        super().__init__()
        self.episode_ids = episode_ids
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.norm_stats = norm_stats
        self.is_sim = None

        self.sequence_length = sequence_length
        self.random_start = random_start
        self.action_padding = action_padding
        self.use_relative_pose = use_relative_pose

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = self.episode_ids[index]
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as f:
            is_sim = False
            qpos8_all = f['/observations/qpos'][()]   # shape (E,8)
            action_all = f['/action'][()]             # shape (E,action_dim)
            # ... 读图像等
            episode_len = qpos8_all.shape[0]

        # 选取 [actual_length]
        if self.random_start and episode_len>self.sequence_length:
            start_ts = np.random.randint(0, episode_len-self.sequence_length)
        else:
            start_ts = 0
        actual_len = min(self.sequence_length, episode_len - start_ts)

        qpos8_seq = qpos8_all[start_ts : start_ts+actual_len]   # [actual_len,8]
        action_seq = action_all[start_ts : start_ts+actual_len]

        # 1) 先转成 10D
        # [actual_len,10]
        pose10_seq = []
        for t in range(actual_len):
            pose10_seq.append(qpos8_to_pose10d(qpos8_seq[t]))
        pose10_seq = np.stack(pose10_seq, axis=0)

        # 2) 做相对变换 (仅 pos+rot6d, gripper 不变)
        if self.use_relative_pose and actual_len>0:
            base_pose10 = pose10_seq[-1]  # [10]
            # 先 -> mat4x4 + gripper
            base_mat = pose10d_to_mat(base_pose10[:-1]) # 这里要注意: pose10中最后一维是 grip? 
            # Actually we put grip at index=9. So pose10[:9] is pos+rot6. We'll handle carefully
            # Let's parse the last dimension as grip
            base_gripper = base_pose10[9]

            # 逐帧相对
            new_pose10_seq = []
            for t in range(actual_len):
                grip_t = pose10_seq[t][9]
                mat_t  = pose10d_to_mat(pose10_seq[t][:-1]) # pose10[t,:9] => [pos+rot6d(6)]
                
                # out_mat = inv(base_mat) * mat_t
                out_mat = convert_pose_mat_rep(
                    mat_t,
                    base_pose_mat=base_mat,
                    pose_rep='relative',
                    backward=False
                )
                # 变回 [pos+rot6d(6),不含gripper], 
                # 但 mat_to_pose10d 需要 gripper, 作为传参
                out_pose10 = mat_to_pose10d(out_mat, grip_t)
                new_pose10_seq.append(out_pose10)
            pose10_seq = np.stack(new_pose10_seq, axis=0) # [actual_len,10]

        # 3) padding 到 T => shape [T,10]
        T = self.sequence_length if self.action_padding else actual_len
        pose10_pad = np.zeros((T,10), dtype=np.float32)
        action_pad = np.zeros((T, action_seq.shape[1]), dtype=np.float32)
        is_pad = np.zeros((T,), dtype=bool)
        
        pose10_pad[:actual_len]  = pose10_seq
        action_pad[:actual_len]  = action_seq
        if actual_len < T:
            is_pad[actual_len:] = True

        # (图像等省略)
        # => 变成Tensor & 归一化
        pose10_tensor   = torch.from_numpy(pose10_pad)
        action_tensor   = torch.from_numpy(action_pad)
        is_pad_tensor   = torch.from_numpy(is_pad)

        # 归一化: qpos_mean, qpos_std => 10维
        pose10_tensor = (pose10_tensor - self.norm_stats["qpos_mean"]) / self.norm_stats["qpos_std"]
        action_tensor = (action_tensor - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        # 最后返回 (images, pose10_tensor, action_tensor, is_pad_tensor)
        # 让模型直接拿 10 维 eef + rot6d + gripper
        images_tensor = torch.zeros((T,3,64,64), dtype=torch.float32) # placeholder
        return images_tensor, pose10_tensor, action_tensor, is_pad_tensor


def get_norm_stats(dataset_dir, num_episodes):
    from relative_data_process_utils import qpos8_to_pose10d

    all_pose10 = []
    all_action = []
    for ep_id in range(num_episodes):
        path = os.path.join(dataset_dir, f'episode_{ep_id}.hdf5')
        with h5py.File(path,'r') as f:
            qpos8 = f['/observations/qpos'][()]  # shape [E,8]
            action = f['/action'][()]
        # batch convert => 10D
        pose10_list = [ qpos8_to_pose10d(row) for row in qpos8 ]
        pose10_arr  = np.stack(pose10_list, axis=0) # [E,10]

        all_pose10.append(torch.from_numpy(pose10_arr))
        all_action.append(torch.from_numpy(action))

    all_pose10 = torch.cat(all_pose10, dim=0)    # [total,10]
    all_action= torch.cat(all_action, dim=0)     # [total, action_dim]

    pose_mean = all_pose10.mean(dim=0, keepdim=True)
    pose_std  = all_pose10.std(dim=0, keepdim=True)
    pose_std  = torch.clip(pose_std, 1e-2, float('inf'))

    act_mean = all_action.mean(dim=0, keepdim=True)
    act_std  = all_action.std(dim=0, keepdim=True)
    act_std  = torch.clip(act_std, 1e-2, float('inf'))

    stats = {
        "qpos_mean":   pose_mean.numpy().squeeze(),
        "qpos_std":    pose_std.numpy().squeeze(),
        "action_mean": act_mean.numpy().squeeze(),
        "action_std":  act_std.numpy().squeeze()
    }
    return stats



def load_data(dataset_dir, num_episodes, camera_names, batch_size_train, batch_size_val):
    from torch.utils.data import DataLoader

    # 8:2 split
    train_ratio = 0.8
    indices = np.random.permutation(num_episodes)
    split_pt = int(train_ratio * num_episodes)
    train_ids = indices[:split_pt]
    val_ids   = indices[split_pt:]

    # 1) 统计
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # 2) 构建 dataset
    train_dataset = EpisodicDataset(
        episode_ids=train_ids,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        sequence_length=16,
        random_start=True,
        action_padding=True,
        use_relative_pose=True
    )
    val_dataset = EpisodicDataset(
        episode_ids=val_ids,
        dataset_dir=dataset_dir,
        camera_names=camera_names,
        norm_stats=norm_stats,
        sequence_length=16,
        random_start=False,
        action_padding=True,
        use_relative_pose=True
    )

    # 3) DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size_train,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    return train_loader, val_loader, norm_stats, train_dataset.is_sim


def make_policy(policy_class, policy_config):
    if policy_class == "ACT":
        policy = ACTPolicy(policy_config)
    elif policy_class == "CNNMLP":
        policy = CNNMLPPolicy(policy_config)
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return policy

def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise ValueError(f"Unknown policy class: {policy_class}")
    return optimizer

### env utils

def sample_box_pose():
    x_range = [0.0, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    cube_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    cube_quat = np.array([1, 0, 0, 0])
    return np.concatenate([cube_position, cube_quat])

def sample_insertion_pose():
    # Peg
    x_range = [0.1, 0.2]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    peg_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    peg_quat = np.array([1, 0, 0, 0])
    peg_pose = np.concatenate([peg_position, peg_quat])

    # Socket
    x_range = [-0.2, -0.1]
    y_range = [0.4, 0.6]
    z_range = [0.05, 0.05]

    ranges = np.vstack([x_range, y_range, z_range])
    socket_position = np.random.uniform(ranges[:, 0], ranges[:, 1])

    socket_quat = np.array([1, 0, 0, 0])
    socket_pose = np.concatenate([socket_position, socket_quat])

    return peg_pose, socket_pose

### helper functions

def get_image(images, camera_names, device='cuda'):
    curr_images = []
    for cam_name in camera_names:
        curr_image = rearrange(images[cam_name], 'h w c -> c h w')
        curr_images.append(curr_image)
    curr_image = np.stack(curr_images, axis=0)
    # cv2.imshow("cam", curr_image)
    # cv2.waitKey(0) 
    curr_image = torch.from_numpy(curr_image / 255.0).float().to(device).unsqueeze(0)
    return curr_image

def compute_dict_mean(epoch_dicts):
    result = {k: None for k in epoch_dicts[0]}
    num_items = len(epoch_dicts)
    for k in result:
        value_sum = 0
        for epoch_dict in epoch_dicts:
            value_sum += epoch_dict[k]
        result[k] = value_sum / num_items
    return result

def detach_dict(d):
    new_d = dict()
    for k, v in d.items():
        new_d[k] = v.detach()
    return new_d

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def pos2pwm(pos:np.ndarray) -> np.ndarray:
    """
    :param pos: numpy array of joint positions in range [-pi, pi]
    :return: numpy array of pwm values in range [0, 4096]
    """ 
    return (pos / 3.14 + 1.) * 2048
    
def pwm2pos(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm values in range [0, 4096]
    :return: numpy array of joint positions in range [-pi, pi]
    """
    return (pwm / 2048 - 1) * 3.14

def pwm2vel(pwm:np.ndarray) -> np.ndarray:
    """
    :param pwm: numpy array of pwm/s joint velocities
    :return: numpy array of rad/s joint velocities 
    """
    return pwm * 3.14 / 2048

def vel2pwm(vel:np.ndarray) -> np.ndarray:
    """
    :param vel: numpy array of rad/s joint velocities
    :return: numpy array of pwm/s joint velocities
    """
    return vel * 2048 / 3.14
    
def pwm2norm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of pwm values in range [0, 4096]
    :return: numpy array of values in range [0, 1]
    """
    return x / 4096
    
def norm2pwm(x:np.ndarray) -> np.ndarray:
    """
    :param x: numpy array of values in range [0, 1]
    :return: numpy array of pwm values in range [0, 4096]
    """
    return x * 4096
