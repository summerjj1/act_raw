# 单臂 ACT 单机多卡训练 (DDP)
#
# 多卡启动（推荐）:
#   torchrun --nproc_per_node=4 train_single_arm_ddp.py [--task TASK]
#
# 单卡也可直接运行（不设 RANK 时自动退化为单进程）:
#   python train_single_arm_ddp.py [--task TASK]
import sys
import h5py
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import os
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)
DETR_PATH = os.path.join(PROJECT_ROOT, 'detr')
sys.path.append(DETR_PATH)
from config.config import SINGLE_ARM_TASK_CONFIG, SINGLE_ARM_POLICY_CONFIG, SINGLE_ARM_TRAIN_CONFIG, WANDB_CONFIG
import pickle
import re
import argparse
import shutil
from copy import deepcopy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from model.utils import make_policy, load_data_single_arm, set_seed
from tqdm import tqdm

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


def parse_episode_ranges(s):
    """解析 '0-30,50-100' 为 [(0,30), (50,100)]，区间闭区间 [a,b]"""
    if not s or not s.strip():
        return None
    ranges = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-', 1)
            ranges.append((int(a.strip()), int(b.strip())))
        else:
            i = int(part)
            ranges.append((i, i))
    return ranges if ranges else None


def episode_number(filename):
    name = os.path.splitext(filename)[0]
    nums = re.findall(r'\d+', name)
    return int(nums[-1]) if nums else -1


parser = argparse.ArgumentParser()
parser.add_argument('--task', type=str, default="K001_single_arm")
parser.add_argument('--data-format', choices=('hdf5', 'raw10_hdf5'), default='hdf5')
parser.add_argument('--raw-black-level', type=int, default=64)
parser.add_argument('--raw-white-level', type=int, default=1023)
parser.add_argument('--raw-gamma', type=float, default=2.2)
args = parser.parse_args()
task = args.task

task_cfg = SINGLE_ARM_TASK_CONFIG.copy()
train_cfg = SINGLE_ARM_TRAIN_CONFIG
policy_config = SINGLE_ARM_POLICY_CONFIG.copy()
checkpoint_dir = os.path.join(train_cfg['checkpoint_dir'], task)


def get_device():
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    if torch.cuda.is_available():
        device = torch.device(f'cuda:{local_rank}')
    else:
        device = torch.device('cpu')
    return device, local_rank


def setup_ddp():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def cleanup_ddp():
    if dist.is_initialized():
        dist.destroy_process_group()


def unwrap_policy(policy):
    return policy.module if isinstance(policy, DDP) else policy


def forward_pass(data, policy, device):
    image_data, qpos_data, action_data, is_pad = data
    image_data = image_data.to(device)
    qpos_data = qpos_data.to(device)
    action_data = action_data.to(device)
    is_pad = is_pad.to(device)
    return policy(qpos_data, image_data, action_data, is_pad)


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs - 1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs - 1, len(validation_history)), val_values, label='validation')
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


def reduce_loss_sums(loss_sums, count):
    if dist.is_initialized():
        for key in loss_sums:
            dist.all_reduce(loss_sums[key], op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    denom = count.clamp(min=1).float()
    return {k: (v / denom).item() for k, v in loss_sums.items()}


def train_bc(rank, world_size, train_dataloader, val_dataloader, train_sampler, policy_config, task, num_episodes, device):
    is_main = rank == 0
    if WANDB_AVAILABLE and WANDB_CONFIG.get('enabled', True) and is_main:
        wandb_cfg = {
            'task': task,
            'num_episodes': num_episodes,
            **{k: v for k, v in train_cfg.items()},
            **{k: v for k, v in policy_config.items() if k != 'device' and isinstance(v, (int, float, str, bool, list))},
            **{k: v for k, v in task_cfg.items() if k != 'dataset_dir' and isinstance(v, (int, float, str, bool, list))},
        }
        wandb.init(
            project=WANDB_CONFIG.get('single_arm_project', 'OneStar_ACT_single_arm'),
            name=task,
            config=wandb_cfg,
        )

    import detr.main as detr_main
    detr_main.device = str(device)

    policy = make_policy(policy_config['policy_class'], policy_config).to(device)
    if world_size > 1:
        if device.type == 'cuda':
            policy = DDP(policy, device_ids=[device.index], find_unused_parameters=True)
        else:
            policy = DDP(policy, find_unused_parameters=True)

    policy_to_save = unwrap_policy(policy)
    optimizer = policy_to_save.configure_optimizers()
    if is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)

    train_history = []
    validation_history = []
    min_val_loss = float('inf')
    best_ckpt_info = None
    grad_accum = train_cfg.get('grad_accum', 1)
    checkpoint_interval_steps = train_cfg.get('checkpoint_interval_steps', 0)
    optimizer_steps = 0
    num_epochs = train_cfg['num_epochs']

    def save_step_checkpoint(step):
        if is_main and checkpoint_interval_steps > 0 and step % checkpoint_interval_steps == 0:
            ckpt_path = os.path.join(
                checkpoint_dir,
                f"policy_step_{step}_seed_{train_cfg['seed']}.ckpt",
            )
            torch.save(policy_to_save.state_dict(), ckpt_path)
            print(f'Saved checkpoint at optimizer step {step}: {ckpt_path}')
    epoch_iter = tqdm(range(num_epochs), desc='Training', unit='epoch', dynamic_ncols=True) if is_main else range(num_epochs)
    loss_keys = ['loss', 'l1', 'kl']

    for epoch in epoch_iter:
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if is_main:
            epoch_iter.set_description(f'Epoch {epoch}')

        # validation
        policy.eval()
        with torch.inference_mode():
            val_sums = {k: torch.tensor(0.0, device=device) for k in loss_keys}
            epoch_val_count = torch.tensor(0, device=device, dtype=torch.long)
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy, device)
                batch_size = data[0].shape[0]
                for k in loss_keys:
                    if k in forward_dict:
                        val_sums[k] = val_sums[k] + forward_dict[k].detach() * batch_size
                epoch_val_count = epoch_val_count + batch_size
            epoch_val_losses = reduce_loss_sums(val_sums, epoch_val_count)
            epoch_val_loss = epoch_val_losses['loss']

            if is_main:
                validation_history.append({k: torch.tensor(v) for k, v in epoch_val_losses.items()})
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy_to_save.state_dict()))
                print(f'Val loss:   {epoch_val_loss:.5f}  (l1={epoch_val_losses["l1"]:.5f}, kl={epoch_val_losses["kl"]:.5f})')
                epoch_iter.set_postfix(val_loss=f'{epoch_val_loss:.5f}', refresh=False)
                if WANDB_AVAILABLE and WANDB_CONFIG.get('enabled', True):
                    wandb.log({f'val/{k}': v for k, v in epoch_val_losses.items()}, step=epoch)

        # training
        policy.train()
        optimizer.zero_grad()
        num_train_batches = 0
        train_sums = {k: torch.tensor(0.0, device=device) for k in loss_keys}
        epoch_train_count = torch.tensor(0, device=device, dtype=torch.long)
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy, device)
            loss = forward_dict['loss'] / grad_accum
            loss.backward()
            if (batch_idx + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                optimizer_steps += 1
                save_step_checkpoint(optimizer_steps)
            batch_size = data[0].shape[0]
            for k in loss_keys:
                if k in forward_dict:
                    train_sums[k] = train_sums[k] + forward_dict[k].detach() * batch_size
            epoch_train_count = epoch_train_count + batch_size
            num_train_batches = batch_idx + 1
        if num_train_batches > 0 and num_train_batches % grad_accum != 0:
            optimizer.step()
            optimizer.zero_grad()
            optimizer_steps += 1
            save_step_checkpoint(optimizer_steps)

        epoch_train_losses = reduce_loss_sums(train_sums, epoch_train_count)
        epoch_train_loss = epoch_train_losses['loss']

        if is_main:
            if num_train_batches == 0:
                print(f'[Warn] Epoch {epoch}: no training batches. Skipping.')
            else:
                train_history.append({k: torch.tensor(v) for k, v in epoch_train_losses.items()})
                print(f'Train loss: {epoch_train_loss:.5f}  (l1={epoch_train_losses["l1"]:.5f}, kl={epoch_train_losses["kl"]:.5f})')
                epoch_iter.set_postfix(val_loss=f'{min_val_loss:.5f}', train_loss=f'{epoch_train_loss:.5f}', refresh=True)
                if WANDB_AVAILABLE and WANDB_CONFIG.get('enabled', True):
                    wandb.log({f'train/{k}': v for k, v in epoch_train_losses.items()}, step=epoch)

            if epoch % 200 == 0 and train_history and validation_history:
                plot_history(train_history, validation_history, epoch, checkpoint_dir, train_cfg['seed'])

        if dist.is_initialized():
            dist.barrier()

    if is_main:
        ckpt_path = os.path.join(checkpoint_dir, 'policy_last.ckpt')
        torch.save(policy_to_save.state_dict(), ckpt_path)
        if best_ckpt_info is not None:
            best_ckpt_path = os.path.join(checkpoint_dir, 'policy_best.ckpt')
            torch.save(best_ckpt_info[2], best_ckpt_path)
            print(f'Saved best checkpoint (epoch {best_ckpt_info[0]}, val_loss {best_ckpt_info[1]:.5f})')
        if WANDB_AVAILABLE and WANDB_CONFIG.get('enabled', True):
            wandb.finish()


if __name__ == '__main__':
    rank, world_size, local_rank = setup_ddp()
    device, _ = get_device()
    os.environ['DEVICE'] = str(device)

    try:
        set_seed(train_cfg['seed'] + rank)

        if rank == 0:
            os.makedirs(checkpoint_dir, exist_ok=True)
            train_info_src = os.path.join(CURRENT_DIR, 'train_info.txt')
            if os.path.isfile(train_info_src):
                shutil.copy2(train_info_src, os.path.join(checkpoint_dir, 'train_info.txt'))
                print(f'Copied train_info.txt to {checkpoint_dir}')

        data_dir = task_cfg['dataset_dir']
        episode_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.hdf5') and not f.endswith('.bak')])

        episode_ranges_cfg = task_cfg.get('episode_ranges')
        episode_ranges = parse_episode_ranges(episode_ranges_cfg if isinstance(episode_ranges_cfg, str) else None)
        if episode_ranges is not None:
            episode_files = [f for f in episode_files if any(a <= episode_number(f) <= b for a, b in episode_ranges)]
            if rank == 0:
                print(f'Episode ranges (config): {episode_ranges_cfg} -> using {len(episode_files)} episodes')

        num_episodes = len(episode_files)
        if num_episodes == 0:
            raise FileNotFoundError(f'No .hdf5 files found in {data_dir} (or none in given episode_ranges)')

        if args.data_format == 'raw10_hdf5':
            sample_path = os.path.join(data_dir, episode_files[0])
            with h5py.File(sample_path, 'r') as sample:
                encoding = sample.attrs.get('image_encoding', '')
                if isinstance(encoding, bytes):
                    encoding = encoding.decode('utf-8')
            if encoding != 'bggr10_planes_uint16':
                raise ValueError(
                    f'{sample_path}: expected image_encoding=bggr10_planes_uint16, got {encoding!r}'
                )
            policy_config.update({
                'image_channels': 4,
                'image_mean': [0.406, 0.456, 0.456, 0.485],
                'image_std': [0.225, 0.224, 0.224, 0.229],
            })

        if rank == 0:
            print(f'Found {num_episodes} episodes, world_size={world_size}, data_format={args.data_format}')

        train_dataset, val_dataset, stats = load_data_single_arm(
            data_dir,
            num_episodes,
            task_cfg['camera_names'],
            train_cfg['batch_size_train'],
            train_cfg['batch_size_val'],
            episode_files=episode_files,
            episode_len=task_cfg['episode_len'],
            raw_black_level=args.raw_black_level,
            raw_white_level=args.raw_white_level,
            raw_gamma=args.raw_gamma,
            return_datasets=True,
        )

        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
            train_dataloader = DataLoader(
                train_dataset,
                batch_size=train_cfg['batch_size_train'],
                sampler=train_sampler,
                shuffle=False,
                pin_memory=True,
                num_workers=4,
                prefetch_factor=2,
            )
            val_dataloader = DataLoader(
                val_dataset,
                batch_size=train_cfg['batch_size_val'],
                sampler=val_sampler,
                shuffle=False,
                pin_memory=True,
                num_workers=2,
                prefetch_factor=2,
            )
        else:
            train_sampler = None
            train_dataloader = DataLoader(train_dataset, batch_size=train_cfg['batch_size_train'], shuffle=True, pin_memory=True, num_workers=4, prefetch_factor=2)
            val_dataloader = DataLoader(val_dataset, batch_size=train_cfg['batch_size_val'], shuffle=True, pin_memory=True, num_workers=2, prefetch_factor=2)

        if rank == 0:
            stats_path = os.path.join(checkpoint_dir, 'dataset_stats.pkl')
            with open(stats_path, 'wb') as f:
                pickle.dump(stats, f)

        train_bc(rank, world_size, train_dataloader, val_dataloader, train_sampler, policy_config, task, num_episodes, device)
    finally:
        cleanup_ddp()
