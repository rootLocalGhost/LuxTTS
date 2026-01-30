import argparse
import collections
import json
import logging
import os
import socket
import subprocess
import sys
import warnings
from collections import defaultdict
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
import torch
from packaging import version
from torch import distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

# Dynamically handle XPU/AMP imports based on available PyTorch version
if hasattr(torch, 'xpu') and torch.xpu.is_available():
    try:
        from torch.xpu.amp import GradScaler
    except ImportError:
        # Handle newer PyTorch versions where AMP might be unified
        if hasattr(torch.amp, "GradScaler"):
            from torch.amp import GradScaler
        else:
             # Create a fallback for inference if scaler is not found
            class GradScaler:
                def __init__(self, **kwargs): pass
                def scale(self, loss): return loss
                def step(self, optimizer): optimizer.step()
                def update(self): pass
else:
    # Use CUDA as fallback if XPU is not available
    if hasattr(torch.amp, "GradScaler"):
        from torch.amp import GradScaler
    else:
        from torch.cuda.amp import GradScaler

Pathlike = Union[str, Path]

class AttributeDict(dict):
    """Dictionary subclass that allows attribute-style access to values"""
    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError(f"No such attribute '{key}'")
    def __setattr__(self, key, value):
        self[key] = value
    def __delattr__(self, key):
        if key in self:
            del self[key]
            return
        raise AttributeError(f"No such attribute '{key}'")
    def __str__(self, indent: int = 2):
        tmp = {}
        for k, v in self.items():
            if isinstance(v, (Path, torch.device, torch.dtype)):
                v = str(v)
            tmp[k] = v
        return json.dumps(tmp, indent=indent, sort_keys=True)

class MetricsTracker(collections.defaultdict):
    """Track and aggregate training metrics"""
    def __init__(self):
        super(MetricsTracker, self).__init__(int)
    def __add__(self, other: "MetricsTracker") -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v
        for k, v in other.items():
            if v - v == 0:
                ans[k] = ans[k] + v
        return ans
    def __mul__(self, alpha: float) -> "MetricsTracker":
        ans = MetricsTracker()
        for k, v in self.items():
            ans[k] = v * alpha
        return ans
    def __str__(self) -> str:
        ans_frames = ""
        ans_utterances = ""
        for k, v in self.norm_items():
            norm_value = "%.4g" % v
            if "utt_" not in k:
                ans_frames += str(k) + "=" + str(norm_value) + ", "
            else:
                ans_utterances += str(k) + "=" + str(norm_value)
                if k == "utt_duration":
                    ans_utterances += " frames, "
                elif k == "utt_pad_proportion":
                    ans_utterances += ", "
                else:
                    raise ValueError(f"Unexpected key: {k}")
        frames = "%.2f" % self["frames"]
        ans_frames += "over " + str(frames) + " frames. "
        if ans_utterances != "":
            utterances = "%.2f" % self["utterances"]
            ans_utterances += "over " + str(utterances) + " utterances."
        return ans_frames + ans_utterances
    def norm_items(self) -> List[Tuple[str, float]]:
        num_frames = self["frames"] if "frames" in self else 1
        num_utterances = self["utterances"] if "utterances" in self else 1
        ans = []
        for k, v in self.items():
            if k == "frames" or k == "utterances":
                continue
            norm_value = (
                float(v) / num_frames if "utt_" not in k else float(v) / num_utterances
            )
            ans.append((k, norm_value))
        return ans
    def reduce(self, device):
        keys = sorted(self.keys())
        s = torch.tensor([float(self[k]) for k in keys], device=device)
        dist.all_reduce(s, op=dist.ReduceOp.SUM)
        for k, v in zip(keys, s.cpu().tolist()):
            self[k] = v
    def write_summary(
        self,
        tb_writer: SummaryWriter,
        prefix: str,
        batch_idx: int,
    ) -> None:
        for k, v in self.norm_items():
            tb_writer.add_scalar(prefix + k, v, batch_idx)

@contextmanager
def torch_autocast(device_type="xpu", **kwargs):
    """Context manager for automatic mixed precision with XPU support"""
    # Check if XPU is available, otherwise fall back to CUDA
    use_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()

    if version.parse(torch.__version__) >= version.parse("2.4.0"):
        # For PyTorch 2.4+, use unified AMP
        target_device = "xpu" if use_xpu else "cuda"
        with torch.amp.autocast(device_type=target_device, **kwargs):
            yield
    else:
        # For older PyTorch versions
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            if use_xpu:
                # Use XPU AMP if available
                if hasattr(torch.xpu, 'amp'):
                    with torch.xpu.amp.autocast(**kwargs):
                        yield
                else:
                    yield # No autocast if not found
            else:
                with torch.cuda.amp.autocast(**kwargs):
                    yield

def create_grad_scaler(device="xpu", **kwargs):
    """Create gradient scaler with XPU support"""
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        try:
            return torch.xpu.amp.GradScaler(**kwargs)
        except:
             if version.parse(torch.__version__) >= version.parse("2.3.0"):
                from torch.amp import GradScaler
                return GradScaler(device='xpu', **kwargs)
             return None

    if version.parse(torch.__version__) >= version.parse("2.3.0"):
        from torch.amp import GradScaler
        return GradScaler(device=device, **kwargs)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=FutureWarning)
            return torch.cuda.amp.GradScaler(**kwargs)

def setup_dist(
    rank=None,
    world_size=None,
    master_port=None,
    use_ddp_launch=False,
    master_addr=None,
):
    """Set up distributed training with XPU support"""
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = (
            "localhost" if master_addr is None else str(master_addr)
        )
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12354" if master_port is None else str(master_port)

    backend = "nccl"
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        backend = "ccl" # Use Intel OneCCL for XPU

    if use_ddp_launch is False:
        dist.init_process_group(backend, rank=rank, world_size=world_size)
        if backend == "ccl":
            torch.xpu.set_device(rank)
        else:
            torch.cuda.set_device(rank)
    else:
        dist.init_process_group(backend)

def cleanup_dist():
    """Clean up distributed training resources"""
    dist.destroy_process_group()

def prepare_input(
    params: AttributeDict,
    batch: dict,
    device: torch.device,
    return_tokens: bool = True,
    return_feature: bool = True,
    return_audio: bool = False,
):
    """Prepare input tensors for model training/inference"""
    return_list = []
    if return_tokens:
        return_list += [batch["tokens"]]
    if return_feature:
        features = batch["features"].to(device)
        features_lens = batch["features_lens"].to(device)
        return_list += [features * params.feat_scale, features_lens]
    if return_audio:
        return_list += [batch["audio"], batch["audio_lens"]]
    return return_list

def prepare_avg_tokens_durations(features_lens, tokens_lens):
    """Calculate average token durations for alignment"""
    tokens_durations = []
    for i in range(len(features_lens)):
        utt_duration = features_lens[i]
        avg_token_duration = utt_duration // tokens_lens[i]
        tokens_durations.append([avg_token_duration] * tokens_lens[i])
    return tokens_durations

def pad_labels(y: List[List[int]], pad_id: int, device: torch.device):
    """Pad sequences to same length for batch processing"""
    y = [token_ids + [pad_id] for token_ids in y]
    length = max([len(token_ids) for token_ids in y])
    y = [token_ids + [pad_id] * (length - len(token_ids)) for token_ids in y]
    return torch.tensor(y, dtype=torch.int64, device=device)

def get_tokens_index(durations: List[List[int]], num_frames: int) -> torch.Tensor:
    """Get token indices for each frame in the sequence"""
    durations = [x + [num_frames - sum(x)] for x in durations]
    batch_size = len(durations)
    ans = torch.zeros(batch_size, num_frames, dtype=torch.int64)
    for b in range(batch_size):
        this_dur = durations[b]
        cur_frame = 0
        for i, d in enumerate(this_dur):
            ans[b, cur_frame : cur_frame + d] = i
            cur_frame += d
        assert cur_frame == num_frames, (cur_frame, num_frames)
    return ans

def to_int_tuple(s: Union[str, int]):
    """Convert string to integer tuple"""
    if isinstance(s, int):
        return (s,)
    return tuple(map(int, s.split(",")))

def get_adjusted_batch_count(params: AttributeDict) -> float:
    """Calculate adjusted batch count for training"""
    return (
        params.batch_idx_train
        * (params.max_duration * params.world_size)
        / params.ref_duration
    )

def set_batch_count(model: Union[nn.Module, DDP], batch_count: float) -> None:
    """Set batch count for model modules"""
    if isinstance(model, DDP):
        model = model.module
    for name, module in model.named_modules():
        if hasattr(module, "batch_count"):
            module.batch_count = batch_count
        if hasattr(module, "name"):
            module.name = name

def condition_time_mask(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """Create random time masks for augmentation"""
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.rand_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask

def condition_time_mask_suffix(
    features_lens: torch.Tensor,
    mask_percent: Tuple[float, float],
    max_len: int = 0,
) -> torch.Tensor:
    """Create time masks at the end of sequences for augmentation"""
    mask_size = (
        torch.zeros_like(features_lens, dtype=torch.float32).uniform_(*mask_percent)
        * features_lens
    ).to(torch.int64)
    mask_starts = (
        torch.ones_like(mask_size, dtype=torch.float32) * (features_lens - mask_size)
    ).to(torch.int64)
    mask_ends = mask_starts + mask_size
    max_len = max(max_len, features_lens.max())
    seq_range = torch.arange(0, max_len, device=features_lens.device)
    mask = (seq_range[None, :] >= mask_starts[:, None]) & (
        seq_range[None, :] < mask_ends[:, None]
    )
    return mask

def make_pad_mask(lengths: torch.Tensor, max_len: int = 0) -> torch.Tensor:
    """Create padding mask for variable-length sequences"""
    assert lengths.ndim == 1, lengths.ndim
    max_len = max(max_len, lengths.max())
    n = lengths.size(0)
    seq_range = torch.arange(0, max_len, device=lengths.device)
    expaned_lengths = seq_range.unsqueeze(0).expand(n, max_len)
    return expaned_lengths >= lengths.unsqueeze(-1)

def str2bool(v):
    """Convert string to boolean value"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")

def setup_logger(
    log_filename: Pathlike,
    log_level: str = "info",
    use_console: bool = True,
) -> None:
    """Set up logging with timestamp and rank information"""
    now = datetime.now()
    date_time = now.strftime("%Y-%m-%d-%H-%M-%S")
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        formatter = f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] ({rank}/{world_size}) %(message)s"
        log_filename = f"{log_filename}-{date_time}-{rank}"
    else:
        formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"
        log_filename = f"{log_filename}-{date_time}"
    os.makedirs(os.path.dirname(log_filename), exist_ok=True)
    level = logging.ERROR
    if log_level == "debug":
        level = logging.DEBUG
    elif log_level == "info":
        level = logging.INFO
    elif log_level == "warning":
        level = logging.WARNING
    elif log_level == "critical":
        level = logging.CRITICAL
    logging.basicConfig(
        filename=log_filename,
        format=formatter,
        level=level,
        filemode="w",
        force=True,
    )
    if use_console:
        console = logging.StreamHandler()
        console.setLevel(level)
        console.setFormatter(logging.Formatter(formatter))
        logging.getLogger("").addHandler(console)

def get_git_sha1():
    """Get git commit hash"""
    try:
        git_commit = (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
        dirty_commit = (
            len(
                subprocess.run(
                    ["git", "diff", "--shortstat"],
                    check=True,
                    stdout=subprocess.PIPE,
                )
                .stdout.decode()
                .rstrip("\n")
                .strip()
            )
            > 0
        )
        git_commit = git_commit + "-dirty" if dirty_commit else git_commit + "-clean"
    except:
        return None
    return git_commit

def get_git_date():
    """Get git commit date"""
    try:
        git_date = (
            subprocess.run(
                ["git", "log", "-1", "--format=%ad", "--date=local"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:
        return None
    return git_date

def get_git_branch_name():
    """Get git branch name"""
    try:
        git_date = (
            subprocess.run(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                check=True,
                stdout=subprocess.PIPE,
            )
            .stdout.decode()
            .rstrip("\n")
            .strip()
        )
    except:
        return None
    return git_date

def get_env_info() -> Dict[str, Any]:
    """Get environment information including hardware and software details"""
    info = {
        "torch-version": str(torch.__version__),
        "python-version": sys.version[:4],
        "zipvoice-git-branch": get_git_branch_name(),
        "zipvoice-git-sha1": get_git_sha1(),
        "zipvoice-git-date": get_git_date(),
        "zipvoice-path": str(Path(__file__).resolve().parent.parent),
        "hostname": socket.gethostname(),
        "IP address": socket.gethostbyname(socket.gethostname()),
    }
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        info["torch-xpu-available"] = True
        info["torch-xpu-device"] = torch.xpu.get_device_name(0)
    else:
        info["torch-cuda-available"] = torch.cuda.is_available()
    return info

def get_parameter_groups_with_lrs(
    model: nn.Module,
    lr: float,
    include_names: bool = False,
    freeze_modules: List[str] = [],
    unfreeze_modules: List[str] = [],
) -> List[dict]:
    """Group parameters with different learning rates"""
    assert not (len(freeze_modules) and len(unfreeze_modules))
    flat_lr_scale = defaultdict(lambda: 1.0)
    names = []
    for name, m in model.named_modules():
        names.append(name)
        if hasattr(m, "lr_scale"):
            flat_lr_scale[name] = m.lr_scale
    lr_to_params = defaultdict(list)
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            logging.info(f"Remove {name} from parameter")
            continue
        split_name = name.split(".")
        prefix = split_name[0]
        if len(freeze_modules) > 0:
            if prefix == "module":
                module_name = split_name[1]
                if module_name in freeze_modules:
                    logging.info(f"Remove {name} from parameters")
                    continue
            else:
                if prefix in freeze_modules:
                    logging.info(f"Remove {name} from parameters")
                    continue
        elif len(unfreeze_modules) > 0:
            if prefix == "module":
                module_name = split_name[1]
                if module_name not in unfreeze_modules:
                    logging.info(f"Remove {name} from parameters")
                    continue
            else:
                if prefix not in unfreeze_modules:
                    logging.info(f"Remove {name} from parameters")
                    continue
        cur_lr = lr * flat_lr_scale[prefix]
        if prefix != "":
            cur_lr *= flat_lr_scale[""]
        for part in split_name[1:]:
            prefix = ".".join([prefix, part])
            cur_lr *= flat_lr_scale[prefix]
        lr_to_params[cur_lr].append((name, parameter) if include_names else parameter)
    if include_names:
        return [{"named_params": pairs, "lr": lr} for lr, pairs in lr_to_params.items()]
    else:
        return [{"params": params, "lr": lr} for lr, params in lr_to_params.items()]