"""Backend-agnostic accelerator selection for CUDA and Ascend NPU runtimes.

CUDA code paths keep their original behavior. NPU support activates only when
torch_npu is importable and reports a usable device; everything degrades to CPU
when neither accelerator is present.
"""
import contextlib
import importlib.util

import torch


def npu_available():
    """Check whether a usable Ascend NPU backend is present.

    Returns:
        bool: True only if torch_npu imports and exposes at least one device.
    """
    if importlib.util.find_spec('torch_npu') is None:
        return False
    import torch_npu  # noqa: F401  (import registers the private ``npu`` backend)
    return bool(torch.npu.is_available() and torch.npu.device_count() > 0)


def resolve_device(local_rank=0):
    """Select the accelerator device owned by this process.

    Args:
        local_rank (int): per-node device ordinal, matching the RANK/LOCAL_RANK
            launcher convention already used by :func:`timejev.engine.setup`.

    Returns:
        torch.device: ``npu`` when available, otherwise ``cuda``, otherwise ``cpu``.
    """
    if npu_available():
        return torch.device('npu', local_rank)
    if torch.cuda.is_available():
        return torch.device('cuda', local_rank)
    return torch.device('cpu')


def set_device(device):
    """Bind the current process to its accelerator ordinal.

    Args:
        device (torch.device): target device from :func:`resolve_device`.

    Returns:
        None. No-op on CPU.
    """
    if device.type == 'npu':
        torch.npu.set_device(device)
    elif device.type == 'cuda':
        torch.cuda.set_device(device)


def manual_seed_all(seed):
    """Seed the CPU generator and every visible accelerator device.

    Args:
        seed (int): RNG seed shared by torch, NumPy and Python ``random`` callers.

    Returns:
        None. Backends without a usable device are skipped.
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if importlib.util.find_spec('torch_npu') is not None:
        import torch_npu  # noqa: F401
        if torch.npu.is_available():
            torch.npu.manual_seed_all(seed)


def distributed_backend(device):
    """Return the torch.distributed backend matching the accelerator.

    Args:
        device (torch.device): resolved process device.

    Returns:
        str: ``hccl`` on NPU, ``nccl`` on CUDA, ``gloo`` on CPU.
    """
    return {'npu': 'hccl', 'cuda': 'nccl'}.get(device.type, 'gloo')


def synchronize(device):
    """Wait until all asynchronous work queued on the device completes.

    Args:
        device (torch.device): device to synchronize.

    Returns:
        None. No-op on CPU.
    """
    if device.type == 'npu':
        torch.npu.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize()


@contextlib.contextmanager
def autocast_bf16(device):
    """Run BF16 autocast on accelerators; leave CPU computation unchanged.

    Args:
        device (torch.device): device whose autocast backend should be used.

    Yields:
        None. Equivalent to ``torch.autocast('cuda', ...)`` when running on CUDA.
    """
    if device.type == 'cpu':
        with contextlib.nullcontext():
            yield
    else:
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            yield


def flash_attention_available():
    """Check whether the CUDA-only flash-attn package is importable.

    Returns:
        bool: True when ``flash_attn`` is installed in the current environment.
    """
    return importlib.util.find_spec('flash_attn') is not None


def attention_implementation(device, override='auto'):
    """Resolve the transformers attention implementation for a device.

    Args:
        device (torch.device): resolved process device.
        override (str): ``auto`` or an explicit transformers implementation name
            such as ``sdpa`` / ``flash_attention_2`` / ``eager``.

    Returns:
        str: implementation name accepted by ``AutoModel.from_pretrained``.
        CUDA with flash-attn preserves the official ``flash_attention_2`` default;
        every other backend falls back to the portable SDPA implementation.
    """
    if override and override != 'auto':
        return override
    if device.type == 'cuda' and flash_attention_available():
        return 'flash_attention_2'
    return 'sdpa'


def enable_torch_linear_attention(model):
    """Replace FLA Triton linear-attention kernels with transformers torch fallbacks.

    Needed only when the installed flash-linear-attention build cannot execute on
    the active accelerator (for example Ascend without matching Triton-Ascend
    kernels). The replacements are the pure-torch implementations shipped by
    transformers itself, so model math is unchanged. Dense and MoE Qwen3.5 layers
    each use the fallback functions defined in their own transformers module.

    Args:
        model (nn.Module): VideoJev model returned by ``from_pretrained``.

    Returns:
        int: number of GatedDeltaNet layers switched to the torch implementation.
    """
    from transformers.models.qwen3_5 import modeling_qwen3_5 as dense
    from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as moe

    fallbacks = {
        'Qwen3_5GatedDeltaNet': (dense.torch_chunk_gated_delta_rule,
                                 dense.torch_recurrent_gated_delta_rule,
                                 dense.torch_causal_conv1d_update),
        'Qwen3_5MoeGatedDeltaNet': (moe.torch_chunk_gated_delta_rule,
                                    moe.torch_recurrent_gated_delta_rule,
                                    moe.torch_causal_conv1d_update),
    }
    switched = 0
    for module in model.modules():
        fallback = fallbacks.get(type(module).__name__)
        if fallback is None:
            continue
        chunk_fn, recurrent_fn, conv_update = fallback
        module.chunk_gated_delta_rule = chunk_fn
        module.recurrent_gated_delta_rule = recurrent_fn
        module.causal_conv1d_fn = None
        module.causal_conv1d_update = conv_update
        switched += 1
    return switched
