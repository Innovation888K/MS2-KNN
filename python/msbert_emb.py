# -*- coding: utf-8 -*-
"""
MSBERT embedding builder for official MSBERTModel.py.

This module adapts:
    for_git/msbert/MSBERTModel.py

Expected official model API:
    from MSBERTModel import MSBERT
    model.predict(input_id, intensity)

Input shapes:
    input_id:  torch.LongTensor, shape [B, L]
    intensity: torch.FloatTensor, shape [B, 1, L]

Output:
    embedding: numpy.ndarray, shape [N, hidden]
"""

import sys
import time
from pathlib import Path

import numpy as np

try:
    import torch
except ImportError as e:
    raise ImportError("MSBERT requires PyTorch. Please install torch first.") from e

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


# ---------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------
try:
    from config import PipelineConfig
except ImportError:
    from .config import PipelineConfig

try:
    from spec2vec_emb import (
        get_config_value,
        get_msdata_length,
        get_spectrum_peaks,
        cast_embedding_dtype,
    )
except ImportError:
    from .spec2vec_emb import (
        get_config_value,
        get_msdata_length,
        get_spectrum_peaks,
        cast_embedding_dtype,
    )

try:
    from utils import l2_normalize
except ImportError:
    try:
        from .utils import l2_normalize
    except ImportError:
        def l2_normalize(x, eps=1e-12):
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            norms = np.maximum(norms, eps)
            return x / norms


# ---------------------------------------------------------------------
# Path / device helpers
# ---------------------------------------------------------------------
def resolve_path(path_value):
    if path_value is None:
        return None

    return Path(path_value).expanduser()


def get_msbert_device(config: PipelineConfig):
    device = str(get_config_value(config, "msbert_device", "cuda"))

    if device.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[msbert] CUDA requested but not available. Falling back to CPU.",
            flush=True,
        )
        return "cpu"

    return device


def log_cuda_memory(prefix="[msbert]", device="cuda"):
    """
    Print CUDA memory usage.

    allocated:
        Real memory occupied by live tensors.

    reserved:
        Memory reserved by PyTorch CUDA caching allocator.
        nvidia-smi usually shows this value, so it may look larger.
    """

    if not str(device).startswith("cuda"):
        return

    if not torch.cuda.is_available():
        return

    dev = torch.device(device)

    allocated = torch.cuda.memory_allocated(dev) / 1024**3
    reserved = torch.cuda.memory_reserved(dev) / 1024**3
    max_allocated = torch.cuda.max_memory_allocated(dev) / 1024**3

    print(
        f"{prefix} cuda memory: "
        f"allocated={allocated:.3f} GiB, "
        f"reserved={reserved:.3f} GiB, "
        f"max_allocated={max_allocated:.3f} GiB",
        flush=True,
    )


def safe_torch_load(path, map_location="cpu"):
    """
    Compatible torch.load wrapper.

    Newer PyTorch versions may support weights_only.
    Older versions do not.
    """

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


# ---------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------
def _extract_state_dict(checkpoint):
    """
    Extract state_dict from common checkpoint formats.

    Returns:
        state_dict, full_model
    """

    if isinstance(checkpoint, torch.nn.Module):
        return None, checkpoint

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported MSBERT checkpoint type: {type(checkpoint)}")

    for key in [
        "state_dict",
        "model_state_dict",
        "net_state_dict",
        "msbert_state_dict",
    ]:
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key], None

    if "model" in checkpoint:
        if isinstance(checkpoint["model"], torch.nn.Module):
            return None, checkpoint["model"]

        if isinstance(checkpoint["model"], dict):
            return checkpoint["model"], None

    # Maybe checkpoint itself is a raw state_dict.
    if len(checkpoint) > 0 and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        return checkpoint, None

    raise ValueError(
        "Could not extract state_dict from MSBERT checkpoint. "
        f"Checkpoint keys: {list(checkpoint.keys())[:50]}"
    )


def _clean_state_dict_keys(state_dict):
    """
    Remove common prefixes from checkpoint keys.
    """

    cleaned = {}

    for key, value in state_dict.items():
        new_key = key

        changed = True
        while changed:
            changed = False

            for prefix in [
                "module.",
                "model.",
                "msbert.",
                "net.",
            ]:
                if new_key.startswith(prefix):
                    new_key = new_key[len(prefix):]
                    changed = True

        cleaned[new_key] = value

    return cleaned


def _infer_msbert_params_from_state_dict(state_dict, config: PipelineConfig):
    """
    Infer MSBERT architecture parameters from checkpoint when possible.
    """

    vocab_size = int(get_config_value(config, "msbert_vocab_size", 1001))
    hidden = int(get_config_value(config, "msbert_hidden", 512))
    n_layers = int(get_config_value(config, "msbert_n_layers", 6))

    # From your MSBERTModel.py:
    # self.embedding.token.weight
    if "embedding.token.weight" in state_dict:
        weight = state_dict["embedding.token.weight"]
        vocab_size = int(weight.shape[0])
        hidden = int(weight.shape[1])

    # Fallback:
    # self.fc2 = nn.Linear(hidden, vocab_size, bias=False)
    if "fc2.weight" in state_dict:
        weight = state_dict["fc2.weight"]
        vocab_size = int(weight.shape[0])
        hidden = int(weight.shape[1])

    layer_ids = []

    for key in state_dict.keys():
        if key.startswith("transformer_blocks."):
            parts = key.split(".")
            if len(parts) >= 2:
                try:
                    layer_ids.append(int(parts[1]))
                except Exception:
                    pass

    if len(layer_ids) > 0:
        n_layers = max(layer_ids) + 1

    attn_heads = int(get_config_value(config, "msbert_attn_heads", 16))
    dropout = float(get_config_value(config, "msbert_dropout", 0.1))
    max_len = int(get_config_value(config, "msbert_max_len", 100))
    max_pred = int(get_config_value(config, "msbert_max_pred", 2))

    return {
        "vocab_size": vocab_size,
        "hidden": hidden,
        "n_layers": n_layers,
        "attn_heads": attn_heads,
        "dropout": dropout,
        "max_len": max_len,
        "max_pred": max_pred,
    }


def get_model_vocab_size(model, fallback=1001):
    """
    Get vocab size from loaded MSBERT model.
    """

    try:
        return int(model.embedding.token.weight.shape[0])
    except Exception:
        return int(fallback)


def get_model_hidden_size(model, fallback=512):
    """
    Get hidden dimension from loaded MSBERT model.
    """

    try:
        return int(model.embedding.token.weight.shape[1])
    except Exception:
        return int(fallback)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def load_msbert_model(config: PipelineConfig):
    """
    Load official MSBERT model from:

        config.msbert_repo_dir / "MSBERTModel.py"

    Required config:
        msbert_repo_dir
        msbert_checkpoint_path
    """

    repo_dir = resolve_path(get_config_value(config, "msbert_repo_dir", None))
    ckpt_path = resolve_path(get_config_value(config, "msbert_checkpoint_path", None))

    if repo_dir is None:
        raise ValueError("config.msbert_repo_dir is required for MSBERT.")

    if not repo_dir.exists():
        raise FileNotFoundError(f"MSBERT repo_dir does not exist: {repo_dir}")

    model_py = repo_dir / "MSBERTModel.py"

    if not model_py.exists():
        raise FileNotFoundError(
            f"MSBERTModel.py not found under msbert_repo_dir: {model_py}"
        )

    if ckpt_path is None:
        raise ValueError("config.msbert_checkpoint_path is required for MSBERT.")

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MSBERT checkpoint does not exist: {ckpt_path}"
        )

    repo_dir = repo_dir.resolve()
    ckpt_path = ckpt_path.resolve()

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    from MSBERTModel import MSBERT

    device = get_msbert_device(config)

    print(f"[msbert] repo_dir={repo_dir}", flush=True)
    print(f"[msbert] checkpoint={ckpt_path}", flush=True)
    print(f"[msbert] device={device}", flush=True)

    checkpoint = safe_torch_load(ckpt_path, map_location="cpu")
    state_dict, full_model = _extract_state_dict(checkpoint)

    if full_model is not None:
        model = full_model
        print("[msbert] Loaded full torch.nn.Module from checkpoint.", flush=True)

    else:
        state_dict = _clean_state_dict_keys(state_dict)
        params = _infer_msbert_params_from_state_dict(state_dict, config)

        print(
            "[msbert] model params: "
            f"vocab_size={params['vocab_size']}, "
            f"hidden={params['hidden']}, "
            f"n_layers={params['n_layers']}, "
            f"attn_heads={params['attn_heads']}, "
            f"dropout={params['dropout']}, "
            f"max_len={params['max_len']}, "
            f"max_pred={params['max_pred']}",
            flush=True,
        )

        model = MSBERT(
            vocab_size=params["vocab_size"],
            hidden=params["hidden"],
            n_layers=params["n_layers"],
            attn_heads=params["attn_heads"],
            dropout=params["dropout"],
            max_len=params["max_len"],
            max_pred=params["max_pred"],
        )

        strict_load = bool(get_config_value(config, "msbert_strict_load", True))

        try:
            load_result = model.load_state_dict(
                state_dict,
                strict=strict_load,
            )

            if not strict_load:
                missing = getattr(load_result, "missing_keys", [])
                unexpected = getattr(load_result, "unexpected_keys", [])

                if len(missing) > 0:
                    print(
                        f"[msbert] WARNING missing keys: {missing[:20]}",
                        flush=True,
                    )
                    if len(missing) > 20:
                        print(
                            f"[msbert] ... and {len(missing) - 20} more missing keys",
                            flush=True,
                        )

                if len(unexpected) > 0:
                    print(
                        f"[msbert] WARNING unexpected keys: {unexpected[:20]}",
                        flush=True,
                    )
                    if len(unexpected) > 20:
                        print(
                            f"[msbert] ... and {len(unexpected) - 20} more unexpected keys",
                            flush=True,
                        )

        except RuntimeError as e:
            raise RuntimeError(
                "Failed to load MSBERT checkpoint. "
                "This usually means checkpoint architecture does not match "
                "vocab_size / hidden / n_layers / attn_heads.\n"
                f"Original error:\n{repr(e)}"
            ) from e

    model = model.to(device)
    model.eval()

    if not hasattr(model, "predict"):
        raise AttributeError(
            "Loaded MSBERT model does not have .predict(input_id, intensity)."
        )

    runtime_vocab_size = get_model_vocab_size(
        model,
        fallback=get_config_value(config, "msbert_vocab_size", 1001),
    )
    runtime_hidden = get_model_hidden_size(
        model,
        fallback=get_config_value(config, "msbert_hidden", 512),
    )

    print(
        f"[msbert] runtime model: vocab_size={runtime_vocab_size}, hidden={runtime_hidden}",
        flush=True,
    )

    return model, device, runtime_vocab_size, runtime_hidden


# ---------------------------------------------------------------------
# Spectrum preprocessing
# ---------------------------------------------------------------------
def _aggregate_duplicate_tokens(token_ids, intensities, mode="sum"):
    """
    Aggregate duplicate integer m/z tokens.
    """

    if token_ids.size == 0:
        return token_ids.astype(np.int64), intensities.astype(np.float32)

    unique_tokens, inverse = np.unique(token_ids, return_inverse=True)

    if mode == "max":
        agg = np.zeros((unique_tokens.shape[0],), dtype=np.float32)

        for i in range(unique_tokens.shape[0]):
            agg[i] = np.max(intensities[inverse == i])

    else:
        agg = np.zeros((unique_tokens.shape[0],), dtype=np.float32)
        np.add.at(agg, inverse, intensities.astype(np.float32, copy=False))

    return unique_tokens.astype(np.int64), agg.astype(np.float32)


def _mzs_to_token_ids(mzs, config: PipelineConfig):
    """
    Convert m/z values to integer MSBERT token ids.

    Default:
        round(m/z)

    Optional config:
        msbert_mz_to_token_method: "round" / "floor" / "ceil"
    """

    mz_min = float(get_config_value(config, "msbert_mz_min", 50.0))
    mz_bin_size = float(get_config_value(config, "msbert_mz_bin_size", 1.0))
    token_offset = int(get_config_value(config, "msbert_token_offset", 0))
    method = str(get_config_value(config, "msbert_mz_to_token_method", "round")).lower()

    if mz_bin_size <= 0:
        raise ValueError("msbert_mz_bin_size must be positive.")

    if abs(mz_bin_size - 1.0) < 1e-12:
        values = mzs
    else:
        values = (mzs - mz_min) / mz_bin_size

    if method == "floor":
        token_ids = np.floor(values)
    elif method == "ceil":
        token_ids = np.ceil(values)
    else:
        token_ids = np.rint(values)

    token_ids = token_ids.astype(np.int64) + token_offset

    return token_ids


def prepare_msbert_inputs(
    mzs,
    intensities,
    config: PipelineConfig,
    runtime_vocab_size=None,
):
    """
    Convert one MS/MS spectrum to official MSBERT inputs.

    Returns:
        input_ids:     int64 array, shape [seq_len]
        intensity_arr: float32 array, shape [seq_len]

    Later batched as:
        input_ids_tensor:  [B, L]
        intensity_tensor:  [B, 1, L]
    """

    mz_min = float(get_config_value(config, "msbert_mz_min", 50.0))
    mz_max = float(get_config_value(config, "msbert_mz_max", 1000.0))

    if runtime_vocab_size is None:
        vocab_size = int(get_config_value(config, "msbert_vocab_size", 1001))
    else:
        vocab_size = int(runtime_vocab_size)

    max_token_id = vocab_size - 1

    # 0 = padding, 1 = mask token in official model.
    # Real m/z token should usually be >= 2.
    min_token_id = int(get_config_value(config, "msbert_min_token_id", 2))

    min_peaks = int(get_config_value(config, "msbert_min_peaks", 1))
    max_peaks = int(get_config_value(config, "msbert_max_peaks", 100))
    seq_len = int(get_config_value(config, "msbert_seq_len", max_peaks))

    normalize_intensity = bool(
        get_config_value(config, "msbert_normalize_intensity", True)
    )
    intensity_power = float(
        get_config_value(config, "msbert_intensity_power", 1.0)
    )

    select_top_by = str(
        get_config_value(config, "msbert_select_top_peaks_by", "intensity")
    ).lower()

    final_sort_by = str(
        get_config_value(config, "msbert_final_sort_by", "mz")
    ).lower()

    duplicate_mode = str(
        get_config_value(config, "msbert_duplicate_token_aggregate", "sum")
    ).lower()

    skip_invalid = bool(
        get_config_value(config, "msbert_skip_invalid_spectra", False)
    )

    renormalize_after_aggregate = bool(
        get_config_value(config, "msbert_renormalize_after_aggregate", False)
    )

    mzs = np.asarray(mzs, dtype=np.float32).reshape(-1)
    intensities = np.asarray(intensities, dtype=np.float32).reshape(-1)

    empty_input = (
        mzs.size == 0
        or intensities.size == 0
        or seq_len <= 0
    )

    if empty_input:
        if skip_invalid:
            return None

        return (
            np.zeros((seq_len,), dtype=np.int64),
            np.zeros((seq_len,), dtype=np.float32),
        )

    n = min(mzs.shape[0], intensities.shape[0])
    mzs = mzs[:n]
    intensities = intensities[:n]

    valid = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (intensities > 0)
        & (mzs >= mz_min)
        & (mzs <= mz_max)
    )

    mzs = mzs[valid]
    intensities = intensities[valid]

    if mzs.size < min_peaks:
        if skip_invalid:
            return None

    input_ids = np.zeros((seq_len,), dtype=np.int64)
    intensity_arr = np.zeros((seq_len,), dtype=np.float32)

    if mzs.size == 0:
        return input_ids, intensity_arr

    token_ids = _mzs_to_token_ids(mzs, config)

    token_valid = (
        (token_ids >= min_token_id)
        & (token_ids <= max_token_id)
    )

    token_ids = token_ids[token_valid]
    intensities = intensities[token_valid]

    if token_ids.size == 0:
        if skip_invalid:
            return None

        return input_ids, intensity_arr

    if normalize_intensity:
        max_i = float(np.max(intensities))

        if max_i > 0:
            intensities = intensities / max_i

    if intensity_power != 1.0:
        intensities = np.power(intensities, intensity_power)

    token_ids, intensities = _aggregate_duplicate_tokens(
        token_ids=token_ids,
        intensities=intensities,
        mode=duplicate_mode,
    )

    if renormalize_after_aggregate and intensities.size > 0:
        max_i = float(np.max(intensities))

        if max_i > 0:
            intensities = intensities / max_i

    # Select top peaks.
    if max_peaks > 0 and token_ids.size > max_peaks:
        if select_top_by == "mz":
            order = np.argsort(token_ids)
        else:
            order = np.argsort(-intensities)

        keep = order[:max_peaks]
        token_ids = token_ids[keep]
        intensities = intensities[keep]

    # Final ordering before model input.
    if final_sort_by == "intensity":
        order = np.argsort(-intensities)
    else:
        order = np.argsort(token_ids)

    token_ids = token_ids[order]
    intensities = intensities[order]

    take = min(seq_len, token_ids.shape[0])

    if take > 0:
        input_ids[:take] = token_ids[:take]
        intensity_arr[:take] = intensities[:take]

    return input_ids, intensity_arr


# ---------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------
def infer_msbert_embedding_dim(
    model,
    device,
    config: PipelineConfig,
    runtime_vocab_size=None,
):
    """
    Infer output embedding dimension from model.predict().
    """

    seq_len = int(
        get_config_value(
            config,
            "msbert_seq_len",
            get_config_value(config, "msbert_max_peaks", 100),
        )
    )

    min_token_id = int(get_config_value(config, "msbert_min_token_id", 2))

    if runtime_vocab_size is None:
        runtime_vocab_size = get_model_vocab_size(
            model,
            fallback=get_config_value(config, "msbert_vocab_size", 1001),
        )

    safe_token_id = min(max(min_token_id, 0), int(runtime_vocab_size) - 1)

    input_ids = torch.zeros(
        (1, seq_len),
        dtype=torch.long,
        device=device,
    )

    intensity = torch.zeros(
        (1, 1, seq_len),
        dtype=torch.float32,
        device=device,
    )

    if seq_len > 0:
        input_ids[0, 0] = safe_token_id
        intensity[0, 0, 0] = 1.0

    with torch.inference_mode():
        output = model.predict(input_ids, intensity)

    if isinstance(output, (tuple, list)):
        output = output[0]

    if not isinstance(output, torch.Tensor):
        raise ValueError(
            f"model.predict() returned unsupported type: {type(output)}"
        )

    if output.ndim == 3:
        output = output.squeeze(1)

    if output.ndim != 2:
        raise ValueError(
            f"Invalid MSBERT predict output shape: {tuple(output.shape)}"
        )

    emb_dim = int(output.shape[1])

    # del input_ids
    # del intensity
    # del output

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))
        torch.cuda.empty_cache()

    return emb_dim


def _safe_normalize_embeddings(embeddings):
    """
    L2 normalize embeddings safely.

    If project utils.l2_normalize handles zero rows, this is redundant.
    Kept here to prevent NaN rows from empty spectra.
    """

    embeddings = np.asarray(embeddings)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    valid = norms.squeeze(1) > 0

    out = embeddings.copy()

    out[valid] = out[valid] / norms[valid]

    return out


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------
def build_msbert_embeddings(msdata, config: PipelineConfig, limit=None):
    """
    Build MSBERT embeddings for spectra in msdata.

    Pipeline:
        spectrum peaks
        -> integer m/z token ids
        -> intensity vector
        -> model.predict(input_ids, intensity)
        -> embedding matrix
        -> L2 normalization
    """

    model, device, runtime_vocab_size, runtime_hidden = load_msbert_model(config)

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(torch.device(device))
        log_cuda_memory(prefix="[msbert] after model load", device=device)

    n_total = get_msdata_length(msdata)
    n = n_total if limit is None else min(int(limit), n_total)

    batch_size = int(get_config_value(config, "msbert_batch_size", 512))
    batch_size = max(batch_size, 1)

    progress_bar = bool(get_config_value(config, "msbert_progress_bar", True))
    progress_every = int(get_config_value(config, "msbert_progress_every", 5000))

    use_amp = bool(get_config_value(config, "msbert_use_amp", True))
    amp_dtype_name = str(get_config_value(config, "msbert_amp_dtype", "float16")).lower()

    if amp_dtype_name == "bfloat16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16

    amp_enabled = use_amp and str(device).startswith("cuda")

    amp_cache_enabled = bool(
        get_config_value(config, "msbert_amp_cache_enabled", False)
    )

    cuda_empty_cache_every = int(
        get_config_value(config, "msbert_cuda_empty_cache_every", 20)
    )

    log_gpu_memory_every = int(
        get_config_value(config, "msbert_log_gpu_memory_every", 50)
    )

    embedding_dim = infer_msbert_embedding_dim(
        model=model,
        device=device,
        config=config,
        runtime_vocab_size=runtime_vocab_size,
    )

    print(
        f"[msbert] n_spectra={n}, "
        f"embedding_dim={embedding_dim}, "
        f"runtime_hidden={runtime_hidden}, "
        f"runtime_vocab_size={runtime_vocab_size}, "
        f"batch_size={batch_size}, "
        f"amp={amp_enabled}, "
        f"amp_cache_enabled={amp_cache_enabled}",
        flush=True,
    )

    embeddings = np.zeros((n, embedding_dim), dtype=np.float32)

    input_id_batch = []
    intensity_batch = []
    pos_batch = []

    failed = 0
    invalid = 0
    flush_count = 0

    t0 = time.time()

    def flush_batch():
        nonlocal input_id_batch
        nonlocal intensity_batch
        nonlocal pos_batch
        nonlocal embeddings
        nonlocal flush_count

        if len(pos_batch) == 0:
            return

        flush_count += 1

        input_ids_np = np.stack(input_id_batch, axis=0).astype(np.int64, copy=False)
        intensity_np = np.stack(intensity_batch, axis=0).astype(np.float32, copy=False)

        input_ids_tensor = None
        intensity_tensor = None
        output = None
        output_cpu = None
        output_np = None

        try:
            input_ids_tensor = torch.from_numpy(input_ids_np).to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            intensity_tensor = torch.from_numpy(intensity_np[:, None, :]).to(
                device=device,
                dtype=torch.float32,
                non_blocking=True,
            )

            with torch.inference_mode():
                if amp_enabled:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=amp_dtype,
                        cache_enabled=amp_cache_enabled,
                    ):
                        output = model.predict(input_ids_tensor, intensity_tensor)
                else:
                    output = model.predict(input_ids_tensor, intensity_tensor)

            if isinstance(output, (tuple, list)):
                output = output[0]

            if not isinstance(output, torch.Tensor):
                raise ValueError(
                    f"MSBERT output is not tensor: {type(output)}"
                )

            if output.ndim == 3:
                output = output.squeeze(1)

            if output.ndim != 2:
                raise ValueError(
                    f"Invalid MSBERT output shape: {tuple(output.shape)}"
                )

            if output.shape[1] != embedding_dim:
                raise ValueError(
                    f"MSBERT embedding dim changed: "
                    f"expected {embedding_dim}, got {output.shape[1]}"
                )

            # Move to CPU immediately and copy to detach from temporary tensor storage.
            output_cpu = output.detach().to("cpu", dtype=torch.float32)
            output_np = output_cpu.numpy().copy()

            embeddings[np.asarray(pos_batch, dtype=np.int64), :] = output_np

        finally:
            # Clear GPU tensor references.
            del input_ids_tensor
            del intensity_tensor
            del output

            # Clear CPU temporaries.
            del output_cpu
            del output_np
            del input_ids_np
            del intensity_np

            input_id_batch = []
            intensity_batch = []
            pos_batch = []

            if str(device).startswith("cuda") and torch.cuda.is_available():
                need_sync = False

                if log_gpu_memory_every > 0 and flush_count % log_gpu_memory_every == 0:
                    need_sync = True

                if cuda_empty_cache_every > 0 and flush_count % cuda_empty_cache_every == 0:
                    need_sync = True

                if need_sync:
                    try:
                        torch.cuda.synchronize(torch.device(device))
                    except Exception:
                        pass

                if log_gpu_memory_every > 0 and flush_count % log_gpu_memory_every == 0:
                    log_cuda_memory(
                        prefix=f"[msbert] after batch {flush_count}",
                        device=device,
                    )

                if cuda_empty_cache_every > 0 and flush_count % cuda_empty_cache_every == 0:
                    torch.cuda.empty_cache()

    if progress_bar and tqdm is not None:
        iterator = tqdm(
            range(n),
            desc="msbert encode",
            unit="spectra",
            dynamic_ncols=True,
        )
    else:
        iterator = range(n)

    for i in iterator:
        try:
            mzs, intensities = get_spectrum_peaks(msdata, i, config)

            prepared = prepare_msbert_inputs(
                mzs=mzs,
                intensities=intensities,
                config=config,
                runtime_vocab_size=runtime_vocab_size,
            )

            if prepared is None:
                invalid += 1
                continue

            input_ids, intensity_arr = prepared

            input_id_batch.append(input_ids)
            intensity_batch.append(intensity_arr)
            pos_batch.append(i)

            if len(pos_batch) >= batch_size:
                flush_batch()

        except Exception as e:
            failed += 1

            if failed <= 20:
                print(
                    f"[msbert] WARNING failed at index {i}: {repr(e)}",
                    flush=True,
                )

        if not (progress_bar and tqdm is not None):
            done = i + 1

            if progress_every > 0 and (
                done % progress_every == 0 or done == n
            ):
                elapsed = time.time() - t0
                speed = done / elapsed if elapsed > 0 else 0.0
                eta = (n - done) / speed if speed > 0 else 0.0

                print(
                    f"[msbert] {done}/{n} "
                    f"({100.0 * done / max(n, 1):.2f}%), "
                    f"failed={failed}, invalid={invalid}, "
                    f"speed={speed:.1f} spectra/s, eta={eta / 60:.1f} min",
                    flush=True,
                )

    flush_batch()

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
        log_cuda_memory(prefix="[msbert] before normalization", device=device)

    print("[msbert] L2 normalizing embeddings...", flush=True)

    # Use safe normalization to avoid NaNs from zero rows.
    embeddings = _safe_normalize_embeddings(embeddings).astype(np.float32, copy=False)

    embeddings = cast_embedding_dtype(embeddings, config)

    elapsed = time.time() - t0

    print(
        f"[msbert] finished. "
        f"shape={embeddings.shape}, dtype={embeddings.dtype}, "
        f"failed={failed}, invalid={invalid}, elapsed={elapsed:.1f}s",
        flush=True,
    )

    if str(device).startswith("cuda") and torch.cuda.is_available():
        log_cuda_memory(prefix="[msbert] finished", device=device)

    return embeddings
def build_msbert_embeddings_from_mzml(
    mzml_path,
    output_path,
    config,
    *,
    ms_level=2,
    max_spectra=None,
    limit=None,
    batch_size=None,
):
    """
    Direct MSBERT mzML inference.

    This function does NOT call build_msbert_embeddings(...).
    """

    try:
        from mzml_input import load_mzml_spectra, save_mzml_embeddings_npz
    except ImportError:
        from .mzml_input import load_mzml_spectra, save_mzml_embeddings_npz

    device = get_msbert_device(config)

    model = load_msbert_model(config)

    if isinstance(model, tuple):
        model = model[0]

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "load_msbert_model(config) must return torch.nn.Module "
            f"or tuple whose first item is torch.nn.Module, got {type(model)}"
        )

    model = model.to(device)
    model.eval()

    vocab_size = get_model_vocab_size(
        model,
        fallback=int(get_config_value(config, "msbert_vocab_size", 1001)),
    )

    max_len = int(get_config_value(config, "msbert_max_len", 100))

    if batch_size is None:
        batch_size = int(get_config_value(config, "msbert_batch_size", 64))

    mz_bin_size = float(get_config_value(config, "msbert_mz_bin_size", 1.0))
    mz_min = float(get_config_value(config, "msbert_mz_min", 0.0))

    if mz_bin_size <= 0:
        raise ValueError(f"msbert_mz_bin_size must be positive, got {mz_bin_size}")

    spectra = load_mzml_spectra(
        mzml_path,
        ms_level=ms_level,
        max_spectra=max_spectra,
        dtype=np.float32,
    )

    if limit is not None:
        spectra = spectra[: int(limit)]

    def make_batch(batch_spectra):
        bsz = len(batch_spectra)

        input_id = np.zeros((bsz, max_len), dtype=np.int64)
        intensity = np.zeros((bsz, 1, max_len), dtype=np.float32)

        for row, spec in enumerate(batch_spectra):
            mzs = np.asarray(spec["mz"], dtype=np.float32).reshape(-1)
            intensities = np.asarray(spec["intensity"], dtype=np.float32).reshape(-1)

            valid = (
                np.isfinite(mzs)
                & np.isfinite(intensities)
                & (mzs > 0)
                & (intensities > 0)
            )

            mzs = mzs[valid]
            intensities = intensities[valid]

            if mzs.size == 0:
                continue

            if mzs.size > max_len:
                order = np.argsort(intensities)[::-1][:max_len]
                mzs = mzs[order]
                intensities = intensities[order]

            order = np.argsort(mzs)
            mzs = mzs[order]
            intensities = intensities[order]

            token_ids = np.floor((mzs - mz_min) / mz_bin_size).astype(np.int64) + 1
            token_ids = np.clip(token_ids, 1, vocab_size - 1)

            max_intensity = np.max(intensities)

            if max_intensity > 0:
                intensities = intensities / max_intensity

            n = min(token_ids.size, max_len)

            input_id[row, :n] = token_ids[:n]
            intensity[row, 0, :n] = intensities[:n]

        input_id = torch.from_numpy(input_id).to(device=device, dtype=torch.long)
        intensity = torch.from_numpy(intensity).to(device=device, dtype=torch.float32)

        return input_id, intensity

    all_embeddings = []

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(spectra), batch_size),
            desc="[MSBERT-mzML] Encoding",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + batch_size, len(spectra))
            batch_spectra = spectra[start:end]

            input_id, intensity = make_batch(batch_spectra)

            if hasattr(model, "predict"):
                output = model.predict(input_id, intensity)
            else:
                output = model(input_id, intensity)

            if isinstance(output, tuple) or isinstance(output, list):
                output = output[0]

            if isinstance(output, dict):
                for key in ["embedding", "embeddings", "z", "output"]:
                    if key in output:
                        output = output[key]
                        break

            output = output.detach().cpu().numpy()
            all_embeddings.append(output)

    embeddings = np.vstack(all_embeddings)

    normalize = bool(get_config_value(config, "msbert_normalize_embeddings", True))

    if normalize:
        embeddings = l2_normalize(embeddings)

    embeddings = cast_embedding_dtype(embeddings, config)

    saved_path = save_mzml_embeddings_npz(
        output_path,
        embeddings,
        spectra,
        mzml_path=mzml_path,
        method="msbert",
        extra={
            "vocab_size": vocab_size,
            "max_len": max_len,
            "mz_bin_size": mz_bin_size,
            "mz_min": mz_min,
        },
    )

    print(f"[MSBERT-mzML] Saved embeddings to: {saved_path}")

    return embeddings

