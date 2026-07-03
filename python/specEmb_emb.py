# -*- coding: utf-8 -*-
"""
SpecEmbedding embedding builder for the provided SiameseModel.

Expected external model.py contains:

    class SiameseModel(nn.Module):
        def __init__(
            self,
            embedding_dim: int,
            n_head: int,
            n_layer: int,
            dim_feedward: int,
            dim_target: int,
            *,
            lambda_params=(1e-3, 1e3),
            feedward_activation='relu',
            dropout=0.1,
            dropout_last_layer=False,
            norm_first=True,
        )

        def forward(self, mz, intensity, mask):
            ...

This module converts each spectrum into:
    mz        : torch.FloatTensor, shape [B, L]
    intensity : torch.FloatTensor, shape [B, L]
    mask      : torch.BoolTensor,  shape [B, L]
                True  = padding
                False = valid peak

Public functions:
    build_specEmb_embeddings(...)
    build_specemb_embeddings(...)
    build_spec_embedding_embeddings(...)
    load_specEmb_model(...)
"""

import os
import re
import sys
import time
import math
import traceback
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED

import numpy as np

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

try:
    import torch
except ImportError as e:
    raise ImportError("SpecEmbedding requires PyTorch. Please install torch first.") from e

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
    try:
        from .config import PipelineConfig
    except ImportError:
        PipelineConfig = None

try:
    from spec2vec_emb import (
        get_config_value,
        get_msdata_length,
        get_spectrum_peaks,
        cast_embedding_dtype,
    )
except ImportError:
    try:
        from .spec2vec_emb import (
            get_config_value,
            get_msdata_length,
            get_spectrum_peaks,
            cast_embedding_dtype,
        )
    except ImportError as e:
        raise ImportError(
            "Cannot import required helpers from spec2vec_emb.py."
        ) from e

try:
    from io_utils import load_mona_library
except ImportError:
    try:
        from .io_utils import load_mona_library
    except ImportError:
        load_mona_library = None

try:
    from utils import l2_normalize
except ImportError:
    try:
        from .utils import l2_normalize
    except ImportError:
        def l2_normalize(x, eps=1e-12):
            x = np.asarray(x)
            norm = np.linalg.norm(x, axis=1, keepdims=True)
            return x / np.maximum(norm, eps)


__all__ = [
    "build_specEmb_embeddings",
    "build_specemb_embeddings",
    "build_spec_embedding_embeddings",
    "load_specEmb_model",
    "spectrum_to_specEmb_input",
    "prepare_specEmb_batch_from_msdata",
]


# ---------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------
def _cfg(config, keys, default=None):
    if isinstance(keys, str):
        keys = [keys]

    for key in keys:
        value = None

        try:
            value = get_config_value(config, key, None)
        except Exception:
            value = None

        if value is not None:
            return value

        if isinstance(config, dict) and key in config:
            value = config[key]

            if value is not None:
                return value

        if hasattr(config, key):
            value = getattr(config, key)

            if value is not None:
                return value

    return default


def _as_bool(x, default=False):
    if x is None:
        return default

    if isinstance(x, bool):
        return x

    if isinstance(x, (int, np.integer)):
        return bool(x)

    if isinstance(x, str):
        return x.strip().lower() in {"1", "true", "yes", "y", "on"}

    return bool(x)


def _cuda_synchronize_if_needed(device):
    device = torch.device(device)

    if torch.cuda.is_available() and device.type == "cuda":
        torch.cuda.synchronize()


def _resolve_device(config):
    device_name = _cfg(
        config,
        [
            "specemb_device",
            "specEmb_device",
            "specembedding_device",
            "embedding_device",
            "device",
        ],
        "cuda" if torch.cuda.is_available() else "cpu",
    )

    device = torch.device(device_name)

    if device.type == "cuda" and not torch.cuda.is_available():
        print("[SpecEmbedding] CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    return device


def _resolve_output_dtype(config):
    dtype_name = _cfg(
        config,
        [
            "specemb_embedding_dtype",
            "specEmb_embedding_dtype",
            "specembedding_embedding_dtype",
            "embedding_dtype",
            "emb_dtype",
        ],
        "float32",
    )

    try:
        return np.dtype(dtype_name)
    except Exception:
        print(f"[SpecEmbedding] Invalid dtype {dtype_name!r}. Falling back to float32.")
        return np.dtype("float32")


def _maybe_configured_limit(config):
    limit = _cfg(
        config,
        [
            "specemb_limit",
            "specEmb_limit",
            "specembedding_limit",
            "embedding_limit",
            "limit",
            "library_limit",
        ],
        None,
    )

    if limit is None:
        return None

    try:
        limit = int(limit)
    except Exception:
        return None

    if limit <= 0:
        return None

    return limit


def _config_to_plain_dict(config):
    if config is None:
        return {}

    if isinstance(config, dict):
        raw = dict(config)
    else:
        raw = {}

        if hasattr(config, "__dict__"):
            for k, v in vars(config).items():
                if k.startswith("_"):
                    continue

                if callable(v):
                    continue

                raw[k] = v

    clean = {}

    for k, v in raw.items():
        if isinstance(v, Path):
            clean[k] = str(v)
        else:
            clean[k] = v

    return clean


def _worker_get_config_value(config, key, default=None):
    if config is None:
        return default

    if isinstance(config, dict):
        return config.get(key, default)

    return getattr(config, key, default)


# ---------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------
def _resolve_specemb_model_file(config):
    model_file = _cfg(
        config,
        [
            "specemb_model_file",
            "specEmb_model_file",
            "specembedding_model_file",
            "specemb_model_py",
            "specEmb_model_py",
        ],
        None,
    )

    if model_file is not None:
        model_file = Path(model_file).expanduser().resolve()

        if not model_file.exists():
            raise FileNotFoundError(f"SpecEmbedding model file not found: {model_file}")

        return model_file

    root = _cfg(
        config,
        [
            "specemb_root",
            "specEmb_root",
            "specembedding_root",
            "specemb_code_dir",
            "specEmb_code_dir",
        ],
        None,
    )

    if root is None:
        return None

    root = Path(root).expanduser().resolve()

    if root.is_file():
        return root

    candidate_names = [
        "model.py",
        "SpecEmbeddingModel.py",
        "SpecEmbedding.py",
        "spec_embedding.py",
        "specembedding.py",
    ]

    for name in candidate_names:
        candidate = root / name

        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        f"Cannot find SpecEmbedding model file under {root}. "
        f"Tried: {candidate_names}"
    )


def _resolve_specemb_checkpoint_path(config):
    checkpoint_path = _cfg(
        config,
        [
            "specemb_checkpoint_path",
            "specEmb_checkpoint_path",
            "specembedding_checkpoint_path",
            "specemb_model_path",
            "specEmb_model_path",
            "specembedding_model_path",
            "specemb_ckpt_path",
            "specEmb_ckpt_path",
            "specemb_weights_path",
            "specEmb_weights_path",
        ],
        None,
    )

    if checkpoint_path is None:
        return None

    checkpoint_path = Path(checkpoint_path).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"SpecEmbedding checkpoint not found: {checkpoint_path}"
        )

    return checkpoint_path


def _import_module_from_file(model_file):
    model_file = Path(model_file).expanduser().resolve()
    module_dir = str(model_file.parent)

    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    module_name = f"_specembedding_external_{abs(hash(str(model_file)))}"

    spec = importlib.util.spec_from_file_location(module_name, str(model_file))

    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create import spec for: {model_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    return module


def _get_model_class_from_module(module, config):
    requested_class = _cfg(
        config,
        [
            "specemb_model_class",
            "specEmb_model_class",
            "specembedding_model_class",
        ],
        None,
    )

    if requested_class is not None and hasattr(module, requested_class):
        return getattr(module, requested_class), requested_class

    fallback_names = [
        "SiameseModel",
        "SpecEmbedding",
        "SpectrumEmbedding",
        "SpecEmb",
        "SpecEmbeddingModel",
        "Model",
    ]

    for name in fallback_names:
        if hasattr(module, name):
            if requested_class is not None and requested_class != name:
                print(
                    f"[SpecEmbedding] Requested class {requested_class!r} not found. "
                    f"Using fallback class {name!r}."
                )

            return getattr(module, name), name

    available = [x for x in dir(module) if not x.startswith("_")]

    raise ImportError(
        f"Cannot find SpecEmbedding/SiameseModel class in module.\n"
        f"Requested class: {requested_class!r}\n"
        f"Available public names: {available[:80]}"
    )


def _torch_load_checkpoint(checkpoint_path, device):
    try:
        return torch.load(checkpoint_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(checkpoint_path, map_location=device)


def _extract_state_dict_or_model(checkpoint):
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint

    if not isinstance(checkpoint, dict):
        return checkpoint

    for key in [
        "model",
        "module",
        "net",
        "network",
        "siamese_model",
        "specembedding",
        "specEmb",
    ]:
        value = checkpoint.get(key, None)

        if isinstance(value, torch.nn.Module):
            return value

    for key in [
        "state_dict",
        "model_state_dict",
        "net_state_dict",
        "module_state_dict",
        "weights",
    ]:
        value = checkpoint.get(key, None)

        if isinstance(value, dict):
            return value

    return checkpoint


def _strip_prefix_if_present(state_dict, prefix):
    if not isinstance(state_dict, dict):
        return state_dict

    keys = list(state_dict.keys())

    if len(keys) == 0:
        return state_dict

    n_with_prefix = sum(str(k).startswith(prefix) for k in keys)

    if n_with_prefix == 0:
        return state_dict

    # Only strip when most keys have this prefix.
    if n_with_prefix < max(1, int(0.5 * len(keys))):
        return state_dict

    new_state = {}

    for k, v in state_dict.items():
        k_str = str(k)

        if k_str.startswith(prefix):
            k_str = k_str[len(prefix):]

        new_state[k_str] = v

    return new_state


def _canonicalize_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict

    prefixes = [
        "module.",
        "model.",
        "net.",
        "network.",
        "siamese_model.",
        "_model.",
        "encoder.",
    ]

    out = dict(state_dict)

    changed = True

    while changed:
        changed = False

        for prefix in prefixes:
            new_out = _strip_prefix_if_present(out, prefix)

            if new_out is not out and list(new_out.keys()) != list(out.keys()):
                out = new_out
                changed = True

    return out


def _get_checkpoint_hparams(checkpoint):
    if not isinstance(checkpoint, dict):
        return {}

    for key in [
        "hyper_parameters",
        "hparams",
        "config",
        "args",
        "model_hparams",
    ]:
        value = checkpoint.get(key, None)

        if isinstance(value, dict):
            return value

    return {}


def _infer_siamese_arch_from_state_dict(state_dict):
    """
    Infer as much as possible from SiameseModel state_dict.

    n_head cannot be reliably inferred from state_dict because
    MultiheadAttention projection weights depend on embedding_dim, not n_head.
    """
    arch = {}

    if not isinstance(state_dict, dict):
        return arch

    # embedding_dim from Transformer self-attention.
    for key in [
        "_encoder.layers.0.self_attn.in_proj_weight",
        "transformer_encoder.layers.0.self_attn.in_proj_weight",
    ]:
        if key in state_dict:
            weight = state_dict[key]

            if hasattr(weight, "shape") and len(weight.shape) == 2:
                arch["embedding_dim"] = int(weight.shape[1])
                break

    if "embedding_dim" not in arch:
        for key, weight in state_dict.items():
            key = str(key)

            if key.endswith("self_attn.in_proj_weight") and hasattr(weight, "shape"):
                if len(weight.shape) == 2:
                    arch["embedding_dim"] = int(weight.shape[1])
                    break

    # dim_feedward from Transformer linear1.
    for key, weight in state_dict.items():
        key = str(key)

        if key.endswith("linear1.weight") and "_encoder.layers.0" in key:
            if hasattr(weight, "shape") and len(weight.shape) == 2:
                arch["dim_feedward"] = int(weight.shape[0])
                break

    if "dim_feedward" not in arch:
        for key, weight in state_dict.items():
            key = str(key)

            if key.endswith("linear1.weight") and hasattr(weight, "shape"):
                if len(weight.shape) == 2:
                    arch["dim_feedward"] = int(weight.shape[0])
                    break

    # n_layer from _encoder.layers.{i}.
    # n_layer from encoder layer keys.
    layer_ids = set()

    for key in state_dict.keys():
        key = str(key)

        patterns = [
            r"_encoder\.layers\.(\d+)\.",
            r"encoder\.layers\.(\d+)\.",
            r"transformer_encoder\.layers\.(\d+)\.",
            r"module\._encoder\.layers\.(\d+)\.",
            r"model\._encoder\.layers\.(\d+)\.",
        ]

        for pattern in patterns:
            m = re.search(pattern, key)

            if m:
                layer_ids.add(int(m.group(1)))
                break

    if layer_ids:
        arch["n_layer"] = max(layer_ids) + 1

    # dim_target from last Linear in _decoder._layers.
    decoder_linear = []

    for key, weight in state_dict.items():
        key = str(key)

        m = re.search(r"_decoder\._layers\.(\d+)\.weight$", key)

        if m and hasattr(weight, "shape") and len(weight.shape) == 2:
            decoder_linear.append((int(m.group(1)), weight))

    if decoder_linear:
        decoder_linear.sort(key=lambda x: x[0])
        last_weight = decoder_linear[-1][1]
        arch["dim_target"] = int(last_weight.shape[0])

    return arch


def _choose_default_n_head(embedding_dim, preferred=8):
    embedding_dim = int(embedding_dim)

    if preferred is not None:
        preferred = int(preferred)

        if preferred > 0 and embedding_dim % preferred == 0:
            return preferred

    for h in [16, 12, 8, 6, 4, 3, 2, 1]:
        if embedding_dim % h == 0:
            return h

    return 1


def _get_siamese_init_kwargs(config, checkpoint=None, state_dict=None):
    """
    Build kwargs for SiameseModel.

    Priority:
        1. config.specemb_init_kwargs
        2. checkpoint hyper_parameters
        3. state_dict inference
        4. safe defaults
    """
    explicit = _cfg(
        config,
        [
            "specemb_init_kwargs",
            "specEmb_init_kwargs",
            "specembedding_init_kwargs",
            "specemb_model_kwargs",
            "specEmb_model_kwargs",
        ],
        None,
    )

    if explicit is not None:
        if not isinstance(explicit, dict):
            raise TypeError(
                "specemb_init_kwargs / specemb_model_kwargs must be a dict."
            )

        return dict(explicit)

    hparams = _get_checkpoint_hparams(checkpoint)
    inferred = _infer_siamese_arch_from_state_dict(state_dict)

    def get_value(out_name, config_keys, hparam_keys, default):
        value = _cfg(config, config_keys, None)

        if value is not None:
            return value

        if out_name in inferred:
            return inferred[out_name]

        for key in hparam_keys:
            if key in hparams and hparams[key] is not None:
                return hparams[key]

        return default

    embedding_dim = int(
        get_value(
            "embedding_dim",
            [
                "specemb_embedding_model_dim",
                "specemb_transformer_dim",
                "specemb_siamese_embedding_dim",
                "specEmb_embedding_model_dim",
            ],
            [
                "embedding_dim",
                "d_model",
                "hidden_size",
            ],
            512,
        )
    )

    preferred_n_head = _cfg(
        config,
        [
            "specemb_n_head",
            "specEmb_n_head",
            "specembedding_n_head",
            "specemb_num_heads",
            "specEmb_num_heads",
            "n_head",
        ],
        None,
    )

    if preferred_n_head is None:
        for key in ["n_head", "num_heads", "n_heads"]:
            if key in hparams:
                preferred_n_head = hparams[key]
                break

    n_head = _choose_default_n_head(
        embedding_dim=embedding_dim,
        preferred=preferred_n_head if preferred_n_head is not None else 8,
    )

    n_layer = int(
        get_value(
            "n_layer",
            [
                "specemb_n_layer",
                "specEmb_n_layer",
                "specembedding_n_layer",
                "specemb_num_layers",
                "specEmb_num_layers",
                "n_layer",
            ],
            [
                "n_layer",
                "num_layers",
                "n_layers",
            ],
            6,
        )
    )

    dim_feedward = int(
        get_value(
            "dim_feedward",
            [
                "specemb_dim_feedward",
                "specEmb_dim_feedward",
                "specembedding_dim_feedward",
                "specemb_dim_feedforward",
                "specEmb_dim_feedforward",
                "dim_feedward",
                "dim_feedforward",
            ],
            [
                "dim_feedward",
                "dim_feedforward",
                "dim_feed_forward",
                "ffn_dim",
            ],
            2048,
        )
    )

    dim_target = int(
        get_value(
            "dim_target",
            [
                "specemb_dim_target",
                "specEmb_dim_target",
                "specembedding_dim_target",
                "specemb_embedding_dim",
                "specEmb_embedding_dim",
                "specembedding_embedding_dim",
                "dim_target",
            ],
            [
                "dim_target",
                "embedding_output_dim",
                "output_dim",
                "embedding_dim_out",
            ],
            512,
        )
    )

    lambda_min = float(
        _cfg(
            config,
            [
                "specemb_lambda_min",
                "specEmb_lambda_min",
                "specembedding_lambda_min",
            ],
            hparams.get("lambda_min", 1e-3),
        )
    )

    lambda_max = float(
        _cfg(
            config,
            [
                "specemb_lambda_max",
                "specEmb_lambda_max",
                "specembedding_lambda_max",
            ],
            hparams.get("lambda_max", 1e3),
        )
    )

    feedward_activation = str(
        _cfg(
            config,
            [
                "specemb_feedward_activation",
                "specEmb_feedward_activation",
                "specembedding_feedward_activation",
                "specemb_activation",
                "SpecEmb_activation",
            ],
            hparams.get("feedward_activation", hparams.get("activation", "relu")),
        )
    )

    dropout = float(
        _cfg(
            config,
            [
                "specemb_dropout",
                "specEmb_dropout",
                "specembedding_dropout",
                "dropout",
            ],
            hparams.get("dropout", 0.1),
        )
    )

    dropout_last_layer = _as_bool(
        _cfg(
            config,
            [
                "specemb_dropout_last_layer",
                "specEmb_dropout_last_layer",
                "specembedding_dropout_last_layer",
            ],
            hparams.get("dropout_last_layer", False),
        ),
        False,
    )

    norm_first = _as_bool(
        _cfg(
            config,
            [
                "specemb_norm_first",
                "specEmb_norm_first",
                "specembedding_norm_first",
            ],
            hparams.get("norm_first", True),
        ),
        True,
    )

    kwargs = {
        "embedding_dim": embedding_dim,
        "n_head": n_head,
        "n_layer": n_layer,
        "dim_feedward": dim_feedward,
        "dim_target": dim_target,
        "lambda_params": (lambda_min, lambda_max),
        "feedward_activation": feedward_activation,
        "dropout": dropout,
        "dropout_last_layer": dropout_last_layer,
        "norm_first": norm_first,
    }

    return kwargs


def _instantiate_model(ModelClass, class_name, config, checkpoint=None, state_dict=None):
    if class_name == "SiameseModel":
        kwargs = _get_siamese_init_kwargs(
            config=config,
            checkpoint=checkpoint,
            state_dict=state_dict,
        )

        print("[SpecEmbedding] SiameseModel init kwargs:")
        for k, v in kwargs.items():
            print(f"  {k}: {v}")

        return ModelClass(**kwargs)

    explicit = _cfg(
        config,
        [
            "specemb_init_kwargs",
            "specEmb_init_kwargs",
            "specembedding_init_kwargs",
            "specemb_model_kwargs",
            "specEmb_model_kwargs",
        ],
        None,
    )

    if explicit is not None:
        if not isinstance(explicit, dict):
            raise TypeError("specemb_init_kwargs must be a dict.")

        try:
            return ModelClass(**dict(explicit))
        except Exception as e:
            raise RuntimeError(
                f"Failed to instantiate {class_name} with specemb_init_kwargs: {repr(e)}"
            ) from e

    try:
        return ModelClass()
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate {class_name}. "
            f"Please provide config.specemb_init_kwargs."
        ) from e


def _load_state_dict_flexible(model, state_dict, strict_preference=True):
    state_dict = _canonicalize_state_dict(state_dict)

    if not isinstance(state_dict, dict):
        raise RuntimeError(
            f"Unsupported checkpoint format. Expected state_dict-like dict, got {type(state_dict)}."
        )

    strict_preference = bool(strict_preference)

    if strict_preference:
        try:
            model.load_state_dict(state_dict, strict=True)
            print("[SpecEmbedding] Checkpoint loaded with strict=True.")
            return model
        except Exception as e:
            print("[SpecEmbedding] strict=True loading failed. Retrying strict=False.")
            print(f"[SpecEmbedding] strict=True error: {repr(e)}")

    incompatible = model.load_state_dict(state_dict, strict=False)

    missing = getattr(incompatible, "missing_keys", [])
    unexpected = getattr(incompatible, "unexpected_keys", [])

    print("[SpecEmbedding] Checkpoint loaded with strict=False.")
    print(f"[SpecEmbedding] Missing keys    : {len(missing)}")
    print(f"[SpecEmbedding] Unexpected keys : {len(unexpected)}")

    if len(missing) > 0:
        print("[SpecEmbedding] First missing keys:")
        for k in missing[:20]:
            print(f"  - {k}")

    if len(unexpected) > 0:
        print("[SpecEmbedding] First unexpected keys:")
        for k in unexpected[:20]:
            print(f"  - {k}")

    return model


def load_specEmb_model(config, device=None):
    if device is None:
        device = _resolve_device(config)

    device = torch.device(device)

    model_file = _resolve_specemb_model_file(config)
    checkpoint_path = _resolve_specemb_checkpoint_path(config)

    if model_file is None and checkpoint_path is None:
        raise ValueError(
            "SpecEmbedding requires config.specemb_model_file and/or "
            "config.specemb_checkpoint_path."
        )

    checkpoint = None
    state_or_model = None
    state_dict = None

    if checkpoint_path is not None:
        print(f"[SpecEmbedding] Loading checkpoint: {checkpoint_path}")
        checkpoint = _torch_load_checkpoint(checkpoint_path, device=device)
        state_or_model = _extract_state_dict_or_model(checkpoint)

        if isinstance(state_or_model, dict):
            state_dict = _canonicalize_state_dict(state_or_model)

    if model_file is None:
        if isinstance(state_or_model, torch.nn.Module):
            model = state_or_model
        else:
            raise ValueError(
                "No specemb_model_file was provided, and checkpoint does not contain "
                "a full torch.nn.Module."
            )
    else:
        print(f"[SpecEmbedding] Importing model file: {model_file}")
        module = _import_module_from_file(model_file)
        ModelClass, class_name = _get_model_class_from_module(module, config)

        model = _instantiate_model(
            ModelClass=ModelClass,
            class_name=class_name,
            config=config,
            checkpoint=checkpoint,
            state_dict=state_dict,
        )

        if state_dict is not None:
            strict_load = _as_bool(
                _cfg(
                    config,
                    [
                        "specemb_strict_load",
                        "specEmb_strict_load",
                        "specembedding_strict_load",
                    ],
                    True,
                ),
                True,
            )

            model = _load_state_dict_flexible(
                model=model,
                state_dict=state_dict,
                strict_preference=strict_load,
            )
        elif checkpoint_path is not None and isinstance(state_or_model, torch.nn.Module):
            model = state_or_model
        else:
            print(
                "[SpecEmbedding] WARNING: checkpoint path is missing or unsupported. "
                "Using randomly initialized model."
            )

    model = model.to(device)
    model.eval()

    print(f"[SpecEmbedding] Model ready on device: {device}")

    return model


# ---------------------------------------------------------------------
# Spectrum preprocessing
# ---------------------------------------------------------------------
def _coerce_peaks_to_numpy(peaks):
    if peaks is None:
        return np.zeros((0, 2), dtype=np.float32)

    if isinstance(peaks, np.ndarray):
        arr = peaks

        if arr.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        if arr.ndim == 1:
            arr = arr.reshape(-1, 2)

        if arr.ndim != 2 or arr.shape[1] < 2:
            raise ValueError(f"Peak array must have shape [N, >=2], got {arr.shape}")

        return arr[:, :2].astype(np.float32, copy=False)

    if isinstance(peaks, (tuple, list)) and len(peaks) == 2:
        mz = np.asarray(peaks[0], dtype=np.float32)
        intensity = np.asarray(peaks[1], dtype=np.float32)

        if mz.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.stack([mz, intensity], axis=1).astype(np.float32, copy=False)

    if isinstance(peaks, dict):
        mz_key = None
        intensity_key = None

        for k in ["mz", "mzs", "mass", "masses", "peak_mz", "mz_array"]:
            if k in peaks:
                mz_key = k
                break

        for k in [
            "intensity",
            "intensities",
            "i",
            "peak_intensity",
            "intensity_array",
        ]:
            if k in peaks:
                intensity_key = k
                break

        if mz_key is not None and intensity_key is not None:
            mz = np.asarray(peaks[mz_key], dtype=np.float32)
            intensity = np.asarray(peaks[intensity_key], dtype=np.float32)

            if mz.size == 0:
                return np.zeros((0, 2), dtype=np.float32)

            return np.stack([mz, intensity], axis=1).astype(np.float32, copy=False)

    if hasattr(peaks, "peaks"):
        return _coerce_peaks_to_numpy(peaks.peaks)

    if hasattr(peaks, "mz") and hasattr(peaks, "intensities"):
        mz = np.asarray(peaks.mz, dtype=np.float32)
        intensity = np.asarray(peaks.intensities, dtype=np.float32)

        if mz.size == 0:
            return np.zeros((0, 2), dtype=np.float32)

        return np.stack([mz, intensity], axis=1).astype(np.float32, copy=False)

    raise TypeError(f"Unsupported peaks type for SpecEmbedding: {type(peaks)}")


def _normalize_intensity(intensity, config):
    intensity = np.asarray(intensity, dtype=np.float32)

    if intensity.size == 0:
        return intensity

    mode = str(
        _cfg(
            config,
            [
                "specemb_intensity_norm",
                "specEmb_intensity_norm",
                "specembedding_intensity_norm",
            ],
            "max",
        )
    ).lower()

    power = float(
        _cfg(
            config,
            [
                "specemb_intensity_power",
                "specEmb_intensity_power",
                "specembedding_intensity_power",
                "embedding_intensity_power",
            ],
            1.0,
        )
    )

    intensity = np.maximum(intensity, 0.0)

    if power != 1.0:
        intensity = np.power(intensity, power)

    if mode in {"none", "raw", "false", "no"}:
        return intensity.astype(np.float32, copy=False)

    if mode in {"sqrt", "sqrt_max"}:
        intensity = np.sqrt(np.maximum(intensity, 0.0))

    elif mode in {"log", "log1p", "log_max"}:
        intensity = np.log1p(np.maximum(intensity, 0.0))

    if mode in {"sum", "tic"}:
        s = float(np.sum(intensity))

        if s > 0:
            intensity = intensity / s

        return intensity.astype(np.float32, copy=False)

    max_i = float(np.max(intensity)) if intensity.size else 0.0

    if max_i > 0:
        intensity = intensity / max_i

    return intensity.astype(np.float32, copy=False)


def _filter_select_sort_peaks(peaks, config):
    arr = _coerce_peaks_to_numpy(peaks)

    if arr.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    mz = arr[:, 0].astype(np.float32, copy=False)
    intensity = arr[:, 1].astype(np.float32, copy=False)

    mz_min = float(
        _cfg(
            config,
            [
                "specemb_min_mz",
                "specEmb_min_mz",
                "specembedding_min_mz",
                "embedding_mz_min",
                "min_mz",
            ],
            50.0,
        )
    )

    mz_max = float(
        _cfg(
            config,
            [
                "specemb_max_mz",
                "specEmb_max_mz",
                "specembedding_max_mz",
                "embedding_mz_max",
                "max_mz",
            ],
            1200.0,
        )
    )

    valid = (
        np.isfinite(mz)
        & np.isfinite(intensity)
        & (intensity > 0)
        & (mz >= mz_min)
        & (mz <= mz_max)
    )

    mz = mz[valid]
    intensity = intensity[valid]

    if mz.size == 0:
        return np.zeros((0, 2), dtype=np.float32)

    max_peaks = int(
        _cfg(
            config,
            [
                "specemb_max_peaks",
                "specEmb_max_peaks",
                "specembedding_max_peaks",
                "embedding_top_n_peaks",
            ],
            100,
        )
    )

    if max_peaks <= 0:
        raise ValueError(f"specemb_max_peaks must be positive, got {max_peaks}")

    select_top_by = str(
        _cfg(
            config,
            [
                "specemb_select_top_peaks_by",
                "specEmb_select_top_peaks_by",
                "specembedding_select_top_peaks_by",
            ],
            "intensity",
        )
    ).lower()

    final_sort_by = str(
        _cfg(
            config,
            [
                "specemb_final_sort_by",
                "specEmb_final_sort_by",
                "specembedding_final_sort_by",
            ],
            "mz",
        )
    ).lower()

    if mz.size > max_peaks:
        if select_top_by in {"mz", "mass"}:
            order = np.argsort(mz)[:max_peaks]
        else:
            order = np.argpartition(intensity, -max_peaks)[-max_peaks:]

        mz = mz[order]
        intensity = intensity[order]

    intensity = _normalize_intensity(intensity, config)

    if final_sort_by in {"intensity", "i"}:
        order = np.argsort(-intensity)
    else:
        order = np.argsort(mz)

    mz = mz[order]
    intensity = intensity[order]

    return np.stack([mz, intensity], axis=1).astype(np.float32, copy=False)


def spectrum_to_specEmb_input(peaks, config):
    """
    Convert one spectrum to SiameseModel input.

    Returns dict:
        mz        : [L]
        intensity : [L]
        mask      : [L], bool, True means padding.
    """
    arr = _filter_select_sort_peaks(peaks, config)

    max_peaks = int(
        _cfg(
            config,
            [
                "specemb_max_peaks",
                "specEmb_max_peaks",
                "specembedding_max_peaks",
                "embedding_top_n_peaks",
            ],
            100,
        )
    )

    mz_out = np.zeros((max_peaks,), dtype=np.float32)
    intensity_out = np.zeros((max_peaks,), dtype=np.float32)
    mask_out = np.ones((max_peaks,), dtype=np.bool_)

    if arr.size == 0:
        # Avoid division by zero in SiameseModel mean pooling.
        mask_out[0] = False
        return {
            "mz": mz_out,
            "intensity": intensity_out,
            "mask": mask_out,
        }

    length = min(arr.shape[0], max_peaks)

    mz_out[:length] = arr[:length, 0]
    intensity_out[:length] = arr[:length, 1]
    mask_out[:length] = False

    return {
        "mz": mz_out,
        "intensity": intensity_out,
        "mask": mask_out,
    }


def prepare_specEmb_batch_from_msdata(msdata, batch_indices, config):
    mz_list = []
    intensity_list = []
    mask_list = []

    for idx in batch_indices:
        peaks = get_spectrum_peaks(msdata, int(idx), config)
        item = spectrum_to_specEmb_input(peaks, config)

        mz_list.append(item["mz"])
        intensity_list.append(item["intensity"])
        mask_list.append(item["mask"])

    batch = {
        "mz": np.stack(mz_list, axis=0).astype(np.float32, copy=False),
        "intensity": np.stack(intensity_list, axis=0).astype(np.float32, copy=False),
        "mask": np.stack(mask_list, axis=0).astype(np.bool_, copy=False),
    }

    return batch


# ---------------------------------------------------------------------
# Tensor conversion and prediction
# ---------------------------------------------------------------------
def _numpy_batch_to_torch(batch_np, device, pin_memory=False):
    device = torch.device(device)

    mz = torch.as_tensor(batch_np["mz"], dtype=torch.float32)
    intensity = torch.as_tensor(batch_np["intensity"], dtype=torch.float32)
    mask = torch.as_tensor(batch_np["mask"], dtype=torch.bool)

    if pin_memory and torch.cuda.is_available() and device.type == "cuda":
        mz = mz.pin_memory()
        intensity = intensity.pin_memory()
        mask = mask.pin_memory()

    mz = mz.to(device, non_blocking=True)
    intensity = intensity.to(device, non_blocking=True)
    mask = mask.to(device, non_blocking=True)

    return {
        "mz": mz,
        "intensity": intensity,
        "mask": mask,
    }


def _extract_output_array(output):
    if isinstance(output, torch.Tensor):
        arr = output.detach().float().cpu().numpy()
    elif isinstance(output, np.ndarray):
        arr = output.astype(np.float32, copy=False)
    elif isinstance(output, dict):
        for key in [
            "embedding",
            "embeddings",
            "spectrum_embedding",
            "spec_embedding",
            "features",
            "z",
            "output",
        ]:
            if key in output:
                return _extract_output_array(output[key])

        raise ValueError(
            f"SpecEmbedding output is dict but no known embedding key found: "
            f"{list(output.keys())}"
        )
    elif isinstance(output, (tuple, list)):
        if len(output) == 0:
            raise ValueError("SpecEmbedding output is empty list/tuple.")

        return _extract_output_array(output[0])
    else:
        raise TypeError(f"Unsupported SpecEmbedding output type: {type(output)}")

    if arr.ndim == 1:
        arr = arr[None, :]

    if arr.ndim == 2:
        return arr.astype(np.float32, copy=False)

    if arr.ndim == 3:
        if arr.shape[1] == 1:
            return arr[:, 0, :].astype(np.float32, copy=False)

        return arr.mean(axis=1).astype(np.float32, copy=False)

    return arr.reshape(arr.shape[0], -1).astype(np.float32, copy=False)


def _predict_specEmb_batch(model, batch, device, config, use_amp=False):
    device = torch.device(device)

    forward_method = _cfg(
        config,
        [
            "specemb_forward_method",
            "specEmb_forward_method",
            "specembedding_forward_method",
        ],
        None,
    )

    if forward_method is not None:
        if not hasattr(model, forward_method):
            raise AttributeError(
                f"SpecEmbedding model has no method {forward_method!r}."
            )

        fn = getattr(model, forward_method)
    else:
        fn = model

    if use_amp and device.type == "cuda":
        amp_dtype_name = str(
            _cfg(
                config,
                [
                    "specemb_amp_dtype",
                    "specEmb_amp_dtype",
                    "specembedding_amp_dtype",
                ],
                "float16",
            )
        ).lower()

        amp_dtype = torch.bfloat16 if amp_dtype_name == "bfloat16" else torch.float16

        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            output = fn(
                batch["mz"],
                batch["intensity"],
                batch["mask"],
            )
    else:
        output = fn(
            batch["mz"],
            batch["intensity"],
            batch["mask"],
        )

    emb_np = _extract_output_array(output)

    if emb_np.ndim != 2:
        raise ValueError(
            f"SpecEmbedding embedding must be 2D [B, D], got {emb_np.shape}"
        )

    return emb_np.astype(np.float32, copy=False)


# ---------------------------------------------------------------------
# Multiprocessing workers
# ---------------------------------------------------------------------
_SPECEMB_CPU_WORKER_MSDATA = None
_SPECEMB_CPU_WORKER_CONFIG = None


def _specEmb_cpu_worker_init(config_dict):
    global _SPECEMB_CPU_WORKER_MSDATA
    global _SPECEMB_CPU_WORKER_CONFIG

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    _SPECEMB_CPU_WORKER_CONFIG = SimpleNamespace(**dict(config_dict))

    if load_mona_library is None:
        raise RuntimeError(
            "load_mona_library is unavailable. "
            "Cannot use multiprocessing SpecEmbedding preprocessing."
        )

    mona_path = _worker_get_config_value(
        _SPECEMB_CPU_WORKER_CONFIG,
        "mona_hdf5_path",
        None,
    )

    if mona_path is None:
        mona_path = _worker_get_config_value(
            _SPECEMB_CPU_WORKER_CONFIG,
            "library_hdf5_path",
            None,
        )

    if mona_path is None:
        mona_path = _worker_get_config_value(
            _SPECEMB_CPU_WORKER_CONFIG,
            "mona_path",
            None,
        )

    if mona_path is None:
        raise ValueError(
            "Multiprocessing SpecEmbedding preprocessing requires "
            "config.mona_hdf5_path or config.library_hdf5_path."
        )

    _SPECEMB_CPU_WORKER_MSDATA = load_mona_library(Path(mona_path))

    print(
        f"[SpecEmbedding CPU worker] pid={os.getpid()}, opened library: {mona_path}",
        flush=True,
    )


def _specEmb_prepare_batch_worker(batch_indices):
    global _SPECEMB_CPU_WORKER_MSDATA
    global _SPECEMB_CPU_WORKER_CONFIG

    if _SPECEMB_CPU_WORKER_MSDATA is None:
        raise RuntimeError("Worker msdata is not initialized.")

    batch_indices = [int(x) for x in batch_indices]

    batch = prepare_specEmb_batch_from_msdata(
        _SPECEMB_CPU_WORKER_MSDATA,
        batch_indices,
        _SPECEMB_CPU_WORKER_CONFIG,
    )

    return {
        "indices": np.asarray(batch_indices, dtype=np.int64),
        "batch": batch,
    }


def _iter_batch_indices(n_spectra, batch_size):
    for start in range(0, n_spectra, batch_size):
        end = min(start + batch_size, n_spectra)
        yield list(range(start, end))


# ---------------------------------------------------------------------
# Embedding loops
# ---------------------------------------------------------------------
def _build_specEmb_embeddings_single_loop(
    model,
    msdata,
    n_spectra,
    config,
    device,
    embedding_dim=None,
    dtype=np.float32,
):
    batch_size = int(
        _cfg(
            config,
            [
                "specemb_batch_size",
                "specEmb_batch_size",
                "specembedding_batch_size",
                "embedding_batch_size",
                "batch_size",
            ],
            1024,
        )
    )

    use_amp = _as_bool(
        _cfg(
            config,
            [
                "specemb_use_amp",
                "specEmb_use_amp",
                "specembedding_use_amp",
                "use_amp",
            ],
            False,
        ),
        False,
    )

    do_profile = _as_bool(
        _cfg(
            config,
            [
                "specemb_profile",
                "specEmb_profile",
                "specembedding_profile",
                "embedding_profile",
                "profile",
            ],
            True,
        ),
        True,
    )

    normalize_embeddings = _as_bool(
        _cfg(
            config,
            [
                "specemb_normalize_embeddings",
                "specEmb_normalize_embeddings",
                "specembedding_normalize_embeddings",
                "normalize_embeddings",
                "l2_normalize_embeddings",
            ],
            True,
        ),
        True,
    )

    if batch_size <= 0:
        raise ValueError(f"specemb_batch_size must be positive, got {batch_size}")

    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model.eval()

    embeddings = None
    n_done = 0

    t_total_start = time.perf_counter()
    t_prepare = 0.0
    t_transfer = 0.0
    t_infer = 0.0
    t_store = 0.0

    if tqdm is not None:
        pbar = tqdm(total=n_spectra, desc="SpecEmbedding", unit="spectra")
    else:
        pbar = None

    with torch.inference_mode():
        for batch_indices in _iter_batch_indices(n_spectra, batch_size):
            t0 = time.perf_counter()

            batch_np = prepare_specEmb_batch_from_msdata(
                msdata,
                batch_indices,
                config,
            )

            t1 = time.perf_counter()

            batch_torch = _numpy_batch_to_torch(
                batch_np,
                device=device,
                pin_memory=device.type == "cuda",
            )

            if do_profile:
                _cuda_synchronize_if_needed(device)

            t2 = time.perf_counter()

            emb_np = _predict_specEmb_batch(
                model=model,
                batch=batch_torch,
                device=device,
                config=config,
                use_amp=use_amp,
            )

            if do_profile:
                _cuda_synchronize_if_needed(device)

            t3 = time.perf_counter()

            if embeddings is None:
                if embedding_dim is None:
                    embedding_dim = int(emb_np.shape[1])

                embeddings = np.zeros(
                    (n_spectra, embedding_dim),
                    dtype=dtype,
                )

            if emb_np.shape[0] != len(batch_indices):
                raise ValueError(
                    f"SpecEmbedding output batch size mismatch: "
                    f"got {emb_np.shape[0]}, expected {len(batch_indices)}"
                )

            if emb_np.shape[1] != embeddings.shape[1]:
                raise ValueError(
                    f"SpecEmbedding embedding dim mismatch: "
                    f"got {emb_np.shape[1]}, expected {embeddings.shape[1]}"
                )

            embeddings[np.asarray(batch_indices, dtype=np.int64)] = emb_np.astype(
                dtype,
                copy=False,
            )

            t4 = time.perf_counter()

            t_prepare += t1 - t0
            t_transfer += t2 - t1
            t_infer += t3 - t2
            t_store += t4 - t3

            n_batch = len(batch_indices)
            n_done += n_batch

            if pbar is not None:
                pbar.update(n_batch)

    if pbar is not None:
        pbar.close()

    if embeddings is None:
        raise RuntimeError("No SpecEmbedding embeddings were generated.")

    if normalize_embeddings:
        embeddings = l2_normalize(embeddings)

    elapsed = time.perf_counter() - t_total_start

    if do_profile:
        spectra_per_sec = n_done / max(elapsed, 1e-12)

        print("")
        print("[SpecEmbedding single-loop profiling]")
        print(f"  n_spectra  : {n_done}")
        print(f"  batch_size : {batch_size}")
        print(f"  use_amp    : {use_amp}")
        print(f"  normalize  : {normalize_embeddings}")
        print(f"  elapsed    : {elapsed:.2f} s")
        print(f"  throughput : {spectra_per_sec:.2f} spectra/s")
        print(f"  prepare    : {t_prepare:.2f} s")
        print(f"  transfer   : {t_transfer:.2f} s")
        print(f"  infer      : {t_infer:.2f} s")
        print(f"  store      : {t_store:.2f} s")
        print("")

    return embeddings


def _build_specEmb_embeddings_multiprocess_loop(
    model,
    n_spectra,
    config,
    device,
    embedding_dim=None,
    dtype=np.float32,
):
    batch_size = int(
        _cfg(
            config,
            [
                "specemb_batch_size",
                "specEmb_batch_size",
                "specembedding_batch_size",
                "embedding_batch_size",
                "batch_size",
            ],
            1024,
        )
    )

    num_workers = int(
        _cfg(
            config,
            [
                "specemb_num_workers",
                "specEmb_num_workers",
                "specembedding_num_workers",
                "num_workers",
            ],
            4,
        )
    )

    max_pending_batches = int(
        _cfg(
            config,
            [
                "specemb_max_pending_batches",
                "specEmb_max_pending_batches",
                "specembedding_max_pending_batches",
                "specemb_pending_batches",
            ],
            max(2, num_workers * 2),
        )
    )

    use_amp = _as_bool(
        _cfg(
            config,
            [
                "specemb_use_amp",
                "specEmb_use_amp",
                "specembedding_use_amp",
                "use_amp",
            ],
            False,
        ),
        False,
    )

    do_profile = _as_bool(
        _cfg(
            config,
            [
                "specemb_profile",
                "specEmb_profile",
                "specembedding_profile",
                "embedding_profile",
                "profile",
            ],
            True,
        ),
        True,
    )

    normalize_embeddings = _as_bool(
        _cfg(
            config,
            [
                "specemb_normalize_embeddings",
                "specEmb_normalize_embeddings",
                "specembedding_normalize_embeddings",
                "normalize_embeddings",
                "l2_normalize_embeddings",
            ],
            True,
        ),
        True,
    )

    if batch_size <= 0:
        raise ValueError(f"specemb_batch_size must be positive, got {batch_size}")

    if num_workers <= 0:
        raise ValueError(
            "_build_specEmb_embeddings_multiprocess_loop requires num_workers > 0."
        )

    if max_pending_batches <= 0:
        max_pending_batches = max(2, num_workers * 2)

    device = torch.device(device)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    model.eval()

    config_dict = _config_to_plain_dict(config)

    embeddings = None
    n_done = 0

    t_total_start = time.perf_counter()
    t_wait_cpu = 0.0
    t_transfer = 0.0
    t_infer = 0.0
    t_store = 0.0

    if tqdm is not None:
        pbar = tqdm(total=n_spectra, desc="SpecEmbedding", unit="spectra")
    else:
        pbar = None

    batch_iter = iter(_iter_batch_indices(n_spectra, batch_size))
    pending = set()

    def submit_next(executor):
        try:
            batch_indices = next(batch_iter)
        except StopIteration:
            return False

        fut = executor.submit(_specEmb_prepare_batch_worker, batch_indices)
        pending.add(fut)

        return True

    with ProcessPoolExecutor(
        max_workers=num_workers,
        initializer=_specEmb_cpu_worker_init,
        initargs=(config_dict,),
    ) as executor:

        for _ in range(max_pending_batches):
            ok = submit_next(executor)

            if not ok:
                break

        with torch.inference_mode():
            while pending:
                t0 = time.perf_counter()

                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                t1 = time.perf_counter()
                t_wait_cpu += t1 - t0

                for fut in done:
                    try:
                        item = fut.result()
                    except Exception:
                        traceback.print_exc()
                        raise

                    while len(pending) < max_pending_batches:
                        ok = submit_next(executor)

                        if not ok:
                            break

                    batch_indices = item["indices"]
                    batch_np = item["batch"]

                    t2 = time.perf_counter()

                    batch_torch = _numpy_batch_to_torch(
                        batch_np,
                        device=device,
                        pin_memory=device.type == "cuda",
                    )

                    if do_profile:
                        _cuda_synchronize_if_needed(device)

                    t3 = time.perf_counter()

                    emb_np = _predict_specEmb_batch(
                        model=model,
                        batch=batch_torch,
                        device=device,
                        config=config,
                        use_amp=use_amp,
                    )

                    if do_profile:
                        _cuda_synchronize_if_needed(device)

                    t4 = time.perf_counter()

                    if embeddings is None:
                        if embedding_dim is None:
                            embedding_dim = int(emb_np.shape[1])

                        embeddings = np.zeros(
                            (n_spectra, embedding_dim),
                            dtype=dtype,
                        )

                    if emb_np.shape[0] != len(batch_indices):
                        raise ValueError(
                            f"SpecEmbedding output batch size mismatch: "
                            f"got {emb_np.shape[0]}, expected {len(batch_indices)}"
                        )

                    if emb_np.shape[1] != embeddings.shape[1]:
                        raise ValueError(
                            f"SpecEmbedding embedding dim mismatch: "
                            f"got {emb_np.shape[1]}, expected {embeddings.shape[1]}"
                        )

                    embeddings[batch_indices] = emb_np.astype(dtype, copy=False)

                    t5 = time.perf_counter()

                    t_transfer += t3 - t2
                    t_infer += t4 - t3
                    t_store += t5 - t4

                    n_batch = len(batch_indices)
                    n_done += n_batch

                    if pbar is not None:
                        pbar.update(n_batch)

    if pbar is not None:
        pbar.close()

    if embeddings is None:
        raise RuntimeError("No SpecEmbedding embeddings were generated.")

    if normalize_embeddings:
        embeddings = l2_normalize(embeddings)

    elapsed = time.perf_counter() - t_total_start

    if do_profile:
        spectra_per_sec = n_done / max(elapsed, 1e-12)

        print("")
        print("[SpecEmbedding multiprocessing profiling]")
        print(f"  n_spectra          : {n_done}")
        print(f"  batch_size         : {batch_size}")
        print(f"  num_workers        : {num_workers}")
        print(f"  max_pending_batches: {max_pending_batches}")
        print(f"  use_amp            : {use_amp}")
        print(f"  normalize          : {normalize_embeddings}")
        print(f"  elapsed            : {elapsed:.2f} s")
        print(f"  throughput         : {spectra_per_sec:.2f} spectra/s")
        print(f"  wait_cpu_batch     : {t_wait_cpu:.2f} s")
        print(f"  transfer           : {t_transfer:.2f} s")
        print(f"  infer              : {t_infer:.2f} s")
        print(f"  store              : {t_store:.2f} s")
        print("")

    return embeddings


# ---------------------------------------------------------------------
# Public build function
# ---------------------------------------------------------------------
def _is_config_candidate(obj):
    if obj is None:
        return False

    if isinstance(obj, (str, Path)):
        return False

    if isinstance(obj, dict):
        return True

    for key in [
        "mona_hdf5_path",
        "library_hdf5_path",
        "specemb_model_file",
        "specEmb_model_file",
        "specembedding_model_file",
        "specemb_checkpoint_path",
        "specEmb_checkpoint_path",
        "specemb_batch_size",
        "specEmb_batch_size",
    ]:
        if hasattr(obj, key):
            return True

    if PipelineConfig is not None and isinstance(obj, PipelineConfig):
        return True

    return False


def _parse_build_args(*args, **kwargs):
    """
    Compatible call styles:
        build_specEmb_embeddings(msdata, config)
        build_specEmb_embeddings(msdata, split_name, config)
        build_specEmb_embeddings(config, msdata)
        build_specEmb_embeddings(config=config, msdata=msdata)
        build_specEmb_embeddings(config=config)
    """
    msdata = kwargs.pop("msdata", None)
    config = kwargs.pop("config", None)

    if msdata is None:
        msdata = kwargs.pop("msdata_lib", None)

    if config is None:
        config = kwargs.pop("cfg", None)

    if config is None:
        for obj in reversed(args):
            if _is_config_candidate(obj):
                config = obj
                break

    if msdata is None:
        for obj in args:
            if obj is config:
                continue

            if isinstance(obj, str):
                continue

            if isinstance(obj, Path):
                continue

            msdata = obj
            break

    if config is None:
        if PipelineConfig is not None:
            config = PipelineConfig()
        else:
            raise ValueError(
                "config is required for build_specEmb_embeddings, "
                "but no config was provided and PipelineConfig is unavailable."
            )

    return msdata, config


def _open_msdata_from_config_if_needed(msdata, config):
    if msdata is not None:
        return msdata

    if load_mona_library is None:
        raise ValueError(
            "msdata was not provided and load_mona_library is unavailable."
        )

    mona_path = _cfg(
        config,
        [
            "mona_hdf5_path",
            "library_hdf5_path",
            "mona_path",
            "msdata_path",
        ],
        None,
    )

    if mona_path is None:
        raise ValueError(
            "msdata was not provided. Please provide msdata or set config.mona_hdf5_path."
        )

    mona_path = Path(mona_path)

    if not mona_path.exists():
        raise FileNotFoundError(f"MoNA HDF5 path not found: {mona_path}")

    print(f"[SpecEmbedding] Loading MoNA library: {mona_path}")

    return load_mona_library(mona_path)


def build_specEmb_embeddings(*args, **kwargs):
    msdata, config = _parse_build_args(*args, **kwargs)

    msdata = _open_msdata_from_config_if_needed(msdata, config)

    n_spectra = int(get_msdata_length(msdata))

    limit = _maybe_configured_limit(config)

    if limit is not None:
        n_spectra = min(n_spectra, limit)

    if n_spectra <= 0:
        raise ValueError("msdata contains no spectra.")

    device = _resolve_device(config)
    output_dtype = _resolve_output_dtype(config)

    embedding_dim = _cfg(
        config,
        [
            "specemb_dim_target",
            "specEmb_dim_target",
            "specembedding_dim_target",
            "specemb_embedding_dim",
            "specEmb_embedding_dim",
            "specembedding_embedding_dim",
            "embedding_dim",
        ],
        None,
    )

    if embedding_dim is not None:
        try:
            embedding_dim = int(embedding_dim)
        except Exception:
            embedding_dim = None

    batch_size = int(
        _cfg(
            config,
            [
                "specemb_batch_size",
                "specEmb_batch_size",
                "specembedding_batch_size",
                "embedding_batch_size",
                "batch_size",
            ],
            1024,
        )
    )

    num_workers = int(
        _cfg(
            config,
            [
                "specemb_num_workers",
                "specEmb_num_workers",
                "specembedding_num_workers",
                "num_workers",
            ],
            0,
        )
    )

    input_mode = _cfg(
        config,
        [
            "specemb_input_mode",
            "specEmb_input_mode",
            "specembedding_input_mode",
        ],
        None,
    )

    if input_mode is not None and str(input_mode).lower() not in {
        "siamese",
        "separate",
        "mz_intensity",
        "mz_intensity_mask",
    }:
        print(
            f"[SpecEmbedding] NOTE: config specemb_input_mode={input_mode!r}, "
            f"but provided SiameseModel requires mz/intensity/mask. "
            f"Using Siamese input automatically."
        )

    print("[SpecEmbedding] Starting embedding generation")
    print(f"[SpecEmbedding] n_spectra   : {n_spectra}")
    print(f"[SpecEmbedding] device      : {device}")
    print(f"[SpecEmbedding] input_mode  : siamese_mz_intensity_mask")
    print(f"[SpecEmbedding] batch_size  : {batch_size}")
    print(f"[SpecEmbedding] num_workers : {num_workers}")
    print(f"[SpecEmbedding] output_dtype: {output_dtype}")

    t_model0 = time.perf_counter()
    model = load_specEmb_model(config, device=device)
    t_model1 = time.perf_counter()

    print(f"[SpecEmbedding] Model loaded in {t_model1 - t_model0:.2f} s")

    t0 = time.perf_counter()

    if num_workers > 0:
        print(
            f"[SpecEmbedding] Using multiprocessing CPU preprocessing: "
            f"{num_workers} workers"
        )

        embeddings = _build_specEmb_embeddings_multiprocess_loop(
            model=model,
            n_spectra=n_spectra,
            config=config,
            device=device,
            embedding_dim=embedding_dim,
            dtype=output_dtype,
        )

    else:
        print("[SpecEmbedding] Using single-process preprocessing")

        embeddings = _build_specEmb_embeddings_single_loop(
            model=model,
            msdata=msdata,
            n_spectra=n_spectra,
            config=config,
            device=device,
            embedding_dim=embedding_dim,
            dtype=output_dtype,
        )

    elapsed = time.perf_counter() - t0

    print(
        f"[SpecEmbedding] Finished embedding {n_spectra} spectra "
        f"in {elapsed:.2f} s, throughput={n_spectra / max(elapsed, 1e-12):.2f} spectra/s"
    )

    try:
        embeddings = cast_embedding_dtype(embeddings, config)
    except Exception:
        embeddings = embeddings.astype(output_dtype, copy=False)

    return embeddings


# Aliases for routing compatibility.
build_specemb_embeddings = build_specEmb_embeddings
build_spec_embedding_embeddings = build_specEmb_embeddings
def build_specEmb_embeddings_from_mzml(
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
    Direct SpecEmbedding mzML inference.

    This function does NOT call:
        - build_specEmb_embeddings
        - build_specemb_embeddings
        - build_spec_embedding_embeddings
    """

    try:
        from mzml_input import load_mzml_spectra, save_mzml_embeddings_npz
    except ImportError:
        from .mzml_input import load_mzml_spectra, save_mzml_embeddings_npz

    device = _resolve_device(config)

    loaded = load_specEmb_model(config)

    if isinstance(loaded, tuple):
        model = loaded[0]
    else:
        model = loaded

    if not isinstance(model, torch.nn.Module):
        raise TypeError(
            "load_specEmb_model(config) must return a torch.nn.Module "
            f"or tuple whose first item is a torch.nn.Module, got {type(model)}"
        )

    model = model.to(device)
    model.eval()

    spectra = load_mzml_spectra(
        mzml_path,
        ms_level=ms_level,
        max_spectra=max_spectra,
        dtype=np.float32,
    )

    if limit is not None:
        spectra = spectra[: int(limit)]

    if batch_size is None:
        batch_size = int(
            _cfg(
                config,
                [
                    "specemb_batch_size",
                    "specEmb_batch_size",
                    "specembedding_batch_size",
                    "embedding_batch_size",
                    "batch_size",
                ],
                64,
            )
        )

    max_len = int(
        _cfg(
            config,
            [
                "specemb_max_len",
                "specEmb_max_len",
                "specembedding_max_len",
                "embedding_max_peaks",
                "max_peaks",
                "embedding_top_n_peaks",
            ],
            200,
        )
    )

    all_embeddings = []

    def make_batch(batch_spectra):
        bsz = len(batch_spectra)

        mz_batch = np.zeros((bsz, max_len), dtype=np.float32)
        intensity_batch = np.zeros((bsz, max_len), dtype=np.float32)
        mask_batch = np.ones((bsz, max_len), dtype=bool)

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

            max_intensity = np.max(intensities)

            if max_intensity > 0:
                intensities = intensities / max_intensity

            n = min(mzs.size, max_len)

            mz_batch[row, :n] = mzs[:n]
            intensity_batch[row, :n] = intensities[:n]
            mask_batch[row, :n] = False

        mz_tensor = torch.from_numpy(mz_batch).to(device=device)
        intensity_tensor = torch.from_numpy(intensity_batch).to(device=device)
        mask_tensor = torch.from_numpy(mask_batch).to(device=device)

        return mz_tensor, intensity_tensor, mask_tensor

    with torch.inference_mode():
        for start in tqdm(
            range(0, len(spectra), batch_size),
            desc="[SpecEmb-mzML] Encoding",
            unit="batch",
            dynamic_ncols=True,
        ):
            end = min(start + batch_size, len(spectra))
            batch_spectra = spectra[start:end]

            mz_tensor, intensity_tensor, mask_tensor = make_batch(batch_spectra)

            output = model(mz_tensor, intensity_tensor, mask_tensor)

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

    normalize = bool(
        _cfg(
            config,
            [
                "specemb_normalize_embeddings",
                "specEmb_normalize_embeddings",
                "normalize_embeddings",
            ],
            True,
        )
    )

    if normalize:
        embeddings = l2_normalize(embeddings)

    embeddings = embeddings.astype(_resolve_output_dtype(config), copy=False)

    saved_path = save_mzml_embeddings_npz(
        output_path,
        embeddings,
        spectra,
        mzml_path=mzml_path,
        method="specEmb",
    )

    print(f"[SpecEmb-mzML] Saved embeddings to: {saved_path}")

    return embeddings

