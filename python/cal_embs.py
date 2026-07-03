import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict

import numpy as np

from io_utils import load_mona_library
from dreams_emb import build_dreams_embeddings
from spec2vec_emb import (
    get_config_value,
    get_msdata_length,
    build_spec2vec_embeddings,
)
from ms2deepscore_emb import build_ms2deepscore_embeddings
from binned_emb import (
    build_binned_embeddings,
    build_neutral_loss_binned_embeddings,
)
from msbert_emb import build_msbert_embeddings
from specEmb_emb import build_specEmb_embeddings



SUPPORTED_EMBEDDING_MODELS = (
    "dreams",
    "spec2vec",
    "ms2deepscore",
    "binned",
    "neutral_loss_binned",
    "msbert",
    'specemb'
)


@dataclass
class EmbeddingCacheInfo:
    model_name: str
    split_name: str
    cache_path: str
    meta_path: str
    n_spectra: int
    embedding_dim: int
    dtype: str
    created_at: str
    elapsed_seconds: float
    source_path: str
    limit: int | None
    config_hash: str


@dataclass
class EmbeddingResult:
    model_name: str
    embeddings: np.ndarray
    cache_info: EmbeddingCacheInfo | None = None


def now_string():
    return time.strftime("%Y-%m-%d %H:%M:%S")


def make_embedding_config_fingerprint(config, split_name="library", limit=None):
    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    parts = {
        "split_name": split_name,
        "embedding_model": model,
        "limit": limit,

        # dtype / memory-relevant settings
        "use_float32": get_config_value(config, "use_float32", False),
        "embedding_dtype": str(get_config_value(config, "embedding_dtype", "")),
        "binned_dtype": str(get_config_value(config, "binned_dtype", "")),

        # source paths
        "mona_hdf5_path": str(get_config_value(config, "mona_hdf5_path", "")),
        "query_mzml_path": str(get_config_value(config, "query_mzml_path", "")),

        # binned parameters
        "embedding_mz_min": get_config_value(config, "embedding_mz_min", None),
        "embedding_mz_max": get_config_value(config, "embedding_mz_max", None),
        "embedding_bin_size": get_config_value(config, "embedding_bin_size", None),
        "embedding_intensity_power": get_config_value(config, "embedding_intensity_power", None),
        "embedding_top_n_peaks": get_config_value(config, "embedding_top_n_peaks", None),

        # neutral loss parameters
        "neutral_loss_mz_min": get_config_value(config, "neutral_loss_mz_min", None),
        "neutral_loss_mz_max": get_config_value(config, "neutral_loss_mz_max", None),
        "neutral_loss_weight": get_config_value(config, "neutral_loss_weight", None),

        # spec2vec parameters
        "spec2vec_model_path": str(get_config_value(config, "spec2vec_model_path", "")),
        "spec2vec_positive_model_path": str(get_config_value(config, "spec2vec_positive_model_path", "")),
        "spec2vec_negative_model_path": str(get_config_value(config, "spec2vec_negative_model_path", "")),
        "spec2vec_intensity_weighting_power": get_config_value(
            config,
            "spec2vec_intensity_weighting_power",
            None,
        ),

        # ms2deepscore parameters
        "ms2deepscore_model_path": str(get_config_value(config, "ms2deepscore_model_path", "")),
        # msbert parameters
        "msbert_repo_dir": str(get_config_value(config, "msbert_repo_dir", "")),
        "msbert_checkpoint_path": str(get_config_value(config, "msbert_checkpoint_path", "")),
        "msbert_device": str(get_config_value(config, "msbert_device", "")),
        "msbert_batch_size": get_config_value(config, "msbert_batch_size", None),

        "msbert_vocab_size": get_config_value(config, "msbert_vocab_size", None),
        "msbert_hidden": get_config_value(config, "msbert_hidden", None),
        "msbert_n_layers": get_config_value(config, "msbert_n_layers", None),
        "msbert_attn_heads": get_config_value(config, "msbert_attn_heads", None),
        "msbert_dropout": get_config_value(config, "msbert_dropout", None),
        "msbert_max_len": get_config_value(config, "msbert_max_len", None),
        "msbert_max_pred": get_config_value(config, "msbert_max_pred", None),

        "msbert_mz_min": get_config_value(config, "msbert_mz_min", None),
        "msbert_mz_max": get_config_value(config, "msbert_mz_max", None),
        "msbert_mz_bin_size": get_config_value(config, "msbert_mz_bin_size", None),
        "msbert_token_offset": get_config_value(config, "msbert_token_offset", None),
        "msbert_min_token_id": get_config_value(config, "msbert_min_token_id", None),

        "msbert_min_peaks": get_config_value(config, "msbert_min_peaks", None),
        "msbert_max_peaks": get_config_value(config, "msbert_max_peaks", None),
        "msbert_seq_len": get_config_value(config, "msbert_seq_len", None),

        "msbert_normalize_intensity": get_config_value(config, "msbert_normalize_intensity", None),
        "msbert_intensity_power": get_config_value(config, "msbert_intensity_power", None),
        "msbert_select_top_peaks_by": get_config_value(config, "msbert_select_top_peaks_by", None),
        "msbert_final_sort_by": get_config_value(config, "msbert_final_sort_by", None),
        "msbert_duplicate_token_aggregate": get_config_value(config, "msbert_duplicate_token_aggregate", None),

        "msbert_skip_invalid_spectra": get_config_value(config, "msbert_skip_invalid_spectra", None),
        "msbert_use_amp": get_config_value(config, "msbert_use_amp", None),
        "msbert_amp_dtype": get_config_value(config, "msbert_amp_dtype", None),
        "msbert_strict_load": get_config_value(config, "msbert_strict_load", None),
    }

    text = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def get_embedding_cache_paths(config, split_name="library", limit=None):
    from pathlib import Path

    model = str(get_config_value(config, "embedding_model", "dreams")).lower()
    cache_dir = Path(get_config_value(config, "embedding_cache_dir", "outputs/embedding_cache"))
    cache_dir.mkdir(parents=True, exist_ok=True)

    config_hash = make_embedding_config_fingerprint(
        config=config,
        split_name=split_name,
        limit=limit,
    )

    if limit is None:
        prefix = split_name
    else:
        prefix = f"{split_name}_limit{int(limit)}"

    cache_path = cache_dir / f"{prefix}_{model}_{config_hash}.npy"
    meta_path = cache_dir / f"{prefix}_{model}_{config_hash}.json"

    return cache_path, meta_path, config_hash


def print_embedding_summary(embeddings, name="embeddings"):
    """
    Print summary safely.

    For huge dense matrices, never run np.isfinite() on the whole matrix,
    because that may allocate a massive temporary bool array.
    """

    print(f"[{name}] shape = {embeddings.shape}")
    print(f"[{name}] dtype = {embeddings.dtype}")

    if embeddings.size > 0 and len(embeddings.shape) == 2:
        sample_n = min(1000, embeddings.shape[0])
        sample = np.asarray(embeddings[:sample_n])

        finite_ratio = float(np.isfinite(sample).mean())
        row_norms = np.linalg.norm(sample, axis=1)

        print(f"[{name}] sample_rows = {sample_n}")
        print(f"[{name}] sample_finite_ratio = {finite_ratio:.6f}")
        print(
            f"[{name}] sample_rows_norm: "
            f"min={np.nanmin(row_norms):.6f}, "
            f"mean={np.nanmean(row_norms):.6f}, "
            f"max={np.nanmax(row_norms):.6f}"
        )


def build_embeddings_from_msdata(msdata, config, limit=None, existing_dreams_embeddings=None):
    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    if model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding_model: {model}. "
            f"Supported models: {SUPPORTED_EMBEDDING_MODELS}"
        )

    print(f"[embedding] model = {model}")

    if model == "dreams":
        return build_dreams_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
            existing_dreams_embeddings=existing_dreams_embeddings,
        )

    if model == "binned":
        return build_binned_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
        )

    if model == "neutral_loss_binned":
        return build_neutral_loss_binned_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
        )

    if model == "spec2vec":
        return build_spec2vec_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
        )

    if model == "ms2deepscore":
        return build_ms2deepscore_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
        )
    if model == "msbert":
        return build_msbert_embeddings(
            msdata=msdata,
            config=config,
            limit=limit,
        )

    if model == "specemb":
        return build_specEmb_embeddings(
            msdata,
            config,
        )

    raise ValueError(f"Unsupported embedding_model: {model}")


def preencode_mona_library(config, limit=None, force=False):
    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    cache_path, meta_path, config_hash = get_embedding_cache_paths(
        config=config,
        split_name="library",
        limit=limit,
    )

    if cache_path.exists() and meta_path.exists() and not force:
        print(f"[cache] Found existing embedding cache: {cache_path}")

        embeddings = np.load(cache_path, mmap_mode="r")

        print_embedding_summary(embeddings, name=f"library_{model}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cache_info = EmbeddingCacheInfo(**meta)

        return EmbeddingResult(
            model_name=model,
            embeddings=embeddings,
            cache_info=cache_info,
        )

    print("[mona] Loading MoNA library...")
    msdata_lib, existing_embs_lib, lib_cols = load_mona_library(config)

    print(f"[mona] Available columns: {lib_cols}")
    print(f"[mona] Library spectra: {get_msdata_length(msdata_lib)}")

    start = time.time()

    try:
        embeddings = build_embeddings_from_msdata(
            msdata=msdata_lib,
            config=config,
            limit=limit,
            existing_dreams_embeddings=existing_embs_lib if model == "dreams" else None,
        )

    finally:
        try:
            if hasattr(msdata_lib, "close"):
                msdata_lib.close()
        except Exception:
            pass

    elapsed = time.time() - start

    print_embedding_summary(embeddings, name=f"library_{model}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache] Saving embeddings to: {cache_path}")
    np.save(cache_path, embeddings)

    cache_info = EmbeddingCacheInfo(
        model_name=model,
        split_name="library",
        cache_path=str(cache_path),
        meta_path=str(meta_path),
        n_spectra=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        dtype=str(embeddings.dtype),
        created_at=now_string(),
        elapsed_seconds=float(elapsed),
        source_path=str(get_config_value(config, "mona_hdf5_path", "")),
        limit=None if limit is None else int(limit),
        config_hash=config_hash,
    )

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(asdict(cache_info), f, ensure_ascii=False, indent=2)

    print(f"[cache] Saved embeddings to: {cache_path}")
    print(f"[cache] Saved metadata to: {meta_path}")
    print(f"[done] elapsed_seconds = {elapsed:.2f}")

    return EmbeddingResult(
        model_name=model,
        embeddings=embeddings,
        cache_info=cache_info,
    )


def main():
    import multiprocessing

    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Pre-encode MoNA library embeddings for chimera pipeline."
    )

    parser.add_argument(
        "--model",
        type=str,
        default="specemb",
        choices=SUPPORTED_EMBEDDING_MODELS,
        help="Embedding model to use.",
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only encode first N spectra for testing. Default: encode full library.",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force recompute even if cache exists.",
    )

    parser.add_argument(
        "--float32",
        action="store_true",
        help="Force output embeddings to float32.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of workers for binned / neutral_loss_binned static multiprocessing.",
    )

    args = parser.parse_args()

    try:
        from config import PipelineConfig
    except ImportError:
        from .config import PipelineConfig

    config = PipelineConfig()

    if args.model is not None:
        config.embedding_model = args.model

    if args.float32:
        config.use_float32 = True
        config.embedding_dtype = "float32"
        config.binned_dtype = "float32"

    if args.workers is not None:
        config.binned_num_workers = int(args.workers)

    if args.force:
        config.force_recompute_embeddings = True

    preencode_mona_library(
        config=config,
        limit=args.limit,
        force=args.force or get_config_value(config, "force_recompute_embeddings", False),
    )


if __name__ == "__main__":
    main()
