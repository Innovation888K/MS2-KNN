# -*- coding: utf-8 -*-

"""
ms2_knn.py

MS2-KNN workflow for resolving chimeric small-molecule MS/MS spectra.

This script reproduces the Chimera-KNN / MS2-KNN workflow without modifying
chimera_knn_batch.py.

Workflow
--------
1. Load chimera query embeddings and reference-library embeddings.
2. Compute or load cached Top-K cosine nearest neighbors.
3. For each query spectrum:
   - Apply adaptive-K truncation to remove low-similarity background candidates.
   - Estimate the number of precursor/component groups using GMM on candidate precursor m/z.
   - Cluster retained Top-K candidates by embedding vectors into the estimated number of groups.
   - Optionally compute cluster-level MCS/scaffold evidence.
   - Compute candidate-level backbone-confidence scores from MCS/scaffold support.
   - Apply two-level pruning:
       a. cluster-level pruning: remove unsupported groups;
       b. compound-level pruning: skip structurally unsupported candidates within retained groups.
   - Select Top-N candidates per retained cluster.
4. Save neighbor, recommendation, scaffold, confidence, manifest, and evaluation files.
5. Optionally evaluate predicted component count against chimera HDF5 ground truth.

Notes
-----
- This script does not modify chimera_knn_batch.py.
- For manuscript-style backbone-confidence scoring and two-level pruning,
  both COMPUTE_SCAFFOLD and USE_STRUCTURE_CONFIDENCE should be True.
"""

from __future__ import annotations

import os
import json
import warnings
import multiprocessing as mp
from pathlib import Path
from typing import Any
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

import matplotlib.pyplot as plt
import seaborn as sns

from adaptive_k_utils import (
    estimate_adaptive_k_piecewise_bic_general,
    compact_adaptive_info_for_row,
    summarize_adaptive_k,
)

# =============================================================================
# 0. Import original workflow utilities without modifying them
# =============================================================================

try:
    import chimera_knn_batch as ck
except ImportError:
    from . import chimera_knn_batch as ck


# =============================================================================
# 1. User configuration
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent

QUERY_DIR = Path(
    r"D:\亚结构注释\for_git\chimera_pipeline\outputs\mona_chimera_dataset_equal_200k_random"
)

LIBRARY_DIR = Path(
    r"D:\亚结构注释\for_git\chimera_pipeline\outputs\embedding_cache"
)

METADATA_CSV = LIBRARY_DIR / "library_metadata.csv"

CHIMERA_HDF5_PATH = Path(
    r"D:\亚结构注释\mona_processed\mona_chimera_dataset.hdf5"
)

OUT_DIR = Path(
    r"D:\亚结构注释\for_git\chimera_pipeline\outputs\mona_chimera_dataset_equal_200k_random_eval"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)


# -----------------------------------------------------------------------------
# Adaptive-K parameters
# -----------------------------------------------------------------------------

# Whether to enable adaptive candidate truncation.
USE_ADAPTIVE_K = True

# Initial Top-K for cosine retrieval.
# Adaptive-K truncates candidates from this initial Top-K list.
KNN_TOPK = 100

# Final retained candidate number bounds after adaptive-K truncation.
ADAPTIVE_K_MIN = 12
ADAPTIVE_K_MAX = 100

# Maximum number of piecewise-linear segments in BIC-based score-rank fitting.
# For example, 4 segments means at most 3 breakpoints.
ADAPTIVE_MAX_SEGMENTS = 4

# Minimum number of ranked candidates in each piecewise-linear segment.
ADAPTIVE_MIN_SEGMENT_SIZE = 8

# Fallback K when no reliable breakpoint is detected.
ADAPTIVE_FALLBACK_K = 100

# If the score range is too small, no clear signal/background boundary is assumed.
ADAPTIVE_MIN_SCORE_RANGE = 0.03

# If a multi-segment model is selected, treat the last segment as background
# and retain candidates before the last breakpoint.
ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT = True

# Whether to save adaptive-K diagnostic fields into neighbor CSV files.
SAVE_ADAPTIVE_K_DIAGNOSTICS = True


# -----------------------------------------------------------------------------
# Main workflow parameters
# -----------------------------------------------------------------------------

# Number of post-KNN worker processes.
# None means cpu_count - 1.
N_POSTPROCESS_WORKERS = 10

# Whether to merge chunk-level CSV files into method-level CSV files.
MERGE_CHUNK_CSV = True

# Maximum number of precursor m/z groups allowed in GMM.
MAX_PRECURSOR_GROUPS = 20

# Number of final recommended molecules selected from each retained cluster.
TOP_N_PER_GROUP = 3

# Top-K retrieval parameters.
USE_CUDA_KNN = True
CUDA_QUERY_BATCH_SIZE = 128
CUDA_USE_FLOAT16 = True
CPU_BATCH_SIZE = 128

# Progress report interval for post-processing workers.
POSTPROCESS_PROGRESS_EVERY = 500

# Whether to compute cluster-level MCS/scaffold evidence.
#
# IMPORTANT:
# To reproduce manuscript-style backbone-confidence scoring and two-level pruning,
# this must be True together with USE_STRUCTURE_CONFIDENCE=True.
COMPUTE_SCAFFOLD = True

# MCS/scaffold computation parameters.
MAX_MOLS_PER_CLUSTER = 100
MCS_TIMEOUT = 20
MIN_NUM_ATOMS = 5
MIN_CLUSTER_SIZE = 1

# Limit the number of queries for debugging. Use None for full run.
LIMIT_QUERIES = None

# Run only selected methods. None means all matched query-library pairs.
METHOD_WHITELIST = None
# METHOD_WHITELIST = ["dreams"]


# -----------------------------------------------------------------------------
# Structure confidence parameters
# -----------------------------------------------------------------------------

# Whether to compute MCS/scaffold-based backbone confidence.
USE_STRUCTURE_CONFIDENCE = True

# Confidence level thresholds.
STRUCT_CONF_HIGH = 0.70
STRUCT_CONF_MEDIUM = 0.40

# Minimum MCS atom count required for reliable structural support.
STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF = 5

# Backbone-confidence weights.
STRUCT_WEIGHT_CLUSTER_SUPPORT = 0.30
STRUCT_WEIGHT_OVERLAP = 0.35
STRUCT_WEIGHT_CANDIDATE_COVERAGE = 0.25
STRUCT_WEIGHT_EXACT_MATCH = 0.10


# -----------------------------------------------------------------------------
# Worker globals
# -----------------------------------------------------------------------------

_WORKER_LIB = None
_WORKER_LIB_EMBS = None
_WORKER_TOP_INDICES = None
_WORKER_TOP_SCORES = None
_WORKER_QUERY_FILE = None
_WORKER_LIB_FILE = None
_WORKER_METHOD = None
_WORKER_PROGRESS_QUEUE = None


# =============================================================================
# 2. Small helper functions
# =============================================================================

def _safe_float(x, default=np.nan):
    """
    Safely convert a value to float.

    Parameters
    ----------
    x:
        Input value.
    default:
        Returned value when conversion fails or x is missing.

    Returns
    -------
    float
        Converted float value or default.
    """
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    """
    Safely convert a value to int.

    Parameters
    ----------
    x:
        Input value.
    default:
        Returned value when conversion fails or x is missing.

    Returns
    -------
    int
        Converted integer value or default.
    """
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _normalize_smiles_like(x: Any) -> str:
    """
    Normalize a SMILES-like value into a clean string.

    None, NaN-like strings, and null-like values are converted to empty strings.

    Parameters
    ----------
    x:
        Input SMILES-like value.

    Returns
    -------
    str
        Cleaned SMILES string.
    """
    if x is None:
        return ""

    s = str(x)

    if s.lower() in {"nan", "none", "null"}:
        return ""

    return s.strip()


def _structure_confidence_level(conf: float) -> str:
    """
    Convert a numeric backbone-confidence score into a categorical level.

    Parameters
    ----------
    conf:
        Backbone-confidence score.

    Returns
    -------
    str
        One of {"High", "Medium", "Low"}.
    """
    if not np.isfinite(conf):
        return "Low"

    if conf >= STRUCT_CONF_HIGH:
        return "High"

    if conf >= STRUCT_CONF_MEDIUM:
        return "Medium"

    return "Low"


def save_json(obj, path: str | Path):
    """
    Save a Python object as a UTF-8 JSON file.

    Parameters
    ----------
    obj:
        JSON-serializable object.
    path:
        Output JSON path.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def split_query_ranges(n_query: int, n_workers: int):
    """
    Split query indices into continuous chunks for multiprocessing.

    Parameters
    ----------
    n_query:
        Total number of query spectra.
    n_workers:
        Number of worker processes.

    Returns
    -------
    list[tuple[int, int]]
        Half-open query index ranges: [(start, end), ...].
    """
    n_query = int(n_query)
    n_workers = int(max(1, n_workers))
    n_workers = min(n_workers, n_query)

    base = n_query // n_workers
    rem = n_query % n_workers

    ranges = []
    start = 0

    for i in range(n_workers):
        size = base + (1 if i < rem else 0)
        end = start + size
        ranges.append((start, end))
        start = end

    return ranges


def merge_csv_files(csv_paths, out_csv):
    """
    Merge multiple chunk-level CSV files into one CSV without loading all data.

    Parameters
    ----------
    csv_paths:
        Input CSV file paths.
    out_csv:
        Output merged CSV file path.

    Returns
    -------
    pathlib.Path
        Path to the merged CSV file.
    """
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    csv_paths = [Path(p) for p in csv_paths if Path(p).exists()]

    if len(csv_paths) == 0:
        pd.DataFrame().to_csv(out_csv, index=False, encoding="utf-8-sig")
        return out_csv

    first_file = True

    with open(out_csv, "w", encoding="utf-8-sig", newline="") as fout:
        for p in tqdm(csv_paths, desc=f"[merge csv] {out_csv.name}", unit="file"):
            with open(p, "r", encoding="utf-8-sig", newline="") as fin:
                for line_i, line in enumerate(fin):
                    if first_file:
                        fout.write(line)
                    else:
                        if line_i == 0:
                            continue
                        fout.write(line)

            first_file = False

    print(f"[saved] {out_csv}", flush=True)
    return out_csv


# =============================================================================
# 3. Embedding loading and validation
# =============================================================================

def load_query_embedding_file(path):
    """
    Load query embedding file.

    Query embeddings can be:
        - .npy: raw embedding matrix with shape (N, D)
        - .npz: contains "embeddings" or "embedding"

    This loader is intentionally separated from ck.load_embedding_file(), because
    the original loader treats .npy files as library embeddings and requires
    metadata_csv.

    Parameters
    ----------
    path:
        Path to query embedding file.

    Returns
    -------
    dict
        Query embedding dictionary with optional query metadata.
    """
    path = Path(path)

    method = ck.normalize_method_name(
        ck.infer_method_from_filename(path)
    )

    if path.suffix.lower() == ".npy":
        embeddings = np.load(path, mmap_mode="r")

        return {
            "path": path,
            "embeddings": embeddings,
            "method": method,
            "precursor_mz": None,
            "smiles": None,
            "inchikey": None,
            "name": None,
            "formula": None,
        }

    if path.suffix.lower() == ".npz":
        data = np.load(path, allow_pickle=True)

        if "embeddings" in data:
            embeddings = data["embeddings"]
        elif "embedding" in data:
            embeddings = data["embedding"]
        else:
            raise KeyError(
                f"No 'embeddings' or 'embedding' key found in query npz: {path}"
            )

        if "method" in data:
            try:
                method = ck.normalize_method_name(str(np.asarray(data["method"]).item()))
            except Exception:
                method = ck.normalize_method_name(str(data["method"]))

        return {
            "path": path,
            "embeddings": embeddings,
            "method": method,
            "precursor_mz": np.asarray(data["precursor_mz"]) if "precursor_mz" in data else None,
            "smiles": np.asarray(data["smiles"]).astype(str) if "smiles" in data else None,
            "inchikey": np.asarray(data["inchikey"]).astype(str) if "inchikey" in data else None,
            "name": np.asarray(data["name"]).astype(str) if "name" in data else None,
            "formula": np.asarray(data["formula"]).astype(str) if "formula" in data else None,
        }

    raise ValueError(f"Unsupported query embedding file: {path}")


def ensure_2d_embedding(x: np.ndarray, name: str) -> np.ndarray:
    """
    Ensure that an embedding array is two-dimensional.

    Singleton dimensions are squeezed before validation.

    Parameters
    ----------
    x:
        Input embedding array.
    name:
        Name used in error messages.

    Returns
    -------
    numpy.ndarray
        Two-dimensional embedding matrix.

    Raises
    ------
    ValueError
        If the input cannot be converted to a 2D matrix.
    """
    x = np.asarray(x)

    if x.ndim == 3 and 1 in x.shape:
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")

    return x


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Row-wise L2-normalize an embedding matrix.

    After normalization, cosine similarity can be computed as a dot product.

    Parameters
    ----------
    x:
        Input matrix with shape (N, D).
    eps:
        Minimum norm to avoid division by zero.

    Returns
    -------
    numpy.ndarray
        L2-normalized float32 matrix.
    """
    x = np.asarray(x, dtype=np.float32)

    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)

    return x / norms


def find_query_library_pairs(
        query_dir: Path,
        library_dir: Path,
        method_whitelist=None,
):
    """
    Match query embedding files and library embedding files by method name.

    Method names are inferred and normalized using chimera_knn_batch.py helpers.

    Parameters
    ----------
    query_dir:
        Directory containing query embedding files.
    library_dir:
        Directory containing library embedding files.
    method_whitelist:
        Optional list of method names to include.

    Returns
    -------
    list[tuple[str, pathlib.Path, pathlib.Path]]
        Matched tuples: (method, query_file, library_file).
    """
    query_files = sorted([
        *query_dir.glob("*.npy"),
        *query_dir.glob("*.npz"),
    ])

    library_files = sorted(library_dir.glob("library_*.npy"))

    pairs = []

    for q in query_files:
        q_method = ck.normalize_method_name(ck.infer_method_from_filename(q))

        if not q_method:
            continue

        if method_whitelist is not None and q_method not in method_whitelist:
            continue

        for lib in library_files:
            lib_method = ck.normalize_method_name(ck.infer_method_from_filename(lib))

            if q_method == lib_method:
                pairs.append((q_method, q, lib))

    return pairs


# =============================================================================
# 4. Top-K retrieval
# =============================================================================

def knn_cosine_torch_library_on_gpu(
        query_embs,
        lib_embs,
        k=100,
        query_batch_size=256,
        use_float16=True,
        device=None,
):
    """
    Compute cosine Top-K retrieval with the full library resident on GPU.

    Workflow:
        1. Normalize query and library embeddings on CPU.
        2. Move the full library matrix to GPU once.
        3. Process query embeddings in batches.
        4. Compute cosine similarity using matrix multiplication.
        5. Return Top-K library indices and similarity scores.

    Parameters
    ----------
    query_embs:
        Query embedding matrix with shape (N_query, D).
    lib_embs:
        Library embedding matrix with shape (N_library, D).
    k:
        Number of nearest neighbors to retrieve.
    query_batch_size:
        Number of query embeddings per GPU batch.
    use_float16:
        Whether to use float16 on CUDA.
    device:
        Optional torch device. If None, CUDA is used when available.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Top-K indices and cosine similarity scores.
    """
    import time
    import torch

    query_embs = ensure_2d_embedding(
        np.asarray(query_embs, dtype=np.float32),
        "query_embs",
    )

    if not isinstance(lib_embs, np.ndarray):
        lib_embs = np.asarray(lib_embs)

    lib_embs = ensure_2d_embedding(lib_embs, "lib_embs")

    if query_embs.shape[1] != lib_embs.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_embs.shape}, lib={lib_embs.shape}"
        )

    n_query = int(query_embs.shape[0])
    n_lib = int(lib_embs.shape[0])
    k = min(int(k), n_lib)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    device = torch.device(device)

    if device.type != "cuda":
        print("[KNN-GPU] CUDA not available, this function will run on CPU.", flush=True)

    dtype = torch.float16 if device.type == "cuda" and use_float16 else torch.float32

    print(
        f"[KNN-GPU] query={query_embs.shape}, lib={lib_embs.shape}, "
        f"k={k}, query_batch_size={query_batch_size}, device={device}, dtype={dtype}",
        flush=True,
    )

    t0 = time.time()
    query_norm = l2_normalize_np(query_embs).astype(np.float32, copy=False)
    lib_norm = l2_normalize_np(np.asarray(lib_embs, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )
    print(f"[KNN-GPU] CPU normalization done in {time.time() - t0:.2f}s", flush=True)

    t1 = time.time()
    lib_gpu = torch.as_tensor(lib_norm, device=device, dtype=dtype).contiguous()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[KNN-GPU] library moved to GPU once in {time.time() - t1:.2f}s; "
        f"lib_gpu.shape={tuple(lib_gpu.shape)}",
        flush=True,
    )

    all_indices = np.empty((n_query, k), dtype=np.int64)
    all_scores = np.empty((n_query, k), dtype=np.float32)

    for q_start in tqdm(
            range(0, n_query, int(query_batch_size)),
            desc="[KNN-GPU] query batches",
            unit="batch",
    ):
        q_end = min(q_start + int(query_batch_size), n_query)

        q_np = query_norm[q_start:q_end]
        q_gpu = torch.as_tensor(q_np, device=device, dtype=dtype).contiguous()

        with torch.inference_mode():
            sim = q_gpu @ lib_gpu.T

            top_scores, top_indices = torch.topk(
                sim,
                k=k,
                dim=1,
                largest=True,
                sorted=True,
            )

        all_scores[q_start:q_end] = top_scores.detach().float().cpu().numpy()
        all_indices[q_start:q_end] = top_indices.detach().cpu().numpy().astype(np.int64)

        del q_gpu, sim, top_scores, top_indices

    del lib_gpu

    if device.type == "cuda":
        torch.cuda.empty_cache()

    return all_indices, all_scores


def knn_cosine_cpu_blockwise(
        query_embs,
        lib_embs,
        k: int = 100,
        batch_size: int = 128,
):
    """
    CPU fallback for cosine Top-K retrieval.

    This delegates to ck.knn_cosine_many() from chimera_knn_batch.py.

    Parameters
    ----------
    query_embs:
        Query embedding matrix.
    lib_embs:
        Library embedding matrix.
    k:
        Number of nearest neighbors.
    batch_size:
        Query batch size.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Top-K indices and cosine similarity scores.
    """
    return ck.knn_cosine_many(
        query_embs=query_embs,
        lib_embs=lib_embs,
        knn_k=k,
        batch_size=batch_size,
    )


def run_topk_retrieval(
        query_embs,
        lib_embs,
        out_prefix: Path,
        k: int = 100,
        use_cuda: bool = True,
):
    """
    Run or load cached Top-K cosine retrieval.

    Parameters
    ----------
    query_embs:
        Query embedding matrix.
    lib_embs:
        Library embedding matrix.
    out_prefix:
        Output prefix for cache files.
    k:
        Number of nearest neighbors.
    use_cuda:
        Whether to use GPU retrieval when available.

    Returns
    -------
    tuple
        indices_mmap, scores_mmap, idx_path, score_path.
    """
    out_prefix = Path(out_prefix)

    idx_path = out_prefix.with_suffix(f".top{k}.idx.npy")
    score_path = out_prefix.with_suffix(f".top{k}.score.npy")

    if idx_path.exists() and score_path.exists():
        print(f"[cache] TopK exists: {idx_path.name}", flush=True)

        indices = np.load(idx_path, mmap_mode="r")
        scores = np.load(score_path, mmap_mode="r")

        return indices, scores, idx_path, score_path

    if use_cuda:
        indices, scores = knn_cosine_torch_library_on_gpu(
            query_embs=query_embs,
            lib_embs=lib_embs,
            k=k,
            query_batch_size=CUDA_QUERY_BATCH_SIZE,
            use_float16=CUDA_USE_FLOAT16,
            device=None,
        )
    else:
        indices, scores = knn_cosine_cpu_blockwise(
            query_embs=query_embs,
            lib_embs=lib_embs,
            k=k,
            batch_size=CPU_BATCH_SIZE,
        )

    np.save(idx_path, indices)
    np.save(score_path, scores)

    print(f"[saved] {idx_path}", flush=True)
    print(f"[saved] {score_path}", flush=True)

    indices_mmap = np.load(idx_path, mmap_mode="r")
    scores_mmap = np.load(score_path, mmap_mode="r")

    return indices_mmap, scores_mmap, idx_path, score_path


# =============================================================================
# 5. Candidate decomposition
# =============================================================================

def cluster_topk_by_embedding(
        topk_embeddings: np.ndarray,
        n_clusters: int,
) -> np.ndarray:
    """
    Cluster retained candidates by embedding distance.

    Candidate embeddings are L2-normalized, converted to cosine distances, and
    clustered using agglomerative clustering.

    Parameters
    ----------
    topk_embeddings:
        Candidate embedding matrix with shape (K, D).
    n_clusters:
        Number of clusters to form.

    Returns
    -------
    numpy.ndarray
        Integer cluster labels.
    """
    x = np.asarray(topk_embeddings, dtype=np.float32)

    if x.ndim != 2:
        raise ValueError(f"topk_embeddings must be 2D, got shape={x.shape}")

    n = x.shape[0]

    if n == 0:
        return np.array([], dtype=int)

    n_clusters = int(n_clusters)
    n_clusters = max(1, min(n_clusters, n))

    if n_clusters == 1:
        return np.zeros(n, dtype=int)

    x = l2_normalize_np(x)
    dist = cosine_distances(x)

    try:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric="precomputed",
            linkage="average",
        )
    except TypeError:
        model = AgglomerativeClustering(
            n_clusters=n_clusters,
            affinity="precomputed",
            linkage="average",
        )

    labels = model.fit_predict(dist)

    return labels.astype(int)


def build_neighbor_df_gmm_x_embedding_cluster(
        *,
        query_index: int,
        neighbor_indices: np.ndarray,
        neighbor_scores: np.ndarray,
        lib: dict[str, Any],
        lib_embs: np.ndarray,
        max_precursor_groups: int = 5,
        use_adaptive_k: bool = False,
) -> pd.DataFrame:
    """
    Build the candidate neighbor table for one query spectrum.

    Steps:
        1. Receive precomputed Top-K indices and similarity scores.
        2. Optionally apply adaptive-K truncation.
        3. Estimate precursor/component group count using GMM on candidate precursor m/z.
        4. Cluster retained candidate embeddings into the estimated number of groups.
        5. Return candidate-level table with metadata and diagnostics.

    Parameters
    ----------
    query_index:
        Query spectrum index.
    neighbor_indices:
        Top-K library indices.
    neighbor_scores:
        Top-K cosine similarity scores.
    lib:
        Library dictionary from ck.load_embedding_file().
    lib_embs:
        Library embedding matrix.
    max_precursor_groups:
        Maximum GMM precursor-group count.
    use_adaptive_k:
        Whether to use adaptive-K truncation.

    Returns
    -------
    pandas.DataFrame
        Candidate neighbor table for one query.
    """
    lib_precursor_mz = lib["precursor_mz"]
    lib_smiles = lib.get("smiles", None)
    lib_inchikey = lib.get("inchikey", None)
    lib_name = lib.get("name", None)
    lib_formula = lib.get("formula", None)

    neighbor_indices = np.asarray(neighbor_indices, dtype=np.int64)
    neighbor_scores = np.asarray(neighbor_scores, dtype=np.float32)

    valid = neighbor_indices >= 0
    neighbor_indices = neighbor_indices[valid]
    neighbor_scores = neighbor_scores[valid]

    if len(neighbor_indices) == 0:
        return pd.DataFrame()

    original_knn_k = int(len(neighbor_indices))

    adaptive_info = {
        "adaptive_k": original_knn_k,
        "selected_segments": 0,
        "selected_model": 0,
        "breakpoints": [],
        "last_breakpoint": -1,
        "bic_by_segments": {},
        "sse_by_segments": {},
        "score_range": np.nan,
        "reason": "disabled",
    }

    if use_adaptive_k:
        adaptive_info = estimate_adaptive_k_piecewise_bic_general(
            neighbor_scores,
            k_min=ADAPTIVE_K_MIN,
            k_max=min(ADAPTIVE_K_MAX, original_knn_k),
            max_segments=ADAPTIVE_MAX_SEGMENTS,
            min_segment_size=ADAPTIVE_MIN_SEGMENT_SIZE,
            fallback_k=ADAPTIVE_FALLBACK_K,
            min_score_range=ADAPTIVE_MIN_SCORE_RANGE,
            keep_before_last_segment=ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT,
        )

        adaptive_k = int(adaptive_info["adaptive_k"])
        adaptive_k = int(max(1, min(adaptive_k, original_knn_k)))

        neighbor_indices = neighbor_indices[:adaptive_k]
        neighbor_scores = neighbor_scores[:adaptive_k]
    else:
        adaptive_k = original_knn_k

    top_precursor_mz = np.asarray(lib_precursor_mz[neighbor_indices], dtype=np.float32)

    _, estimated_x = ck.estimate_precursor_groups_gmm(
        top_precursor_mz,
        max_groups=max_precursor_groups,
    )

    estimated_x = int(estimated_x)
    estimated_x = max(1, min(estimated_x, len(neighbor_indices)))

    top_embeddings = np.asarray(lib_embs[neighbor_indices], dtype=np.float32)

    cluster_labels = cluster_topk_by_embedding(
        topk_embeddings=top_embeddings,
        n_clusters=estimated_x,
    )

    rows = []

    for rank_i, lib_idx in enumerate(neighbor_indices):
        row = {
            "query_index": int(query_index),
            "library_index": int(lib_idx),
            "rank": int(rank_i + 1),
            "similarity": float(neighbor_scores[rank_i]),
            "cluster": int(cluster_labels[rank_i]),
            "estimated_n_precursor_groups": int(estimated_x),
            "library_precursor_mz": float(lib_precursor_mz[lib_idx]),
            "adaptive_k": int(adaptive_k),
            "original_knn_k": int(original_knn_k),
        }

        if SAVE_ADAPTIVE_K_DIAGNOSTICS:
            row.update(compact_adaptive_info_for_row(adaptive_info))

        if lib_smiles is not None:
            row["smiles"] = str(lib_smiles[lib_idx])
            row["library_smiles"] = str(lib_smiles[lib_idx])

        if lib_inchikey is not None:
            row["inchikey"] = str(lib_inchikey[lib_idx])
            row["library_inchikey"] = str(lib_inchikey[lib_idx])

        if lib_name is not None:
            row["name"] = str(lib_name[lib_idx])
            row["library_name"] = str(lib_name[lib_idx])

        if lib_formula is not None:
            row["formula"] = str(lib_formula[lib_idx])
            row["library_formula"] = str(lib_formula[lib_idx])

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 6. Scaffold, backbone confidence, and pruning
# =============================================================================

def empty_scaffold_df_for_query(query_index, query_npz, lib_npy, method):
    """
    Create an empty scaffold/MCS table for one query.

    This preserves a stable output schema when scaffold computation is disabled,
    unavailable, or failed.

    Parameters
    ----------
    query_index:
        Query spectrum index.
    query_npz:
        Query embedding file path.
    lib_npy:
        Library embedding file path.
    method:
        Embedding method name.

    Returns
    -------
    pandas.DataFrame
        One-row placeholder scaffold table.
    """
    return pd.DataFrame(
        [
            {
                "query_index": int(query_index),
                "query_file": str(query_npz),
                "library_file": str(lib_npy),
                "method": method,
                "cluster": np.nan,
                "smarts": "",
                "scaffold_smiles": "",
                "n_molecules": 0,
                "mean_similarity": np.nan,
                "max_similarity": np.nan,
                "confidence": np.nan,
                "mcs_num_atoms": 0,
                "mcs_num_bonds": 0,
                "mcs_canceled": False,
            }
        ]
    )


def compute_scaffold_for_neighbor_df(
        neighbor_df: pd.DataFrame,
        *,
        query_index: int,
        query_npz: Path,
        lib_npy: Path,
        method: str,
) -> pd.DataFrame:
    """
    Compute cluster-level MCS/scaffold evidence for one query.

    This calls ck.build_cluster_scaffold_df() without modifying
    chimera_knn_batch.py.

    Parameters
    ----------
    neighbor_df:
        Candidate neighbor table for one query.
    query_index:
        Query spectrum index.
    query_npz:
        Query embedding file path.
    lib_npy:
        Library embedding file path.
    method:
        Embedding method name.

    Returns
    -------
    pandas.DataFrame
        Cluster-level scaffold/MCS table.
    """
    if neighbor_df is None or len(neighbor_df) == 0:
        return empty_scaffold_df_for_query(query_index, query_npz, lib_npy, method)

    try:
        scaffold_df = ck.build_cluster_scaffold_df(
            neighbor_df,
            cluster_col="cluster",
            smiles_col="smiles",
            similarity_col="similarity",
            min_cluster_size=MIN_CLUSTER_SIZE,
            max_mols_per_cluster=MAX_MOLS_PER_CLUSTER,
            mcs_timeout=MCS_TIMEOUT,
            min_num_atoms=MIN_NUM_ATOMS,
            deduplicate=True,
        )
    except Exception as e:
        print(
            f"[scaffold] WARNING failed query={query_index}, error={repr(e)}",
            flush=True,
        )
        scaffold_df = pd.DataFrame()

    if scaffold_df is None or len(scaffold_df) == 0:
        return empty_scaffold_df_for_query(query_index, query_npz, lib_npy, method)

    scaffold_df = scaffold_df.copy()
    scaffold_df["query_index"] = int(query_index)
    scaffold_df["query_file"] = str(query_npz)
    scaffold_df["library_file"] = str(lib_npy)
    scaffold_df["method"] = method

    return scaffold_df


def build_confidence_df_from_scaffold(scaffold_all_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert cluster scaffold/MCS output into a standardized confidence table.

    Parameters
    ----------
    scaffold_all_df:
        Concatenated scaffold table.

    Returns
    -------
    pandas.DataFrame
        Standardized scaffold confidence table.
    """
    expected_cols = [
        "query_file",
        "library_file",
        "method",
        "query_index",
        "cluster",
        "smarts",
        "scaffold_smiles",
        "n_molecules",
        "mean_similarity",
        "max_similarity",
        "confidence",
        "mcs_num_atoms",
        "mcs_num_bonds",
        "mcs_canceled",
    ]

    scaffold_all_df = scaffold_all_df.copy()

    for col in expected_cols:
        if col not in scaffold_all_df.columns:
            scaffold_all_df[col] = np.nan

    confidence_df = scaffold_all_df[expected_cols].copy()

    confidence_df = confidence_df.rename(
        columns={
            "cluster": "precursor_group_id",
            "smarts": "scaffold_smarts",
            "confidence": "scaffold_confidence",
            "n_molecules": "scaffold_n_molecules",
            "mean_similarity": "scaffold_mean_similarity",
            "max_similarity": "scaffold_max_similarity",
        }
    )

    return confidence_df


def add_backbone_confidence_to_neighbor_df(
        neighbor_df: pd.DataFrame,
        scaffold_df: pd.DataFrame | None = None,
        *,
        cluster_col: str = "cluster",
        smiles_col: str = "smiles",
        similarity_col: str = "similarity",
        max_mols_per_cluster: int = MAX_MOLS_PER_CLUSTER,
        min_scaffold_atoms_for_conf: int = STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF,
        weight_cluster_support: float = STRUCT_WEIGHT_CLUSTER_SUPPORT,
        weight_overlap: float = STRUCT_WEIGHT_OVERLAP,
        weight_candidate_coverage: float = STRUCT_WEIGHT_CANDIDATE_COVERAGE,
        weight_exact_match: float = STRUCT_WEIGHT_EXACT_MATCH,
) -> pd.DataFrame:
    """
    Add MCS/scaffold-based backbone confidence to each candidate row.

    The manuscript-style score is:

        C_backbone =
            w_cluster  * C_cluster
          + w_overlap  * R_overlap
          + w_coverage * R_coverage
          + w_exact    * I_exact

    This evaluates whether each candidate is structurally supported by its own
    latent-space cluster, not by query SMILES.

    Parameters
    ----------
    neighbor_df:
        Candidate table for one query.
    scaffold_df:
        Cluster-level MCS/scaffold table from compute_scaffold_for_neighbor_df().
    cluster_col:
        Cluster-label column.
    smiles_col:
        Candidate SMILES column.
    similarity_col:
        Similarity-score column. Kept for interface consistency.
    max_mols_per_cluster:
        Maximum cluster size used for capped cluster-size support.
    min_scaffold_atoms_for_conf:
        Minimum MCS atom count for reliable support.
    weight_cluster_support:
        Weight of cluster support.
    weight_overlap:
        Weight of scaffold/MCS overlap.
    weight_candidate_coverage:
        Weight of candidate coverage.
    weight_exact_match:
        Weight of exact scaffold agreement.

    Returns
    -------
    pandas.DataFrame
        Candidate table with backbone-confidence and MCS/scaffold columns.
    """
    if neighbor_df is None or len(neighbor_df) == 0:
        return neighbor_df

    out = neighbor_df.copy()

    default_cols = {
        "cluster_scaffold_smiles": "",
        "cluster_mcs_smarts": "",
        "cluster_mcs_num_atoms": 0,
        "cluster_mcs_num_bonds": 0,
        "cluster_scaffold_confidence": np.nan,
        "cluster_n_molecules": 0,
        "candidate_scaffold_smiles": "",
        "candidate_scaffold_num_atoms": 0,
        "backbone_cluster_support": 0.0,
        "backbone_scaffold_overlap": 0.0,
        "backbone_candidate_coverage": 0.0,
        "backbone_exact_scaffold_match": 0.0,
        "backbone_confidence_score": 0.0,
        "backbone_confidence_level": "Low",
        "backbone_confidence_pass": False,
        "backbone_confidence_reason": "not_computed",
    }

    for col, value in default_cols.items():
        if col not in out.columns:
            out[col] = value

    if scaffold_df is None or len(scaffold_df) == 0:
        cluster_sizes = out.groupby(cluster_col).size().to_dict()

        c_cluster = out[cluster_col].map(
            lambda g: min(1.0, float(cluster_sizes.get(g, 0)) / float(max_mols_per_cluster))
        ).astype(float)

        out["cluster_n_molecules"] = out[cluster_col].map(lambda g: int(cluster_sizes.get(g, 0)))
        out["backbone_cluster_support"] = c_cluster
        out["backbone_confidence_score"] = (
            weight_cluster_support * out["backbone_cluster_support"]
        )
        out["backbone_confidence_level"] = out["backbone_confidence_score"].map(
            _structure_confidence_level
        )
        out["backbone_confidence_pass"] = (
            out["backbone_confidence_score"] >= STRUCT_CONF_MEDIUM
        )
        out["backbone_confidence_reason"] = "cluster_size_only_no_mcs"

        return out

    scaffold_info = scaffold_df.copy()

    rename_map = {}

    if "smarts" in scaffold_info.columns:
        rename_map["smarts"] = "cluster_mcs_smarts"

    if "scaffold_smiles" in scaffold_info.columns:
        rename_map["scaffold_smiles"] = "cluster_scaffold_smiles"

    if "mcs_num_atoms" in scaffold_info.columns:
        rename_map["mcs_num_atoms"] = "cluster_mcs_num_atoms"

    if "mcs_num_bonds" in scaffold_info.columns:
        rename_map["mcs_num_bonds"] = "cluster_mcs_num_bonds"

    if "confidence" in scaffold_info.columns:
        rename_map["confidence"] = "cluster_scaffold_confidence"

    if "n_molecules" in scaffold_info.columns:
        rename_map["n_molecules"] = "cluster_n_molecules"

    scaffold_info = scaffold_info.rename(columns=rename_map)

    keep_cols = [
        cluster_col,
        "cluster_mcs_smarts",
        "cluster_scaffold_smiles",
        "cluster_mcs_num_atoms",
        "cluster_mcs_num_bonds",
        "cluster_scaffold_confidence",
        "cluster_n_molecules",
    ]

    for col in keep_cols:
        if col not in scaffold_info.columns:
            scaffold_info[col] = np.nan

    scaffold_info = scaffold_info[keep_cols].drop_duplicates(subset=[cluster_col])

    drop_before_merge = [
        "cluster_mcs_smarts",
        "cluster_scaffold_smiles",
        "cluster_mcs_num_atoms",
        "cluster_mcs_num_bonds",
        "cluster_scaffold_confidence",
        "cluster_n_molecules",
    ]

    out = out.drop(
        columns=[c for c in drop_before_merge if c in out.columns],
        errors="ignore",
    )

    out = out.merge(
        scaffold_info,
        on=cluster_col,
        how="left",
    )

    candidate_scaffold_smiles = []
    candidate_scaffold_num_atoms = []

    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold

        rdkit_available = True
    except Exception:
        rdkit_available = False

    for _, row in out.iterrows():
        smi = _normalize_smiles_like(row.get(smiles_col, ""))

        if not rdkit_available or not smi:
            candidate_scaffold_smiles.append("")
            candidate_scaffold_num_atoms.append(0)
            continue

        try:
            mol = Chem.MolFromSmiles(smi)

            if mol is None:
                candidate_scaffold_smiles.append("")
                candidate_scaffold_num_atoms.append(0)
                continue

            scaffold_mol = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold_mol) if scaffold_mol is not None else ""
            n_atoms = int(scaffold_mol.GetNumAtoms()) if scaffold_mol is not None else 0

            candidate_scaffold_smiles.append(scaffold_smi)
            candidate_scaffold_num_atoms.append(n_atoms)

        except Exception:
            candidate_scaffold_smiles.append("")
            candidate_scaffold_num_atoms.append(0)

    out["candidate_scaffold_smiles"] = candidate_scaffold_smiles
    out["candidate_scaffold_num_atoms"] = candidate_scaffold_num_atoms

    out["cluster_mcs_num_atoms"] = out["cluster_mcs_num_atoms"].map(_safe_int)
    out["cluster_mcs_num_bonds"] = out["cluster_mcs_num_bonds"].map(_safe_int)
    out["cluster_n_molecules"] = out["cluster_n_molecules"].map(_safe_int)
    out["cluster_scaffold_confidence"] = out["cluster_scaffold_confidence"].map(_safe_float)

    out["cluster_scaffold_smiles"] = out["cluster_scaffold_smiles"].fillna("").astype(str)
    out["cluster_mcs_smarts"] = out["cluster_mcs_smarts"].fillna("").astype(str)

    size_support = out["cluster_n_molecules"].map(
        lambda n: min(1.0, float(max(0, int(n))) / float(max_mols_per_cluster))
    )

    scaffold_conf = out["cluster_scaffold_confidence"].astype(float)

    out["backbone_cluster_support"] = np.where(
        np.isfinite(scaffold_conf) & (scaffold_conf > 0),
        scaffold_conf,
        size_support,
    ).astype(float)

    out["backbone_cluster_support"] = np.clip(
        out["backbone_cluster_support"],
        0.0,
        1.0,
    )

    n_match = out["cluster_mcs_num_atoms"].astype(float)
    n_scaffold = out["candidate_scaffold_num_atoms"].map(lambda x: max(float(x), 1.0))

    out["backbone_scaffold_overlap"] = np.clip(
        n_match / n_scaffold,
        0.0,
        1.0,
    )

    out["backbone_candidate_coverage"] = (
        out["cluster_mcs_num_atoms"].astype(int) >= int(min_scaffold_atoms_for_conf)
    ).astype(float)

    out["backbone_exact_scaffold_match"] = (
        out["candidate_scaffold_smiles"].astype(str)
        == out["cluster_scaffold_smiles"].astype(str)
    ).astype(float)

    empty_scaffold_mask = (
        out["candidate_scaffold_smiles"].astype(str).str.len() == 0
    ) | (
        out["cluster_scaffold_smiles"].astype(str).str.len() == 0
    )

    out.loc[empty_scaffold_mask, "backbone_exact_scaffold_match"] = 0.0

    out["backbone_confidence_score"] = (
        weight_cluster_support * out["backbone_cluster_support"].astype(float)
        + weight_overlap * out["backbone_scaffold_overlap"].astype(float)
        + weight_candidate_coverage * out["backbone_candidate_coverage"].astype(float)
        + weight_exact_match * out["backbone_exact_scaffold_match"].astype(float)
    )

    no_valid_mcs = out["cluster_mcs_num_atoms"].astype(int) <= 0

    out.loc[no_valid_mcs, "backbone_confidence_score"] = 0.0

    out["backbone_confidence_score"] = np.clip(
        out["backbone_confidence_score"].astype(float),
        0.0,
        1.0,
    )

    out["backbone_confidence_level"] = out["backbone_confidence_score"].map(
        _structure_confidence_level
    )

    out["backbone_confidence_pass"] = (
        out["backbone_confidence_score"] >= STRUCT_CONF_MEDIUM
    )

    out["backbone_confidence_reason"] = np.where(
        no_valid_mcs,
        "no_valid_mcs_or_scaffold",
        "mcs_scaffold_supported",
    )

    return out


def summarize_top_recommendations_with_compound_pruning(
        neighbor_df: pd.DataFrame,
        scaffold_df: pd.DataFrame | None = None,
        *,
        top_n_per_group: int = TOP_N_PER_GROUP,
        cluster_col: str = "cluster",
        similarity_col: str = "similarity",
        use_structure_confidence: bool = USE_STRUCTURE_CONFIDENCE,
        confidence_col: str = "backbone_confidence_score",
        confidence_threshold: float = STRUCT_CONF_MEDIUM,
) -> pd.DataFrame:
    """
    Select Top-N recommendations per cluster with compound-level pruning.

    For each cluster:
        1. Sort candidates by similarity.
        2. If structure confidence is enabled, remove candidates below threshold.
        3. Select top_n_per_group candidates from the remaining candidates.

    Parameters
    ----------
    neighbor_df:
        Candidate table for one query.
    scaffold_df:
        Cluster scaffold table. Used only if confidence scores are absent.
    top_n_per_group:
        Number of recommendations per cluster.
    cluster_col:
        Cluster-label column.
    similarity_col:
        Similarity-score column.
    use_structure_confidence:
        Whether to apply confidence-based pruning.
    confidence_col:
        Backbone-confidence score column.
    confidence_threshold:
        Minimum score required for retention.

    Returns
    -------
    pandas.DataFrame
        Final recommendation table.
    """
    if neighbor_df is None or len(neighbor_df) == 0:
        return pd.DataFrame()

    df = neighbor_df.copy()

    if use_structure_confidence:
        if confidence_col not in df.columns:
            df = add_backbone_confidence_to_neighbor_df(
                neighbor_df=df,
                scaffold_df=scaffold_df,
                cluster_col=cluster_col,
                similarity_col=similarity_col,
            )

        df[confidence_col] = df[confidence_col].map(_safe_float)

    rows = []

    for cluster_id, g in df.groupby(cluster_col, sort=True):
        g = g.copy()

        if similarity_col in g.columns:
            g = g.sort_values(similarity_col, ascending=False)
        elif "rank" in g.columns:
            g = g.sort_values("rank", ascending=True)

        if use_structure_confidence:
            g_pass = g[
                g[confidence_col].astype(float) >= float(confidence_threshold)
            ].copy()

            if len(g_pass) == 0:
                continue

            g = g_pass

            if similarity_col in g.columns:
                g = g.sort_values(similarity_col, ascending=False)
            elif "rank" in g.columns:
                g = g.sort_values("rank", ascending=True)

        selected = g.head(int(top_n_per_group)).copy()

        selected["recommendation_rank_in_cluster"] = np.arange(
            1,
            len(selected) + 1,
            dtype=int,
        )

        selected["recommendation_cluster"] = cluster_id
        selected["compound_level_pruning_used"] = bool(use_structure_confidence)
        selected["compound_level_confidence_threshold"] = (
            float(confidence_threshold) if use_structure_confidence else np.nan
        )

        rows.append(selected)

    if len(rows) == 0:
        return pd.DataFrame()

    rec = pd.concat(rows, ignore_index=True)

    sort_cols = []

    if "query_index" in rec.columns:
        sort_cols.append("query_index")

    if cluster_col in rec.columns:
        sort_cols.append(cluster_col)

    if "recommendation_rank_in_cluster" in rec.columns:
        sort_cols.append("recommendation_rank_in_cluster")

    if sort_cols:
        rec = rec.sort_values(sort_cols).reset_index(drop=True)

    return rec


# =============================================================================
# 7. Parallel post-processing
# =============================================================================

def _postprocess_worker_init(
        lib_npy_path,
        metadata_csv,
        top_idx_path,
        top_score_path,
        query_file,
        method,
        progress_queue,
):
    """
    Initialize one multiprocessing worker for post-KNN processing.

    Each worker opens large arrays independently using memory mapping. This avoids
    sending large arrays from the main process to child processes.

    Parameters
    ----------
    lib_npy_path:
        Path to library embedding file.
    metadata_csv:
        Path to library metadata CSV.
    top_idx_path:
        Path to cached Top-K index array.
    top_score_path:
        Path to cached Top-K score array.
    query_file:
        Path to query embedding file.
    method:
        Embedding method name.
    progress_queue:
        Queue for progress reporting.
    """
    global _WORKER_LIB
    global _WORKER_LIB_EMBS
    global _WORKER_TOP_INDICES
    global _WORKER_TOP_SCORES
    global _WORKER_QUERY_FILE
    global _WORKER_LIB_FILE
    global _WORKER_METHOD
    global _WORKER_PROGRESS_QUEUE

    lib_npy_path = Path(lib_npy_path)
    metadata_csv = Path(metadata_csv)
    top_idx_path = Path(top_idx_path)
    top_score_path = Path(top_score_path)

    _WORKER_LIB = ck.load_embedding_file(
        lib_npy_path,
        metadata_csv=metadata_csv,
    )

    _WORKER_LIB_EMBS = _WORKER_LIB["embeddings"]
    _WORKER_TOP_INDICES = np.load(top_idx_path, mmap_mode="r")
    _WORKER_TOP_SCORES = np.load(top_score_path, mmap_mode="r")

    _WORKER_QUERY_FILE = Path(query_file)
    _WORKER_LIB_FILE = lib_npy_path
    _WORKER_METHOD = str(method)
    _WORKER_PROGRESS_QUEUE = progress_queue

    print(
        f"[worker init] pid={os.getpid()}, method={_WORKER_METHOD}, "
        f"lib_embs.shape={_WORKER_LIB_EMBS.shape}, "
        f"top_indices.shape={_WORKER_TOP_INDICES.shape}",
        flush=True,
    )


def _process_query_chunk_worker(args):
    """
    Process a continuous query-index range inside one worker.

    For every query, this function builds neighbor candidates, computes scaffold
    evidence, calculates backbone confidence, applies pruning, and saves chunk
    outputs.

    Parameters
    ----------
    args:
        Tuple containing:
        chunk_id, q_start, q_end, out_dir, compute_scaffold

    Returns
    -------
    dict
        Saved chunk file paths and chunk metadata.
    """
    (
        chunk_id,
        q_start,
        q_end,
        out_dir,
        compute_scaffold,
    ) = args

    global _WORKER_LIB
    global _WORKER_LIB_EMBS
    global _WORKER_TOP_INDICES
    global _WORKER_TOP_SCORES
    global _WORKER_QUERY_FILE
    global _WORKER_LIB_FILE
    global _WORKER_METHOD
    global _WORKER_PROGRESS_QUEUE

    out_dir = Path(out_dir)
    chunk_dir = out_dir / "chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    q_start = int(q_start)
    q_end = int(q_end)
    chunk_id = int(chunk_id)

    n_chunk = q_end - q_start

    all_neighbors = []
    all_recommendations = []
    all_scaffolds = []

    estimated_x = np.empty(n_chunk, dtype=np.int64)
    adaptive_k_arr = np.empty(n_chunk, dtype=np.int64)
    adaptive_segments_arr = np.empty(n_chunk, dtype=np.int64)

    done_since_report = 0

    try:
        for local_i, q_idx in enumerate(range(q_start, q_end)):
            neighbor_indices = np.asarray(_WORKER_TOP_INDICES[q_idx], dtype=np.int64)
            neighbor_scores = np.asarray(_WORKER_TOP_SCORES[q_idx], dtype=np.float32)

            neighbor_df = build_neighbor_df_gmm_x_embedding_cluster(
                query_index=q_idx,
                neighbor_indices=neighbor_indices,
                neighbor_scores=neighbor_scores,
                lib=_WORKER_LIB,
                lib_embs=_WORKER_LIB_EMBS,
                max_precursor_groups=MAX_PRECURSOR_GROUPS,
                use_adaptive_k=USE_ADAPTIVE_K,
            )

            if neighbor_df is not None and len(neighbor_df) > 0:
                neighbor_df["query_file"] = str(_WORKER_QUERY_FILE)
                neighbor_df["library_file"] = str(_WORKER_LIB_FILE)
                neighbor_df["method"] = _WORKER_METHOD

                estimated_x[local_i] = int(
                    neighbor_df["estimated_n_precursor_groups"].iloc[0]
                )

                if "adaptive_k" in neighbor_df.columns:
                    adaptive_k_arr[local_i] = int(neighbor_df["adaptive_k"].iloc[0])
                else:
                    adaptive_k_arr[local_i] = int(len(neighbor_indices))

                if "adaptive_selected_segments" in neighbor_df.columns:
                    adaptive_segments_arr[local_i] = int(
                        neighbor_df["adaptive_selected_segments"].iloc[0]
                    )
                elif "adaptive_selected_model" in neighbor_df.columns:
                    adaptive_segments_arr[local_i] = int(
                        neighbor_df["adaptive_selected_model"].iloc[0]
                    )
                else:
                    adaptive_segments_arr[local_i] = 0
            else:
                estimated_x[local_i] = 0
                adaptive_k_arr[local_i] = 0
                adaptive_segments_arr[local_i] = 0
                neighbor_df = pd.DataFrame()

            if compute_scaffold:
                scaffold_df = compute_scaffold_for_neighbor_df(
                    neighbor_df,
                    query_index=q_idx,
                    query_npz=_WORKER_QUERY_FILE,
                    lib_npy=_WORKER_LIB_FILE,
                    method=_WORKER_METHOD,
                )
            else:
                scaffold_df = empty_scaffold_df_for_query(
                    q_idx,
                    _WORKER_QUERY_FILE,
                    _WORKER_LIB_FILE,
                    _WORKER_METHOD,
                )

            if USE_STRUCTURE_CONFIDENCE and compute_scaffold:
                neighbor_df = add_backbone_confidence_to_neighbor_df(
                    neighbor_df=neighbor_df,
                    scaffold_df=scaffold_df,
                    cluster_col="cluster",
                    smiles_col="smiles",
                    similarity_col="similarity",
                    max_mols_per_cluster=MAX_MOLS_PER_CLUSTER,
                    min_scaffold_atoms_for_conf=STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF,
                )

            recommendation_df = summarize_top_recommendations_with_compound_pruning(
                neighbor_df=neighbor_df,
                scaffold_df=scaffold_df,
                top_n_per_group=TOP_N_PER_GROUP,
                cluster_col="cluster",
                similarity_col="similarity",
                use_structure_confidence=bool(USE_STRUCTURE_CONFIDENCE and compute_scaffold),
                confidence_col="backbone_confidence_score",
                confidence_threshold=STRUCT_CONF_MEDIUM,
            )

            if recommendation_df is not None and len(recommendation_df) > 0:
                recommendation_df["query_file"] = str(_WORKER_QUERY_FILE)
                recommendation_df["library_file"] = str(_WORKER_LIB_FILE)
                recommendation_df["method"] = _WORKER_METHOD
            else:
                recommendation_df = pd.DataFrame()

            all_neighbors.append(neighbor_df)
            all_scaffolds.append(scaffold_df)
            all_recommendations.append(recommendation_df)

            done_since_report += 1

            if done_since_report >= POSTPROCESS_PROGRESS_EVERY:
                if _WORKER_PROGRESS_QUEUE is not None:
                    _WORKER_PROGRESS_QUEUE.put(done_since_report)
                done_since_report = 0

    finally:
        if done_since_report > 0 and _WORKER_PROGRESS_QUEUE is not None:
            _WORKER_PROGRESS_QUEUE.put(done_since_report)

    neighbor_all_df = (
        pd.concat(all_neighbors, ignore_index=True)
        if len(all_neighbors)
        else pd.DataFrame()
    )

    scaffold_all_df = (
        pd.concat(all_scaffolds, ignore_index=True)
        if len(all_scaffolds)
        else pd.DataFrame()
    )

    recommendation_all_df = (
        pd.concat(all_recommendations, ignore_index=True)
        if len(all_recommendations)
        else pd.DataFrame()
    )

    confidence_df = build_confidence_df_from_scaffold(scaffold_all_df)

    prefix = f"chunk_{chunk_id:04d}_q{q_start}_{q_end}"

    neighbor_csv = chunk_dir / f"{prefix}.neighbors.csv"
    recommendation_csv = chunk_dir / f"{prefix}.chimera_recommendations.csv"
    scaffold_csv = chunk_dir / f"{prefix}.cluster_scaffolds.csv"
    confidence_csv = chunk_dir / f"{prefix}.scaffold_confidence.csv"
    estimated_x_npy = chunk_dir / f"{prefix}.estimated_n_precursor_groups.npy"
    adaptive_k_npy = chunk_dir / f"{prefix}.adaptive_k.npy"
    adaptive_segments_npy = chunk_dir / f"{prefix}.adaptive_selected_segments.npy"

    neighbor_all_df.to_csv(neighbor_csv, index=False, encoding="utf-8-sig")
    recommendation_all_df.to_csv(recommendation_csv, index=False, encoding="utf-8-sig")
    scaffold_all_df.to_csv(scaffold_csv, index=False, encoding="utf-8-sig")
    confidence_df.to_csv(confidence_csv, index=False, encoding="utf-8-sig")

    np.save(estimated_x_npy, estimated_x)
    np.save(adaptive_k_npy, adaptive_k_arr)
    np.save(adaptive_segments_npy, adaptive_segments_arr)

    return {
        "chunk_id": chunk_id,
        "q_start": q_start,
        "q_end": q_end,
        "pid": int(os.getpid()),
        "n_query": int(n_chunk),
        "neighbor_csv": str(neighbor_csv),
        "recommendation_csv": str(recommendation_csv),
        "scaffold_csv": str(scaffold_csv),
        "confidence_csv": str(confidence_csv),
        "estimated_x_npy": str(estimated_x_npy),
        "adaptive_k_npy": str(adaptive_k_npy),
        "adaptive_segments_npy": str(adaptive_segments_npy),
    }


def process_queries_after_topk_parallel_mmap(
        *,
        query_npz: Path,
        lib_npy: Path,
        method: str,
        top_idx_path: Path,
        top_score_path: Path,
        out_dir: Path,
        n_workers: int | None = None,
        compute_scaffold: bool = False,
        merge_chunk_csv: bool = True,
):
    """
    Run multiprocessing post-processing after Top-K retrieval.

    Large arrays are opened inside each worker with mmap.

    Parameters
    ----------
    query_npz:
        Query embedding file path.
    lib_npy:
        Library embedding file path.
    method:
        Embedding method name.
    top_idx_path:
        Cached Top-K index array path.
    top_score_path:
        Cached Top-K score array path.
    out_dir:
        Output directory for this method.
    n_workers:
        Number of worker processes.
    compute_scaffold:
        Whether to compute MCS/scaffold evidence.
    merge_chunk_csv:
        Whether to merge chunk-level CSV files.

    Returns
    -------
    dict
        Output arrays, paths, summaries, and manifest path.
    """
    query_npz = Path(query_npz)
    lib_npy = Path(lib_npy)
    top_idx_path = Path(top_idx_path)
    top_score_path = Path(top_score_path)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    top_indices_mmap = np.load(top_idx_path, mmap_mode="r")
    n_query = int(top_indices_mmap.shape[0])
    del top_indices_mmap

    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)

    n_workers = int(max(1, n_workers))
    n_workers = min(n_workers, n_query)

    ranges = split_query_ranges(n_query, n_workers)

    print(
        f"[parallel-mmap] method={method}, n_query={n_query}, "
        f"n_workers={n_workers}",
        flush=True,
    )

    for i, (s, e) in enumerate(ranges):
        print(f"[parallel-mmap] chunk {i:04d}: q{s}:{e}", flush=True)

    tasks = []

    for chunk_id, (q_start, q_end) in enumerate(ranges):
        tasks.append(
            (
                int(chunk_id),
                int(q_start),
                int(q_end),
                str(out_dir),
                bool(compute_scaffold),
            )
        )

    ctx = mp.get_context("spawn")

    manager = ctx.Manager()
    progress_queue = manager.Queue()

    chunk_results = []

    with ProcessPoolExecutor(
            max_workers=n_workers,
            mp_context=ctx,
            initializer=_postprocess_worker_init,
            initargs=(
                    str(lib_npy),
                    str(METADATA_CSV),
                    str(top_idx_path),
                    str(top_score_path),
                    str(query_npz),
                    str(method),
                    progress_queue,
            ),
    ) as ex:
        futures = [ex.submit(_process_query_chunk_worker, task) for task in tasks]
        future_set = set(futures)

        with tqdm(
                total=n_query,
                desc=f"[{method}] postprocess queries",
                unit="query",
                mininterval=1.0,
        ) as pbar:
            completed_queries = 0

            while future_set:
                while True:
                    try:
                        inc = progress_queue.get_nowait()
                    except Exception:
                        break

                    inc = int(inc)
                    completed_queries += inc
                    pbar.update(inc)

                done_futures = [f for f in list(future_set) if f.done()]

                for fut in done_futures:
                    future_set.remove(fut)

                    res = fut.result()
                    chunk_results.append(res)

                    print(
                        f"\n[chunk done] chunk={res['chunk_id']:04d}, "
                        f"q={res['q_start']}:{res['q_end']}, "
                        f"pid={res['pid']}",
                        flush=True,
                    )

                if future_set:
                    import time
                    time.sleep(0.2)

            while True:
                try:
                    inc = progress_queue.get_nowait()
                except Exception:
                    break

                inc = int(inc)
                completed_queries += inc
                pbar.update(inc)

            if completed_queries < n_query:
                pbar.update(n_query - completed_queries)

    chunk_results = sorted(chunk_results, key=lambda x: x["chunk_id"])

    estimated_parts = [
        np.load(res["estimated_x_npy"])
        for res in chunk_results
    ]

    estimated_x = np.concatenate(estimated_parts, axis=0)

    adaptive_k_parts = [
        np.load(res["adaptive_k_npy"])
        for res in chunk_results
        if "adaptive_k_npy" in res
    ]

    adaptive_segments_parts = [
        np.load(res["adaptive_segments_npy"])
        for res in chunk_results
        if "adaptive_segments_npy" in res
    ]

    if len(adaptive_k_parts) > 0:
        adaptive_k_arr = np.concatenate(adaptive_k_parts, axis=0)
    else:
        adaptive_k_arr = np.full_like(estimated_x, fill_value=-1, dtype=np.int64)

    if len(adaptive_segments_parts) > 0:
        adaptive_segments_arr = np.concatenate(adaptive_segments_parts, axis=0)
    else:
        adaptive_segments_arr = np.full_like(estimated_x, fill_value=-1, dtype=np.int64)

    stem = f"{query_npz.stem}__vs__{lib_npy.stem}"

    estimated_x_path = out_dir / f"{stem}.estimated_n_precursor_groups.npy"
    adaptive_k_path = out_dir / f"{stem}.adaptive_k.npy"
    adaptive_segments_path = out_dir / f"{stem}.adaptive_selected_segments.npy"

    np.save(estimated_x_path, estimated_x)
    np.save(adaptive_k_path, adaptive_k_arr)
    np.save(adaptive_segments_path, adaptive_segments_arr)

    print(f"[saved] {estimated_x_path}", flush=True)
    print(f"[saved] {adaptive_k_path}", flush=True)
    print(f"[saved] {adaptive_segments_path}", flush=True)

    adaptive_summary = summarize_adaptive_k(
        adaptive_k_arr,
        selected_segments=adaptive_segments_arr,
    )

    print(f"[adaptive-k] {adaptive_summary}", flush=True)

    manifest_path = out_dir / f"{stem}.chunk_manifest.json"

    save_json(
        {
            "method": str(method),
            "query_file": str(query_npz),
            "library_file": str(lib_npy),
            "top_idx_path": str(top_idx_path),
            "top_score_path": str(top_score_path),
            "n_query": int(n_query),
            "n_workers": int(n_workers),

            "knn_topk": int(KNN_TOPK),
            "max_precursor_groups": int(MAX_PRECURSOR_GROUPS),
            "top_n_per_group": int(TOP_N_PER_GROUP),

            "compute_scaffold": bool(compute_scaffold),
            "max_mols_per_cluster": int(MAX_MOLS_PER_CLUSTER),
            "mcs_timeout": int(MCS_TIMEOUT),
            "min_num_atoms": int(MIN_NUM_ATOMS),
            "min_cluster_size": int(MIN_CLUSTER_SIZE),

            "use_structure_confidence": bool(USE_STRUCTURE_CONFIDENCE),
            "struct_conf_high": float(STRUCT_CONF_HIGH),
            "struct_conf_medium": float(STRUCT_CONF_MEDIUM),
            "struct_min_scaffold_atoms_for_conf": int(STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF),
            "struct_weight_cluster_support": float(STRUCT_WEIGHT_CLUSTER_SUPPORT),
            "struct_weight_overlap": float(STRUCT_WEIGHT_OVERLAP),
            "struct_weight_candidate_coverage": float(STRUCT_WEIGHT_CANDIDATE_COVERAGE),
            "struct_weight_exact_match": float(STRUCT_WEIGHT_EXACT_MATCH),
            "compound_level_pruning": bool(USE_STRUCTURE_CONFIDENCE and compute_scaffold),
            "compound_level_pruning_threshold": float(STRUCT_CONF_MEDIUM),

            "use_adaptive_k": bool(USE_ADAPTIVE_K),
            "adaptive_k_min": int(ADAPTIVE_K_MIN),
            "adaptive_k_max": int(ADAPTIVE_K_MAX),
            "adaptive_max_segments": int(ADAPTIVE_MAX_SEGMENTS),
            "adaptive_min_segment_size": int(ADAPTIVE_MIN_SEGMENT_SIZE),
            "adaptive_fallback_k": int(ADAPTIVE_FALLBACK_K),
            "adaptive_min_score_range": float(ADAPTIVE_MIN_SCORE_RANGE),
            "adaptive_keep_before_last_segment": bool(ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT),
            "adaptive_summary": adaptive_summary,

            "chunks": chunk_results,
        },
        manifest_path,
    )

    print(f"[saved] {manifest_path}", flush=True)

    merged_paths = {}

    if merge_chunk_csv:
        merge_specs = [
            ("neighbor_csv", out_dir / f"{stem}.neighbors.csv"),
            ("recommendation_csv", out_dir / f"{stem}.chimera_recommendations.csv"),
            ("scaffold_csv", out_dir / f"{stem}.cluster_scaffolds.csv"),
            ("confidence_csv", out_dir / f"{stem}.scaffold_confidence.csv"),
        ]

        for key, merged_path in merge_specs:
            csvs = [res[key] for res in chunk_results]
            merge_csv_files(csvs, merged_path)
            merged_paths[key] = str(merged_path)

    return {
        "estimated_x": estimated_x,
        "estimated_x_path": str(estimated_x_path),
        "adaptive_k": adaptive_k_arr,
        "adaptive_k_path": str(adaptive_k_path),
        "adaptive_selected_segments": adaptive_segments_arr,
        "adaptive_selected_segments_path": str(adaptive_segments_path),
        "adaptive_summary": adaptive_summary,
        "chunk_results": chunk_results,
        "merged_paths": merged_paths,
        "manifest_path": str(manifest_path),
    }


# =============================================================================
# 8. Evaluation and plotting
# =============================================================================

def load_chimera_true_component_count(h5_path: str | Path, limit=None) -> np.ndarray | None:
    """
    Load ground-truth component counts from a chimera HDF5 dataset.

    Parameters
    ----------
    h5_path:
        HDF5 path containing "component_count".
    limit:
        Optional maximum number of entries.

    Returns
    -------
    numpy.ndarray or None
        One-dimensional integer array or None if unavailable.
    """
    h5_path = Path(h5_path)

    if not h5_path.exists():
        print(f"[eval] WARNING: chimera HDF5 not found: {h5_path}", flush=True)
        return None

    with h5py.File(h5_path, "r") as f:
        if "component_count" not in f:
            print("[eval] WARNING: component_count not found in HDF5.", flush=True)
            return None

        n = f["component_count"].shape[0]

        if limit is not None:
            n = min(int(limit), n)

        y = f["component_count"][:n]

    return np.asarray(y, dtype=int).reshape(-1)


def evaluate_component_count_by_true_count(
        true_count: np.ndarray,
        pred_count: np.ndarray,
) -> dict[str, Any]:
    """
    Evaluate predicted component counts against ground truth.

    Parameters
    ----------
    true_count:
        Ground-truth component counts.
    pred_count:
        Predicted component counts.

    Returns
    -------
    dict
        Overall accuracy and per-class accuracy for true counts 2 and 3.
    """
    true_count = np.asarray(true_count, dtype=int).reshape(-1)
    pred_count = np.asarray(pred_count, dtype=int).reshape(-1)

    n = min(len(true_count), len(pred_count))
    true_count = true_count[:n]
    pred_count = pred_count[:n]

    out = {
        "n_eval": int(n),
        "overall_accuracy": float(np.mean(true_count == pred_count)) if n else np.nan,
    }

    for k in [2, 3]:
        mask = true_count == k
        total = int(mask.sum())

        if total == 0:
            correct = 0
            acc = np.nan
        else:
            correct = int((pred_count[mask] == k).sum())
            acc = float(correct / total)

        out[f"accuracy_true_{k}"] = acc
        out[f"correct_true_{k}"] = correct
        out[f"total_true_{k}"] = total

    return out


def plot_component_accuracy_pdf(eval_df: pd.DataFrame, out_pdf: str | Path):
    """
    Plot component-count prediction accuracy as a PDF.

    Parameters
    ----------
    eval_df:
        Evaluation dataframe.
    out_pdf:
        Output PDF path.
    """
    out_pdf = Path(out_pdf)
    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    rows = []

    for _, r in eval_df.iterrows():
        method = r["method"]

        rows.append(
            {
                "method": method,
                "true_component_count": "2 components",
                "accuracy": r.get("accuracy_true_2", np.nan),
                "correct": r.get("correct_true_2", np.nan),
                "total": r.get("total_true_2", np.nan),
            }
        )

        rows.append(
            {
                "method": method,
                "true_component_count": "3 components",
                "accuracy": r.get("accuracy_true_3", np.nan),
                "correct": r.get("correct_true_3", np.nan),
                "total": r.get("total_true_3", np.nan),
            }
        )

    plot_df = pd.DataFrame(rows)

    plt.figure(figsize=(12, 5.5))

    sns.barplot(
        data=plot_df,
        x="method",
        y="accuracy",
        hue="true_component_count",
    )

    plt.xticks(rotation=35, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Accuracy")
    plt.xlabel("Embedding method")
    plt.title("Component-count prediction accuracy by true component count")
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.legend(title="True component count", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(out_pdf, format="pdf", bbox_inches="tight")
    plt.close()

    print(f"[saved] {out_pdf}", flush=True)


# =============================================================================
# 9. Main
# =============================================================================

def main():
    """
    Run the complete MS2-KNN workflow.

    Main steps:
        1. Find matched query-library embedding pairs.
        2. Load embeddings.
        3. Run or load cached Top-K retrieval.
        4. Run multiprocessing post-KNN processing.
        5. Save outputs and optional evaluation.
    """
    warnings.filterwarnings("ignore")

    print("=" * 100)
    print("[MS2-KNN] Starting")
    print(f"QUERY_DIR         : {QUERY_DIR}")
    print(f"LIBRARY_DIR       : {LIBRARY_DIR}")
    print(f"METADATA_CSV      : {METADATA_CSV}")
    print(f"OUT_DIR           : {OUT_DIR}")
    print(f"KNN_TOPK          : {KNN_TOPK}")
    print(f"COMPUTE_SCAFFOLD  : {COMPUTE_SCAFFOLD}")
    print(f"USE_STRUCTURE_CONF: {USE_STRUCTURE_CONFIDENCE}")
    print("=" * 100)

    pairs = find_query_library_pairs(
        QUERY_DIR,
        LIBRARY_DIR,
        method_whitelist=METHOD_WHITELIST,
    )

    print(f"[pairs] found {len(pairs)} matched query-library pairs")

    if len(pairs) == 0:
        raise RuntimeError("No matched query-library pairs found.")

    all_eval_rows = []

    true_component_count = load_chimera_true_component_count(
        CHIMERA_HDF5_PATH,
        limit=LIMIT_QUERIES,
    )

    for method, query_npz, lib_npy in pairs:
        print("\n" + "=" * 100)
        print(f"[method] {method}")
        print(f"[query]  {query_npz}")
        print(f"[lib]    {lib_npy}")

        method_out_dir = OUT_DIR / method
        method_out_dir.mkdir(parents=True, exist_ok=True)

        query = load_query_embedding_file(query_npz)
        lib = ck.load_embedding_file(lib_npy, metadata_csv=METADATA_CSV)

        query_embs = ensure_2d_embedding(query["embeddings"], "query_embs")
        lib_embs = ensure_2d_embedding(lib["embeddings"], "lib_embs")

        if LIMIT_QUERIES is not None:
            query_embs = query_embs[: int(LIMIT_QUERIES)]

        print(f"[shape] query={query_embs.shape}, lib={lib_embs.shape}", flush=True)

        out_prefix = method_out_dir / f"{query_npz.stem}__vs__{lib_npy.stem}"

        _, _, top_idx_path, top_score_path = run_topk_retrieval(
            query_embs=query_embs,
            lib_embs=lib_embs,
            out_prefix=out_prefix,
            k=KNN_TOPK,
            use_cuda=USE_CUDA_KNN,
        )

        result = process_queries_after_topk_parallel_mmap(
            query_npz=query_npz,
            lib_npy=lib_npy,
            method=method,
            top_idx_path=top_idx_path,
            top_score_path=top_score_path,
            out_dir=method_out_dir,
            n_workers=N_POSTPROCESS_WORKERS,
            compute_scaffold=COMPUTE_SCAFFOLD,
            merge_chunk_csv=MERGE_CHUNK_CSV,
        )

        estimated_x = result["estimated_x"]

        if true_component_count is not None:
            eval_res = evaluate_component_count_by_true_count(
                true_count=true_component_count,
                pred_count=estimated_x,
            )

            eval_row = {
                "method": method,
                "query_file": str(query_npz),
                "library_file": str(lib_npy),
                **eval_res,
            }

            all_eval_rows.append(eval_row)

            pd.DataFrame([eval_row]).to_csv(
                method_out_dir / f"{method}.component_count_eval.csv",
                index=False,
                encoding="utf-8-sig",
            )

            pred_df = pd.DataFrame(
                {
                    "query_index": np.arange(len(estimated_x), dtype=int),
                    "predicted_component_count": estimated_x,
                    "true_component_count": true_component_count[: len(estimated_x)],
                }
            )

            pred_df["adaptive_k"] = result["adaptive_k"][: len(pred_df)]
            pred_df["adaptive_selected_segments"] = result["adaptive_selected_segments"][: len(pred_df)]

            pred_df.to_csv(
                method_out_dir / f"{method}.component_count_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(f"[eval] {method}: {eval_res}", flush=True)

    if all_eval_rows:
        eval_df = pd.DataFrame(all_eval_rows)

        eval_csv = OUT_DIR / "combined_component_count_eval.csv"
        eval_df.to_csv(eval_csv, index=False, encoding="utf-8-sig")

        plot_component_accuracy_pdf(
            eval_df,
            OUT_DIR / "component_count_accuracy_by_true_count.pdf",
        )

        print(f"[saved] {eval_csv}", flush=True)

    config = {
        "KNN_TOPK": KNN_TOPK,
        "MAX_PRECURSOR_GROUPS": MAX_PRECURSOR_GROUPS,
        "TOP_N_PER_GROUP": TOP_N_PER_GROUP,

        "USE_CUDA_KNN": USE_CUDA_KNN,
        "CUDA_QUERY_BATCH_SIZE": CUDA_QUERY_BATCH_SIZE,
        "CUDA_USE_FLOAT16": CUDA_USE_FLOAT16,
        "CPU_BATCH_SIZE": CPU_BATCH_SIZE,

        "USE_ADAPTIVE_K": USE_ADAPTIVE_K,
        "ADAPTIVE_K_MIN": ADAPTIVE_K_MIN,
        "ADAPTIVE_K_MAX": ADAPTIVE_K_MAX,
        "ADAPTIVE_MAX_SEGMENTS": ADAPTIVE_MAX_SEGMENTS,
        "ADAPTIVE_MIN_SEGMENT_SIZE": ADAPTIVE_MIN_SEGMENT_SIZE,
        "ADAPTIVE_FALLBACK_K": ADAPTIVE_FALLBACK_K,
        "ADAPTIVE_MIN_SCORE_RANGE": ADAPTIVE_MIN_SCORE_RANGE,
        "ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT": ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT,
        "SAVE_ADAPTIVE_K_DIAGNOSTICS": SAVE_ADAPTIVE_K_DIAGNOSTICS,

        "COMPUTE_SCAFFOLD": COMPUTE_SCAFFOLD,
        "MAX_MOLS_PER_CLUSTER": MAX_MOLS_PER_CLUSTER,
        "MCS_TIMEOUT": MCS_TIMEOUT,
        "MIN_NUM_ATOMS": MIN_NUM_ATOMS,
        "MIN_CLUSTER_SIZE": MIN_CLUSTER_SIZE,

        "USE_STRUCTURE_CONFIDENCE": USE_STRUCTURE_CONFIDENCE,
        "STRUCT_CONF_HIGH": STRUCT_CONF_HIGH,
        "STRUCT_CONF_MEDIUM": STRUCT_CONF_MEDIUM,
        "STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF": STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF,
        "STRUCT_WEIGHT_CLUSTER_SUPPORT": STRUCT_WEIGHT_CLUSTER_SUPPORT,
        "STRUCT_WEIGHT_OVERLAP": STRUCT_WEIGHT_OVERLAP,
        "STRUCT_WEIGHT_CANDIDATE_COVERAGE": STRUCT_WEIGHT_CANDIDATE_COVERAGE,
        "STRUCT_WEIGHT_EXACT_MATCH": STRUCT_WEIGHT_EXACT_MATCH,

        "LIMIT_QUERIES": LIMIT_QUERIES,
        "METHOD_WHITELIST": METHOD_WHITELIST,
        "N_POSTPROCESS_WORKERS": N_POSTPROCESS_WORKERS,
        "MERGE_CHUNK_CSV": MERGE_CHUNK_CSV,
    }

    save_json(config, OUT_DIR / "run_config.json")

    print("=" * 100)
    print("[DONE]")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
