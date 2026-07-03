# -*- coding: utf-8 -*-

"""
Reproduce Chimera-KNN workflow without modifying chimera_knn_batch.py.

Workflow:
1. Load chimera query embeddings and library embeddings.
2. Compute TopK cosine retrieval, optionally using CUDA.
3. For each query:
   - Use TopK precursor m/z values.
   - Estimate component/group count x by GMM.
   - Cluster TopK candidates by embedding vectors into x clusters.
   - Select Top3 candidates from each cluster by similarity.
   - Optionally compute scaffold/confidence tables.
4. Save all results.
5. Optionally evaluate predicted component count against chimera HDF5 ground truth.

This script does NOT modify chimera_knn_batch.py.
"""

from __future__ import annotations
from adaptive_k_utils import (
    estimate_adaptive_k_piecewise_bic_general,
    compact_adaptive_info_for_row,
    summarize_adaptive_k,
)
import os
import json
import math
import warnings
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances
from rdkit import Chem

import matplotlib.pyplot as plt
import seaborn as sns
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed

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

# -------------------------------------------------------------------------
# Adaptive K parameters
# -------------------------------------------------------------------------
# 是否启用自适应 K。
USE_ADAPTIVE_K = True

# CUDA KNN 仍然先取 Top100，adaptive K 是从 Top100 里再截断。
# defalut 100
KNN_TOPK = 100

# adaptive_k 的上下限。
# 注意：这是最终保留候选数，不是分段数。
# defalut 12
ADAPTIVE_K_MIN = 12
# defalut 100
ADAPTIVE_K_MAX = 100

# BIC 最多允许 score-rank 曲线分成几段。
# 1 段 = 不切；5 段 = 最多 4 个断点。
ADAPTIVE_MAX_SEGMENTS = 4

# 分段线性拟合时，每一段至少多少个点。
# Top100 + max 5 segments 时，8 比较稳。
# defalut 8
ADAPTIVE_MIN_SEGMENT_SIZE = 8

# BIC 选择一段模型时，说明没有可靠断点，回退到这个 K。
# defalut 100
ADAPTIVE_FALLBACK_K = 100

# 如果 score 总下降幅度太小，认为没有明显信号/背景分界。
ADAPTIVE_MIN_SCORE_RANGE = 0.03

# 统一规则：
# 选中 m>=2 段时，最后一段视为背景尾部，保留最后一个断点之前的全部。
ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT = True

# 是否把 adaptive-k 诊断信息写进 neighbor CSV。
SAVE_ADAPTIVE_K_DIAGNOSTICS = True

# -------------------------------------------------------------------------
# Main workflow parameters
# -------------------------------------------------------------------------
# Post-KNN 多进程数量。
# None 表示自动使用 cpu_count - 1。
N_POSTPROCESS_WORKERS = 10

# 每个 worker 是否单独保存 chunk 文件。
SAVE_CHUNK_FILES = True

# 是否在最后合并所有 chunk CSV。
# 全量 20w 时，all_neighbors.csv 会很大，合并 CSV 会慢。
MERGE_CHUNK_CSV = True

# 每条 chimera query 取相似度 Top100 候选。
# defalut 100
KNN_TOPK = 100

# GMM 最多允许分成多少个 precursor m/z group。
# 如果你的 simulated chimera 只有 AB / ABC，可以设 3。
# 如果想保留原工作流弹性，可以设 5。
# defalut 4
MAX_PRECURSOR_GROUPS = 20

# 每个 embedding cluster 中推荐 Top3 分子。
#defalut 1
TOP_N_PER_GROUP = 3

# KNN batch 参数。
USE_CUDA_KNN = True
CUDA_QUERY_BATCH_SIZE = 128
CUDA_DB_BLOCK_SIZE = 50000
CUDA_USE_FLOAT16 = True
POSTPROCESS_PROGRESS_EVERY = 500

CPU_BATCH_SIZE = 128

# 是否计算 scaffold / 结构置信度。
# 全量 20w 时这个会很重，建议先 False 跑通主结果。
COMPUTE_SCAFFOLD = False

# 如果 COMPUTE_SCAFFOLD=True，每个 cluster 用多少分子做 MCS/scaffold。
MAX_MOLS_PER_CLUSTER = 100
MCS_TIMEOUT = 20
MIN_NUM_ATOMS = 5
MIN_CLUSTER_SIZE = 1

# 全量设 None；快速测试可以设 1000 / 5000。
LIMIT_QUERIES = None

# 如果只跑某些方法，填列表；None 表示自动匹配全部。
METHOD_WHITELIST = None
# METHOD_WHITELIST = ["dreams"]
_WORKER_LIB = None
_WORKER_LIB_EMBS = None
_WORKER_TOP_INDICES = None
_WORKER_TOP_SCORES = None
_WORKER_QUERY_FILE = None
_WORKER_LIB_FILE = None
_WORKER_METHOD = None
_WORKER_PROGRESS_QUEUE = None

# -------------------------------------------------------------------------
# Structure confidence parameters
# -------------------------------------------------------------------------
USE_STRUCTURE_CONFIDENCE = True

# MCS / scaffold overlap 相关阈值。
STRUCT_MCS_THRESHOLD = 0.50

# 置信度分级阈值。
STRUCT_CONF_HIGH = 0.70
STRUCT_CONF_MEDIUM = 0.40

# 如果 scaffold 原子数太少，结构置信度不可靠。
STRUCT_MIN_SCAFFOLD_ATOMS_FOR_CONF = 5

# cluster support 权重。
STRUCT_WEIGHT_CLUSTER_SUPPORT = 0.30
STRUCT_WEIGHT_OVERLAP = 0.35
STRUCT_WEIGHT_CANDIDATE_COVERAGE = 0.25
STRUCT_WEIGHT_EXACT_MATCH = 0.10


def _safe_float(x, default=np.nan):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def _safe_int(x, default=0):
    try:
        if pd.isna(x):
            return default
        return int(x)
    except Exception:
        return default


def _normalize_smiles_like(x: Any) -> str:
    if x is None:
        return ""
    s = str(x)
    if s.lower() in {"nan", "none", "null"}:
        return ""
    return s.strip()


def _structure_confidence_level(conf: float) -> str:
    if not np.isfinite(conf):
        return "Low"
    if conf >= STRUCT_CONF_HIGH:
        return "High"
    if conf >= STRUCT_CONF_MEDIUM:
        return "Medium"
    return "Low"


def add_structure_confidence_to_recommendations(
        recommendation_df,
        query_smiles=None,
        compute_scaffold=True,
        **kwargs,
):
    """
    Add structure/scaffold confidence information to recommendation_df.

    This version is robust for multiprocessing:
      - imports RDKit inside worker if needed;
      - auto-detects candidate SMILES column;
      - does not assume candidate_smiles variable already exists;
      - returns original dataframe safely if SMILES/RDKit is unavailable.

    Expected candidate SMILES column can be one of:
      smiles, candidate_smiles, library_smiles, reference_smiles, lib_smiles
    """

    if recommendation_df is None:
        return recommendation_df

    if len(recommendation_df) == 0:
        return recommendation_df

    recommendation_df = recommendation_df.copy()

    if not compute_scaffold:
        return recommendation_df

    # ------------------------------------------------------------
    # 1. Import RDKit inside function for Windows multiprocessing
    # ------------------------------------------------------------
    try:
        from rdkit import Chem
        from rdkit.Chem.Scaffolds import MurckoScaffold
        from rdkit import DataStructs
        from rdkit.Chem import AllChem
    except Exception as e:
        recommendation_df["structure_confidence"] = "rdkit_unavailable"
        recommendation_df["structure_error"] = str(e)
        return recommendation_df

    # ------------------------------------------------------------
    # 2. Find candidate SMILES column
    # ------------------------------------------------------------
    candidate_smiles_col_candidates = [
        "candidate_smiles",
        "library_smiles",
        "reference_smiles",
        "lib_smiles",
        "smiles",
        "SMILES",
        "canonical_smiles",
        "CanonicalSMILES",
    ]

    candidate_smiles_col = None

    for col in candidate_smiles_col_candidates:
        if col in recommendation_df.columns:
            candidate_smiles_col = col
            break

    if candidate_smiles_col is None:
        recommendation_df["structure_confidence"] = "no_candidate_smiles_column"
        recommendation_df["candidate_scaffold"] = ""
        return recommendation_df

    # ------------------------------------------------------------
    # 3. Resolve query SMILES
    # ------------------------------------------------------------
    query_mol = None
    query_scaffold_smiles = ""

    if query_smiles is None:
        query_smiles_col_candidates = [
            "query_smiles",
            "query_SMILES",
            "mixture_smiles",
            "true_smiles",
            "target_smiles",
        ]

        for col in query_smiles_col_candidates:
            if col in recommendation_df.columns:
                vals = recommendation_df[col].dropna().astype(str).values
                vals = [x for x in vals if x.strip() and x.lower() != "nan"]
                if len(vals) > 0:
                    query_smiles = vals[0]
                    break

    if query_smiles is not None:
        query_smiles = str(query_smiles).strip()

        if query_smiles and query_smiles.lower() != "nan":
            try:
                query_mol = Chem.MolFromSmiles(query_smiles)
                if query_mol is not None:
                    query_scaffold = MurckoScaffold.GetScaffoldForMol(query_mol)
                    query_scaffold_smiles = Chem.MolToSmiles(query_scaffold)
            except Exception:
                query_mol = None
                query_scaffold_smiles = ""

    # ------------------------------------------------------------
    # 4. Process candidate rows
    # ------------------------------------------------------------
    candidate_scaffolds = []
    candidate_valid_smiles = []
    scaffold_match = []
    tanimoto_to_query = []
    structure_confidence = []

    query_fp = None

    if query_mol is not None:
        try:
            query_fp = AllChem.GetMorganFingerprintAsBitVect(query_mol, 2, nBits=2048)
        except Exception:
            query_fp = None

    for _, row in recommendation_df.iterrows():

        candidate_smiles = row.get(candidate_smiles_col, "")

        if candidate_smiles is None:
            candidate_smiles = ""

        candidate_smiles = str(candidate_smiles).strip()

        if candidate_smiles == "" or candidate_smiles.lower() == "nan":
            candidate_valid_smiles.append(False)
            candidate_scaffolds.append("")
            scaffold_match.append(False)
            tanimoto_to_query.append(np.nan)
            structure_confidence.append("missing_candidate_smiles")
            continue

        try:
            mol = Chem.MolFromSmiles(candidate_smiles)

            if mol is None:
                candidate_valid_smiles.append(False)
                candidate_scaffolds.append("")
                scaffold_match.append(False)
                tanimoto_to_query.append(np.nan)
                structure_confidence.append("invalid_candidate_smiles")
                continue

            candidate_valid_smiles.append(True)

            cand_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            cand_scaffold_smiles = Chem.MolToSmiles(cand_scaffold)
            candidate_scaffolds.append(cand_scaffold_smiles)

            if query_scaffold_smiles:
                is_scaffold_match = cand_scaffold_smiles == query_scaffold_smiles
            else:
                is_scaffold_match = False

            scaffold_match.append(bool(is_scaffold_match))

            if query_fp is not None:
                try:
                    cand_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    tanimoto = float(DataStructs.TanimotoSimilarity(query_fp, cand_fp))
                except Exception:
                    tanimoto = np.nan
            else:
                tanimoto = np.nan

            tanimoto_to_query.append(tanimoto)

            if is_scaffold_match:
                conf = "same_scaffold"
            elif np.isfinite(tanimoto) and tanimoto >= 0.7:
                conf = "high_similarity"
            elif np.isfinite(tanimoto) and tanimoto >= 0.4:
                conf = "medium_similarity"
            elif np.isfinite(tanimoto):
                conf = "low_similarity"
            else:
                conf = "candidate_valid_no_query"

            structure_confidence.append(conf)

        except Exception as e:
            candidate_valid_smiles.append(False)
            candidate_scaffolds.append("")
            scaffold_match.append(False)
            tanimoto_to_query.append(np.nan)
            structure_confidence.append("structure_error")

    recommendation_df["candidate_smiles_col"] = candidate_smiles_col
    recommendation_df["candidate_valid_smiles"] = candidate_valid_smiles
    recommendation_df["candidate_scaffold"] = candidate_scaffolds
    recommendation_df["query_scaffold"] = query_scaffold_smiles
    recommendation_df["same_scaffold_as_query"] = scaffold_match
    recommendation_df["tanimoto_to_query"] = tanimoto_to_query
    recommendation_df["structure_confidence"] = structure_confidence

    return recommendation_df


# =============================================================================
# 2. Basic helpers
# =============================================================================
def split_query_ranges(n_query: int, n_workers: int):
    """
    将 n_query 平均切成 n_workers 个连续区间。

    返回:
        [(start0, end0), (start1, end1), ...]
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
    合并多个 chunk CSV，不一次性读入内存。
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


def load_query_embedding_file(path):
    """
    Load chimera query embedding file.

    Query embeddings can be:
        - .npy: raw embedding matrix, shape=(N, D)
        - .npz: contains key "embeddings"

    This function is intentionally separate from ck.load_embedding_file(),
    because ck.load_embedding_file() treats .npy as library embeddings
    and requires metadata_csv.
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


def l2_normalize_np(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    return x / norms


def ensure_2d_embedding(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x)

    if x.ndim == 3 and 1 in x.shape:
        x = np.squeeze(x)

    if x.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape={x.shape}")

    return x


def load_chimera_true_component_count(h5_path: str | Path, limit=None) -> np.ndarray | None:
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


def save_json(obj, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
    每个 worker 启动时执行一次。

    大数组不由主进程传入：
    - library embedding: worker 自己 mmap
    - top_indices/top_scores: worker 自己 mmap
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
    worker 处理一个 query 区间。

    args:
        chunk_id, q_start, q_end, out_dir, compute_scaffold
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
                    adaptive_segments_arr[local_i] = int(neighbor_df["adaptive_selected_segments"].iloc[0])
                elif "adaptive_selected_model" in neighbor_df.columns:
                    adaptive_segments_arr[local_i] = int(neighbor_df["adaptive_selected_model"].iloc[0])
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

            recommendation_df = ck.summarize_top_recommendations_from_neighbor_df(
                neighbor_df=neighbor_df,
                scaffold_df=scaffold_df,
                top_n_per_group=TOP_N_PER_GROUP,
            )

            if recommendation_df is not None and len(recommendation_df) > 0:
                if USE_STRUCTURE_CONFIDENCE:
                    recommendation_df = add_structure_confidence_to_recommendations(
                        recommendation_df=recommendation_df,
                        neighbor_df=neighbor_df,
                        scaffold_df=scaffold_df,
                    )

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
        # 把最后不足 POSTPROCESS_PROGRESS_EVERY 的部分也汇报给主进程
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
    多进程后处理。

    重要：
    - 主进程不传 lib
    - 主进程不传 top_indices/top_scores 数组
    - worker 自己 mmap 读取：
        lib_npy
        top_idx_path
        top_score_path
    """
    query_npz = Path(query_npz)
    lib_npy = Path(lib_npy)
    top_idx_path = Path(top_idx_path)
    top_score_path = Path(top_score_path)
    out_dir = Path(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    # 只读 shape，不把数组读入内存
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
                # 1. 先尽量消费 worker 汇报的 query 进度
                while True:
                    try:
                        inc = progress_queue.get_nowait()
                    except Exception:
                        break

                    inc = int(inc)
                    completed_queries += inc
                    pbar.update(inc)

                # 2. 检查哪些 chunk 完成了
                done_futures = [f for f in list(future_set) if f.done()]

                for fut in done_futures:
                    future_set.remove(fut)

                    # 如果 worker 抛错，这里会直接抛出来
                    res = fut.result()
                    chunk_results.append(res)

                    print(
                        f"\n[chunk done] chunk={res['chunk_id']:04d}, "
                        f"q={res['q_start']}:{res['q_end']}, "
                        f"pid={res['pid']}",
                        flush=True,
                    )

                # 3. 避免 while 空转占 CPU
                if future_set:
                    import time
                    time.sleep(0.2)

            # 4. futures 都结束后，再清空队列里剩余进度
            while True:
                try:
                    inc = progress_queue.get_nowait()
                except Exception:
                    break

                inc = int(inc)
                completed_queries += inc
                pbar.update(inc)

            # 5. 防止极少数情况下进度条少一点点
            if completed_queries < n_query:
                pbar.update(n_query - completed_queries)

    chunk_results = sorted(chunk_results, key=lambda x: x["chunk_id"])

    # 合并 estimated_x
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

    # 保存 manifest
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
            "compute_scaffold": bool(compute_scaffold),
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
# 3. CUDA / CPU TopK retrieval
# =============================================================================

def knn_cosine_torch_blockwise(
        query_embs,
        lib_embs,
        k: int = 100,
        query_batch_size: int = 128,
        db_block_size: int = 50000,
        use_float16: bool = True,
        device: str | None = None,
):
    """
    Blockwise cosine TopK retrieval using PyTorch/CUDA.

    This avoids materializing the full query x library matrix.
    """
    import torch

    query_embs = ensure_2d_embedding(np.asarray(query_embs, dtype=np.float32), "query_embs")
    lib_embs = ensure_2d_embedding(np.asarray(lib_embs, dtype=np.float32), "lib_embs")

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

    dtype = torch.float16 if device.type == "cuda" and use_float16 else torch.float32

    print(
        f"[KNN-CUDA] query={query_embs.shape}, lib={lib_embs.shape}, "
        f"k={k}, q_batch={query_batch_size}, db_block={db_block_size}, "
        f"device={device}, dtype={dtype}",
        flush=True,
    )

    query_embs = l2_normalize_np(query_embs)
    lib_embs = l2_normalize_np(lib_embs)

    all_indices = np.empty((n_query, k), dtype=np.int64)
    all_scores = np.empty((n_query, k), dtype=np.float32)

    for q_start in tqdm(
            range(0, n_query, int(query_batch_size)),
            desc="[KNN-CUDA] query batches",
            unit="batch",
    ):
        q_end = min(q_start + int(query_batch_size), n_query)
        q_np = query_embs[q_start:q_end]
        batch_n = q_end - q_start

        q = torch.as_tensor(q_np, device=device, dtype=dtype)

        cur_scores = torch.full(
            (batch_n, k),
            -float("inf"),
            device=device,
            dtype=torch.float32,
        )

        cur_indices = torch.full(
            (batch_n, k),
            -1,
            device=device,
            dtype=torch.long,
        )

        for db_start in range(0, n_lib, int(db_block_size)):
            db_end = min(db_start + int(db_block_size), n_lib)
            db_np = lib_embs[db_start:db_end]

            db = torch.as_tensor(db_np, device=device, dtype=dtype)

            with torch.inference_mode():
                sim = q @ db.T
                sim = sim.float()

                local_k = min(k, sim.shape[1])

                block_scores, block_local_idx = torch.topk(
                    sim,
                    k=local_k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )

                block_global_idx = block_local_idx + int(db_start)

                merged_scores = torch.cat([cur_scores, block_scores], dim=1)
                merged_indices = torch.cat([cur_indices, block_global_idx], dim=1)

                cur_scores, order = torch.topk(
                    merged_scores,
                    k=k,
                    dim=1,
                    largest=True,
                    sorted=True,
                )

                cur_indices = torch.gather(merged_indices, 1, order)

            del db, sim, block_scores, block_local_idx, block_global_idx
            del merged_scores, merged_indices, order

        all_scores[q_start:q_end] = cur_scores.detach().cpu().numpy().astype(np.float32)
        all_indices[q_start:q_end] = cur_indices.detach().cpu().numpy().astype(np.int64)

        del q, cur_scores, cur_indices

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
    CPU fallback. Calls original ck.knn_cosine_many.
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
    Run or load cached TopK retrieval.
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

    # 重新 mmap 打开，避免后面主进程持有大数组副本
    indices_mmap = np.load(idx_path, mmap_mode="r")
    scores_mmap = np.load(score_path, mmap_mode="r")

    return indices_mmap, scores_mmap, idx_path, score_path


# =============================================================================
# 4. GMM component estimate + embedding clustering
# =============================================================================

def cluster_topk_by_embedding(
        topk_embeddings: np.ndarray,
        n_clusters: int,
) -> np.ndarray:
    """
    Cluster TopK candidates by embedding vectors into n_clusters.
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
    Build neighbor dataframe using the requested workflow:

    1. TopK candidate indices/scores are already given.
    2. Use TopK precursor m/z to estimate x by GMM.
    3. Cluster TopK embeddings into x clusters.
    4. Return neighbor table.
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

    # GMM estimates x only.
    _, estimated_x = ck.estimate_precursor_groups_gmm(
        top_precursor_mz,
        max_groups=max_precursor_groups,
    )

    estimated_x = int(estimated_x)
    estimated_x = max(1, min(estimated_x, len(neighbor_indices)))

    # Embedding clustering gives final cluster labels.
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
# 5. Scaffold / recommendation processing
# =============================================================================

def empty_scaffold_df_for_query(query_index, query_npz, lib_npy, method):
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
    Compute scaffold table for one query using original build_cluster_scaffold_df.
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


def process_queries_after_topk(
        *,
        query_npz: Path,
        lib_npy: Path,
        method: str,
        query_embs: np.ndarray,
        lib_embs: np.ndarray,
        lib: dict[str, Any],
        top_indices: np.ndarray,
        top_scores: np.ndarray,
        out_dir: Path,
):
    """
    Build neighbors, recommendations, scaffolds, and confidence tables.
    """
    n_query = top_indices.shape[0]

    all_neighbors = []
    all_recommendations = []
    all_scaffolds = []

    estimated_x = np.empty(n_query, dtype=np.int64)

    for q_idx in tqdm(
            range(n_query),
            desc=f"[{method}] GMM-x + embedding-cluster + recommendation",
            unit="query",
    ):
        neighbor_df = build_neighbor_df_gmm_x_embedding_cluster(
            query_index=q_idx,
            neighbor_indices=neighbor_indices,
            neighbor_scores=neighbor_scores,
            lib=_WORKER_LIB,
            lib_embs=_WORKER_LIB_EMBS,
            max_precursor_groups=MAX_PRECURSOR_GROUPS,
            use_adaptive_k=USE_ADAPTIVE_K,
        )

        if len(neighbor_df) > 0:
            neighbor_df["query_file"] = str(query_npz)
            neighbor_df["library_file"] = str(lib_npy)
            neighbor_df["method"] = method

            estimated_x[q_idx] = int(neighbor_df["estimated_n_precursor_groups"].iloc[0])
        else:
            estimated_x[q_idx] = 0

        if COMPUTE_SCAFFOLD:
            scaffold_df = compute_scaffold_for_neighbor_df(
                neighbor_df,
                query_index=q_idx,
                query_npz=query_npz,
                lib_npy=lib_npy,
                method=method,
            )
        else:
            scaffold_df = empty_scaffold_df_for_query(
                q_idx,
                query_npz,
                lib_npy,
                method,
            )

        recommendation_df = ck.summarize_top_recommendations_from_neighbor_df(
            neighbor_df=neighbor_df,
            scaffold_df=scaffold_df,
            top_n_per_group=TOP_N_PER_GROUP,
        )
        if recommendation_df is not None and len(recommendation_df) > 0:
            if USE_STRUCTURE_CONFIDENCE:
                recommendation_df = add_structure_confidence_to_recommendations(
                    recommendation_df=recommendation_df,
                    neighbor_df=neighbor_df,
                    scaffold_df=scaffold_df,
                )

        if recommendation_df is not None and len(recommendation_df) > 0:
            recommendation_df["query_file"] = str(query_npz)
            recommendation_df["library_file"] = str(lib_npy)
            recommendation_df["method"] = method

        all_neighbors.append(neighbor_df)
        all_scaffolds.append(scaffold_df)
        all_recommendations.append(recommendation_df)

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

    stem = f"{query_npz.stem}__vs__{lib_npy.stem}"

    neighbor_csv = out_dir / f"{stem}.neighbors.csv"
    recommendation_csv = out_dir / f"{stem}.chimera_recommendations.csv"
    scaffold_csv = out_dir / f"{stem}.cluster_scaffolds.csv"
    confidence_csv = out_dir / f"{stem}.scaffold_confidence.csv"
    estimated_x_npy = out_dir / f"{stem}.estimated_n_precursor_groups.npy"

    neighbor_all_df.to_csv(neighbor_csv, index=False, encoding="utf-8-sig")
    recommendation_all_df.to_csv(recommendation_csv, index=False, encoding="utf-8-sig")
    scaffold_all_df.to_csv(scaffold_csv, index=False, encoding="utf-8-sig")
    confidence_df.to_csv(confidence_csv, index=False, encoding="utf-8-sig")
    np.save(estimated_x_npy, estimated_x)

    print(f"[saved] {neighbor_csv}", flush=True)
    print(f"[saved] {recommendation_csv}", flush=True)
    print(f"[saved] {scaffold_csv}", flush=True)
    print(f"[saved] {confidence_csv}", flush=True)
    print(f"[saved] {estimated_x_npy}", flush=True)

    return {
        "neighbors": neighbor_all_df,
        "recommendations": recommendation_all_df,
        "scaffolds": scaffold_all_df,
        "confidence": confidence_df,
        "estimated_x": estimated_x,
    }


# =============================================================================
# 6. Evaluation and PDF plots
# =============================================================================

def evaluate_component_count_by_true_count(
        true_count: np.ndarray,
        pred_count: np.ndarray,
) -> dict[str, Any]:
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
    Plot all-method component-count accuracy for true 2 and true 3.
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
# 7. Method matching
# =============================================================================

def find_query_library_pairs(
        query_dir: Path,
        library_dir: Path,
        method_whitelist=None,
):
    query_files = sorted(query_dir.glob("*.npy"))
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


def knn_cosine_torch_library_on_gpu(
        query_embs,
        lib_embs,
        k=100,
        query_batch_size=256,
        use_float16=True,
        device=None,
):
    """
    Fast cosine TopK retrieval with full library resident on GPU.

    Best when the full library embedding matrix fits in GPU memory.

    Workflow:
        1. Normalize query and library on CPU.
        2. Move full library embedding to GPU once.
        3. For each query batch:
            sim = query_batch @ library_gpu.T
            topk(sim)
        4. Return top-k indices and scores.

    This avoids repeated CPU->GPU transfer of library blocks.
    """
    import torch
    import time

    query_embs = ensure_2d_embedding(np.asarray(query_embs, dtype=np.float32), "query_embs")

    # 不要强制 np.asarray(lib_embs, dtype=np.float32) 太早复制 memmap。
    # 这里分情况处理。
    if not isinstance(lib_embs, np.ndarray):
        lib_embs = np.asarray(lib_embs)

    lib_embs = ensure_2d_embedding(lib_embs, "lib_embs")

    if query_embs.shape[1] != lib_embs.shape[1]:
        raise ValueError(
            f"Embedding dimension mismatch: query={query_embs.shape}, lib={lib_embs.shape}"
        )

    n_query = int(query_embs.shape[0])
    n_lib = int(lib_embs.shape[0])
    dim = int(query_embs.shape[1])
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

    # ------------------------------------------------------------------
    # Normalize on CPU.
    # ------------------------------------------------------------------
    t0 = time.time()
    query_norm = l2_normalize_np(query_embs).astype(np.float32, copy=False)

    # 对 memmap library，这一步会读完整库一次，是必要的。
    # 但只做一次。
    lib_norm = l2_normalize_np(np.asarray(lib_embs, dtype=np.float32)).astype(
        np.float32,
        copy=False,
    )

    print(f"[KNN-GPU] CPU normalization done in {time.time() - t0:.2f}s", flush=True)

    # ------------------------------------------------------------------
    # Move full library to GPU once.
    # ------------------------------------------------------------------
    t1 = time.time()

    lib_gpu = torch.as_tensor(lib_norm, device=device, dtype=dtype)

    # 保证是 contiguous，matmul 更稳。
    lib_gpu = lib_gpu.contiguous()

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(
        f"[KNN-GPU] library moved to GPU once in {time.time() - t1:.2f}s; "
        f"lib_gpu.shape={tuple(lib_gpu.shape)}",
        flush=True,
    )

    all_indices = np.empty((n_query, k), dtype=np.int64)
    all_scores = np.empty((n_query, k), dtype=np.float32)

    # ------------------------------------------------------------------
    # Query batches.
    # ------------------------------------------------------------------
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

            # topk 在 fp16 sim 上可能返回 fp16 score；
            # 保存前转 float32。
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


# =============================================================================
# 8. Main
# =============================================================================

def main():
    warnings.filterwarnings("ignore")

    print("=" * 100)
    print("[Chimera-KNN reproduce] Starting")
    print(f"QUERY_DIR      : {QUERY_DIR}")
    print(f"LIBRARY_DIR    : {LIBRARY_DIR}")
    print(f"METADATA_CSV   : {METADATA_CSV}")
    print(f"OUT_DIR        : {OUT_DIR}")
    print(f"KNN_TOPK       : {KNN_TOPK}")
    print(f"COMPUTE_SCAFFOLD: {COMPUTE_SCAFFOLD}")
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
    merged_neighbors = []
    merged_recommendations = []
    merged_scaffolds = []
    merged_confidences = []

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

        top_indices, top_scores, top_idx_path, top_score_path = run_topk_retrieval(
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

            if "adaptive_k" in result:
                pred_df["adaptive_k"] = result["adaptive_k"][: len(pred_df)]

            if "adaptive_selected_segments" in result:
                pred_df["adaptive_selected_segments"] = result["adaptive_selected_segments"][: len(pred_df)]

            pred_df.to_csv(
                method_out_dir / f"{method}.component_count_predictions.csv",
                index=False,
                encoding="utf-8-sig",
            )

            print(f"[eval] {method}: {eval_res}", flush=True)

    print("\n" + "=" * 100)
    print("[merge] saving merged outputs")

    if merged_neighbors:
        pd.concat(merged_neighbors, ignore_index=True).to_csv(
            OUT_DIR / "all_neighbors.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if merged_recommendations:
        pd.concat(merged_recommendations, ignore_index=True).to_csv(
            OUT_DIR / "all_chimera_recommendations.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if merged_scaffolds:
        pd.concat(merged_scaffolds, ignore_index=True).to_csv(
            OUT_DIR / "all_cluster_scaffolds.csv",
            index=False,
            encoding="utf-8-sig",
        )

    if merged_confidences:
        pd.concat(merged_confidences, ignore_index=True).to_csv(
            OUT_DIR / "all_scaffold_confidence.csv",
            index=False,
            encoding="utf-8-sig",
        )

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
        "USE_ADAPTIVE_K": USE_ADAPTIVE_K,
        "ADAPTIVE_K_MIN": ADAPTIVE_K_MIN,
        "ADAPTIVE_K_MAX": ADAPTIVE_K_MAX,
        "ADAPTIVE_MAX_SEGMENTS": ADAPTIVE_MAX_SEGMENTS,
        "ADAPTIVE_MIN_SEGMENT_SIZE": ADAPTIVE_MIN_SEGMENT_SIZE,
        "ADAPTIVE_FALLBACK_K": ADAPTIVE_FALLBACK_K,
        "ADAPTIVE_MIN_SCORE_RANGE": ADAPTIVE_MIN_SCORE_RANGE,
        "ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT": ADAPTIVE_KEEP_BEFORE_LAST_SEGMENT,
        "SAVE_ADAPTIVE_K_DIAGNOSTICS": SAVE_ADAPTIVE_K_DIAGNOSTICS,
    }

    save_json(config, OUT_DIR / "run_config.json")

    print("=" * 100)
    print("[DONE]")
    print(f"Results saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
