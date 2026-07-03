import os
import time
import shutil
import threading
import queue as queue_module
import multiprocessing as mp
from pathlib import Path
from types import SimpleNamespace
from dataclasses import asdict, is_dataclass
from tqdm import tqdm
import numpy as np
import io_utils
from io_utils import get_precursor_mz
from spec2vec_emb import (
    get_config_value,
    get_msdata_length,
    get_spectrum_peaks,
)
import h5py

class H5MSDataLite:
    """
    Lightweight HDF5-backed MSData replacement.

    This class is designed to bypass dreams.utils.data.MSData.load(),
    while still supporting the subset of methods/behaviors commonly used
    by pre-encoding and retrieval pipelines.

    Supported:
        - len(msdata)
        - msdata.columns()
        - "col" in msdata
        - msdata["col"]
        - msdata.get_values("col", idx=None)
        - msdata.get_spectra(idx=None)
        - msdata.get_prec_mzs(idx=None)
        - msdata.get_smiles(idx=None)
        - msdata.get_charges(idx=None)
        - msdata.close()
    """

    def __init__(
        self,
        hdf5_path,
        mode="r",
        spectrum_col="spectrum",
        precursor_mz_col="precursor_mz",
        embedding_col="DreaMS_embedding",
    ):
        self.hdf5_pth = Path(hdf5_path)
        self.path = self.hdf5_pth
        self.mode = mode

        self.spectrum_col = spectrum_col
        self.precursor_mz_col = precursor_mz_col
        self.embedding_col = embedding_col

        self.f = h5py.File(str(self.hdf5_pth), mode)
        self.data = self.f
        self.in_mem = False

        # ---- Required checks ----
        if self.spectrum_col not in self.f:
            raise ValueError(
                f'Column "{self.spectrum_col}" is not present in dataset {self.hdf5_pth}. '
                f"Available columns: {list(self.f.keys())}"
            )

        if self.precursor_mz_col not in self.f:
            raise ValueError(
                f'Column "{self.precursor_mz_col}" is not present in dataset {self.hdf5_pth}. '
                f"Available columns: {list(self.f.keys())}"
            )

        spec_shape = self.f[self.spectrum_col].shape

        if len(spec_shape) != 3 or spec_shape[1] != 2:
            raise ValueError(
                f'Shape of "{self.spectrum_col}" must be (num_spectra, 2, num_peaks), '
                f"but got {spec_shape}."
            )

        # DreaMS / MoNA style: every root-level dataset should have same first dim.
        n = spec_shape[0]
        bad_cols = []

        for k in self.f.keys():
            obj = self.f[k]

            if isinstance(obj, h5py.Group):
                bad_cols.append((k, "GROUP", None))
                continue

            if not hasattr(obj, "shape") or len(obj.shape) == 0:
                continue

            if obj.shape[0] != n:
                bad_cols.append((k, obj.shape, obj.dtype))

        if bad_cols:
            msg = "\n".join([f"  {x}" for x in bad_cols])
            raise ValueError(
                f"Some HDF5 root objects are not compatible table columns. "
                f"All datasets must have first dimension = {n}.\n{msg}"
            )

        self.num_spectra = int(n)

    def __len__(self):
        return self.num_spectra

    def __contains__(self, col):
        """
        Critical fix:
        allows `"DreaMS_embedding" in msdata_lib`
        without Python falling back to __getitem__(0).
        """
        return isinstance(col, str) and col in self.f.keys()

    def __getitem__(self, col):
        """
        Column access only.

        Example:
            msdata["spectrum"]
            msdata["precursor_mz"]

        Integer row access is intentionally not supported here.
        """
        if not isinstance(col, str):
            raise TypeError(
                f"H5MSDataLite.__getitem__ expects a string column name, "
                f"but got {type(col)}: {col}."
            )

        return self.get_values(col)

    def columns(self):
        return list(self.f.keys())

    def keys(self):
        return self.columns()

    def get_values(self, col, idx=None, decode_strings=True):
        if col not in self.f:
            raise KeyError(
                f'Column "{col}" is not present in {self.hdf5_pth}. '
                f"Available columns: {list(self.f.keys())}"
            )

        ds = self.f[col]

        if idx is None:
            val = ds[:]
        else:
            val = ds[idx]

        if decode_strings:
            val = self._decode_if_needed(val)

        return val

    @staticmethod
    def _decode_if_needed(val):
        """
        Decode bytes/object string arrays safely.
        Keep numeric arrays unchanged.
        """
        if isinstance(val, bytes):
            return val.decode("utf-8", errors="ignore")

        if isinstance(val, np.bytes_):
            return val.astype(str)

        if isinstance(val, np.ndarray):
            if val.dtype.kind == "S":
                return np.char.decode(val, "utf-8", errors="ignore")

            if val.dtype == object:
                flat = val.reshape(-1)
                decoded = [
                    x.decode("utf-8", errors="ignore") if isinstance(x, bytes) else x
                    for x in flat
                ]
                return np.asarray(decoded, dtype=object).reshape(val.shape)

        return val

    def get_spectra(self, idx=None):
        return self.get_values(self.spectrum_col, idx=idx, decode_strings=False)

    def get_prec_mzs(self, idx=None):
        return self.get_values(self.precursor_mz_col, idx=idx, decode_strings=False)

    def get_smiles(self, idx=None):
        return self.get_values("smiles", idx=idx, decode_strings=True)

    def get_charges(self, idx=None):
        return self.get_values("charge", idx=idx, decode_strings=False)

    def get_adducts(self, idx=None):
        return self.get_values("adduct", idx=idx, decode_strings=True)

    def close(self):
        try:
            self.f.close()
        except Exception:
            pass

    def __del__(self):
        self.close()

    def __repr__(self):
        return (
            f"H5MSDataLite(pth={self.hdf5_pth}, "
            f"in_mem=False) with {self.num_spectra:,} spectra."
        )
def load_mona_library(config):
    """
    Load MoNA or chimera HDF5 library without using dreams.utils.data.MSData.load().

    Returns
    -------
    msdata_lib:
        H5MSDataLite object.

    existing_embs_lib:
        h5py Dataset for DreaMS embeddings if present, otherwise None.

        Important:
            This returns a lazy HDF5 dataset, not a fully loaded numpy array.
            Downstream can slice it by batches.

    lib_cols:
        List of root-level HDF5 dataset names.
    """

    # ------------------------------------------------------------
    # 1. Resolve source path from config
    # ------------------------------------------------------------
    mona_hdf5_path = None

    # 支持 dict config
    if isinstance(config, dict):
        for key in [
            "source_path",
            "mona_hdf5_path",
            "library_path",
            "lib_path",
            "mona_path",
        ]:
            if key in config:
                mona_hdf5_path = config[key]
                break

        # 如果你的 config 是嵌套结构，也兼容一下
        if mona_hdf5_path is None:
            for parent_key in ["library", "data", "mona"]:
                if parent_key in config and isinstance(config[parent_key], dict):
                    for key in [
                        "source_path",
                        "mona_hdf5_path",
                        "library_path",
                        "lib_path",
                        "mona_path",
                    ]:
                        if key in config[parent_key]:
                            mona_hdf5_path = config[parent_key][key]
                            break

                    if mona_hdf5_path is not None:
                        break

    # 支持 OmegaConf / argparse Namespace / 普通对象 config
    else:
        for key in [
            "source_path",
            "mona_hdf5_path",
            "library_path",
            "lib_path",
            "mona_path",
        ]:
            if hasattr(config, key):
                mona_hdf5_path = getattr(config, key)
                break

        # 嵌套对象
        if mona_hdf5_path is None:
            for parent_key in ["library", "data", "mona"]:
                if hasattr(config, parent_key):
                    parent = getattr(config, parent_key)
                    for key in [
                        "source_path",
                        "mona_hdf5_path",
                        "library_path",
                        "lib_path",
                        "mona_path",
                    ]:
                        if hasattr(parent, key):
                            mona_hdf5_path = getattr(parent, key)
                            break

                    if mona_hdf5_path is not None:
                        break

    if mona_hdf5_path is None:
        raise ValueError(
            "Could not find MoNA library path in config. "
            "Expected one of: source_path, mona_hdf5_path, library_path, lib_path, mona_path."
        )

    mona_hdf5_path = Path(mona_hdf5_path)

    print("[data] Loading spectra for split=library...")
    print(f"[data] source_path = {mona_hdf5_path}")

    if not mona_hdf5_path.exists():
        raise FileNotFoundError(f"MoNA library HDF5 does not exist: {mona_hdf5_path}")

    # ------------------------------------------------------------
    # 2. Direct HDF5-backed loading
    # ------------------------------------------------------------
    msdata_lib = H5MSDataLite(
        hdf5_path=mona_hdf5_path,
        mode="r",
        spectrum_col="spectrum",
        precursor_mz_col="precursor_mz",
        embedding_col="DreaMS_embedding",
    )

    lib_cols = msdata_lib.columns()

    print(f"[data] Loaded library: {msdata_lib}")
    print(f"[data] Library columns: {lib_cols}")

    # ------------------------------------------------------------
    # 3. Existing embeddings
    # ------------------------------------------------------------
    existing_embs_lib = None

    if "DreaMS_embedding" in msdata_lib:
        emb_ds = msdata_lib.f["DreaMS_embedding"]

        if len(emb_ds.shape) != 2:
            print(
                f"[data] WARNING: DreaMS_embedding exists but shape is invalid: {emb_ds.shape}. "
                f"Will ignore it."
            )
            existing_embs_lib = None
        elif emb_ds.shape[0] != len(msdata_lib):
            print(
                f"[data] WARNING: DreaMS_embedding first dimension {emb_ds.shape[0]} "
                f"does not match number of spectra {len(msdata_lib)}. Will ignore it."
            )
            existing_embs_lib = None
        else:
            # Important:
            # Do not load the entire embedding matrix into memory here.
            # Return h5py.Dataset so downstream can slice lazily.
            existing_embs_lib = emb_ds
            print(f"[data] Found existing DreaMS_embedding: shape={emb_ds.shape}, dtype={emb_ds.dtype}")
    else:
        print("[data] No existing DreaMS_embedding column found.")

    return msdata_lib, existing_embs_lib, lib_cols

def config_to_worker_dict(config):
    """
    Convert config object to a pickle-friendly dict for multiprocessing workers.
    """

    if is_dataclass(config):
        return asdict(config)

    if isinstance(config, dict):
        return dict(config)

    result = {}

    if hasattr(config, "__dict__"):
        for key, value in vars(config).items():
            if key.startswith("_"):
                continue
            if callable(value):
                continue
            result[key] = value

    return result


def dict_to_worker_config(config_dict):
    """
    Rebuild a config object inside worker process.

    Prefer PipelineConfig if available, because some project functions may expect
    a PipelineConfig-like object. Fallback to SimpleNamespace.
    """

    try:
        from config import PipelineConfig

        config = PipelineConfig()

        for key, value in dict(config_dict).items():
            try:
                setattr(config, key, value)
            except Exception:
                pass

        return config

    except Exception:
        pass

    try:
        from .config import PipelineConfig

        config = PipelineConfig()

        for key, value in dict(config_dict).items():
            try:
                setattr(config, key, value)
            except Exception:
                pass

        return config

    except Exception:
        pass

    return SimpleNamespace(**dict(config_dict))


def get_embedding_dtype_from_config(config):
    """
    Resolve dense binned embedding dtype.

    Important:
    For binned embeddings, default to float32 to avoid accidental float64
    memory explosion.

    Recognized config keys:
    - binned_dtype
    - embedding_dtype
    - use_float32
    """

    dtype_name = get_config_value(
        config,
        "binned_dtype",
        get_config_value(config, "embedding_dtype", None),
    )

    if dtype_name is not None:
        dtype_name = str(dtype_name).strip().lower()

        if dtype_name in {"float16", "half", "fp16"}:
            return np.dtype(np.float16)

        if dtype_name in {"float32", "single", "fp32"}:
            return np.dtype(np.float32)

        if dtype_name in {"float64", "double", "fp64"}:
            return np.dtype(np.float64)

        raise ValueError(f"Unsupported binned dtype: {dtype_name}")

    if bool(get_config_value(config, "use_float32", False)):
        return np.dtype(np.float32)

    # Safer default for dense binned vectors.
    return np.dtype(np.float32)


def estimate_dense_array_size_gb(n_rows, n_cols, dtype=np.float32):
    itemsize = np.dtype(dtype).itemsize
    size_bytes = int(n_rows) * int(n_cols) * itemsize
    return size_bytes / 1024**3


def l2_normalize_inplace_rows(x, eps=1e-12, chunk_size=8192):
    """
    In-place row-wise L2 normalization.

    Avoids creating a second full-size dense matrix.
    """

    n = x.shape[0]

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)

        block = x[start:end]

        # Use float32 working precision for stable norms, especially if block is float16.
        work = block.astype(np.float32, copy=False)
        norms = np.sqrt(np.sum(work * work, axis=1, keepdims=True))
        norms = np.maximum(norms, eps)

        block /= norms.astype(block.dtype, copy=False)

    return x


def _build_dense_binned_shard_static_worker(task):
    """
    Static shard worker for binned / neutral-loss-binned embeddings.

    Each worker:
    - opens its own MoNA library handle
    - processes one contiguous index range
    - builds one dense shard
    - normalizes rows in-place
    - saves shard to .npy
    """

    (
        worker_id,
        config_dict,
        mode,
        start,
        end,
        mz_min,
        mz_max,
        bin_size,
        n_bins,
        dtype_name,
        shard_path,
        worker_log_every,
        progress_queue,
        progress_update_every,
    ) = task

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    config = dict_to_worker_config(config_dict)
    dtype = np.dtype(dtype_name)

    print(
        f"[{mode}-worker {worker_id}] pid={os.getpid()} "
        f"range={start}:{end}, n_bins={n_bins}, dtype={dtype.name}",
        flush=True,
    )

    t0 = time.time()
    try:
        msdata, existing_embs_lib, lib_cols = io_utils.load_mona_library(config)
    except Exception as e:
        msdata, existing_embs_lib, lib_cols = load_mona_library(config)

    shard_n = int(end) - int(start)
    shard = np.zeros((shard_n, n_bins), dtype=dtype)

    failed = 0
    empty = 0
    out_of_range = 0
    no_precursor = 0
    last_progress_sent = 0

    for local_pos, global_idx in enumerate(range(start, end)):
        try:
            if mode == "neutral_loss":
                precursor_mz = get_precursor_mz(msdata, global_idx)

                if not np.isfinite(precursor_mz):
                    no_precursor += 1
                    continue

                precursor_mz = float(precursor_mz)
            else:
                precursor_mz = None

            mzs, intensities = get_spectrum_peaks(msdata, global_idx, config)

            mzs = np.asarray(mzs, dtype=np.float64).reshape(-1)
            intensities = np.asarray(intensities, dtype=dtype).reshape(-1)

            if mzs.size == 0 or intensities.size == 0:
                empty += 1
                continue

            if mzs.shape[0] != intensities.shape[0]:
                raise ValueError(
                    f"m/z and intensity length mismatch: "
                    f"{mzs.shape[0]} vs {intensities.shape[0]}"
                )

            finite = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)

            if not np.any(finite):
                empty += 1
                continue

            mzs = mzs[finite]
            intensities = intensities[finite]

            if mode == "neutral_loss":
                values = precursor_mz - mzs
            else:
                values = mzs

            bin_idx = np.floor((values - mz_min) / bin_size).astype(np.int64)
            valid = (bin_idx >= 0) & (bin_idx < n_bins)

            if not np.any(valid):
                out_of_range += 1
                continue

            np.add.at(
                shard[local_pos],
                bin_idx[valid],
                intensities[valid].astype(dtype, copy=False),
            )

        except Exception as e:
            failed += 1

            if failed <= 10:
                print(
                    f"[{mode}-worker {worker_id}] WARNING: failed at index "
                    f"{global_idx}: {repr(e)}",
                    flush=True,
                )

        done = local_pos + 1

        # Send lightweight progress updates to main process.
        if progress_queue is not None and progress_update_every > 0:
            if done - last_progress_sent >= progress_update_every or done == shard_n:
                delta = done - last_progress_sent

                if delta > 0:
                    try:
                        progress_queue.put(
                            {
                                "delta": int(delta),
                                "worker_id": int(worker_id),
                            }
                        )
                    except Exception:
                        pass

                    last_progress_sent = done

        # Optional per-worker logs. Disabled by default to keep tqdm clean.
        if worker_log_every > 0 and (
                done % worker_log_every == 0 or done == shard_n
        ):
            elapsed = time.time() - t0
            speed = done / elapsed if elapsed > 0 else 0.0

            print(
                f"[{mode}-worker {worker_id}] "
                f"{done}/{shard_n}, failed={failed}, empty={empty}, "
                f"out_of_range={out_of_range}, no_precursor={no_precursor}, "
                f"speed={speed:.1f} spectra/s",
                flush=True,
            )

    try:
        if hasattr(msdata, "close"):
            msdata.close()
    except Exception:
        pass

    # Row-wise L2 normalization is independent, so it is safe to normalize per shard.
    l2_normalize_inplace_rows(shard)

    shard_path = Path(shard_path)
    shard_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(shard_path, shard)

    elapsed = time.time() - t0

    print(
        f"[{mode}-worker {worker_id}] saved {shard_path}, "
        f"shape={shard.shape}, failed={failed}, empty={empty}, "
        f"out_of_range={out_of_range}, no_precursor={no_precursor}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )

    return {
        "worker_id": int(worker_id),
        "mode": str(mode),
        "start": int(start),
        "end": int(end),
        "shard_path": str(shard_path),
        "failed": int(failed),
        "empty": int(empty),
        "out_of_range": int(out_of_range),
        "no_precursor": int(no_precursor),
        "elapsed": float(elapsed),
    }


def _build_dense_binned_embeddings_parallel_static(
    msdata,
    config,
    limit=None,
    mode="peak",
    mz_min=0.0,
    mz_max=2000.0,
    bin_size=0.1,
):
    """
    Shared static multiprocessing builder.

    mode:
    - "peak": normal binned peaks
    - "neutral_loss": precursor_mz - fragment_mz binned features
    """

    if mode not in {"peak", "neutral_loss"}:
        raise ValueError(f"Unsupported binned mode: {mode}")

    mz_min = float(mz_min)
    mz_max = float(mz_max)
    bin_size = float(bin_size)

    if mz_max <= mz_min:
        raise ValueError(f"{mode}: mz_max must be larger than mz_min.")

    if bin_size <= 0:
        raise ValueError(f"{mode}: bin_size must be positive.")

    n_bins = int(np.ceil((mz_max - mz_min) / bin_size))

    n_total = get_msdata_length(msdata)
    n = n_total if limit is None else min(int(limit), n_total)

    dtype = get_embedding_dtype_from_config(config)
    dtype_name = dtype.name

    estimated_gb = estimate_dense_array_size_gb(n, n_bins, dtype=dtype)

    default_workers = max(1, min((os.cpu_count() or 4) - 1, 8))

    num_workers = int(
        get_config_value(
            config,
            "binned_num_workers",
            get_config_value(config, "embedding_num_workers", default_workers),
        )
    )

    num_workers = max(1, min(num_workers, n if n > 0 else 1))

    progress_bar_enabled = bool(
        get_config_value(config, "binned_progress_bar", True)
    )

    progress_update_every = int(
        get_config_value(config, "binned_progress_update_every", 1000)
    )

    progress_update_every = max(progress_update_every, 1)

    worker_log_every = int(
        get_config_value(config, "binned_worker_log_every", 0)
    )

    cache_dir = Path(
        get_config_value(config, "embedding_cache_dir", "outputs/embedding_cache")
    )

    shard_root = Path(
        get_config_value(
            config,
            "binned_shard_dir",
            cache_dir / "_binned_static_shards",
        )
    )

    run_id = f"{mode}_{int(time.time())}_pid{os.getpid()}"
    shard_dir = shard_root / run_id
    shard_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[{mode}] static multiprocessing enabled",
        flush=True,
    )
    print(
        f"[{mode}] n_spectra={n}, n_bins={n_bins}, dtype={dtype_name}, "
        f"estimated_dense_size={estimated_gb:.3f} GiB",
        flush=True,
    )
    print(
        f"[{mode}] mz_min={mz_min}, mz_max={mz_max}, bin_size={bin_size}",
        flush=True,
    )
    print(
        f"[{mode}] num_workers={num_workers}, shard_dir={shard_dir}",
        flush=True,
    )

    if n <= 0:
        return np.zeros((0, n_bins), dtype=dtype)

    config_dict = config_to_worker_dict(config)

    t0 = time.time()

    ctx = mp.get_context("spawn")

    manager = ctx.Manager()
    progress_queue = manager.Queue()

    progress_desc = f"{mode} encode"

    progress_thread = threading.Thread(
        target=_consume_progress_queue,
        args=(
            progress_queue,
            n,
            progress_desc,
            progress_bar_enabled,
        ),
        daemon=True,
    )

    progress_thread.start()

    tasks = []

    for worker_id in range(num_workers):
        start = worker_id * n // num_workers
        end = (worker_id + 1) * n // num_workers

        shard_path = shard_dir / f"{mode}_shard_{worker_id:03d}_{start}_{end}.npy"

        tasks.append(
            (
                worker_id,
                config_dict,
                mode,
                start,
                end,
                mz_min,
                mz_max,
                bin_size,
                n_bins,
                dtype_name,
                str(shard_path),
                worker_log_every,
                progress_queue,
                progress_update_every,
            )
        )

    try:
        with ctx.Pool(processes=num_workers) as pool:
            results = pool.map(_build_dense_binned_shard_static_worker, tasks)

    finally:
        try:
            progress_queue.put(None)
        except Exception:
            pass

        try:
            progress_thread.join(timeout=10.0)
        except Exception:
            pass

        try:
            manager.shutdown()
        except Exception:
            pass

    results = sorted(results, key=lambda x: x["worker_id"])

    preprocess_elapsed = time.time() - t0

    total_failed = sum(int(r["failed"]) for r in results)
    total_empty = sum(int(r["empty"]) for r in results)
    total_out_of_range = sum(int(r["out_of_range"]) for r in results)
    total_no_precursor = sum(int(r["no_precursor"]) for r in results)

    print(
        f"[{mode}] shard building finished. "
        f"failed={total_failed}, empty={total_empty}, "
        f"out_of_range={total_out_of_range}, no_precursor={total_no_precursor}, "
        f"elapsed={preprocess_elapsed:.1f}s",
        flush=True,
    )

    print(
        f"[{mode}] merging shards into dense matrix...",
        flush=True,
    )

    merge_t0 = time.time()

    # np.empty avoids the extra zero-fill cost. Every row is filled by exactly one shard.
    embeddings = np.empty((n, n_bins), dtype=dtype)

    if progress_bar_enabled and tqdm is not None:
        merge_iter = tqdm(
            results,
            desc=f"{mode} merge",
            unit="shard",
            dynamic_ncols=True,
        )
    else:
        merge_iter = results

    for r in merge_iter:
        start = int(r["start"])
        end = int(r["end"])
        shard_path = r["shard_path"]

        shard = np.load(shard_path, mmap_mode="r")

        expected_shape = (end - start, n_bins)

        if shard.shape != expected_shape:
            raise ValueError(
                f"Shard shape mismatch for {shard_path}: "
                f"expected {expected_shape}, got {shard.shape}"
            )

        embeddings[start:end, :] = shard

        del shard

        if not (progress_bar_enabled and tqdm is not None):
            print(
                f"[{mode}] merged shard worker={r['worker_id']} "
                f"range={start}:{end}",
                flush=True,
            )

    merge_elapsed = time.time() - merge_t0

    keep_shards = bool(get_config_value(config, "binned_keep_shards", False))

    if not keep_shards:
        try:
            shutil.rmtree(shard_dir)
            print(f"[{mode}] removed temporary shard_dir: {shard_dir}", flush=True)
        except Exception as e:
            print(
                f"[{mode}] WARNING: failed to remove shard_dir {shard_dir}: {repr(e)}",
                flush=True,
            )
    else:
        print(f"[{mode}] kept temporary shard_dir: {shard_dir}", flush=True)

    elapsed = time.time() - t0

    print(
        f"[{mode}] finished. shape={embeddings.shape}, dtype={embeddings.dtype}, "
        f"preprocess_elapsed={preprocess_elapsed:.1f}s, "
        f"merge_elapsed={merge_elapsed:.1f}s, total_elapsed={elapsed:.1f}s",
        flush=True,
    )

    return embeddings


def build_binned_embeddings(msdata, config, limit=None):
    """
    Build normal peak binned embeddings with static multiprocessing.
    """

    mz_min = float(get_config_value(config, "embedding_mz_min", 0.0))
    mz_max = float(get_config_value(config, "embedding_mz_max", 2000.0))
    bin_size = float(get_config_value(config, "embedding_bin_size", 0.1))

    return _build_dense_binned_embeddings_parallel_static(
        msdata=msdata,
        config=config,
        limit=limit,
        mode="peak",
        mz_min=mz_min,
        mz_max=mz_max,
        bin_size=bin_size,
    )


def build_neutral_loss_binned_embeddings(msdata, config, limit=None):
    """
    Build peak binned + neutral-loss binned embeddings.

    This is more memory-heavy than normal binned because the final vector is:
        [peak_binned, neutral_loss_weight * loss_binned]

    The function builds peak first, copies it into the final matrix, frees peak,
    then builds loss, copies it, frees loss, then normalizes the combined matrix.
    """

    peak_mz_min = float(get_config_value(config, "embedding_mz_min", 0.0))
    peak_mz_max = float(get_config_value(config, "embedding_mz_max", 2000.0))
    bin_size = float(get_config_value(config, "embedding_bin_size", 0.1))

    loss_min = float(get_config_value(config, "neutral_loss_mz_min", 0.0))
    loss_max = float(get_config_value(config, "neutral_loss_mz_max", 1000.0))

    if loss_max <= loss_min:
        raise ValueError("neutral_loss_mz_max must be larger than neutral_loss_mz_min.")

    dtype = get_embedding_dtype_from_config(config)

    n_total = get_msdata_length(msdata)
    n = n_total if limit is None else min(int(limit), n_total)

    peak_bins = int(np.ceil((peak_mz_max - peak_mz_min) / bin_size))
    loss_bins = int(np.ceil((loss_max - loss_min) / bin_size))
    total_bins = peak_bins + loss_bins

    estimated_gb = estimate_dense_array_size_gb(n, total_bins, dtype=dtype)

    print(
        f"[neutral_loss_binned] final_dim={total_bins}, "
        f"peak_bins={peak_bins}, loss_bins={loss_bins}, "
        f"estimated_final_dense_size={estimated_gb:.3f} GiB",
        flush=True,
    )

    peak_emb = _build_dense_binned_embeddings_parallel_static(
        msdata=msdata,
        config=config,
        limit=limit,
        mode="peak",
        mz_min=peak_mz_min,
        mz_max=peak_mz_max,
        bin_size=bin_size,
    )

    combined = np.empty((n, total_bins), dtype=dtype)
    combined[:, :peak_bins] = peak_emb

    del peak_emb

    loss_emb = _build_dense_binned_embeddings_parallel_static(
        msdata=msdata,
        config=config,
        limit=limit,
        mode="neutral_loss",
        mz_min=loss_min,
        mz_max=loss_max,
        bin_size=bin_size,
    )

    neutral_loss_weight = float(get_config_value(config, "neutral_loss_weight", 1.0))

    np.multiply(
        loss_emb,
        neutral_loss_weight,
        out=combined[:, peak_bins:],
        casting="unsafe",
    )

    del loss_emb

    print("[neutral_loss_binned] L2 normalizing combined matrix in-place...", flush=True)

    l2_normalize_inplace_rows(combined)

    print(
        f"[neutral_loss_binned] finished. shape={combined.shape}, "
        f"dtype={combined.dtype}",
        flush=True,
    )

    return combined

def _consume_progress_queue(progress_queue, total, desc="binned", enabled=True):
    """
    Consume progress events from worker processes and show one global progress bar.

    Workers send:
        {"delta": int, "worker_id": int}

    Main process sends:
        None

    to stop this consumer.
    """

    total = int(total)
    completed = 0
    t0 = time.time()
    last_print = t0

    use_tqdm = enabled and tqdm is not None

    if use_tqdm:
        with tqdm(
            total=total,
            desc=desc,
            unit="spectra",
            dynamic_ncols=True,
            smoothing=0.05,
        ) as pbar:
            while True:
                msg = progress_queue.get()

                if msg is None:
                    break

                delta = int(msg.get("delta", 0))

                if delta <= 0:
                    continue

                completed += delta
                pbar.update(delta)

        return

    # Fallback when tqdm is not installed or disabled.
    while True:
        try:
            msg = progress_queue.get(timeout=1.0)
        except queue_module.Empty:
            now = time.time()

            if now - last_print >= 10.0:
                elapsed = now - t0
                speed = completed / elapsed if elapsed > 0 else 0.0
                pct = 100.0 * completed / total if total > 0 else 100.0
                eta = (total - completed) / speed if speed > 0 else float("inf")

                print(
                    f"[{desc}] progress {completed}/{total} "
                    f"({pct:.2f}%), speed={speed:.1f} spectra/s, "
                    f"eta={eta / 60:.1f} min",
                    flush=True,
                )

                last_print = now

            continue

        if msg is None:
            break

        delta = int(msg.get("delta", 0))

        if delta <= 0:
            continue

        completed += delta

        now = time.time()

        if now - last_print >= 10.0 or completed >= total:
            elapsed = now - t0
            speed = completed / elapsed if elapsed > 0 else 0.0
            pct = 100.0 * completed / total if total > 0 else 100.0
            eta = (total - completed) / speed if speed > 0 else 0.0

            print(
                f"[{desc}] progress {completed}/{total} "
                f"({pct:.2f}%), speed={speed:.1f} spectra/s, "
                f"eta={eta / 60:.1f} min",
                flush=True,
            )

            last_print = now

def build_binned_embeddings_from_mzml(
    mzml_path,
    output_path,
    config,
    *,
    mode="binned",
    ms_level=2,
    max_spectra=None,
    limit=None,
):
    """
    Direct mzML binned embedding builder.

    mode:
        - "binned":
            output = peak_binned
            dim = peak_bins

        - "neutral_loss" / "neutral_loss_binned":
            output = [peak_binned, neutral_loss_weight * loss_binned]
            dim = peak_bins + loss_bins

    This matches library-side build_neutral_loss_binned_embeddings(...):
        1. build peak binned
        2. L2 normalize peak rows
        3. build neutral-loss binned
        4. L2 normalize loss rows
        5. multiply loss part by neutral_loss_weight
        6. concatenate
        7. L2 normalize combined rows
    """

    try:
        from mzml_input import load_mzml_spectra, save_mzml_embeddings_npz
    except ImportError:
        from .mzml_input import load_mzml_spectra, save_mzml_embeddings_npz

    mode = str(mode).lower().strip().replace("-", "_")

    if mode not in {"binned", "neutral_loss", "neutral_loss_binned"}:
        raise ValueError(f"Unsupported binned mode: {mode}")

    spectra = load_mzml_spectra(
        mzml_path,
        ms_level=ms_level,
        max_spectra=max_spectra,
        dtype=np.float32,
    )

    if limit is not None:
        spectra = spectra[: int(limit)]

    # Use the same config keys as library-side binned embedding builder.
    peak_mz_min = float(
        get_config_value(
            config,
            "embedding_mz_min",
            get_config_value(config, "embedding_mz_min", 50.0),
        )
    )

    peak_mz_max = float(
        get_config_value(
            config,
            "embedding_mz_max",
            get_config_value(config, "embedding_mz_max", 1200.0),
        )
    )

    bin_size = float(
        get_config_value(
            config,
            "embedding_bin_size",
            get_config_value(config, "embedding_bin_size", 1),
        )
    )

    if bin_size <= 0:
        raise ValueError(f"embedding_bin_size must be positive, got {bin_size}")

    if peak_mz_max <= peak_mz_min:
        raise ValueError(
            f"Invalid peak bin range: "
            f"peak_mz_min={peak_mz_min}, peak_mz_max={peak_mz_max}"
        )

    peak_bins = int(np.ceil((peak_mz_max - peak_mz_min) / bin_size))

    dtype = get_embedding_dtype_from_config(config)

    use_combined_neutral_loss = mode in {"neutral_loss", "neutral_loss_binned"}

    if use_combined_neutral_loss:
        loss_min = float(get_config_value(config, "neutral_loss_mz_min", 0.0))
        loss_max = float(get_config_value(config, "neutral_loss_mz_max", 1000.0))

        if loss_max <= loss_min:
            raise ValueError(
                f"Invalid neutral-loss bin range: "
                f"neutral_loss_mz_min={loss_min}, neutral_loss_mz_max={loss_max}"
            )

        loss_bins = int(np.ceil((loss_max - loss_min) / bin_size))
        total_bins = peak_bins + loss_bins

        neutral_loss_weight = float(
            get_config_value(config, "neutral_loss_weight", 1.0)
        )

        embeddings = np.zeros((len(spectra), total_bins), dtype=dtype)

        print(
            f"[neutral_loss_binned-mzML] peak_bins={peak_bins}, "
            f"loss_bins={loss_bins}, total_bins={total_bins}, "
            f"bin_size={bin_size}, dtype={dtype}",
            flush=True,
        )

    else:
        loss_min = None
        loss_max = None
        loss_bins = 0
        total_bins = peak_bins
        neutral_loss_weight = 1.0

        embeddings = np.zeros((len(spectra), peak_bins), dtype=dtype)

        print(
            f"[binned-mzML] peak_bins={peak_bins}, "
            f"bin_size={bin_size}, dtype={dtype}",
            flush=True,
        )

    for i, spec in enumerate(
        tqdm(
            spectra,
            desc=f"[{mode}-mzML] Encoding",
            unit="spectra",
            dynamic_ncols=True,
        )
    ):
        mzs = np.asarray(spec["mz"], dtype=np.float64).reshape(-1)
        intensities = np.asarray(spec["intensity"], dtype=dtype).reshape(-1)

        valid = (
            np.isfinite(mzs)
            & np.isfinite(intensities)
            & (intensities > 0)
        )

        mzs = mzs[valid]
        intensities = intensities[valid]

        if mzs.size == 0:
            continue

        # -------------------------------------------------------------
        # 1. Peak binned part
        # -------------------------------------------------------------
        peak_bin_idx = np.floor((mzs - peak_mz_min) / bin_size).astype(np.int64)
        peak_valid = (peak_bin_idx >= 0) & (peak_bin_idx < peak_bins)

        if np.any(peak_valid):
            np.add.at(
                embeddings[i, :peak_bins],
                peak_bin_idx[peak_valid],
                intensities[peak_valid].astype(dtype, copy=False),
            )

        # -------------------------------------------------------------
        # 2. Neutral-loss binned part
        # -------------------------------------------------------------
        if use_combined_neutral_loss:
            precursor_mz = float(spec.get("precursor_mz", np.nan))

            if not np.isfinite(precursor_mz):
                continue

            losses = precursor_mz - mzs

            loss_bin_idx = np.floor((losses - loss_min) / bin_size).astype(np.int64)
            loss_valid = (loss_bin_idx >= 0) & (loss_bin_idx < loss_bins)

            if np.any(loss_valid):
                np.add.at(
                    embeddings[i, peak_bins:],
                    loss_bin_idx[loss_valid],
                    intensities[loss_valid].astype(dtype, copy=False),
                )

    # -----------------------------------------------------------------
    # Normalize exactly like library-side logic
    # -----------------------------------------------------------------
    if use_combined_neutral_loss:
        print(
            "[neutral_loss_binned-mzML] L2 normalizing peak part...",
            flush=True,
        )
        l2_normalize_inplace_rows(embeddings[:, :peak_bins])

        print(
            "[neutral_loss_binned-mzML] L2 normalizing neutral-loss part...",
            flush=True,
        )
        l2_normalize_inplace_rows(embeddings[:, peak_bins:])

        print(
            f"[neutral_loss_binned-mzML] applying neutral_loss_weight={neutral_loss_weight}...",
            flush=True,
        )
        np.multiply(
            embeddings[:, peak_bins:],
            neutral_loss_weight,
            out=embeddings[:, peak_bins:],
            casting="unsafe",
        )

        print(
            "[neutral_loss_binned-mzML] L2 normalizing combined matrix...",
            flush=True,
        )
        l2_normalize_inplace_rows(embeddings)

        save_method = "neutral_loss_binned"

        extra = {
            "mode": save_method,
            "peak_mz_min": peak_mz_min,
            "peak_mz_max": peak_mz_max,
            "neutral_loss_mz_min": loss_min,
            "neutral_loss_mz_max": loss_max,
            "bin_size": bin_size,
            "peak_bins": peak_bins,
            "loss_bins": loss_bins,
            "total_bins": total_bins,
            "neutral_loss_weight": neutral_loss_weight,
        }

    else:
        print(
            "[binned-mzML] L2 normalizing peak matrix...",
            flush=True,
        )
        l2_normalize_inplace_rows(embeddings)

        save_method = "binned"

        extra = {
            "mode": save_method,
            "mz_min": peak_mz_min,
            "mz_max": peak_mz_max,
            "bin_size": bin_size,
            "n_bins": peak_bins,
        }

    saved_path = save_mzml_embeddings_npz(
        output_path,
        embeddings,
        spectra,
        mzml_path=mzml_path,
        method=save_method,
        extra=extra,
    )

    print(f"[{mode}-mzML] Saved embeddings to: {saved_path}", flush=True)
    print(f"[{mode}-mzML] shape={embeddings.shape}", flush=True)

    return embeddings
