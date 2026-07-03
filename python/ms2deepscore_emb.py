import argparse
import json
import os
import subprocess
import sys
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import h5py
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
import io_utils
from io_utils import get_precursor_mz
from utils import l2_normalize
from spec2vec_emb import (
    get_config_value,
    get_msdata_length,
    cast_embedding_dtype,
    get_spectrum_peaks,
)
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
def load_mona_library(mona_hdf5_path=None):
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
def build_ms2deepscore_runtime_context(config):
    """
    Build runtime context.

    For speed, use a fixed ionmode by default.
    """

    default_ion_mode = get_config_value(
        config,
        "ms2deepscore_default_ion_mode",
        get_config_value(config, "ms2deepscore_ion_mode", "positive"),
    )

    default_ion_mode = normalize_ion_mode(default_ion_mode)

    if default_ion_mode not in {"positive", "negative"}:
        raise ValueError(
            "ms2deepscore_default_ion_mode must be 'positive' or 'negative'. "
            f"Got: {default_ion_mode}"
        )

    return SimpleNamespace(
        default_ion_mode=default_ion_mode,
    )
_MS2DS_WORKER_CONFIG = None
_MS2DS_WORKER_MSDATA = None
_MS2DS_WORKER_IONMODE = None
_MS2DS_WORKER_PEAK_DTYPE = None


def _ms2ds_cpu_worker_init(config_dict, ionmode, peak_dtype_name):
    """
    Initializer for each CPU worker process.

    Important:
    Each process opens its own MoNA library handle.
    Do NOT share msdata_lib / HDF5 handle across processes.
    """

    global _MS2DS_WORKER_CONFIG
    global _MS2DS_WORKER_MSDATA
    global _MS2DS_WORKER_IONMODE
    global _MS2DS_WORKER_PEAK_DTYPE

    os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    _MS2DS_WORKER_CONFIG = dict_to_config(config_dict)
    _MS2DS_WORKER_IONMODE = ionmode

    peak_dtype_name = str(peak_dtype_name).lower()

    if peak_dtype_name in {"float64", "double"}:
        _MS2DS_WORKER_PEAK_DTYPE = np.float64
    else:
        _MS2DS_WORKER_PEAK_DTYPE = np.float32

    print(
        f"[ms2deepscore-cpu-worker] pid={os.getpid()} loading MoNA library...",
        flush=True,
    )
    try:
        msdata_lib, existing_embs_lib, lib_cols = io_utils.load_mona_library(_MS2DS_WORKER_CONFIG)
    except Exception as e:
        msdata_lib, existing_embs_lib, lib_cols = load_mona_library(r"D:\亚结构注释\mona_processed\mona_chimera_dataset_equal_200k.hdf5")

    _MS2DS_WORKER_MSDATA = msdata_lib

    print(
        f"[ms2deepscore-cpu-worker] pid={os.getpid()} ready. "
        f"peak_dtype={_MS2DS_WORKER_PEAK_DTYPE}, ionmode={_MS2DS_WORKER_IONMODE}",
        flush=True,
    )
def _ms2ds_build_spectrum_chunk(task):
    """
    CPU worker task.

    Build matchms.Spectrum objects for index range [start, end).

    Returns:
        {
            "start": int,
            "end": int,
            "spectra": list[matchms.Spectrum],
            "valid_positions": list[int],
            "failed": int,
        }
    """

    global _MS2DS_WORKER_CONFIG
    global _MS2DS_WORKER_MSDATA
    global _MS2DS_WORKER_IONMODE
    global _MS2DS_WORKER_PEAK_DTYPE

    start, end = task

    if _MS2DS_WORKER_CONFIG is None or _MS2DS_WORKER_MSDATA is None:
        raise RuntimeError("MS2DeepScore CPU worker is not initialized.")

    try:
        from matchms import Spectrum
    except ImportError as e:
        raise ImportError(
            "matchms is required for MS2DeepScore worker process."
        ) from e

    spectra = []
    valid_positions = []
    failed = 0

    for pos, idx in enumerate(range(start, end)):
        try:
            mzs, intensities = get_spectrum_peaks(
                _MS2DS_WORKER_MSDATA,
                idx,
                _MS2DS_WORKER_CONFIG,
            )

            precursor_mz = get_precursor_mz(_MS2DS_WORKER_MSDATA, idx)

            mzs = np.asarray(mzs, dtype=_MS2DS_WORKER_PEAK_DTYPE).reshape(-1)
            intensities = np.asarray(
                intensities,
                dtype=_MS2DS_WORKER_PEAK_DTYPE,
            ).reshape(-1)

            if mzs.shape[0] != intensities.shape[0]:
                raise ValueError(
                    f"m/z and intensity length mismatch: "
                    f"{mzs.shape[0]} vs {intensities.shape[0]}"
                )

            mask = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)

            mzs = mzs[mask]
            intensities = intensities[mask]

            if mzs.size == 0:
                raise ValueError("no valid peaks")

            metadata = {
                "ionmode": _MS2DS_WORKER_IONMODE,
                "ion_mode": _MS2DS_WORKER_IONMODE,
                "polarity": _MS2DS_WORKER_IONMODE,
            }

            if np.isfinite(precursor_mz):
                precursor_mz = float(precursor_mz)
                metadata["precursor_mz"] = precursor_mz
                metadata["parent_mass"] = precursor_mz

            spectrum = Spectrum(
                mz=mzs,
                intensities=intensities,
                metadata=metadata,
            )

            spectra.append(spectrum)
            valid_positions.append(pos)

        except Exception:
            failed += 1

    return {
        "start": start,
        "end": end,
        "spectra": spectra,
        "valid_positions": valid_positions,
        "failed": failed,
    }
def encode_spectrum_chunk_on_gpu(
    model,
    chunk,
    embedding_dim,
    config,
    device=None,
):
    """
    Encode one CPU-built Spectrum chunk on GPU.

    If batch encoding fails, fallback to single-spectrum encoding.
    """

    start = int(chunk["start"])
    end = int(chunk["end"])
    spectra = chunk["spectra"]
    valid_positions = chunk["valid_positions"]
    build_failed = int(chunk["failed"])

    batch_size = end - start

    batch_embeddings = np.zeros((batch_size, embedding_dim), dtype=np.float32)

    if len(spectra) == 0:
        return batch_embeddings, build_failed

    try:
        emb_valid = ms2deepscore_spectra_to_vectors(
            model=model,
            spectra=spectra,
            device=device,
            progress_bar=False,
        )

        emb_valid = np.asarray(emb_valid, dtype=np.float32)

        if emb_valid.ndim != 2:
            raise ValueError(f"Batch embedding has invalid shape: {emb_valid.shape}")

        if emb_valid.shape[0] != len(spectra):
            raise ValueError(
                f"Batch embedding row mismatch: expected {len(spectra)}, "
                f"got {emb_valid.shape[0]}"
            )

        if emb_valid.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {embedding_dim}, "
                f"got {emb_valid.shape[1]}"
            )

        for row, pos in enumerate(valid_positions):
            batch_embeddings[pos, :] = emb_valid[row]

        return batch_embeddings, build_failed

    except Exception as batch_error:
        print(
            f"[ms2deepscore] WARNING: GPU batch encoding failed for "
            f"{start}:{end}; fallback to single. Error: {repr(batch_error)}",
            flush=True,
        )

    failed = build_failed

    for spectrum, pos in zip(spectra, valid_positions):
        try:
            emb = ms2deepscore_spectrum_to_vector(
                model=model,
                spectrum=spectrum,
                device=device,
            )

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if emb.size != embedding_dim:
                raise ValueError(
                    f"Single embedding dimension mismatch: "
                    f"expected {embedding_dim}, got {emb.size}"
                )

            batch_embeddings[pos, :] = emb

        except Exception as e:
            failed += 1
            print(
                f"[ms2deepscore] WARNING: single GPU encoding failed at "
                f"global index {start + pos}; filled with zeros. "
                f"Error: {repr(e)}",
                flush=True,
            )

    return batch_embeddings, failed
def infer_embedding_dim_from_chunk(
    model,
    chunk,
    device=None,
):
    """
    Infer embedding dimension from the first valid Spectrum in a CPU-built chunk.
    """

    spectra = chunk["spectra"]

    if len(spectra) == 0:
        return None, None

    for spectrum in spectra:
        try:
            emb = ms2deepscore_spectrum_to_vector(
                model=model,
                spectrum=spectrum,
                device=device,
            )

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if emb.size > 0:
                return int(emb.size), emb

        except Exception as e:
            print(
                f"[ms2deepscore] WARNING: failed to infer embedding dim "
                f"from one spectrum: {repr(e)}",
                flush=True,
            )

    return None, None

def preload_mona_spectra_to_memory(msdata_lib, config, n, runtime_context):
    """
    Preload all required MoNA spectral data into RAM.

    This avoids repeatedly reading from the original msdata object during
    MS2DeepScore embedding.

    Cached fields:
    - mz arrays
    - intensity arrays
    - precursor_mz
    - fixed ionmode from runtime_context
    """

    peak_dtype_name = str(
        get_config_value(config, "ms2deepscore_preload_peak_dtype", "float32")
    ).lower()

    if peak_dtype_name in {"float64", "double"}:
        peak_dtype = np.float64
    else:
        peak_dtype = np.float32

    print(
        f"[ms2deepscore] Preloading MoNA spectra into RAM: n={n}, peak_dtype={peak_dtype}",
        flush=True,
    )

    mzs_list = [None] * n
    intensities_list = [None] * n
    precursor_mzs = np.full(n, np.nan, dtype=np.float32)
    valid_mask = np.zeros(n, dtype=bool)

    failed = 0
    t0 = time.time()

    report_every = int(
        get_config_value(config, "ms2deepscore_preload_report_every", 50000)
    )
    report_every = max(report_every, 1)

    for i in range(n):
        try:
            mzs, intensities = get_spectrum_peaks(msdata_lib, i, config)
            precursor_mz = get_precursor_mz(msdata_lib, i)

            mzs = np.asarray(mzs, dtype=peak_dtype).reshape(-1)
            intensities = np.asarray(intensities, dtype=peak_dtype).reshape(-1)

            if mzs.shape[0] != intensities.shape[0]:
                raise ValueError(
                    f"m/z and intensity length mismatch: "
                    f"{mzs.shape[0]} vs {intensities.shape[0]}"
                )

            mask = np.isfinite(mzs) & np.isfinite(intensities) & (intensities > 0)

            mzs = mzs[mask]
            intensities = intensities[mask]

            if mzs.size == 0:
                raise ValueError("no valid peaks")

            mzs_list[i] = mzs
            intensities_list[i] = intensities
            valid_mask[i] = True

            if np.isfinite(precursor_mz):
                precursor_mzs[i] = float(precursor_mz)

        except Exception as e:
            failed += 1

            if failed <= 20:
                print(
                    f"[ms2deepscore] WARNING: preload failed at index {i}: {repr(e)}",
                    flush=True,
                )
            elif failed == 21:
                print(
                    "[ms2deepscore] further preload failures suppressed...",
                    flush=True,
                )

        if (i + 1) % report_every == 0 or (i + 1) == n:
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed if elapsed > 0 else 0.0

            n_valid = int(valid_mask[: i + 1].sum())

            print(
                f"[ms2deepscore] preloaded {i + 1}/{n}, "
                f"valid={n_valid}, failed={failed}, "
                f"speed={speed:.1f} spectra/s",
                flush=True,
            )

    elapsed = time.time() - t0

    n_valid = int(valid_mask.sum())

    print(
        f"[ms2deepscore] preload finished: n={n}, valid={n_valid}, "
        f"failed={failed}, elapsed={elapsed:.1f}s",
        flush=True,
    )

    return SimpleNamespace(
        mzs_list=mzs_list,
        intensities_list=intensities_list,
        precursor_mzs=precursor_mzs,
        valid_mask=valid_mask,
        ionmode=runtime_context.default_ion_mode,
        n=n,
        failed=failed,
        peak_dtype=str(np.dtype(peak_dtype)),
    )
def cached_spectrum_to_matchms_spectrum(cache, idx):
    """
    Build one matchms.Spectrum from preloaded RAM cache.
    No access to original msdata_lib.
    """

    try:
        from matchms import Spectrum
    except ImportError as e:
        raise ImportError(
            "matchms is required for MS2DeepScore. "
            "Install it inside the MS2DeepScore Python environment."
        ) from e

    if idx < 0 or idx >= cache.n:
        raise IndexError(f"Spectrum index out of range: {idx}")

    if not bool(cache.valid_mask[idx]):
        raise ValueError(f"Cached spectrum at index {idx} is invalid.")

    mzs = cache.mzs_list[idx]
    intensities = cache.intensities_list[idx]

    if mzs is None or intensities is None:
        raise ValueError(f"Cached spectrum at index {idx} is empty.")

    metadata = {
        "ionmode": cache.ionmode,
        "ion_mode": cache.ionmode,
        "polarity": cache.ionmode,
    }

    precursor_mz = cache.precursor_mzs[idx]

    if np.isfinite(precursor_mz):
        precursor_mz = float(precursor_mz)
        metadata["precursor_mz"] = precursor_mz
        metadata["parent_mass"] = precursor_mz

    return Spectrum(
        mz=mzs,
        intensities=intensities,
        metadata=metadata,
    )


def build_matchms_spectra_batch_from_cache(cache, indices):
    """
    Build one batch of matchms.Spectrum from RAM cache.
    """

    spectra = []
    valid_positions = []
    failed_count = 0

    for pos, idx in enumerate(indices):
        try:
            spectrum = cached_spectrum_to_matchms_spectrum(cache, idx)
            spectra.append(spectrum)
            valid_positions.append(pos)

        except Exception as e:
            failed_count += 1

            if failed_count <= 5:
                print(
                    f"[ms2deepscore] WARNING: failed to build cached Spectrum "
                    f"at index {idx}: {repr(e)}",
                    flush=True,
                )

    if failed_count > 5:
        print(
            f"[ms2deepscore] WARNING: suppressed {failed_count - 5} more "
            f"cached Spectrum-build failures in this batch.",
            flush=True,
        )

    return spectra, valid_positions, failed_count

def get_project_root():
    return Path(__file__).resolve().parents[1]


def _jsonable_value(x):
    if isinstance(x, Path):
        return str(x)

    if isinstance(x, tuple):
        return [_jsonable_value(v) for v in x]

    if isinstance(x, list):
        return [_jsonable_value(v) for v in x]

    if isinstance(x, dict):
        return {str(k): _jsonable_value(v) for k, v in x.items()}

    if isinstance(x, (str, int, float, bool)) or x is None:
        return x

    try:
        json.dumps(x)
        return x
    except Exception:
        return str(x)


def config_to_dict(config):
    data = {}

    try:
        raw = vars(config)
    except Exception:
        raw = {}

    for k, v in raw.items():
        if k.startswith("_"):
            continue

        data[k] = _jsonable_value(v)

    return data


def dict_to_config(data):
    return SimpleNamespace(**data)


def msdata_to_matchms_spectrum(msdata, idx, config):
    """
    Convert one spectrum from msdata to matchms.Spectrum.

    Important for MS2DeepScore:
    The model may require metadata['ionmode'] to be either:
        'positive'
    or:
        'negative'
    """

    try:
        from matchms import Spectrum
    except ImportError as e:
        raise ImportError(
            "matchms is required for MS2DeepScore. "
            "Install it inside the MS2DeepScore Python environment."
        ) from e

    mzs, intensities = get_spectrum_peaks(msdata, idx, config)
    precursor_mz = get_precursor_mz(msdata, idx)

    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if mzs.ndim != 1:
        mzs = mzs.reshape(-1)

    if intensities.ndim != 1:
        intensities = intensities.reshape(-1)

    if mzs.shape[0] != intensities.shape[0]:
        raise ValueError(
            f"m/z and intensity length mismatch at index {idx}: "
            f"{mzs.shape[0]} vs {intensities.shape[0]}"
        )

    finite_mask = np.isfinite(mzs) & np.isfinite(intensities)

    mzs = mzs[finite_mask]
    intensities = intensities[finite_mask]

    positive_intensity_mask = intensities > 0

    mzs = mzs[positive_intensity_mask]
    intensities = intensities[positive_intensity_mask]

    if mzs.size == 0:
        raise ValueError(f"Spectrum at index {idx} has no valid peaks.")

    ion_mode = get_ion_mode_for_spectrum(msdata, idx, config)

    metadata = {
        "ionmode": ion_mode,
        "ion_mode": ion_mode,
        "polarity": ion_mode,
    }

    if np.isfinite(precursor_mz):
        precursor_mz = float(precursor_mz)

        metadata["precursor_mz"] = precursor_mz
        metadata["parent_mass"] = precursor_mz

    spectrum = Spectrum(
        mz=mzs,
        intensities=intensities,
        metadata=metadata,
    )

    return spectrum



def load_ms2deepscore_model_object(model_path):
    """
    Load MS2DeepScore model.

    Important:
    compute_embedding_array() expects the raw SiameseSpectralModel-like object.
    Do NOT wrap it with MS2DeepScore(raw_model) here.
    """

    from ms2deepscore.models import load_model

    model_path = str(model_path)

    try:
        raw_model = load_model(model_path, allow_legacy=True)
    except TypeError:
        # For older ms2deepscore versions that do not support allow_legacy.
        raw_model = load_model(model_path)

    return raw_model


def ms2deepscore_spectra_to_vectors(
    model,
    spectra,
    device=None,
    progress_bar=False,
):
    """
    Convert a batch of matchms Spectrum objects to MS2DeepScore embedding vectors.

    Current confirmed API:
        compute_embedding_array(model, spectra, datatype='numpy', device=None, progress_bar=True)
    """

    from ms2deepscore.models import compute_embedding_array

    emb = compute_embedding_array(
        model,
        spectra,
        datatype="numpy",
        device=device,
        progress_bar=progress_bar,
    )

    emb = np.asarray(emb)

    if emb.ndim != 2:
        raise ValueError(f"Unexpected MS2DeepScore embedding array shape: {emb.shape}")

    return emb.astype(np.float32, copy=False)


def ms2deepscore_spectrum_to_vector(
    model,
    spectrum,
    device=None,
):
    """
    Convert one matchms Spectrum object to one MS2DeepScore embedding vector.
    """

    emb = ms2deepscore_spectra_to_vectors(
        model=model,
        spectra=[spectrum],
        device=device,
        progress_bar=False,
    )

    if emb.shape[0] != 1:
        raise ValueError(f"Expected one embedding, got shape: {emb.shape}")

    return emb[0].astype(np.float32, copy=False)


def find_first_valid_embedding_from_cache(
    model,
    cache,
    n,
    device=None,
    max_probe=1000,
):
    """
    Find first valid embedding using preloaded RAM cache.
    """

    probe_n = min(int(max_probe), int(n))

    print(
        f"[ms2deepscore] Probing first valid spectrum from RAM cache "
        f"within first {probe_n} spectra...",
        flush=True,
    )

    last_error = None

    for i in range(probe_n):
        try:
            spectrum = cached_spectrum_to_matchms_spectrum(cache, i)

            emb = ms2deepscore_spectrum_to_vector(
                model=model,
                spectrum=spectrum,
                device=device,
            )

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if emb.size <= 0:
                raise ValueError("Empty embedding vector.")

            print(
                f"[ms2deepscore] First valid spectrum index = {i}, "
                f"embedding_dim = {emb.size}",
                flush=True,
            )

            return i, emb

        except Exception as e:
            last_error = e

            if i < 20 or i == probe_n - 1:
                print(
                    f"[ms2deepscore] probe failed at index {i}: {repr(e)}",
                    flush=True,
                )
            elif i == 20:
                print(
                    "[ms2deepscore] further probe failures suppressed...",
                    flush=True,
                )

    raise RuntimeError(
        "Could not find any valid spectrum for MS2DeepScore embedding "
        f"within first {probe_n} cached spectra. Last error: {repr(last_error)}"
    )



def encode_batch_with_fallback_from_cache(
    model,
    cache,
    indices,
    embedding_dim,
    config,
    device=None,
):
    """
    Encode one batch using preloaded RAM cache.

    No repeated access to original msdata_lib.
    """

    indices = list(indices)
    batch_size = len(indices)

    if batch_size == 0:
        return np.zeros((0, embedding_dim), dtype=np.float32), 0

    t0 = time.time()

    spectra, valid_positions, build_failed = build_matchms_spectra_batch_from_cache(
        cache=cache,
        indices=indices,
    )

    t1 = time.time()

    batch_embeddings = np.zeros((batch_size, embedding_dim), dtype=np.float32)

    if len(spectra) == 0:
        return batch_embeddings, build_failed

    try:
        emb_valid = ms2deepscore_spectra_to_vectors(
            model=model,
            spectra=spectra,
            device=device,
            progress_bar=False,
        )

        t2 = time.time()

        emb_valid = np.asarray(emb_valid, dtype=np.float32)

        if emb_valid.ndim != 2:
            raise ValueError(f"Batch embedding has invalid shape: {emb_valid.shape}")

        if emb_valid.shape[0] != len(spectra):
            raise ValueError(
                f"Batch embedding row mismatch: expected {len(spectra)}, "
                f"got {emb_valid.shape[0]}"
            )

        if emb_valid.shape[1] != embedding_dim:
            raise ValueError(
                f"Embedding dimension mismatch: expected {embedding_dim}, "
                f"got {emb_valid.shape[1]}"
            )

        for row, pos in enumerate(valid_positions):
            batch_embeddings[pos, :] = emb_valid[row]

        t3 = time.time()

        show_timing = bool(
            get_config_value(config, "ms2deepscore_show_timing", True)
        )

        if show_timing and (
            indices[0] == 0
            or (indices[-1] + 1) % max(batch_size * 20, 10000) == 0
        ):
            print(
                f"[ms2deepscore][timing] batch {indices[0]}:{indices[-1] + 1}, "
                f"build_from_cache={t1 - t0:.3f}s, "
                f"model_compute={t2 - t1:.3f}s, "
                f"assign={t3 - t2:.3f}s, "
                f"valid={len(spectra)}/{batch_size}",
                flush=True,
            )

        return batch_embeddings, build_failed

    except Exception as batch_error:
        print(
            f"[ms2deepscore] WARNING: batch encoding failed for "
            f"{indices[0]}:{indices[-1] + 1}; fallback to single. "
            f"Error: {repr(batch_error)}",
            flush=True,
        )

    failed = build_failed

    # fallback 也复用已经从 cache 构造好的 spectra，不再碰原始 msdata
    for spectrum, pos, idx in zip(
        spectra,
        valid_positions,
        [indices[p] for p in valid_positions],
    ):
        try:
            emb = ms2deepscore_spectrum_to_vector(
                model=model,
                spectrum=spectrum,
                device=device,
            )

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if emb.size != embedding_dim:
                raise ValueError(
                    f"Single embedding dimension mismatch at index {idx}: "
                    f"expected {embedding_dim}, got {emb.size}"
                )

            batch_embeddings[pos, :] = emb

        except Exception as e:
            failed += 1

            print(
                f"[ms2deepscore] WARNING: failed at index {idx}; "
                f"filled with zeros. Error: {repr(e)}",
                flush=True,
            )

    return batch_embeddings, failed

def build_ms2deepscore_runtime_context(config):
    """
    Build runtime context.

    For speed, use a fixed ionmode by default.
    """

    default_ion_mode = get_config_value(
        config,
        "ms2deepscore_default_ion_mode",
        get_config_value(config, "ms2deepscore_ion_mode", "positive"),
    )

    default_ion_mode = normalize_ion_mode(default_ion_mode)

    if default_ion_mode not in {"positive", "negative"}:
        raise ValueError(
            "ms2deepscore_default_ion_mode must be 'positive' or 'negative'. "
            f"Got: {default_ion_mode}"
        )

    return SimpleNamespace(
        default_ion_mode=default_ion_mode,
    )


def build_ms2deepscore_embeddings_direct(config, limit=None,mona_hdf5_path=None):
    """
    Build MS2DeepScore embeddings with multi-core CPU preprocessing
    and main-process GPU encoding.

    Design:
    - CPU workers independently load MoNA.
    - CPU workers build matchms.Spectrum chunks.
    - Main process receives chunks and runs GPU embedding.
    """

    model_path = get_config_value(config, "ms2deepscore_model_path", None)

    if model_path is None or str(model_path).strip() == "":
        raise ValueError(
            "config.ms2deepscore_model_path is required when embedding_model='ms2deepscore'."
        )

    model_path = Path(model_path).expanduser()

    if not model_path.is_absolute():
        model_path = (get_project_root() / model_path).resolve()
    else:
        model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"MS2DeepScore model file does not exist: {model_path}")

    batch_size = int(get_config_value(config, "ms2deepscore_batch_size", 2048))
    batch_size = max(batch_size, 1)

    num_workers = int(
        get_config_value(
            config,
            "ms2deepscore_num_workers",
            max(1, min((os.cpu_count() or 4) - 1, 8)),
        )
    )
    num_workers = max(num_workers, 1)

    max_in_flight = int(
        get_config_value(
            config,
            "ms2deepscore_max_in_flight_chunks",
            max(2, num_workers * 2),
        )
    )
    max_in_flight = max(max_in_flight, 1)

    device = get_config_value(config, "ms2deepscore_device", "cuda")

    if device is not None and str(device).strip() == "":
        device = None

    peak_dtype_name = str(
        get_config_value(config, "ms2deepscore_preload_peak_dtype", "float32")
    ).lower()

    runtime_context = build_ms2deepscore_runtime_context(config)

    print("[ms2deepscore] Loading MoNA library in main process for length check...", flush=True)
    try:
        msdata_lib, existing_embs_lib, lib_cols = io_utils.load_mona_library(_MS2DS_WORKER_CONFIG)
    except Exception:
        msdata_lib, existing_embs_lib, lib_cols = load_mona_library(r"D:\亚结构注释\mona_processed\mona_chimera_dataset_equal_200k.hdf5")

    print(f"[ms2deepscore] Available columns: {lib_cols}", flush=True)

    n_total = get_msdata_length(msdata_lib)
    n = n_total if limit is None else min(int(limit), n_total)

    try:
        if hasattr(msdata_lib, "close"):
            msdata_lib.close()
    except Exception:
        pass

    msdata_lib = None

    print(f"[ms2deepscore] n_spectra = {n}", flush=True)
    print(f"[ms2deepscore] batch_size = {batch_size}", flush=True)
    print(f"[ms2deepscore] num_cpu_workers = {num_workers}", flush=True)
    print(f"[ms2deepscore] max_in_flight_chunks = {max_in_flight}", flush=True)
    print(f"[ms2deepscore] device = {device}", flush=True)
    print(f"[ms2deepscore] default ionmode = {runtime_context.default_ion_mode}", flush=True)
    print(f"[ms2deepscore] peak_dtype = {peak_dtype_name}", flush=True)

    if n <= 0:
        return cast_embedding_dtype(np.zeros((0, 0), dtype=np.float32), config)

    print(f"[ms2deepscore] Loading model: {model_path}", flush=True)

    model = load_ms2deepscore_model_object(model_path)

    config_dict = config_to_dict(config)

    chunk_ranges = [
        (start, min(start + batch_size, n))
        for start in range(0, n, batch_size)
    ]

    total_chunks = len(chunk_ranges)

    print(
        f"[ms2deepscore] total_chunks = {total_chunks}",
        flush=True,
    )

    embeddings = None
    embedding_dim = None
    first_emb = None
    first_emb_global_index = None

    completed_spectra = 0
    completed_chunks = 0
    total_failed = 0

    start_time = time.time()
    last_report_time = start_time

    # Windows 下明确使用 spawn，避免 fork/HDF5/CUDA 状态污染。
    mp_context = mp.get_context("spawn")

    def submit_next_chunk(executor, chunk_iter, futures):
        try:
            task = next(chunk_iter)
        except StopIteration:
            return False

        future = executor.submit(_ms2ds_build_spectrum_chunk, task)
        futures[future] = task
        return True

    chunk_iter = iter(chunk_ranges)
    futures = {}

    with ProcessPoolExecutor(
        max_workers=num_workers,
        mp_context=mp_context,
        initializer=_ms2ds_cpu_worker_init,
        initargs=(
            config_dict,
            runtime_context.default_ion_mode,
            peak_dtype_name,
        ),
    ) as executor:

        for _ in range(min(max_in_flight, total_chunks)):
            submit_next_chunk(executor, chunk_iter, futures)

        while futures:
            done, _ = wait(
                futures.keys(),
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                task = futures.pop(future)
                task_start, task_end = task

                t_chunk0 = time.time()

                try:
                    chunk = future.result()
                except Exception as e:
                    total_failed += task_end - task_start
                    completed_spectra += task_end - task_start
                    completed_chunks += 1

                    print(
                        f"[ms2deepscore] ERROR: CPU worker failed for chunk "
                        f"{task_start}:{task_end}. Filled with zeros later. "
                        f"Error: {repr(e)}",
                        flush=True,
                    )

                    submit_next_chunk(executor, chunk_iter, futures)
                    continue

                t_chunk1 = time.time()

                chunk_start = int(chunk["start"])
                chunk_end = int(chunk["end"])
                chunk_size = chunk_end - chunk_start

                # 第一次拿到有效 spectra 时，推断 embedding_dim 并分配最终矩阵。
                if embedding_dim is None:
                    embedding_dim, first_emb = infer_embedding_dim_from_chunk(
                        model=model,
                        chunk=chunk,
                        device=device,
                    )

                    if embedding_dim is not None:
                        embeddings = np.zeros((n, embedding_dim), dtype=np.float32)

                        # 找 first_emb 对应的 global index。
                        # infer 用的是 chunk["spectra"] 里的第一条可用谱图。
                        # 这里简单再扫一次，找到第一条可成功编码的 valid position。
                        for spectrum, pos in zip(
                            chunk["spectra"],
                            chunk["valid_positions"],
                        ):
                            try:
                                test_emb = ms2deepscore_spectrum_to_vector(
                                    model=model,
                                    spectrum=spectrum,
                                    device=device,
                                )
                                test_emb = np.asarray(test_emb, dtype=np.float32).reshape(-1)

                                if test_emb.size == embedding_dim:
                                    first_emb = test_emb
                                    first_emb_global_index = chunk_start + int(pos)
                                    embeddings[first_emb_global_index, :] = first_emb
                                    break
                            except Exception:
                                pass

                        print(
                            f"[ms2deepscore] embedding_dim = {embedding_dim}, "
                            f"first_valid_index = {first_emb_global_index}",
                            flush=True,
                        )

                # 如果当前 chunk 没有任何有效 spectra，且 embedding_dim 还未知，先跳过。
                # 后面 embeddings 分配后，这些位置自然保持零。
                if embedding_dim is not None:
                    t_gpu0 = time.time()

                    batch_embeddings, failed = encode_spectrum_chunk_on_gpu(
                        model=model,
                        chunk=chunk,
                        embedding_dim=embedding_dim,
                        config=config,
                        device=device,
                    )

                    t_gpu1 = time.time()

                    embeddings[chunk_start:chunk_end, :] = batch_embeddings

                    if (
                        first_emb_global_index is not None
                        and chunk_start <= first_emb_global_index < chunk_end
                    ):
                        embeddings[first_emb_global_index, :] = first_emb

                    total_failed += int(failed)

                else:
                    # 还没有找到任何可用于推断维度的谱图。
                    total_failed += int(chunk["failed"])
                    t_gpu0 = time.time()
                    t_gpu1 = t_gpu0

                completed_spectra += chunk_size
                completed_chunks += 1

                t_chunk2 = time.time()

                show_timing = bool(
                    get_config_value(config, "ms2deepscore_show_timing", True)
                )

                now = time.time()

                should_report = (
                    completed_chunks == 1
                    or completed_chunks == total_chunks
                    or completed_spectra >= n
                    or now - last_report_time >= 10.0
                )

                if should_report:
                    elapsed = now - start_time
                    speed = completed_spectra / elapsed if elapsed > 0 else 0.0

                    valid_in_chunk = len(chunk["spectra"])

                    print(
                        f"[ms2deepscore] completed_chunks={completed_chunks}/{total_chunks}, "
                        f"completed_spectra={completed_spectra}/{n} "
                        f"({100.0 * completed_spectra / n:.2f}%), "
                        f"last_chunk={chunk_start}:{chunk_end}, "
                        f"valid_in_chunk={valid_in_chunk}/{chunk_size}, "
                        f"failed_total={total_failed}, "
                        f"speed={speed:.1f} spectra/s",
                        flush=True,
                    )

                    last_report_time = now

                if show_timing and completed_chunks <= 5:
                    print(
                        f"[ms2deepscore][timing] chunk {chunk_start}:{chunk_end}, "
                        f"receive_result={t_chunk1 - t_chunk0:.3f}s, "
                        f"gpu_encode={t_gpu1 - t_gpu0:.3f}s, "
                        f"postprocess={t_chunk2 - t_gpu1:.3f}s",
                        flush=True,
                    )

                # 主进程处理完一个 chunk 后，继续提交新 chunk。
                submit_next_chunk(executor, chunk_iter, futures)

    if embeddings is None or embedding_dim is None:
        raise RuntimeError(
            "Could not infer MS2DeepScore embedding dimension. "
            "No valid spectrum was successfully encoded."
        )

    print("[ms2deepscore] L2 normalizing embeddings...", flush=True)

    embeddings = l2_normalize(embeddings)
    embeddings = cast_embedding_dtype(embeddings, config)

    elapsed = time.time() - start_time

    print(
        f"[ms2deepscore] finished. shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}, failed={total_failed}, "
        f"elapsed={elapsed:.1f}s",
        flush=True,
    )

    return embeddings




def get_ms2deepscore_python_path(config):
    python_path = get_config_value(config, "ms2deepscore_python_path", None)

    if python_path is None or str(python_path).strip() == "":
        return sys.executable

    python_path = Path(python_path).expanduser()

    if python_path.is_dir():
        candidate = python_path / "Scripts" / "python.exe"

        if candidate.exists():
            return str(candidate)

        candidate = python_path / "bin" / "python"

        if candidate.exists():
            return str(candidate)

    if not python_path.exists():
        raise FileNotFoundError(
            f"MS2DeepScore Python executable does not exist: {python_path}"
        )

    return str(python_path)


def run_ms2deepscore_subprocess(config, limit=None):
    python_exe = get_ms2deepscore_python_path(config)
    project_root = get_project_root()

    cache_dir = Path(
        get_config_value(
            config,
            "embedding_cache_dir",
            "outputs/embedding_cache",
        )
    ).expanduser()

    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    cache_dir = cache_dir.resolve()

    job_dir = cache_dir / "_ms2deepscore_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"job_{int(time.time())}_{os.getpid()}"
    config_json = (job_dir / f"{job_id}_config.json").resolve()
    output_npy = (job_dir / f"{job_id}_embeddings.npy").resolve()

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, ensure_ascii=False, indent=2)

    if not config_json.exists():
        raise FileNotFoundError(
            f"Failed to create MS2DeepScore subprocess config file: {config_json}"
        )

    script_path = Path(__file__).resolve()

    cmd = [
        str(python_exe),
        str(script_path),
        "--worker",
        "--config-json",
        str(config_json),
        "--output-npy",
        str(output_npy),
    ]

    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])

    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    module_dir = str(Path(__file__).resolve().parent)
    project_root_str = str(project_root)

    old_pythonpath = env.get("PYTHONPATH", "")

    if old_pythonpath:
        env["PYTHONPATH"] = (
            project_root_str
            + os.pathsep
            + module_dir
            + os.pathsep
            + old_pythonpath
        )
    else:
        env["PYTHONPATH"] = project_root_str + os.pathsep + module_dir

    print(f"[ms2deepscore] subprocess python = {python_exe}", flush=True)
    print(f"[ms2deepscore] subprocess script = {script_path}", flush=True)
    print(f"[ms2deepscore] subprocess config = {config_json}", flush=True)
    print(f"[ms2deepscore] subprocess output = {output_npy}", flush=True)
    print(f"[ms2deepscore] subprocess cwd = {project_root}", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"MS2DeepScore subprocess failed with exit code {return_code}."
        )

    if not output_npy.exists():
        raise FileNotFoundError(
            f"MS2DeepScore subprocess finished but output file was not created: {output_npy}"
        )

    embeddings = np.load(output_npy)

    keep_temp = bool(get_config_value(config, "ms2deepscore_keep_temp_files", False))

    if not keep_temp:
        try:
            config_json.unlink(missing_ok=True)
            output_npy.unlink(missing_ok=True)
        except Exception:
            pass

    return cast_embedding_dtype(embeddings, config)


def build_ms2deepscore_embeddings(msdata, config, limit=None):
    run_in_subprocess = bool(
        get_config_value(config, "ms2deepscore_run_in_subprocess", True)
    )

    if run_in_subprocess:
        return run_ms2deepscore_subprocess(config=config, limit=limit)

    return build_ms2deepscore_embeddings_direct(config=config, limit=limit)

def normalize_ion_mode(value):
    """
    Normalize ion mode / polarity value to 'positive' or 'negative'.

    MS2DeepScore metadata feature generator expects:
        ionmode == 'positive'
    or:
        ionmode == 'negative'
    """

    if value is None:
        return None

    if isinstance(value, bytes):
        try:
            value = value.decode("utf-8", errors="ignore")
        except Exception:
            value = str(value)

    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        value = value.reshape(-1)[0]

    if isinstance(value, (list, tuple)):
        if len(value) == 0:
            return None
        value = value[0]

    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")

    if isinstance(value, (int, float, np.integer, np.floating)):
        if not np.isfinite(value):
            return None

        if value > 0:
            return "positive"

        if value < 0:
            return "negative"

        return None

    text = str(value).strip().lower()

    if text == "" or text in {"none", "nan", "null", "unknown", "na", "n/a"}:
        return None

    text = (
        text.replace(" ", "")
        .replace("_", "")
        .replace("-", "")
        .replace("/", "")
    )

    positive_values = {
        "positive",
        "pos",
        "p",
        "+",
        "1",
        "+1",
        "esi+",
        "esi positive",
        "esipositive",
        "positiveionmode",
        "positiveion",
        "positivepolarity",
    }

    negative_values = {
        "negative",
        "neg",
        "n",
        "-",
        "-1",
        "esi-",
        "esi negative",
        "esinegative",
        "negativeionmode",
        "negativeion",
        "negativepolarity",
    }

    if text in positive_values:
        return "positive"

    if text in negative_values:
        return "negative"

    if "positive" in text or text.endswith("pos") or "esi+" in text:
        return "positive"

    if "negative" in text or text.endswith("neg") or "esi-" in text:
        return "negative"

    return None


def _safe_scalar_value(x):
    """
    Convert common one-element containers from pandas / h5py / numpy to scalar.
    """

    if x is None:
        return None

    if isinstance(x, bytes):
        return x.decode("utf-8", errors="ignore")

    if isinstance(x, np.ndarray):
        if x.size == 0:
            return None
        return _safe_scalar_value(x.reshape(-1)[0])

    if isinstance(x, (list, tuple)):
        if len(x) == 0:
            return None
        return _safe_scalar_value(x[0])

    try:
        # pandas scalar NA
        if hasattr(x, "item"):
            return x.item()
    except Exception:
        pass

    return x


def get_metadata_value_from_msdata(msdata, idx, keys):
    """
    Try to fetch metadata value from various msdata-like containers.

    This is intentionally defensive because load_mona_library() may return:
    - pandas DataFrame-like objects
    - AnnData-like objects
    - h5py-like objects
    - list of dicts
    - custom objects
    """

    # 1. AnnData-like: msdata.obs
    try:
        obs = getattr(msdata, "obs", None)

        if obs is not None:
            for key in keys:
                if key in obs.columns:
                    return _safe_scalar_value(obs.iloc[idx][key])
    except Exception:
        pass

    # 2. pandas DataFrame-like
    try:
        columns = getattr(msdata, "columns", None)

        if columns is not None:
            for key in keys:
                if key in columns:
                    return _safe_scalar_value(msdata.iloc[idx][key])
    except Exception:
        pass

    # 3. h5py-like or dict-like root datasets
    try:
        for key in keys:
            if key in msdata:
                return _safe_scalar_value(msdata[key][idx])
    except Exception:
        pass

    # 4. dict-like metadata container
    try:
        metadata = getattr(msdata, "metadata", None)

        if isinstance(metadata, dict):
            for key in keys:
                if key in metadata:
                    value = metadata[key]

                    try:
                        return _safe_scalar_value(value[idx])
                    except Exception:
                        return _safe_scalar_value(value)
    except Exception:
        pass

    # 5. list-like records
    try:
        item = msdata[idx]

        if isinstance(item, dict):
            for key in keys:
                if key in item:
                    return _safe_scalar_value(item[key])

            metadata = item.get("metadata", None)

            if isinstance(metadata, dict):
                for key in keys:
                    if key in metadata:
                        return _safe_scalar_value(metadata[key])
    except Exception:
        pass

    # 6. object attributes on one spectrum-like item
    try:
        item = msdata[idx]

        metadata = getattr(item, "metadata", None)

        if isinstance(metadata, dict):
            for key in keys:
                if key in metadata:
                    return _safe_scalar_value(metadata[key])

        for key in keys:
            if hasattr(item, key):
                return _safe_scalar_value(getattr(item, key))
    except Exception:
        pass

    return None


def get_ion_mode_for_spectrum(msdata, idx, config):
    """
    Get ion mode for one spectrum.

    Priority:
    1. Per-spectrum metadata from msdata
    2. Config-level default
    3. Fallback default, controlled by config
    """

    ion_mode_keys = [
        "ionmode",
        "ion_mode",
        "ionization_mode",
        "polarity",
        "polarity_mode",
        "scan_polarity",
        "charge_mode",
        "ms_ionmode",
        "precursor_ionmode",
    ]

    raw_value = get_metadata_value_from_msdata(msdata, idx, ion_mode_keys)
    ion_mode = normalize_ion_mode(raw_value)

    if ion_mode is not None:
        return ion_mode

    config_keys = [
        "ms2deepscore_ion_mode",
        "ms2deepscore_default_ion_mode",
        "ion_mode",
        "ionmode",
        "polarity",
    ]

    for key in config_keys:
        raw_value = get_config_value(config, key, None)
        ion_mode = normalize_ion_mode(raw_value)

        if ion_mode is not None:
            return ion_mode

    # 最后的兜底值。
    # 如果你的库是负离子模式，请把这里改成 "negative"，
    # 或者在 config 中设置 ms2deepscore_default_ion_mode = "negative"。
    fallback = get_config_value(config, "ms2deepscore_fallback_ion_mode", "positive")
    ion_mode = normalize_ion_mode(fallback)

    if ion_mode is None:
        raise ValueError(
            "Cannot determine ion mode. Please set config.ms2deepscore_default_ion_mode "
            "to 'positive' or 'negative'."
        )

    return ion_mode

def worker_main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--config-json", type=str, required=True)
    parser.add_argument("--output-npy", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    config_json = Path(args.config_json).expanduser()
    output_npy = Path(args.output_npy).expanduser()

    if not config_json.is_absolute():
        config_json = (Path.cwd() / config_json).resolve()
    else:
        config_json = config_json.resolve()

    if not output_npy.is_absolute():
        output_npy = (Path.cwd() / output_npy).resolve()
    else:
        output_npy = output_npy.resolve()

    print(f"[ms2deepscore-worker] cwd = {Path.cwd()}", flush=True)
    print(f"[ms2deepscore-worker] config_json = {config_json}", flush=True)
    print(f"[ms2deepscore-worker] output_npy = {output_npy}", flush=True)

    if not config_json.exists():
        raise FileNotFoundError(
            f"MS2DeepScore worker config json does not exist: {config_json}"
        )

    with open(config_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = dict_to_config(data)

    embeddings = build_ms2deepscore_embeddings_direct(
        config=config,
        limit=args.limit,
    )

    output_npy.parent.mkdir(parents=True, exist_ok=True)

    np.save(output_npy, embeddings)

    print(f"[ms2deepscore] saved subprocess embeddings to: {output_npy}", flush=True)
    print(f"[ms2deepscore] shape = {embeddings.shape}", flush=True)
    print(f"[ms2deepscore] dtype = {embeddings.dtype}", flush=True)
def build_ms2deepscore_embeddings_from_mzml_direct(
    mzml_path,
    output_path,
    config,
    *,
    ms_level=2,
    max_spectra=None,
    limit=None,
):
    """
    Build MS2DeepScore query embeddings from mzML.

    This function mirrors the library-side MS2DeepScore logic:

    Library side:
        1. load MS2DeepScore model
        2. build matchms.Spectrum with fixed ionmode
        3. batch encode by compute_embedding_array(...)
        4. invalid spectra -> zero rows
        5. L2 normalize full matrix
        6. cast_embedding_dtype(...)

    mzML query side:
        1. load mzML spectra
        2. convert each mzML spectrum dict to matchms.Spectrum
        3. batch encode with fallback
        4. invalid spectra -> zero rows
        5. L2 normalize full matrix
        6. cast_embedding_dtype(...)
        7. save as .npz by save_mzml_embeddings_npz(...)
    """

    try:
        from mzml_input import load_mzml_spectra, save_mzml_embeddings_npz
    except ImportError:
        from .mzml_input import load_mzml_spectra, save_mzml_embeddings_npz

    try:
        from matchms import Spectrum
    except ImportError as e:
        raise ImportError(
            "matchms is required for MS2DeepScore mzML query embedding. "
            "Please run this in the MS2DeepScore environment."
        ) from e

    # ------------------------------------------------------------
    # 1. Resolve model path exactly like library-side direct builder
    # ------------------------------------------------------------
    model_path = get_config_value(config, "ms2deepscore_model_path", None)

    if model_path is None or str(model_path).strip() == "":
        raise ValueError(
            "config.ms2deepscore_model_path is required for "
            "build_ms2deepscore_embeddings_from_mzml(...)."
        )

    model_path = Path(model_path).expanduser()

    if not model_path.is_absolute():
        model_path = (get_project_root() / model_path).resolve()
    else:
        model_path = model_path.resolve()

    if not model_path.exists():
        raise FileNotFoundError(f"MS2DeepScore model file does not exist: {model_path}")

    device = get_config_value(config, "ms2deepscore_device", "cuda")

    if device is not None and str(device).strip() == "":
        device = None

    batch_size = int(get_config_value(config, "ms2deepscore_batch_size", 2048))
    batch_size = max(batch_size, 1)

    runtime_context = build_ms2deepscore_runtime_context(config)
    ionmode = runtime_context.default_ion_mode

    peak_dtype_name = str(
        get_config_value(config, "ms2deepscore_preload_peak_dtype", "float32")
    ).lower()

    if peak_dtype_name in {"float64", "double"}:
        peak_dtype = np.float64
    else:
        peak_dtype = np.float32

    print(f"[ms2deepscore-mzML] mzML path = {mzml_path}", flush=True)
    print(f"[ms2deepscore-mzML] output   = {output_path}", flush=True)
    print(f"[ms2deepscore-mzML] model    = {model_path}", flush=True)
    print(f"[ms2deepscore-mzML] device   = {device}", flush=True)
    print(f"[ms2deepscore-mzML] ionmode  = {ionmode}", flush=True)
    print(f"[ms2deepscore-mzML] batch_size = {batch_size}", flush=True)

    # ------------------------------------------------------------
    # 2. Load mzML spectra
    # ------------------------------------------------------------
    spectra_raw = load_mzml_spectra(
        mzml_path,
        ms_level=ms_level,
        max_spectra=max_spectra,
        dtype=peak_dtype,
    )

    if limit is not None:
        spectra_raw = spectra_raw[: int(limit)]

    n = len(spectra_raw)

    print(f"[ms2deepscore-mzML] n_spectra = {n}", flush=True)

    if n <= 0:
        embeddings = np.zeros((0, 0), dtype=np.float32)

        saved_path = save_mzml_embeddings_npz(
            output_path,
            embeddings,
            spectra_raw,
            mzml_path=mzml_path,
            method="ms2deepscore",
            extra={
                "model_path": str(model_path),
                "device": str(device),
                "ionmode": ionmode,
                "batch_size": batch_size,
                "n_spectra": n,
            },
        )

        print(f"[ms2deepscore-mzML] Saved empty embeddings to: {saved_path}", flush=True)
        return embeddings

    # ------------------------------------------------------------
    # 3. Load model
    # ------------------------------------------------------------
    print(f"[ms2deepscore-mzML] Loading model: {model_path}", flush=True)
    model = load_ms2deepscore_model_object(model_path)

    # ------------------------------------------------------------
    # 4. Local converter: mzML dict -> matchms.Spectrum
    # ------------------------------------------------------------
    def mzml_dict_to_matchms_spectrum(spec, local_index):
        mzs = np.asarray(spec.get("mz", []), dtype=peak_dtype).reshape(-1)
        intensities = np.asarray(spec.get("intensity", []), dtype=peak_dtype).reshape(-1)

        if mzs.shape[0] != intensities.shape[0]:
            raise ValueError(
                f"m/z and intensity length mismatch at mzML index {local_index}: "
                f"{mzs.shape[0]} vs {intensities.shape[0]}"
            )

        mask = (
            np.isfinite(mzs)
            & np.isfinite(intensities)
            & (intensities > 0)
        )

        mzs = mzs[mask]
        intensities = intensities[mask]

        if mzs.size == 0:
            raise ValueError(f"mzML spectrum index {local_index} has no valid peaks.")

        precursor_mz = spec.get("precursor_mz", np.nan)

        try:
            precursor_mz = float(precursor_mz)
        except Exception:
            precursor_mz = np.nan

        metadata = {
            "ionmode": ionmode,
            "ion_mode": ionmode,
            "polarity": ionmode,
        }

        if np.isfinite(precursor_mz):
            metadata["precursor_mz"] = precursor_mz
            metadata["parent_mass"] = precursor_mz

        return Spectrum(
            mz=mzs,
            intensities=intensities,
            metadata=metadata,
        )

    # ------------------------------------------------------------
    # 5. Probe first valid spectrum to infer embedding_dim
    # ------------------------------------------------------------
    embedding_dim = None
    first_valid_index = None
    first_valid_embedding = None

    max_probe = int(get_config_value(config, "ms2deepscore_mzml_max_probe", 1000))
    max_probe = max(1, min(max_probe, n))

    print(
        f"[ms2deepscore-mzML] Probing first valid spectrum within first {max_probe} spectra...",
        flush=True,
    )

    last_probe_error = None

    for i in range(max_probe):
        try:
            spectrum = mzml_dict_to_matchms_spectrum(spectra_raw[i], i)

            emb = ms2deepscore_spectrum_to_vector(
                model=model,
                spectrum=spectrum,
                device=device,
            )

            emb = np.asarray(emb, dtype=np.float32).reshape(-1)

            if emb.size <= 0:
                raise ValueError("Empty MS2DeepScore embedding.")

            embedding_dim = int(emb.size)
            first_valid_index = int(i)
            first_valid_embedding = emb

            print(
                f"[ms2deepscore-mzML] first_valid_index = {first_valid_index}, "
                f"embedding_dim = {embedding_dim}",
                flush=True,
            )

            break

        except Exception as e:
            last_probe_error = e

            if i < 20 or i == max_probe - 1:
                print(
                    f"[ms2deepscore-mzML] probe failed at index {i}: {repr(e)}",
                    flush=True,
                )
            elif i == 20:
                print(
                    "[ms2deepscore-mzML] further probe failures suppressed...",
                    flush=True,
                )

    if embedding_dim is None:
        raise RuntimeError(
            "Could not infer MS2DeepScore embedding dimension from mzML query spectra. "
            f"Last error: {repr(last_probe_error)}"
        )

    embeddings = np.zeros((n, embedding_dim), dtype=np.float32)

    if first_valid_index is not None and first_valid_embedding is not None:
        embeddings[first_valid_index, :] = first_valid_embedding

    # ------------------------------------------------------------
    # 6. Batch encode all mzML spectra
    # ------------------------------------------------------------
    total_failed = 0
    completed = 0
    t0 = time.time()

    show_timing = bool(get_config_value(config, "ms2deepscore_show_timing", True))

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)

        t_batch0 = time.time()

        batch_spectra = []
        valid_positions = []
        build_failed = 0

        for pos, idx in enumerate(range(start, end)):
            try:
                spectrum = mzml_dict_to_matchms_spectrum(spectra_raw[idx], idx)
                batch_spectra.append(spectrum)
                valid_positions.append(pos)

            except Exception as e:
                build_failed += 1

                if build_failed <= 5:
                    print(
                        f"[ms2deepscore-mzML] WARNING: failed to build Spectrum "
                        f"at index {idx}: {repr(e)}",
                        flush=True,
                    )

        if build_failed > 5:
            print(
                f"[ms2deepscore-mzML] WARNING: suppressed {build_failed - 5} more "
                f"Spectrum-build failures in batch {start}:{end}",
                flush=True,
            )

        batch_embeddings = np.zeros((end - start, embedding_dim), dtype=np.float32)

        if len(batch_spectra) > 0:
            try:
                emb_valid = ms2deepscore_spectra_to_vectors(
                    model=model,
                    spectra=batch_spectra,
                    device=device,
                    progress_bar=False,
                )

                emb_valid = np.asarray(emb_valid, dtype=np.float32)

                if emb_valid.ndim != 2:
                    raise ValueError(
                        f"Batch embedding has invalid shape: {emb_valid.shape}"
                    )

                if emb_valid.shape[0] != len(batch_spectra):
                    raise ValueError(
                        f"Batch embedding row mismatch: expected {len(batch_spectra)}, "
                        f"got {emb_valid.shape[0]}"
                    )

                if emb_valid.shape[1] != embedding_dim:
                    raise ValueError(
                        f"Embedding dimension mismatch: expected {embedding_dim}, "
                        f"got {emb_valid.shape[1]}"
                    )

                for row, pos in enumerate(valid_positions):
                    batch_embeddings[pos, :] = emb_valid[row]

                failed = build_failed

            except Exception as batch_error:
                print(
                    f"[ms2deepscore-mzML] WARNING: batch encoding failed for "
                    f"{start}:{end}; fallback to single. Error: {repr(batch_error)}",
                    flush=True,
                )

                failed = build_failed

                for spectrum, pos in zip(batch_spectra, valid_positions):
                    global_idx = start + int(pos)

                    try:
                        emb = ms2deepscore_spectrum_to_vector(
                            model=model,
                            spectrum=spectrum,
                            device=device,
                        )

                        emb = np.asarray(emb, dtype=np.float32).reshape(-1)

                        if emb.size != embedding_dim:
                            raise ValueError(
                                f"Single embedding dimension mismatch at index {global_idx}: "
                                f"expected {embedding_dim}, got {emb.size}"
                            )

                        batch_embeddings[pos, :] = emb

                    except Exception as e:
                        failed += 1

                        print(
                            f"[ms2deepscore-mzML] WARNING: failed at index {global_idx}; "
                            f"filled with zeros. Error: {repr(e)}",
                            flush=True,
                        )

        else:
            failed = build_failed

        embeddings[start:end, :] = batch_embeddings

        # 保留 probe 得到的首个 embedding，避免它被某些 batch fallback 差异覆盖。
        if (
            first_valid_index is not None
            and first_valid_embedding is not None
            and start <= first_valid_index < end
        ):
            embeddings[first_valid_index, :] = first_valid_embedding

        total_failed += int(failed)
        completed += end - start

        t_batch1 = time.time()

        elapsed = time.time() - t0
        speed = completed / elapsed if elapsed > 0 else 0.0

        print(
            f"[ms2deepscore-mzML] completed={completed}/{n} "
            f"({100.0 * completed / n:.2f}%), "
            f"batch={start}:{end}, "
            f"valid_in_batch={len(batch_spectra)}/{end - start}, "
            f"failed_total={total_failed}, "
            f"speed={speed:.1f} spectra/s",
            flush=True,
        )

        if show_timing:
            print(
                f"[ms2deepscore-mzML][timing] batch {start}:{end}, "
                f"elapsed={t_batch1 - t_batch0:.3f}s",
                flush=True,
            )

    # ------------------------------------------------------------
    # 7. Normalize + cast exactly like library-side builder
    # ------------------------------------------------------------
    print("[ms2deepscore-mzML] L2 normalizing embeddings...", flush=True)

    embeddings = l2_normalize(embeddings)
    embeddings = cast_embedding_dtype(embeddings, config)

    elapsed = time.time() - t0

    print(
        f"[ms2deepscore-mzML] finished. shape={embeddings.shape}, "
        f"dtype={embeddings.dtype}, failed={total_failed}, elapsed={elapsed:.1f}s",
        flush=True,
    )

    # ------------------------------------------------------------
    # 8. Save query embeddings npz
    # ------------------------------------------------------------
    saved_path = save_mzml_embeddings_npz(
        output_path,
        embeddings,
        spectra_raw,
        mzml_path=mzml_path,
        method="ms2deepscore",
        extra={
            "model_path": str(model_path),
            "device": str(device),
            "ionmode": ionmode,
            "batch_size": batch_size,
            "embedding_dim": int(embedding_dim),
            "n_spectra": int(n),
            "failed": int(total_failed),
            "ms_level": int(ms_level),
            "max_spectra": None if max_spectra is None else int(max_spectra),
            "limit": None if limit is None else int(limit),
        },
    )

    print(f"[ms2deepscore-mzML] Saved embeddings to: {saved_path}", flush=True)

    return embeddings
def run_ms2deepscore_mzml_subprocess(
    mzml_path,
    output_path,
    config,
    *,
    ms_level=2,
    max_spectra=None,
    limit=None,
):
    """
    Run MS2DeepScore mzML query embedding in a separate Python environment.

    This mirrors run_ms2deepscore_subprocess(...), but for mzML query files.

    Parent process:
        - does not import ms2deepscore
        - writes config json
        - launches this same script with --mzml-worker

    Child process:
        - runs inside ms2deepscore_python_path
        - imports ms2deepscore
        - builds mzML query embeddings
        - saves .npz
    """

    python_exe = get_ms2deepscore_python_path(config)
    project_root = get_project_root()

    mzml_path = Path(mzml_path).expanduser()
    output_path = Path(output_path).expanduser()

    if not mzml_path.is_absolute():
        mzml_path = (Path.cwd() / mzml_path).resolve()
    else:
        mzml_path = mzml_path.resolve()

    if not output_path.is_absolute():
        output_path = (Path.cwd() / output_path).resolve()
    else:
        output_path = output_path.resolve()

    if not mzml_path.exists():
        raise FileNotFoundError(f"mzML file does not exist: {mzml_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(
        get_config_value(
            config,
            "embedding_cache_dir",
            "outputs/embedding_cache",
        )
    ).expanduser()

    if not cache_dir.is_absolute():
        cache_dir = project_root / cache_dir

    cache_dir = cache_dir.resolve()

    job_dir = cache_dir / "_ms2deepscore_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)

    job_id = f"mzml_job_{int(time.time())}_{os.getpid()}"
    config_json = (job_dir / f"{job_id}_config.json").resolve()

    with open(config_json, "w", encoding="utf-8") as f:
        json.dump(config_to_dict(config), f, ensure_ascii=False, indent=2)

    if not config_json.exists():
        raise FileNotFoundError(
            f"Failed to create MS2DeepScore mzML subprocess config file: {config_json}"
        )

    script_path = Path(__file__).resolve()

    cmd = [
        str(python_exe),
        str(script_path),
        "--mzml-worker",
        "--config-json",
        str(config_json),
        "--mzml-path",
        str(mzml_path),
        "--output-npz",
        str(output_path),
        "--ms-level",
        str(int(ms_level)),
    ]

    if max_spectra is not None:
        cmd.extend(["--max-spectra", str(int(max_spectra))])

    if limit is not None:
        cmd.extend(["--limit", str(int(limit))])

    env = os.environ.copy()
    env.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

    module_dir = str(Path(__file__).resolve().parent)
    project_root_str = str(project_root)

    old_pythonpath = env.get("PYTHONPATH", "")

    pythonpath_parts = [
        project_root_str,
        module_dir,
    ]

    if old_pythonpath:
        pythonpath_parts.append(old_pythonpath)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath_parts)

    print(f"[ms2deepscore-mzML] subprocess python = {python_exe}", flush=True)
    print(f"[ms2deepscore-mzML] subprocess script = {script_path}", flush=True)
    print(f"[ms2deepscore-mzML] subprocess config = {config_json}", flush=True)
    print(f"[ms2deepscore-mzML] mzML path = {mzml_path}", flush=True)
    print(f"[ms2deepscore-mzML] subprocess output = {output_path}", flush=True)
    print(f"[ms2deepscore-mzML] subprocess cwd = {project_root}", flush=True)

    process = subprocess.Popen(
        cmd,
        cwd=str(project_root),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    assert process.stdout is not None

    for line in process.stdout:
        print(line, end="")

    return_code = process.wait()

    if return_code != 0:
        raise RuntimeError(
            f"MS2DeepScore mzML subprocess failed with exit code {return_code}."
        )

    if not output_path.exists():
        raise FileNotFoundError(
            f"MS2DeepScore mzML subprocess finished but output file was not created: "
            f"{output_path}"
        )

    keep_temp = bool(get_config_value(config, "ms2deepscore_keep_temp_files", False))

    if not keep_temp:
        try:
            config_json.unlink(missing_ok=True)
        except Exception:
            pass

    data = np.load(output_path, allow_pickle=True)

    if "embeddings" not in data:
        raise KeyError(
            f"MS2DeepScore mzML output npz has no 'embeddings' key: {output_path}. "
            f"Available keys: {list(data.keys())}"
        )

    embeddings = data["embeddings"]

    print(
        f"[ms2deepscore-mzML] loaded subprocess output: "
        f"shape={embeddings.shape}, dtype={embeddings.dtype}",
        flush=True,
    )

    return embeddings

def build_ms2deepscore_embeddings_from_mzml(
    mzml_path,
    output_path,
    config,
    *,
    ms_level=2,
    max_spectra=None,
    limit=None,
):
    """
    Public mzML query builder.

    By default, run MS2DeepScore in a separate Python environment,
    because ms2deepscore often depends on a different Python / package stack.

    Config key:
        ms2deepscore_run_in_subprocess = True
        ms2deepscore_python_path = path/to/ms2deepscore/python.exe
    """

    run_in_subprocess = bool(
        get_config_value(config, "ms2deepscore_run_in_subprocess", True)
    )

    if run_in_subprocess:
        return run_ms2deepscore_mzml_subprocess(
            mzml_path=mzml_path,
            output_path=output_path,
            config=config,
            ms_level=ms_level,
            max_spectra=max_spectra,
            limit=limit,
        )

    return build_ms2deepscore_embeddings_from_mzml_direct(
        mzml_path=mzml_path,
        output_path=output_path,
        config=config,
        ms_level=ms_level,
        max_spectra=max_spectra,
        limit=limit,
    )
def worker4query():
    """
    Worker for mzML query embedding.

    This worker is separate from worker_main().
    It is used only for MS2DeepScore query embedding from mzML files.

    Parent process launches this script with:
        --mzml-worker
        --config-json
        --mzml-path
        --output-npz
    """

    parser = argparse.ArgumentParser(
        description="MS2DeepScore mzML query embedding worker"
    )

    parser.add_argument("--mzml-worker", action="store_true")
    parser.add_argument("--query-worker", action="store_true")

    parser.add_argument("--config-json", type=str, required=True)
    parser.add_argument("--mzml-path", type=str, required=True)
    parser.add_argument("--output-npz", type=str, required=True)

    parser.add_argument("--ms-level", type=int, default=2)
    parser.add_argument("--max-spectra", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    config_json = Path(args.config_json).expanduser()
    mzml_path = Path(args.mzml_path).expanduser()
    output_npz = Path(args.output_npz).expanduser()

    if not config_json.is_absolute():
        config_json = (Path.cwd() / config_json).resolve()
    else:
        config_json = config_json.resolve()

    if not mzml_path.is_absolute():
        mzml_path = (Path.cwd() / mzml_path).resolve()
    else:
        mzml_path = mzml_path.resolve()

    if not output_npz.is_absolute():
        output_npz = (Path.cwd() / output_npz).resolve()
    else:
        output_npz = output_npz.resolve()

    print(f"[ms2deepscore-query-worker] cwd = {Path.cwd()}", flush=True)
    print(f"[ms2deepscore-query-worker] config_json = {config_json}", flush=True)
    print(f"[ms2deepscore-query-worker] mzml_path = {mzml_path}", flush=True)
    print(f"[ms2deepscore-query-worker] output_npz = {output_npz}", flush=True)
    print(f"[ms2deepscore-query-worker] ms_level = {args.ms_level}", flush=True)
    print(f"[ms2deepscore-query-worker] max_spectra = {args.max_spectra}", flush=True)
    print(f"[ms2deepscore-query-worker] limit = {args.limit}", flush=True)

    if not config_json.exists():
        raise FileNotFoundError(
            f"MS2DeepScore query worker config json does not exist: {config_json}"
        )

    if not mzml_path.exists():
        raise FileNotFoundError(
            f"mzML file does not exist: {mzml_path}"
        )

    with open(config_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    config = dict_to_config(data)

    output_npz.parent.mkdir(parents=True, exist_ok=True)

    build_ms2deepscore_embeddings_from_mzml_direct(
        mzml_path=mzml_path,
        output_path=output_npz,
        config=config,
        ms_level=args.ms_level,
        max_spectra=args.max_spectra,
        limit=args.limit,
    )

    if not output_npz.exists():
        raise FileNotFoundError(
            f"MS2DeepScore query worker finished but output npz was not created: "
            f"{output_npz}"
        )

    try:
        data = np.load(output_npz, allow_pickle=True)

        if "embeddings" in data:
            embeddings = data["embeddings"]
            print(
                f"[ms2deepscore-query-worker] saved npz: {output_npz}, "
                f"shape={embeddings.shape}, dtype={embeddings.dtype}",
                flush=True,
            )
        else:
            print(
                f"[ms2deepscore-query-worker] saved npz: {output_npz}, "
                f"keys={list(data.keys())}",
                flush=True,
            )

    except Exception as e:
        print(
            f"[ms2deepscore-query-worker] WARNING: output npz exists but could not "
            f"be inspected. Error: {repr(e)}",
            flush=True,
        )

    print("[ms2deepscore-query-worker] done.", flush=True)

if __name__ == "__main__":
    if "--mzml-worker" in sys.argv or "--query-worker" in sys.argv:
        worker4query()
    else:
        worker_main()
