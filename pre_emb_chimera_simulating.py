import argparse
import hashlib
import json
import time
from dataclasses import dataclass, asdict
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
import typing as T
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from dreams.utils import data as du
from dreams.utils import dformats
from dreams.definitions import DREAMS_EMBEDDING, SPECTRUM
from dreams.api import PreTrainedModel
from dreams.models.dreams.dreams import DreaMS as DreaMSModel
from dreams.models.heads.heads import FineTuningHead
from utils import l2_normalize
import h5py
import numpy as np
from pathlib import Path

def resolve_spectra_input_for_dreams(spectra):
    """
    Resolve input for local DreaMS encoding.

    Accepted:
        - str / Path
        - dreams.utils.data.MSData
        - H5MSDataLite with .hdf5_pth or .path
    """
    if isinstance(spectra, du.MSData):
        return spectra

    if isinstance(spectra, (str, Path)):
        return Path(spectra)

    if hasattr(spectra, "hdf5_pth"):
        return Path(spectra.hdf5_pth)

    if hasattr(spectra, "path"):
        return Path(spectra.path)

    raise TypeError(
        "Unsupported spectra input for DreaMS local encoding. "
        f"Got type={type(spectra)}. "
        "Expected str / Path / dreams.utils.data.MSData / H5MSDataLite with .hdf5_pth or .path."
    )

def dreams_predictions_local(
    model_ckpt: T.Union[PreTrainedModel, FineTuningHead, DreaMSModel, Path, str],
    spectra: T.Union[Path, str, du.MSData, object],
    model_cls=None,
    batch_size=32,
    progress_bar=True,
    n_highest_peaks=None,
    title="",
    logger_pth=None,
    store_preds=False,
    **msdata_kwargs,
):
    """
    Local replacement of dreams_predictions.

    Difference from original DreaMS version:
        - accepts H5MSDataLite-like object;
        - if input has .pth, converts it to Path first;
        - then uses du.MSData.load(...) exactly like DreaMS.
    """

    # ------------------------------------------------------------------
    # Load pre-trained model
    # ------------------------------------------------------------------
    if not isinstance(model_ckpt, PreTrainedModel):
        if isinstance(model_ckpt, str):
            if "/" in model_ckpt or "\\" in model_ckpt:
                model_ckpt = Path(model_ckpt)
            else:
                title = model_ckpt

        if isinstance(model_ckpt, str):
            model_ckpt = PreTrainedModel.from_name(model_ckpt)

        elif isinstance(model_ckpt, Path):
            model_ckpt = PreTrainedModel.from_ckpt(
                model_ckpt,
                model_cls,
                n_highest_peaks,
            )

        else:
            model_ckpt = PreTrainedModel(
                model_ckpt,
                n_highest_peaks,
            )

    # ------------------------------------------------------------------
    # Initialize spectrum preprocessing
    # ------------------------------------------------------------------
    spec_preproc = du.SpectrumPreprocessor(
        dformat=dformats.DataFormatA(),
        n_highest_peaks=model_ckpt.n_highest_peaks,
    )

    # ------------------------------------------------------------------
    # Load a dataset of spectra
    # ------------------------------------------------------------------
    spectra = resolve_spectra_input_for_dreams(spectra)

    if not isinstance(spectra, du.MSData):
        if isinstance(spectra, str):
            spectra = Path(spectra)

        print(
            f"[dreams-local] Loading spectra with du.MSData.load:\n"
            f"  spectra={spectra}\n"
            f"  mode={'a' if store_preds else 'r'}",
            flush=True,
        )

        msdata = du.MSData.load(
            spectra,
            mode="a" if store_preds else "r",
            **msdata_kwargs,
        )

    else:
        msdata = spectra

        if msdata.mode != "a" and store_preds:
            raise ValueError(
                "Adding new columns is allowed only in append mode. "
                "Initialize msdata as `MSData(..., mode='a')` to add new columns."
            )

    spectra_dataset = msdata.to_torch_dataset(spec_preproc)

    dataloader = DataLoader(
        spectra_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
    )

    # ------------------------------------------------------------------
    # Setup logger writing progress to a file
    # ------------------------------------------------------------------
    if logger_pth:
        from dreams.utils import io

        logger = io.setup_logger(logger_pth)
        tqdm_logger = io.TqdmToLogger(logger)
    else:
        tqdm_logger = None

    # ------------------------------------------------------------------
    # Compute predictions
    # ------------------------------------------------------------------
    model = model_ckpt.model

    if not title:
        title = "DreaMS_prediction"

    num_samples = len(spectra_dataset)

    if num_samples == 0:
        raise ValueError("[dreams-local] Empty spectra dataset.")

    first_batch = next(iter(dataloader))

    with torch.inference_mode():
        first_pred = model(
            first_batch[SPECTRUM].to(
                device=model.device,
                dtype=model.dtype,
            )
        )

    output_shape = first_pred.shape[1:]

    preds = torch.zeros(
        (num_samples, *output_shape),
        dtype=model.dtype,
    )

    progress_bar_obj = tqdm(
        total=num_samples,
        desc="Computing " + title.replace("_", " ") + "s",
        disable=not progress_bar,
        file=tqdm_logger if logger_pth else None,
    )

    start_idx = 0

    for batch in dataloader:
        with torch.inference_mode():
            pred = model(
                batch[SPECTRUM].to(
                    device=model.device,
                    dtype=model.dtype,
                )
            )

            cur_batch_size = pred.shape[0]

            preds[start_idx:start_idx + cur_batch_size] = pred.cpu()

            start_idx += cur_batch_size
            progress_bar_obj.update(cur_batch_size)

    progress_bar_obj.close()

    preds = preds.numpy()

    if store_preds:
        msdata.add_column(title, preds)

    return preds
def dreams_embeddings_local(
    spectra,
    batch_size=32,
    progress_bar=True,
    logger_pth=None,
    store_embs=False,
    **msdata_kwargs,
):
    """
    Local replacement of dreams_embeddings.

    Accepts:
        - Path / str
        - du.MSData
        - H5MSDataLite-like object with .pth
    """
    return dreams_predictions_local(
        DREAMS_EMBEDDING,
        spectra,
        batch_size=batch_size,
        progress_bar=progress_bar,
        logger_pth=logger_pth,
        store_preds=store_embs,
        **msdata_kwargs,
    )
def force_recompute_dreams_embeddings_for_preencode(
    msdata,
    config,
    limit=None,
):
    """
    Force real DreaMS embedding computation for pre-encoding.

    This function:
        - accepts your H5MSDataLite object;
        - does not call build_dreams_embeddings();
        - does not use existing_dreams_embeddings;
        - calls local rewritten dreams_embeddings_local();
        - returns normalized embeddings.
    """
    batch_size = int(getattr(config, "dreams_batch_size", 32))
    use_float32 = bool(getattr(config, "use_float32", True))

    print(
        f"[dreams-force] Start real DreaMS encoding.\n"
        f"  input_type={type(msdata)}\n"
        f"  batch_size={batch_size}\n"
        f"  limit={limit}",
        flush=True,
    )

    embs = dreams_embeddings_local(
        msdata,
        batch_size=batch_size,
        progress_bar=True,
        store_embs=False,
    )

    embs = np.asarray(embs, dtype=np.float32)

    if limit is not None:
        embs = embs[: int(limit)]

    if embs.ndim != 2:
        raise ValueError(
            f"[dreams-force] Expected 2D embeddings, got shape={embs.shape}"
        )

    if embs.shape[0] == 0:
        raise ValueError("[dreams-force] Got zero embedding rows.")

    if embs.shape[1] == 0:
        raise ValueError("[dreams-force] Got zero embedding dimension.")

    finite_rows = np.isfinite(embs).all(axis=1)
    nan_rows = np.isnan(embs).any(axis=1)
    inf_rows = np.isinf(embs).any(axis=1)

    clean = np.nan_to_num(
        embs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    norms_raw = np.linalg.norm(clean, axis=1)
    zero_norm_rows = norms_raw == 0

    print(
        f"[dreams-force/raw] "
        f"shape={embs.shape}, dtype={embs.dtype}, "
        f"finite_rows={int(finite_rows.sum())}/{embs.shape[0]}, "
        f"nan_rows={int(nan_rows.sum())}, "
        f"inf_rows={int(inf_rows.sum())}, "
        f"zero_norm_rows={int(zero_norm_rows.sum())}, "
        f"norm_min={float(np.nanmin(norms_raw))}, "
        f"norm_median={float(np.nanmedian(norms_raw))}, "
        f"norm_max={float(np.nanmax(norms_raw))}",
        flush=True,
    )

    if int(finite_rows.sum()) != embs.shape[0]:
        raise ValueError(
            "[dreams-force] DreaMS produced NaN/Inf embeddings. Refusing to save."
        )

    if int(zero_norm_rows.sum()) == embs.shape[0]:
        raise ValueError(
            "[dreams-force] DreaMS produced all-zero embeddings. Refusing to save."
        )

    embs = l2_normalize(embs)

    clean_norm = np.nan_to_num(
        embs,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    norms = np.linalg.norm(clean_norm, axis=1)

    print(
        f"[dreams-force/normalized] "
        f"shape={embs.shape}, dtype={embs.dtype}, "
        f"finite_rows={int(np.isfinite(embs).all(axis=1).sum())}/{embs.shape[0]}, "
        f"nan_rows={int(np.isnan(embs).any(axis=1).sum())}, "
        f"inf_rows={int(np.isinf(embs).any(axis=1).sum())}, "
        f"zero_norm_rows={int((norms == 0).sum())}, "
        f"norm_min={float(np.nanmin(norms))}, "
        f"norm_median={float(np.nanmedian(norms))}, "
        f"norm_max={float(np.nanmax(norms))}",
        flush=True,
    )

    if use_float32:
        embs = embs.astype(np.float32, copy=False)

    return embs


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

SUPPORTED_EMBEDDING_MODELS = (
    "dreams",
    "spec2vec",
    "ms2deepscore",
    "binned",
    "neutral_loss_binned",
    "msbert",
    "specemb",
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


def make_embedding_config_fingerprint(
    config,
    split_name="library",
    limit=None,
    source_path_override=None,
):
    """
    Build a short fingerprint for the embedding cache.

    The hash includes:
    - dataset split name
    - source path
    - embedding model
    - model-relevant configuration
    - limit

    This prevents different datasets or different model settings from
    accidentally sharing the same cache file.
    """

    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    source_path_for_hash = (
        str(source_path_override)
        if source_path_override is not None
        else str(get_config_value(config, "mona_hdf5_path", ""))
    )

    parts = {
        "split_name": split_name,
        "embedding_model": model,
        "limit": limit,

        # dtype / memory-relevant settings
        "use_float32": get_config_value(config, "use_float32", False),
        "embedding_dtype": str(get_config_value(config, "embedding_dtype", "")),
        "binned_dtype": str(get_config_value(config, "binned_dtype", "")),

        # source paths
        "source_path": source_path_for_hash,
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


def get_embedding_cache_paths(
    config,
    split_name="library",
    limit=None,
    cache_dir_override=None,
    cache_prefix=None,
    source_path_override=None,
):
    """
    Resolve embedding cache and metadata paths.

    Examples:
    - split_name="library", cache_prefix=None
      -> library_specemb_xxxxxx.npy

    - split_name="simulated_chimera", cache_prefix="sim_chimera"
      -> sim_chimera_specemb_xxxxxx.npy
    """

    from pathlib import Path

    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    if cache_dir_override is None:
        cache_dir = Path(get_config_value(config, "embedding_cache_dir", "outputs/embedding_cache"))
    else:
        cache_dir = Path(cache_dir_override)

    cache_dir.mkdir(parents=True, exist_ok=True)

    config_hash = make_embedding_config_fingerprint(
        config=config,
        split_name=split_name,
        limit=limit,
        source_path_override=source_path_override,
    )

    if cache_prefix is not None and str(cache_prefix).strip():
        prefix = str(cache_prefix).strip()
    else:
        if limit is None:
            prefix = str(split_name)
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


def build_embeddings_from_msdata(
    msdata,
    config,
    limit=None,
    existing_dreams_embeddings=None,
):
    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    if model not in SUPPORTED_EMBEDDING_MODELS:
        raise ValueError(
            f"Unsupported embedding_model: {model}. "
            f"Supported models: {SUPPORTED_EMBEDDING_MODELS}"
        )

    print(f"[embedding] model = {model}")

    if model == "dreams":
        return force_recompute_dreams_embeddings_for_preencode(
            msdata=msdata,
            config=config,
            limit=limit,
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


def preencode_mona_library(
    config,
    limit=None,
    force=False,
    split_name="library",
    cache_dir=None,
    cache_prefix=None,
    source_path=None,
):
    """
    Pre-encode spectra loaded by load_mona_library(config).

    This function is now dataset-split aware. It can be reused for:
    - original MoNA library
    - simulated chimera library
    - held-out library
    - simulated query set

    Parameters
    ----------
    config:
        PipelineConfig or similar config object.
    limit:
        Encode only the first N spectra.
    force:
        Recompute even if cache exists.
    split_name:
        Dataset split name, used in hash and default filename prefix.
    cache_dir:
        Optional custom output directory for embedding cache.
    cache_prefix:
        Optional custom filename prefix.
    source_path:
        Optional source data path override. Currently this overrides
        config.mona_hdf5_path before calling load_mona_library(config).
    """

    model = str(get_config_value(config, "embedding_model", "dreams")).lower()

    if source_path is not None:
        try:
            config.mona_hdf5_path = str(source_path)
        except Exception:
            pass

    effective_source_path = (
        str(source_path)
        if source_path is not None
        else str(get_config_value(config, "mona_hdf5_path", ""))
    )

    cache_path, meta_path, config_hash = get_embedding_cache_paths(
        config=config,
        split_name=split_name,
        limit=limit,
        cache_dir_override=cache_dir,
        cache_prefix=cache_prefix,
        source_path_override=effective_source_path,
    )

    if cache_path.exists() and meta_path.exists() and not force:
        print(f"[cache] Found existing embedding cache: {cache_path}")

        embeddings = np.load(cache_path, mmap_mode="r")

        print_embedding_summary(embeddings, name=f"{split_name}_{model}")

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        cache_info = EmbeddingCacheInfo(**meta)

        return EmbeddingResult(
            model_name=model,
            embeddings=embeddings,
            cache_info=cache_info,
        )

    print(f"[data] Loading spectra for split={split_name}...")
    print(f"[data] source_path = {effective_source_path}")

    msdata_lib, existing_embs_lib, lib_cols = load_mona_library(config)

    print(f"[data] Available columns: {lib_cols}")
    print(f"[data] Spectra count: {get_msdata_length(msdata_lib)}")

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

    print_embedding_summary(embeddings, name=f"{split_name}_{model}")

    cache_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[cache] Saving embeddings to: {cache_path}")
    np.save(cache_path, embeddings)

    cache_info = EmbeddingCacheInfo(
        model_name=model,
        split_name=str(split_name),
        cache_path=str(cache_path),
        meta_path=str(meta_path),
        n_spectra=int(embeddings.shape[0]),
        embedding_dim=int(embeddings.shape[1]),
        dtype=str(embeddings.dtype),
        created_at=now_string(),
        elapsed_seconds=float(elapsed),
        source_path=effective_source_path,
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
        description="Pre-encode MoNA or MoNA-like library embeddings for chimera pipeline."
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
        default=True,
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
        default=10,
        help="Number of workers for binned / neutral_loss_binned static multiprocessing.",
    )

    parser.add_argument(
        "--split-name",
        type=str,
        default="simulated_chimera",
        help=(
            "Name of the dataset split to encode. "
            "Examples: library, simulated_chimera, simulated_query, heldout_library."
        ),
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=r'D:\亚结构注释\for_git\chimera_pipeline\outputs/mona_chimera_dataset_equal_200k_random',
        help=(
            "Custom embedding cache directory. "
            "If not set, config.embedding_cache_dir will be used."
        ),
    )

    parser.add_argument(
        "--cache-prefix",
        type=str,
        default=None,
        help=(
            "Custom cache filename prefix. "
            "If not set, split-name or split-name_limitN will be used."
        ),
    )

    parser.add_argument(
        "--source-path",
        type=str,
        default=r"D:\亚结构注释\mona_processed\mona_chimera_dataset_equal_200k_random.hdf5",
        help=(
            "Optional source data path override. "
            "Currently this overrides config.mona_hdf5_path and is included in the cache hash."
        ),
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

    if args.cache_dir is not None:
        config.embedding_cache_dir = args.cache_dir

    if args.source_path is not None:
        config.mona_hdf5_path = args.source_path

    if args.force:
        config.force_recompute_embeddings = True

    preencode_mona_library(
        config=config,
        limit=args.limit,
        force=args.force or get_config_value(config, "force_recompute_embeddings", False),
        split_name=args.split_name,
        cache_dir=args.cache_dir,
        cache_prefix=args.cache_prefix,
        source_path=args.source_path,
    )


if __name__ == "__main__":
    main()
