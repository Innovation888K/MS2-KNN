import os

os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

from dataclasses import dataclass
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import io_utils
import h5py
import numpy as np
from tqdm import tqdm


from io_utils import get_columns, get_value, get_precursor_mz
from utils import l2_normalize



@dataclass
class Spec2VecLiteDocument:
    words: list
    weights: object = None


_SPEC2VEC_WORKER_MSDATA = None
_SPEC2VEC_WORKER_CONFIG = None

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
def get_project_root():
    return Path(__file__).resolve().parents[1]


def get_default_model_dir():
    model_dir = get_project_root() / "model"
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def get_config_value(config, name, default=None):
    return getattr(config, name, default)


def get_msdata_length(msdata):
    try:
        return len(msdata)
    except Exception as e:
        raise TypeError(
            "Cannot determine length of MSData object. "
            "Expected msdata to support len(msdata)."
        ) from e


def cast_embedding_dtype(embeddings, config):
    embeddings = np.asarray(embeddings)

    if get_config_value(config, "use_float32", False):
        return embeddings.astype(np.float32, copy=False)

    return embeddings.astype(float, copy=False)


def infer_ion_mode_from_msdata(msdata, idx):
    cols = get_columns(msdata)

    candidate_cols = [
        "adduct",
        "ADDUCT",
        "precursor_adduct",
        "library_adduct",
        "ionmode",
        "ion_mode",
        "polarity",
        "charge",
    ]

    values = []

    for col in candidate_cols:
        if col not in cols:
            continue

        try:
            v = get_value(msdata, col, idx, default="")
        except Exception:
            v = ""

        if v is not None:
            values.append(str(v))

    text = " ".join(values).lower().strip()

    if text == "":
        return "unknown"

    negative_tokens = [
        "negative",
        "neg",
        "[m-h]-",
        "m-h",
        "-h",
        "]-",
        "-",
    ]

    positive_tokens = [
        "positive",
        "pos",
        "[m+h]+",
        "m+h",
        "m+na",
        "m+k",
        "+h",
        "]+",
        "+",
    ]

    for token in negative_tokens:
        if token in text:
            return "negative"

    for token in positive_tokens:
        if token in text:
            return "positive"

    return "unknown"


def get_spec2vec_default_paths():
    model_dir = get_default_model_dir()

    positive_path = model_dir / "spec2vec_mona_positive.model"
    negative_path = model_dir / "spec2vec_mona_negative.model"

    return positive_path, negative_path


def get_spec2vec_model_paths(config):
    default_positive_path, default_negative_path = get_spec2vec_default_paths()

    positive_path = get_config_value(config, "spec2vec_positive_model_path", None)
    negative_path = get_config_value(config, "spec2vec_negative_model_path", None)

    if positive_path is None:
        positive_path = default_positive_path
    else:
        positive_path = Path(positive_path)

    if negative_path is None:
        negative_path = default_negative_path
    else:
        negative_path = Path(negative_path)

    return positive_path, negative_path


def clean_peaks(mzs, intensities, config):
    mzs = np.asarray(mzs, dtype=float)
    intensities = np.asarray(intensities, dtype=float)

    if mzs.ndim != 1:
        mzs = mzs.reshape(-1)

    if intensities.ndim != 1:
        intensities = intensities.reshape(-1)

    n = min(len(mzs), len(intensities))
    mzs = mzs[:n]
    intensities = intensities[:n]

    mask = (
        np.isfinite(mzs)
        & np.isfinite(intensities)
        & (mzs > 0)
        & (intensities > 0)
    )

    mzs = mzs[mask]
    intensities = intensities[mask]

    if len(mzs) == 0:
        return mzs, intensities

    top_n = get_config_value(config, "embedding_top_n_peaks", 200)

    if top_n is not None and int(top_n) > 0 and len(mzs) > int(top_n):
        order = np.argsort(intensities)[::-1][: int(top_n)]
        mzs = mzs[order]
        intensities = intensities[order]

    max_intensity = np.max(intensities)

    if max_intensity > 0:
        intensities = intensities / max_intensity

    intensity_power = get_config_value(config, "embedding_intensity_power", 0.5)

    if intensity_power is not None:
        intensities = intensities ** float(intensity_power)

    mz_order = np.argsort(mzs)
    mzs = mzs[mz_order]
    intensities = intensities[mz_order]

    return mzs, intensities


def get_spectrum_peaks(msdata, idx, config):
    cols = get_columns(msdata)

    peak_array_candidate_cols = get_config_value(
        config,
        "peak_array_candidate_cols",
        (
            "peaks",
            "spectrum",
            "peak_array",
            "mz_intensity_array",
            "mz_intensity",
        ),
    )

    mz_candidate_cols = get_config_value(
        config,
        "mz_candidate_cols",
        (
            "mz",
            "mzs",
            "mz_array",
            "mz_values",
            "peak_mz",
        ),
    )

    intensity_candidate_cols = get_config_value(
        config,
        "intensity_candidate_cols",
        (
            "intensity",
            "intensities",
            "intensity_array",
            "intensity_values",
            "peak_intensity",
        ),
    )

    for col in peak_array_candidate_cols:
        if col not in cols:
            continue

        arr = get_value(msdata, col, idx, default=None)

        if arr is None:
            continue

        arr = np.asarray(arr)

        if arr.ndim == 2 and arr.shape[1] >= 2:
            mzs = arr[:, 0]
            intensities = arr[:, 1]
            return clean_peaks(mzs, intensities, config)

    mz_col = None
    intensity_col = None

    for col in mz_candidate_cols:
        if col in cols:
            mz_col = col
            break

    for col in intensity_candidate_cols:
        if col in cols:
            intensity_col = col
            break

    if mz_col is not None and intensity_col is not None:
        mzs = get_value(msdata, mz_col, idx, default=None)
        intensities = get_value(msdata, intensity_col, idx, default=None)

        if mzs is not None and intensities is not None:
            return clean_peaks(mzs, intensities, config)

    raise KeyError(
        "Cannot read spectrum peaks from MSData. "
        f"Index={idx}. Available columns: {cols}"
    )


def msdata_to_matchms_spectrum(msdata, idx, config):
    try:
        from matchms import Spectrum
    except ImportError as e:
        raise ImportError(
            "matchms is required for this embedding model. "
            "Install with: pip install matchms"
        ) from e

    mzs, intensities = get_spectrum_peaks(msdata, idx, config)
    precursor_mz = get_precursor_mz(msdata, idx)

    metadata = {}

    if np.isfinite(precursor_mz):
        metadata["precursor_mz"] = float(precursor_mz)

    return Spectrum(
        mz=np.asarray(mzs, dtype=float),
        intensities=np.asarray(intensities, dtype=float),
        metadata=metadata,
    )


def spectrum_document_to_vector(doc, model, intensity_power=0.5):
    vector_size = int(model.vector_size)
    vec = np.zeros(vector_size, dtype=float)
    weight_sum = 0.0

    words = list(doc.words)
    weights = getattr(doc, "weights", None)

    if weights is None:
        weights = np.ones(len(words), dtype=float)
    else:
        weights = np.asarray(weights, dtype=float)

    weights = weights ** float(intensity_power)

    for word, weight in zip(words, weights):
        if word not in model.wv:
            continue

        vec += float(weight) * np.asarray(model.wv[word], dtype=float)
        weight_sum += float(weight)

    if weight_sum > 0:
        vec = vec / weight_sum

    return vec


def init_spec2vec_read_worker(config):
    global _SPEC2VEC_WORKER_MSDATA
    global _SPEC2VEC_WORKER_CONFIG

    _SPEC2VEC_WORKER_CONFIG = config
    try:
        msdata_lib, existing_embs_lib, lib_cols = io_utils.load_mona_library(config)
    except Exception as e:
        msdata_lib, existing_embs_lib, lib_cols = load_mona_library(r"D:\亚结构注释\mona_processed\mona_chimera_dataset.hdf5")
    _SPEC2VEC_WORKER_MSDATA = msdata_lib


def make_spec2vec_lite_document_read_worker(task):
    global _SPEC2VEC_WORKER_MSDATA
    global _SPEC2VEC_WORKER_CONFIG

    idx, n_decimals = task

    try:
        from matchms import Spectrum
        from spec2vec import SpectrumDocument

        msdata = _SPEC2VEC_WORKER_MSDATA
        config = _SPEC2VEC_WORKER_CONFIG

        if msdata is None:
            raise RuntimeError("Worker MSData is not initialized.")

        mzs, intensities = get_spectrum_peaks(msdata, idx, config)
        precursor_mz = get_precursor_mz(msdata, idx)
        ion_mode = infer_ion_mode_from_msdata(msdata, idx)

        if ion_mode == "unknown":
            ion_mode = "positive"

        metadata = {}

        if np.isfinite(precursor_mz):
            metadata["precursor_mz"] = float(precursor_mz)

        spectrum = Spectrum(
            mz=np.asarray(mzs, dtype=float),
            intensities=np.asarray(intensities, dtype=float),
            metadata=metadata,
        )

        doc = SpectrumDocument(
            spectrum,
            n_decimals=int(n_decimals),
        )

        words = list(doc.words)
        weights = getattr(doc, "weights", None)

        if weights is not None:
            weights = np.asarray(weights, dtype=float)

        lite_doc = Spec2VecLiteDocument(
            words=words,
            weights=weights,
        )

        return idx, lite_doc, ion_mode, None

    except Exception as e:
        return idx, None, "unknown", str(e)


def build_spec2vec_documents_parallel(msdata, config, n):
    n_decimals = int(get_config_value(config, "spec2vec_n_decimals", 2))

    default_workers = max(1, (os.cpu_count() or 4) - 2)

    document_workers = int(
        get_config_value(
            config,
            "spec2vec_document_workers",
            default_workers,
        )
    )

    max_pending = int(
        get_config_value(
            config,
            "spec2vec_document_max_pending",
            document_workers * 16,
        )
    )

    document_workers = max(1, document_workers)
    max_pending = max(document_workers, max_pending)

    print(f"[spec2vec] document_workers = {document_workers}")
    print(f"[spec2vec] document_max_pending = {max_pending}")
    print("[spec2vec] read + document construction are both running in worker processes")

    documents = [None] * n
    ion_modes = ["unknown"] * n

    positive_documents = []
    negative_documents = []

    n_failed_document = 0

    def handle_future(future, pbar):
        nonlocal n_failed_document

        try:
            idx, doc, ion_mode, error = future.result()
        except Exception as e:
            n_failed_document += 1
            tqdm.write(f"[spec2vec] WARNING: worker crashed: {e}")
            pbar.update(1)
            return

        if error is not None:
            n_failed_document += 1

            if isinstance(idx, int) and 0 <= idx < n:
                documents[idx] = None
                ion_modes[idx] = "unknown"

            tqdm.write(
                f"[spec2vec] WARNING: failed to build document "
                f"at index {idx}: {error}"
            )

            pbar.update(1)
            return

        documents[idx] = doc
        ion_modes[idx] = ion_mode

        if ion_mode == "positive":
            positive_documents.append(doc)
        elif ion_mode == "negative":
            negative_documents.append(doc)

        pbar.update(1)

    with ProcessPoolExecutor(
        max_workers=document_workers,
        initializer=init_spec2vec_read_worker,
        initargs=(config,),
    ) as executor:
        pending = set()

        with tqdm(
            total=n,
            desc="[spec2vec] Building documents",
            unit="spectra",
            dynamic_ncols=True,
        ) as pbar:

            for i in range(n):
                future = executor.submit(
                    make_spec2vec_lite_document_read_worker,
                    (
                        i,
                        n_decimals,
                    ),
                )

                pending.add(future)

                if len(pending) >= max_pending:
                    done, pending = wait(
                        pending,
                        return_when=FIRST_COMPLETED,
                    )

                    for finished in done:
                        handle_future(finished, pbar)

                    pbar.set_postfix(
                        {
                            "pos": len(positive_documents),
                            "neg": len(negative_documents),
                            "fail": n_failed_document,
                        }
                    )

            while pending:
                done, pending = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for finished in done:
                    handle_future(finished, pbar)

                pbar.set_postfix(
                    {
                        "pos": len(positive_documents),
                        "neg": len(negative_documents),
                        "fail": n_failed_document,
                    }
                )

    print(f"[spec2vec] document build finished.")
    print(f"[spec2vec] positive documents = {len(positive_documents)}")
    print(f"[spec2vec] negative documents = {len(negative_documents)}")
    print(f"[spec2vec] failed document = {n_failed_document}")

    return documents, ion_modes, positive_documents, negative_documents


def build_spec2vec_embeddings(msdata, config, limit=None):
    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError(
            "Spec2Vec dependencies are missing. "
            "Install with: pip install matchms spec2vec gensim"
        ) from e

    positive_model_path, negative_model_path = get_spec2vec_model_paths(config)

    force_recompute = bool(
        get_config_value(config, "force_recompute_embeddings", False)
    )

    n_total = get_msdata_length(msdata)
    n = n_total if limit is None else min(int(limit), n_total)

    print(f"[spec2vec] n_spectra = {n}")
    print(f"[spec2vec] positive_model_path = {positive_model_path}")
    print(f"[spec2vec] negative_model_path = {negative_model_path}")

    print("[spec2vec] Building SpectrumDocument objects with multiprocessing...")

    documents, ion_modes, positive_documents, negative_documents = (
        build_spec2vec_documents_parallel(
            msdata=msdata,
            config=config,
            n=n,
        )
    )

    n_positive = len(positive_documents)
    n_negative = len(negative_documents)

    print(f"[spec2vec] positive documents = {n_positive}")
    print(f"[spec2vec] negative documents = {n_negative}")

    if n_positive == 0 and n_negative == 0:
        raise RuntimeError("No valid SpectrumDocument objects were created.")

    positive_model_path.parent.mkdir(parents=True, exist_ok=True)
    negative_model_path.parent.mkdir(parents=True, exist_ok=True)

    def train_spec2vec_word2vec(docs, mode_name):
        if len(docs) == 0:
            raise RuntimeError(f"No documents available for Spec2Vec {mode_name} training.")

        sentences = [list(doc.words) for doc in docs]

        print(f"[spec2vec] Training {mode_name} Word2Vec model on {len(sentences)} spectra...")

        model = Word2Vec(
            sentences=sentences,
            vector_size=int(get_config_value(config, "spec2vec_vector_size", 300)),
            window=int(get_config_value(config, "spec2vec_window", 500)),
            min_count=int(get_config_value(config, "spec2vec_min_count", 1)),
            workers=int(get_config_value(config, "spec2vec_workers", 4)),
            epochs=int(get_config_value(config, "spec2vec_epochs", 20)),
            sg=1,
            negative=5,
        )

        return model

    if positive_model_path.exists() and not force_recompute:
        print(f"[spec2vec] Loading positive model: {positive_model_path}")
        positive_model = Word2Vec.load(str(positive_model_path))
    else:
        positive_model = train_spec2vec_word2vec(
            docs=positive_documents if n_positive > 0 else negative_documents,
            mode_name="positive",
        )
        positive_model.save(str(positive_model_path))
        print(f"[spec2vec] Saved positive model to: {positive_model_path}")

    if negative_model_path.exists() and not force_recompute:
        print(f"[spec2vec] Loading negative model: {negative_model_path}")
        negative_model = Word2Vec.load(str(negative_model_path))
    else:
        if n_negative > 0:
            negative_model = train_spec2vec_word2vec(
                docs=negative_documents,
                mode_name="negative",
            )
        else:
            print(
                "[spec2vec] WARNING: no negative documents found. "
                "Using positive model as negative fallback."
            )
            negative_model = positive_model

        negative_model.save(str(negative_model_path))
        print(f"[spec2vec] Saved negative model to: {negative_model_path}")

    positive_vector_size = int(positive_model.vector_size)
    negative_vector_size = int(negative_model.vector_size)

    if positive_vector_size != negative_vector_size:
        raise ValueError(
            "Positive and negative Spec2Vec models have different vector sizes: "
            f"{positive_vector_size} vs {negative_vector_size}"
        )

    vector_size = positive_vector_size

    embeddings = []

    print("[spec2vec] Converting documents to embeddings...")

    for i, doc in enumerate(
        tqdm(
            documents,
            desc="[spec2vec] Encoding embeddings",
            unit="spectra",
            dynamic_ncols=True,
        )
    ):
        if doc is None:
            emb = np.zeros(vector_size, dtype=float)
            embeddings.append(emb)
            continue

        ion_mode = ion_modes[i]

        if ion_mode == "negative":
            model = negative_model
        else:
            model = positive_model

        emb = spectrum_document_to_vector(
            doc=doc,
            model=model,
            intensity_power=get_config_value(
                config,
                "spec2vec_intensity_weighting_power",
                0.5,
            ),
        )

        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    embeddings = l2_normalize(embeddings)

    return cast_embedding_dtype(embeddings, config)

def _load_mzml_as_spec2vec_msdata(
    mzml_path,
    *,
    ms_level=2,
    min_peaks=1,
    max_spectra=None,
    dtype=np.float32,
):
    """
    Load mzML file and convert it into a lightweight msdata-like table.

    This function is designed for Spec2Vec inference only.

    Returned object is a pandas.DataFrame with columns:
        - scan_id
        - mz
        - intensity
        - precursor_mz
        - ionmode
        - ms_level

    These column names are compatible with get_spectrum_peaks(...)
    and infer_ion_mode_from_msdata(...).
    """

    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError(
            "Reading mzML requires pandas. Install it with: pip install pandas"
        ) from e

    try:
        from pyteomics import mzml
    except ImportError as e:
        raise ImportError(
            "Reading mzML requires pyteomics. Install it with: pip install pyteomics"
        ) from e

    mzml_path = Path(mzml_path).expanduser().resolve()

    if not mzml_path.exists():
        raise FileNotFoundError(f"mzML file not found: {mzml_path}")

    def _get_ms_level(spec):
        value = spec.get("ms level", spec.get("msLevel", None))

        try:
            return int(value)
        except Exception:
            return None

    def _get_scan_id(spec, idx):
        for key in ["id", "scan_id", "spectrum title", "title"]:
            value = spec.get(key, None)

            if value is not None:
                return str(value)

        return f"spectrum_{idx}"

    def _get_precursor_mz(spec):
        try:
            precursor_list = spec.get("precursorList", {})
            precursors = precursor_list.get("precursor", [])

            if len(precursors) > 0:
                precursor = precursors[0]

                selected_ion_list = precursor.get("selectedIonList", {})
                selected_ions = selected_ion_list.get("selectedIon", [])

                if len(selected_ions) > 0:
                    ion = selected_ions[0]

                    for key in [
                        "selected ion m/z",
                        "isolation window target m/z",
                        "precursor m/z",
                    ]:
                        if key in ion:
                            return float(ion[key])

                isolation_window = precursor.get("isolationWindow", {})

                for key in [
                    "isolation window target m/z",
                    "selected ion m/z",
                    "precursor m/z",
                ]:
                    if key in isolation_window:
                        return float(isolation_window[key])

        except Exception:
            pass

        for key in [
            "precursor_mz",
            "precursor m/z",
            "selected ion m/z",
            "isolation window target m/z",
        ]:
            try:
                if key in spec:
                    return float(spec[key])
            except Exception:
                pass

        return np.nan

    def _get_ionmode(spec):
        """
        Return positive / negative / unknown.

        mzML polarity information is not always present.
        """

        text_parts = []

        for key in [
            "positive scan",
            "negative scan",
            "scan polarity",
            "polarity",
            "ionmode",
            "ion_mode",
        ]:
            if key in spec:
                text_parts.append(str(key))
                text_parts.append(str(spec.get(key)))

        text = " ".join(text_parts).lower()

        if "negative scan" in text or "negative" in text or "neg" in text:
            return "negative"

        if "positive scan" in text or "positive" in text or "pos" in text:
            return "positive"

        return "unknown"

    rows = []

    with mzml.read(str(mzml_path)) as reader:
        for raw_idx, spec in enumerate(reader):
            current_ms_level = _get_ms_level(spec)

            if ms_level is not None and current_ms_level != int(ms_level):
                continue

            mzs = spec.get("m/z array", None)
            intensities = spec.get("intensity array", None)

            if mzs is None or intensities is None:
                continue

            mzs = np.asarray(mzs, dtype=dtype).reshape(-1)
            intensities = np.asarray(intensities, dtype=dtype).reshape(-1)

            n = min(len(mzs), len(intensities))
            mzs = mzs[:n]
            intensities = intensities[:n]

            mask = (
                np.isfinite(mzs)
                & np.isfinite(intensities)
                & (mzs > 0)
                & (intensities > 0)
            )

            mzs = mzs[mask]
            intensities = intensities[mask]

            if len(mzs) < int(min_peaks):
                continue

            rows.append(
                {
                    "scan_id": _get_scan_id(spec, raw_idx),
                    "mz": mzs,
                    "intensity": intensities,
                    "precursor_mz": _get_precursor_mz(spec),
                    "ionmode": _get_ionmode(spec),
                    "ms_level": current_ms_level,
                }
            )

            if max_spectra is not None and len(rows) >= int(max_spectra):
                break

    if len(rows) == 0:
        raise RuntimeError(
            f"No usable spectra were found in mzML file: {mzml_path}. "
            f"Check ms_level={ms_level}, centroiding, and peak arrays."
        )

    return pd.DataFrame(rows)


def _save_spec2vec_mzml_embeddings(
    output_path,
    embeddings,
    msdata,
    *,
    mzml_path,
    positive_model_path,
    negative_model_path,
):
    """
    Save mzML Spec2Vec embeddings with minimal metadata.
    """

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    scan_ids = (
        msdata["scan_id"].to_numpy(dtype=object)
        if "scan_id" in msdata.columns
        else np.asarray([], dtype=object)
    )

    precursor_mz = (
        msdata["precursor_mz"].to_numpy(dtype=np.float32)
        if "precursor_mz" in msdata.columns
        else np.asarray([], dtype=np.float32)
    )

    ionmode = (
        msdata["ionmode"].to_numpy(dtype=object)
        if "ionmode" in msdata.columns
        else np.asarray([], dtype=object)
    )

    np.savez_compressed(
        output_path,
        embeddings=np.asarray(embeddings),
        scan_ids=scan_ids,
        precursor_mz=precursor_mz,
        ionmode=ionmode,
        mzml_path=np.asarray(str(mzml_path)),
        method=np.asarray("spec2vec"),
        positive_model_path=np.asarray(str(positive_model_path)),
        negative_model_path=np.asarray(str(negative_model_path)),
    )

    return output_path


def build_spec2vec_embeddings_from_mzml(
    mzml_path,
    output_path,
    config,
    *,
    ms_level=2,
    max_spectra=None,
    limit=None,
    allow_missing_negative_model=True,
):
    """
    Inference-only Spec2Vec encoding for mzML.

    Important:
    This function does NOT call build_spec2vec_embeddings(...).
    It never trains Word2Vec models.
    """

    try:
        from gensim.models import Word2Vec
    except ImportError as e:
        raise ImportError(
            "Spec2Vec mzML inference requires gensim. "
            "Install with: pip install gensim"
        ) from e

    try:
        from mzml_input import load_mzml_spectra, save_mzml_embeddings_npz
    except ImportError:
        from .mzml_input import load_mzml_spectra, save_mzml_embeddings_npz

    positive_model_path, negative_model_path = get_spec2vec_model_paths(config)
    positive_model_path = Path(positive_model_path).expanduser().resolve()
    negative_model_path = Path(negative_model_path).expanduser().resolve()

    if not positive_model_path.exists():
        raise FileNotFoundError(
            "Positive Spec2Vec model is required for mzML inference. "
            f"Missing file: {positive_model_path}"
        )

    if not negative_model_path.exists() and not allow_missing_negative_model:
        raise FileNotFoundError(
            "Negative Spec2Vec model is required because "
            "allow_missing_negative_model=False. "
            f"Missing file: {negative_model_path}"
        )

    spectra = load_mzml_spectra(
        mzml_path,
        ms_level=ms_level,
        max_spectra=max_spectra,
        dtype=np.float32,
    )

    if limit is not None:
        spectra = spectra[: int(limit)]

    if len(spectra) == 0:
        raise RuntimeError("No spectra available for Spec2Vec mzML inference.")

    print(f"[spec2vec-mzML] n_spectra = {len(spectra)}")
    print(f"[spec2vec-mzML] positive_model_path = {positive_model_path}")
    print(f"[spec2vec-mzML] negative_model_path = {negative_model_path}")

    positive_model = Word2Vec.load(str(positive_model_path))

    if negative_model_path.exists():
        negative_model = Word2Vec.load(str(negative_model_path))
    else:
        print(
            "[spec2vec-mzML] WARNING: negative model missing. "
            "Using positive model as fallback."
        )
        negative_model = positive_model

    if int(positive_model.vector_size) != int(negative_model.vector_size):
        raise ValueError(
            "Positive and negative Spec2Vec models have different vector sizes: "
            f"{positive_model.vector_size} vs {negative_model.vector_size}"
        )

    vector_size = int(positive_model.vector_size)
    embeddings = []

    for spec in tqdm(
        spectra,
        desc="[spec2vec-mzML] Encoding",
        unit="spectra",
        dynamic_ncols=True,
    ):
        mzs, intensities = clean_peaks(
            spec["mz"],
            spec["intensity"],
            config,
        )

        if len(mzs) == 0:
            embeddings.append(np.zeros(vector_size, dtype=float))
            continue

        words = []
        weights = []

        mz_rounding_decimals = int(
            get_config_value(config, "spec2vec_mz_rounding_decimals", 2)
        )

        for mz, intensity in zip(mzs, intensities):
            word = f"peak@{round(float(mz), mz_rounding_decimals)}"
            words.append(word)
            weights.append(float(intensity))

        doc = Spec2VecLiteDocument(
            words=words,
            weights=np.asarray(weights, dtype=float),
        )

        ion_mode = spec.get("ionmode", "unknown")

        if ion_mode == "negative":
            model = negative_model
        else:
            model = positive_model

        emb = spectrum_document_to_vector(
            doc=doc,
            model=model,
            intensity_power=get_config_value(
                config,
                "spec2vec_intensity_weighting_power",
                0.5,
            ),
        )

        embeddings.append(emb)

    embeddings = np.vstack(embeddings)
    embeddings = l2_normalize(embeddings)
    embeddings = cast_embedding_dtype(embeddings, config)

    saved_path = save_mzml_embeddings_npz(
        output_path,
        embeddings,
        spectra,
        mzml_path=mzml_path,
        method="spec2vec",
        extra={
            "positive_model_path": str(positive_model_path),
            "negative_model_path": str(negative_model_path),
        },
    )

    print(f"[spec2vec-mzML] Saved embeddings to: {saved_path}")

    return embeddings

