import numpy as np
from io_utils import get_columns, get_value
from utils import l2_normalize
import os
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")
from dreams.utils.spectra import PeakListModifiedCosine
from dreams.utils.data import MSData
from dreams.api import dreams_embeddings
from dreams.definitions import *


def get_config_value(config, name, default=None):
    return getattr(config, name, default)
def _is_valid_openms_data_path(path_value):
    """
    Check whether OPENMS_DATA_PATH points to a plausible OpenMS share directory.

    Expected directory usually looks like:
        .../site-packages/pyopenms/share/OpenMS

    It commonly contains subdirectories such as:
        CHEMISTRY, CV, INI, MAPPING
    """

    if path_value is None:
        return False

    try:
        p = Path(path_value).expanduser().resolve()
    except Exception:
        return False

    if not p.exists() or not p.is_dir():
        return False

    marker_names = [
        "CHEMISTRY",
        "CV",
        "INI",
        "MAPPING",
        "examples",
    ]

    for name in marker_names:
        if (p / name).exists():
            return True

    return False


def _ensure_openms_data_path(openms_data_path=None, *, verbose=True):
    """
    Ensure OPENMS_DATA_PATH is valid before calling DreaMS MSData.load(...).

    This is necessary because DreaMS may use pyOpenMS internally.
    If OPENMS_DATA_PATH points to a broken path, OpenMS can terminate the process
    with a fatal error.
    """

    import os
    import sys
    import site
    from pathlib import Path

    # 1. Explicit user-provided path has highest priority.
    if openms_data_path is not None:
        openms_data_path = Path(openms_data_path).expanduser().resolve()

        if not _is_valid_openms_data_path(openms_data_path):
            raise RuntimeError(
                "Provided openms_data_path is not a valid OpenMS data directory:\n"
                f"  {openms_data_path}\n\n"
                "It should usually look like:\n"
                "  .../site-packages/pyopenms/share/OpenMS"
            )

        os.environ["OPENMS_DATA_PATH"] = str(openms_data_path)

        if verbose:
            print(
                f"[DreaMS-mzML] OPENMS_DATA_PATH set explicitly: "
                f"{openms_data_path}",
                flush=True,
            )

        return str(openms_data_path)

    # 2. Keep current env var only if it is valid.
    current = os.environ.get("OPENMS_DATA_PATH", None)

    if _is_valid_openms_data_path(current):
        if verbose:
            print(
                f"[DreaMS-mzML] Existing OPENMS_DATA_PATH is valid: {current}",
                flush=True,
            )

        return current

    if verbose and current:
        print(
            "[DreaMS-mzML] Existing OPENMS_DATA_PATH is invalid and will be ignored:\n"
            f"  {current}",
            flush=True,
        )

    # 3. Try to discover pyopenms installation path.
    candidates = []

    try:
        import pyopenms

        pyopenms_dir = Path(pyopenms.__file__).expanduser().resolve().parent
        candidates.extend(
            [
                pyopenms_dir / "share" / "OpenMS",
                pyopenms_dir.parent / "pyopenms" / "share" / "OpenMS",
            ]
        )
    except Exception:
        pass

    # 4. Common venv / conda locations.
    candidates.extend(
        [
            Path(sys.prefix) / "Lib" / "site-packages" / "pyopenms" / "share" / "OpenMS",
            Path(sys.prefix) / "lib" / "site-packages" / "pyopenms" / "share" / "OpenMS",
            Path(sys.prefix) / "Library" / "share" / "OpenMS",
            Path(sys.prefix) / "share" / "OpenMS",
        ]
    )

    # 5. site-packages locations.
    try:
        for sp in site.getsitepackages():
            candidates.append(Path(sp) / "pyopenms" / "share" / "OpenMS")
    except Exception:
        pass

    try:
        candidates.append(Path(site.getusersitepackages()) / "pyopenms" / "share" / "OpenMS")
    except Exception:
        pass

    # Remove duplicates while preserving order.
    seen = set()
    unique_candidates = []

    for c in candidates:
        try:
            c = Path(c).expanduser().resolve()
            key = str(c).lower()
        except Exception:
            continue

        if key in seen:
            continue

        seen.add(key)
        unique_candidates.append(c)

    for candidate in unique_candidates:
        if _is_valid_openms_data_path(candidate):
            os.environ["OPENMS_DATA_PATH"] = str(candidate)

            if verbose:
                print(
                    f"[DreaMS-mzML] OPENMS_DATA_PATH fixed: {candidate}",
                    flush=True,
                )

            return str(candidate)

    candidate_text = "\n".join(f"  - {c}" for c in unique_candidates)

    raise RuntimeError(
        "Could not find a valid OpenMS data directory for pyOpenMS.\n\n"
        "Tried candidates:\n"
        f"{candidate_text}\n\n"
        "Fix options:\n"
        "1. Reinstall pyopenms in the current environment:\n"
        "     pip install --force-reinstall pyopenms\n\n"
        "2. Or pass openms_data_path explicitly, for example:\n"
        "     build_dreams_embeddings_from_mzml(...,\n"
        "         openms_data_path=r'C:\\path\\to\\site-packages\\pyopenms\\share\\OpenMS')\n\n"
        "3. On Windows, avoid virtualenv paths with non-ASCII characters if OpenMS "
        "fails to handle them correctly."
    )


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


def load_dreams_embeddings_from_msdata(msdata, config, limit=None):
    cols = get_columns(msdata)

    candidate_cols = []

    try:
        from dreams.definitions import DREAMS_EMBEDDING

        candidate_cols.append(DREAMS_EMBEDDING)
    except Exception:
        pass

    candidate_cols.extend(
        [
            "DREAMS_EMBEDDING",
            "dreams_embedding",
            "DreaMS_embedding",
            "embedding",
            "embeddings",
        ]
    )

    emb_col = None

    for col in candidate_cols:
        if col in cols:
            emb_col = col
            break

    if emb_col is None:
        raise KeyError(
            "Cannot find DreaMS embedding column in MSData. "
            f"Available columns: {cols}"
        )

    n_total = get_msdata_length(msdata)
    n = n_total if limit is None else min(int(limit), n_total)

    embeddings = []

    for i in range(n):
        emb = get_value(msdata, emb_col, i, default=None)

        if emb is None:
            raise ValueError(f"Missing DreaMS embedding at index {i}")

        embeddings.append(np.asarray(emb, dtype=float).reshape(-1))

    embeddings = np.vstack(embeddings)
    embeddings = l2_normalize(embeddings)

    return cast_embedding_dtype(embeddings, config)


def build_dreams_embeddings(msdata, config, limit=None, existing_dreams_embeddings=None):
    if existing_dreams_embeddings is not None:
        embeddings = np.asarray(existing_dreams_embeddings)

        if limit is not None:
            embeddings = embeddings[: int(limit)]

        embeddings = l2_normalize(embeddings)
        return cast_embedding_dtype(embeddings, config)

    return load_dreams_embeddings_from_msdata(
        msdata=msdata,
        config=config,
        limit=limit,
    )

def build_dreams_embeddings_from_mzml(
    config
):
    from pathlib import Path
    in_pth = Path(config.query_mzml_path)
    msdata = MSData.load(in_pth)
    embs = dreams_embeddings(msdata)
    embs_array = np.vstack([np.asarray(e, dtype=np.float32).reshape(1, -1) for e in embs])
    output_path = r"out/example.dreams.npz"
    save_path=np.savez_compressed(
        output_path,
        embeddings=embs_array,
        mzml_path=np.asarray(str(in_pth)),
        method=np.asarray("dreams"),
        n_spectra=np.asarray(embs_array.shape[0], dtype=np.int64),
        embedding_dim=np.asarray(embs_array.shape[1], dtype=np.int64),
    )
    print(f"[dreams-mzML] Saved embeddings to: {save_path}")
    return embs_array