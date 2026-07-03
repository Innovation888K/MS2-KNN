#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Parallel generation of synthetic AB and ABC chimeric MS/MS spectra from MoNA HDF5.

This version defaults to:
    AB  = 10,000 spectra, mixed as 1:1
    ABC = 10,000 spectra, mixed as 1:1:1
    total = 20,000 synthetic chimeric spectra

Input format:
    spectrum:      (N, 2, 128) float32
    precursor_mz:  (N,) float64
    num_peaks:     (N,) int64
    smiles, inchikey, mona_id, name, etc.

Output root-level DreaMS-compatible HDF5:
    /spectrum
    /precursor_mz
    /num_peaks
    /smiles
    /inchikey
    /mona_id
    ...

Ground-truth chimera fields:
    /chimera_type
    /component_count
    /precursor_mz_list
    /mixing_weights
    /source_mona_idx
    /source_mona_id
    /source_inchikey
    /source_smiles
    /source_name
"""

import argparse
import os
import random
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import h5py
import numpy as np
from tqdm import tqdm


# ============================================================
# Global data for workers
# ============================================================

G = {}


def init_worker(shared_data, seed_base):
    """
    Initializer for each worker process.

    On Linux fork can share memory copy-on-write.
    On Windows spawn will pickle data to workers.
    Since user said memory is not a concern, this is acceptable,
    but for very large data Windows may still take time to start.
    """
    global G
    G = shared_data

    pid = os.getpid()
    seed = int(seed_base + pid) % (2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)


# ============================================================
# Utilities
# ============================================================

def decode_array(ds):
    """
    Decode an HDF5 string/object dataset into numpy object array of str.
    """
    arr = ds[:]
    out = np.empty(arr.shape[0], dtype=object)

    for i, x in enumerate(arr):
        if isinstance(x, bytes):
            out[i] = x.decode("utf-8", errors="ignore")
        else:
            out[i] = str(x)

    return out


def normalize_intensity(intensity, mode="max"):
    intensity = np.asarray(intensity, dtype=np.float32)

    if intensity.size == 0:
        return intensity

    if mode == "max":
        m = float(np.max(intensity))
        if m > 0:
            return (intensity / m * 1000.0).astype(np.float32)
    elif mode == "sum":
        s = float(np.sum(intensity))
        if s > 0:
            return (intensity / s).astype(np.float32)
    elif mode == "none":
        return intensity.astype(np.float32)
    else:
        raise ValueError(f"Unknown normalize mode: {mode}")

    return intensity.astype(np.float32)


def extract_valid_peaks_from_array(spec_2x128, num_peaks=None):
    """
    Fast extraction from fixed spectrum.

    If num_peaks is reliable, use first num_peaks peaks.
    Otherwise remove zero mz/intensity.
    """
    if num_peaks is not None and num_peaks > 0:
        n = min(int(num_peaks), spec_2x128.shape[1])
        mz = spec_2x128[0, :n].astype(np.float32, copy=False)
        intensity = spec_2x128[1, :n].astype(np.float32, copy=False)

        mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        mz = mz[mask]
        intensity = intensity[mask]
    else:
        mz = spec_2x128[0, :].astype(np.float32, copy=False)
        intensity = spec_2x128[1, :].astype(np.float32, copy=False)

        mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
        mz = mz[mask]
        intensity = intensity[mask]

    if mz.size <= 1:
        return mz.astype(np.float32), intensity.astype(np.float32)

    order = np.argsort(mz)
    return mz[order].astype(np.float32), intensity[order].astype(np.float32)


def pack_spectrum_fixed(mz, intensity, max_peaks=128):
    """
    Keep top max_peaks by intensity, then sort by mz.
    """
    mz = np.asarray(mz, dtype=np.float32)
    intensity = np.asarray(intensity, dtype=np.float32)

    mask = np.isfinite(mz) & np.isfinite(intensity) & (mz > 0) & (intensity > 0)
    mz = mz[mask]
    intensity = intensity[mask]

    if mz.size > max_peaks:
        idx = np.argpartition(intensity, -max_peaks)[-max_peaks:]
        mz = mz[idx]
        intensity = intensity[idx]

    if mz.size > 1:
        order = np.argsort(mz)
        mz = mz[order]
        intensity = intensity[order]

    spec = np.zeros((2, max_peaks), dtype=np.float32)
    n = int(min(mz.size, max_peaks))

    if n > 0:
        spec[0, :n] = mz[:n]
        spec[1, :n] = intensity[:n]

    return spec, n


def merge_peaks_fast(mz_arrays, intensity_arrays, weights, mz_tol=0.01):
    """
    Merge peaks from multiple spectra.

    Each component intensity is first normalized to max=1000,
    then multiplied by component weight.

    For equal mixing:
        AB  weights = [0.5, 0.5]
        ABC weights = [1/3, 1/3, 1/3]
    """
    weights = np.asarray(weights, dtype=np.float32)
    weights = weights / np.sum(weights)

    total_len = sum(len(x) for x in mz_arrays)
    if total_len == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    all_mz = np.empty(total_len, dtype=np.float32)
    all_int = np.empty(total_len, dtype=np.float32)

    pos = 0
    for mz, inten, w in zip(mz_arrays, intensity_arrays, weights):
        n = len(mz)
        if n == 0:
            continue

        inten = normalize_intensity(inten, mode="max") * float(w)

        all_mz[pos:pos+n] = mz
        all_int[pos:pos+n] = inten
        pos += n

    if pos == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)

    all_mz = all_mz[:pos]
    all_int = all_int[:pos]

    order = np.argsort(all_mz)
    all_mz = all_mz[order]
    all_int = all_int[order]

    merged_mz = []
    merged_int = []

    start = 0
    n = len(all_mz)

    while start < n:
        end = start + 1

        while end < n and all_mz[end] - all_mz[end - 1] <= mz_tol:
            end += 1

        cmz = all_mz[start:end]
        cint = all_int[start:end]
        s = float(np.sum(cint))

        if s > 0:
            merged_mz.append(float(np.sum(cmz * cint) / s))
            merged_int.append(s)

        start = end

    merged_mz = np.asarray(merged_mz, dtype=np.float32)
    merged_int = np.asarray(merged_int, dtype=np.float32)
    merged_int = normalize_intensity(merged_int, mode="max")

    return merged_mz, merged_int


def sample_weights(k, mode="equal", min_weight=0.05):
    """
    Return component mixing weights.

    mode="equal":
        k=2 -> [0.5, 0.5]
        k=3 -> [0.3333, 0.3333, 0.3333]

    mode="dirichlet" or "uniform":
        random weights with lower-bound min_weight.
    """
    if mode == "equal":
        return np.ones(k, dtype=np.float32) / float(k)

    for _ in range(100):
        if mode == "dirichlet":
            w = np.random.dirichlet(np.ones(k)).astype(np.float32)
        elif mode == "uniform":
            w = np.random.uniform(0.2, 1.0, size=k).astype(np.float32)
            w = w / np.sum(w)
        else:
            raise ValueError(f"Unknown weight mode: {mode}")

        if float(np.min(w)) >= min_weight:
            return w

    return np.ones(k, dtype=np.float32) / float(k)


def join_values(arr, indices, sep=" | "):
    return sep.join(arr[indices].tolist())


# ============================================================
# Load all MoNA into memory
# ============================================================

def load_mona_to_memory(
    input_path,
    min_peaks=10,
    min_precursor_mz=50.0,
    require_smiles=True,
    require_inchikey=True,
    ion_mode_filter=None,
    adduct_filter=None,
):
    """
    Load required MoNA datasets into memory.
    """
    print(f"Input file: {input_path}")

    with h5py.File(input_path, "r") as f:
        n = f["precursor_mz"].shape[0]
        print(f"Total spectra: {n}")

        print("Loading numeric arrays...")
        spectrum = f["spectrum"][:]
        precursor_mz = f["precursor_mz"][:]
        num_peaks = f["num_peaks"][:]
        charge = f["charge"][:]
        mona_idx = f["mona_idx"][:]

        string_fields = [
            "COLLISION_ENERGY",
            "FORMULA",
            "FRAGMENTATION_MODE",
            "IDENTIFIER",
            "INSTRUMENT",
            "INSTRUMENT_TYPE",
            "ION_MODE",
            "adduct",
            "inchikey",
            "mona_id",
            "name",
            "smiles",
            "source",
        ]

        strings = {}
        print("Loading and decoding string arrays...")
        for field in tqdm(string_fields, desc="Decoding strings"):
            strings[field] = decode_array(f[field])

    valid = np.ones(n, dtype=bool)
    valid &= np.isfinite(precursor_mz)
    valid &= precursor_mz >= min_precursor_mz
    valid &= num_peaks >= min_peaks

    if require_smiles:
        smiles = strings["smiles"]
        smiles_ok = np.array([
            (s is not None) and (str(s).strip() != "") and (str(s).strip().lower() not in {"nan", "none", "null"})
            for s in smiles
        ], dtype=bool)
        valid &= smiles_ok

    if require_inchikey:
        inchikey = strings["inchikey"]
        inchikey_ok = np.array([
            (s is not None) and (str(s).strip() != "") and (str(s).strip().lower() not in {"nan", "none", "null"})
            for s in inchikey
        ], dtype=bool)
        valid &= inchikey_ok

    if ion_mode_filter is not None:
        allowed = {x.lower() for x in ion_mode_filter}
        ion_ok = np.array([
            str(s).strip().lower() in allowed
            for s in strings["ION_MODE"]
        ], dtype=bool)
        valid &= ion_ok

    if adduct_filter is not None:
        allowed = {x.lower() for x in adduct_filter}
        adduct_ok = np.array([
            str(s).strip().lower() in allowed
            for s in strings["adduct"]
        ], dtype=bool)
        valid &= adduct_ok

    valid_indices = np.where(valid)[0].astype(np.int64)
    print(f"Valid spectra after filtering: {len(valid_indices)}")

    data = {
        "spectrum": spectrum,
        "precursor_mz": precursor_mz,
        "num_peaks": num_peaks,
        "charge": charge,
        "mona_idx": mona_idx,
        "strings": strings,
        "valid_indices": valid_indices,
    }

    return data


def build_window_bins_from_memory(data, window_size=20.0):
    precursor = data["precursor_mz"]
    valid_indices = data["valid_indices"]

    bin_ids = np.floor(precursor[valid_indices] / window_size).astype(np.int64)

    bins = defaultdict(list)
    for idx, b in zip(valid_indices, bin_ids):
        bins[int(b)].append(int(idx))

    bins = {b: np.asarray(v, dtype=np.int64) for b, v in bins.items() if len(v) >= 3}

    print(f"Valid DIA windows: {len(bins)}")

    if len(bins) == 0:
        raise RuntimeError("No valid DIA windows. Relax filters.")

    data["bins"] = bins
    data["bin_keys"] = np.asarray(list(bins.keys()), dtype=np.int64)

    return data


# ============================================================
# Worker generation
# ============================================================

def sample_components_from_memory(k, require_distinct_inchikey=True, max_tries=1000):
    """
    Sample k components from one DIA bin.
    """
    bins = G["bins"]
    bin_keys = G["bin_keys"]
    inchikey = G["strings"]["inchikey"]

    for _ in range(max_tries):
        b = int(bin_keys[np.random.randint(0, len(bin_keys))])
        pool = bins[b]

        if len(pool) < k:
            continue

        picked = np.random.choice(pool, size=k, replace=False).astype(np.int64)

        if require_distinct_inchikey:
            keys = inchikey[picked]
            if len(set(keys.tolist())) < k:
                continue

        return picked, b

    raise RuntimeError("Failed to sample components.")


def generate_chunk_worker(task):
    """
    Generate a chunk of chimera spectra.

    Return a dict containing arrays for this chunk.
    """
    (
        start,
        n_chunk,
        k,
        chimera_type,
        window_size,
        mz_tol,
        weight_mode,
        min_weight,
        max_peaks,
        require_distinct_inchikey,
    ) = task

    spectrum = G["spectrum"]
    precursor_mz = G["precursor_mz"]
    num_peaks = G["num_peaks"]
    charge = G["charge"]
    mona_idx = G["mona_idx"]
    S = G["strings"]

    out_spectrum = np.zeros((n_chunk, 2, max_peaks), dtype=np.float32)
    out_num_peaks = np.zeros(n_chunk, dtype=np.int64)
    out_precursor_mz = np.zeros(n_chunk, dtype=np.float64)
    out_charge = np.zeros(n_chunk, dtype=np.int64)
    out_mona_idx = np.arange(start, start + n_chunk, dtype=np.int64)

    max_components = 3
    out_component_count = np.full(n_chunk, k, dtype=np.int64)
    out_precursor_mz_list = np.zeros((n_chunk, max_components), dtype=np.float64)
    out_mixing_weights = np.zeros((n_chunk, max_components), dtype=np.float32)
    out_source_mona_idx = np.full((n_chunk, max_components), -1, dtype=np.int64)
    out_dia_window_id = np.zeros(n_chunk, dtype=np.int64)
    out_dia_window_lower = np.zeros(n_chunk, dtype=np.float64)
    out_dia_window_upper = np.zeros(n_chunk, dtype=np.float64)

    str_fields = [
        "COLLISION_ENERGY",
        "FORMULA",
        "FRAGMENTATION_MODE",
        "IDENTIFIER",
        "INSTRUMENT",
        "INSTRUMENT_TYPE",
        "ION_MODE",
        "adduct",
        "inchikey",
        "mona_id",
        "name",
        "smiles",
        "source",
        "chimera_type",
    ]

    out_str = {field: np.empty(n_chunk, dtype=object) for field in str_fields}

    source_str_fields = [
        "source_mona_id",
        "source_inchikey",
        "source_smiles",
        "source_name",
    ]
    out_source_str = {
        field: np.empty((n_chunk, max_components), dtype=object)
        for field in source_str_fields
    }

    for i in range(n_chunk):
        global_i = start + i

        picked, window_id = sample_components_from_memory(
            k=k,
            require_distinct_inchikey=require_distinct_inchikey,
        )

        # Default now:
        # AB  -> [0.5, 0.5]
        # ABC -> [1/3, 1/3, 1/3]
        weights = sample_weights(k, mode=weight_mode, min_weight=min_weight)

        mz_arrays = []
        int_arrays = []

        for idx in picked:
            mz, inten = extract_valid_peaks_from_array(
                spectrum[idx],
                num_peaks=int(num_peaks[idx]),
            )
            mz_arrays.append(mz)
            int_arrays.append(inten)

        merged_mz, merged_int = merge_peaks_fast(
            mz_arrays,
            int_arrays,
            weights=weights,
            mz_tol=mz_tol,
        )

        packed, n_pk = pack_spectrum_fixed(
            merged_mz,
            merged_int,
            max_peaks=max_peaks,
        )

        out_spectrum[i] = packed
        out_num_peaks[i] = n_pk

        pmz_list = precursor_mz[picked].astype(np.float64)

        # DreaMS-compatible single precursor_mz.
        # Here use the first component precursor as representative.
        out_precursor_mz[i] = float(pmz_list[0])
        out_charge[i] = int(charge[picked[0]])
        out_component_count[i] = k

        out_precursor_mz_list[i, :k] = pmz_list
        out_mixing_weights[i, :k] = weights
        out_source_mona_idx[i, :k] = picked

        out_dia_window_id[i] = int(window_id)
        out_dia_window_lower[i] = float(window_id * window_size)
        out_dia_window_upper[i] = float((window_id + 1) * window_size)

        chimera_id = f"chimera_{chimera_type}_{global_i:06d}"

        out_str["COLLISION_ENERGY"][i] = join_values(S["COLLISION_ENERGY"], picked)
        out_str["FORMULA"][i] = join_values(S["FORMULA"], picked)
        out_str["FRAGMENTATION_MODE"][i] = join_values(S["FRAGMENTATION_MODE"], picked)
        out_str["IDENTIFIER"][i] = join_values(S["IDENTIFIER"], picked)
        out_str["INSTRUMENT"][i] = join_values(S["INSTRUMENT"], picked)
        out_str["INSTRUMENT_TYPE"][i] = join_values(S["INSTRUMENT_TYPE"], picked)
        out_str["ION_MODE"][i] = join_values(S["ION_MODE"], picked)
        out_str["adduct"][i] = join_values(S["adduct"], picked)
        out_str["inchikey"][i] = join_values(S["inchikey"], picked)
        out_str["mona_id"][i] = chimera_id
        out_str["name"][i] = join_values(S["name"], picked, sep=" + ")
        out_str["smiles"][i] = join_values(S["smiles"], picked)
        out_str["source"][i] = "synthetic_chimera_from_MoNA"
        out_str["chimera_type"][i] = chimera_type

        for j in range(max_components):
            if j < k:
                idx = picked[j]
                out_source_str["source_mona_id"][i, j] = S["mona_id"][idx]
                out_source_str["source_inchikey"][i, j] = S["inchikey"][idx]
                out_source_str["source_smiles"][i, j] = S["smiles"][idx]
                out_source_str["source_name"][i, j] = S["name"][idx]
            else:
                out_source_str["source_mona_id"][i, j] = ""
                out_source_str["source_inchikey"][i, j] = ""
                out_source_str["source_smiles"][i, j] = ""
                out_source_str["source_name"][i, j] = ""

    result = {
        "start": start,
        "n": n_chunk,
        "spectrum": out_spectrum,
        "num_peaks": out_num_peaks,
        "precursor_mz": out_precursor_mz,
        "charge": out_charge,
        "mona_idx": out_mona_idx,
        "component_count": out_component_count,
        "precursor_mz_list": out_precursor_mz_list,
        "mixing_weights": out_mixing_weights,
        "source_mona_idx": out_source_mona_idx,
        "dia_window_id": out_dia_window_id,
        "dia_window_lower": out_dia_window_lower,
        "dia_window_upper": out_dia_window_upper,
        "str": out_str,
        "source_str": out_source_str,
    }

    return result


# ============================================================
# HDF5 writing
# ============================================================

def create_output_root(out, n, max_peaks=128, max_components=3):
    """
    Create root-level MoNA/DreaMS-compatible HDF5 datasets.

    DreaMS MSData.load() expects columns directly under root:
        /spectrum
        /precursor_mz
        /num_peaks
        /smiles
        /inchikey
        /mona_id
        ...
    """
    str_dt = h5py.string_dtype(encoding="utf-8")

    out.create_dataset("COLLISION_ENERGY", shape=(n,), dtype=str_dt)
    out.create_dataset("FORMULA", shape=(n,), dtype=str_dt)
    out.create_dataset("FRAGMENTATION_MODE", shape=(n,), dtype=str_dt)
    out.create_dataset("IDENTIFIER", shape=(n,), dtype=str_dt)
    out.create_dataset("INSTRUMENT", shape=(n,), dtype=str_dt)
    out.create_dataset("INSTRUMENT_TYPE", shape=(n,), dtype=str_dt)
    out.create_dataset("ION_MODE", shape=(n,), dtype=str_dt)
    out.create_dataset("adduct", shape=(n,), dtype=str_dt)

    out.create_dataset("charge", shape=(n,), dtype=np.int64)
    out.create_dataset("inchikey", shape=(n,), dtype=str_dt)
    out.create_dataset("mona_id", shape=(n,), dtype=str_dt)
    out.create_dataset("mona_idx", shape=(n,), dtype=np.int64)
    out.create_dataset("name", shape=(n,), dtype=str_dt)
    out.create_dataset("num_peaks", shape=(n,), dtype=np.int64)
    out.create_dataset("precursor_mz", shape=(n,), dtype=np.float64)
    out.create_dataset("smiles", shape=(n,), dtype=str_dt)
    out.create_dataset("source", shape=(n,), dtype=str_dt)

    out.create_dataset("spectrum", shape=(n, 2, max_peaks), dtype=np.float32)

    out.create_dataset(
        "DreaMS_embedding",
        shape=(n, 1024),
        dtype=np.float32,
        fillvalue=np.nan,
    )

    out.create_dataset("chimera_type", shape=(n,), dtype=str_dt)
    out.create_dataset("component_count", shape=(n,), dtype=np.int64)

    out.create_dataset("precursor_mz_list", shape=(n, max_components), dtype=np.float64)
    out.create_dataset("mixing_weights", shape=(n, max_components), dtype=np.float32)
    out.create_dataset("source_mona_idx", shape=(n, max_components), dtype=np.int64)

    out.create_dataset("source_mona_id", shape=(n, max_components), dtype=str_dt)
    out.create_dataset("source_inchikey", shape=(n, max_components), dtype=str_dt)
    out.create_dataset("source_smiles", shape=(n, max_components), dtype=str_dt)
    out.create_dataset("source_name", shape=(n, max_components), dtype=str_dt)

    out.create_dataset("dia_window_id", shape=(n,), dtype=np.int64)
    out.create_dataset("dia_window_lower", shape=(n,), dtype=np.float64)
    out.create_dataset("dia_window_upper", shape=(n,), dtype=np.float64)

    return out


def write_chunk_to_group(grp, chunk):
    start = int(chunk["start"])
    end = start + int(chunk["n"])
    sl = slice(start, end)

    grp["spectrum"][sl] = chunk["spectrum"]
    grp["num_peaks"][sl] = chunk["num_peaks"]
    grp["precursor_mz"][sl] = chunk["precursor_mz"]
    grp["charge"][sl] = chunk["charge"]
    grp["mona_idx"][sl] = chunk["mona_idx"]

    grp["component_count"][sl] = chunk["component_count"]
    grp["precursor_mz_list"][sl, :] = chunk["precursor_mz_list"]
    grp["mixing_weights"][sl, :] = chunk["mixing_weights"]
    grp["source_mona_idx"][sl, :] = chunk["source_mona_idx"]
    grp["dia_window_id"][sl] = chunk["dia_window_id"]
    grp["dia_window_lower"][sl] = chunk["dia_window_lower"]
    grp["dia_window_upper"][sl] = chunk["dia_window_upper"]

    for field, arr in chunk["str"].items():
        if arr is None:
            continue
        if field in grp:
            grp[field][sl] = arr.astype(str)

    for field, arr in chunk["source_str"].items():
        grp[field][sl, :] = arr.astype(str)


# ============================================================
# Orchestration
# ============================================================

def make_tasks(start_offset, n_total, chunk_size, k, chimera_type, window_size, mz_tol,
               weight_mode, min_weight, max_peaks, require_distinct_inchikey):
    """
    Create generation tasks with absolute output indices.

    AB:
        start_offset = 0

    ABC:
        start_offset = n_ab
    """
    tasks = []

    for local_start in range(0, n_total, chunk_size):
        n_chunk = min(chunk_size, n_total - local_start)
        absolute_start = start_offset + local_start

        tasks.append((
            absolute_start,
            n_chunk,
            k,
            chimera_type,
            window_size,
            mz_tol,
            weight_mode,
            min_weight,
            max_peaks,
            require_distinct_inchikey,
        ))

    return tasks


def run_parallel_for_root(
    grp,
    start_offset,
    n_total,
    k,
    chimera_type,
    data,
    n_workers,
    chunk_size,
    window_size,
    mz_tol,
    weight_mode,
    min_weight,
    max_peaks,
    require_distinct_inchikey,
    seed,
):
    """
    Generate AB or ABC spectra and write into root-level datasets.

    grp is actually the HDF5 root file handle.
    """
    tasks = make_tasks(
        start_offset=start_offset,
        n_total=n_total,
        chunk_size=chunk_size,
        k=k,
        chimera_type=chimera_type,
        window_size=window_size,
        mz_tol=mz_tol,
        weight_mode=weight_mode,
        min_weight=min_weight,
        max_peaks=max_peaks,
        require_distinct_inchikey=require_distinct_inchikey,
    )

    print(
        f"{chimera_type}: {len(tasks)} chunks, "
        f"start_offset={start_offset}, chunk_size={chunk_size}, workers={n_workers}, "
        f"weight_mode={weight_mode}"
    )

    with ProcessPoolExecutor(
        max_workers=n_workers,
        initializer=init_worker,
        initargs=(data, seed),
    ) as ex:
        futures = [ex.submit(generate_chunk_worker, task) for task in tasks]

        pbar = tqdm(total=n_total, desc=f"Writing {chimera_type}")

        for fut in as_completed(futures):
            chunk = fut.result()
            write_chunk_to_group(grp, chunk)
            pbar.update(chunk["n"])

        pbar.close()


def generate_chimera_parallel(
    input_path,
    output_path,
    n_ab=10_000,
    n_abc=10_000,
    n_workers=None,
    chunk_size=1000,
    window_size=20.0,
    mz_tol=0.01,
    min_peaks=10,
    min_precursor_mz=50.0,
    weight_mode="equal",
    min_weight=0.05,
    max_peaks=128,
    seed=42,
    ion_mode_filter=None,
    adduct_filter=None,
    require_smiles=True,
    require_inchikey=True,
    require_distinct_inchikey=True,
):
    if n_workers is None:
        n_workers = max(1, os.cpu_count() - 1)

    print("=" * 100)
    print("[generate chimera]")
    print(f"input_path                 : {input_path}")
    print(f"output_path                : {output_path}")
    print(f"n_ab                       : {n_ab}")
    print(f"n_abc                      : {n_abc}")
    print(f"total                      : {n_ab + n_abc}")
    print(f"weight_mode                : {weight_mode}")
    print(f"AB mixing                  : 1:1" if weight_mode == "equal" else "AB mixing                  : random")
    print(f"ABC mixing                 : 1:1:1" if weight_mode == "equal" else "ABC mixing                 : random")
    print(f"n_workers                  : {n_workers}")
    print(f"chunk_size                 : {chunk_size}")
    print(f"window_size                : {window_size}")
    print(f"mz_tol                     : {mz_tol}")
    print(f"require_distinct_inchikey  : {require_distinct_inchikey}")
    print("=" * 100)

    data = load_mona_to_memory(
        input_path=input_path,
        min_peaks=min_peaks,
        min_precursor_mz=min_precursor_mz,
        require_smiles=require_smiles,
        require_inchikey=require_inchikey,
        ion_mode_filter=ion_mode_filter,
        adduct_filter=adduct_filter,
    )

    data = build_window_bins_from_memory(data, window_size=window_size)

    total_n = int(n_ab + n_abc)

    with h5py.File(output_path, "w") as out:
        out.attrs["source_library"] = "MoNA"
        out.attrs["dataset_type"] = "synthetic_chimeric_msms_parallel_root_dreams_compatible"
        out.attrs["layout"] = "root_level"
        out.attrs["input_path"] = input_path

        out.attrs["n_total"] = int(total_n)
        out.attrs["n_chimera_AB"] = int(n_ab)
        out.attrs["n_chimera_ABC"] = int(n_abc)
        out.attrs["ab_start"] = int(0)
        out.attrs["ab_end"] = int(n_ab)
        out.attrs["abc_start"] = int(n_ab)
        out.attrs["abc_end"] = int(n_ab + n_abc)

        out.attrs["n_workers"] = int(n_workers)
        out.attrs["chunk_size"] = int(chunk_size)
        out.attrs["window_size"] = float(window_size)
        out.attrs["mz_merge_tolerance"] = float(mz_tol)
        out.attrs["min_peaks_input"] = int(min_peaks)
        out.attrs["min_precursor_mz"] = float(min_precursor_mz)
        out.attrs["weight_mode"] = weight_mode
        out.attrs["min_weight"] = float(min_weight)
        out.attrs["max_peaks"] = int(max_peaks)
        out.attrs["seed"] = int(seed)

        grp_all = create_output_root(
            out,
            n=total_n,
            max_peaks=max_peaks,
            max_components=3,
        )

        run_parallel_for_root(
            grp=grp_all,
            start_offset=0,
            n_total=n_ab,
            k=2,
            chimera_type="AB",
            data=data,
            n_workers=n_workers,
            chunk_size=chunk_size,
            window_size=window_size,
            mz_tol=mz_tol,
            weight_mode=weight_mode,
            min_weight=min_weight,
            max_peaks=max_peaks,
            require_distinct_inchikey=require_distinct_inchikey,
            seed=seed,
        )

        run_parallel_for_root(
            grp=grp_all,
            start_offset=n_ab,
            n_total=n_abc,
            k=3,
            chimera_type="ABC",
            data=data,
            n_workers=n_workers,
            chunk_size=chunk_size,
            window_size=window_size,
            mz_tol=mz_tol,
            weight_mode=weight_mode,
            min_weight=min_weight,
            max_peaks=max_peaks,
            require_distinct_inchikey=require_distinct_inchikey,
            seed=seed + 100000,
        )

    print(f"Saved DreaMS-compatible root-level output to: {output_path}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Parallel generation of AB and ABC synthetic chimeric spectra from MoNA HDF5."
    )

    parser.add_argument(
        "--input",
        default=r"D:\亚结构注释\mona_processed\mona_dreams_dataset.hdf5",
        help="Input MoNA HDF5 path.",
    )

    parser.add_argument(
        "--output",
        default=r"D:\亚结构注释\mona_processed\mona_chimera_dataset_equal_200k_random.hdf5",
        help="Output chimera HDF5 path.",
    )

    # Default now: total 20k = 10k AB + 10k ABC
    parser.add_argument("--n_ab", type=int, default=10_0000)
    parser.add_argument("--n_abc", type=int, default=10_0000)

    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of worker processes.",
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=1000,
        help="Number of spectra generated per worker task.",
    )

    parser.add_argument("--window_size", type=float, default=20.0)
    parser.add_argument("--mz_tol", type=float, default=0.01)
    parser.add_argument("--min_peaks", type=int, default=10)
    parser.add_argument("--min_precursor_mz", type=float, default=50.0)

    parser.add_argument(
        "--weight_mode",
        type=str,
        default="dirichlet",
        choices=["equal", "dirichlet", "uniform"],
        help="equal gives AB=1:1 and ABC=1:1:1.",
    )

    parser.add_argument("--min_weight", type=float, default=0.05)
    parser.add_argument("--max_peaks", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--ion_mode",
        nargs="*",
        default=None,
        help="Optional ION_MODE filter, e.g. --ion_mode positive Positive POSITIVE.",
    )

    parser.add_argument(
        "--adduct",
        nargs="*",
        default=None,
        help="Optional adduct filter, e.g. --adduct [M+H]+ [M-H]-.",
    )

    parser.add_argument("--allow_missing_smiles", action="store_true")
    parser.add_argument("--allow_missing_inchikey", action="store_true")
    parser.add_argument("--allow_same_inchikey_components", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    generate_chimera_parallel(
        input_path=args.input,
        output_path=args.output,
        n_ab=args.n_ab,
        n_abc=args.n_abc,
        n_workers=args.workers,
        chunk_size=args.chunk_size,
        window_size=args.window_size,
        mz_tol=args.mz_tol,
        min_peaks=args.min_peaks,
        min_precursor_mz=args.min_precursor_mz,
        weight_mode=args.weight_mode,
        min_weight=args.min_weight,
        max_peaks=args.max_peaks,
        seed=args.seed,
        ion_mode_filter=args.ion_mode,
        adduct_filter=args.adduct,
        require_smiles=not args.allow_missing_smiles,
        require_inchikey=not args.allow_missing_inchikey,
        require_distinct_inchikey=not args.allow_same_inchikey_components,
    )


if __name__ == "__main__":
    main()
