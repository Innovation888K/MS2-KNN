# Data — per-query benchmark results

This folder contains the per-query results of MS2-KNN and the compared methods for the two experimental benchmarks described in the manuscript. The large-scale synthetic benchmark (10,000 binary and 10,000 ternary chimeric queries) is not included due to size and is available from the corresponding author upon request.

## `selfcir/` — experimentally acquired standard mixtures

17 binary pairs measured at five ratios (4:1, 3:2, 1:1, 2:3, 1:4) plus 3 ternary groups at 1:1:1 (88 samples in total). MS2-KNN was run against a self-built reference library (the measured spectra of the standards themselves); reverse spectral search, a CHIMERYS-inspired NNLS deconvolution baseline, and global similarity search were compared under an identical protocol.

| file | description |
|---|---|
| `selfcir_query_manifest.csv` | one row per query (the representative chimeric scan of each mixture sample) with source file, group, ratio, chimera type (AB/ABC), precursor m/z, and true InChIKeys/names |
| `selfcir_library_metadata.csv` | rows of the self-built reference library (target-precursor MS/MS spectra of the measured standards retained for library search) |
| `selfcir_standard_metadata.csv` | the 43 standards with [M+H]+ m/z, molecular formula, SMILES, and InChIKey |
| `selfcir_per_query_recommendations.csv` | full MS2-KNN output per query (candidate clusters, recommended compounds, backbone-confidence scores, scaffold assignments, cluster-level pruning flags) |
| `selfcir_per_scan_recovery.csv` | per-scan recovery of true compounds |
| `selfcir_per_sample_recovery.csv` | per-sample number of true components resolved (0/1/2 for binary, 0–3 for ternary) for each compared method |
| `selfcir_tool_summary.csv` | per-method summary statistics (full-recovery counts and rates) |

## `real_mzml/` — 20 real single-herb traditional Chinese medicine extracts

DDA acquisitions de-replicated by precursor m/z to approximately 7,800 representative MS/MS queries per benchmark, searched against the MoNA reference library with the same four methods.

| file | description |
|---|---|
| `real_mzml_query_meta.csv` | one row per query with query index, sample ID, and herb name |
| `real_mzml_identification_all_hits.csv` | per-query identifications of the four compared methods (MS2-KNN after two-level pruning, global search, reverse spectral search, CHIMERYS-inspired NNLS) with rank, score, and rank-in-scan |
| `real_mzml_identification_summary.csv` | per-sample × method counts of distinct identified compounds and total recommendations |
| `real_mzml_identification_compounds.csv` | per-sample × method compound-level details (number of supporting scans, best score) |
| `real_mzml_method_specificity.csv` | herb-level specificity statistics (mean pairwise Jaccard overlap, ubiquitous compounds present in ≥18 of 20 herbs, mean yield per herb) |
| `real_mzml_per_query_recommendations.csv.gz` | full MS2-KNN output per query (gzip-compressed; open directly with `pandas.read_csv(..., compression="gzip")`) |

## Notes

- InChIKey values are given as the first 14 characters (skeleton level).
- CHIMERYS itself is a proteomics tool; the compared baseline is a CHIMERYS-inspired sparse non-negative-least-squares spectral deconvolution implemented for small molecules.
- Reference: the manuscript and its supplementary data describe how each benchmark was generated.
