Comparative Analysis of Gun Violence Coverage

Setup
- Use Python 3.9+ (a local `.venv` is recommended).
- Install dependencies: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
- The script will download `en_core_web_sm` on the first run.
- To rerun everything end-to-end: execute Tasks 1–6 in order; outputs will come under `outputs/task*`.

Task 1: Context Extraction
- Run `python task1_coref_extraction.py --output-dir outputs/task1` (add `--device cuda` if you have a GPU).
- Inputs: 100 source articles under `cnn_five_para`, `FOX_five_para`, `NYT_five_para`, and `WSJ_five_para`.
- Outputs:
  - `outputs/task1/all_contexts.json`: resolved text plus victim/shooter context lists for every article.
  - `outputs/task1/contexts/<outlet>/<article>.json`: per-article context slices (victim and shooter sentences separated).

Task 2: Description Extraction
- Run `python task2_description_extraction.py --input outputs/task1/all_contexts.json --output-dir outputs/task2`.
- Outputs:
  - `outputs/task2/descriptions/<outlet>/<article>.json`: sentence-level descriptions and deduped phrase lists for shooter/victim mentions.
  - `outputs/task2/all_descriptions.json`: aggregate descriptions across outlets.

Task 3: Description Clustering
- Run `python task3_description_clustering.py --input outputs/task2/all_descriptions.json --output-dir outputs/task3`.
- Outputs:
  - Cluster memberships: `outputs/task3/clusters/shooter_clusters.json`, `outputs/task3/clusters/victim_clusters.json`.
  - Plots: `outputs/task3/shooter_umap.png`, `outputs/task3/victim_umap.png` (with legends), and cluster size bars `outputs/task3/shooter_cluster_sizes.png`, `outputs/task3/victim_cluster_sizes.png`.
  - Summary: `outputs/task3/summary.json`.

Task 4: Manual Cluster Evaluation
- See `docs/task4_manual_cluster_evaluation.md` for lexical/semantic assessment, purity notes, and refinements (merging tactical shooter clusters, collapsing fragment clusters to noise).

Task 5: Cross-Outlet Frequency Analysis
- Run `python task5_frequency_analysis.py --input-dir outputs/task3/clusters --output-dir outputs/task5`.
- Outputs:
  - Counts/proportions CSVs: `outputs/task5/<role>_counts.csv`, `outputs/task5/<role>_proportions.csv`.
  - Heatmaps: `outputs/task5/<role>_counts_heatmap.png`, `outputs/task5/<role>_proportions_heatmap.png`.
  - Grouped bars: `outputs/task5/<role>_counts_bar.png`.
 - These charts let you visually compare how often each framing shows up in each outlet.

Task 6: Statistical Testing
 - Run `python task6_hypothesis_testing.py --counts-dir outputs/task5 --output outputs/task6_chi2_results.json`.
 - See `docs/task6_statistical_testing.md` for hypotheses, chi-squared results, and interpretations.

Quick reference table
| Task | Command | Key Outputs |
| --- | --- | --- |
| 1 | `python task1_coref_extraction.py` | `outputs/task1/all_contexts.json`, per-article context JSONs |
| 2 | `python task2_description_extraction.py` | `outputs/task2/all_descriptions.json`, per-article description JSONs |
| 3 | `python task3_description_clustering.py` | Cluster JSONs, `shooter_umap.png`, `victim_umap.png`, cluster size bars |
| 4 | (manual review) | Notes in `docs/task4_manual_cluster_evaluation.md` |
| 5 | `python task5_frequency_analysis.py` | Counts/proportions CSVs, heatmaps, bar charts |
| 6 | `python task6_hypothesis_testing.py` | `outputs/task6_chi2_results.json`, interpretations in docs |

Notes
- Coreference resolution uses the `fastcoref` spaCy component; the code force-pins `aiohttp<3.12` for compatibility with Python 3.9.
- The keyword lists for shooter/victim sentences are intentionally conservative; adjust in `task1_coref_extraction.py` if you need broader coverage.
- Clustering uses SBERT embeddings with DBSCAN plus a small k-means refinement on the main shooter cluster; re-run Task 3 if you tweak parameters or labels.
- Frequency tables and visuals (Task 5) derive from the current cluster mapping; re-run Tasks 5 and 6 after any cluster changes.
- All scripts are CLI-friendly; use the `--help` flag on any script to see available options.
