Task 5 - Cross-Outlet Frequency Analysis

Approach (≈120 words)
- Inputs: cluster assignments from Task 3 (`outputs/task3/clusters/*.json`) with refined labels mapping new cluster IDs to readable names.
- Processing: `task5_frequency_analysis.py` flattens per-phrase records, builds counts (clusters × outlet) and column-normalized proportions, then saves tables and visuals. Shooter labels: school/event framing, gunman identity/age, shooter actions, location fragments, barricade/standoff, locked-room references, noise. Victim labels: child/student victims, aftermath/casualties, counts/inclusion, coverage/inclusion, noise.
- Outputs: counts/proportions CSVs and three plots per role—heatmaps for counts and proportions, grouped bar chart for counts, stacked bar chart for proportions.

Files
- Tables: `outputs/task5/<role>_counts.csv`, `outputs/task5/<role>_proportions.csv`.
- Plots: `outputs/task5/<role>_counts_heatmap.png`, `outputs/task5/<role>_proportions_heatmap.png`, `outputs/task5/<role>_counts_bar.png`, `outputs/task5/<role>_proportions_bar.png`.
- Code: `task5_frequency_analysis.py`.

Notes
- Labels come from the refined cluster mapping; rerun Task 5 (and Task 6) if cluster labels change.
- Counts/proportions are per outlet; proportions sum to 1.0 within each outlet column.
