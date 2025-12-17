Task 6 - Chi-Squared Homogeneity Tests

Setup
- Data: Cluster counts from Task 5 (`outputs/task5/shooter_counts.csv`, `victim_counts.csv`). Victim micro-clusters were collapsed into a single “victim descriptors” bucket; shooter kept three clusters (“gunman descriptors”, “tactical / standoff”, “noise/fragments”).
- Test: Chi-Squared Test of Homogeneity per role on the top clusters (all clusters available after collapse).

Hypotheses (for each listed cluster per role)
- H0: The proportion of descriptions in this cluster is the same across CNN, Fox, NYT, and WSJ.
- H1: At least one outlet’s proportion differs.

Explicitly for the tested tables:
- Shooter H0: The outlet distributions for “gunman descriptors”, “tactical/standoff”, and “noise/fragments” are identical across CNN, Fox, NYT, and WSJ.
- Victim H0: The outlet distributions for “victim descriptors” and “noise/fragments” are identical across CNN, Fox, NYT, and WSJ.

Results (α=0.05)
- Shooter clusters (3×4 table; dof=6): χ²=22.16, p=0.00113 → reject H0. Shooter framing distribution differs by outlet. Notably, WSJ concentrates more tactical/standoff (7/20%) while others have near-zero; gunman descriptors vary moderately.
- Victim clusters (2×4 table; dof=3): χ²=16.44, p=0.00092 → reject H0. Victim framing distribution differs by outlet. NYT shows a higher fraction of noise/fragments (36/108 ≈33%) versus others (10–21%); WSJ has the lowest noise share.

Interpretation
- Shooter: WSJ over-represents “tactical / standoff” relative to expected; others under-represent that frame, driving significance.
- Victim: NYT’s higher fragment/noise share and WSJ’s lower share drive deviation from homogeneous proportions.

Files
- Code: `task6_hypothesis_testing.py`
- Outputs: `outputs/task6_chi2_results.json` (stats, p-values, expected counts, contingency tables).
