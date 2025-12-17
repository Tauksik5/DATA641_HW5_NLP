Task 3 - Description Clustering

Approach
- Embeddings: used SBERT (`all-MiniLM-L6-v2`) to embed each deduped shooter/victim phrase from Task 2. Rationale: sentence-transformers give contextual phrase-level vectors that capture short noun/verb phrases better than static word vectors, while remaining lightweight for 300–400 phrases. Embeddings are L2-normalized for cosine DBSCAN.
- Clustering: ran DBSCAN with a small grid search over eps ∈ {0.4, 0.5, 0.6, 0.7} and min_samples ∈ {3, 4, 5}. For each combination, silhouette score (cosine) was computed when ≥2 non-noise clusters were present; the best-scoring model was chosen and refit. Fallback eps=0.7/min_samples=4 is used only if all candidates collapse to noise. This avoids hand-picking epsilon while keeping the model deterministic.
- Labels: top tokens per cluster are shown in the JSON; manual overrides applied for interpretability:
  - Shooter clusters: (0) “gunman / school shooter descriptors” (age + role + action), (1) “location / reporting fragments” (short Texas/attribution fragments), (2) “barricade / standoff actions”, (3) “locked-room references”; noise holds 59 miscellaneous phrases.
  - Victim clusters: (0) “child/student victim descriptors” (counts/ages/roles), (1) “shooting impact on victims” (shot/wounded), (2) “aftermath / casualties” (left dead/hospitalized), (3) “counts / inclusion phrasing”; noise holds 75 items.
- Visualization: UMAP (n_neighbors=15, min_dist=0.1, cosine, seed=42) projects embeddings to 2D; scatter plots saved as `outputs/task3/shooter_umap.png` and `outputs/task3/victim_umap.png`.

Outputs
- Cluster JSONs with labels and per-phrase membership: `outputs/task3/clusters/shooter_clusters.json`, `outputs/task3/clusters/victim_clusters.json`.
- Aggregate: `outputs/task3/summary.json` plus the UMAP PNGs above.

Notes / limitations
- Short fragments and extraction noise are isolated into a “noise” cluster; they are retained for transparency but can be pruned downstream.
- DBSCAN parameters were tuned on this corpus only; if you expand the phrase set, rerun the grid search to avoid over-/under-clustering.
- Victim micro-clusters can be collapsed (see Task 5/6 scripts) to simplify downstream frequency and significance testing.
