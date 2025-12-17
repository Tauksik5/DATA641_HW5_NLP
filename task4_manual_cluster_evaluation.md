Task 4 - Manual Cluster Evaluation

Clusters reviewed from `outputs/task3/clusters/*`.

Lexical/semantic coherence & purity
- Shooter 0 (“gunman / school shooter descriptors”): High lexical overlap (gunman/shooter + age/role), consistent semantics (perpetrator identity/actions). Kept.
- Shooter 1 (“location / reporting fragments”): Low coherence; artifacts (“texte”, partial phrases). Misclassifications from extraction noise. Merged into noise.
- Shooter 2 & 3 (“barricade / standoff actions” + “locked-room references”): Small but coherent action/context phrasing (barricaded, locked room). Merged into a single “tactical / standoff” bucket for stability.
- Shooter noise: Catch-all for fragments and stray verbs.
- Victim 0 (“child/student victim descriptors”): Strong lexical and semantic alignment (child/student/teacher counts/ages). Kept.
- Victim 1 (“shooting impact on victims”): Shot/wounded phrasing; coherent but tiny. Kept separate for visibility.
- Victim 2 (“aftermath / casualties”): Dead/left/hospitalized phrases; coherent. Kept.
- Victim 3 (“counts / inclusion phrasing”): Small (counts/inclusion wording); borderline—kept but monitored.
- Victim noise: Fragments and stray verbs; not semantically coherent.

Refinements applied
- Shooter: cluster 1 collapsed into noise; clusters 2+3 merged to “tactical / standoff”.
- Victim: clusters retained, but noise explicitly tracked; small clusters flagged as low-support.
- Refined labels and mapping baked into `task5_frequency_analysis.py` via `REFINED_LABELS`, used for frequency/proportion reporting.

Problematic clusters / causes
- Fragment clusters stem from extraction noise (partial spans like “texte”, “where lock”). Root cause: imperfect phrase extraction (short tokens or markup leakage).
- Small clusters are sensitive to DBSCAN parameters; keeping them merged/flagged reduces over-interpretation.

Outcome
- Final reporting uses the refined mapping (merge/suppress noisy shooter clusters; merge shooter tactical clusters) to improve purity while preserving the main semantic themes.
