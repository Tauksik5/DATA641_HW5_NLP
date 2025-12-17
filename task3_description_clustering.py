"""
Task 3: Cluster descriptive phrases for shooter/victim portrayals.

Steps:
1) Load phrases from Task 2 aggregate output.
2) Embed phrases with SBERT (all-MiniLM-L6-v2).
3) Tune DBSCAN (eps/min_samples grid) via silhouette score.
4) Save cluster assignments and UMAP visualizations.
"""
import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import umap
from sentence_transformers import SentenceTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

# Common stopwords trimmed from labels
STOPWORDS = {
    "the",
    "a",
    "an",
    "and",
    "of",
    "in",
    "at",
    "to",
    "for",
    "on",
    "by",
    "with",
    "after",
    "into",
    "was",
    "were",
    "is",
    "are",
    "be",
    "been",
}


# Load phrases and role metadata from Task 2 aggregate JSON
def load_phrases(path: Path) -> List[Tuple[str, str, str, str]]:
    """Return list of (phrase, role, outlet, article)."""
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for rec in data:
        outlet = rec.get("outlet")
        article = rec.get("article")
        for p in rec.get("shooter_phrases", []):
            rows.append((p, "shooter", outlet, article))
        for p in rec.get("victim_phrases", []):
            rows.append((p, "victim", outlet, article))
    return rows


# Tune DBSCAN over small grid using silhouette score
def choose_dbscan(
    embeddings: np.ndarray, eps_grid: List[float], min_samples_grid: List[int]
) -> DBSCAN:
    best = None
    best_score = -math.inf
    for eps in eps_grid:
        for ms in min_samples_grid:
            model = DBSCAN(eps=eps, min_samples=ms, metric="cosine")
            labels = model.fit_predict(embeddings)
            # valid if >=2 clusters (exclude noise)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters < 2:
                continue
            try:
                score = silhouette_score(embeddings, labels, metric="cosine")
            except Exception:
                continue
            if score > best_score:
                best_score = score
                best = model
    if best is None:
        # fallback conservative params
        best = DBSCAN(eps=0.7, min_samples=4, metric="cosine").fit(embeddings)
    else:
        best.fit(embeddings)
    return best


# Build a short token-based label for a cluster
def top_tokens(phrases: List[str], k: int = 3) -> str:
    counts = Counter()
    for p in phrases:
        for tok in p.lower().replace("/", " ").split():
            tok = tok.strip(",.?!\"'()[]")
            if tok and tok not in STOPWORDS:
                counts[tok] += 1
    if not counts:
        return "misc"
    return ", ".join([w for w, _ in counts.most_common(k)])


# Cluster phrases for one role, label, and save visuals/JSON
def cluster_role(
    rows: List[Tuple[str, str, str, str]],
    role: str,
    model: SentenceTransformer,
    output_dir: Path,
):
    phrases = [(p, outlet, article) for p, r, outlet, article in rows if r == role]
    texts = [p for p, _, _ in phrases]
    if not texts:
        return None
    embeddings = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

    dbscan = choose_dbscan(embeddings, eps_grid=[0.35, 0.4, 0.45, 0.5], min_samples_grid=[2, 3, 4])
    labels = dbscan.labels_.tolist()

    # Optional sub-clustering to add finer-grained groups
    if role == "shooter":
        main_ids = [i for i, l in enumerate(labels) if l == 0]
        if len(main_ids) >= 9:
            sub = KMeans(n_clusters=3, random_state=42, n_init=10).fit(embeddings[main_ids])
            for idx, sublab in zip(main_ids, sub.labels_):
                labels[idx] = 10 + int(sublab)

    if role == "victim":
        size_counts = Counter(labels)
        for cid, count in list(size_counts.items()):
            if cid != -1 and count < 5:
                for i, l in enumerate(labels):
                    if l == cid:
                        labels[i] = -1

    labels = np.array(labels)

    clusters: Dict[int, Dict] = defaultdict(lambda: {"phrases": [], "outlets": Counter()})
    for label, (phrase, outlet, article) in zip(labels, phrases):
        clusters[label]["phrases"].append({"phrase": phrase, "outlet": outlet, "article": article})
        clusters[label]["outlets"][outlet] += 1

    label_map: Dict[int, str] = {}
    # Build summaries
    summary = []
    for cid, info in clusters.items():
        phrases_list = [p["phrase"] for p in info["phrases"]]
        auto_label = "noise" if cid == -1 else top_tokens(phrases_list)
        manual_overrides = {
            ("shooter", 10): "school/event framing",
            ("shooter", 11): "gunman identity/age descriptors",
            ("shooter", 12): "shooter actions (kill/shoot)",
            ("shooter", 1): "location / reporting fragments",
            ("shooter", 2): "barricade / standoff actions",
            ("shooter", 3): "locked-room references",
            ("victim", 0): "child/student victim descriptors",
            ("victim", 2): "aftermath / casualties",
            ("victim", 3): "counts / inclusion phrasing",
            ("victim", 4): "coverage/inclusion phrasing",
        }
        human_label = manual_overrides.get((role, cid), auto_label)
        label_map[cid] = human_label
        summary.append(
            {
                "cluster": int(cid),
                "label": human_label,
                "auto_label": auto_label,
                "size": len(info["phrases"]),
                "top_outlets": info["outlets"].most_common(),
                "phrases": info["phrases"],
            }
        )

    (output_dir / "clusters").mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "clusters" / f"{role}_clusters.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    # UMAP projection
    reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, metric="cosine", random_state=42, n_components=2
    )
    points = reducer.fit_transform(embeddings)
    fig, ax = plt.subplots(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    cmap = plt.get_cmap("tab20", len(unique_labels))
    color_map = {}
    for idx, cid in enumerate(unique_labels):
        if cid == -1:
            color_map[cid] = "#8c8c8c"
        else:
            color_map[cid] = cmap(idx)
    colors = [color_map[cid] for cid in labels]
    ax.scatter(points[:, 0], points[:, 1], c=colors, s=12, alpha=0.85)
    ax.set_title(f"{role.capitalize()} descriptions clusters")
    ax.set_xticks([])
    ax.set_yticks([])
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=color_map[cid], markersize=6)
        for cid in unique_labels
    ]
    legend_labels = [label_map.get(cid, f"cluster {cid}") for cid in unique_labels]
    ax.legend(handles, legend_labels, title="Clusters", loc="upper left", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    fig_path = output_dir / f"{role}_umap.png"
    plt.savefig(fig_path, dpi=300)
    plt.close(fig)

    # Cluster size bar plot
    size_items = sorted(
        [(label_map.get(cid, f"cluster {cid}"), len(info["phrases"])) for cid, info in clusters.items()],
        key=lambda x: x[1],
        reverse=True,
    )
    labels_bar = [item[0] for item in size_items]
    sizes_bar = [item[1] for item in size_items]
    fig, ax = plt.subplots(figsize=(8, 4))
    positions = list(range(len(labels_bar)))
    ax.bar(positions, sizes_bar, color="#4477aa")
    ax.set_xticks(positions)
    ax.set_ylabel("Count")
    ax.set_title(f"{role.capitalize()} cluster sizes")
    ax.set_xticklabels(labels_bar, rotation=30, ha="right")
    plt.tight_layout()
    size_path = output_dir / f"{role}_cluster_sizes.png"
    plt.savefig(size_path, dpi=300)
    plt.close(fig)

    return {
        "phrases": texts,
        "labels": labels.tolist(),
        "umap_points": points.tolist(),
        "cluster_summary": summary,
    }


# CLI entrypoint
def main():
    parser = argparse.ArgumentParser(description="Cluster descriptive phrases for shooters/victims.")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs") / "task2" / "all_descriptions.json",
        help="Path to Task 2 aggregate descriptions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "task3",
        help="Directory to store clustering outputs.",
    )
    args = parser.parse_args()

    rows = load_phrases(args.input)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for role in ["shooter", "victim"]:
        res = cluster_role(rows, role, model, args.output_dir)
        results[role] = {
            "clusters": res["cluster_summary"] if res else [],
        }

    (args.output_dir / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
