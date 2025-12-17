"""
Task 5: Cross-outlet frequency/proportion analysis for refined clusters.

Reads Task 3 cluster outputs, applies manual remapping of small/fragment clusters,
and produces count/proportion tables plus heatmaps per role.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# Mapping raw cluster ids to refined labels for reporting
REFINED_LABELS = {
    "shooter": {
        10: "school/event framing",
        11: "gunman identity/age descriptors",
        12: "shooter actions",
        1: "location fragments",
        2: "barricade/standoff",
        3: "locked-room references",
        -1: "noise/fragments",
    },
    "victim": {
        0: "child/student victims",
        2: "aftermath/casualties",
        3: "counts/inclusion",
        4: "coverage/inclusion",
        -1: "noise/fragments",
    },
}


def load_clusters(path: Path) -> List[Dict]:
    # Read cluster JSON from Task 3
    return json.loads(path.read_text(encoding="utf-8"))


def expand_rows(cluster_data: List[Dict], role: str) -> List[Dict]:
    # Flatten cluster JSON into per-phrase rows with refined labels
    rows: List[Dict] = []
    mapping = REFINED_LABELS[role]
    for cluster in cluster_data:
        cid = cluster["cluster"]
        label = mapping.get(cid, "noise/fragments")
        for p in cluster["phrases"]:
            rows.append(
                {
                    "role": role,
                    "cluster_id": cid,
                    "cluster_label": label,
                    "outlet": p["outlet"],
                    "article": p["article"],
                    "phrase": p["phrase"],
                }
            )
    return rows


def pivot_counts(df: pd.DataFrame) -> pd.DataFrame:
    # Counts table: clusters x outlet
    table = (
        df.pivot_table(
            index="cluster_label",
            columns="outlet",
            values="phrase",
            aggfunc="count",
            fill_value=0,
        )
        .sort_index()
    )
    return table


def pivot_props(df: pd.DataFrame) -> pd.DataFrame:
    # Proportion table normalized per outlet
    counts = pivot_counts(df)
    col_sums = counts.sum(axis=0)
    props = counts.divide(col_sums, axis=1).fillna(0)
    return props


def plot_heatmap(data: pd.DataFrame, title: str, path: Path, fmt: str = ".2f"):
    # Save an annotated heatmap
    plt.figure(figsize=(8, max(3, 0.6 * len(data))))
    sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_grouped_bars(data: pd.DataFrame, title: str, path: Path):
    # Save grouped bar chart for counts
    ax = data.T.plot(kind="bar", figsize=(8, 5))
    ax.set_ylabel("Count")
    ax.set_title(title)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


# Save stacked bar chart for proportions
def plot_stacked_props(props: pd.DataFrame, title: str, path: Path):
    props.T.plot(kind="bar", stacked=True, figsize=(8, 5))
    plt.ylabel("Proportion")
    plt.title(title)
    plt.legend(title="Cluster", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def main():
    # CLI entrypoint for Task 5 reporting
    parser = argparse.ArgumentParser(description="Cross-outlet frequency analysis for clusters.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("outputs") / "task3" / "clusters",
        help="Directory containing *_clusters.json from Task 3.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs") / "task5",
        help="Directory to write frequency tables and plots.",
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for role in ["shooter", "victim"]:
        clusters = load_clusters(args.input_dir / f"{role}_clusters.json")
        rows = expand_rows(clusters, role)
        df = pd.DataFrame(rows)

        counts = pivot_counts(df)
        props = pivot_props(df)

        counts_path = args.output_dir / f"{role}_counts.csv"
        props_path = args.output_dir / f"{role}_proportions.csv"
        counts.to_csv(counts_path)
        props.to_csv(props_path)

        plot_heatmap(
            counts,
            title=f"{role.capitalize()} cluster counts by outlet",
            path=args.output_dir / f"{role}_counts_heatmap.png",
            fmt="d",
        )
        plot_heatmap(
            props,
            title=f"{role.capitalize()} cluster proportions by outlet",
            path=args.output_dir / f"{role}_proportions_heatmap.png",
            fmt=".2f",
        )
        plot_grouped_bars(
            counts,
            title=f"{role.capitalize()} cluster counts (grouped bars)",
            path=args.output_dir / f"{role}_counts_bar.png",
        )
        plot_stacked_props(
            props,
            title=f"{role.capitalize()} cluster proportions (stacked bars)",
            path=args.output_dir / f"{role}_proportions_bar.png",
        )


if __name__ == "__main__":
    main()
