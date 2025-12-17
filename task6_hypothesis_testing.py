"""
Task 6: Chi-Squared Test of Homogeneity for framing clusters across outlets.

Uses Task 5 counts to build contingency tables for the top 3 clusters per role
and reports chi2 stats, degrees of freedom, and p-values.
"""
import argparse
from pathlib import Path
from typing import List

import pandas as pd
from scipy.stats import chi2_contingency


def load_counts(path: Path) -> pd.DataFrame:
    # Read counts CSV from Task 5
    return pd.read_csv(path, index_col=0)


def top_clusters(df: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    # Select top-k clusters by total count
    totals = df.sum(axis=1).sort_values(ascending=False)
    top = totals.head(k).index
    return df.loc[top]


def run_chi2(table: pd.DataFrame):
    # Run chi-squared homogeneity test on contingency table
    chi2, p, dof, exp = chi2_contingency(table)
    return {"chi2": chi2, "p_value": p, "dof": dof, "expected": exp.tolist()}


def main():
    # CLI entrypoint for Task 6
    parser = argparse.ArgumentParser(description="Chi-squared homogeneity tests for framing clusters.")
    parser.add_argument(
        "--counts-dir",
        type=Path,
        default=Path("outputs") / "task5",
        help="Directory containing <role>_counts.csv from Task 5.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs") / "task6_chi2_results.json",
        help="Path to write chi-squared results JSON.",
    )
    args = parser.parse_args()

    results = {}
    for role in ["shooter", "victim"]:
        df = load_counts(args.counts_dir / f"{role}_counts.csv")
        top = top_clusters(df, k=min(3, len(df)))
        res = run_chi2(top)
        res["clusters"] = top.index.tolist()
        res["table"] = top.to_dict()
        results[role] = res

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(pd.Series(results).to_json(indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
