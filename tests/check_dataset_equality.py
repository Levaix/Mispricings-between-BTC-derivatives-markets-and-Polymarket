"""
Load options_smile_event_probabilities.parquet and btc_probability_comparison_empirical_master.parquet.
Report row counts and confirm equality of key statistics:
  mean(probability_spread), median(probability_spread), mean(prob_option), mean(prob_yes).
Saves report under tests.
"""

from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
PROB_PARQUET = CODE_ROOT / "Actual data" / "Option Probabilities" / "options_smile_event_probabilities.parquet"
MASTER_PARQUET = CODE_ROOT / "Actual data" / "btc_probability_comparison_empirical_master.parquet"
REPORT_PATH = CODE_ROOT / "tests" / "dataset_equality_report.txt"


def main() -> None:
    lines = [
        "Dataset equality check (key variables)",
        "======================================",
        "",
    ]

    df_prob = pd.read_parquet(PROB_PARQUET)
    df_master = pd.read_parquet(MASTER_PARQUET)

    n_prob = len(df_prob)
    n_master = len(df_master)

    lines.append("Row counts:")
    lines.append(f"  options_smile_event_probabilities.parquet:  {n_prob}")
    lines.append(f"  btc_probability_comparison_empirical_master.parquet:  {n_master}")
    lines.append("")

    # On full dataset, restrict to rows with valid probability_spread for comparison
    df_prob_valid = df_prob.dropna(subset=["probability_spread", "prob_option", "prob_yes"])
    n_prob_valid = len(df_prob_valid)

    def stats(df: pd.DataFrame) -> dict:
        return {
            "mean_probability_spread": float(df["probability_spread"].mean()),
            "median_probability_spread": float(df["probability_spread"].median()),
            "mean_prob_option": float(df["prob_option"].mean()),
            "mean_prob_yes": float(df["prob_yes"].mean()),
        }

    s_prob = stats(df_prob_valid)
    s_master = stats(df_master)

    lines.append("Key statistics (full: rows with non-null probability_spread; master: all rows):")
    lines.append("")
    lines.append("                         full (valid rows)    master")
    lines.append(f"  n rows                {n_prob_valid}              {n_master}")
    lines.append(f"  mean(probability_spread)   {s_prob['mean_probability_spread']:.10f}   {s_master['mean_probability_spread']:.10f}")
    lines.append(f"  median(probability_spread) {s_prob['median_probability_spread']:.10f}   {s_master['median_probability_spread']:.10f}")
    lines.append(f"  mean(prob_option)          {s_prob['mean_prob_option']:.10f}   {s_master['mean_prob_option']:.10f}")
    lines.append(f"  mean(prob_yes)             {s_prob['mean_prob_yes']:.10f}   {s_master['mean_prob_yes']:.10f}")
    lines.append("")

    tol = 1e-12
    identical = (
        abs(s_prob["mean_probability_spread"] - s_master["mean_probability_spread"]) <= tol
        and abs(s_prob["median_probability_spread"] - s_master["median_probability_spread"]) <= tol
        and abs(s_prob["mean_prob_option"] - s_master["mean_prob_option"]) <= tol
        and abs(s_prob["mean_prob_yes"] - s_master["mean_prob_yes"]) <= tol
        and n_prob_valid == n_master
    )
    if identical:
        lines.append("Result: Statistics are IDENTICAL (and row counts match).")
    else:
        lines.append("Result: Statistics DIFFER (or row counts differ).")
    lines.append("")

    text = "\n".join(lines)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
