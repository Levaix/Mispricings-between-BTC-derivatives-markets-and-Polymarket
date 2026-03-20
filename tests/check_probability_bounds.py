"""
Check that prob_yes and prob_option lie within [0, 1].
Report min/max and counts of prob_option < 0 and prob_option > 1.
Uses: Actual data/Option Probabilities/options_smile_event_probabilities.parquet
      Actual data/btc_probability_comparison_empirical_master.parquet
"""

from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
PROB_PARQUET = CODE_ROOT / "Actual data" / "Option Probabilities" / "options_smile_event_probabilities.parquet"
MASTER_PARQUET = CODE_ROOT / "Actual data" / "btc_probability_comparison_empirical_master.parquet"
REPORT_PATH = CODE_ROOT / "tests" / "probability_bounds_report.txt"


def check_bounds(df: pd.DataFrame, label: str) -> list[str]:
    lines = [f"--- {label} ---", ""]
    py = pd.to_numeric(df["prob_yes"], errors="coerce")
    po = pd.to_numeric(df["prob_option"], errors="coerce")
    valid_py = py.notna()
    valid_po = po.notna()

    lines.append(f"min(prob_yes)   = {py.min():.6g}" if valid_py.any() else "min(prob_yes)   = N/A (no valid)")
    lines.append(f"max(prob_yes)   = {py.max():.6g}" if valid_py.any() else "max(prob_yes)   = N/A (no valid)")
    lines.append("")
    lines.append(f"min(prob_option) = {po.min():.6g}" if valid_po.any() else "min(prob_option) = N/A (no valid)")
    lines.append(f"max(prob_option) = {po.max():.6g}" if valid_po.any() else "max(prob_option) = N/A (no valid)")
    lines.append("")

    n_lt0 = (po < 0).sum()
    n_gt1 = (po > 1).sum()
    lines.append("Observations with prob_option out of [0, 1]:")
    lines.append(f"  prob_option < 0:  {int(n_lt0)}")
    lines.append(f"  prob_option > 1:  {int(n_gt1)}")
    lines.append("")
    return lines


def main() -> None:
    all_lines = [
        "Probability bounds check",
        "========================",
        "",
    ]

    if PROB_PARQUET.exists():
        df_prob = pd.read_parquet(PROB_PARQUET)
        all_lines.extend(check_bounds(df_prob, "options_smile_event_probabilities.parquet"))
    else:
        all_lines.append(f"File not found: {PROB_PARQUET}")
        all_lines.append("")

    if MASTER_PARQUET.exists():
        df_master = pd.read_parquet(MASTER_PARQUET)
        all_lines.extend(check_bounds(df_master, "btc_probability_comparison_empirical_master.parquet"))
    else:
        all_lines.append(f"File not found: {MASTER_PARQUET}")
        all_lines.append("")

    text = "\n".join(all_lines)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
