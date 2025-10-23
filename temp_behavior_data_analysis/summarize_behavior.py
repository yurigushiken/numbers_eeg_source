import json
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data_behavior" / "data_UTF8"
    output_dir = Path(__file__).resolve().parent
    output_dir.mkdir(parents=True, exist_ok=True)

    subject_summaries = {}
    aggregate_block_orders = []
    aggregate_block_sets = []
    all_blocks_counter = Counter()

    print("Behavioral dataset summary (block procedures and Target ACC usage)\n")

    for csv_path in sorted(data_dir.glob("Subject*.csv")):
        df = pd.read_csv(csv_path, on_bad_lines="warn", low_memory=False)
        subject_key = csv_path.stem
        block_series = df.get("Procedure[Block]")
        block_values = []
        block_counts = {}
        if block_series is not None:
            block_values = [str(v) for v in block_series.dropna().unique().tolist()]
            block_counts = Counter(str(v) for v in block_series.dropna())
            aggregate_block_orders.append(block_values)
            aggregate_block_sets.append(set(block_values))
            all_blocks_counter.update(block_counts)

        acc_cols = [col for col in df.columns if "ACC" in col and "Overall" not in col]
        acc_usage = defaultdict(dict)
        if block_series is not None and acc_cols:
            for block in block_counts:
                block_mask = block_series == block
                for col in acc_cols:
                    non_null_count = int(df.loc[block_mask, col].notna().sum())
                    if non_null_count > 0:
                        acc_usage[block][col] = non_null_count

        summary = {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "block_values": block_values,
            "block_counts": {k: int(v) for k, v in block_counts.items()},
            "acc_columns": acc_cols,
            "acc_usage_by_block": {blk: usage for blk, usage in acc_usage.items()},
        }
        subject_summaries[subject_key] = summary

        print(f"{subject_key}: rows={summary['rows']}, columns={summary['columns']}")
        if block_values:
            print(f"  Block procedures ({len(block_values)}): {', '.join(block_values)}")
        else:
            print("  Block procedures: <none>")
        print("  ACC columns: " + (", ".join(acc_cols) if acc_cols else "<none>"))
        if acc_usage:
            for blk, usage in acc_usage.items():
                top_col = max(usage.items(), key=lambda item: item[1])[0]
                print(f"    {blk}: primary ACC column {top_col} (non-null {usage[top_col]})")
                if len(usage) > 1:
                    extras = {k: v for k, v in usage.items() if k != top_col}
                    print(f"      additional ACC columns: {extras}")
        else:
            print("    <no ACC usage detected>")
        print("")

    output_dir.joinpath("behavior_summary.json").write_text(json.dumps(subject_summaries, indent=2))

    aggregate_info = {
        "block_orders": aggregate_block_orders,
        "block_sets": [sorted(list(s)) for s in aggregate_block_sets],
        "block_frequency": {k: int(v) for k, v in all_blocks_counter.items()},
    }
    output_dir.joinpath("behavior_aggregate.json").write_text(json.dumps(aggregate_info, indent=2))

    print("Summary artifacts written to:")
    print(f"  {output_dir / 'behavior_summary.json'}")
    print(f"  {output_dir / 'behavior_aggregate.json'}")


if __name__ == "__main__":
    main()
