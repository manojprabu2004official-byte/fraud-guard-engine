import polars as pl
from pathlib import Path

# Get the directory where this script file is saved
SCRIPT_DIR = Path(__file__).resolve().parent

# Automatically create an absolute path to your files
raw_file = SCRIPT_DIR / "PS_20174392719_1491204439457_log.csv"
output_parquet = SCRIPT_DIR / "paysim_cleaned.parquet"

print("Optimizing PaySim dataset using Polars LazyScan...")
# ... the rest of your polars pipeline code remains exactly the same!

# Execute lazy pipeline for optimal memory allocation
df_clean = (
    pl.scan_csv(raw_file)
    # 1. Filter only fraud-bearing transaction types
    .filter(pl.col("type").is_in(["TRANSFER", "CASH_OUT"]))
    # 2. Compute error deltas and encode binary type
    .with_columns([
        # Origin & Destination balance ledger discrepancies
        (pl.col("oldbalanceOrg") - pl.col("amount") - pl.col("newbalanceOrig")).alias("errorBalanceOrig"),
        (pl.col("oldbalanceDest") + pl.col("amount") - pl.col("newbalanceDest")).alias("errorBalanceDest"),
        
        # Binary encoding: 1 for TRANSFER, 0 for CASH_OUT
        (pl.col("type") == "TRANSFER").cast(pl.UInt8).alias("is_transfer"),
        
        # Relative drain ratio (handles divide-by-zero safely)
        pl.when(pl.col("oldbalanceOrg") > 0)
          .then(pl.col("amount") / pl.col("oldbalanceOrg"))
          .otherwise(0.0)
          .alias("drain_ratio")
    ])
    .collect()
)

# 3. Export to compressed Parquet format
df_clean.write_parquet(output_parquet, compression="snappy")

print(f"Dataset successfully cleaned and compressed!")
print(f"Reduced row count: {df_clean.height:,} rows")
print(f"Columns ready for ML: {df_clean.columns}")