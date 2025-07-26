import os
import subprocess
import pandas as pd
import csv
import tempfile
import shutil

# ===== CONFIG =====
# Download from https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar
RAR_FILE_PATH = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.rar"
OUTPUT_CSV = "processed_azfunc_trace.csv"
# Optionally trim top rows, None if no trim
TOP_NUM_ROWS_TO_RETRIEVE = 5000

# ===== STEP 1: Extract to a temporary directory using unar =====
temp_dir = tempfile.mkdtemp()
print(f"Extracting {RAR_FILE_PATH} to a temporary folder...")

result = subprocess.run(["unar", "-o", temp_dir, "-f", RAR_FILE_PATH], capture_output=True, text=True)
if result.returncode != 0:
    print("Error during extraction:\n", result.stderr)
    shutil.rmtree(temp_dir)
    raise RuntimeError("Extraction failed.")
else:
    print(result.stdout)

# ===== STEP 2: Find extracted file (txt or csv) =====
extracted_files = [f for f in os.listdir(temp_dir) if f.endswith(".csv") or f.endswith(".txt")]
if not extracted_files:
    shutil.rmtree(temp_dir)
    raise FileNotFoundError("No CSV or TXT files found inside the RAR archive!")
data_file_path = os.path.join(temp_dir, extracted_files[0])
print(f"Found data file: {data_file_path}")

# ===== STEP 3: Auto-detect delimiter & load =====
print(f"Loading {data_file_path}...")

with open(data_file_path, 'r', newline='', encoding='utf-8') as f:
    sample = f.read(4096)
    dialect = csv.Sniffer().sniff(sample)
    delimiter = dialect.delimiter

data_frame = pd.read_csv(data_file_path, sep=delimiter)

print(f"Loaded {len(data_frame)} rows and {len(data_frame.columns)} columns.")
print("Columns:", list(data_frame.columns)[:20], "...")

# ===== STEP 4: Compute start_timestamp in microseconds =====
if {"end_timestamp", "duration"}.issubset(data_frame.columns):
    # Compute start time in µs (end and duration are in ms)
    data_frame["start_timestamp_ms"] = (data_frame["end_timestamp"] - data_frame["duration"]) * 1000
    print("Added start_timestamp_ms column.")
else:
    print("Warning: 'end_timestamp' or 'duration' column missing; cannot compute start_timestamp_ms.")

# sort the data frame by start_timestamp_ms
data_frame = data_frame.sort_values(by="start_timestamp_ms").reset_index(drop=True)

if TOP_NUM_ROWS_TO_RETRIEVE:
    data_frame = data_frame.head(TOP_NUM_ROWS_TO_RETRIEVE)
    print(f"Trimmed to top {TOP_NUM_ROWS_TO_RETRIEVE} rows.")

# ===== STEP 5: Save as processed CSV =====
data_frame.to_csv(OUTPUT_CSV, index=False)
print(f"Processed CSV written to {OUTPUT_CSV}.")

# ===== STEP 6: Clean up temporary directory =====
shutil.rmtree(temp_dir)
print("Temporary files removed.")