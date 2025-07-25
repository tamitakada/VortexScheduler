import pandas as pd
import matplotlib.pyplot as plt

# https://github.com/alibaba/clusterdata/tree/master/cluster-trace-gpu-v2025

SOURCE_CSV_FILE_PATH = 'disaggregated_DLRM_trace_2025.csv'
WRITE_TO_CSV_FILE_PATH = 'filtered_trace.csv'

TOP_ROWS_TO_RETRIEVE = None

COLUMNS_TO_KEEP = ['instance_sn', 'role', 'app_name', 'creation_time', 'gpu_request']

# Read the CSV
data_frame = pd.read_csv(SOURCE_CSV_FILE_PATH)

# Filter and make a copy to avoid SettingWithCopyWarning
filtered_data_frame = data_frame[data_frame['creation_time'].notna()].copy()

print(filtered_data_frame.head(5))

# Normalize creation time
earliest_time = filtered_data_frame['creation_time'].min()
filtered_data_frame.loc[:, 'creation_time'] = filtered_data_frame['creation_time'] - earliest_time

# Keep only selected columns
final_trace = filtered_data_frame[COLUMNS_TO_KEEP]

# Optionally trim top rows
if TOP_ROWS_TO_RETRIEVE:
    final_trace = final_trace.head(TOP_ROWS_TO_RETRIEVE)

# Write the DataFrame to a CSV file
final_trace.to_csv(WRITE_TO_CSV_FILE_PATH, index=False)
print("DataFrame has been written to", WRITE_TO_CSV_FILE_PATH, ", #lines:", len(final_trace))