import pandas as pd

# Download from https://github.com/Azure/AzurePublicDataset/blob/master/data/AzureLLMInferenceTrace_code.csv

# File paths
# SOURCE_CSV_FILE_PATH = 'AzureLLMInferenceTrace_code_2023.csv'
# WRITE_TO_CSV_FILE_PATH = 'processed_llm_code_trace.csv'

SOURCE_CSV_FILE_PATH = 'AzureLLMInferenceTrace_conv_2023.csv'
WRITE_TO_CSV_FILE_PATH = 'processed_llm_conv_trace.csv'

# Optionally trim top rows, None if no trim
TOP_NUM_ROWS_TO_RETRIEVE = 8000

# Read the CSV
data_frame = pd.read_csv(SOURCE_CSV_FILE_PATH)

# Convert TIMESTAMP to datetime
data_frame['TIMESTAMP'] = pd.to_datetime(data_frame['TIMESTAMP'])

# Sort by TIMESTAMP
data_frame = data_frame.sort_values(by='TIMESTAMP').reset_index(drop=True)

# Convert to nanoseconds relative to the first timestamp
first_time = data_frame['TIMESTAMP'].iloc[0]
data_frame['start_timestamp_ms'] = ((data_frame['TIMESTAMP'] - first_time).dt.total_seconds() * 1e3).astype('int64')

# Drop the old TIMESTAMP column
data_frame = data_frame.drop(columns=['TIMESTAMP'])

if TOP_NUM_ROWS_TO_RETRIEVE:
    data_frame = data_frame.head(TOP_NUM_ROWS_TO_RETRIEVE)

# Save to CSV
data_frame.to_csv(WRITE_TO_CSV_FILE_PATH, index=False)
print("Processed CSV written to", WRITE_TO_CSV_FILE_PATH, ", #lines:", len(data_frame))