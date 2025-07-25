import pandas as pd
import matplotlib.pyplot as plt


# https://github.com/alibaba/clusterdata/blob/master/cluster-trace-gpu-v2023/README.md

csv_file_path = 'requests_trace.csv'

data_frame = pd.read_csv(csv_file_path)

filtered_data_frame = data_frame[data_frame['gpu_spec'] == 'T4']

print(filtered_data_frame.head())

earliest_time = filtered_data_frame['creation_time'].min()
filtered_data_frame['creation_time'] = filtered_data_frame['creation_time'] - earliest_time
filtered_data_frame['creation_time'] = filtered_data_frame['creation_time'] / 2

filtered_data_frame = filtered_data_frame.drop(columns=['gpu_spec', 'num_gpu'])


filtered_data_frame['interval'] = filtered_data_frame['creation_time'].diff().fillna(0).astype(int)

first_1000_rows = filtered_data_frame.head(1000)



csv_file_path = 'filtered_trace.csv'

# Write the DataFrame to a CSV file
first_1000_rows.to_csv(csv_file_path, index=False)  # Set index=False to exclude row numbers

print("DataFrame has been written to", csv_file_path)