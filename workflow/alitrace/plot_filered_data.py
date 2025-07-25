import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'filtered_trace.csv'

data_frame = pd.read_csv(csv_file_path)
data_frame['creation_time'] = data_frame['creation_time'] / 1000.0


first_1000_rows = data_frame.head(800)

# num_bins = 500
bin_width = 1
num_bins = int((first_1000_rows['creation_time'].max() - first_1000_rows['creation_time'].min()) / bin_width)

# Create the histogram plot
plt.hist(first_1000_rows['creation_time'], bins=num_bins, edgecolor='black')

# Add labels and title
plt.xlabel('timeline(s)')
plt.ylabel('Frequency')
plt.title('Histogram of creation_time')

# Display the plot
plt.show()