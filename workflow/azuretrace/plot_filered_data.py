import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot start_timestamp histogram from CSV')
    parser.add_argument('csv_file_path', help='Path to the CSV file')
    parser.add_argument('--num_rows', type=int, default=5000, help='Number of rows to consider, None for all')
    parser.add_argument('--bin_width', type=float, default=1.0, help='Bin width for histogram')
    args = parser.parse_args()

    # Read CSV
    data_frame = pd.read_csv(args.csv_file_path)
    data_frame['start_timestamp'] = data_frame['start_timestamp_ms'] / 1e3  # Convert from microseconds to seconds 

    # Take first N rows
    if args.num_rows is not None:
        data_frame = data_frame.head(args.num_rows)
    else:
        args.num_rows = len(data_frame)
    subset = data_frame.head(args.num_rows)
    
    # start = 15000
    # end = 19000
    # subset = data_frame.iloc[start:end]

    # Compute number of bins
    num_bins = int((subset['start_timestamp'].max() - subset['start_timestamp'].min()) / args.bin_width)

    # Plot histogram
    plt.hist(subset['start_timestamp'], bins=num_bins, edgecolor='black')
    plt.xlabel('timeline (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Azure Function Invocation Start Time Distribution')
    plt.show()

if __name__ == '__main__':
    main()