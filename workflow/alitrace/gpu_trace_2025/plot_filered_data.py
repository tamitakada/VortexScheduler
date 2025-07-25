import pandas as pd
import matplotlib.pyplot as plt
import argparse

def main():
    parser = argparse.ArgumentParser(description='Plot creation_time histogram from CSV')
    parser.add_argument('csv_file_path', help='Path to the CSV file')
    parser.add_argument('--num_rows', type=int, default=800, help='Number of rows to consider')
    parser.add_argument('--bin_width', type=float, default=10.0, help='Bin width for histogram')
    args = parser.parse_args()

    # Read CSV
    data_frame = pd.read_csv(args.csv_file_path)
    data_frame['creation_time'] = data_frame['creation_time'] #/ 1000.0

    # Take first N rows
    subset = data_frame.head(args.num_rows)

    # Compute number of bins
    num_bins = int((subset['creation_time'].max() - subset['creation_time'].min()) / args.bin_width)

    # Plot histogram
    plt.hist(subset['creation_time'], bins=num_bins, edgecolor='black')
    plt.xlabel('timeline (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of creation_time')
    plt.show()

if __name__ == '__main__':
    main()