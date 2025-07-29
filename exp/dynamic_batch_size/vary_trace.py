"""
This script is used to generate the customized trace for the simulation, based on real traces.
"""
import random
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np

def extract_request_arrival_timestamps(trace_file_path):
    '''
    @param trace_file_path: the path to the trace file
    @return: a list of arrival times of the requests
    '''
    df = pd.read_csv(trace_file_path)
    return df['start_timestamp_ms'].tolist()

def generate_trace_with_multiple_concurrent_users(trace_file_path, num_users):
    '''
    Generate a trace with multiple concurrent users, with the arrival times of the requests overlapping.
    @param trace_file_path: the path to the trace file
    @param num_users: the number of users
    @return: a list of arrival times of the requests
    '''
    arrival_times = extract_request_arrival_timestamps(trace_file_path)
    arrival_times.sort()
    group_size = len(arrival_times) // num_users
    arrival_times_groups = [arrival_times[i*group_size:(i+1)*group_size] for i in range(num_users)]
    # Overlap the arrival times of the users, as if they start at the same time 0+offset_time*random_double(0, 1)
    concurrent_arrival_times = []
    for i in range(num_users):
        offset_time = random.uniform(0, 1)
        normalized_arrival_times = [arrival_time - arrival_times_groups[i][0] + offset_time for arrival_time in arrival_times_groups[i]]
        concurrent_arrival_times.extend(normalized_arrival_times)
    concurrent_arrival_times.sort()
    return concurrent_arrival_times

def generate_trace_with_simple_compression(trace_file_path, compression_factor):
    '''
    Simple compression, just divide the arrival times by the compression factor
    @param trace_file_path: the path to the trace file
    @param compression_factor: the compression factor
    @return: a list of arrival times of the requests
    '''
    arrival_times = extract_request_arrival_timestamps(trace_file_path)
    compressed_arrival_times = [arrival_time * 1.0 / compression_factor for arrival_time in arrival_times]
    return compressed_arrival_times


def plot_trace_pattern(arrival_times, bin_width=1.0):
    '''
    Plot the histogram of the arrival times
    @param arrival_times: a list of arrival times
    @param bin_width: the width of the bins, default is 1.0 second
    @return: None
    '''
    # Convert from microseconds to seconds 
    np_arrival_times = np.array(arrival_times)
    np_arrival_times = np_arrival_times * 1.0 / 1e3

    # Compute number of bins
    num_bins = int((np_arrival_times.max() - np_arrival_times.min()) / bin_width)

    # Plot histogram
    plt.hist(np_arrival_times, bins=num_bins, edgecolor='black')
    plt.xlabel('timeline (s)')
    plt.ylabel('Frequency')
    plt.title('Histogram of Generated Arrival Time Distribution')
    plt.show()

# Example usage - you can change the file path as needed
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot start_timestamp histogram from CSV')
    parser.add_argument('csv_file_path', help='Path to the CSV file')
    parser.add_argument('--gen_type', help='The type of the trace to generate', choices=['original','compression', 'multi_users'], default='original')
    args = parser.parse_args()

    
    # Extract timestamps
    arrival_times = extract_request_arrival_timestamps(args.csv_file_path)
    if args.gen_type == 'compression':
        arrival_times = generate_trace_with_simple_compression(args.csv_file_path, 10)
    elif args.gen_type == 'multi_users':
        arrival_times = generate_trace_with_multiple_concurrent_users(args.csv_file_path, 10)
    else:
        arrival_times = extract_request_arrival_timestamps(args.csv_file_path)
    plot_trace_pattern(arrival_times)
    
    # You can now use timestamp_list for your analysis
    print(f"\nTotal number of timestamps: {len(arrival_times)}")
    print(f"Timestamp range: {min(arrival_times)} to {max(arrival_times)} ms")


    