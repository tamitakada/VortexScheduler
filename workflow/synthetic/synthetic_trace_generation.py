import numpy as np
import matplotlib.pyplot as plt

# def generate_synthetic_trace(mean, cva2, size):
#     # Calculate the shape and scale parameters based on mean and CVA2
#     shape = mean**2 / cva2**2
#     scale = cva2**2 / mean

#     # Generate synthetic inter-arrival times from the gamma distribution
#     inter_arrival_times = np.random.gamma(shape, scale, size)

#     # Compute request times from inter-arrival times
#     request_times = np.cumsum(inter_arrival_times)

#     return request_times

def generate_synthetic_trace(mean, cv, size):
    # Calculate the shape (k) and scale (θ) parameters from mean and CV
    k = (mean / cv) ** 2
    theta = cv ** 2 / mean

    # Generate inter-arrival times from the gamma distribution
    inter_arrival_times = np.random.gamma(k, theta, size)
    print("gamma is: ",inter_arrival_times)
    samples = np.random.exponential(k, size)
    print("exponential is ", samples)
    samples = [mean] * size
    print("uniform is ", samples)

    # Calculate arrival times from inter-arrival times
    arrival_times = np.cumsum(inter_arrival_times)

    return arrival_times


def generate_exponential(interval_ms, size):
    pts = []
    for i in range(size):
        pts.append(np.random.exponential(interval_ms))
    sorted_data = np.sort(pts)
    # Calculate the cumulative probabilities
    cumulative_probabilities = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

    # Plot the cumulative distribution curve
    plt.plot(sorted_data, cumulative_probabilities, marker='o', color='blue')
    plt.xlabel('Values')
    plt.ylabel('Cumulative Probability')
    plt.title('Cumulative Distribution Curve (CDC)')
    
    
    samples = np.random.exponential(interval_ms, size)
    sorted_samples = np.sort(samples)
    plt.plot(sorted_samples, cumulative_probabilities, marker='o', color='red')

    plt.grid(True)
    plt.show()

generate_exponential(5, 1000)


# # Parameters
# mean = 5.0
# cva2 = 5.0
# size = 10  # Number of synthetic requests

# # Generate a synthetic trace
# request_times = generate_synthetic_trace(mean, cva2, size)

# # Calculate the number of requests arriving every second
# seconds = np.arange(0, int(np.ceil(request_times[-1])) + 1)
# requests_per_second = np.histogram(request_times, bins=seconds)[0]

# # Create the dot plot
# plt.plot(seconds[:-1], requests_per_second, 'bo', markersize=2)
# plt.xlabel("Time (seconds)")
# plt.ylabel("Number of Requests")
# plt.title("Dot Plot of Requests Arriving Every Second")
# plt.grid(True)
# plt.show()
