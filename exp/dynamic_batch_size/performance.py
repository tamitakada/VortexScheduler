#!/usr/bin/env python3
"""
Performance Analysis Script

This script analyzes the performance of the scheduler by reading finished requests
and calculating latency statistics.
"""

import json
import pandas as pd
import numpy as np
import logging

def load_finished_requests():
    """Load finished requests from JSON file"""
    try:
        with open('output/finished_reqs.json', 'r') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} finished requests")
        return data
    except FileNotFoundError:
        print("Error: output/finished_reqs.json not found")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in output/finished_reqs.json")
        return None

def calculate_latency_stats(requests, logger):
    """Calculate latency statistics for all requests"""
    logger.info("="*50)
    logger.info("LATENCY ANALYSIS")
    logger.info("="*50)
    
    # Calculate latency for each request
    latencies = []
    for req in requests:
        # Latency = finish_time - arrival_time
        if req['finish_time'] is not None:
            latency = req['finish_time'] - req['arrival_time']
            latencies.append(latency)

    
    # Convert to numpy array for easier calculations
    latencies = np.array(latencies)
    
    # Calculate statistics
    avg_latency = np.mean(latencies)
    median_latency = np.median(latencies)
    p90_latency = np.percentile(latencies, 90)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Log results
    logger.info(f"Total requests analyzed: {len(latencies)}")
    logger.info(f"Average latency: {avg_latency:.3f} ms")
    logger.info(f"Median latency: {median_latency:.3f} ms")
    logger.info(f"90th percentile (P90): {p90_latency:.3f} ms")
    logger.info(f"95th percentile (P95): {p95_latency:.3f} ms")
    logger.info(f"99th percentile (P99): {p99_latency:.3f} ms")
    
    # Additional statistics
    min_latency = np.min(latencies)
    max_latency = np.max(latencies)
    std_latency = np.std(latencies)
    
    logger.info("Additional statistics:")
    logger.info(f"Minimum latency: {min_latency:.3f} ms")
    logger.info(f"Maximum latency: {max_latency:.3f} ms")
    logger.info(f"Standard deviation: {std_latency:.3f} ms")
    
    return {
        'count': len(latencies),
        'avg': avg_latency,
        'median': median_latency,
        'p90': p90_latency,
        'p95': p95_latency,
        'p99': p99_latency,
        'min': min_latency,
        'max': max_latency,
        'std': std_latency,
        'latencies': latencies
    }

def analyze_by_batch_size(requests):
    """Analyze latency by batch size"""
    print("\n" + "="*50)
    print("LATENCY BY BATCH SIZE")
    print("="*50)
    
    # Group requests by batch size
    batch_stats = {}
    for req in requests:
        if req['finish_time'] is not None:
            batch_size = req['batch_size']
            latency = req['finish_time'] - req['arrival_time']
        
            if batch_size not in batch_stats:
                batch_stats[batch_size] = []
        
        batch_stats[batch_size].append(latency)
    
    # Calculate statistics for each batch size
    print("Batch Size | Count | Avg Latency | Median | P90 | P95 | P99")
    print("-" * 70)
    
    for batch_size in sorted(batch_stats.keys()):
        latencies = np.array(batch_stats[batch_size])
        
        avg = np.mean(latencies)
        median = np.median(latencies)
        p90 = np.percentile(latencies, 90)
        p95 = np.percentile(latencies, 95)
        p99 = np.percentile(latencies, 99)
        
        print(f"{batch_size:10d} | {len(latencies):5d} | {avg:11.3f} | {median:6.3f} | {p90:5.3f} | {p95:5.3f} | {p99:5.3f}")

def analyze_queue_time(requests, logger):
    """Analyze queue time statistics"""
    logger.info("="*50)
    logger.info("QUEUE TIME ANALYSIS")
    logger.info("="*50)
    
    queue_times = []
    for req in requests:
        if req['finish_time'] is not None:
            queue_time = req['queue_time']
            queue_times.append(queue_time)
    
    queue_times = np.array(queue_times)
    
    logger.info(f"Average queue time: {np.mean(queue_times):.3f} ms")
    logger.info(f"Median queue time: {np.median(queue_times):.3f} ms")
    logger.info(f"90th percentile queue time: {np.percentile(queue_times, 90):.3f} ms")
    logger.info(f"95th percentile queue time: {np.percentile(queue_times, 95):.3f} ms")
    logger.info(f"99th percentile queue time: {np.percentile(queue_times, 99):.3f} ms")
    logger.info(f"Maximum queue time: {np.max(queue_times):.3f} ms")

def analyze_slo_satisfaction(requests, logger):
    """Analyze SLO satisfaction rates for a fixed SLO threshold"""
    logger.info("="*60)
    logger.info("SLO SATISFACTION ANALYSIS")
    logger.info("="*60)
    
    # Count requests by status
    satisfied = 0
    dropped = 0
    not_satisfied = 0
    total = 0
    
    for req in requests:
        if req['finish_time'] is not None:
            if req['finish_time']  <= req['deadline']:
                satisfied += 1
            else:
                not_satisfied += 1
        else:
            dropped += 1
        total += 1
    
    # Calculate percentages
    satisfied_percentage = (satisfied / total) * 100 if total > 0 else 0
    not_satisfied_percentage = (not_satisfied / total) * 100 if total > 0 else 0
    dropped_percentage = (dropped / total) * 100 if total > 0 else 0
    
    # Log results with fixed-width columns
    logger.info(f"{'Metric':<20} {'Count':<10} {'Percentage %':<12}")
    logger.info("-" * 42)
    logger.info(f"{'SLO Satisfied':<20} {satisfied:<10} {satisfied_percentage:<11.1f}")
    logger.info(f"{'Not Satisfied':<20} {not_satisfied:<10} {not_satisfied_percentage:<11.1f}")
    logger.info(f"{'Dropped':<20} {dropped:<10} {dropped_percentage:<11.1f}")
    logger.info(f"{'Total':<20} {total:<10} {'100.0':<12}")
    logger.info("-" * 42)
    # logger.info(f"SLO Threshold: {slo_threshold} ms")



def get_performance_metrics(requests, logger):
    """Main function to run the performance analysis"""
    logger.info("="*50)    
    logger.info("PERFORMANCE ANALYSIS")
    logger.info("="*50)
    
    # Load finished requests
    # requests = load_finished_requests()
    if requests is None:
        return
    
    # Calculate overall latency statistics
    stats = calculate_latency_stats(requests, logger)
    
    # Analyze by batch size
    # analyze_by_batch_size(requests)
    
    # Analyze queue times
    analyze_queue_time(requests, logger)
    
    # Analyze SLO satisfaction with a fixed SLO threshold
    # slo_threshold = 100.0  # 100ms SLO threshold - you can modify this value
    analyze_slo_satisfaction(requests, logger)
    

    

# if __name__ == "__main__":
#     get_performance_metrics()
