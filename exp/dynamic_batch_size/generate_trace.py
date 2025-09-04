from collections import deque
import csv
import enum
from hmac import new
import json
import logging
import argparse
import random
import numpy as np
from typing import List, Tuple


def generate_slo_factors(num_requests: int, min_slo: float, max_slo: float, 
                        distribution: str = 'uniform') -> List[float]:
    """
    Generate random SLO factors within the specified range.
    
    Args:
        num_requests: Number of SLO factors to generate
        min_slo: Minimum SLO factor value
        max_slo: Maximum SLO factor value
        distribution: Distribution type ('uniform', 'normal', 'exponential')
    
    Returns:
        List of generated SLO factors
    """
    if distribution == 'uniform':
        return [random.uniform(min_slo, max_slo) for _ in range(num_requests)]
    elif distribution == 'normal':
        # Use normal distribution with mean at center of range
        mean = (min_slo + max_slo) / 2
        std = (max_slo - min_slo) / 6  # 99.7% of values within range
        slo_factors = np.random.normal(mean, std, num_requests)
        # Clip values to ensure they stay within bounds
        return np.clip(slo_factors, min_slo, max_slo).tolist()
    elif distribution == 'exponential':
        # Use exponential distribution, scaled to fit the range
        scale = (max_slo - min_slo) / 4
        slo_factors = np.random.exponential(scale, num_requests) + min_slo
        return np.clip(slo_factors, min_slo, max_slo).tolist()
    else:
        raise ValueError(f"Unknown distribution: {distribution}")


def save_to_csv(slo_factors: List[float], output_file: str, 
                include_timestamps: bool = True, arrival_rate: float = 1.0):
    """
    Save SLO factors to CSV file with optional timestamps.
    
    Args:
        slo_factors: List of SLO factors
        output_file: Output CSV file path
        include_timestamps: Whether to include arrival timestamps
        arrival_rate: Average requests per second for timestamp generation
    """
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['request_id', 'slo_factor']
        if include_timestamps:
            fieldnames.append('arrival_time')
        
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        current_time = 0.0
        for i, slo_factor in enumerate(slo_factors):
            row = {
                'request_id': i,
                'slo_factor': round(slo_factor, 4)
            }
            
            if include_timestamps:
                # Generate arrival time using exponential distribution
                inter_arrival_time = random.expovariate(arrival_rate)
                current_time += inter_arrival_time
                row['arrival_time'] = round(current_time, 4)
            
            writer.writerow(row)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Generate random SLO factors and save to CSV')
    parser.add_argument('--min_slo', type=float, required=True, 
                       help='Minimum SLO factor value')
    parser.add_argument('--max_slo', type=float, required=True, 
                       help='Maximum SLO factor value')
    parser.add_argument('--num_requests', type=int, default=1000,
                       help='Number of requests to generate (default: 1000)')
    parser.add_argument('--output_file', type=str, default='slo_factors.csv',
                       help='Output CSV file path (default: slo_factors.csv)')
    parser.add_argument('--distribution', type=str, default='uniform',
                       choices=['uniform', 'normal', 'exponential'],
                       help='Distribution type for SLO factors (default: uniform)')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for reproducibility')
    parser.add_argument('--no_timestamps', action='store_true',
                       help='Exclude arrival timestamps from output')
    parser.add_argument('--arrival_rate', type=float, default=1.0,
                       help='Average requests per second for timestamp generation (default: 1.0)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.min_slo >= args.max_slo:
        parser.error("min_slo must be less than max_slo")
    
    if args.num_requests <= 0:
        parser.error("num_requests must be positive")
    
    # Set random seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    # Configure logging
    if args.verbose:
        logging.basicConfig(level=logging.INFO)
    else:
        logging.basicConfig(level=logging.WARNING)
    
    logger = logging.getLogger(__name__)
    
    logger.info(f"Generating {args.num_requests} SLO factors")
    logger.info(f"Range: [{args.min_slo}, {args.max_slo}]")
    logger.info(f"Distribution: {args.distribution}")
    logger.info(f"Output file: {args.output_file}")
    
    # Generate SLO factors
    slo_factors = generate_slo_factors(
        num_requests=args.num_requests,
        min_slo=args.min_slo,
        max_slo=args.max_slo,
        distribution=args.distribution
    )
    
    # Save to CSV
    save_to_csv(
        slo_factors=slo_factors,
        output_file=args.output_file,
        include_timestamps=not args.no_timestamps,
        arrival_rate=args.arrival_rate
    )
    
    # Print summary statistics
    logger.info(f"Generated {len(slo_factors)} SLO factors")
    logger.info(f"Min SLO: {min(slo_factors):.4f}")
    logger.info(f"Max SLO: {max(slo_factors):.4f}")
    logger.info(f"Mean SLO: {np.mean(slo_factors):.4f}")
    logger.info(f"Std SLO: {np.std(slo_factors):.4f}")
    logger.info(f"Saved to: {args.output_file}")


if __name__ == "__main__":
    main()