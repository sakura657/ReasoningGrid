#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import argparse
import logging
from collections import defaultdict

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Default paths
DEFAULT_RESULTS_PATH = "/home/ubuntu/sober-reasoning/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/all_experiments_results.json"
DEFAULT_OUTPUT_PATH = "/home/ubuntu/sober-reasoning/output/deepseek-ai_DeepSeek-R1-Distill-Qwen-1.5B/analysis_summary.json"

# List of seed values to analyze
SEEDS = [0, 1, 2, 3, 4, 42, 100, 123, 2023, 777]

def load_results(file_path):
    """
    Load the results from the JSON file.
    
    Args:
        file_path (str): Path to the JSON file containing results.
        
    Returns:
        dict: The loaded JSON data or None if loading fails.
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        logging.info(f"Successfully loaded results from {file_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading results from {file_path}: {e}")
        return None

def compute_weighted_average(values, weights=None):
    """
    Compute weighted average of values with optional weights.
    
    Args:
        values (list): List of values to average.
        weights (list, optional): List of weights. If None, equal weights are used.
        
    Returns:
        float: The weighted average.
    """
    if weights is None:
        return np.mean(values)
    return np.average(values, weights=weights)

def compute_weighted_std_error(values, weights=None):
    """
    Compute weighted standard error of the mean.
    
    Args:
        values (list): List of values.
        weights (list, optional): List of weights. If None, equal weights are used.
        
    Returns:
        float: The weighted standard error.
    """
    if len(values) <= 1:
        return 0.0
        
    if weights is None:
        # Standard error of the mean = standard deviation / sqrt(n)
        return np.std(values, ddof=1) / np.sqrt(len(values))
    
    # For weighted calculations
    weights = np.array(weights)
    values = np.array(values)
    
    # Effective sample size for weighted data
    n_eff = np.sum(weights) ** 2 / np.sum(weights ** 2)
    
    # Weighted mean
    weighted_mean = np.sum(weights * values) / np.sum(weights)
    
    # Weighted variance
    weighted_var = np.sum(weights * ((values - weighted_mean) ** 2)) / np.sum(weights)
    
    # Weighted standard error
    return np.sqrt(weighted_var / n_eff)

def analyze_by_seed(data):
    """
    Analyze results by seed, averaging across all configurations for each seed.
    
    Args:
        data (dict): The loaded results data.
        
    Returns:
        dict: A dictionary containing average results for each seed.
    """
    seed_results = {}
    
    # Group experiments by seed
    for seed in SEEDS:
        seed_experiments = []
        
        # Find all experiments with this seed
        for exp_name, exp_data in data.items():
            try:
                exp_seed = exp_data["configuration"]["seed"]
                if exp_seed == str(seed):
                    seed_experiments.append(exp_data)
            except KeyError:
                continue
                
        if not seed_experiments:
            logging.warning(f"No experiments found for seed {seed}")
            continue
            
        # Calculate average results for this seed
        accuracy_values = []
        accuracy_stderr_values = []
        token_length_values = []
        token_length_stderr_values = []
        
        for exp in seed_experiments:
            try:
                accuracy_values.append(exp["results"]["accuracy"])
                accuracy_stderr_values.append(exp["results"]["accuracy_stderr"])
                token_length_values.append(exp["results"]["avg_token_length"])
                token_length_stderr_values.append(exp["results"]["token_length_stderr"])
            except KeyError as e:
                logging.warning(f"Missing key in experiment: {e}")
                continue
                
        # Calculate averages
        avg_accuracy = np.mean(accuracy_values)
        avg_accuracy_stderr = compute_weighted_std_error(accuracy_values)
        avg_token_length = np.mean(token_length_values)
        avg_token_length_stderr = compute_weighted_std_error(token_length_values)
        
        # Store results
        seed_results[str(seed)] = {
            "seed": seed,
            "results": {
                "accuracy": avg_accuracy,
                "accuracy_stderr": avg_accuracy_stderr,
                "avg_token_length": avg_token_length,
                "token_length_stderr": avg_token_length_stderr,
                "num_experiments": len(seed_experiments),
                "num_samples": seed_experiments[0]["results"].get("num_samples", 30)  # Assuming all have same sample count
            }
        }
        
        logging.info(f"Seed {seed}: Processed {len(seed_experiments)} experiments, " 
                    f"avg accuracy: {avg_accuracy:.4f}, "
                    f"avg token length: {avg_token_length:.2f}")
    
    return seed_results

def analyze_by_config(data):
    """
    Analyze results by configuration, averaging across seeds for each config.
    
    Args:
        data (dict): The loaded results data.
        
    Returns:
        dict: A dictionary containing average results for each configuration.
    """
    # Initialize storage for different configurations
    config_results = {}
    
    # Group experiments by configuration
    config_experiments = defaultdict(list)
    
    for exp_name, exp_data in data.items():
        try:
            # Extract configuration parameters
            temp = exp_data["configuration"]["temperature"]
            top_p = exp_data["configuration"]["top_p"]
            dtype = exp_data["configuration"]["dtype"]
            
            # Create a configuration key
            config_key = f"temp_{temp}_topp_{top_p}_dtype_{dtype}"
            
            config_experiments[config_key].append(exp_data)
        except KeyError:
            continue
    
    # Calculate average results for each configuration
    for config_key, experiments in config_experiments.items():
        if not experiments:
            continue
            
        # Get configuration details from the first experiment
        first_exp = experiments[0]
        temp = first_exp["configuration"]["temperature"]
        top_p = first_exp["configuration"]["top_p"]
        dtype = first_exp["configuration"]["dtype"]
        
        # Extract metrics from all experiments with this configuration
        accuracy_values = []
        accuracy_stderr_values = []
        token_length_values = []
        token_length_stderr_values = []
        
        for exp in experiments:
            try:
                accuracy_values.append(exp["results"]["accuracy"])
                accuracy_stderr_values.append(exp["results"]["accuracy_stderr"])
                token_length_values.append(exp["results"]["avg_token_length"])
                token_length_stderr_values.append(exp["results"]["token_length_stderr"])
            except KeyError as e:
                logging.warning(f"Missing key in experiment: {e}")
                continue
        
        # Calculate weighted averages
        # Use inverse of stderr as weights (higher precision = higher weight)
        accuracy_weights = [1 / (stderr if stderr > 0 else 1e-10) for stderr in accuracy_stderr_values]
        token_weights = [1 / (stderr if stderr > 0 else 1e-10) for stderr in token_length_stderr_values]
        
        avg_accuracy = compute_weighted_average(accuracy_values, accuracy_weights)
        avg_accuracy_stderr = compute_weighted_std_error(accuracy_values, accuracy_weights)
        avg_token_length = compute_weighted_average(token_length_values, token_weights)
        avg_token_length_stderr = compute_weighted_std_error(token_length_values, token_weights)
        
        # Store results
        config_results[config_key] = {
            "configuration": {
                "temperature": temp,
                "top_p": top_p,
                "dtype": dtype
            },
            "results": {
                "accuracy": avg_accuracy,
                "accuracy_stderr": avg_accuracy_stderr,
                "avg_token_length": avg_token_length,
                "token_length_stderr": avg_token_length_stderr,
                "num_experiments": len(experiments),
                "num_samples": experiments[0]["results"].get("num_samples", 30)  # Assuming all have same sample count
            }
        }
        
        logging.info(f"Config temp={temp}, top_p={top_p}, dtype={dtype}: "
                    f"Processed {len(experiments)} experiments, "
                    f"avg accuracy: {avg_accuracy:.4f}, "
                    f"avg token length: {avg_token_length:.2f}")
    
    return config_results

def find_best_configurations(config_results):
    """
    Find the best configurations based on accuracy.
    
    Args:
        config_results (dict): Results grouped by configuration.
        
    Returns:
        dict: Dictionary containing the best configurations.
    """
    best_configs = {
        "best_accuracy": {
            "config": None,
            "value": 0
        },
        "best_token_efficiency": {
            "config": None,
            "value": float('inf')
        },
        "best_combined": {
            "config": None,
            "accuracy": 0,
            "token_length": float('inf')
        }
    }
    
    # Find best accuracy and token efficiency
    for config_key, data in config_results.items():
        accuracy = data["results"]["accuracy"]
        token_length = data["results"]["avg_token_length"]
        
        # Best accuracy
        if accuracy > best_configs["best_accuracy"]["value"]:
            best_configs["best_accuracy"]["value"] = accuracy
            best_configs["best_accuracy"]["config"] = config_key
            
        # Best token efficiency (lowest length)
        if token_length < best_configs["best_token_efficiency"]["value"]:
            best_configs["best_token_efficiency"]["value"] = token_length
            best_configs["best_token_efficiency"]["config"] = config_key
            
        # Best combined (highest accuracy with reasonable token length)
        # We can define this as configurations with accuracy within 90% of best accuracy
        # and token length not more than 20% above minimum
        if accuracy > 0.9 * best_configs["best_accuracy"]["value"] and \
           (best_configs["best_combined"]["config"] is None or \
            accuracy > best_configs["best_combined"]["accuracy"]):
            best_configs["best_combined"]["accuracy"] = accuracy
            best_configs["best_combined"]["token_length"] = token_length
            best_configs["best_combined"]["config"] = config_key
    
    return best_configs

def main():
    parser = argparse.ArgumentParser(description='Analyze LLM experiment results')
    
    parser.add_argument('--results-path', type=str, default=DEFAULT_RESULTS_PATH,
                        help=f'Path to the results JSON file (default: {DEFAULT_RESULTS_PATH})')
    
    parser.add_argument('--output-path', type=str, default=DEFAULT_OUTPUT_PATH,
                        help=f'Path to save the analysis results (default: {DEFAULT_OUTPUT_PATH})')
    
    args = parser.parse_args()
    
    # Load results
    data = load_results(args.results_path)
    if data is None:
        return
    
    # Analyze by seed
    seed_results = analyze_by_seed(data)
    
    # Analyze by configuration
    config_results = analyze_by_config(data)
    
    # Find best configurations
    best_configs = find_best_configurations(config_results)
    
    # Compile final results
    final_results = {
        "seed_analysis": seed_results,
        "configuration_analysis": config_results,
        "best_configurations": best_configs
    }
    
    # Save results
    try:
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
        
        with open(args.output_path, 'w') as f:
            json.dump(final_results, f, indent=2)
        logging.info(f"Successfully saved analysis results to {args.output_path}")
        
        # Also save a more readable CSV for seed analysis
        seed_df = pd.DataFrame([
            {
                "seed": int(seed),
                "accuracy": data["results"]["accuracy"],
                "accuracy_stderr": data["results"]["accuracy_stderr"],
                "avg_token_length": data["results"]["avg_token_length"],
                "token_length_stderr": data["results"]["token_length_stderr"],
                "num_experiments": data["results"]["num_experiments"]
            }
            for seed, data in seed_results.items()
        ])
        
        csv_path = os.path.join(os.path.dirname(args.output_path), "seed_analysis.csv")
        seed_df.to_csv(csv_path, index=False)
        logging.info(f"Saved seed analysis CSV to {csv_path}")
        
        # Also save a more readable CSV for configuration analysis
        config_df = pd.DataFrame([
            {
                "temperature": data["configuration"]["temperature"],
                "top_p": data["configuration"]["top_p"],
                "dtype": data["configuration"]["dtype"],
                "accuracy": data["results"]["accuracy"],
                "accuracy_stderr": data["results"]["accuracy_stderr"],
                "avg_token_length": data["results"]["avg_token_length"],
                "token_length_stderr": data["results"]["token_length_stderr"],
                "num_experiments": data["results"]["num_experiments"]
            }
            for config, data in config_results.items()
        ])
        
        csv_path = os.path.join(os.path.dirname(args.output_path), "config_analysis.csv")
        config_df.to_csv(csv_path, index=False)
        logging.info(f"Saved configuration analysis CSV to {csv_path}")
        
    except Exception as e:
        logging.error(f"Error saving analysis results: {e}")

if __name__ == "__main__":
    main()
