import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import sys

# Default to the path in the new config
DEFAULT_LOG_DIR = "./logs/lidar_fast"

def plot_training_results(log_dir):
    """
    Finds eval.csv files and plots the reward curve.
    """
    if not os.path.exists(log_dir):
        print(f"Error: Directory {log_dir} does not exist.")
        print("Check if training has started or if 'work_dir' in config matches.")
        return

    # Look for eval.csv recursively
    csv_files = glob.glob(os.path.join(log_dir, "**", "eval.csv"), recursive=True)
    
    if not csv_files:
        print(f"No eval.csv files found yet in {log_dir}")
        print("Wait for the first evaluation step (step 5000 with new config).")
        return

    plt.figure(figsize=(10, 6))
    found_data = False
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Filter valid columns
            step_col = 'step' if 'step' in df.columns else 'total_time'
            # Look for reward columns
            reward_keys = [k for k in df.columns if 'reward' in k or 'return' in k]
            
            if step_col in df.columns and reward_keys:
                reward_col = reward_keys[0]
                seed = os.path.basename(os.path.dirname(csv_file))
                
                # Plot
                plt.plot(df[step_col], df[reward_col], linewidth=2, marker='o', label=f"Seed {seed}")
                found_data = True
                
                print(f"Plotting data from {csv_file} ({len(df)} points)")
                print(f"Current Best Reward: {df[reward_col].max():.2f}")
        except Exception as e:
            print(f"Skipping {csv_file}: {e}")

    if found_data:
        plt.xlabel("Environment Steps")
        plt.ylabel("Episode Reward")
        plt.title("Lidar Navigation Training Progress")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig("training_plot.png")
        print(f"\n[Success] Saved 'training_plot.png'. Download it to view.")
    else:
        print("Found CSV files but could not extract data.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG_DIR
    plot_training_results(path)
