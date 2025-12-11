import os
import re
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

def parse_terminal_log(filepath, label):
    """
    Parses the raw terminal log format:
    train  E: 12,284  I: 945,535  R: 410.6  S: 1.0  T: 13:22:05
    """
    data = []
    
    print(f"Reading log file: {filepath}")
    if not os.path.exists(filepath):
        print(f"ERROR: File not found: {filepath}")
        return None

    # Regex pattern to extract I (Iteration/Step) and S (Success)
    # Looking for 'I:' followed by number (with commas) and 'S:' followed by float
    pattern = re.compile(r"I:\s*([\d,]+).*?S:\s*([\d\.]+)")
    
    with open(filepath, 'r') as f:
        for line in f:
            if "train" in line and "S:" in line:
                match = pattern.search(line)
                if match:
                    try:
                        # Extract Step (remove commas)
                        step = int(match.group(1).replace(',', ''))
                        # Extract Success Rate
                        success = float(match.group(2))
                        
                        data.append({'Step': step, 'Success Rate': success, 'Method': label})
                    except ValueError:
                        continue

    if not data:
        print(f"WARNING: No valid training data found in {filepath}")
        return None
        
    df = pd.DataFrame(data)
    
    # Sort just in case logs got mixed
    df = df.sort_values('Step')
    
    # Calculate Smooth Moving Average (window=50 for noisy logs)
    df['Success Rate (Smoothed)'] = df['Success Rate'].rolling(window=50, min_periods=1).mean()
    
    print(f" -> Found {len(df)} data points for {label}")
    return df

def quantify_performance(df_a, df_b):
    print("\n" + "="*50)
    print("      QUANTITATIVE HYPOTHESIS VERIFICATION")
    print("="*50)

    stats = []

    for label, df in [("Curve A (Fine-tune)", df_a), ("Curve B (Scratch)", df_b)]:
        if df is None or df.empty:
            print(f"Skipping {label} (No Data)")
            continue
            
        max_success = df['Success Rate (Smoothed)'].max()
        final_success = df['Success Rate (Smoothed)'].iloc[-100:].mean() # Avg of last 100 steps
        
        # Find step where success first crossed 0.5 (50%) consistently
        # We assume "consistently" means the smoothed line stays above 0.5
        cross_50 = df[df['Success Rate (Smoothed)'] >= 0.5]
        step_50 = cross_50['Step'].iloc[0] if not cross_50.empty else float('inf')
        
        step_50_str = f"{step_50:,.0f}" if step_50 != float('inf') else "Never"

        print(f"\nMethod: {label}")
        print(f"  - Peak Success:       {max_success:.1%}")
        print(f"  - Final Success:      {final_success:.1%}")
        print(f"  - Steps to 50% Win:   {step_50_str}")
        
        stats.append((label, step_50))

    # Verification Logic
    print("\n" + "-"*50)
    print("CONCLUSION:")
    
    if len(stats) == 2:
        name_a, step_a = stats[0] # Fine-tune
        name_b, step_b = stats[1] # Scratch
        
        if step_a < step_b:
            if step_b == float('inf'):
                 print(f"✅ HYPOTHESIS CONFIRMED: Fine-tuning reached 50% success, while Scratch never did.")
            else:
                speedup = step_b / step_a
                print(f"✅ HYPOTHESIS CONFIRMED: Fine-tuning learned {speedup:.1f}x faster than Scratch.")
                print(f"   (Fine-tune took {step_a:,.0f} steps vs Scratch {step_b:,.0f} steps)")
        else:
            print("❌ HYPOTHESIS FAILED: Fine-tuning did not learn faster in this run.")
            print("   (Check if encoder freezing or domain gap is interfering)")

def plot_comparison(log_file_a, log_file_b):
    df_a = parse_terminal_log(log_file_a, "Fine-tune (Curve A)")
    df_b = parse_terminal_log(log_file_b, "Scratch (Curve B)")
    
    if df_a is None or df_b is None:
        print("Error: Could not load one or both log files.")
        return

    # Combine data
    combined = pd.concat([df_a, df_b])

    # Plotting
    plt.figure(figsize=(12, 7))
    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)
    
    # 1. Plot raw data faintly (optional, for reference)
    sns.lineplot(data=combined, x='Step', y='Success Rate', hue='Method', 
                 alpha=0.15, legend=False, palette=['#2ca02c', '#d62728'])

    # 2. Plot Smoothed Trend Line (Bold)
    ax = sns.lineplot(data=combined, x='Step', y='Success Rate (Smoothed)', hue='Method', 
                 linewidth=3, palette=['#2ca02c', '#d62728'])

    plt.title('Transfer Learning Efficiency: Success Rate over Time', fontsize=16, pad=15)
    plt.xlabel('Training Steps', fontsize=14)
    plt.ylabel('Success Rate (Smoothed)', fontsize=14)
    plt.ylim(-0.05, 1.05)
    
    # Add 50% threshold line
    plt.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
    plt.text(combined['Step'].min(), 0.51, ' 50% Success Threshold', color='gray', fontsize=10)

    plt.tight_layout()
    output_path = "final_comparison_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"\nGraph saved to: {os.path.abspath(output_path)}")
    plt.show()
    
    # Run numbers
    quantify_performance(df_a, df_b)

if __name__ == "__main__":
    # --- INSTRUCTIONS ---
    # 1. Locate your 'terminal_log.txt' files for both experiments.
    # 2. Paste their absolute paths below.
    
    # Example:
    # LOG_A = "/home/rl-group4/tdmpc2/outputs/finetune_run/terminal_log.txt"
    # LOG_B = "/home/rl-group4/tdmpc2/outputs/scratch_run/terminal_log.txt"
    
    # REPLACE THESE PATHS WITH YOUR ACTUAL FILES
    LOG_A = "/home/rl-group4/tdmpc2/tdmpc2/outputs/2025-12-10/12-03-36/terminal_log.txt" 
    LOG_B = "/home/rl-group4/tdmpc2/tdmpc2/outputs/2025-12-10/01-59-34/terminal_log.txt"

    plot_comparison(LOG_A, LOG_B)
