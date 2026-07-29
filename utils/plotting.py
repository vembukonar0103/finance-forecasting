# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd
# pyrefly: ignore [missing-import]
import numpy as np
import os

def plot_forecast(actual_dates, actual_values, pred_dates, pred_values, output_path, title="Forecast vs Actual", quantiles=None):
    """
    Plots historical values and predicted future values, saving the image to output_path.
    quantiles can be a dict of: { 'p10': [...], 'p25': [...], 'p75': [...], 'p90': [...] } for shaded confidence bands.
    """
    plt.figure(figsize=(12, 6))
    
    # Plot historical actuals
    plt.plot(actual_dates, actual_values, label="Actual (Historical)", color="blue", linewidth=2)
    
    # Plot point forecast
    plt.plot(pred_dates, pred_values, label="Forecast Point (Median)", color="red", linestyle="--", linewidth=2)
    
    # Plot quantiles if available
    if quantiles:
        if "p10" in quantiles and "p90" in quantiles:
            plt.fill_between(pred_dates, quantiles["p10"], quantiles["p90"], color="red", alpha=0.15, label="10%-90% Confidence Interval")
        if "p25" in quantiles and "p75" in quantiles:
            plt.fill_between(pred_dates, quantiles["p25"], quantiles["p75"], color="red", alpha=0.25, label="25%-75% Confidence Interval")
            
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Date", fontsize=12)
    plt.ylabel("Price", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper left")
    plt.tight_layout()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    plt.savefig(output_path, dpi=150)
    plt.close()
