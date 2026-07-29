import sys
import os
import pandas as pd
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import load_data
from models.timesfm_model import TimesFMForecast
from utils.plotting import plot_forecast

def main():
    ticker = "AAPL"
    print(f"Loading data for {ticker} (last 500 days)...")
    # Retrieve 500 days of data
    df = load_data(ticker, period="2y", interval="1d")
    print(f"Loaded {df.shape[0]} rows.")
    
    # Hold out the last 30 days
    holdout_days = 30
    df_context = df.iloc[:-holdout_days]
    df_holdout = df.iloc[-holdout_days:]
    
    print(f"Context data shape: {df_context.shape}")
    print(f"Holdout data shape: {df_holdout.shape}")
    
    print("Initializing TimesFM Forecast Wrapper...")
    model = TimesFMForecast()
    
    print("Running forecasting...")
    pred_df, prob_plus_5 = model.predict(df_context, target_col='close', horizon=holdout_days)
    
    print("\n--- Inference Output Shape Check ---")
    print(f"Predictions DataFrame shape: {pred_df.shape}")
    print(f"Columns: {list(pred_df.columns)}")
    
    print("\n--- Day-30 Forecast Details ---")
    day_30_idx = holdout_days - 1
    print(f"Day-30 Date: {pred_df.index[day_30_idx].strftime('%Y-%m-%d')}")
    print(f"Point Forecast (Median): {pred_df['point'].iloc[day_30_idx]:.2f}")
    print(f"P10-P90 Band: [{pred_df['p10'].iloc[day_30_idx]:.2f}, {pred_df['p90'].iloc[day_30_idx]:.2f}]")
    print(f"P25-P75 Band: [{pred_df['p25'].iloc[day_30_idx]:.2f}, {pred_df['p75'].iloc[day_30_idx]:.2f}]")
    print(f"Heuristic P(>= +5% in 30 days): {prob_plus_5:.4f} ({prob_plus_5*100:.2f}%)")
    
    # Verify how many actuals fell inside the P10-P90 band
    actuals = df_holdout['close'].values
    p10_vals = pred_df['p10'].values
    p90_vals = pred_df['p90'].values
    
    inside_count = sum(1 for act, p10, p90 in zip(actuals, p10_vals, p90_vals) if p10 <= act <= p90)
    hit_rate = inside_count / holdout_days
    print(f"\nHoldout Band Accuracy (P10-P90): {inside_count}/{holdout_days} days ({hit_rate*100:.2f}%)")
    
    # Plotting
    plot_path = f"plots/{ticker}_timesfm_forecast.png"
    print(f"Saving forecast plot to {plot_path}...")
    
    # Select trailing actuals to show context in plot (e.g. 100 days)
    plot_actuals = df.tail(100)
    actual_dates = pd.to_datetime(plot_actuals['timestamps'])
    actual_values = plot_actuals['close']
    
    pred_dates = pd.to_datetime(pred_df.index)
    
    quantiles_dict = {
        'p10': pred_df['p10'].values,
        'p25': pred_df['p25'].values,
        'p75': pred_df['p75'].values,
        'p90': pred_df['p90'].values
    }
    
    plot_forecast(
        actual_dates=actual_dates,
        actual_values=actual_values,
        pred_dates=pred_dates,
        pred_values=pred_df['point'].values,
        output_path=plot_path,
        title=f"{ticker} 30-Day TimesFM Forecast & Quantile Bands",
        quantiles=quantiles_dict
    )
    print("Forecast run completed successfully!")

if __name__ == "__main__":
    main()
