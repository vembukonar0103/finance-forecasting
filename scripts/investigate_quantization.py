"""
scripts/investigate_quantization.py
-----------------------------------
Investigation script for MPS model quantization.
Checks dynamic INT8 quantization for Kronos and FP16 (.half()) for TimesFM.
"""
import torch
import time
import os
import sys
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.kronos_model import KronosForecast
from models.timesfm_model import TimesFMForecast
from utils.data_loader import load_data

def check_device(model, model_name):
    devices = set(p.device.type for p in model.parameters())
    print(f"[{model_name}] Parameter devices: {devices}")
    return "cpu" not in devices and "mps" in devices

def investigate_kronos():
    print("=== Kronos INT8 Quantization Investigation ===")
    if not torch.backends.mps.is_available():
        print("MPS not available. Skipping.")
        return
        
    print("Loading original Kronos model onto MPS...")
    kronos = KronosForecast()
    df = load_data("AAPL")
    
    # Warmup
    _ = kronos.predict(df)
    
    # Time original
    start = time.time()
    for _ in range(3):
        _ = kronos.predict(df)
    end = time.time()
    orig_time = (end - start) / 3
    print(f"Original Kronos (FP32) Avg Inference Time: {orig_time:.4f}s")
    check_device(kronos.model, "Kronos-FP32")
    
    # Quantize
    print("\nApplying Dynamic INT8 Quantization to Kronos...")
    try:
        quantized_model = torch.quantization.quantize_dynamic(
            kronos.model,
            {torch.nn.Linear}, 
            dtype=torch.qint8
        )
        kronos.model = quantized_model
        
        check_device(kronos.model, "Kronos-INT8")
        
        # Test if it runs without error
        try:
            start = time.time()
            for _ in range(3):
                _ = kronos.predict(df)
            end = time.time()
            quant_time = (end - start) / 3
            print(f"Quantized Kronos (INT8) Avg Inference Time: {quant_time:.4f}s")
            print(f"Speedup: {orig_time / quant_time:.2f}x")
        except Exception as e:
            print(f"Error during INT8 inference: {e}")
            
    except Exception as e:
        print(f"Error during quantization: {e}")

def investigate_timesfm():
    print("\n=== TimesFM FP16 Quantization Investigation ===")
    if not torch.backends.mps.is_available():
        print("MPS not available. Skipping.")
        return
        
    print("Loading original TimesFM model onto MPS...")
    timesfm = TimesFMForecast()
    df = load_data("AAPL")
    
    # Warmup
    _ = timesfm.predict(df, horizon=30)
    
    # Time original
    start = time.time()
    for _ in range(3):
        _ = timesfm.predict(df, horizon=30)
    end = time.time()
    orig_time = (end - start) / 3
    # Apply Half Precision
    print("\nApplying .half() (FP16) to TimesFM...")
    try:
        if hasattr(timesfm.model, 'model'):
            timesfm.model.model = timesfm.model.model.half()
        else:
            timesfm.model = timesfm.model.half()
        
        # Test if it runs without error
        try:
            start = time.time()
            for _ in range(3):
                _ = timesfm.predict(df, horizon=30)
            end = time.time()
            quant_time = (end - start) / 3
            print(f"Quantized TimesFM (FP16) Avg Inference Time: {quant_time:.4f}s")
            print(f"Speedup: {orig_time / quant_time:.2f}x")
        except Exception as e:
            print(f"Error during FP16 inference: {e}")
            
    except Exception as e:
        print(f"Error during FP16 conversion: {e}")

if __name__ == "__main__":
    investigate_kronos()
    investigate_timesfm()
