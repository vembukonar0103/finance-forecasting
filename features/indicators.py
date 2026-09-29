import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import talib

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes 25 technical indicators using TA-Lib library and volatility regime features
    on the input DataFrame.
    The input DataFrame should contain 'open', 'high', 'low', 'close', and 'volume' columns.
    
    Returns a DataFrame with columns:
      - 'timestamp' (YYYY-MM-DD string)
      - 'event_timestamp' (datetime64[ns, UTC] for Feast point-in-time correctness)
      - 'created_timestamp' (datetime64[ns, UTC])
      - 25 technical indicator columns:
        sma10, sma20, sma50, sma200, ema10, ema20, ema50, ema200,
        rsi, macd, macd_sig, macd_hist, bb_upper, bb_middle, bb_lower,
        atr, adx, cci, slowk, slowd, willr, obv, adosc, volume_sma20, relative_volume
      - Custom volatility regime features:
        bb_width, atr_percentile_20d, rsi_slope_5d
    """
    # Normalize columns to lowercase to handle different cases
    df = df.copy()
    df.columns = [str(col).lower() for col in df.columns]
    
    # If standard timestamp columns are missing, reset index to handle Date/Datetime index
    time_cols = ['timestamps', 'timestamp', 'date', 'datetime']
    ts_col = None
    for tc in time_cols:
        if tc in df.columns:
            ts_col = tc
            break
            
    if ts_col is None:
        df = df.reset_index()
        df.columns = [str(col).lower() for col in df.columns]
        for tc in time_cols:
            if tc in df.columns:
                ts_col = tc
                break
                
    if ts_col is None:
        raise KeyError("Could not find date/timestamp column or Date/Datetime index in DataFrame.")

    # Sort by timestamp to ensure calculations are correct
    df = df.sort_values(ts_col).copy()
    
    o = df['open'].astype(float).values
    h = df['high'].astype(float).values
    l = df['low'].astype(float).values
    c = df['close'].astype(float).values
    v = df['volume'].astype(float).values
    
    feats = pd.DataFrame()
    
    # Datetime handling for Feast point-in-time correctness
    dt_series = pd.to_datetime(df[ts_col])
    if dt_series.dt.tz is None:
        dt_utc = dt_series.dt.tz_localize('UTC')
    else:
        dt_utc = dt_series.dt.tz_convert('UTC')
        
    feats['timestamp'] = dt_utc.dt.strftime('%Y-%m-%d')
    feats['event_timestamp'] = dt_utc
    feats['created_timestamp'] = pd.Timestamp.now(tz='UTC')
    
    # 1-4. Simple Moving Averages
    feats['sma10'] = talib.SMA(c, timeperiod=10)
    feats['sma20'] = talib.SMA(c, timeperiod=20)
    feats['sma50'] = talib.SMA(c, timeperiod=50)
    feats['sma200'] = talib.SMA(c, timeperiod=200)
    
    # 5-8. Exponential Moving Averages
    feats['ema10'] = talib.EMA(c, timeperiod=10)
    feats['ema20'] = talib.EMA(c, timeperiod=20)
    feats['ema50'] = talib.EMA(c, timeperiod=50)
    feats['ema200'] = talib.EMA(c, timeperiod=200)
    
    # 9. Relative Strength Index (RSI)
    feats['rsi'] = talib.RSI(c, timeperiod=14)
    
    # 10-12. MACD
    macd, macdsignal, macdhist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    feats['macd'] = macd
    feats['macd_sig'] = macdsignal
    feats['macd_hist'] = macdhist
    
    # 13-15. Bollinger Bands
    upper, middle, lower = talib.BBANDS(c, timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
    feats['bb_upper'] = upper
    feats['bb_middle'] = middle
    feats['bb_lower'] = lower
    
    # 16. Average True Range (ATR)
    feats['atr'] = talib.ATR(h, l, c, timeperiod=14)
    
    # 17. Average Directional Index (ADX)
    feats['adx'] = talib.ADX(h, l, c, timeperiod=14)
    
    # 18. Commodity Channel Index (CCI)
    feats['cci'] = talib.CCI(h, l, c, timeperiod=14)
    
    # 19-20. Stochastic Oscillator
    slowk, slowd = talib.STOCH(h, l, c, fastk_period=5, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    feats['slowk'] = slowk
    feats['slowd'] = slowd
    
    # 21. Williams %R
    feats['willr'] = talib.WILLR(h, l, c, timeperiod=14)

    # 22. On-Balance Volume (OBV)
    feats['obv'] = talib.OBV(c, v)

    # 23. Chaikin A/D Oscillator (ADOSC)
    feats['adosc'] = talib.ADOSC(h, l, c, v, fastperiod=3, slowperiod=10)

    # 24-25. Volume-based Indicators
    feats['volume_sma20'] = talib.SMA(v, timeperiod=20)
    # Handle division by zero or NaN safely
    feats['relative_volume'] = np.where(feats['volume_sma20'] > 0, v / feats['volume_sma20'], 1.0)
    
    # Custom Volatility Regime Features
    # 1. Bollinger Band Width
    bb_mid = np.where(feats['bb_middle'] > 0, feats['bb_middle'], 1e-5)
    feats['bb_width'] = (feats['bb_upper'] - feats['bb_lower']) / bb_mid
    
    # 2. 20-day rolling ATR percentile rank
    feats['atr_percentile_20d'] = feats['atr'].rolling(window=20, min_periods=5).apply(
        lambda s: pd.Series(s).rank(pct=True).iloc[-1] if len(s) > 0 else 0.5,
        raw=False
    ).fillna(0.5)
    
    # 3. 5-day RSI slope
    feats['rsi_slope_5d'] = ((feats['rsi'] - feats['rsi'].shift(5)) / 5.0).fillna(0.0)
    
    return feats
