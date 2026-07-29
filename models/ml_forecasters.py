import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from features.indicators import build_features

class MLForecaster:
    def __init__(self, model_type="ridge", use_technical=False):
        """
        Parameters
        ----------
        model_type : str
            "ridge" or "rf" (Random Forest)
        use_technical : bool
            If True, uses returns lags + technical indicators.
            If False, uses only returns lags (Base Model).
        """
        self.model_type = model_type
        self.use_technical = use_technical
        self.models = {}  # Store models for each horizon h = 1..30
        self.scalers = {} # Scalers for each horizon h = 1..30
        self.best_params = {}
        
    def _prepare_features(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        """
        Builds feature matrix X and matches it with standard columns.
        """
        # Ensure standard columns are present
        df = df.copy()
        df.columns = [str(col).lower() for col in df.columns]
        
        # Calculate returns lags
        close = df['close'].astype(float)
        ret = close.pct_change()
        
        feats_df = pd.DataFrame(index=df.index)
        feats_df['ret_lag1'] = ret.shift(1)
        feats_df['ret_lag2'] = ret.shift(2)
        feats_df['ret_lag5'] = ret.shift(5)
        feats_df['ret_lag10'] = ret.shift(10)
        feats_df['ret_lag20'] = ret.shift(20)
        
        if self.use_technical:
            # Build all 25 technical indicators
            tech_df = build_features(df)
            # Align indices: tech_df has timestamp, let's merge or assign
            # Drop timestamp since we align by index
            tech_cols = [c for c in tech_df.columns if c not in ('timestamp', 'timestamps', 'date')]
            for col in tech_cols:
                feats_df[col] = tech_df[col].astype(float)
                
        # Forward fill and backward fill NaNs
        feats_df = feats_df.ffill().bfill()
        
        # Fill any remaining NaNs with 0.0
        feats_df = feats_df.fillna(0.0)
        
        return feats_df, close

    def fit(self, df: pd.DataFrame, horizon: int = 30, val_days: int = 60):
        """
        Fits 30 separate models (one for each horizon step h = 1..30).
        Performs grid search on the validation split to select hyperparameters.
        
        Parameters
        ----------
        df : pd.DataFrame
            The context data (up to date T)
        horizon : int
            Prediction horizon
        val_days : int
            Number of days from the end of the context to use as validation
        """
        # Separate training and validation indices
        N = len(df)
        if N <= val_days + 50:
            # Too short, use fallback parameters and fit on all
            train_idx = N
        else:
            train_idx = N - val_days
            
        df_train = df.iloc[:train_idx]
        df_val = df.iloc[train_idx:]
        
        # Prepare features for train and validation
        X_all, close_all = self._prepare_features(df)
        
        X_train = X_all.iloc[:train_idx]
        close_train = close_all.iloc[:train_idx]
        
        X_val = X_all.iloc[train_idx:]
        close_val = close_all.iloc[train_idx:]
        
        # Hyperparameter search grid
        if self.model_type == "ridge":
            param_grid = [0.01, 0.1, 1.0, 10.0, 100.0]
        else:
            param_grid = [3, 5, 8] # max_depth for Random Forest
            
        # We find the best hyperparameter by evaluating across all horizons on the validation set
        best_param = param_grid[len(param_grid) // 2] # Default fallback
        
        if N > val_days + 50:
            param_losses = {p: 0.0 for p in param_grid}
            
            # Temporary models for grid search
            for p in param_grid:
                total_mae = 0.0
                horizon_count = 0
                for h in range(1, horizon + 1):
                    # Target: return at h steps ahead
                    # r_{t, h} = close_{t+h} / close_t - 1
                    y_train = (close_train.shift(-h) / close_train - 1.0).dropna()
                    X_tr_h = X_train.loc[y_train.index]
                    
                    if len(y_train) < 10:
                        continue
                        
                    # Standardize features
                    scaler = StandardScaler()
                    X_tr_h_scaled = scaler.fit_transform(X_tr_h)
                    
                    # Instantiate model
                    if self.model_type == "ridge":
                        model = Ridge(alpha=p)
                    else:
                        model = RandomForestRegressor(max_depth=p, n_estimators=50, random_state=42)
                        
                    model.fit(X_tr_h_scaled, y_train)
                    
                    # Predict on validation
                    y_val_actual = (close_val.shift(-h) / close_val - 1.0).dropna()
                    if len(y_val_actual) == 0:
                        continue
                    X_val_h = X_val.loc[y_val_actual.index]
                    X_val_h_scaled = scaler.transform(X_val_h)
                    
                    y_val_pred = model.predict(X_val_h_scaled)
                    
                    mae = np.mean(np.abs(y_val_actual - y_val_pred))
                    total_mae += mae
                    horizon_count += 1
                    
                if horizon_count > 0:
                    param_losses[p] = total_mae / horizon_count
                    
            best_param = min(param_losses, key=param_losses.get)
            
        self.best_params[self.model_type] = best_param
        
        # Now refit the models for each horizon h on the FULL context df using the best parameter
        for h in range(1, horizon + 1):
            y_all = (close_all.shift(-h) / close_all - 1.0).dropna()
            X_all_h = X_all.loc[y_all.index]
            
            scaler = StandardScaler()
            X_all_h_scaled = scaler.fit_transform(X_all_h)
            
            if self.model_type == "ridge":
                model = Ridge(alpha=best_param)
            else:
                model = RandomForestRegressor(max_depth=best_param, n_estimators=100, random_state=42)
                
            model.fit(X_all_h_scaled, y_all)
            
            self.models[h] = model
            self.scalers[h] = scaler

    def predict(self, df: pd.DataFrame, horizon: int = 30) -> pd.DataFrame:
        """
        Predicts close prices for the next `horizon` days.
        Uses the last row of df as features to predict return at each step,
        then reconstructs future prices from the last close price.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with index of future dates (business days or calendar days)
            containing column 'point'.
        """
        # Ensure we have standard columns and sorted by date
        df_sorted = df.sort_values('timestamps').copy()
        last_close = df_sorted['close'].iloc[-1]
        last_timestamp = pd.to_datetime(df_sorted['timestamps'].iloc[-1])
        
        # Features at the last timestamp
        X_all, _ = self._prepare_features(df_sorted)
        last_features = X_all.iloc[-1:].copy()
        
        # Generate predictions for each horizon h
        pred_returns = []
        for h in range(1, horizon + 1):
            if h not in self.models:
                # Fallback to zero return if model not trained
                pred_returns.append(0.0)
                continue
                
            model = self.models[h]
            scaler = self.scalers[h]
            
            last_features_scaled = scaler.transform(last_features)
            pred_ret = model.predict(last_features_scaled)[0]
            pred_returns.append(pred_ret)
            
        # Reconstruct future prices
        pred_prices = [last_close * (1.0 + r) for r in pred_returns]
        
        # Generate future business days (default, to be aligned later)
        # Note: We can guess frequency based on calendar days diff, or standard B
        freq = 'B'
        if len(df_sorted) > 2:
            time_diffs = pd.to_datetime(df_sorted['timestamps']).diff().dropna()
            if time_diffs.min().days == 1 and time_diffs.max().days == 1:
                freq = 'D' # trades calendar days (BTC-USD)
                
        future_dates = pd.date_range(
            start=last_timestamp,
            periods=horizon + 1,
            freq=freq
        )[1:]
        
        pred_df = pd.DataFrame({
            'point': pred_prices
        }, index=future_dates)
        
        return pred_df
