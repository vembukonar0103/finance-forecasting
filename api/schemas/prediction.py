from pydantic import BaseModel
from typing import Dict, Any, List, Optional

class PredictionResponse(BaseModel):
    ticker: str
    model_used: str = "timesfm"
    generated_at: str
    current_price: float
    predicted_price: float
    horizon_days: int = 30
    expected_return_pct: float
    confidence_score: float
    prediction_interval: Dict[str, float]
    technical_indicators: Dict[str, Optional[float]]
    signal: str
    reasoning: List[str]
    
    # Extra optional fields to preserve existing functionality/tests
    timesfm_quantiles: Optional[Dict[str, Any]] = None
    kronos_pattern: Optional[Dict[str, Any]] = None

class KronosInferenceResponse(BaseModel):
    ticker: str
    timeframe: str
    pattern: str
    label: str
    confidence: float
    forecast_step: int
    tokenization_ms: float
    cache_hit: bool
    attention_heatmap: Optional[Dict[str, Any]] = None

class TimesFMInferenceResponse(BaseModel):
    ticker: str
    horizon_days: int
    timestamps: List[str]
    point_forecast: List[float]
    p10: List[float]
    p25: List[float]
    p50: List[float]
    p75: List[float]
    p90: List[float]
    probability_plus_5_pct: float

class TimesFMHorizonQuantiles(BaseModel):
    timestamps: List[str]
    point_forecast: List[float]
    p10: List[float]
    p25: List[float]
    p50: List[float]
    p75: List[float]
    p90: List[float]
    probability_plus_5_pct: float
    inference_ms: float
    cache_hit: bool
    fallback: bool = False
    provider: str = "timesfm"

class TimesFMFanChartResponse(BaseModel):
    ticker: str
    horizons: Dict[str, TimesFMHorizonQuantiles]
