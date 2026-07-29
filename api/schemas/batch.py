from pydantic import BaseModel, Field
from typing import List, Dict, Any
from api.schemas.prediction import PredictionResponse

class BatchPredictionRequest(BaseModel):
    tickers: List[str] = Field(..., min_length=1, max_length=10, description="List of 1 to 10 ticker symbols.")

class BatchPredictionResponse(BaseModel):
    results: List[PredictionResponse]
    failed: List[Dict[str, Any]]
    duration_ms: float
