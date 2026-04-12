from pydantic import BaseModel
from typing import Optional


class StockInfo(BaseModel):
    ticker: str
    name: str
    sector: str
    min_date: str
    max_date: str
    n_rows: int


class AlgorithmInfo(BaseModel):
    id: str
    name: str
    description: str
    env_type: str


class TrainRequest(BaseModel):
    tickers: list[str]
    algorithm: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_iterations: int = 100
    steps_per_iter: int = 2048


class JobStatus(BaseModel):
    job_id: str
    status: str  # "pending", "training", "backtesting", "completed", "failed"
    progress: float = 0.0
    current_iteration: int = 0
    total_iterations: int = 0
    error: Optional[str] = None
    results: Optional[dict] = None
