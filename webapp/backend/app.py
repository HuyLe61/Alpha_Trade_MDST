import sys
import os
import uuid
import asyncio
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

# In Docker the ML modules live under /project; locally they're two levels up
PROJECT_ROOT = Path("/project") if Path("/project/envs").exists() else Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

STATIC_DIR = Path(__file__).resolve().parent / "static"

from database import init_db, get_available_tickers, load_stock_data
from schemas import StockInfo, AlgorithmInfo, TrainRequest, JobStatus
from training import run_training_job


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Auto-populate stock data on first boot (empty volume)
    if not get_available_tickers():
        print("Database empty — fetching stock data from yfinance...", flush=True)
        from fetch_data import fetch_and_store
        fetch_and_store()
    yield

app = FastAPI(title="AlgoTrade API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store
jobs: dict[str, dict] = {}
# Per-job message queues for WebSocket streaming
job_queues: dict[str, list[asyncio.Queue]] = {}

ALGORITHMS = [
    AlgorithmInfo(
        id="mvp",
        name="MVP (Custom PPO)",
        description="Our custom PPO agent with portfolio weight allocation, Sharpe reward, and classical baseline comparisons (equal-weight, buy-and-hold, min-variance).",
        env_type="portfolio",
    ),
    AlgorithmInfo(
        id="ppo",
        name="PPO (Stable Baselines3)",
        description="Proximal Policy Optimization — on-policy, clipped surrogate objective. Stable and sample-efficient.",
        env_type="naive",
    ),
    AlgorithmInfo(
        id="ddpg",
        name="DDPG (Stable Baselines3)",
        description="Deep Deterministic Policy Gradient — off-policy, continuous actions. Uses replay buffer and target networks.",
        env_type="naive",
    ),
    AlgorithmInfo(
        id="sac",
        name="SAC (Stable Baselines3)",
        description="Soft Actor-Critic — off-policy, entropy-regularized. Maximizes reward and policy entropy for robust exploration.",
        env_type="naive",
    ),
    AlgorithmInfo(
        id="a2c",
        name="A2C (Stable Baselines3)",
        description="Advantage Actor-Critic — on-policy, synchronous. Simple and fast baseline for continuous control.",
        env_type="naive",
    ),
    AlgorithmInfo(
        id="td3",
        name="TD3 (Stable Baselines3)",
        description="Twin Delayed DDPG — off-policy with twin critics, delayed policy updates, and target smoothing for stability.",
        env_type="naive",
    ),
]


@app.get("/api/stocks", response_model=list[StockInfo])
def list_stocks():
    return get_available_tickers()


@app.get("/api/algorithms", response_model=list[AlgorithmInfo])
def list_algorithms():
    return ALGORITHMS


@app.post("/api/train")
def start_training(req: TrainRequest):
    job_id = str(uuid.uuid4())[:8]
    jobs[job_id] = {
        "job_id": job_id,
        "status": "pending",
        "progress": 0.0,
        "current_iteration": 0,
        "total_iterations": req.n_iterations,
        "error": None,
        "results": None,
    }
    job_queues[job_id] = []

    def progress_callback(data: dict):
        jobs[job_id].update(data)
        for q in job_queues.get(job_id, []):
            try:
                q.put_nowait(data)
            except asyncio.QueueFull:
                pass

    thread = threading.Thread(
        target=_run_job,
        args=(job_id, req, progress_callback),
        daemon=True,
    )
    thread.start()
    return {"job_id": job_id}


def _run_job(job_id: str, req: TrainRequest, callback):
    try:
        callback({"status": "training", "progress": 0.0})
        results = run_training_job(
            tickers=req.tickers,
            algorithm=req.algorithm,
            train_start=req.train_start,
            train_end=req.train_end,
            test_start=req.test_start,
            test_end=req.test_end,
            n_iterations=req.n_iterations,
            steps_per_iter=req.steps_per_iter,
            progress_callback=callback,
        )
        callback({"status": "completed", "progress": 1.0, "results": results})
    except Exception as e:
        import traceback
        traceback.print_exc()
        callback({"status": "failed", "error": str(e)})


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str):
    if job_id not in jobs:
        return {"error": "Job not found"}
    return jobs[job_id]


@app.websocket("/ws/train/{job_id}")
async def ws_training(websocket: WebSocket, job_id: str):
    await websocket.accept()
    queue: asyncio.Queue = asyncio.Queue(maxsize=500)

    if job_id not in job_queues:
        job_queues[job_id] = []
    job_queues[job_id].append(queue)

    try:
        # Send current state immediately
        if job_id in jobs:
            await websocket.send_json(jobs[job_id])

        while True:
            try:
                data = await asyncio.wait_for(queue.get(), timeout=1.0)
                await websocket.send_json(_make_serializable(data))
                if data.get("status") in ("completed", "failed"):
                    break
            except asyncio.TimeoutError:
                # Check if job finished while we were waiting
                if job_id in jobs and jobs[job_id].get("status") in ("completed", "failed"):
                    await websocket.send_json(_make_serializable(jobs[job_id]))
                    break
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in job_queues:
            try:
                job_queues[job_id].remove(queue)
            except ValueError:
                pass


def _make_serializable(obj):
    """Convert numpy types to Python natives for JSON serialization."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: _make_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_make_serializable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# --- SPA static file serving (production) ---
if STATIC_DIR.exists():
    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file = STATIC_DIR / full_path
        if file.is_file():
            return FileResponse(file)
        return FileResponse(STATIC_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port)
