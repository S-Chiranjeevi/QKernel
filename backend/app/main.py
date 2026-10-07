"""Persistent API worker. Heavy scientific dependencies are loaded on demand."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
# Use the mounted persistent volume on hosted workers, /tmp on serverless, and
# the project directory for local development.
SERVERLESS = bool(os.environ.get("VERCEL"))
_data_dir = os.environ.get("QKERNEL_DATA_DIR")
_writable = (Path(_data_dir).expanduser() if _data_dir else
             Path("/tmp/qkernel") if SERVERLESS or not os.access(ROOT, os.W_OK) else ROOT)
RESULTS = _writable / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
DATA_DIR = _writable / "datasets"
DATA_DIR.mkdir(parents=True, exist_ok=True)
KERNEL_CACHE = RESULTS / "kernel-cache"
KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
KERNEL_CACHE_LIMIT = 512 * 1024 * 1024
os.environ.setdefault("MPLCONFIGDIR", str(_writable / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 250 * 1024 * 1024


def _kernel_cache_get_or_compute(cache_key: str, compute):
    """Load a cached pair of kernel matrices, or compute and persist the pair."""
    cache_file = KERNEL_CACHE / f"{cache_key}.npz"
    import numpy as np

    if cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as cached:
                return cached["train"], cached["test"], True
        except Exception:
            # A truncated or incompatible cache entry should not block a run.
            cache_file.unlink(missing_ok=True)

    train, test = compute()
    try:
        temp_cache = KERNEL_CACHE / f"{cache_key}.{uuid.uuid4().hex}.tmp.npz"
        np.savez_compressed(temp_cache, train=train, test=test)
        temp_cache.replace(cache_file)
        cache_size = sum(path.stat().st_size for path in KERNEL_CACHE.glob("*.npz"))
        if cache_size > KERNEL_CACHE_LIMIT:
            cached_files = sorted(KERNEL_CACHE.glob("*.npz"), key=lambda path: path.stat().st_mtime)
            for old_cache in cached_files:
                if old_cache == cache_file:
                    continue
                old_size = old_cache.stat().st_size
                old_cache.unlink(missing_ok=True)
                cache_size -= old_size
                if cache_size <= KERNEL_CACHE_LIMIT:
                    break
    except OSError:
        # The cache is an optimization; a read-only or full disk must not fail a run.
        pass
    return train, test, False
REQUIRED = {*(f"V{i}" for i in range(1, 29)), "Time", "Amount", "Class"}
IDEAL_SEEDS = [42, 123, 456, 789, 1000, 2024, 2025, 2026, 31415, 27182]
dataset: dict[str, Any] = {"frame": None, "name": None, "sha256": None, "synthetic": False}
jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()

app = FastAPI(title="Q-Fraud Intelligence API", version="0.1.0",
              description="Local software simulation research API. No quantum hardware or cloud services.")
_allowed_origins = [origin.strip() for origin in os.environ.get(
    "CORS_ALLOW_ORIGINS", "http://127.0.0.1:5173,http://localhost:5173"
).split(",") if origin.strip()]
app.add_middleware(CORSMiddleware, allow_origins=_allowed_origins,
                   allow_credentials=False, allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
                   allow_headers=["Content-Type"])


def _pd():
    try:
        import pandas as pd
        return pd
    except ImportError as exc:
        raise HTTPException(503, "Dataset support needs pandas. Install backend/requirements.txt.") from exc


def _load_csv(raw: bytes, name: str):
    pd = _pd()
    import numpy as np
    try:
        frame = pd.read_csv(io.BytesIO(raw))
    except Exception as exc:
        raise HTTPException(400, "Could not read this CSV. Check its encoding and delimiter.") from exc
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise HTTPException(400, f"Required column '{missing[0]}' was not found.")
    if frame.empty:
        raise HTTPException(400, "The CSV has a header but no transaction rows.")
    if frame[list(REQUIRED)].isna().any().any():
        col = str(frame[list(REQUIRED)].columns[frame[list(REQUIRED)].isna().any()][0])
        raise HTTPException(400, f"Column '{col}' contains missing values. Clean the CSV and try again.")
    for col in sorted(REQUIRED):
        try: numeric=pd.to_numeric(frame[col],errors="raise")
        except Exception as exc: raise HTTPException(400,f"Column '{col}' must contain numeric values.") from exc
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise HTTPException(400,f"Column '{col}' contains an infinite or non-finite value.")
        frame[col]=numeric
    values = set(frame["Class"].unique().tolist())
    if not values.issubset({0, 1, 0.0, 1.0}):
        raise HTTPException(400, "Column 'Class' must contain only 0 and 1.")
    dataset.update(frame=frame, name=name, sha256=hashlib.sha256(raw).hexdigest(), synthetic=False)
    if _data_dir:
        # Keep the most recently uploaded dataset on the mounted volume so a
        # normal service restart does not require the user to upload it again.
        saved_csv = DATA_DIR / "uploaded.csv"
        saved_meta = DATA_DIR / "uploaded.json"
        temp_csv = DATA_DIR / f"uploaded.{uuid.uuid4().hex}.tmp"
        try:
            temp_csv.write_bytes(raw)
            temp_csv.replace(saved_csv)
            saved_meta.write_text(json.dumps({"name": name}), encoding="utf-8")
        except OSError as exc:
            temp_csv.unlink(missing_ok=True)
            raise HTTPException(507, "Dataset validated, but the persistent data disk could not save it. Check available disk space.") from exc
    return _statistics(frame)


@app.on_event("startup")
def restore_uploaded_dataset():
    saved_csv = DATA_DIR / "uploaded.csv"
    saved_meta = DATA_DIR / "uploaded.json"
    if _data_dir and saved_csv.is_file():
        try:
            metadata = json.loads(saved_meta.read_text(encoding="utf-8")) if saved_meta.is_file() else {}
            _load_csv(saved_csv.read_bytes(), str(metadata.get("name") or "uploaded.csv"))
        except Exception as exc:
            # A corrupt saved file should not prevent the API health endpoint
            # from starting; the user can upload a corrected CSV.
            print(f"Could not restore persisted dataset: {exc}")


def _statistics(frame):
    n = int(len(frame)); fraud = int((frame["Class"] == 1).sum())
    return {"name": dataset["name"], "sha256": dataset["sha256"], "rows": n,
            "features": int(len([c for c in frame.columns if c not in {"Class", "Time"}])),
            "fraud": fraud, "legitimate": n - fraud, "fraud_rate": fraud / n if n else 0,
            "duplicates": int(frame.duplicated().sum()), "missing_values": int(frame.isna().sum().sum()),
            "class_counts": [{"label": "Legitimate", "count": n - fraud}, {"label": "Fraud", "count": fraud}],
            "synthetic": bool(dataset.get("synthetic", False))}


@app.get("/api/health")
def health():
    default_csv = (ROOT / "datasets" / "raw" / "creditcard.csv").is_file()
    serverless = SERVERLESS
    persistent_volume = bool(_data_dir)
    remote_worker = os.environ.get("QKERNEL_DEPLOYMENT") == "render"
    return {"status": "ok", "mode": "local software simulation",
            "execution_mode": "vercel-serverless" if serverless else "persistent-worker" if remote_worker else "local-worker",
            "persistent_background_jobs": not serverless,
            "persistent_results": persistent_volume or (not serverless and not remote_worker),
            "supports_local_paths": not serverless and not remote_worker,
            "dataset_loaded": dataset["frame"] is not None,
            "dataset_name": dataset.get("name"), "default_dataset_available": default_csv,
            "results_count": len(list(RESULTS.glob("*/run.json")))}


@app.post("/api/dataset/demo")
def generate_demo_dataset():
    """Generate a realistic synthetic credit-card dataset for immediate research testing."""
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Dataset sessions are temporary on Vercel. Connect a persistent FastAPI worker for experiments.")
    pd = _pd()
    import numpy as np
    rng = np.random.default_rng(42)
    n_legit = 1000
    n_fraud = 200
    total = n_legit + n_fraud
    v_cols = [f"V{i}" for i in range(1, 29)]
    scales = np.exp(-np.linspace(0, 2.5, 28))
    x_legit = rng.normal(0, scales, size=(n_legit, 28))
    x_fraud = rng.normal(0, scales, size=(n_fraud, 28))
    x_fraud[:, 13] -= 3.5  # V14
    x_fraud[:, 11] -= 3.0  # V12
    x_fraud[:, 9] -= 2.5   # V10
    x_fraud[:, 16] -= 2.0  # V17
    x_fraud[:, 3] += 2.0   # V4
    X = np.vstack([x_legit, x_fraud])
    classes = np.array([0] * n_legit + [1] * n_fraud)
    amounts_legit = rng.lognormal(mean=3.0, sigma=1.2, size=n_legit)
    amounts_fraud = rng.lognormal(mean=3.8, sigma=1.5, size=n_fraud)
    amounts = np.concatenate([amounts_legit, amounts_fraud]).round(2)
    times = np.sort(rng.uniform(0, 172800, size=total)).round(1)
    indices = rng.permutation(total)
    data = {col: X[indices, i] for i, col in enumerate(v_cols)}
    data["Time"] = times[indices]
    data["Amount"] = amounts[indices]
    data["Class"] = classes[indices]
    frame = pd.DataFrame(data)
    csv_bytes = frame.to_csv(index=False).encode("utf-8")
    dataset.update(frame=frame, name="synthetic_creditcard_demo.csv",
                   sha256=hashlib.sha256(csv_bytes).hexdigest(), synthetic=True)
    return _statistics(frame)


@app.post("/api/dataset/load-default")
def load_default_dataset():
    """Load default dataset from datasets/raw/creditcard.csv if it exists, otherwise generate demo dataset."""
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Default dataset loading is unavailable on Vercel. Connect a persistent FastAPI worker.")
    default_path = ROOT / "datasets" / "raw" / "creditcard.csv"
    if default_path.is_file():
        return _load_csv(default_path.read_bytes(), default_path.name)
    return generate_demo_dataset()


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Dataset uploads are not durable on Vercel. Connect a persistent FastAPI worker for training and scoring.")
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(400, "Choose a CSV file.")
    raw = await file.read(MAX_UPLOAD + 1)
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "CSV exceeds the 250 MB upload limit.")
    return _load_csv(raw, Path(file.filename).name)


class PathRequest(BaseModel):
    path: str = Field(min_length=1, max_length=1000)


@app.post("/api/dataset/use-path")
def use_path(req: PathRequest):
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Local filesystem paths are available only in local development.")
    path = Path(req.path).expanduser().resolve()
    if not path.is_file() or path.suffix.lower() != ".csv":
        raise HTTPException(400, "Choose an existing local .csv file.")
    if path.stat().st_size > MAX_UPLOAD:
        raise HTTPException(413, "CSV exceeds the 250 MB limit.")
    try:
        return _load_csv(path.read_bytes(), path.name)
    except OSError as exc:
        raise HTTPException(400, "The local CSV could not be read. Check the path and permissions.") from exc


@app.get("/api/dataset/statistics")
def statistics():
    if dataset["frame"] is None:
        raise HTTPException(404, "No dataset is loaded. Upload a CSV or enter a local path first.")
    return _statistics(dataset["frame"])


class CircuitRequest(BaseModel):
    qubits: int = Field(default=4, ge=2, le=8)
    repeats: int = Field(default=1, ge=1, le=3)
    entangle: bool = True
    x: list[float] | None = None


@app.get("/api/circuit")
def circuit_get(qubits: int = 4, repeats: int = 1, entangle: bool = True,
                x: str | None = Query(default=None, description="Comma-separated feature-map angles from 0 to pi")):
    try:
        values = [float(value.strip()) for value in x.split(",")] if x else None
    except ValueError as exc:
        raise HTTPException(400, "Circuit angles must be comma-separated numbers.") from exc
    return _circuit(CircuitRequest(qubits=qubits, repeats=repeats, entangle=entangle, x=values))


def _circuit(req: CircuitRequest):
    try:
        from qiskit import QuantumCircuit
    except ImportError as exc:
        raise HTTPException(503, "Circuit drawing needs Qiskit. Install backend/requirements.txt.") from exc
    qc = QuantumCircuit(req.qubits)
    zero = req.x or [0.0] * req.qubits
    if len(zero) != req.qubits or any(not math.isfinite(v) for v in zero):
        raise HTTPException(400, "The circuit input vector must have one finite value per qubit.")
    for _ in range(req.repeats):
        for i, v in enumerate(zero): qc.h(i); qc.rz(2 * v, i)
        if req.entangle:
            for i in range(req.qubits):
                for j in range(i + 1, req.qubits): qc.rzz(2 * (math.pi - zero[i]) * (math.pi - zero[j]), i, j)
    try: text = qc.draw(output="text", fold=100).single_string()
    except Exception: text = str(qc.draw(output="text", fold=100))
    svg = None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = qc.draw(output="mpl")
        buffer = io.StringIO()
        fig.savefig(buffer, format="svg", bbox_inches="tight")
        plt.close(fig)
        svg = buffer.getvalue()
    except Exception:
        pass
    transpiled = qc.decompose(reps=3)
    counts = transpiled.count_ops(); two = sum(n for gate, n in counts.items() if gate in {"cx", "cz", "swap"})
    return {"text": text, "svg": svg, "depth": transpiled.depth(), "gate_count": sum(counts.values()),
            "two_qubit_gate_count": two, "counts": dict(counts), "qubits": req.qubits, "repeats": req.repeats,
            "entangled": req.entangle, "label": "Software simulation"}


@app.get("/api/runtime-estimate")
def runtime_estimate(qubits: int = 4, repeats: int = 1, fraud_labels: int = 10, seeds: int = 10, entangle: bool = True):
    """Calibrate state generation and vectorized products like the benchmark worker."""
    if qubits not in (4, 6, 8) or repeats not in (1, 2, 3) or fraud_labels not in (6, 10, 20, 50) or not 1 <= seeds <= 10:
        raise HTTPException(400, "Choose supported qubit, repeat, fraud-label and seed values.")
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise HTTPException(503, "Runtime calibration needs NumPy and Qiskit. Install backend/requirements.txt.") from exc
    def state(values):
        qc=QuantumCircuit(qubits)
        for _ in range(repeats):
            for i,x in enumerate(values):qc.h(i);qc.rz(2*float(x),i)
            if entangle:
                for i in range(qubits):
                    for j in range(i+1,qubits):qc.rzz(2*(math.pi-values[i])*(math.pi-values[j]),i,j)
        return np.asarray(Statevector.from_instruction(qc).data)
    rng=np.random.default_rng(17);cal_train=24;cal_test=12
    values=rng.uniform(0,math.pi,(cal_train+cal_test,qubits));t0=time.perf_counter()
    states=np.stack([state(x) for x in values]);state_seconds=(time.perf_counter()-t0)/len(values)
    left,right=states[:cal_train],states[cal_train:];t0=time.perf_counter()
    np.abs(left@left.conj().T)**2;np.abs(right@left.conj().T)**2
    dense_products=cal_train*cal_train+cal_test*cal_train
    product_seconds=(time.perf_counter()-t0)/dense_products
    train=200+fraud_labels;test=500
    unique_train=train*(train+1)//2;test_train=test*train
    entries=(unique_train+test_train)*seeds
    estimated=(state_seconds*(train+test)+product_seconds*(train*train+test*train))*seeds
    return {"label":"Software simulation","mode":"ideal exact statevector","qubits":qubits,"repeats":repeats,"entangle":entangle,
            "calibration_states":len(values),"calibration_matrix_rows":[cal_train,cal_test],"seconds_per_state":state_seconds,
            "seconds_per_dense_product_entry":product_seconds,"estimated_kernel_entries":entries,
            "estimated_kernel_seconds":estimated,"train_rows":train,"test_rows":test,
            "message":"Estimate calibrates the worker's state generation and vectorized matrix products. It covers kernel construction only; CV, model fitting, storage and machine load add time, and small-matrix timing is an extrapolation."}


@app.get("/api/runtime-estimate/grid")
def runtime_grid_estimate(type: Literal["scarcity-curve", "entanglement-ablation", "scaling-ablation"],
                         qubits: int = 6, fraud_labels: int = 10, repeats: int = 1,
                         seeds: int = 3, entangle: bool = True):
    """Estimate every kernel configuration before launching a multi-run sweep."""
    if qubits not in (4, 6, 8) or repeats not in (1, 2, 3) or fraud_labels not in (6, 10, 20, 50) or not 1 <= seeds <= 10:
        raise HTTPException(400, "Choose supported qubit, repeat, fraud-label and seed values.")
    if type == "entanglement-ablation":
        configs = [(k, qubits, repeats, toggle) for k in (6, 10, 20, 50) for toggle in (True, False)]
    elif type == "scaling-ablation":
        configs = [(fraud_labels, q, r, entangle) for q in (4, 6, 8) for r in (1, 2, 3)]
    else:
        configs = [(k, qubits, repeats, entangle) for k in (6, 10, 20, 50)]
    estimates = [runtime_estimate(q, r, k, seeds, toggle) for k, q, r, toggle in configs]
    return {"type": type, "configuration_count": len(configs), "seed_count": seeds,
            "estimated_kernel_entries": sum(item["estimated_kernel_entries"] for item in estimates),
            "estimated_kernel_seconds": sum(item["estimated_kernel_seconds"] for item in estimates),
            "per_configuration": [{"fraud_labels": k, "qubits": q, "repeats": r, "entangle": toggle,
                                   "estimated_kernel_seconds": result["estimated_kernel_seconds"]}
                                  for (k, q, r, toggle), result in zip(configs, estimates)],
            "message": "Kernel-construction estimate only; model fitting, CV, storage, and machine load add time."}


@app.get("/api/runtime-estimate/noise")
def noise_runtime_estimate(preset: Literal["LOW","MEDIUM","HIGH"]="MEDIUM", shots: int=256, mitigation: bool=False, seeds: int=1):
    if shots not in (256,1024,4096): raise HTTPException(400,"Choose 256, 1024 or 4096 shots.")
    if seeds not in (1,3,5): raise HTTPException(400,"Choose 1, 3 or 5 seeds for the local demo estimate.")
    try:
        import numpy as np
        from qiskit import transpile
        from qiskit_aer import AerSimulator
        from .noise_lab import feature_map,fold,make_noise,PRESETS
    except ImportError as exc: raise HTTPException(503,"Runtime calibration needs Qiskit Aer. Install backend/requirements.txt.") from exc
    a=np.array([.23,.71,1.18,2.04]);b=np.array([.41,.93,1.44,2.37]);base=feature_map(a).compose(feature_map(b).inverse()); model=make_noise(preset)
    sim=AerSimulator(noise_model=model,method="density_matrix"); elapsed=0.0
    for scale in (1,3,5) if mitigation else (1,):
        qc=fold(base,scale);qc.measure_all();compiled=transpile(qc,sim,optimization_level=0)
        t0=time.perf_counter();sim.run(compiled,shots=shots,seed_simulator=5).result().get_counts();elapsed+=time.perf_counter()-t0
    ntrain=40;ntest=20;per_scale=ntrain*(ntrain+1)//2+ntest*ntrain;entries=per_scale*(3 if mitigation else 1)
    measured_scales=3 if mitigation else 1
    per_entry=elapsed/(measured_scales)
    p2=PRESETS[preset][1]
    gates_per_pair=2*math.comb(4,2)
    signal={str(scale):float((1-p2)**(gates_per_pair*scale)) for scale in ((1,3,5) if mitigation else (1,))}
    entries*=seeds
    return {"label":"Software simulation","mode":"parameterized Aer noise","preset":preset,"shots":shots,"mitigation":mitigation,"seed_count":seeds,"calibration_circuits":measured_scales,"estimated_kernel_entries":entries,"estimated_seconds":per_entry*entries,"train_rows":ntrain,"test_rows":ntest,"estimated_two_qubit_gates_per_pair":gates_per_pair,"estimated_surviving_signal":signal,"signal_warning":signal.get("1",1)<0.2,"message":"Local estimate from measured Aer circuit time; compilation, PCA, fitting, disk I/O and machine load add time. Noise Lab uses a balanced 60-row demo per seed."}


class PairRequest(BaseModel):
    x: list[float]
    y: list[float]
    repeats: int = Field(default=1, ge=1, le=3)
    entangle: bool = True


@app.post("/api/kernel/pair")
def kernel_pair(req: PairRequest):
    try:
        import numpy as np
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise HTTPException(503, "Kernel exploration needs NumPy and Qiskit. Install backend/requirements.txt.") from exc
    if len(req.x) != len(req.y) or len(req.x) not in (2, 4, 6, 8):
        raise HTTPException(400, "Both vectors must have the same supported size: 2, 4, 6 or 8.")
    if any(not math.isfinite(v) or v < 0 or v > math.pi for v in [*req.x, *req.y]):
        raise HTTPException(400, "Vector values must be finite angles between 0 and pi.")
    def state(vec):
        q = QuantumCircuit(len(vec))
        for _ in range(req.repeats):
            for i, v in enumerate(vec): q.h(i); q.rz(2*v, i)
            if req.entangle:
                for i in range(len(vec)):
                    for j in range(i+1, len(vec)): q.rzz(2*(math.pi-vec[i])*(math.pi-vec[j]), i, j)
        return np.asarray(Statevector.from_instruction(q).data)
    sx, sy = state(req.x), state(req.y)
    probs_x, probs_y = (np.abs(sx)**2).tolist(), (np.abs(sy)**2).tolist()
    return {"similarity": float(abs(np.vdot(sx, sy))**2), "probabilities_x": probs_x,
            "probabilities_y": probs_y, "label": "Software simulation", "formula": "|<psi(x)|psi(y)>|^2"}


class JobRequest(BaseModel):
    type: Literal["benchmark", "noise", "mitigation", "demo", "entanglement-ablation", "scaling-ablation", "shots-sweep", "judge-demo", "scarcity-curve"]
    config: dict[str, Any] = Field(default_factory=dict)


def _decorate_run(run_id: str, group_id: str, group_type: str, grid_config: dict[str, Any] | None = None):
    path = RESULTS / run_id / "run.json"
    run = json.loads(path.read_text(encoding="utf-8"))
    run["experiment_group"] = {"id": group_id, "type": group_type, "configuration": grid_config or {}}
    path.write_text(json.dumps(run, indent=2), encoding="utf-8")


def _run_benchmark_grid(parent_id: str, frame, dataset_name: str, dataset_hash: str,
                        synthetic: bool, configurations: list[dict[str, Any]], group_type: str):
    completed: list[str] = []
    total = len(configurations)
    jobs[parent_id]["run_ids"] = completed
    for index, config in enumerate(configurations):
        if jobs[parent_id].get("cancel_requested"):
            jobs[parent_id].update(status="cancelled", message="Sweep cancelled between configurations.", run_ids=completed)
            return
        child_id = uuid.uuid4().hex[:12]
        jobs[child_id] = {"job_id": child_id, "type": "benchmark", "status": "queued", "progress": 0,
                          "message": f"Configuration {index+1}/{total}", "cancel_requested": False,
                          "synthetic": synthetic, "parent_job_id": parent_id}
        jobs[parent_id].update(status="running", progress=int(index / total * 100),
                               message=f"Configuration {index+1}/{total}: k={config['fraud_labels']}, q={config['qubits']}, reps={config['repeats']}, entangle={'on' if config['entangle'] else 'off'}")
        _run_benchmark(child_id, frame.copy(), int(config["fraud_labels"]), int(config["seed_count"]),
                       IDEAL_SEEDS[:int(config["seed_count"])], int(config["qubits"]), int(config["repeats"]),
                       bool(config["entangle"]), bool(config.get("log_amount", False)), synthetic,
                       dataset_name, dataset_hash, parent_id=parent_id, grid_index=index, grid_total=total)
        child = jobs[child_id]
        if child.get("status") != "completed":
            jobs[parent_id].update(status=child.get("status", "failed"), message=child.get("message", "A sweep configuration failed."), run_ids=completed)
            return
        run_id = child["run_id"]
        _decorate_run(run_id, parent_id, group_type, config)
        completed.append(run_id)
    jobs[parent_id].update(status="completed", progress=100, run_ids=completed,
                           message=f"{group_type.replace('-', ' ').title()} complete: {len(completed)} configurations saved.")


def _run_noise_sweep(parent_id: str, frame, qubits: int, preset: str, seed_count: int,
                     synthetic: bool, dataset_name: str, dataset_hash: str):
    completed: list[str] = []
    jobs[parent_id]["run_ids"] = completed
    for index, shots in enumerate((256, 1024, 4096)):
        if jobs[parent_id].get("cancel_requested"):
            jobs[parent_id].update(status="cancelled", message="Shot sweep cancelled between shot settings.", run_ids=completed)
            return
        child_id = uuid.uuid4().hex[:12]
        jobs[child_id] = {"job_id": child_id, "type": "noise", "status": "queued", "progress": 0,
                          "message": f"Shot setting {index+1}/3", "cancel_requested": False,
                          "synthetic": synthetic, "parent_job_id": parent_id}
        jobs[parent_id].update(status="running", progress=int(index / 3 * 100), message=f"Running {shots} shots · seed {seed_count}.")
        _run_noise_job(child_id, frame.copy(), qubits, preset, shots, False, seed_count, synthetic, dataset_name, dataset_hash,
                       parent_id=parent_id, parent_start=index*100/3, parent_span=100/3)
        child = jobs[child_id]
        if child.get("status") != "completed":
            jobs[parent_id].update(status=child.get("status", "failed"), message=child.get("message", "Shot sweep failed."), run_ids=completed)
            return
        _decorate_run(child["run_id"], parent_id, "shots-sweep", {"shots": shots, "noise_preset": preset})
        completed.append(child["run_id"])
    jobs[parent_id].update(status="completed", progress=100, run_ids=completed,
                           message="Shots-vs-PR-AUC sweep complete. Review mean ± standard deviation across seeds.")


def _run_judge_demo(parent_id: str, frame, dataset_name: str, dataset_hash: str, synthetic: bool,
                     budget: int, preset: str):
    run_ids: list[str] = []
    jobs[parent_id]["run_ids"] = run_ids
    stages = ("benchmark", "noise", "mitigation")
    for stage_index, stage in enumerate(stages):
        if jobs[parent_id].get("cancel_requested"):
            jobs[parent_id].update(status="cancelled", message="Judge demo cancelled between stages.", run_ids=run_ids)
            return
        child_id = uuid.uuid4().hex[:12]
        jobs[child_id] = {"job_id": child_id, "type": stage, "status": "queued", "progress": 0,
                          "message": f"Judge demo: {stage}", "cancel_requested": False,
                          "synthetic": synthetic, "parent_job_id": parent_id}
        jobs[parent_id].update(status="running", progress=stage_index * 33,
                               message=f"Judge demo stage {stage_index+1}/3: {stage.title()}.")
        if stage == "benchmark":
            _run_benchmark(child_id, frame.copy(), 6, 3, IDEAL_SEEDS[:3], 4, 1, True, False,
                           synthetic, dataset_name, dataset_hash, parent_id=parent_id,
                           grid_index=stage_index, grid_total=3)
        else:
            _run_noise_job(child_id, frame.copy(), 4, preset, 256, stage == "mitigation", 3,
                           synthetic, dataset_name, dataset_hash, parent_id=parent_id,
                           parent_start=stage_index*100/3, parent_span=100/3)
        child = jobs[child_id]
        if child.get("status") != "completed":
            jobs[parent_id].update(status=child.get("status", "failed"),
                                   message=f"Judge demo stopped during {stage}: {child.get('message', 'stage failed')}", run_ids=run_ids)
            return
        _decorate_run(child["run_id"], parent_id, "judge-demo", {"stage": stage, "synthetic": synthetic})
        run_ids.append(child["run_id"])
    jobs[parent_id].update(status="completed", progress=100, run_ids=run_ids,
                           message="Judge demo complete: benchmark, noisy run and mitigation are saved.")


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if req.type == "judge-demo" and dataset["frame"] is None:
        generate_demo_dataset()
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Long-running experiments require a persistent FastAPI worker. Vercel serverless functions cannot guarantee background jobs or persistent results.")
    if dataset["frame"] is None:
        raise HTTPException(409, "Load and validate a dataset before starting an experiment.")
    try:
        import numpy, pandas, sklearn
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        if req.type in ("noise","mitigation","demo","shots-sweep","judge-demo"):
            import qiskit_aer
    except ImportError:
        raise HTTPException(503, "Benchmark dependencies are missing. Install backend/requirements.txt, then restart the API.")
    frame = dataset["frame"].copy()
    if req.type in ("entanglement-ablation", "scaling-ablation", "scarcity-curve"):
        seeds = int(req.config.get("seed_count", 3))
        k = int(req.config.get("fraud_labels", 10))
        entangle = bool(req.config.get("entangle", True))
        budget = int(req.config.get("time_budget_seconds", 7200))
        if seeds < 1 or seeds > 10 or k not in (6, 10, 20, 50):
            raise HTTPException(400, "Choose 1–10 seeds and fraud-label count 6, 10, 20 or 50.")
        if req.type == "entanglement-ablation":
            qubits = int(req.config.get("qubits", 4)); repeats = int(req.config.get("repeats", 1))
            if qubits not in (4, 6, 8) or repeats not in (1, 2, 3):
                raise HTTPException(400, "Choose supported qubits and repeat values.")
            configurations = [{"fraud_labels": label_count, "qubits": qubits, "repeats": repeats,
                               "entangle": toggle, "seed_count": seeds}
                              for label_count in (6, 10, 20, 50) for toggle in (True, False)]
        elif req.type == "scaling-ablation":
            configurations = [{"fraud_labels": k, "qubits": n_qubits, "repeats": n_repeats,
                               "entangle": entangle, "seed_count": seeds}
                              for n_qubits in (4, 6, 8) for n_repeats in (1, 2, 3)]
        else:
            curve_qubits = int(req.config.get("qubits", 6)); curve_repeats = int(req.config.get("repeats", 1))
            if curve_qubits not in (4, 6, 8) or curve_repeats not in (1, 2, 3):
                raise HTTPException(400, "Choose 4, 6 or 8 qubits and 1, 2 or 3 repeats.")
            configurations = [{"fraud_labels": label_count, "qubits": curve_qubits,
                               "repeats": curve_repeats, "entangle": entangle,
                               "seed_count": seeds} for label_count in (6, 10, 20, 50)]
        max_k = max(item["fraud_labels"] for item in configurations)
        if int((frame.Class == 1).sum()) < 100 + max_k or int((frame.Class == 0).sum()) < 600:
            raise HTTPException(400, "The full sweep needs at least 150 fraud and 600 legitimate rows.")
        estimate_seconds = sum(float(runtime_estimate(item["qubits"], item["repeats"], item["fraud_labels"], seeds, item["entangle"])["estimated_kernel_seconds"]) for item in configurations)
        if budget < 60 or budget > 86400:
            raise HTTPException(400, "Choose a time budget between 1 minute and 24 hours.")
        if estimate_seconds > budget:
            raise HTTPException(413, f"This {len(configurations)}-configuration sweep estimates {estimate_seconds/60:.1f} minutes for kernels alone. Increase the budget or reduce the seed count.")
        group_id = uuid.uuid4().hex[:12]
        group_type = {"entanglement-ablation":"entanglement-ablation", "scaling-ablation":"feature-map-scaling",
                      "scarcity-curve":"scarcity-curve"}[req.type]
        jobs[group_id] = {"job_id": group_id, "type": req.type, "status": "queued", "progress": 0,
                          "message": f"Queued {len(configurations)} configurations.", "cancel_requested": False,
                          "synthetic": bool(dataset.get("synthetic", False)), "run_ids": [],
                          "estimated_kernel_seconds": estimate_seconds}
        threading.Thread(target=_run_benchmark_grid, args=(group_id, frame, dataset["name"], dataset["sha256"],
                          bool(dataset.get("synthetic", False)), configurations, group_type), daemon=True).start()
        return {"job_id": group_id, "status": "queued", "message": f"Queued {len(configurations)} configurations; estimated kernel time {estimate_seconds/60:.1f} minutes."}

    if req.type == "shots-sweep":
        preset = str(req.config.get("noise_preset", "MEDIUM")).upper()
        seeds = int(req.config.get("seed_count", 3)); budget = int(req.config.get("time_budget_seconds", 7200))
        if preset not in ("LOW", "MEDIUM", "HIGH") or seeds not in (3, 5):
            raise HTTPException(400, "Choose a noise preset and 3 or 5 independent seeds.")
        if int((frame.Class == 1).sum()) < 30 or int((frame.Class == 0).sum()) < 30:
            raise HTTPException(400, "The shots sweep needs at least 30 fraud and 30 legitimate rows.")
        estimate_seconds = sum(float(noise_runtime_estimate(preset, shot, False, seeds)["estimated_seconds"]) for shot in (256, 1024, 4096))
        if estimate_seconds > budget:
            raise HTTPException(413, f"The three-shot sweep estimates {estimate_seconds/60:.1f} minutes. Increase the time budget or use fewer seeds.")
        group_id = uuid.uuid4().hex[:12]
        jobs[group_id] = {"job_id": group_id, "type": req.type, "status": "queued", "progress": 0,
                          "message": "Queued shot settings 256, 1024 and 4096.", "cancel_requested": False,
                          "synthetic": bool(dataset.get("synthetic", False)), "run_ids": [],
                          "estimated_seconds": estimate_seconds}
        threading.Thread(target=_run_noise_sweep, args=(group_id, frame, 4, preset, seeds,
                          bool(dataset.get("synthetic", False)), dataset["name"], dataset["sha256"]), daemon=True).start()
        return {"job_id": group_id, "status": "queued", "message": f"Queued 256/1024/4096 shot sweep, {seeds} seeds per setting."}

    if req.type == "judge-demo":
        preset = str(req.config.get("noise_preset", "LOW")).upper()
        budget = int(req.config.get("time_budget_seconds", 7200))
        if preset not in ("LOW", "MEDIUM", "HIGH"):
            raise HTTPException(400, "Choose a LOW, MEDIUM or HIGH noise preset.")
        if int((frame.Class == 1).sum()) < 106 or int((frame.Class == 0).sum()) < 600:
            raise HTTPException(400, "The judge demo needs at least 106 fraud and 600 legitimate rows. Load a dataset or use Generate demo dataset.")
        estimated = float(runtime_estimate(4, 1, 6, 3, True)["estimated_kernel_seconds"])
        estimated += 2 * float(noise_runtime_estimate(preset, 256, True, 3)["estimated_seconds"])
        if estimated > budget:
            raise HTTPException(413, f"The complete judge demo estimates {estimated/60:.1f} minutes of kernel/simulator time. Increase the time budget.")
        group_id = uuid.uuid4().hex[:12]
        jobs[group_id] = {"job_id": group_id, "type": req.type, "status": "queued", "progress": 0,
                          "message": "Queued benchmark → noisy simulation → mitigation.", "cancel_requested": False,
                          "synthetic": bool(dataset.get("synthetic", False)), "run_ids": [],
                          "estimated_seconds": estimated}
        threading.Thread(target=_run_judge_demo, args=(group_id, frame, dataset["name"], dataset["sha256"],
                          bool(dataset.get("synthetic", False)), budget, preset), daemon=True).start()
        return {"job_id": group_id, "status": "queued", "estimated_seconds": estimated,
                "message": f"Queued benchmark → noisy simulation → mitigation. Estimated kernel/simulator time: {estimated/60:.1f} min; setup and model fitting add time."}

    if req.type in ("noise","mitigation","demo"):
        qubits=int(req.config.get("qubits",4)); shots=int(req.config.get("shots",256)); preset=str(req.config.get("noise_preset","MEDIUM")).upper()
        seed_count=int(req.config.get("seed_count",3))
        budget=int(req.config.get("time_budget_seconds",1800))
        if qubits!=4: raise HTTPException(400,"The bounded Aer noise demonstration currently supports 4 qubits.")
        if shots not in (256,1024,4096): raise HTTPException(400,"Choose 256, 1024 or 4096 shots.")
        if preset not in ("LOW","MEDIUM","HIGH"): raise HTTPException(400,"Choose a LOW, MEDIUM or HIGH noise preset.")
        if seed_count not in (3,5): raise HTTPException(400,"Noise and mitigation experiments require 3 or 5 independent seeds.")
        if budget<60 or budget>86400: raise HTTPException(400,"Choose a time budget between 60 seconds and 24 hours.")
        if int((frame.Class==1).sum())<30 or int((frame.Class==0).sum())<30:
            raise HTTPException(400,"This local noise demonstration needs at least 30 fraud and 30 legitimate transactions.")
        estimate=noise_runtime_estimate(preset,shots,req.type=="mitigation",seed_count)
        if estimate["estimated_seconds"]>budget: raise HTTPException(413,"The local Aer simulation exceeds the selected time budget. Lower the shot count or choose a smaller experiment.")
        job_id=uuid.uuid4().hex[:12]
        jobs[job_id]={"job_id":job_id,"type":req.type,"status":"queued","progress":0,"message":"Waiting for a local Aer worker.","cancel_requested":False,"synthetic":bool(dataset.get("synthetic",False)),"label":"Software simulation"}
        threading.Thread(target=_run_noise_job,args=(job_id,frame,qubits,preset,shots,req.type=="mitigation",seed_count,bool(dataset.get("synthetic",False)),dataset["name"],dataset["sha256"]),daemon=True).start()
        return {"job_id":job_id,"status":"queued","message":f"{seed_count}-seed balanced 60-row software simulation queued."}
    k = int(req.config.get("fraud_labels", 10))
    seeds = int(req.config.get("seed_count", 10))
    qubits = int(req.config.get("qubits", 6))
    repeats=int(req.config.get("repeats",1));entangle=bool(req.config.get("entangle",True))
    log_amount=bool(req.config.get("log_amount",False))
    if k not in (6, 10, 20, 50) or seeds < 1 or seeds > 10 or qubits not in (4, 6, 8) or repeats not in (1,2,3):
        raise HTTPException(400, "Use k=6, 10, 20 or 50; 1–10 seeds; 4, 6 or 8 qubits; and 1–3 repeats.")
    budget = int(req.config.get("time_budget_seconds", 1800))
    if budget < 60 or budget > 86400:
        raise HTTPException(400, "Choose a time budget between 60 seconds and 24 hours.")
    estimate = runtime_estimate(qubits, repeats, k, seeds)
    if estimate["estimated_kernel_seconds"] > budget:
        raise HTTPException(413, "This quantum benchmark is too large for the selected time budget. Reduce seeds, qubits or fraud-label count.")
    if int((frame.Class == 1).sum()) < 100 + k or int((frame.Class == 0).sum()) < 600:
        raise HTTPException(400, "This protocol needs at least 100 fraud rows, k additional training frauds, and 600 legitimate rows.")
    if log_amount and bool((frame.Amount<0).any()): raise HTTPException(400,"log1p(Amount) needs non-negative amount values. Turn it off or clean the CSV.")
    job_id = uuid.uuid4().hex[:12]
    seed_values = IDEAL_SEEDS[:seeds]
    jobs[job_id] = {"job_id": job_id, "type": "benchmark", "status": "queued", "progress": 0,
                    "message": "Waiting for a local worker.", "cancel_requested": False,
                    "synthetic": bool(dataset.get("synthetic",False)), "label": "Software simulation"}
    threading.Thread(target=_run_benchmark, args=(job_id, frame, k, seeds, seed_values, qubits, repeats, entangle, log_amount, bool(dataset.get("synthetic",False)), dataset["name"], dataset["sha256"]), daemon=True).start()
    return {"job_id": job_id, "status": "queued", "message": "Benchmark queued on this computer."}


def _run_benchmark(job_id: str, frame, k: int, seed_count: int, seed_values: list[int], qubits: int, repeats: int, entangle: bool, log_amount: bool, synthetic: bool, dataset_name: str, dataset_hash: str, parent_id: str | None = None, grid_index: int = 0, grid_total: int = 1):
    """Run the ideal local benchmark. Outputs are written only after completion."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from sklearn.decomposition import PCA
    from sklearn.metrics import (average_precision_score, f1_score, precision_score, precision_recall_curve,
                                 recall_score, roc_auc_score)
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.base import clone
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{job_id}"
    rows = []; predictions = []; variances = []; circuit_example = None; kernel_artifacts=[]; last_quantum_bundle = None
    def progress(value, message):
        with lock:
            if job_id in jobs:
                jobs[job_id].update(status="running", progress=value, message=message)
            if parent_id and parent_id in jobs:
                if jobs[parent_id].get("cancel_requested"):
                    jobs[job_id]["cancel_requested"] = True
                jobs[parent_id].update(status="running", progress=int(100 * (grid_index + value / 100) / grid_total),
                                       message=f"Configuration {grid_index+1}/{grid_total}: {message}")
    def feature_state(values, repeat=1, entangle=True):
        circuit = QuantumCircuit(qubits)
        for _ in range(repeat):
            for i, x in enumerate(values): circuit.h(i); circuit.rz(2 * float(x), i)
            if entangle:
                for i in range(qubits):
                    for j in range(i + 1, qubits):
                        circuit.rzz(2 * (math.pi - values[i]) * (math.pi - values[j]), i, j)
        return np.asarray(Statevector.from_instruction(circuit).data)
    def matrix(a, b, repeat=1, entangle=True):
        left = np.stack([feature_state(x, repeat, entangle) for x in a])
        if b is a:
            gram=left @ left.conj().T
            upper=np.triu(np.abs(gram)**2)
            return upper+np.triu(upper,1).T
        right = np.stack([feature_state(x, repeat, entangle) for x in b])
        return np.abs(left @ right.conj().T) ** 2
    def f1_threshold(y_true, scores):
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        if len(thresholds)==0: return 0.0
        f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
        return float(thresholds[int(np.argmax(f1))])
    try:
        frame = frame.reset_index(drop=True)
        features = [*(f"V{i}" for i in range(1,29)),"Amount"]
        X = frame[features].to_numpy(dtype=float); y = frame.Class.to_numpy(dtype=int)
        if log_amount: X[:,features.index("Amount")]=np.log1p(X[:,features.index("Amount")])
        positives = np.flatnonzero(y == 1); negatives = np.flatnonzero(y == 0)
        for n, seed in enumerate(seed_values):
            if jobs[job_id].get("cancel_requested") or (parent_id and jobs.get(parent_id, {}).get("cancel_requested")):
                jobs[job_id].update(status="cancelled", message="Run cancelled. No partial run was saved."); return
            rng = np.random.default_rng(seed)
            test_pos = rng.choice(positives, 100, replace=False)
            test_neg = rng.choice(negatives, 400, replace=False)
            remain_pos = np.setdiff1d(positives, test_pos)
            remain_neg = np.setdiff1d(negatives, test_neg)
            train_pos = rng.choice(remain_pos, k, replace=False)
            train_neg = rng.choice(remain_neg, 200, replace=False)
            train_ids = np.r_[train_neg, train_pos]; test_ids = np.r_[test_neg, test_pos]
            rng.shuffle(train_ids); rng.shuffle(test_ids)
            ytr, yte = y[train_ids], y[test_ids]
            t0 = time.perf_counter()
            scaler = StandardScaler().fit(X[train_ids]); xtr = scaler.transform(X[train_ids]); xte = scaler.transform(X[test_ids])
            pca = PCA(n_components=qubits, random_state=seed).fit(xtr)
            ztr = pca.transform(xtr); zte = pca.transform(xte)
            angle = MinMaxScaler(feature_range=(0, math.pi), clip=True).fit(ztr)
            atr, ate = angle.transform(ztr), angle.transform(zte)
            if n == 0:
                circuit_example = atr[0].tolist()
            prep_s = time.perf_counter() - t0; variances.append(pca.explained_variance_ratio_.tolist())
            progress(round(5 + 85*n/seed_count), f"Seed {n+1}/{seed_count} (random state {seed}): computing ideal kernels")
            cache_payload = json.dumps({"dataset": dataset_hash, "seed": int(seed),
                "train": train_ids.tolist(), "test": test_ids.tolist(), "features": features,
                "qubits": qubits, "repeats": repeats, "entangle": entangle,
                "log_amount": log_amount, "preprocessing": "standard-pca-minmax-v1"}, sort_keys=True)
            cache_key = hashlib.sha256(cache_payload.encode()).hexdigest()
            t0 = time.perf_counter()
            Ktr, Kte, cache_hit = _kernel_cache_get_or_compute(
                cache_key, lambda: (matrix(atr, atr, repeats, entangle), matrix(ate, atr, repeats, entangle)))
            if cache_hit:
                jobs[job_id]["cache_hits"] = jobs[job_id].get("cache_hits", 0) + 1
            kernel_s = time.perf_counter()-t0
            order=np.argsort(-ytr,kind="stable")
            kernel_artifacts.append(Ktr[np.ix_(order,order)])
            # Select C with stratified three-fold PR-AUC on the training kernel only.
            folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
            best = (-1.0, 1.0)
            for c in [0.1, 0.3, 1, 3, 10, 30]:
                scores=[]
                for a,b in folds.split(Ktr,ytr):
                    model=SVC(C=c,kernel="precomputed",class_weight="balanced")
                    model.fit(Ktr[np.ix_(a,a)],ytr[a]); scores.append(average_precision_score(ytr[b],model.decision_function(Ktr[np.ix_(b,a)])))
                if float(np.mean(scores))>best[0]:best=(float(np.mean(scores)),c)
            oof=np.zeros(len(ytr),dtype=float)
            for a,b in folds.split(Ktr,ytr):
                fold_model=SVC(C=best[1],kernel="precomputed",class_weight="balanced")
                fold_model.fit(Ktr[np.ix_(a,a)],ytr[a]);oof[b]=fold_model.decision_function(Ktr[np.ix_(b,a)])
            q_threshold=f1_threshold(ytr,oof)
            model=SVC(C=best[1],kernel="precomputed",class_weight="balanced")
            t0=time.perf_counter();model.fit(Ktr,ytr);fit_s=time.perf_counter()-t0
            t0=time.perf_counter();decision=model.decision_function(Kte);pred_s=time.perf_counter()-t0
            pred=(decision>=q_threshold).astype(int)
            values={"PR-AUC":average_precision_score(yte,decision),"recall":recall_score(yte,pred,zero_division=0),"precision":precision_score(yte,pred,zero_division=0),"F1":f1_score(yte,pred,zero_division=0),"ROC-AUC":roc_auc_score(yte,decision),"accuracy":float(np.mean(pred==yte))}
            for metric,value in values.items():rows.append({"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":metric,"value":float(value)})
            for i,idx in enumerate(test_ids):predictions.append({"seed":seed,"row_id":int(idx),"amount":float(frame.iloc[idx]["Amount"]),"actual":int(yte[i]),"quantum_score":float(decision[i]),"quantum_prediction":int(pred[i]),"classical_scores":{},"classical_predictions":{}})
            from sklearn.model_selection import cross_val_predict, cross_val_score
            best_classical_cv=-1.0; best_classical_test=0.0; best_classical_name=""; best_classical_clf=None; best_classical_thresh=0.5; best_classical_reference=None
            for j,name in enumerate(["RBF-SVM","Logistic regression","Random forest","Classical (all features)"]):
                from sklearn.preprocessing import StandardScaler as SS
                tx=X[train_ids] if name=="Classical (all features)" else ztr
                vx=X[test_ids] if name=="Classical (all features)" else zte
                if name in ("RBF-SVM","Classical (all features)"):
                    candidates=[make_pipeline(SS(),SVC(C=c,gamma=g,class_weight="balanced")) for c in [0.1,0.3,1,3,10,30] for g in [0.1,0.3,1,3]]
                elif name=="Logistic regression":
                    candidates=[make_pipeline(SS(),LogisticRegression(C=c,class_weight="balanced",max_iter=1000)) for c in [0.1,0.3,1,3,10,30]]
                else:
                    candidates=[RandomForestClassifier(n_estimators=200,class_weight="balanced",random_state=seed,n_jobs=1)]
                model_scores=[float(np.mean(cross_val_score(clf,tx,ytr,cv=folds,scoring="average_precision",n_jobs=1))) for clf in candidates]
                chosen=int(np.argmax(model_scores));clf=clone(candidates[chosen]);cv_mean=model_scores[chosen]
                oof_method="predict_proba" if name=="Random forest" else "decision_function"
                oof_scores=cross_val_predict(clf,tx,ytr,cv=folds,method=oof_method,n_jobs=1)
                if oof_scores.ndim==2:oof_scores=oof_scores[:,1]
                threshold=f1_threshold(ytr,oof_scores)
                t0=time.perf_counter();clf.fit(tx,ytr);ct_fit=time.perf_counter()-t0
                t0=time.perf_counter();cs=clf.decision_function(vx) if hasattr(clf,"decision_function") else clf.predict_proba(vx)[:,1];ct_pred=time.perf_counter()-t0
                cp=(cs>=threshold).astype(int)
                cvals={"PR-AUC":average_precision_score(yte,cs),"recall":recall_score(yte,cp,zero_division=0),"precision":precision_score(yte,cp,zero_division=0),"F1":f1_score(yte,cp,zero_division=0),"ROC-AUC":roc_auc_score(yte,cs),"accuracy":float(np.mean(cp==yte)),"fit time":ct_fit,"prediction time":ct_pred}
                for metric,value in cvals.items(): rows.append({"seed":seed,"fraud_labels":k,"model":name,"metric":metric,"value":float(value)})
                for row_i,score_value in enumerate(cs): predictions[-len(test_ids)+row_i]["classical_scores"][name]=float(score_value)
                for row_i,predicted_value in enumerate(cp): predictions[-len(test_ids)+row_i]["classical_predictions"][name]=int(predicted_value)
                if cv_mean > best_classical_cv:
                    best_classical_cv=cv_mean; best_classical_test=float(cvals["PR-AUC"]); best_classical_name=name; best_classical_clf=clf; best_classical_thresh=threshold; best_classical_reference=np.asarray(oof_scores,dtype=float).ravel().tolist()
            rows.append({"seed":seed,"fraud_labels":k,"model":"Best classical (CV-selected)","metric":"PR-AUC","value":best_classical_test})
            for prediction in predictions[-len(test_ids):]:
                prediction["best_classical_name"]=best_classical_name
                prediction["best_classical_score"]=prediction["classical_scores"].get(best_classical_name)
            last_quantum_bundle = {
                "qubits": qubits, "repeats": repeats, "entangle": entangle,
                "log_amount": log_amount, "features": features, "q_threshold": q_threshold,
                "quantum_model": model, "quantum_train_angles": atr, "quantum_train_y": ytr,
                "quantum_reference_scores": np.asarray(oof,dtype=float).ravel().tolist(),
                "classical_reference_scores": best_classical_reference,
                "scaler": scaler, "pca": pca, "angle": angle,
                "best_classical_name": best_classical_name, "best_classical_model": best_classical_clf,
                "classical_threshold": best_classical_thresh
            }
            rows.extend([{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"kernel time","value":float(kernel_s)},{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"fit time","value":float(fit_s)},{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"prediction time","value":float(pred_s)},{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"qubits","value":float(qubits)}])
            report_circuit=QuantumCircuit(qubits)
            for _ in range(repeats):
                for i,v in enumerate(atr[0]): report_circuit.h(i);report_circuit.rz(2*float(v),i)
                if entangle:
                    for i in range(qubits):
                        for j in range(i+1,qubits):report_circuit.rzz(2*(math.pi-atr[0][i])*(math.pi-atr[0][j]),i,j)
            decomposed=report_circuit.decompose(reps=3);gate_counts=decomposed.count_ops()
            rows.extend([{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"depth","value":float(decomposed.depth())},{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"gates","value":float(sum(gate_counts.values()))},{"seed":seed,"fraud_labels":k,"model":"Quantum ideal","metric":"two-qubit gates","value":float(sum(v for gate,v in gate_counts.items() if gate in {"cx","cz","swap"}))}])
        # Persist only completed real-data runs. No partial outputs are represented as results.
        import pandas as pd
        summary=[]
        for (model_name,metric), group in pd.DataFrame(rows).groupby(["model","metric"]):
            vals=group.value.to_numpy();summary.append({"model":model_name,"metric":metric,"mean":float(np.mean(vals)),"std":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,"seed_count":int(len(vals))})
        quantum=np.array([r["value"] for r in rows if r["model"]=="Quantum ideal" and r["metric"]=="PR-AUC"])
        classical_by_seed=[]
        for seed in seed_values:
            vals=[r["value"] for r in rows if r["seed"]==seed and r["metric"]=="PR-AUC" and r["model"]=="Best classical (CV-selected)"]
            classical_by_seed.append(max(vals) if vals else 0)
        differences=quantum-np.array(classical_by_seed)
        seed_summaries=[]
        for seed, difference, quantum_value, classical_value in zip(seed_values, differences, quantum, classical_by_seed):
            seed_summaries.append({"seed":int(seed),"quantum_pr_auc":float(quantum_value),
                                   "best_classical_pr_auc":float(classical_value),
                                   "paired_difference":float(difference)})
        verdict="Not enough evidence"; interval=[None,None]
        if seed_count>=2:
            from scipy.stats import t
            m=float(differences.mean());sem=float(differences.std(ddof=1)/math.sqrt(seed_count));critical=float(t.ppf(.975,seed_count-1));interval=[m-critical*sem,m+critical*sem]
            if interval[0]>0 and interval[0]>.01:verdict="Quantum ahead in this setting"
            elif interval[1]<0 and m<-.03:verdict="Classical preferred"
            elif abs(m)<=.03 or interval[0]<=0<=interval[1]:verdict="Quantum comparable (within noise)"
            else:verdict="Classical slightly ahead"
        out=RESULTS/run_id;out.mkdir(parents=True,exist_ok=False)
        pd.DataFrame(rows).to_csv(out/"metrics.csv",index=False)
        prediction_frame=pd.DataFrame(predictions)
        score_frame=prediction_frame.pop("classical_scores").apply(pd.Series).add_prefix("score_")
        class_frame=prediction_frame.pop("classical_predictions").apply(pd.Series).add_prefix("prediction_")
        pd.concat([prediction_frame,score_frame,class_frame],axis=1).to_csv(out/"predictions.csv",index=False)
        np.savez_compressed(out/"kernel_train.npz",**{f"seed_{i}":matrix for i,matrix in enumerate(kernel_artifacts)})
        if last_quantum_bundle:
            try:
                import pickle
                with open(out / "model_bundle.pkl", "wb") as f:
                    pickle.dump(last_quantum_bundle, f)
            except Exception:
                pass
        config={"mode":"benchmark","fraud_labels":k,"seed_count":seed_count,"seeds":seed_values,"train_normal_count":200,"train_fraud_count":k,"test_fraud_count":100,"test_normal_count":400,"qubits":qubits,"repeats":repeats,"entangle":entangle,"log1p_amount":log_amount,"environment":"ideal exact statevector"}
        run={"schema_version":"1.0","run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"synthetic":synthetic,"config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"dataset":{"name":dataset_name,"sha256":dataset_hash,"rows":len(frame),"fraud_count":int(y.sum()),"legitimate_count":int((y==0).sum())},"config":config,"circuit_example":{"qubits":qubits,"repeats":repeats,"entangle":entangle,"angles":circuit_example,"seed":seed_values[0]},"preprocessing":{"fit_on_training_only":True,"features":features,"pca_explained_variance_mean":np.mean(variances,axis=0).tolist()},"kernel_cache":{"enabled":True,"hits":int(jobs[job_id].get("cache_hits",0)),"limit_bytes":KERNEL_CACHE_LIMIT},"artifacts":{"metrics":"metrics.csv","predictions":"predictions.csv","kernel_train":"kernel_train.npz","model_bundle":"model_bundle.pkl"},"kernel_heatmap":{"artifact":"kernel_train.npz","fraud_rows_first":True,"divider_after":k,"matrix_size":200+k,"seed_count":seed_count},"seed_summaries":seed_summaries,"metrics":summary,"verdict":{"overall":verdict,"paired_difference_mean":float(differences.mean()),"interval_95_t":interval},"limitations":["Ideal software statevector simulation; no hardware execution.","This benchmark run reports ideal kernels only. Noisy and mitigated comparisons are separate run types.","Balanced test subset does not represent natural fraud prevalence.","Classical RBF-SVM and logistic regression grids are selected by training-only CV; random forest uses 200 trees.","Decision thresholds are selected from training-only out-of-fold scores; this is a research implementation that still needs independent protocol review."]}
        (out/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
        jobs[job_id].update(status="completed",progress=100,message=f"Ideal benchmark completed. Kernel cache hits: {jobs[job_id].get('cache_hits',0)}.",run_id=run_id)
    except Exception as exc:
        jobs[job_id].update(status="failed",message="The run stopped before a complete result was saved. Check the dataset size and local simulation dependencies, then retry.")


def _run_noise_job(job_id, frame, qubits, preset, shots, mitigate, seed_count, synthetic, dataset_name, dataset_hash,
                   parent_id: str | None = None, parent_start: float = 0, parent_span: float = 100):
    run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{job_id}"
    try:
        import numpy as np
        from .noise_lab import run_lab
        results=[]
        for seed in range(seed_count):
            if jobs[job_id].get("cancel_requested") or (parent_id and jobs.get(parent_id, {}).get("cancel_requested")):
                raise InterruptedError()
            start=seed/seed_count
            def progress(value,message):
                if jobs[job_id].get("cancel_requested"): raise InterruptedError()
                child_progress = int(5+85*(start+value/(100*seed_count)))
                jobs[job_id].update(status="running",progress=child_progress,message=f"Seed {seed+1}/{seed_count}: {message}")
                if parent_id and parent_id in jobs:
                    jobs[parent_id].update(status="running", progress=int(parent_start + parent_span * child_progress / 100),
                                           message=f"{shots} shots · seed {seed+1}/{seed_count}: {message}")
            results.append(run_lab(frame,qubits,preset,shots,seed=seed,mitigate=mitigate,synthetic=synthetic,progress=progress))
        progress(94,"Writing the completed multi-seed simulation result.")
        first=results[0]
        def summary(values):
            import numpy as np
            return {key:{"mean":float(np.mean([item[key] for item in values])),"std":float(np.std([item[key] for item in values],ddof=1)) if len(values)>1 else 0.0,"seed_count":len(values)} for key in values[0]}
        result=dict(first);result["ideal"]= {k:v["mean"] for k,v in summary([x["ideal"] for x in results]).items()};result["noisy"]={k:v["mean"] for k,v in summary([x["noisy"] for x in results]).items()}
        result["ideal_summary"]=summary([x["ideal"] for x in results]);result["noisy_summary"]=summary([x["noisy"] for x in results]);result["seed_count"]=seed_count
        result["seed_summaries"]=[{"seed":i,"ideal":x["ideal"],"noisy":x["noisy"],"mitigation_metrics":x.get("mitigation",{}).get("metrics"),"kernel_summary":x.get("mitigation",{}).get("kernel_summary")} for i,x in enumerate(results)]
        rows=[]
        for environment in ("ideal","noisy"):
            for metric,stat in result[f"{environment}_summary"].items():
                rows.append({"model":environment,"metric":metric,"mean":stat["mean"],"std":stat["std"],"seed_count":seed_count})
        if mitigate:
            methods={}
            method_names=results[0]["mitigation"]["metrics"].keys()
            for method in method_names:
                methods[method]={}
                for phase in ("before_repair","after_repair"):
                    metric_names=results[0]["mitigation"]["metrics"][method][phase].keys()
                    methods[method][phase]={}
                    for metric in metric_names:
                        values=[x["mitigation"]["metrics"][method][phase][metric] for x in results]
                        stat={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)) if seed_count>1 else 0.0,"seed_count":seed_count}
                        methods[method][phase][metric]=stat
                        if phase=="after_repair":rows.append({"model":f"mitigated-{method}","metric":metric,**stat})
                for key in ("minimum_eigenvalue_before","minimum_eigenvalue_after"):
                    values=[x["mitigation"]["metrics"][method][key] for x in results]
                    methods[method][key]={"mean":float(np.mean(values)),"std":float(np.std(values,ddof=1)),"seed_count":seed_count}
            result["mitigation"]=dict(first["mitigation"]);result["mitigation"]["metrics"]=methods
            summaries=[x["mitigation"]["kernel_summary"] for x in results]
            result["mitigation"]["kernel_summary"]={key:(float(np.mean([item[key] for item in summaries])) if key!="extrapolators" else {name:float(np.mean([item[key][name] for item in summaries])) for name in summaries[0][key]}) for key in summaries[0]}
            gaps=[x["ideal"]["PR-AUC"]-x["noisy"]["PR-AUC"] for x in results]
            gap_mean=float(np.mean(gaps));gap_std=float(np.std(gaps,ddof=1))
            mitigated=[x["mitigation"]["metrics"]["linear"]["after_repair"]["PR-AUC"] for x in results]
            gains=[m-x["noisy"]["PR-AUC"] for m,x in zip(mitigated,results)]
            if gap_mean<=0.02 or gap_mean<=gap_std:
                result["recovery_note"]="No meaningful degradation to recover across these seeds."
            elif float(np.mean(gains))<=float(np.std(gains,ddof=1)):
                result["recovery_note"]="Mitigation did not improve this experiment beyond seed variation."
            else:
                recoveries=[(m-x["noisy"]["PR-AUC"])/gap for m,x,gap in zip(mitigated,results,gaps) if gap>max(.02,gap_std)]
                result["recovery"]={"mean":float(np.mean(recoveries)),"std":float(np.std(recoveries,ddof=1)) if len(recoveries)>1 else 0.0,"seed_count":len(recoveries)} if recoveries else None
                result["recovery_note"]="Recovery is the paired mean (mitigated − noisy) / (ideal − noisy), reported only when ideal-to-noisy degradation exceeds 0.02 and its across-seed standard deviation."
        result["protocol"]=f"Quick demo, too small for conclusions · {seed_count} independent balanced 60-row seeds"
        result["limitations"]=["Quick demo, too small for conclusions.",first["balanced_subset_notice"],f"{seed_count} seeds are reported, but each uses only 30 fraud and 30 legitimate rows; this does not meet the full benchmark protocol.","Noise values are a parameterized simulator model, not measured hardware noise.","This demonstration uses fixed model parameters and is not the full benchmark CV protocol."]
        config={"mode":"quick-demo-noise","fraud_labels_per_run":int(first["fraud_labels_train"]),"seed_count":seed_count,"seeds":list(range(seed_count)),"qubits":qubits,"repeats":1,"shots":shots,"noise_preset":preset,"mitigation":mitigate,"noise_values":first["noise_values"]}
        run={"schema_version":"1.0","run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"synthetic":synthetic,"label":"Software simulation","config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"dataset":{"name":dataset_name,"sha256":dataset_hash,"rows":len(frame)},"config":config,"protocol":result["protocol"],"balanced_subset_notice":result["balanced_subset_notice"],"metrics":rows,"verdict":{"overall":"Not enough evidence"},"noise_lab":result,"limitations":result["limitations"]}
        out=RESULTS/run_id;out.mkdir(parents=True,exist_ok=False)
        (out/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
        with (out/"metrics.csv").open("w",newline="",encoding="utf-8") as f:
            writer=csv.DictWriter(f,fieldnames=["model","metric","mean","std","seed_count"]);writer.writeheader();writer.writerows(rows)
        jobs[job_id].update(status="completed",progress=100,message=f"{seed_count}-seed quick demo complete. Small balanced samples are not conclusion-grade.",run_id=run_id)
    except InterruptedError:
        jobs[job_id].update(status="cancelled",message="Run cancelled. No partial result was saved.")
    except Exception:
        jobs[job_id].update(status="failed",message="Noise simulation failed before a result was saved. Check the selected configuration and local Qiskit Aer installation.")


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "That experiment run was not found.")
    return jobs[job_id]


@app.post("/api/jobs/{job_id}/cancel")
def cancel_job(job_id: str):
    if job_id not in jobs: raise HTTPException(404, "That experiment run was not found.")
    jobs[job_id]["cancel_requested"] = True
    return {"message": "Cancellation requested."}


@app.get("/api/results")
def list_results():
    results = []
    for file in RESULTS.glob("*/run.json"):
        try:
            run=json.loads(file.read_text(encoding="utf-8"))
            # Keep the index small; full kernel matrices are returned only by get_result.
            item={k:run.get(k) for k in ("schema_version","run_id","created_at","synthetic","label","config_hash","dataset","config","preprocessing","artifacts","metrics","verdict","limitations","protocol","balanced_subset_notice","experiment_group","kernel_cache","circuit_example","seed_summaries") if k in run}
            lab=run.get("noise_lab",{})
            if lab:
                mit_data = lab.get("mitigation") or {}
                item["noise_lab"]={"recovery_note":lab.get("recovery_note"),"mitigation":{"kernel_summary":mit_data.get("kernel_summary"),"metrics":mit_data.get("metrics")}}
            results.append(item)
        except (OSError, json.JSONDecodeError): continue
    return sorted(results, key=lambda r: r.get("created_at", ""), reverse=True)


@app.get("/api/results/{run_id}")
def get_result(run_id: str):
    path = (RESULTS / run_id / "run.json").resolve()
    if not path.is_relative_to(RESULTS.resolve()) or not path.is_file():
        raise HTTPException(404, "No saved result was found for this run.")
    try: return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc: raise HTTPException(500, "The saved result file is not valid JSON.") from exc


@app.get("/api/results/{run_id}/kernel")
def get_kernel_heatmap(run_id: str, seed: int = 0):
    run=get_result(run_id); artifact=run.get("artifacts",{}).get("kernel_train")
    if not artifact: raise HTTPException(404,"This run did not save a training kernel matrix.")
    path=(RESULTS/run_id/artifact).resolve()
    if not path.is_relative_to((RESULTS/run_id).resolve()) or not path.is_file(): raise HTTPException(404,"Saved kernel artifact is missing or invalid.")
    import numpy as np
    try:
        with np.load(path,allow_pickle=False) as stored:
            key=f"seed_{seed}"
            if key not in stored: raise HTTPException(404,"That seed's kernel matrix was not saved.")
            matrix=stored[key]
    except (OSError,ValueError) as exc: raise HTTPException(400,"Saved kernel matrix could not be read.") from exc
    divider = int(run.get("kernel_heatmap",{}).get("divider_after") or 0)
    # Descriptive kernel geometry only; this is not a prediction metric.
    upper = np.triu(np.ones(matrix.shape, dtype=bool), k=1)
    fraud = np.arange(matrix.shape[0]) < divider
    groups = {
        "within_fraud": upper & fraud[:, None] & fraud[None, :],
        "within_legitimate": upper & ~fraud[:, None] & ~fraud[None, :],
        "cross_class": fraud[:, None] != fraud[None, :],
    }
    geometry = {name: float(matrix[mask].mean()) if mask.any() else None for name, mask in groups.items()}
    return {"run_id":run_id,"seed":seed,"matrix":matrix.tolist(),"fraud_rows_first":True,
            "divider_after":divider,"geometry":geometry,"label":"Software simulation"}


def _generate_markdown_report(run: dict) -> str:
    run_id = run.get("run_id", "Unknown")
    created = run.get("created_at", "Unknown")
    config = run.get("config", {})
    dataset_info = run.get("dataset", {})
    verdict = run.get("verdict", {})
    metrics = run.get("metrics", [])
    prep = run.get("preprocessing", {})
    limitations = run.get("limitations", [])

    lines = [
        f"# Research Experiment Report: `{run_id}`",
        "",
        f"- **Timestamp**: {created}",
        f"- **Configuration Hash**: `{run.get('config_hash', 'N/A')}`",
        f"- **Simulation Mode**: {config.get('mode', 'benchmark')} ({config.get('environment', 'Software Simulation')})",
        "",
        "## Dataset Profile",
        f"- **Dataset Source**: `{dataset_info.get('name', 'N/A')}`",
        f"- **SHA-256 Digest**: `{dataset_info.get('sha256', 'N/A')}`",
        f"- **Total Rows Evaluated**: {dataset_info.get('rows', 'N/A'):,}" if isinstance(dataset_info.get('rows'), int) else f"- **Total Rows**: {dataset_info.get('rows', 'N/A')}",
        f"- **Class Balance**: {dataset_info.get('fraud_count', 'N/A')} Fraud / {dataset_info.get('legitimate_count', 'N/A')} Legitimate",
        "",
        "## Experiment Configuration",
        f"- **Qubits**: {config.get('qubits', 'N/A')}",
        f"- **Circuit Feature-Map Repeats**: {config.get('repeats', 'N/A')}",
        f"- **Entanglement**: {'Pairwise RZZ Enabled' if config.get('entangle') else 'Disabled'}",
        f"- **Fraud Training Labels (k)**: {config.get('fraud_labels', config.get('fraud_labels_per_run', 'N/A'))}",
        f"- **Training Partition**: {config.get('train_normal_count', 200)} Normal + {config.get('train_fraud_count', config.get('fraud_labels', 'N/A'))} Fraud",
        f"- **Test Partition**: {config.get('test_normal_count', 400)} Normal + {config.get('test_fraud_count', 100)} Fraud",
        f"- **Random Seeds ({config.get('seed_count', len(config.get('seeds', [])))} seeds)**: {', '.join(map(str, config.get('seeds', [])))}",
        "",
        "## Performance Metrics Summary",
        "",
        "| Model | Metric | Mean | Std Dev | Seeds |",
        "| :--- | :--- | :---: | :---: | :---: |"
    ]
    for row in metrics:
        m = row.get("mean")
        s = row.get("std", 0.0)
        mean_str = f"{m:.4f}" if isinstance(m, (int, float)) else str(m)
        std_str = f"{s:.4f}" if isinstance(s, (int, float)) else str(s)
        lines.append(f"| {row.get('model')} | {row.get('metric')} | {mean_str} | ± {std_str} | n={row.get('seed_count', 1)} |")

    lines.extend([
        "",
        "## Statistical Verdict & Analysis",
        f"- **Overall Verdict**: **{verdict.get('overall', 'Not enough evidence')}**",
        f"- **Paired Difference (Quantum − Best Classical)**: {verdict.get('paired_difference_mean', 'N/A'):.4f}" if isinstance(verdict.get('paired_difference_mean'), (int, float)) else f"- **Paired Difference**: {verdict.get('paired_difference_mean', 'N/A')}",
        f"- **95% Confidence Interval (Student's t)**: {verdict.get('interval_95_t', ['N/A', 'N/A'])}",
        "",
        "## Preprocessing & Dimensionality Reduction",
        f"- **Feature Space**: 29 features (`V1`–`V28`, `Amount`) reduced to {config.get('qubits', 6)} principal components via PCA.",
        f"- **Mean Explained Variance per Component**: {prep.get('pca_explained_variance_mean', 'N/A')}",
        "",
        "## Research Limitations & Disclaimers"
    ])
    for lim in limitations:
        lines.append(f"- {lim}")

    lab = run.get("noise_lab", {})
    if lab:
        lines.extend([
            "",
            "## Noise & Error-Mitigation Lab",
            f"- **Preset**: {lab.get('noise_preset', 'N/A')} | **Shots**: {lab.get('shots', 'N/A')}",
            f"- **Recovery Note**: {lab.get('recovery_note', 'N/A')}"
        ])

    lines.extend([
        "",
        "---",
        "*Report automatically exported from Q-Fraud Intelligence Local Research Bench.*"
    ])
    return "\n".join(lines)


@app.get("/api/results/{run_id}/export")
def export_result(run_id: str, format: Literal["json", "csv", "md"] = "json"):
    run = get_result(run_id)
    if format == "json": return JSONResponse(run)
    if format == "md":
        return PlainTextResponse(_generate_markdown_report(run), media_type="text/markdown")
    output = io.StringIO(); writer = csv.DictWriter(output, fieldnames=["model", "metric", "mean", "std", "seed_count"])
    writer.writeheader()
    for row in run.get("metrics", []): writer.writerow(row)
    return PlainTextResponse(output.getvalue(), media_type="text/csv")


@app.get("/api/testset/{run_id}")
def testset(run_id: str):
    run = get_result(run_id)
    artifact = run.get("artifacts", {}).get("predictions")
    if not artifact: raise HTTPException(404, "This run did not save per-transaction scores.")
    path=(RESULTS/run_id/artifact).resolve()
    if not path.is_relative_to((RESULTS/run_id).resolve()) or not path.is_file():
        raise HTTPException(404,"Per-transaction score artifact is missing or invalid.")
    try:
        import pandas as pd
        table=pd.read_csv(path)
        table=table.astype(object).where(pd.notna(table),None)
        data=table.to_dict(orient="records")
        for row in data:
            row["classical_scores"]={key[6:]:value for key,value in list(row.items()) if key.startswith("score_") and value is not None}
            row["classical_predictions"]={key[11:]:int(value) for key,value in list(row.items()) if key.startswith("prediction_") and value is not None}
        ordered=sorted(range(len(data)),key=lambda i:(data[i].get("quantum_score") is None,float(data[i].get("quantum_score") or 0)),reverse=True)
        for rank,index in enumerate(ordered,1):
            data[index]["quantum_rank_percentile"]=100.0*(len(data)-rank)/(max(1,len(data)-1))
            best=data[index].get("best_classical_name"); cp=data[index].get("classical_predictions",{}).get(best)
            data[index]["agreement"]=None if cp is None or data[index].get("quantum_prediction") is None else bool(int(cp)==int(data[index]["quantum_prediction"]))
    except Exception as exc:
        raise HTTPException(400,"Saved per-transaction scores could not be read.") from exc
    return {"rows":data,"count":len(data),"message":"Scores are decision values, not calibrated probabilities."}


def _fit_model_bundle_for_run(run_data: dict, frame):
    """Fit a compatible model bundle using the run's config and current dataset."""
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from sklearn.decomposition import PCA
    from sklearn.metrics import precision_recall_curve
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.svm import SVC
    from sklearn.model_selection import StratifiedKFold, cross_val_predict

    config = run_data.get("config", {})
    qubits = int(config.get("qubits", 6))
    repeats = int(config.get("repeats", 1))
    entangle = bool(config.get("entangle", True))
    log_amount = bool(config.get("log1p_amount", False))
    k = int(config.get("fraud_labels", config.get("fraud_labels_per_run", 10)))
    seeds = config.get("seeds") or [42]
    seed = int(seeds[0])

    features = [*(f"V{i}" for i in range(1, 29)), "Amount"]
    X = frame[features].to_numpy(dtype=float)
    y = frame.Class.to_numpy(dtype=int)
    if log_amount:
        amt_idx = features.index("Amount")
        X[:, amt_idx] = np.log1p(np.maximum(0.0, X[:, amt_idx]))

    positives = np.flatnonzero(y == 1)
    negatives = np.flatnonzero(y == 0)
    rng = np.random.default_rng(seed)

    k_actual = min(k, len(positives))
    train_pos = rng.choice(positives, k_actual, replace=False)
    train_neg = rng.choice(negatives, min(200, len(negatives)), replace=False)
    train_ids = np.r_[train_neg, train_pos]
    rng.shuffle(train_ids)
    ytr = y[train_ids]

    scaler = StandardScaler().fit(X[train_ids])
    xtr = scaler.transform(X[train_ids])
    pca = PCA(n_components=qubits, random_state=seed).fit(xtr)
    ztr = pca.transform(xtr)
    angle = MinMaxScaler(feature_range=(0.05 * math.pi, 0.95 * math.pi), clip=True).fit(ztr)
    atr = angle.transform(ztr)

    def feature_state(values):
        circuit = QuantumCircuit(qubits)
        for _ in range(repeats):
            for i, x in enumerate(values):
                circuit.h(i)
                circuit.rz(2 * float(x), i)
            if entangle:
                for i in range(qubits):
                    for j in range(i + 1, qubits):
                        circuit.rzz(2 * (math.pi - values[i]) * (math.pi - values[j]), i, j)
        return np.asarray(Statevector.from_instruction(circuit).data)

    left = np.stack([feature_state(x) for x in atr])
    gram = left @ left.conj().T
    upper = np.triu(np.abs(gram) ** 2)
    Ktr = upper + np.triu(upper, 1).T

    folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    q_model = SVC(C=1.0, kernel="precomputed", class_weight="balanced")
    oof = np.zeros(len(ytr), dtype=float)
    for a, b in folds.split(Ktr, ytr):
        fold_model = SVC(C=1.0, kernel="precomputed", class_weight="balanced")
        fold_model.fit(Ktr[np.ix_(a, a)], ytr[a])
        oof[b] = fold_model.decision_function(Ktr[np.ix_(b, a)])

    prec, rec, threshs = precision_recall_curve(ytr, oof)
    if len(threshs) > 0:
        f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
        q_thresh = float(threshs[int(np.argmax(f1))])
    else:
        q_thresh = 0.0
    q_model.fit(Ktr, ytr)

    classical_model = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=seed, n_jobs=1)
    oof_c = cross_val_predict(classical_model, ztr, ytr, cv=folds, method="predict_proba", n_jobs=1)[:, 1]
    c_prec, c_rec, c_threshs = precision_recall_curve(ytr, oof_c)
    if len(c_threshs) > 0:
        f1_c = 2 * c_prec[:-1] * c_rec[:-1] / np.maximum(c_prec[:-1] + c_rec[:-1], 1e-12)
        c_thresh = float(c_threshs[int(np.argmax(f1_c))])
    else:
        c_thresh = 0.5
    classical_model.fit(ztr, ytr)

    bundle = {
        "qubits": qubits,
        "repeats": repeats,
        "entangle": entangle,
        "log_amount": log_amount,
        "features": features,
        "q_threshold": q_thresh,
        "quantum_model": q_model,
        "quantum_train_angles": atr,
        "quantum_train_y": ytr,
        "quantum_reference_scores": np.asarray(oof,dtype=float).ravel().tolist(),
        "classical_reference_scores": np.asarray(oof_c,dtype=float).ravel().tolist(),
        "scaler": scaler,
        "pca": pca,
        "angle": angle,
        "best_classical_name": "Random forest",
        "best_classical_model": classical_model,
        "classical_threshold": c_thresh,
    }
    return bundle


def _get_or_create_model_bundle(run_id: str | None = None):
    import pickle
    target_run_id = run_id
    if not target_run_id:
        all_runs = sorted(list(RESULTS.glob("*/run.json")), key=lambda p: p.stat().st_mtime, reverse=True)
        if not all_runs:
            raise HTTPException(404, "No completed experiment runs found. Please run a benchmark first.")
        target_run_id = all_runs[0].parent.name

    run_dir = RESULTS / target_run_id
    if not run_dir.is_dir():
        raise HTTPException(404, f"Run '{target_run_id}' not found.")
    bundle_path = run_dir / "model_bundle.pkl"
    if bundle_path.is_file():
        try:
            with open(bundle_path, "rb") as f:
                return pickle.load(f), target_run_id
        except Exception:
            pass

    run_file = run_dir / "run.json"
    if not run_file.is_file():
        raise HTTPException(404, f"Run metadata not found for '{target_run_id}'.")
    run_data = json.loads(run_file.read_text(encoding="utf-8"))

    if dataset["frame"] is None:
        load_default_dataset()

    bundle = _fit_model_bundle_for_run(run_data, dataset["frame"])
    try:
        with open(bundle_path, "wb") as f:
            pickle.dump(bundle, f)
    except Exception:
        pass
    return bundle, target_run_id


class ScoreRequest(BaseModel):
    run_id: str | None = None
    values: dict[str, float] = Field(default_factory=dict)


@app.post("/api/score")
def score(req: ScoreRequest):
    if os.environ.get("VERCEL"):
        raise HTTPException(503, "Persisted model scoring requires a persistent FastAPI worker; Vercel serverless instances do not retain dataset and model state.")
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    request_start = time.perf_counter()
    bundle, active_run_id = _get_or_create_model_bundle(req.run_id)
    model_init_ms = (time.perf_counter() - request_start) * 1000.0
    inference_start = time.perf_counter()
    preprocess_start = time.perf_counter()
    features = bundle["features"]
    raw_vals = [float(req.values.get(f, 0.0)) for f in features]
    raw_array = np.array([raw_vals], dtype=float)
    if bundle.get("log_amount", False):
        amt_idx = features.index("Amount")
        raw_array[0, amt_idx] = np.log1p(max(0.0, raw_array[0, amt_idx]))

    x_scaled = bundle["scaler"].transform(raw_array)
    z = bundle["pca"].transform(x_scaled)
    a = bundle["angle"].transform(z)
    preprocessing_ms = (time.perf_counter() - preprocess_start) * 1000.0

    quantum_state_start = time.perf_counter()
    qubits = bundle["qubits"]
    repeats = bundle["repeats"]
    entangle = bundle["entangle"]

    def feature_state(values):
        qc = QuantumCircuit(qubits)
        for _ in range(repeats):
            for i, x in enumerate(values):
                qc.h(i)
                qc.rz(2 * float(x), i)
            if entangle:
                for i in range(qubits):
                    for j in range(i + 1, qubits):
                        qc.rzz(2 * (math.pi - values[i]) * (math.pi - values[j]), i, j)
        return np.asarray(Statevector.from_instruction(qc).data)

    new_state = feature_state(a[0])
    train_states = np.stack([feature_state(x) for x in bundle["quantum_train_angles"]])
    quantum_state_ms = (time.perf_counter() - quantum_state_start) * 1000.0
    kernel_start = time.perf_counter()
    k_vec = np.abs(train_states @ new_state.conj().T) ** 2
    kernel_eval_ms = (time.perf_counter() - kernel_start) * 1000.0

    q_model = bundle["quantum_model"]
    q_svc_start = time.perf_counter()
    q_score = float(q_model.decision_function([k_vec])[0])
    quantum_svc_ms = (time.perf_counter() - q_svc_start) * 1000.0
    q_thresh = float(bundle["q_threshold"])
    q_pred = int(q_score >= q_thresh)
    # Exact additive SVC decomposition: sum(alpha_i * y_i * K(x_i, x)) + intercept.
    support_ids = np.asarray(q_model.support_, dtype=int)
    dual = np.asarray(q_model.dual_coef_[0], dtype=float)
    support_contributions = dual * k_vec[support_ids]
    order = np.argsort(np.abs(support_contributions))[::-1][:5]
    strongest = max(float(np.max(np.abs(support_contributions))), 1e-12)
    contribution_trace = [{
        "support_vector": int(support_ids[i]) + 1,
        "label": "fraud" if int(bundle["quantum_train_y"][support_ids[i]]) == 1 else "legitimate",
        "kernel_similarity": round(float(k_vec[support_ids[i]]), 6),
        "contribution": round(float(support_contributions[i]), 6),
        "relative_strength": round(float(abs(support_contributions[i]) / strongest), 4),
    } for i in order]
    intercept = float(np.asarray(q_model.intercept_).ravel()[0])
    reconstructed = float(np.sum(support_contributions) + intercept)
    reference = np.asarray(bundle.get("quantum_reference_scores", []), dtype=float)
    rank_percentile = float(np.mean(reference <= q_score) * 100.0) if reference.size else None

    classical_preprocess_start = time.perf_counter()
    clf = bundle["best_classical_model"]
    c_thresh = float(bundle["classical_threshold"])
    c_in = raw_array if bundle.get("best_classical_name") == "Classical (all features)" else z
    classical_preprocessing_ms = (time.perf_counter() - classical_preprocess_start) * 1000.0
    classical_predict_start = time.perf_counter()
    if hasattr(clf, "decision_function"):
        c_score = float(clf.decision_function(c_in)[0])
    else:
        c_score = float(clf.predict_proba(c_in)[0, 1])
    c_pred = int(c_score >= c_thresh)
    classical_prediction_ms = (time.perf_counter() - classical_predict_start) * 1000.0
    classical_ms = classical_preprocessing_ms + classical_prediction_ms

    inference_ms = (time.perf_counter() - inference_start) * 1000.0
    latency_ms = (time.perf_counter() - request_start) * 1000.0
    q_dict = {
        "score": round(q_score, 5),
        "threshold": round(q_thresh, 5),
        "prediction": q_pred,
        "is_fraud": bool(q_pred == 1),
        "decision": "FRAUD" if q_pred == 1 else "LEGITIMATE",
        "verdict": "FRAUD" if q_pred == 1 else "LEGITIMATE",
        "support_vectors_count": len(q_model.support_),
        "rank_percentile": round(rank_percentile, 1) if rank_percentile is not None else None,
        "explanation": {
            "method": "Exact quantum-kernel SVC support-vector decomposition",
            "intercept": round(intercept, 6),
            "reconstructed_score": round(reconstructed, 6),
            "reconstruction_error": round(abs(reconstructed - q_score), 10),
            "top_support_vectors": contribution_trace,
            "caveat": "This explains the model margin through kernel support vectors; it is not a causal or per-feature explanation."
        }
    }
    c_dict = {
        "model_name": bundle.get("best_classical_name", "Random forest"),
        "score": round(c_score, 5),
        "threshold": round(c_thresh, 5),
        "prediction": c_pred,
        "is_fraud": bool(c_pred == 1),
        "decision": "FRAUD" if c_pred == 1 else "LEGITIMATE",
        "verdict": "FRAUD" if c_pred == 1 else "LEGITIMATE"
    }
    latency_breakdown = {
        "total_ms": round(latency_ms, 2),
        "model_initialization_ms": round(model_init_ms, 2),
        "inference_ms": round(inference_ms, 2),
        "preprocessing_ms": round(preprocessing_ms, 2),
        "quantum_state_generation_ms": round(quantum_state_ms, 2),
        "kernel_evaluation_ms": round(kernel_eval_ms, 2),
        "quantum_svc_ms": round(quantum_svc_ms, 2),
        "quantum_kernel_ms": round(quantum_state_ms + kernel_eval_ms, 2),
        "classical_input_selection_ms": round(classical_preprocessing_ms, 2),
        "classical_prediction_ms": round(classical_prediction_ms, 2),
        "classical_ms": round(classical_ms, 2),
        "timing_method": "independently measured wall-clock stages; end-to-end includes model load or fit"
    }
    return {
        "status": "ok",
        "run_id": active_run_id,
        "latency_ms": latency_breakdown,
        "quantum": q_dict,
        "quantum_svc": q_dict,
        "classical": c_dict,
        "agreement": bool(q_pred == c_pred),
        "consensus": {
            "agreement": bool(q_pred == c_pred),
            "verdict": "AGREE" if q_pred == c_pred else "DISAGREE"
        }
    }


@app.get("/api/sample-transactions")
def sample_transactions():
    """Return actually labelled rows from the active dataset; never invent examples."""
    frame = dataset.get("frame")
    if frame is None:
        return {"samples": [], "message": "Load a dataset to use dataset-labelled transaction examples."}
    features = [f"V{i}" for i in range(1, 29)] + ["Time", "Amount"]
    samples = []
    for label, name in ((0, "Dataset-labelled legitimate example"), (1, "Dataset-labelled fraud example")):
        matches = frame.index[frame["Class"] == label]
        if len(matches) == 0:
            continue
        row_id = matches[0]
        row = frame.loc[row_id]
        values = {feature: float(row[feature]) for feature in features}
        samples.append({"id": f"row-{row_id}", "label": name, "actual_class": label,
                        "description": f"Selected from loaded dataset {dataset.get('name')}; row ID {row_id}.",
                        "row_id": str(row_id), "features": values, "values": values})
    return {"samples": samples, "dataset_name": dataset.get("name"), "sha256": dataset.get("sha256"),
            "message": "Examples use actual Class labels from the loaded dataset."}


@app.get("/api/resources")
def resources():
    return RESOURCES


RESOURCES = [
 {"title":"Q-Hack India 2026", "url":"https://www.quantumweek.tech/hackathon", "reason":"Hackathon context and participation details."},
 {"title":"Q-Hack submission guide", "url":"https://www.quantumweek.tech/hackathon/submission", "reason":"Submission expectations."},
 {"title":"Credit Card Fraud Detection dataset", "url":"https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud", "reason":"Dataset source; check license terms before redistribution."},
 {"title":"OpenML dataset record", "url":"https://www.openml.org/d/1597", "reason":"Alternative dataset metadata and access."},
 {"title":"Qiskit documentation", "url":"https://quantum.cloud.ibm.com/docs", "reason":"Circuit and simulator API reference; this app uses no IBM hardware."},
 {"title":"Qiskit Aer simulation guide", "url":"https://quantum.cloud.ibm.com/docs/guides/simulate-with-qiskit-aer", "reason":"Local simulation and noise-model reference."},
 {"title":"Bowles, Ahmed and Schuld (2024)", "url":"https://arxiv.org/abs/2403.07059", "reason":"Benchmarking cautions and classical baselines for QML."},
 {"title":"Havlicek et al. (2019)", "url":"https://doi.org/10.1038/s41586-019-0980-2", "reason":"Quantum feature-space learning background."},
 {"title":"Temme, Bravyi and Gambetta (2017)", "url":"https://doi.org/10.1103/PhysRevLett.119.180509", "reason":"Error-mitigation background for short-depth circuits."},
 {"title":"scikit-learn", "url":"https://scikit-learn.org/", "reason":"Classical model and evaluation documentation."},
 {"title":"NumPy", "url":"https://numpy.org/", "reason":"Numerical computing used in kernel calculations."},
 {"title":"FastAPI", "url":"https://fastapi.tiangolo.com/", "reason":"Backend API documentation."}
]
