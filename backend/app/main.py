"""Local-only API shell. Heavy scientific dependencies are loaded on demand."""
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

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
RESULTS.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
MAX_UPLOAD = 250 * 1024 * 1024
REQUIRED = {*(f"V{i}" for i in range(1, 29)), "Time", "Amount", "Class"}
IDEAL_SEEDS = [42, 123, 456, 789, 1000, 2024, 2025, 2026, 31415, 27182]
dataset: dict[str, Any] = {"frame": None, "name": None, "sha256": None, "synthetic": False}
jobs: dict[str, dict[str, Any]] = {}
lock = threading.Lock()

app = FastAPI(title="Q-Fraud Intelligence API", version="0.1.0",
              description="Local software simulation research API. No quantum hardware or cloud services.")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
                   allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])


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
    return _statistics(frame)


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
    return {"status": "ok", "mode": "local software simulation", "dataset_loaded": dataset["frame"] is not None,
            "results_count": len(list(RESULTS.glob("*/run.json")))}


@app.post("/api/dataset/upload")
async def upload_dataset(file: UploadFile = File(...)):
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
def circuit_get(qubits: int = 4, repeats: int = 1, entangle: bool = True):
    return _circuit(CircuitRequest(qubits=qubits, repeats=repeats, entangle=entangle))


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
    type: Literal["benchmark", "noise", "mitigation", "demo"]
    config: dict[str, Any] = Field(default_factory=dict)


@app.post("/api/jobs")
def create_job(req: JobRequest):
    if dataset["frame"] is None:
        raise HTTPException(409, "Load and validate a dataset before starting an experiment.")
    try:
        import numpy, pandas, sklearn
        from qiskit import QuantumCircuit
        from qiskit.quantum_info import Statevector
        if req.type in ("noise","mitigation","demo"):
            import qiskit_aer
    except ImportError:
        raise HTTPException(503, "Benchmark dependencies are missing. Install backend/requirements.txt, then restart the API.")
    frame = dataset["frame"].copy()
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


def _run_benchmark(job_id: str, frame, k: int, seed_count: int, seed_values: list[int], qubits: int, repeats: int, entangle: bool, log_amount: bool, synthetic: bool, dataset_name: str, dataset_hash: str):
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
    rows = []; predictions = []; variances = []; circuits = {}; kernel_artifacts=[]
    def progress(value, message):
        with lock:
            if job_id in jobs:
                jobs[job_id].update(status="running", progress=value, message=message)
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
            if jobs[job_id].get("cancel_requested"):
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
            prep_s = time.perf_counter() - t0; variances.append(pca.explained_variance_ratio_.tolist())
            progress(round(5 + 85*n/seed_count), f"Seed {n+1}/{seed_count} (random state {seed}): computing ideal kernels")
            t0 = time.perf_counter(); Ktr = matrix(atr, atr, repeats, entangle); Kte = matrix(ate, atr, repeats, entangle); kernel_s = time.perf_counter()-t0
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
            best_classical_cv=-1.0; best_classical_test=0.0; best_classical_name=""
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
                    best_classical_cv=cv_mean; best_classical_test=float(cvals["PR-AUC"]); best_classical_name=name
            rows.append({"seed":seed,"fraud_labels":k,"model":"Best classical (CV-selected)","metric":"PR-AUC","value":best_classical_test})
            for prediction in predictions[-len(test_ids):]:
                prediction["best_classical_name"]=best_classical_name
                prediction["best_classical_score"]=prediction["classical_scores"].get(best_classical_name)
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
        config={"mode":"benchmark","fraud_labels":k,"seed_count":seed_count,"seeds":seed_values,"train_normal_count":200,"train_fraud_count":k,"test_fraud_count":100,"test_normal_count":400,"qubits":qubits,"repeats":repeats,"entangle":entangle,"log1p_amount":log_amount,"environment":"ideal exact statevector"}
        run={"schema_version":"1.0","run_id":run_id,"created_at":datetime.now(timezone.utc).isoformat(),"synthetic":synthetic,"config_hash":hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),"dataset":{"name":dataset_name,"sha256":dataset_hash,"rows":len(frame),"fraud_count":int(y.sum()),"legitimate_count":int((y==0).sum())},"config":config,"preprocessing":{"fit_on_training_only":True,"features":features,"pca_explained_variance_mean":np.mean(variances,axis=0).tolist()},"artifacts":{"metrics":"metrics.csv","predictions":"predictions.csv","kernel_train":"kernel_train.npz"},"kernel_heatmap":{"artifact":"kernel_train.npz","fraud_rows_first":True,"divider_after":k,"matrix_size":200+k,"seed_count":seed_count},"metrics":summary,"verdict":{"overall":verdict,"paired_difference_mean":float(differences.mean()),"interval_95_t":interval},"limitations":["Ideal software statevector simulation; no hardware execution.","This benchmark run reports ideal kernels only. Noisy and mitigated comparisons are separate run types.","Balanced test subset does not represent natural fraud prevalence.","Classical RBF-SVM and logistic regression grids are selected by training-only CV; random forest uses 200 trees.","Decision thresholds are selected from training-only out-of-fold scores; this is a research implementation that still needs independent protocol review."]}
        (out/"run.json").write_text(json.dumps(run,indent=2),encoding="utf-8")
        jobs[job_id].update(status="completed",progress=100,message="Ideal benchmark completed.",run_id=run_id)
    except Exception as exc:
        jobs[job_id].update(status="failed",message="The run stopped before a complete result was saved. Check the dataset size and local simulation dependencies, then retry.")


def _run_noise_job(job_id, frame, qubits, preset, shots, mitigate, seed_count, synthetic, dataset_name, dataset_hash):
    run_id=f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{job_id}"
    try:
        import numpy as np
        from .noise_lab import run_lab
        results=[]
        for seed in range(seed_count):
            if jobs[job_id].get("cancel_requested"): raise InterruptedError()
            start=seed/seed_count
            def progress(value,message):
                if jobs[job_id].get("cancel_requested"): raise InterruptedError()
                jobs[job_id].update(status="running",progress=int(5+85*(start+value/(100*seed_count))),message=f"Seed {seed+1}/{seed_count}: {message}")
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
            item={k:run.get(k) for k in ("schema_version","run_id","created_at","synthetic","label","config_hash","dataset","config","preprocessing","artifacts","metrics","verdict","limitations","protocol","balanced_subset_notice") if k in run}
            lab=run.get("noise_lab",{})
            if lab:
                item["noise_lab"]={"recovery_note":lab.get("recovery_note"),"mitigation":{"kernel_summary":lab.get("mitigation",{}).get("kernel_summary"),"metrics":lab.get("mitigation",{}).get("metrics")}}
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
    return {"run_id":run_id,"seed":seed,"matrix":matrix.tolist(),"fraud_rows_first":True,"divider_after":run.get("kernel_heatmap",{}).get("divider_after"),"label":"Software simulation"}


@app.get("/api/results/{run_id}/export")
def export_result(run_id: str, format: Literal["json", "csv", "md"] = "json"):
    run = get_result(run_id)
    if format == "json": return JSONResponse(run)
    if format == "md":
        lines = [f"# {run.get('run_id', run_id)}", "", f"Replay of a stored real run ({run.get('created_at', 'date unavailable')}, config hash {run.get('config_hash', 'unavailable')}).", "", "## Limitations", ""]
        lines.extend(f"- {v}" for v in run.get("limitations", ["Review the saved run configuration before interpretation."]))
        return PlainTextResponse("\n".join(lines), media_type="text/markdown")
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


class ScoreRequest(BaseModel):
    values: dict[str, float]


@app.post("/api/score")
def score(req: ScoreRequest):
    raise HTTPException(409, "Load a saved run with a compatible model before scoring. Research demonstration, not a production financial decision system.")


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
