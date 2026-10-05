# Q-Fraud Intelligence

A local research prototype for comparing classical fraud classifiers with a
Qiskit quantum-kernel SVM and simulated noise. The software uses Qiskit and
Qiskit Aer on this computer. It does not connect to quantum hardware, IBM
Quantum, credentials, or cloud services. No dataset or benchmark results ship
with the project.

## Start the application

Requirements: Python 3.11 or newer and Node.js 20 or newer.

From the project root, install backend packages and create the local
environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r backend/requirements.txt
```

In one terminal, start the API:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open http://127.0.0.1:5173. The API documentation is at
http://127.0.0.1:8000/docs. After installing dependencies, the application
runs locally and does not need network access for simulation.

## Provide a dataset

Use **Dataset** to upload a CSV (up to 250 MB) or load a local path. The app
checks for `V1`–`V28`, `Time`, `Amount`, and binary `Class` columns, and reports
row counts and class balance from the loaded file. Uploads stay in process
memory; they are not written to disk. Do not expose this local API to a public
network or share saved row-level prediction files without checking the data
license and privacy requirements.

For the canonical ULB/Worldline dataset, place `creditcard.csv` at
`datasets/raw/creditcard.csv`. That directory is ignored by Git. The default
ideal benchmark uses the reproducible seed list
`42, 123, 456, 789, 1000, 2024, 2025, 2026, 31415, 27182`; selecting fewer
seeds uses a prefix of this list.

The ideal benchmark requires at least 100 fraud rows, 600 legitimate rows,
and an additional `k` training fraud rows (k = 6, 10, 20, or 50). It samples
200 legitimate and k fraud training rows, then tests on 100 unseen fraud and
400 legitimate rows. The balanced test subset does not represent natural
fraud prevalence. Preprocessing, PCA, angle scaling and decision thresholds
are fit using training data only.

## What works in this prototype

- Dataset validation and live class-balance statistics.
- Qiskit feature-map inspection and exact statevector kernel-pair explorer.
- Background, cancellable ideal benchmark jobs with ten configurable seeds,
  training-only CV for model selection, classical baselines, stored metrics,
  test-set decision scores, and paired per-seed verdicts.
- An exact local runtime estimate for ideal kernel construction. It does not
  include CV/model fitting and is an extrapolation, not a completion-time
  guarantee.
- Four-qubit Qiskit Aer noise and zero-noise extrapolation demonstrations across
  three or five independent seeds, each using a balanced 60-row subset. These
  remain quick demos, not full-protocol benchmark evidence.
- Result listing/export, saved prediction exploration, and saved ideal
  training-kernel heatmaps.

## Research limits to keep visible

The ideal benchmark and small noisy/ZNE demonstration are separate protocols.
The noise demonstration uses at least three seeds and a fixed 30/30 balanced
subset per seed; it does not implement the full 100-fraud/400-normal test
protocol for noisy or mitigated runs.
The ideal benchmark currently uses a fixed feature-map configuration per run;
there is no automated entanglement/repeat ablation grid or cross-run replay
button. There is no kernel cache, thermal-relaxation noise, shots-versus-PR-AUC
sweep, or full one-click judge demo. The transaction scorer is a UI placeholder;
it does not accept arbitrary transactions or load a persisted model. Saved
runs include JSON and CSV exports; a Markdown report export is not implemented.

Treat every result as prototype research output. Simulation time is not
quantum-hardware performance. Accuracy can mislead when fraud is rare. A
balanced subset changes class prevalence. Small sample comparisons are
uncertain, and this work cannot establish universal quantum advantage.

## Project layout

- `frontend/`: React, TypeScript and Vite user interface.
- `backend/app/`: FastAPI API, dataset checks, Qiskit utilities, benchmark job
  workers, Qiskit Aer noise and mitigation experiment.
- `datasets/`: user-managed local data; files are ignored by Git.
- `results/`: generated experiment folders; files are ignored by Git.
- `frontend/src/content.ts`: centralized application copy.

The UI uses ink `#17232D`, paper `#F4F2EC`, slate `#64717A`, rust `#A94F35`,
moss `#557565`, and gold `#B38A3E`; DM Sans for reading and DM Mono for
measurements. It supports system dark mode, reduced motion, and responsive
layouts.
