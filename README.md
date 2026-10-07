# Q-Fraud Intelligence

A research prototype for comparing classical fraud classifiers with a
Qiskit quantum-kernel SVM and simulated noise. The software uses Qiskit and
Qiskit Aer in a Python worker. It does not connect to quantum hardware, IBM
Quantum, or quantum cloud services. No dataset ships with the project.

## Deploy the hosted app

The frontend is a Vercel static site and the API is a persistent Render web
service. The Render service definition in `render.yaml` installs the Python
dependencies, starts Uvicorn, mounts a persistent disk for experiment results
and uploaded datasets, and allows requests from `https://qkernel.vercel.app`.
The configured worker uses 2 CPU cores, 4 GB memory, and a 5 GB persistent
disk; Render requires a paid service for persistent disks.

1. Push this repository to GitHub and create a Render Blueprint from it. Render
   reads `render.yaml`; wait for the `qkernel-api` deployment to become live.
2. Open `https://qkernel-api.onrender.com/api/health`. A healthy response has
   `status: "ok"` and `execution_mode: "persistent-worker"`.
3. In the Vercel project settings, set the production environment variable
   `VITE_API_URL` to `https://qkernel-api.onrender.com` (no trailing slash),
   then redeploy the frontend. Vite embeds this value during the build.
4. Open `https://qkernel.vercel.app`, confirm it reports a connected persistent
   worker, then upload your CSV and run the experiments.

If Render assigns a different service URL, use that URL in Vercel. The Render
Blueprint's `CORS_ALLOW_ORIGINS` must include the exact frontend origin. The
API's `QKERNEL_DATA_DIR` points to the mounted disk; result files and uploaded
CSV data survive service restarts. Running benchmark jobs stop when a worker
restarts or redeploys, so restart any interrupted jobs.

## Start the application

The full research workflow requires a persistent FastAPI worker. For local
development, run the API and frontend as separate processes:

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

- A judge-oriented research dashboard centered on the scarce-label question,
  with a four-budget scarcity curve populated only from saved benchmark runs,
  paired verdict details, pre-run estimates for complete grids, and a
  rule-based experiment planner.
- A one-click staged judge demo with visible benchmark/noise/mitigation
  progress and saved evidence replay. It is explicitly a staged showcase: the
  noisy and mitigated stages use a separate balanced subset protocol.
- Dataset validation and live class-balance statistics.
- Qiskit feature-map inspection and exact statevector kernel-pair explorer.
- Background, cancellable ideal benchmark jobs with ten configurable seeds,
  training-only CV for model selection, classical baselines, stored metrics,
  test-set decision scores, and paired per-seed verdicts.
- Eight-configuration entanglement ablation (k = 6, 10, 20, 50 × RZZ on/off)
  and nine-configuration feature-map scaling sweep (4/6/8 qubits × 1/2/3
  repeats), each with mean ± standard deviation across seeds.
- A 256/1024/4096 shot sweep with seed standard-deviation error bars, and a
  local one-click judge demo that sequences ideal, noisy, and mitigated runs.
- A SHA-256-keyed, 512 MiB-bounded local kernel-matrix cache keyed by dataset,
  data splits, preprocessing, and feature-map configuration.
- Cross-run configuration replay and a saved-run review with run-specific
  metrics, verdict, configuration, dataset hash, seed summaries, and kernel
  heatmap; the run registry also compares selected configurations.
- A Fraud Case Lab with actual-vs-predicted labels, disagreement counts, and
  filters for fraud-labelled cases each model catches while the other misses.
- Kernel-geometry means (within-fraud, within-legitimate, and cross-class)
  calculated from the saved training matrix. These are descriptive similarities,
  not predictive performance metrics.
- Dataset validation checklist and data provenance. Example transactions are
  selected from the loaded dataset and retain their real `Class` labels; no
  hand-authored sample is presented as labelled data.
- An exact local runtime estimate for ideal kernel construction. It does not
  include CV/model fitting and is an extrapolation, not a completion-time
  guarantee.
- Four-qubit Qiskit Aer noise and zero-noise extrapolation demonstrations across
  three or five independent seeds, each using a balanced 60-row subset. These
  remain quick demos, not full-protocol benchmark evidence.
- Result listing/export, including the Markdown report, saved prediction
  exploration, and saved ideal training-kernel heatmaps.
- Persisted-model live scoring with an exact quantum-kernel SVC support-vector
  contribution trace, a reference percentile from cross-validated training
  scores, and a UI-only threshold policy sandbox. The contribution trace is a
  model-margin explanation, not a per-feature causal explanation.

## Research limits to keep visible

The ideal benchmark and small noisy/ZNE demonstration are separate protocols.
The noise demonstration uses at least three seeds and a fixed 30/30 balanced
subset per seed; it does not implement the full 100-fraud/400-normal test
protocol for noisy or mitigated runs.
There is no same-split, full-protocol Aer noise benchmark yet. The Dashboard
and Noise Lab keep the quick controlled simulation separate from the main
benchmark so judges do not mistake the two result sets for paired evidence.
The ablation grids and shot sweep run as multiple independent jobs and may be
long; the estimator covers kernel/simulator work and is not a completion-time
guarantee. Noise sweeps use balanced demo subsets, not the full ideal benchmark
protocol. Thermal-relaxation noise is not modeled. Live decision scores are not
calibrated probabilities; the threshold sandbox is exploratory and does not
retrain the model or establish a production operating point. Use the persistent
API worker for reproducible experiments; Vercel hosts the static frontend.

Treat every result as prototype research output. Simulation time is not
quantum-hardware performance. Accuracy can mislead when fraud is rare. A
balanced subset changes class prevalence. Small sample comparisons are
uncertain, and this work cannot establish universal quantum advantage.

## Project layout

- `frontend/`: React, TypeScript and Vite user interface.
- `backend/app/`: FastAPI API, dataset checks, Qiskit utilities, benchmark job
  workers, Qiskit Aer noise and mitigation experiment. These research routes
  remain in a consolidated module for this prototype; splitting them into
  separate API/domain packages is future maintenance work.
- `datasets/`: user-managed local data; files are ignored by Git.
- `results/`: generated experiment folders; files are ignored by Git.
- `frontend/src/content.ts`: centralized application copy.

The UI uses ink `#17232D`, paper `#F4F2EC`, slate `#64717A`, rust `#A94F35`,
moss `#557565`, and gold `#B38A3E`; DM Sans for reading and DM Mono for
measurements. It supports system dark mode, reduced motion, and responsive
layouts.

If `VITE_API_URL` is not set in Vercel, the frontend uses same-origin `/api`
requests and the app will not reach the Render worker. Set the variable and
redeploy after creating the backend.
