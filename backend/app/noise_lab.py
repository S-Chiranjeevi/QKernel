"""Small, explicitly exploratory Aer noise and ZNE runs for the local labs."""
from __future__ import annotations
import math
import time
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

PRESETS={"LOW":(0.0005,0.005,0.01),"MEDIUM":(0.001,0.01,0.02),"HIGH":(0.003,0.03,0.05)}

def feature_map(values, repeats=1, entangle=True):
    n=len(values); qc=QuantumCircuit(n)
    for _ in range(repeats):
        for i,x in enumerate(values): qc.h(i); qc.rz(2*float(x),i)
        if entangle:
            for i in range(n):
                for j in range(i+1,n): qc.rzz(2*(math.pi-values[i])*(math.pi-values[j]),i,j)
    return qc

def fold(circuit, scale):
    """Transparent odd-factor unitary folding: G -> G(G†G)^k."""
    if scale not in (1,3,5): raise ValueError("Folding scale must be 1, 3 or 5.")
    out=QuantumCircuit(circuit.num_qubits)
    k=(scale-1)//2
    for inst in circuit.data:
        op=inst.operation; qargs=[circuit.find_bit(q).index for q in inst.qubits]
        if op.name in {"barrier","measure"}:
            out.append(op,qargs); continue
        out.append(op,qargs)
        for _ in range(k):
            out.append(op.inverse(),qargs); out.append(op,qargs)
    out.global_phase=circuit.global_phase
    return out

def make_noise(preset):
    p1,p2,pr=PRESETS[preset]
    model=NoiseModel()
    model.add_all_qubit_quantum_error(depolarizing_error(p1,1),["h","rz"])
    model.add_all_qubit_quantum_error(depolarizing_error(p2,2),["rzz"])
    model.add_all_qubit_readout_error(ReadoutError([[1-pr,pr],[pr,1-pr]]))
    return model

def kernel_circuits(left,right,train_symmetric,scale):
    maps_left=[feature_map(x) for x in left]
    maps_right=maps_left if train_symmetric else [feature_map(x) for x in right]
    pairs=[(i,j) for i in range(len(left)) for j in range(i,len(left))] if train_symmetric else [(i,j) for i in range(len(left)) for j in range(len(right))]
    circuits=[]
    for i,j in pairs:
        qc=maps_left[i].compose(maps_right[j].inverse())
        qc=fold(qc,scale)
        qc.measure_all(); circuits.append(qc)
    return pairs,circuits

def sampled_kernel(left,right,model,shots,seed,scale,train_symmetric):
    pairs,circuits=kernel_circuits(left,right,train_symmetric,scale)
    sim=AerSimulator(noise_model=model,method="density_matrix")
    compiled=transpile(circuits,sim,optimization_level=0)
    counts=sim.run(compiled,shots=shots,seed_simulator=seed).result().get_counts()
    vals=np.array([c.get("0"*left.shape[1],0)/shots for c in counts],dtype=float)
    if train_symmetric:
        mat=np.zeros((len(left),len(left)))
        for (i,j),v in zip(pairs,vals): mat[i,j]=v; mat[j,i]=v
        return mat
    return vals.reshape(len(left),len(right))

def exact_kernel(left,right,symmetric=False):
    a=np.stack([np.asarray(Statevector.from_instruction(feature_map(x)).data) for x in left])
    b=a if symmetric else np.stack([np.asarray(Statevector.from_instruction(feature_map(x)).data) for x in right])
    return np.abs(a@b.conj().T)**2

def repair_train(matrix):
    clipped=np.clip(matrix,0,1); symmetric=(clipped+clipped.T)/2
    vals,vecs=np.linalg.eigh(symmetric); repaired=(vecs*np.maximum(vals,0))@vecs.T
    return symmetric,repaired,vals

def extrapolate(scales):
    y=np.stack(scales,axis=0); x=np.array([1.,3.,5.])
    # Linear and degree-two Richardson extrapolations, evaluated at scale zero.
    linear=np.polynomial.polynomial.polyfit(x,y.reshape(3,-1),1)[0].reshape(y.shape[1:])
    rich=np.polynomial.polynomial.polyfit(x,y.reshape(3,-1),2)[0].reshape(y.shape[1:])
    # Exponential model with zero asymptote: log(y)=log(a)-b*scale.
    safe=np.clip(y,1e-8,1).reshape(3,-1)
    exp0=np.exp(np.polynomial.polynomial.polyfit(x,np.log(safe),1)[0]).reshape(y.shape[1:])
    return {"linear":linear,"richardson":rich,"exponential":exp0}

def score_kernel(ktrain,ktest,ytrain,ytest,repair=False):
    if repair: _,ktrain,_=repair_train(ktrain)
    ktest=np.clip(ktest,0,1)
    model=SVC(C=1,kernel="precomputed",class_weight="balanced")
    t=time.perf_counter();model.fit(ktrain,ytrain);fit=time.perf_counter()-t
    t=time.perf_counter();decision=model.decision_function(ktest);predict=time.perf_counter()-t
    pred=(decision>=0).astype(int)
    return {"PR-AUC":float(average_precision_score(ytest,decision)),"recall":float(recall_score(ytest,pred,zero_division=0)),"precision":float(precision_score(ytest,pred,zero_division=0)),"F1":float(f1_score(ytest,pred,zero_division=0)),"ROC-AUC":float(roc_auc_score(ytest,decision)),"accuracy":float(np.mean(pred==ytest)),"fit_time":fit,"prediction_time":predict},decision

def run_lab(frame,qubits=4,preset="MEDIUM",shots=256,seed=0,mitigate=False,synthetic=False,progress=lambda *_:None):
    """Run one balanced 60-row demonstration. It is never conclusion-grade."""
    features=[*(f"V{i}" for i in range(1,29)),"Amount"]
    y=frame.Class.to_numpy(dtype=int); pos=np.flatnonzero(y==1); neg=np.flatnonzero(y==0)
    if len(pos)<30 or len(neg)<30: raise ValueError("Noise Lab needs at least 30 fraud and 30 legitimate rows for its balanced 60-row demo.")
    rng=np.random.default_rng(seed); ids=np.r_[rng.choice(pos,30,replace=False),rng.choice(neg,30,replace=False)]; rng.shuffle(ids)
    X=frame.iloc[ids][features].to_numpy(dtype=float); labels=y[ids]
    itr,ite=train_test_split(np.arange(60),test_size=1/3,random_state=seed,stratify=labels)
    ytr,yte=labels[itr],labels[ite]
    standard=StandardScaler().fit(X[itr]); xs=standard.transform(X[itr]); xv=standard.transform(X[ite])
    pca=PCA(n_components=qubits,random_state=seed).fit(xs); ztr=pca.transform(xs); zte=pca.transform(xv)
    mm=MinMaxScaler((0,math.pi),clip=True).fit(ztr); atr=mm.transform(ztr); ate=mm.transform(zte)
    progress(10,"Prepared a balanced 60-row demonstration subset.")
    ideal_train=exact_kernel(atr,atr,True); ideal_test=exact_kernel(ate,atr,False)
    ideal_metrics,_=score_kernel(ideal_train,ideal_test,ytr,yte)
    model=make_noise(preset); progress(20,"Building simulated compute-uncompute circuits.")
    noisy_train=sampled_kernel(atr,atr,model,shots,seed,1,True)
    noisy_test=sampled_kernel(ate,atr,model,shots,seed+100,1,False)
    noisy_metrics,_=score_kernel(noisy_train,noisy_test,ytr,yte)
    result={"protocol":"Quick demo, too small for conclusions","balanced_subset_notice":"This subset is used to make simulation feasible and does not represent natural fraud prevalence.","ideal":ideal_metrics,"noisy":noisy_metrics,"noise_preset":preset,"noise_values":{"single_qubit":PRESETS[preset][0],"two_qubit":PRESETS[preset][1],"readout":PRESETS[preset][2]},"shots":shots,"qubits":qubits,"repeats":1,"train_count":len(itr),"test_count":len(ite),"fraud_labels_train":int(ytr.sum()),"synthetic":bool(synthetic)}
    if mitigate:
        scale_train=[];scale_test=[]
        for ix,scale in enumerate((1,3,5)):
            progress(35+ix*15,f"Measuring folded circuits at scale {scale}.")
            scale_train.append(sampled_kernel(atr,atr,model,shots,seed+scale,scale,True))
            scale_test.append(sampled_kernel(ate,atr,model,shots,seed+100+scale,scale,False))
        train_est=extrapolate(scale_train); test_est=extrapolate(scale_test)
        methods={}; matrices={}
        for name,kt in train_est.items():
            raw=np.clip((kt+kt.T)/2,0,1); _,fixed,eigs=repair_train(raw)
            score_before,_=score_kernel(raw,test_est[name],ytr,yte)
            score_after,_=score_kernel(fixed,test_est[name],ytr,yte)
            methods[name]={"before_repair":score_before,"after_repair":score_after,"minimum_eigenvalue_before":float(eigs.min()),"minimum_eigenvalue_after":float(np.linalg.eigvalsh(fixed).min())}
            matrices[name]={"raw_train":raw.tolist(),"repaired_train":fixed.tolist(),"raw_test_train":np.clip(test_est[name],0,1).tolist()}
        def offdiag_mean(matrix):
            vals=matrix[np.triu_indices_from(matrix,k=1)]
            return float(np.mean(vals)) if len(vals) else float(np.mean(matrix))
        result["mitigation"]={"headline_method":"linear","scales":[1,3,5],"metrics":methods,"matrices":matrices,"scale_train_matrices":[x.tolist() for x in scale_train],"label":"Software simulation","kernel_summary":{"ideal":offdiag_mean(ideal_train),"scale_1":offdiag_mean(scale_train[0]),"scale_3":offdiag_mean(scale_train[1]),"scale_5":offdiag_mean(scale_train[2]),"extrapolators":{name:offdiag_mean(mat) for name,mat in train_est.items()}}}
        degradation=ideal_metrics["PR-AUC"]-noisy_metrics["PR-AUC"]
        recovered=methods["linear"]["after_repair"]["PR-AUC"]
        if degradation<=0.02:
            result["recovery_note"]="No meaningful degradation to recover. The run is a one-seed demo, so no confidence interval is calculated."
        elif recovered<=noisy_metrics["PR-AUC"]:
            result["recovery_note"]="Mitigation did not improve this experiment. The run is a one-seed demo, so no confidence interval is calculated."
        else:
            result["recovery_note"]="The mitigated point estimate is above the noisy point estimate, but recovery is withheld because this one-seed demo cannot estimate across-seed uncertainty."
        progress(90,"Completed extrapolation and positive-semidefinite kernel repair.")
    result["kernel_matrices"]={"ideal_train":ideal_train.tolist(),"ideal_test_train":ideal_test.tolist(),"noisy_train":noisy_train.tolist(),"noisy_test_train":noisy_test.tolist()}
    result["limitations"]=["Quick demo, too small for conclusions.",result["balanced_subset_notice"],"One seed; no inferential verdict is calculated.","Noise values are a parameterized simulator model, not measured hardware noise.","This demonstration uses fixed model parameters and is not the full benchmark CV protocol."]
    return result
