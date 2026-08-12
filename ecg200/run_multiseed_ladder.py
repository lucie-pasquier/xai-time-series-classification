"""Multi-seed retrofit for Models 2-4 (CNNs). Seeds 0-4. Saves _multiseed_results.json."""
import json, time, sys
from pathlib import Path
BASE = Path(__file__).resolve().parent                 # ecg200/
sys.path.insert(0, str(BASE.parent))                   # repo root -> import harness/ecg200
import numpy as np
from harness.models.cnn import build_cnn, train_cnn, set_seed, torch_predict_proba, count_parameters
from harness.evaluation.metrics import classification_metrics
from harness.xai.regions import build_region_grid
from ecg200.config import REGION_SIZE_PRIMARY_PCT
from harness.xai.feature_ablation import feature_ablation
from harness.xai.kernel_shap import kernel_shap
from harness.xai.integrated_gradients import integrated_gradients
from harness.xai.deletion_curves import perturbation_curves
from harness.xai.cmi import compute_cmi
from harness.xai.concentration import concentration_from_importances

P=str(BASE/'data'/'processed')+'/'; load=lambda s:(np.load(P+f'X_{s}.npy'), np.load(P+f'y_{s}.npy'))
Xtr,ytr=load('train'); Xva,yva=load('val'); Xte,yte=load('test')
grid=build_region_grid(Xte.shape[1], REGION_SIZE_PRIMARY_PCT); NR=grid.n_regions
SEEDS=[0,1,2,3,4]; ARCHS=['shallow','medium','deep']
PMS=['zero','laplace']   # sample_mean == zero on z-scored data (proven M1-M4)

def prof(A):
    m=np.abs(np.asarray(A)).mean(0); return m/m.max()
def cmi_of(attr_list, pm):
    M,L=[],[]
    for s,a in zip(Xte, attr_list):
        c=perturbation_curves(pp,s,grid,a,method=pm); M.append(c['MoRF']); L.append(c['LeRF'])
    r=compute_cmi(M,L); return r

out={'seeds':SEEDS,'pms':PMS,'note':'sample_mean==zero (z-scored); oracle==FA(zero); concentration from FA(zero) reliance','arch':{}}
t_all=time.time()
for arch in ARCHS:
    out['arch'][arch]={'params':int(count_parameters(build_cnn(arch))),'per_seed':[]}
    for seed in SEEDS:
        t0=time.time()
        set_seed(seed); model=build_cnn(arch); train_cnn(model,Xtr,ytr,Xva,yva,seed=seed)
        pp=torch_predict_proba(model)
        def acc(X,y):
            pr=pp(X); m=classification_metrics(y,pr.argmax(1),pr); return m,pr
        mtr,_=acc(Xtr,ytr); mva,_=acc(Xva,yva); mte,prte=acc(Xte,yte)
        collapsed = (len(np.unique(prte.argmax(1)))==1) or (mte['balanced_accuracy']<0.55)
        rec={'seed':seed,'test_acc':mte['accuracy'],'balanced_accuracy':mte['balanced_accuracy'],
             'f1':mte['f1'],'roc_auc':mte['roc_auc'],'collapsed':bool(collapsed),
             'overfit_gap':float(mva['accuracy']-mte['accuracy'])}
        if not collapsed:
            # zero-PM attributions (reused for agreement, concentration, oracle, zero-CMI)
            FAz=np.array([feature_ablation(pp,s,grid,'zero') for s in Xte])
            KSz=np.array([kernel_shap(pp,s,grid,'zero') for s in Xte])
            IGz=np.array([integrated_gradients(model,s,grid,'zero') for s in Xte])
            pf={'FA':prof(FAz),'KS':prof(KSz),'IG':prof(IGz)}
            rec['agreement']={'FA_KS':float(np.corrcoef(pf['FA'],pf['KS'])[0,1]),
                              'FA_IG':float(np.corrcoef(pf['FA'],pf['IG'])[0,1]),
                              'KS_IG':float(np.corrcoef(pf['KS'],pf['IG'])[0,1])}
            rec['concentration']=float(np.mean([concentration_from_importances(r) for r in FAz]))
            cmi={'FeatureAblation':{},'KernelSHAP':{},'IntegratedGradients':{}}
            pes={'FeatureAblation':{},'KernelSHAP':{},'IntegratedGradients':{}}
            for name,attrs_z,fn in [('FeatureAblation',FAz,feature_ablation),('KernelSHAP',KSz,kernel_shap),('IntegratedGradients',IGz,integrated_gradients)]:
                for pm in PMS:
                    if pm=='zero': attrs=attrs_z
                    elif name=='IntegratedGradients': attrs=[integrated_gradients(model,s,grid,pm) for s in Xte]
                    else: attrs=[fn(pp,s,grid,method=pm) for s in Xte]
                    r=cmi_of(attrs,pm); cmi[name][pm]=float(r['CMI']); pes[name][pm]=float(r['PES'])
            rec['cmi']=cmi; rec['pes']=pes
            rec['oracle']=cmi['FeatureAblation']['zero']  # FA(zero)==oracle
            rng=np.random.RandomState(0)
            rec['random']=float(cmi_of([rng.rand(NR) for _ in Xte],'zero')['CMI'])
        out['arch'][arch]['per_seed'].append(rec)
        print(f"[{arch} seed{seed}] test={mte['accuracy']:.3f} collapsed={collapsed} "
              f"{'CMI-FAz=%.3f conc=%.3f agr=%.2f/%.2f/%.2f'%(rec.get('oracle',float('nan')),rec.get('concentration',float('nan')),*(rec['agreement'].values() if 'agreement' in rec else (0,0,0)))} "
              f"({time.time()-t0:.0f}s)", flush=True)
        json.dump(out, open(BASE/'results'/'metrics'/'multiseed_ladder.json','w'), indent=1)
print(f"TOTAL {time.time()-t_all:.0f}s", flush=True)

# Model 1 determinism check (LogisticRegression is deterministic; KS seed fixed)
from ecg200.linear_baseline import extract_band_power_features, build_linear_baseline
Ftr,_=extract_band_power_features(Xtr)
c1=[build_linear_baseline().fit(Ftr,ytr).predict(extract_band_power_features(Xte)[0]).tolist() for _ in range(3)]
print('Model1 deterministic across refits:', c1[0]==c1[1]==c1[2], flush=True)
json.dump(out, open(BASE/'results'/'metrics'/'multiseed_ladder.json','w'), indent=1)
print('DONE', flush=True)
