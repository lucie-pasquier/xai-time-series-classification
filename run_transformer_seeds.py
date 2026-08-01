"""Model 5 (transformer) — 5-seed train-only pass. Saves results/metrics/transformer_seeds.json.
Recipe: shared CNN train_cnn defaults (Adam lr 1e-3, wd 1e-4, batch 16, 200 ep, patience 30,
class-weighted CE), seed-before-build per seed. Transformer has built-in dropout 0.1."""
import json, time, warnings; warnings.filterwarnings("ignore")
import numpy as np
from src.models.transformer import build_transformer
from src.models.cnn import train_cnn, set_seed, torch_predict_proba, count_parameters
from src.evaluation.metrics import classification_metrics

P='data/processed/'; load=lambda s:(np.load(P+f'X_{s}.npy'), np.load(P+f'y_{s}.npy'))
Xtr,ytr=load('train'); Xva,yva=load('val'); Xte,yte=load('test')
SEEDS=[0,1,2,3,4]
out={'seeds':SEEDS,'params':int(count_parameters(build_transformer())),
     'recipe':'shared CNN train_cnn (Adam lr1e-3 wd1e-4 batch16 200ep patience30 class-weighted CE); dropout 0.1 in model',
     'patch':'per-timestep (patch_size=1), 96 tokens, grid-exact','per_seed':[]}
t0=time.time()
for seed in SEEDS:
    ts=time.time(); set_seed(seed); m=build_transformer(); r=train_cnn(m,Xtr,ytr,Xva,yva,seed=seed)
    pp=torch_predict_proba(m)
    S={}
    for nm,X,y in [('train',Xtr,ytr),('val',Xva,yva),('test',Xte,yte)]:
        pr=pp(X); S[nm]=classification_metrics(y,pr.argmax(1),pr)
    prte=pp(Xte)
    collapsed = (len(np.unique(prte.argmax(1)))==1) or (S['test']['balanced_accuracy']<0.55)
    out['per_seed'].append({'seed':seed,'collapsed':bool(collapsed),
        'test':S['test'],'val':S['val'],'train':S['train'],
        'overfit_gap':float(S['val']['accuracy']-S['test']['accuracy']),
        'best_epoch':r['best_epoch'],'n_epochs':len(r['history']['train_loss']),
        'history':r['history']})
    print(f"[seed{seed}] test={S['test']['accuracy']:.3f} bal={S['test']['balanced_accuracy']:.3f} "
          f"f1={S['test']['f1']:.3f} auc={S['test']['roc_auc']:.3f} gap={out['per_seed'][-1]['overfit_gap']:+.3f} "
          f"collapsed={collapsed} best_ep={r['best_epoch']} ({time.time()-ts:.0f}s)", flush=True)
    json.dump(out, open('results/metrics/transformer_seeds.json','w'))
acc=np.array([r['test']['accuracy'] for r in out['per_seed']])
print(f"\nparams={out['params']:,}  test acc mean±std = {acc.mean():.3f}±{acc.std():.3f}  spread [{acc.min():.3f},{acc.max():.3f}]  collapsed {sum(r['collapsed'] for r in out['per_seed'])}/5  TOTAL {time.time()-t0:.0f}s", flush=True)
print("DONE", flush=True)
