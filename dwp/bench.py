r"""Стенд для сравнения вариантов на ОДНОМ разбиении и по многим разбиениям.

Читает выгрузки `dwp.export` (frames + draft), а не сырые JSON, поэтому
один прогон занимает секунды вместо полутора минут. Логика скопирована
из train.py; воспроизведение сверено с боевым прогоном (seed 7, 4927
матчей): драфт OOF auc 0.5888 и ll 0.6798 совпали до четвёртого знака,
стейт ll 0.5183 против 0.5178 — расхождение от недетерминизма LightGBM.

Зачем отдельный стенд. Итоговая метрика на ОДНОМ отложенном тесте
пляшет от разбиения сильнее, чем от большинства правок: замерено
ст.откл 0.0079 по семи сидам при неизменных данных и модели. Поэтому
режим multi гоняет несколько разбиений и печатает разброс, а не одно
число.

Запуск:
    python -m dwp.bench single --frames data\frames.csv.gz --draft data\draft.csv.gz
    python -m dwp.bench multi  --frames data\frames.csv.gz --draft data\draft.csv.gz --seeds 1,3,5,7,11
    python -m dwp.bench ab --elo-scale 10 --draft-c 0.02 --frames ... --draft ...
"""

import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
EPS=1e-6
LGBM=dict(objective="binary",learning_rate=0.05,num_leaves=31,min_data_in_leaf=200,
          feature_fraction=0.9,bagging_fraction=0.8,bagging_freq=1,lambda_l2=5.0,
          verbosity=-1,num_threads=0)
BASE=["minute","gold_adv","gold_adv_d1","gold_adv_slope5","radiant_towers_lost",
      "dire_towers_lost","tower_adv","rax_adv","roshan_radiant","roshan_dire",
      "minutes_since_roshan","tormentor_radiant","tormentor_dire","draft_logit"]
XP=["xp_adv","xp_adv_d1"]
EXTRA=["kills_adv","kills_adv_d5","nw_top_adv","nw_conc_adv","bb_used_radiant",
       "bb_used_dire","bb_oncd_adv"]
def logit(p): p=np.clip(np.asarray(p,float),EPS,1-EPS); return np.log(p/(1-p))
def sig(z): return 1/(1+np.exp(-z))
def split_ids(ids,ts,seed):
    g=GroupShuffleSplit(n_splits=1,test_size=ts,random_state=seed)
    a,b=next(g.split(np.zeros(len(ids)),groups=ids)); return ids[a],ids[b]
def rel_gap(y,p,n_bins=10):
    e=np.linspace(0,1,n_bins+1); i=np.clip(np.digitize(p,e[1:-1],right=False),0,n_bins-1)
    w=[];gp=[]
    for b in range(n_bins):
        m=i==b
        if m.any(): w.append(m.sum()); gp.append(abs(p[m].mean()-y[m].mean()))
    w=np.array(w,float); gp=np.array(gp)
    return float((gp*w).sum()/w.sum()), float(gp.max())
def fit_cal(method,p,y):
    if method=="none": return None
    if method=="isotonic":
        m=IsotonicRegression(out_of_bounds="clip",y_min=0.,y_max=1.); m.fit(p,y); return m
    lr=LogisticRegression(C=1e6,max_iter=1000); lr.fit(logit(p).reshape(-1,1),y); return lr
def app_cal(c,p):
    if c is None: return np.asarray(p,float)
    if isinstance(c,IsotonicRegression): return c.predict(p)
    return c.predict_proba(logit(p).reshape(-1,1))[:,1]

def run(fr,dr,seed=7,elo_scale=1.0,draft_C=0.05,feats_extra=True,use_xp=True,
        extra_cols=None,split_groups=None,verbose=True,tag="",pool_frac=1.0,pool_seed=0):
    mids=fr.match_id.drop_duplicates().to_numpy()          # порядок загрузки
    dr=dr.set_index("match_id").loc[mids].reset_index()
    H=[c for c in dr.columns if c.startswith("h") and c[1:].isdigit()]
    Xd=np.c_[dr[H].to_numpy(np.float32), dr[["elo_div400"]].to_numpy(np.float32)*elo_scale]
    yd=dr.y.to_numpy()
    if split_groups is None:
        pool_ids,test_ids=split_ids(mids,0.20,seed)
    else:
        gs=pd.Series(split_groups).loc[:].to_numpy()
        g=GroupShuffleSplit(n_splits=1,test_size=0.20,random_state=seed)
        a,b=next(g.split(np.zeros(len(mids)),groups=gs)); pool_ids,test_ids=mids[a],mids[b]
    if pool_frac<1.0:
        # Тест НЕ трогаем: он должен остаться тем же, иначе кривая обучения
        # смешается с составом теста — ровно та ошибка, которую и проверяем.
        rs=np.random.default_rng(pool_seed)
        keep=rs.choice(len(pool_ids),int(round(len(pool_ids)*pool_frac)),replace=False)
        pool_ids=np.sort(pool_ids[keep])
        keep_all=np.isin(mids,np.concatenate([pool_ids,test_ids]))
        mids=mids[keep_all]; Xd=Xd[keep_all]; yd=yd[keep_all]
        fr=fr[fr.match_id.isin(set(mids.tolist()))]
    pm=np.isin(mids,pool_ids); tm=~pm
    est=lambda: CalibratedClassifierCV(LogisticRegression(C=draft_C,max_iter=3000),method="sigmoid",cv=5)
    oof=cross_val_predict(est(),Xd[pm],yd[pm],cv=5,method="predict_proba")[:,1]
    dm=est().fit(Xd[pm],yd[pm])
    pdft=np.empty(len(mids)); pdft[pm]=oof; pdft[tm]=dm.predict_proba(Xd[tm])[:,1]
    draft={"oof_auc":roc_auc_score(yd[pm],oof),"oof_ll":log_loss(yd[pm],oof),
           "test_auc":roc_auc_score(yd[tm],pdft[tm]),"test_ll":log_loss(yd[tm],pdft[tm]),
           "test_base":log_loss(yd[tm],np.full(tm.sum(),yd[pm].mean()))}
    df=fr.copy()
    df["draft_logit"]=df.match_id.map(dict(zip(mids.tolist(),logit(pdft).tolist())))
    feats=BASE+(XP if use_xp else [])+(EXTRA if feats_extra else [])+(extra_cols or [])
    pf,val=split_ids(pool_ids,0.15/0.80,seed+1); fit,cal=split_ids(pf,0.25,seed+2)
    sel=lambda ids: df[df.match_id.isin(set(ids.tolist()))]
    d_f,d_v,d_c,d_t=sel(fit),sel(val),sel(cal),sel(test_ids)
    bo=lgb.train(LGBM,lgb.Dataset(d_f[feats],label=d_f.y),num_boost_round=400,
                 valid_sets=[lgb.Dataset(d_v[feats],label=d_v.y)],
                 callbacks=[lgb.early_stopping(40,verbose=False),lgb.log_evaluation(0)])
    pc=bo.predict(d_c[feats],num_iteration=bo.best_iteration); yc=d_c.y.to_numpy(); gc=d_c.match_id.to_numpy()
    folds=list(GroupKFold(n_splits=5).split(pc,yc,groups=gc))
    best=None
    for meth in ("none","sigmoid","isotonic"):
        o=np.empty(len(yc))
        for tr,te in folds: o[te]=app_cal(fit_cal(meth,pc[tr],yc[tr]),pc[te])
        for e in [0.,1e-3,3e-3,1e-2,2e-2,5e-2,8e-2,0.12]:
            q=np.clip(o,max(e,EPS),1-max(e,EPS)); ec,_=rel_gap(yc,q)
            if best is None or ec<best[0]: best=(ec,meth,e)
    _,meth,eps=best
    cal_f=fit_cal(meth,pc,yc)
    yt=d_t.y.to_numpy(); pr=bo.predict(d_t[feats],num_iteration=bo.best_iteration)
    pt=np.clip(app_cal(cal_f,pr),max(eps,EPS),1-max(eps,EPS))
    ec,mx=rel_gap(yt,pt)
    out=dict(tag=tag,n_test_matches=int(d_t.match_id.nunique()),n_test_rows=len(yt),
             trees=bo.best_iteration,cal=meth,eps=eps,
             ll=log_loss(yt,pt),brier=brier_score_loss(yt,pt),
             acc=float(((pt>=.5)==yt).mean()),auc=roc_auc_score(yt,pt),
             base=log_loss(yt,np.full(len(yt),d_f.y.mean())),ece=ec,maxgap=mx,**{"draft_"+k:v for k,v in draft.items()})
    if verbose:
        print(f"[{tag}] test матчей {out['n_test_matches']} строк {out['n_test_rows']} | деревьев {out['trees']} | кал. {meth} eps {eps:g}")
        print(f"   драфт: OOF auc {draft['oof_auc']:.4f} ll {draft['oof_ll']:.4f} | test auc {draft['test_auc']:.4f} ll {draft['test_ll']:.4f} (база {draft['test_base']:.4f})")
        print(f"   стейт: ll {out['ll']:.4f} brier {out['brier']:.4f} acc {out['acc']:.4f} auc {out['auc']:.4f} | база {out['base']:.4f} | ECE {ec:.3f} макс {mx:.3f}")
    return out,pt,yt,d_t.match_id.to_numpy(),d_t.minute.to_numpy()


def _paired(y,pa,pb,g,nb=2000,seed=0):
    """Парный бутстрап ПО МАТЧАМ — как в compare.py. Строки внутри матча
    скоррелированы, по строкам интервал выходит в разы уже настоящего."""
    rng=np.random.default_rng(seed); u=np.unique(g); idx={k:np.flatnonzero(g==k) for k in u}
    f=lambda yy,pp:-np.mean(yy*np.log(np.clip(pp,EPS,1-EPS))+(1-yy)*np.log(np.clip(1-pp,EPS,1-EPS)))
    d=np.empty(nb)
    for b in range(nb):
        rows=np.concatenate([idx[k] for k in rng.choice(u,len(u),True)])
        d[b]=f(y[rows],pb[rows])-f(y[rows],pa[rows])
    return f(y,pb)-f(y,pa),float(np.percentile(d,2.5)),float(np.percentile(d,97.5)),float((d<0).mean())


def main(argv=None):
    import argparse
    ap=argparse.ArgumentParser(description="Стенд сравнения вариантов на выгрузках.")
    ap.add_argument("mode",choices=["single","multi","ab"])
    ap.add_argument("--frames",required=True)
    ap.add_argument("--draft",required=True)
    ap.add_argument("--seed",type=int,default=7)
    ap.add_argument("--seeds",default="1,3,5,7,11,13,17")
    ap.add_argument("--elo-scale",type=float,default=10.0,help="для режима ab: вариант B")
    ap.add_argument("--draft-c",type=float,default=0.02,help="для режима ab: вариант B")
    a=ap.parse_args(argv)
    fr=pd.read_csv(a.frames); dr=pd.read_csv(a.draft)
    if a.mode=="single":
        run(fr,dr,seed=a.seed,tag=f"seed {a.seed}")
    elif a.mode=="multi":
        seeds=[int(s) for s in a.seeds.split(",")]
        lls=[];aucs=[]
        for s in seeds:
            o,*_=run(fr,dr,seed=s,verbose=False,tag=str(s))
            lls.append(o["ll"]); aucs.append(o["auc"])
            print(f"  seed {s:>3}: стейт ll {o['ll']:.4f}  auc {o['auc']:.4f}  "
                  f"драфт OOF auc {o['draft_oof_auc']:.4f}",flush=True)
        l=np.array(lls)
        print(f"\nlog_loss по {len(seeds)} разбиениям: {l.min():.4f}..{l.max():.4f}, "
              f"среднее {l.mean():.4f}, ст.откл {l.std(ddof=1):.4f}")
        print("Правку меньше этого разброса нельзя принимать по одному разбиению.")
    else:
        oa,pa,y,g,_=run(fr,dr,seed=a.seed,elo_scale=1.0,draft_C=0.05,verbose=False,tag="A")
        ob,pb,y2,g2,_=run(fr,dr,seed=a.seed,elo_scale=a.elo_scale,draft_C=a.draft_c,verbose=False,tag="B")
        assert np.array_equal(y,y2) and np.array_equal(g,g2),"выборки разъехались"
        d,lo,hi,pw=_paired(y.astype(float),pa,pb,g)
        print(f"A: elo*1 C=0.05   стейт ll {oa['ll']:.4f}  драфт OOF auc {oa['draft_oof_auc']:.4f}")
        print(f"B: elo*{a.elo_scale:g} C={a.draft_c:g}  стейт ll {ob['ll']:.4f}  драфт OOF auc {ob['draft_oof_auc']:.4f}")
        print(f"B-A на стейте: {d:+.4f}  95% [{lo:+.4f}, {hi:+.4f}]  P(B лучше) {pw:.3f}")
    return 0


if __name__=="__main__":
    import sys; sys.exit(main())
