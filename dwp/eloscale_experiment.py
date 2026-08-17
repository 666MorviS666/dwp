"""Гипотеза: elo_div400 задавлен регуляризацией, рассчитанной на 127
геройских коэффициентов. Столбец один, разброс 0.185 — штраф C=0.05
съедает его вместе с героями.

Проверка: масштабируем столбец. Коэффициент при x*s ведёт себя как
штраф C*s^2, то есть больший s = слабее штраф именно на elo.
Протокол хронологический, шагающий: обучаемся только на прошлом.

Данные: `python -m dwp.export --what draft` -> data/export_draft.csv.gz.
Раньше здесь стоял абсолютный путь чужой машины (/mnt/user-data/uploads/),
и скрипт не запускался нигде, кроме той машины, — то есть вывод в
config.DRAFT_ELO_SCALE нечем было перепроверить.
"""
import sys
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss

from . import config

SRC = config.DATA_DIR / 'export_draft.csv.gz'
if not SRC.exists():
    sys.exit(f'Нет {SRC}. Сначала: python -m dwp.export --what draft')
d=pd.read_csv(SRC).sort_values('start_time').reset_index(drop=True)
H=[c for c in d.columns if c.startswith('h') and c[1:].isdigit()]
Xh=d[H].values.astype(np.float64); e=d[['elo_div400']].values.astype(np.float64)
y=d.y.values; n=len(y); edges=[int(n*i/5) for i in range(6)]
def walk(s,C):
    ps=[];ys=[];bs=[]
    for k in range(1,5):
        tr=np.arange(edges[k]); te=np.arange(edges[k],edges[k+1])
        X=np.c_[Xh,e*s]
        lr=LogisticRegression(C=C,max_iter=8000).fit(X[tr],y[tr])
        ps.append(lr.predict_proba(X[te])[:,1]); ys.append(y[te])
        bs.append(np.full(len(te),y[tr].mean()))
    p=np.concatenate(ps); yy=np.concatenate(ys); b=np.concatenate(bs)
    return roc_auc_score(yy,p),log_loss(yy,p),log_loss(yy,b)
print('%-6s %-6s %8s %10s %10s'%('масштаб','C','AUC','log_loss','база'))
for s in (1,2,3,5,10,20):
    for C in (0.02,0.05,0.1):
        a,l,b=walk(s,C); print('%-7g %-6g %8.4f %10.4f %10.4f'%(s,C,a,l,b))
