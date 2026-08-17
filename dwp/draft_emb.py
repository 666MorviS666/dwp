"""Драфт по эмбеддингам героев вместо 127 одиночных коэффициентов.

ЗАЧЕМ ИМЕННО ЭТО. Ошибка разложена по фазам: **половина её лежит до 15-й
минуты**, где золота ещё нет, вышек нет, и единственный доступный сигнал —
драфт. А нынешняя драфт-модель слабая: 127 независимых коэффициентов, AUC
около 0.61, девять из десяти коэффициентов живого состава в пределах шума.
Она по построению не знает ни синергий, ни контрпиков — для неё состав
это сумма пятерых, и «эти двое вместе» для неё не существует.

ПОЧЕМУ НЕ ПАРЫ В ЛОБ. 127 героев дают ~8000 пар внутри команды и ~16000
между командами на 6.5 тысячи матчей. Это гарантированное переобучение, и
проверять там нечего. Нужно низкоранговое представление: у каждого героя
вектор из 8-16 чисел, а сила пары — скалярное произведение. Параметров
становится 127*k вместо 15000.

ФОРМА МОДЕЛИ И ПОЧЕМУ ОНА ИМЕННО ТАКАЯ. Обычная факторизационная машина на
кодировке «+1 за Radiant, −1 за Dire» здесь НЕВЕРНА, и это стоит написать
прямо, потому что ошибка незаметная. В FM слагаемое <v_i,v_j>*x_i*x_j для
пары внутри Dire даёт x_i*x_j = (−1)(−1) = +1, то есть синергия Dire
прибавляется к оценке RADIANT. Модель, которая считает, что удачная пара у
противника играет за нас, обучится чему угодно, только не синергии.

Поэтому синергия и контра разведены явно:

    score = b + w*elo
            + Σ_{i∈R} a_i − Σ_{j∈D} a_j                     одиночная сила
            + Σ_{i<j∈R} <u_i,u_j> − Σ_{i<j∈D} <u_i,u_j>     синергия внутри
            + (Σ_R v)^T M (Σ_D v),  M = A − Aᵀ              контра между

`M` антисимметрична НАМЕРЕННО: «i контрит j» обязано означать «j
контрится i-м» с обратным знаком, иначе модель выучит, что оба героя
контрят друг друга одновременно. При перестановке сторон все слагаемые,
кроме `b`, меняют знак — то есть модель по построению не считает, что
Radiant сильнее просто потому, что он Radiant; это остаётся работой `b`.

КАК ПРОВЕРЯЕТСЯ. Шагающим хронологическим протоколом: учимся только на
прошлом, проверяем на следующем куске времени. Так меряют драфт: мета
меняется патчами, и случайное разбиение дало бы модели заглянуть в
будущее. Тот же протокол уже применён к парным данным Stratz
(`eloscale_experiment.py`). Слепой холдаут при этом НЕ трогается вовсе —
он выбрасывается до всего, и весь подбор идёт на остальных матчах.

    python -m dwp.draft_emb --walk 5                # база против эмбеддингов
    python -m dwp.draft_emb --walk 5 --dim 16
    python -m dwp.draft_emb --walk 5 --halflife 45  # плюс веса по свежести
    python -m dwp.draft_emb --walk 5 --ablate       # что даёт каждое слагаемое
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score

from . import config, holdout as HO

DRAFT_CSV = config.DATA_DIR / "export_draft.csv.gz"
OUT_JSON = config.DATA_DIR / "draft_emb.json"
EPS = 1e-6


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


@dataclass
class Draft:
    """Выгрузка драфтов в виде, удобном модели."""
    R: np.ndarray          # (n, 5) индексы героев Radiant
    D: np.ndarray          # (n, 5) индексы героев Dire
    elo: np.ndarray        # (n,)
    y: np.ndarray          # (n,)
    mid: np.ndarray        # (n,)
    t: np.ndarray          # (n,) start_time
    X: np.ndarray          # (n, H+1) матрица базовой модели, как в train.py
    heroes: list[int]

    def take(self, sel) -> "Draft":
        return Draft(self.R[sel], self.D[sel], self.elo[sel], self.y[sel],
                     self.mid[sel], self.t[sel], self.X[sel], self.heroes)

    def __len__(self) -> int:
        return len(self.y)


def load_draft(path: Path | None = None, drop_holdout: bool = True) -> Draft:
    p = path or DRAFT_CSV
    if not p.exists():
        raise FileNotFoundError(
            f"нет выгрузки драфтов {p}.\n"
            f"Что делать: `python -m dwp.export --draft`")
    d = pd.read_csv(p)
    hc = [c for c in d.columns if c.startswith("h") and c[1:].isdigit()]
    heroes = [int(c[1:]) for c in hc]
    Xh = d[hc].to_numpy(dtype=np.int8)
    if drop_holdout:
        keep = ~np.array([HO.is_holdout(m) for m in d["match_id"].to_numpy()])
        d, Xh = d[keep].reset_index(drop=True), Xh[keep]
    # Ровно пять на сторону — проверено на выгрузке; если разошлось,
    # молча дополнять нулями нельзя, лучше упасть с числом.
    pos, neg = (Xh > 0).sum(1), (Xh < 0).sum(1)
    bad = int(((pos != 5) | (neg != 5)).sum())
    if bad:
        raise ValueError(f"в {bad} строках выгрузки не по пять героев на "
                         f"сторону — драфт разобран неверно")
    R = np.argsort(-Xh, axis=1, kind="stable")[:, :5].astype(np.int32)
    D = np.argsort(Xh, axis=1, kind="stable")[:, :5].astype(np.int32)
    elo = d["elo_div400"].to_numpy(dtype=np.float64)
    X = np.concatenate([Xh.astype(np.float32), elo[:, None].astype(np.float32)],
                       axis=1)
    return Draft(R, D, elo, d["y"].to_numpy(dtype=np.float64),
                 d["match_id"].to_numpy(dtype=np.int64),
                 d["start_time"].to_numpy(dtype=np.int64), X, heroes)


# --- модель --------------------------------------------------------------

class EmbDraft:
    """Логистическая модель с эмбеддингами синергии и контры.

    Полный батч и Adam: параметров порядка 127*2*k + k², матчей тысячи —
    это секунды, и дробить на мини-батчи незачем.
    """

    def __init__(self, n_heroes: int, dim: int = 8, l2_a: float = 1.0,
                 l2_emb: float = 1.0,
                 lr: float = 0.02, epochs: int = 3000, patience: int = 300,
                 use_syn: bool = True, use_cnt: bool = True,
                 warm: bool = True, lr_lin: float = 0.002,
                 seed: int = 0) -> None:
        self.H, self.k = n_heroes, dim
        self.l2 = {"a": l2_a, "u": l2_emb, "v": l2_emb, "m": l2_emb}
        self.lr, self.epochs, self.patience = lr, epochs, patience
        # Линейная часть учится ЗАМЕТНО медленнее эмбеддингов и стартует не
        # с нуля, а с готовой базовой модели. Иначе выходит гонка, в которой
        # эмбеддинги успевают запомнить обучающую выборку раньше, чем
        # линейные коэффициенты вообще сойдутся, — на первом заходе так и
        # получилось: лучшая эпоха всегда оказывалась первой.
        self.lr_lin = lr_lin
        self.warm = warm
        self.use_syn, self.use_cnt = use_syn, use_cnt
        self.seed = seed
        self.hist: list[tuple[int, float]] = []

    # --- прямой ход ------------------------------------------------------

    def _parts(self, R, D):
        uR, uD = self.u[R], self.u[D]                      # (n,5,k)
        sR, sD = uR.sum(1), uD.sum(1)                      # (n,k)
        vR, vD = self.v[R], self.v[D]
        pR, pD = vR.sum(1), vD.sum(1)
        return uR, uD, sR, sD, vR, vD, pR, pD

    def score(self, R, D, elo):
        uR, uD, sR, sD, _vR, _vD, pR, pD = self._parts(R, D)
        z = self.b + self.w * elo + self.a[R].sum(1) - self.a[D].sum(1)
        if self.use_syn:
            synR = 0.5 * ((sR ** 2).sum(1) - (uR ** 2).sum((1, 2)))
            synD = 0.5 * ((sD ** 2).sum(1) - (uD ** 2).sum((1, 2)))
            z = z + synR - synD
        if self.use_cnt:
            M = self.A - self.A.T
            z = z + ((pR @ M) * pD).sum(1)
        return z

    def predict_proba(self, d: Draft) -> np.ndarray:
        return _sig(self.score(d.R, d.D, d.elo))

    # --- обучение --------------------------------------------------------

    def fit(self, tr: Draft, va: Draft, w_tr: np.ndarray | None = None
            ) -> "EmbDraft":
        rng = np.random.default_rng(self.seed)
        H, k = self.H, self.k
        base = float(np.clip(tr.y.mean(), EPS, 1 - EPS))
        self.a = np.zeros(H)
        self.b = float(np.log(base / (1 - base)))
        self.w = 0.0
        if self.warm:
            # Тёплый старт от той самой модели, с которой сравниваемся.
            # Тогда эмбеддингам достаётся ОСТАТОК, а не всё сразу, и вопрос
            # ставится честно: есть ли в парах что-то сверх одиночной силы.
            lr0 = fit_baseline(tr)
            self.a = lr0.coef_[0][:H].astype(float).copy()
            self.w = float(lr0.coef_[0][H])
            self.b = float(lr0.intercept_[0])
        self.u = rng.normal(0, 0.01, (H, k))
        self.v = rng.normal(0, 0.01, (H, k))
        self.A = rng.normal(0, 0.01, (k, k))

        names = ["a", "b", "w", "u", "v", "A"]
        lrs = {"a": self.lr_lin, "b": self.lr_lin, "w": self.lr_lin,
               "u": self.lr, "v": self.lr, "A": self.lr}
        m = {n: np.zeros_like(getattr(self, n), dtype=float) for n in names}
        vv = {n: np.zeros_like(getattr(self, n), dtype=float) for n in names}
        b1, b2, eps = 0.9, 0.999, 1e-8

        wt = np.ones(len(tr)) if w_tr is None else np.asarray(w_tr, dtype=float)
        wt = wt / wt.mean()
        best, best_ep, best_state = np.inf, -1, None
        for ep in range(1, self.epochs + 1):
            g_all = self._grads(tr, wt)
            for n in names:
                gr = g_all[n]
                m[n] = b1 * m[n] + (1 - b1) * gr
                vv[n] = b2 * vv[n] + (1 - b2) * gr * gr
                mh = m[n] / (1 - b1 ** ep)
                vh = vv[n] / (1 - b2 ** ep)
                setattr(self, n, getattr(self, n) - lrs[n] * mh / (np.sqrt(vh) + eps))
            if ep % 10 == 0 or ep == 1:
                pv = np.clip(self.predict_proba(va), EPS, 1 - EPS)
                ll = float(-np.mean(va.y * np.log(pv) + (1 - va.y) * np.log(1 - pv)))
                self.hist.append((ep, ll))
                if ll < best - 1e-6:
                    best, best_ep = ll, ep
                    best_state = (self.a.copy(), self.b, self.w,
                                  self.u.copy(), self.v.copy(), self.A.copy())
                elif ep - best_ep >= self.patience:
                    break
        if best_state is not None:
            self.a, self.b, self.w, self.u, self.v, self.A = best_state
        self.best_val, self.best_epoch = best, best_ep
        return self

    def _grads(self, d: Draft, wt: np.ndarray) -> dict:
        R, D, elo, y = d.R, d.D, d.elo, d.y
        n = len(y)
        uR, uD, sR, sD, _vR, _vD, pR, pD = self._parts(R, D)
        z = self.b + self.w * elo + self.a[R].sum(1) - self.a[D].sum(1)
        if self.use_syn:
            z = z + 0.5 * ((sR ** 2).sum(1) - (uR ** 2).sum((1, 2))) \
                  - 0.5 * ((sD ** 2).sum(1) - (uD ** 2).sum((1, 2)))
        M = self.A - self.A.T
        if self.use_cnt:
            z = z + ((pR @ M) * pD).sum(1)
        g = wt * (_sig(z) - y) / n                                  # (n,)

        da = np.zeros_like(self.a)
        np.add.at(da, R, g[:, None])
        np.add.at(da, D, -g[:, None])
        du = np.zeros_like(self.u)
        dv = np.zeros_like(self.v)
        dA = np.zeros_like(self.A)
        if self.use_syn:
            # d(synR)/du_i = sR − u_i для i из R; для D то же со знаком минус.
            np.add.at(du, R, g[:, None, None] * (sR[:, None, :] - uR))
            np.add.at(du, D, -g[:, None, None] * (sD[:, None, :] - uD))
        if self.use_cnt:
            gR = pD @ M.T                                           # d/dpR
            gD = pR @ M                                             # d/dpD
            np.add.at(dv, R, g[:, None, None] * gR[:, None, :])
            np.add.at(dv, D, g[:, None, None] * gD[:, None, :])
            G = pR.T @ (g[:, None] * pD)
            dA = G - G.T          # потому что M = A − Aᵀ
        # Штраф НЕ делится на n: потери усреднены по матчам, значит и
        # регуляризация должна быть в том же масштабе. На первом заходе
        # деление было, эффективный штраф падал в тысячи раз, и эмбеддинги
        # запоминали обучающую выборку за десяток эпох.
        return {
            "a": da + self.l2["a"] * self.a,
            "b": float(g.sum()),
            "w": float((g * elo).sum()),
            "u": du + self.l2["u"] * self.u,
            "v": dv + self.l2["v"] * self.v,
            "A": dA + self.l2["m"] * self.A,
        }

    # --- разбор ----------------------------------------------------------

    def pair_synergy(self, i: int, j: int) -> float:
        return float(self.u[i] @ self.u[j])

    def pair_counter(self, i: int, j: int) -> float:
        M = self.A - self.A.T
        return float(self.v[i] @ M @ self.v[j])


# Сетка штрафа для эмбеддингов. Подбирается на ВНУТРЕННЕЙ проверке внутри
# обучающего куска — то есть на прошлом, а не на том, что модель будет
# предсказывать. Базовой модели такой подбор не нужен: её C уже подобран
# в проекте раньше и зафиксирован в config.DRAFT_C.
L2_GRID = (0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def fit_best(tr: Draft, va: Draft, n_heroes: int, dim: int,
             w_tr=None, use_syn=True, use_cnt=True, seed: int = 0
             ) -> tuple["EmbDraft", float]:
    """Обучить с подбором штрафа по внутренней проверке."""
    best, best_l2 = None, None
    for l2 in L2_GRID:
        m = EmbDraft(n_heroes, dim=dim, l2_emb=l2, use_syn=use_syn,
                     use_cnt=use_cnt, seed=seed).fit(tr, va, w_tr)
        if best is None or m.best_val < best.best_val:
            best, best_l2 = m, l2
    return best, best_l2


# --- база ----------------------------------------------------------------

def fit_baseline(tr: Draft, C: float = None,
                 sample_weight: np.ndarray | None = None) -> LogisticRegression:
    """Ровно то, что делает train.py: одна логистическая на ±1 и elo."""
    lr = LogisticRegression(C=config.DRAFT_C if C is None else C,
                            max_iter=3000, solver="lbfgs")
    lr.fit(tr.X, tr.y, sample_weight=sample_weight)
    return lr


def walk_recency(d: Draft, steps: int, halflives, verbose: bool = True) -> dict:
    """Веса по свежести на ОДНОЙ И ТОЙ ЖЕ базовой модели.

    Отдельно от эмбеддингов намеренно: если менять сразу и модель, и веса,
    то по итогу не узнать, что из двух сработало. Здесь модель одна, а
    меняется только вес матча — экспоненциально падающий с возрастом.

    Это пункт 1 из «Что доработать»: кривая обучения показала, что объём
    выборки для стейт-модели исчерпан, и свежесть — единственное, что ещё
    может дать добор данных. Проверяем на драфт-модели, потому что здесь
    полный протокол стоит полторы минуты, а не час обучения.
    """
    order = np.argsort(d.t, kind="stable")
    d = d.take(order)
    n = len(d)
    edges = [int(round(n * (i + 1) / (steps + 1))) for i in range(steps + 1)]
    pools: dict = {h: [] for h in ([None] + list(halflives))}
    ys = []
    for s in range(steps):
        lo, hi = edges[s], edges[s + 1]
        tr, te = d.take(slice(0, lo)), d.take(slice(lo, hi))
        if len(te) < 100:
            continue
        ys.append(te.y)
        line = []
        for h in pools:
            wt = None if h is None else recency_weights(tr.t, float(tr.t[-1]), h)
            p = fit_baseline(tr, sample_weight=wt).predict_proba(te.X)[:, 1]
            pools[h].append(p)
            line.append(f"{'без весов' if h is None else str(h) + 'д'}"
                        f" {roc_auc_score(te.y, p):.4f}")
        if verbose:
            print(f"  шаг {s + 1}: AUC  " + "   ".join(line))
    y = np.concatenate(ys)
    return {"y": y, "p": {h: np.concatenate(v) for h, v in pools.items()}}


# --- протокол ------------------------------------------------------------

def recency_weights(t: np.ndarray, t_ref: float, halflife_days: float
                    ) -> np.ndarray:
    """Вес матча падает вдвое каждые `halflife_days` до края обучения."""
    age = (float(t_ref) - t.astype(float)) / 86400.0
    return 0.5 ** (np.maximum(0.0, age) / halflife_days)


def _metrics(y, p) -> dict:
    p = np.clip(p, EPS, 1 - EPS)
    return {"ll": float(log_loss(y, p, labels=[0, 1])),
            "auc": float(roc_auc_score(y, p)) if len(set(y.tolist())) > 1 else float("nan"),
            "acc": float(np.mean((p >= 0.5) == (y == 1))), "n": int(len(y))}


def walk_forward(d: Draft, steps: int, dim: int, halflife: float | None,
                 use_syn: bool = True, use_cnt: bool = True,
                 seed: int = 0, verbose: bool = True) -> dict:
    """Учимся на прошлом, проверяем на будущем. Никаких случайных разбиений.

    Внутренняя проверка для ранней остановки берётся из ХВОСТА обучающего
    куска, а не случайно: подбирать число шагов по случайной подвыборке
    прошлого значило бы подбирать его по данным, которых в бою не будет.
    """
    order = np.argsort(d.t, kind="stable")
    d = d.take(order)
    n = len(d)
    edges = [int(round(n * (i + 1) / (steps + 1))) for i in range(steps + 1)]
    rows = []
    pool_y, pool_pb, pool_pe, pool_mid = [], [], [], []
    for s in range(steps):
        lo, hi = edges[s], edges[s + 1]
        tr_all = d.take(slice(0, lo))
        te = d.take(slice(lo, hi))
        cut = int(len(tr_all) * 0.85)
        tr, va = tr_all.take(slice(0, cut)), tr_all.take(slice(cut, len(tr_all)))
        if len(va) < 100 or len(te) < 100:
            continue
        wt = (recency_weights(tr.t, float(tr_all.t[-1]), halflife)
              if halflife else None)
        base = fit_baseline(tr_all)
        pb = base.predict_proba(te.X)[:, 1]
        emb, l2 = fit_best(tr, va, len(d.heroes), dim, wt, use_syn, use_cnt, seed)
        pe = emb.predict_proba(te)
        mb, me = _metrics(te.y, pb), _metrics(te.y, pe)
        rows.append({"step": s + 1, "n_train": len(tr_all), "n_test": len(te),
                     "base": mb, "emb": me, "epoch": emb.best_epoch,
                     "l2": l2, "emb_norm": float(np.abs(emb.u).mean())})
        pool_y.append(te.y)
        pool_pb.append(pb)
        pool_pe.append(pe)
        pool_mid.append(te.mid)
        if verbose:
            print(f"  шаг {s + 1}: учим на {len(tr_all):>5}, "
                  f"проверяем на {len(te):>4}   "
                  f"AUC база {mb['auc']:.4f} -> эмб {me['auc']:.4f}   "
                  f"log loss {mb['ll']:.4f} -> {me['ll']:.4f}   "
                  f"(штраф {l2:g}, эпоха {emb.best_epoch}, "
                  f"|u| {np.abs(emb.u).mean():.4f})")
    if not rows:
        raise RuntimeError("шагов не получилось: слишком мало матчей")
    y = np.concatenate(pool_y)
    pb = np.concatenate(pool_pb)
    pe = np.concatenate(pool_pe)
    return {"steps": rows, "y": y, "p_base": pb, "p_emb": pe,
            "mid": np.concatenate(pool_mid)}


def paired_bootstrap(y, pa, pb, n_boot: int = 2000, seed: int = 0) -> dict:
    """Разность метрик с интервалом. Одна строка = один матч, группировать
    нечего: драфт-модель даёт по одному предсказанию на матч."""
    rng = np.random.default_rng(seed)
    n = len(y)
    dll = np.empty(n_boot)
    dauc = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, n, n)
        if len(set(y[s].tolist())) < 2:
            dll[i] = dauc[i] = np.nan
            continue
        dll[i] = log_loss(y[s], np.clip(pb[s], EPS, 1 - EPS), labels=[0, 1]) \
            - log_loss(y[s], np.clip(pa[s], EPS, 1 - EPS), labels=[0, 1])
        dauc[i] = roc_auc_score(y[s], pb[s]) - roc_auc_score(y[s], pa[s])
    out = {}
    for name, arr, base in (("log_loss", dll,
                             log_loss(y, np.clip(pb, EPS, 1 - EPS), labels=[0, 1])
                             - log_loss(y, np.clip(pa, EPS, 1 - EPS), labels=[0, 1])),
                            ("auc", dauc,
                             roc_auc_score(y, pb) - roc_auc_score(y, pa))):
        a = arr[~np.isnan(arr)]
        lo, hi = np.percentile(a, [2.5, 97.5])
        out[name] = {"delta": float(base), "lo": float(lo), "hi": float(hi),
                     "p_better": float(np.mean(a < 0) if name == "log_loss"
                                       else np.mean(a > 0))}
    return out


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Драфт по эмбеддингам против 127 коэффициентов.")
    ap.add_argument("--walk", type=int, default=5, help="шагов протокола")
    ap.add_argument("--dim", type=int, default=8, help="размерность эмбеддинга")
    ap.add_argument("--halflife", type=float, default=None,
                    help="полураспад веса матча в ДНЯХ (веса по свежести)")
    ap.add_argument("--ablate", action="store_true",
                    help="что даёт каждое слагаемое по отдельности")
    ap.add_argument("--recency", action="store_true",
                    help="замерить ТОЛЬКО веса по свежести, на базовой модели")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)

    t0 = time.time()
    d = load_draft()
    print("=" * 78)
    print(f"Драфтов (без слепого холдаута): {len(d)}   героев: {len(d.heroes)}")
    print(f"Период: {pd.to_datetime(d.t.min(), unit='s').date()} .. "
          f"{pd.to_datetime(d.t.max(), unit='s').date()}")
    print(f"Протокол: шагающий хронологический, {args.walk} шагов; "
          f"эмбеддинг {args.dim}")
    if args.halflife:
        print(f"Веса по свежести: полураспад {args.halflife:g} дней")
    print("=" * 78)

    if args.recency:
        hl = (14.0, 30.0, 60.0, 120.0)
        print("\n--- Веса по свежести: та же модель, разный полураспад ---")
        r = walk_recency(d, args.walk, hl)
        y = r["y"]
        p0 = r["p"][None]
        print(f"\n  {'полураспад':<14}{'log_loss':>10}{'auc':>9}"
              f"{'Δ auc к «без весов»':>22}{'95% интервал':>24}")
        print("  " + "-" * 78)
        for h in [None] + list(hl):
            p = r["p"][h]
            m = _metrics(y, p)
            lab = "без весов" if h is None else f"{h:g} дней"
            if h is None:
                print(f"  {lab:<14}{m['ll']:>10.4f}{m['auc']:>9.4f}"
                      f"{'—':>22}{'':>24}")
                continue
            bs = paired_bootstrap(y, p0, p, n_boot=args.n_boot, seed=args.seed)
            c = bs["auc"]
            print(f"  {lab:<14}{m['ll']:>10.4f}{m['auc']:>9.4f}"
                  f"{c['delta']:>+22.4f}   [{c['lo']:+.4f}, {c['hi']:+.4f}]")
        print("\n  Плюс в последней колонке значит «свежесть помогает».\n"
              "  Значимо — только если весь интервал по одну сторону нуля.")
        rec = {"built_at": time.time(), "walk": args.walk,
               "n_matches": len(d), "halflives": list(hl),
               "rows": {("none" if h is None else f"{h:g}"):
                        dict(_metrics(y, r["p"][h]),
                             **({} if h is None else
                                {"vs_none": paired_bootstrap(
                                    y, p0, r["p"][h], n_boot=args.n_boot,
                                    seed=args.seed)}))
                        for h in [None] + list(hl)}}
        out_rec = args.out.with_name("draft_recency.json")
        out_rec.parent.mkdir(parents=True, exist_ok=True)
        out_rec.write_text(json.dumps(rec, ensure_ascii=False, indent=1,
                                      default=float), encoding="utf-8")
        print(f"\n  записано: {out_rec}")
        return 0

    runs = [("эмбеддинги (синергия + контра)", True, True)]
    if args.ablate:
        runs += [("только синергия", True, False),
                 ("только контра", False, True),
                 ("ни того ни другого (проверка стенда)", False, False)]

    results = {}
    for title, syn, cnt in runs:
        print(f"\n--- {title} ---")
        r = walk_forward(d, args.walk, args.dim, args.halflife,
                         use_syn=syn, use_cnt=cnt, seed=args.seed)
        mb = _metrics(r["y"], r["p_base"])
        me = _metrics(r["y"], r["p_emb"])
        bs = paired_bootstrap(r["y"], r["p_base"], r["p_emb"],
                              n_boot=args.n_boot, seed=args.seed)
        print(f"\n  {'':<10}{'log_loss':>10}{'auc':>9}{'acc':>8}{'n':>8}")
        print(f"  {'база':<10}{mb['ll']:>10.4f}{mb['auc']:>9.4f}"
              f"{mb['acc']:>8.4f}{mb['n']:>8}")
        print(f"  {'эмбеддинг':<10}{me['ll']:>10.4f}{me['auc']:>9.4f}"
              f"{me['acc']:>8.4f}{me['n']:>8}")
        for k in ("log_loss", "auc"):
            c = bs[k]
            print(f"  Δ {k:<8}{c['delta']:>+10.4f}   "
                  f"95% [{c['lo']:+.4f}, {c['hi']:+.4f}]   "
                  f"P(лучше) {c['p_better']:.3f}")
        good = bs["auc"]["lo"] > 0 and bs["log_loss"]["hi"] < 0
        some = bs["auc"]["lo"] > 0 or bs["log_loss"]["hi"] < 0
        print("  ВЫВОД: " + (
            "лучше значимо и по AUC, и по log loss." if good else
            "лучше значимо по одной метрике из двух — этого мало." if some else
            "разница НЕ значима: интервал накрывает ноль."))
        results[title] = {"base": mb, "emb": me, "boot": bs,
                          "steps": [{k: v for k, v in s.items()}
                                    for s in r["steps"]]}

    out = {"built_at": time.time(), "dim": args.dim, "walk": args.walk,
           "halflife_days": args.halflife, "n_matches": len(d),
           "results": results}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1,
                                   default=float), encoding="utf-8")
    print(f"\n  записано: {args.out}   заняло {time.time() - t0:.0f} с")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
