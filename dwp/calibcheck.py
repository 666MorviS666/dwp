"""Условная калибровка: честно ли число на экране ВНУТРИ среза.

Глобальный ECE — среднее по всем строкам. Он может быть маленьким, когда
на одних минутах модель систематически завышает, а на других занижает:
смещения гасят друг друга в общем числе, но человек смотрит одну минуту,
а не среднее по матчу.

Считается то же, что latecheck, с двумя отличиями: отрезки по десять
минут, а не четыре широких, и много разбиений вместо одного. Второе
существеннее: на 50+ минутах в одном тесте десятки матчей, и число
оттуда без разброса читать нельзя.

Читает выгрузки, а не сырые JSON — как bench.py.

Запуск:
    python -m dwp.calibcheck --frames data\\export_frames.csv.gz
                             --draft  data\\export_draft.csv.gz
"""

from __future__ import annotations

import argparse
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from . import config

BANDS = [(0, 10), (10, 20), (20, 30), (30, 40), (40, 50), (50, 10 ** 9)]
EPS = 1e-3


def _label(lo: int, hi: int) -> str:
    return f"{lo}-{hi - 1}" if hi < 10 ** 9 else "50+"


def ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    """Средневзвешенное |предсказано - фактически| по децилям — как в train.ece."""
    if len(y) == 0:
        return float("nan")
    idx = np.clip(np.digitize(p, np.linspace(0, 1, n_bins + 1)[1:-1]), 0, n_bins - 1)
    tot = 0.0
    for b in range(n_bins):
        m = idx == b
        if m.any():
            tot += m.sum() * abs(p[m].mean() - y[m].mean())
    return tot / len(y)


def one_split(fr: pd.DataFrame, dr: pd.DataFrame, seed: int) -> dict:
    mids = dr["match_id"].to_numpy()
    y_m = dr["y"].to_numpy()
    hc = sorted([c for c in dr.columns if c.startswith("h") and c[1:].isdigit()],
                key=lambda c: int(c[1:]))
    # elo_div400 в выгрузке уже умножен на DRAFT_ELO_SCALE (см. draft_matrix).
    Xd = dr[hc + ["elo_div400"]].to_numpy(dtype=np.float64)

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(mids))
    n_test = int(len(mids) * config.TEST_FRACTION)
    test_i, pool_i = perm[:n_test], perm[n_test:]
    n_cal = int(len(pool_i) * config.CALIB_FRACTION)
    cal_i, fit_i = pool_i[:n_cal], pool_i[n_cal:]

    lr = LogisticRegression(C=config.DRAFT_C, max_iter=5000)
    p_pool = cross_val_predict(lr, Xd[pool_i], y_m[pool_i], cv=config.DRAFT_CALIB_FOLDS,
                               method="predict_proba")[:, 1]
    lr.fit(Xd[pool_i], y_m[pool_i])
    p_test = lr.predict_proba(Xd[test_i])[:, 1]
    dl = dict(zip(mids[pool_i], np.log(p_pool / (1 - p_pool))))
    dl.update(dict(zip(mids[test_i], np.log(p_test / (1 - p_test)))))

    feats = [c for c in fr.columns if c not in ("match_id", "y")]

    def block(ii):
        d = fr[fr["match_id"].isin(set(mids[ii].tolist()))]
        X = d[feats].copy()
        X["draft_logit"] = d["match_id"].map(dl)
        return d, X

    d_fit, X_fit = block(fit_i)
    d_cal, X_cal = block(cal_i)
    d_te, X_te = block(test_i)

    params = dict(config.LGBM_PARAMS, seed=seed, bagging_seed=seed,
                  feature_fraction_seed=seed)
    booster = lgb.train(params, lgb.Dataset(X_fit, d_fit["y"]),
                        num_boost_round=config.LGBM_ROUNDS)
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        booster.predict(X_cal), d_cal["y"])
    p = np.clip(iso.predict(booster.predict(X_te)), EPS, 1 - EPS)

    y = d_te["y"].to_numpy()
    mn = d_te["minute"].to_numpy()
    gid = d_te["match_id"].to_numpy()
    out = {"seed": seed, "ECE": ece(y, p)}
    for lo, hi in BANDS:
        m = (mn >= lo) & (mn < hi)
        lab = _label(lo, hi)
        out[f"ECE {lab}"] = ece(y[m], p[m])
        out[f"смещ {lab}"] = float(p[m].mean() - y[m].mean()) if m.any() else np.nan
        out[f"матчей {lab}"] = int(len(np.unique(gid[m])))
    return out


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Условная калибровка по отрезкам минут.")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--seeds", default="1,3,5,7,11")
    args = ap.parse_args(argv)

    fr = pd.read_csv(args.frames)
    dr = pd.read_csv(args.draft)
    seeds = [int(s) for s in args.seeds.split(",")]
    rows = []
    for s in seeds:
        rows.append(one_split(fr, dr, s))
        print(f"  разбиение {s} готово", flush=True)
    df = pd.DataFrame(rows)

    labs = [_label(lo, hi) for lo, hi in BANDS]
    print("\nECE внутри отрезка (глобальный ECE — крайний левый столбец):")
    print(df[["seed", "ECE"] + [f"ECE {l}" for l in labs]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nсмещение (предсказано минус фактически) внутри отрезка:")
    print(df[["seed"] + [f"смещ {l}" for l in labs]].to_string(
        index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nсводка по разбиениям:")
    print(f"  {'срез':<8}{'ECE':>9}{'ст.откл':>10}{'смещение':>11}"
          f"{'ст.откл':>10}{'матчей':>9}")
    print(f"  {'весь':<8}{df['ECE'].mean():>9.4f}{df['ECE'].std(ddof=1):>10.4f}"
          f"{'':>11}{'':>10}{'':>9}")
    for l in labs:
        print(f"  {l:<8}{df[f'ECE {l}'].mean():>9.4f}{df[f'ECE {l}'].std(ddof=1):>10.4f}"
              f"{df[f'смещ {l}'].mean():>+11.4f}{df[f'смещ {l}'].std(ddof=1):>10.4f}"
              f"{df[f'матчей {l}'].mean():>9.0f}")
    print("\nКак читать: если ECE в срезе заметно выше глобального — число на\n"
          "экране на этих минутах честно не настолько, насколько обещает общий\n"
          "ECE. Если при этом смещение меняет знак между разбиениями и его\n"
          "ст.откл больше среднего — это разброс, а не сдвиг, и починить\n"
          "перекалибровкой нельзя: поправка, выученная на одном калибровочном\n"
          "наборе, на следующем окажется с обратным знаком.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
