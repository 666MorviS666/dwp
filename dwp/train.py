"""Обучение двух моделей и честная оценка качества.

Схема разбиения (всё — по match_id, не по строкам):

    все матчи
      ├── test   20%          только финальная оценка
      └── train_pool 80%
            ├── fit   60%     деревья стейт-модели
            ├── val   15%     ранняя остановка
            └── calib 25%     изотоническая калибровка

Драфт-модель учится на всём train_pool, но `draft_logit` для строк
train_pool берётся out-of-fold (cross_val_predict). Иначе prior на своих
же матчах переуверен, стейт-модель научится его недооценивать, а в лайве
получит нормальный prior и начнёт врать.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, cross_val_predict

from . import config, features as F, holdout as HO

EPS = 1e-6


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def fit_calibrator(method: str, p_raw: np.ndarray, y: np.ndarray):
    """Возвращает объект калибратора. 'none' — тождественное отображение."""
    if method == "none":
        return None
    if method == "isotonic":
        m = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        m.fit(p_raw, y)
        return m
    if method == "sigmoid":
        # Платт: логистическая регрессия по логиту сырого выхода. Два
        # параметра вместо ступенчатой функции — на нескольких сотнях
        # матчей это устойчивее изотоники.
        lr = LogisticRegression(C=1e6, max_iter=1000)
        lr.fit(logit(p_raw).reshape(-1, 1), y)
        return lr
    raise ValueError(f"неизвестный калибратор {method!r}")


def apply_calibrator(cal, p_raw: np.ndarray) -> np.ndarray:
    if cal is None:
        return np.asarray(p_raw, dtype=np.float64)
    if isinstance(cal, IsotonicRegression):
        return cal.predict(p_raw)
    return cal.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1]


def split_ids(ids: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """GroupShuffleSplit по match_id. Построчный сплит здесь — главная
    ловушка задачи: минуты одного матча делят лейбл и почти дублируют
    друг друга, тест протекает в трейн и accuracy завышается."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dummy = np.zeros(len(ids))
    a, b = next(gss.split(dummy, groups=ids))
    return ids[a], ids[b]


# --- Метрики ------------------------------------------------------------

def core_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    out = {
        "n": int(len(y)),
        "log_loss": float(log_loss(y, np.clip(p, EPS, 1 - EPS), labels=[0, 1])),
        "brier": float(brier_score_loss(y, p)),
        "acc": float(((p >= 0.5).astype(int) == y).mean()),
    }
    out["auc"] = float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan")
    return out


def baseline_metrics(y_train: np.ndarray, y: np.ndarray) -> dict:
    base = float(np.mean(y_train))
    p = np.full(len(y), base)
    m = core_metrics(y, p)
    m["const_p"] = base
    return m


def reliability_table(y: np.ndarray, p: np.ndarray, groups: np.ndarray,
                      n_bins: int = 10) -> pd.DataFrame:
    """Предсказано против фактического по децилям предсказания.

    Для оверлея это главная таблица: если модель говорит 70%, Radiant
    должен выигрывать в 70% таких случаев. n_matches печатается рядом с
    n_rows, потому что строки внутри матча скоррелированы и эффективный
    размер выборки ближе к числу матчей.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        rows.append({
            "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
            "n_rows": int(m.sum()),
            "n_matches": int(len(np.unique(groups[m]))),
            "pred_mean": float(p[m].mean()),
            "actual": float(y[m].mean()),
            "gap": float(p[m].mean() - y[m].mean()),
        })
    return pd.DataFrame(rows)


def by_minute_bucket(df: pd.DataFrame, p: np.ndarray, y: np.ndarray,
                     base_rate: float, width: int = 10) -> pd.DataFrame:
    buckets = (df["minute"].to_numpy() // width) * width
    rows = []
    for b in sorted(np.unique(buckets)):
        m = buckets == b
        if m.sum() < 20:
            continue
        yy, pp = y[m], p[m]
        base = np.full(m.sum(), base_rate)
        rows.append({
            "minutes": f"{int(b)}-{int(b) + width - 1}",
            "n_rows": int(m.sum()),
            "n_matches": int(df.loc[m, "match_id"].nunique()),
            "log_loss": float(log_loss(yy, np.clip(pp, EPS, 1 - EPS), labels=[0, 1])),
            "brier": float(brier_score_loss(yy, pp)),
            "acc": float(((pp >= 0.5).astype(int) == yy).mean()),
            "ll_base": float(log_loss(yy, base, labels=[0, 1])),
        })
    return pd.DataFrame(rows)


def ece(y: np.ndarray, p: np.ndarray, groups: np.ndarray, n_bins: int = 10) -> float:
    """Средневзвешенное по числу строк |предсказано - фактически| по децилям.

    Устойчивее максимального смещения: последний зависит от одного,
    возможно пустого, дециля. Для оверлея это и есть «насколько честно
    число на экране» одним числом.
    """
    rel = reliability_table(y, p, groups, n_bins)
    if not len(rel):
        return float("nan")
    w = rel["n_rows"].to_numpy(dtype=float)
    return float((rel["gap"].abs().to_numpy() * w).sum() / w.sum())


def _fmt(df: pd.DataFrame) -> str:
    return df.to_string(index=False, float_format=lambda v: f"{v:.4f}")


# --- Основной сценарий --------------------------------------------------

def run(source_dir: Path, use_xp: bool, limit: int | None, seed: int,
        out_path: Path, draft_calib: str = "isotonic",
        state_calib: str = "auto", calib_criterion: str = "logloss",
        live_features: bool = False, extra_features: bool = False,
        gold_norm: str = "off", exact_features: bool = False,
        use_holdout: bool = True, pool_frac: float = 1.0,
        quiet: bool = False, return_artefact: bool = False,
        lgbm_params: dict | None = None, rating: str = "player"):
    t0 = time.time()
    if quiet:
        # Кривая обучения гоняет run() пятнадцать раз; печатать пятнадцать
        # полных отчётов значит спрятать в них итоговую таблицу.
        import io
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            return run(source_dir, use_xp, limit, seed, out_path, draft_calib,
                       state_calib, calib_criterion, live_features,
                       extra_features, gold_norm, exact_features, use_holdout,
                       pool_frac, quiet=False, return_artefact=return_artefact,
                       lgbm_params=lgbm_params, rating=rating)
    matches = F.load_matches(source_dir, limit=limit)
    if not matches:
        print(f"ОШИБКА: в {source_dir} нет матчей. Запустите `python -m dwp.synthetic` "
              f"или `python -m dwp.collect`.", file=sys.stderr)
        return 2
    matches = F.usable_matches(matches)
    # СЛЕПОЙ ХОЛДАУТ. Обычная тестовая выборка честна ровно один раз, а
    # дальше расходуется глазами разработчика: каждое решение «оставить или
    # откатить» принимается по ней. Поэтому часть матчей не показывается
    # обучению вообще никогда, и попадание в неё зависит только от
    # match_id — добор данных разбиение не двигает (см. dwp.holdout).
    n_before = len(matches)
    if use_holdout:
        matches, hidden = HO.split(matches)
        if hidden:
            print(f"[train] в слепом холдауте оставлено {len(hidden)} матчей из "
                  f"{n_before} — обучение их не увидит.\n"
                  f"        Оценить на них: python -m dwp.blindtest --seal "
                  f"--holdout && python -m dwp.blindtest --reveal")
    if len(matches) < 100:
        print(f"ОШИБКА: пригодных матчей всего {len(matches)}, этого мало для обучения.",
              file=sys.stderr)
        return 2

    hero_ids, id2idx, hero_names = F.load_heroes()

    # ДОЛЯ ОБУЧАЮЩЕГО ПУЛА. Урезается только пул, тест остаётся целиком —
    # иначе кривая обучения мерила бы заодно и то, что тест стал меньше, а
    # это другая величина. Отброшенные матчи выбрасываются и из истории Elo
    # тоже: Elo — признак, и оставить в нём исходы матчей, которых обучение
    # «не видело», значило бы протечку, которая занижает эффект объёма.
    all_ids = np.array([int(m["match_id"]) for m in matches], dtype=np.int64)
    pool_ids, test_ids = split_ids(all_ids, config.TEST_FRACTION, seed)
    n_pool_full = len(pool_ids)
    if pool_frac < 1.0:
        if not 0.0 < pool_frac <= 1.0:
            raise ValueError(f"pool_frac={pool_frac}, ожидалось (0, 1]")
        rng = np.random.default_rng(seed * 1_000_003 + 17)
        k = max(50, int(round(len(pool_ids) * pool_frac)))
        pool_ids = np.sort(rng.choice(pool_ids, size=min(k, len(pool_ids)),
                                      replace=False))
        used = set(pool_ids.tolist()) | set(test_ids.tolist())
        matches = [m for m in matches if int(m["match_id"]) in used]

    # Рейтинг: по игрокам или по командам. Умолчание — игроки, и это
    # результат замера, а не вкус: герои без рейтинга дают AUC 0.5465,
    # с командным Elo 0.6001, с рейтингом игроков 0.6138 (шагающий
    # протокол, 10636 матчей вне холдаута; −0.0049 log loss, 95%
    # [−0.0072, −0.0026]). Ключ оставлен, чтобы замер можно было
    # повторить, а не поверить в него.
    elo_team_pre, elo_final = F.build_elo(matches)
    player_pre, player_final = F.build_player_elo(matches)
    elo_pre = player_pre if rating == "player" else elo_team_pre
    n_acc = sum(1 for m in matches if F.match_accounts(m)[0])
    print(f"Рейтинг: {'по игрокам' if rating == 'player' else 'по командам'}"
          f" | составы разобраны у {n_acc} матчей из {len(matches)}"
          f" | игроков в таблице: {len(player_final)}")
    Xd, yd, mids = F.draft_matrix(matches, id2idx, elo_pre)
    pool_mask = np.isin(mids, pool_ids)
    test_mask = np.isin(mids, test_ids)

    print("=" * 74)
    print(f"Источник: {source_dir}")
    if pool_frac < 1.0:
        print(f"Доля обучающего пула: {pool_frac:.2f} — оставлено "
              f"{len(pool_ids)} матчей из {n_pool_full}; тест не урезан")
    print(f"Матчей пригодных: {len(matches)} | train_pool: {pool_mask.sum()} | "
          f"test: {test_mask.sum()} | признаки по опыту: {'да' if use_xp else 'нет (--no-xp)'}")

    # --- 1. Драфт-модель -------------------------------------------------
    def _draft_estimator(method: str):
        base = LogisticRegression(C=config.DRAFT_C, max_iter=3000, solver="lbfgs")
        if method == "none":
            return base
        return CalibratedClassifierCV(base, method=method, cv=config.DRAFT_CALIB_FOLDS)

    # Выбор калибратора сравнивается на out-of-fold внутри train_pool —
    # тест при этом не трогается. Изотоника на нескольких сотнях точек
    # фолда даёт грубые ступени и может проиграть базовой линии по
    # log loss, при этом AUC не падает. Поэтому сравнение печатается
    # всегда, а не прячется за умолчанием.
    oof_cache: dict[str, np.ndarray] = {}
    print("\n--- Калибровка драфт-модели: сравнение на out-of-fold (train_pool) ---")
    print(f"  {'метод':<12}{'log_loss':>10}{'brier':>9}{'auc':>8}")
    for method in ("none", "sigmoid", "isotonic"):
        pr = cross_val_predict(_draft_estimator(method), Xd[pool_mask], yd[pool_mask],
                               cv=config.DRAFT_CALIB_FOLDS, method="predict_proba")[:, 1]
        oof_cache[method] = pr
        mm = core_metrics(yd[pool_mask], pr)
        print(f"  {method:<12}{mm['log_loss']:>10.4f}{mm['brier']:>9.4f}{mm['auc']:>8.4f}")
    oof_base = baseline_metrics(yd[pool_mask], yd[pool_mask])
    print(f"  {'база (конст)':<12}{oof_base['log_loss']:>10.4f}{oof_base['brier']:>9.4f}"
          f"{'-':>8}")

    if draft_calib == "auto":
        chosen = min(oof_cache, key=lambda k: log_loss(yd[pool_mask], oof_cache[k]))
        print(f"  выбран автоматически: {chosen}")
    else:
        chosen = draft_calib
        print(f"  выбран флагом --draft-calib: {chosen}")

    draft_model = _draft_estimator(chosen)
    draft_model.fit(Xd[pool_mask], yd[pool_mask])
    oof = oof_cache[chosen]
    p_draft = np.empty(len(mids))
    p_draft[pool_mask] = oof
    p_draft[test_mask] = draft_model.predict_proba(Xd[test_mask])[:, 1]

    dm_test = core_metrics(yd[test_mask], p_draft[test_mask])
    dm_base = baseline_metrics(yd[pool_mask], yd[test_mask])
    print("\n--- Драфт-модель (одна строка = один матч, до старта) ---")
    print(f"  test  : n={dm_test['n']}  log_loss={dm_test['log_loss']:.4f}  "
          f"brier={dm_test['brier']:.4f}  acc={dm_test['acc']:.4f}  auc={dm_test['auc']:.4f}")
    print(f"  базовая (константа {dm_base['const_p']:.3f}): "
          f"log_loss={dm_base['log_loss']:.4f}  brier={dm_base['brier']:.4f}  "
          f"acc={dm_base['acc']:.4f}")

    # Коэффициенты для интерпретации. CalibratedClassifierCV хранит по
    # одной модели на фолд — усредняем, чтобы получить один вектор вкладов.
    if hasattr(draft_model, "calibrated_classifiers_"):
        coefs = np.mean([cc.estimator.coef_[0]
                         for cc in draft_model.calibrated_classifiers_], axis=0)
        intercept = float(np.mean([cc.estimator.intercept_[0]
                                   for cc in draft_model.calibrated_classifiers_]))
    else:
        coefs = draft_model.coef_[0]
        intercept = float(draft_model.intercept_[0])
    dnames = F.draft_feature_names(hero_ids, hero_names)

    truth_path = config.DATA_DIR / "synthetic_truth.json"
    truth_corr = None
    if truth_path.exists():
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        hs = truth.get("hero_strength", {})
        tv = np.array([hs.get(str(h), np.nan) for h in hero_ids])
        ok = ~np.isnan(tv)
        if ok.sum() > 10:
            truth_corr = float(np.corrcoef(coefs[:len(hero_ids)][ok], tv[ok])[0, 1])
            print(f"  корреляция выученных коэффициентов с истинной силой героев: "
                  f"{truth_corr:.3f}  (синтетика; на реальных данных проверить нечем)")

    order = np.argsort(coefs[:len(hero_ids)])
    print("  сильнейшие по коэффициенту: " +
          ", ".join(f"{dnames[i]} {coefs[i]:+.3f}" for i in order[::-1][:5]))
    print("  слабейшие:                  " +
          ", ".join(f"{dnames[i]} {coefs[i]:+.3f}" for i in order[:5]))
    # Подпись со множителем: без неё невозможно на глаз отличить прогон
    # с DRAFT_ELO_SCALE от прогона без него, а коэффициент делится на s.
    print(f"  коэффициент при elo_diff/400*{config.DRAFT_ELO_SCALE:g}: "
          f"{coefs[len(hero_ids)]:+.3f}")

    # --- 2. Поминутные признаки -----------------------------------------
    prior = dict(zip(mids.tolist(), p_draft.tolist()))
    frames, skipped = [], 0
    parse_flags = {"tower_missing": 0, "rax_missing": 0, "roshan_missing": 0}
    for m in matches:
        parsed = F.parse_objectives(m)
        if not parsed.tower_ok:
            parse_flags["tower_missing"] += 1
        if not parsed.rax_ok:
            parse_flags["rax_missing"] += 1
        if not parsed.roshan_kills:
            parse_flags["roshan_missing"] += 1
        try:
            fr = F.match_state_frame(m, parsed)
        except ValueError:
            skipped += 1
            continue
        fr["draft_logit"] = logit(np.array([prior[int(m["match_id"])]]))[0]
        frames.append(fr)
    df = pd.concat(frames, ignore_index=True)
    if skipped:
        print(f"\n[train] матчей без поминутных данных пропущено: {skipped}")
    print("\n--- Поминутные срезы ---")
    print(f"  строк: {len(df)} из {df['match_id'].nunique()} матчей")
    print(f"  матчей без разобранных вышек: {parse_flags['tower_missing']}, "
          f"бараков: {parse_flags['rax_missing']}, Рошанов: {parse_flags['roshan_missing']}"
          f"   (у таких матчей соответствующие признаки = NaN, не ноль)")

    feats = F.state_features(use_xp, live_only=live_features, extra=extra_features,
                             gold_norm=gold_norm, exact=exact_features)
    if gold_norm != "off":
        print(f"\n  режим --gold-norm {gold_norm}: перевес по золоту как доля "
              f"суммарного нетворса\n    добавлено: {', '.join(F.NORM_STATE_FEATURES)}"
              + (f"\n    убрано:    {', '.join(F.GOLD_NORM_REPLACES)}"
                 if gold_norm == "replace" else ""))
    if live_features:
        print(f"\n  режим --live-features: убраны признаки, которых нет в "
              f"GetRealtimeStats:\n    {', '.join(F.LIVE_UNAVAILABLE)}"
              f"\n  осталось {len(feats)} признаков. Сравните log loss с полной "
              f"моделью — это и есть цена неполноты лайв-источника.")
    # Требование 2: duration коррелирует с исходом и в лайве неизвестна.
    # Проверка стоит копейки, а цена ошибки — модель, отличная офлайн и
    # мёртвая в лайве, что по метрикам не видно.
    banned = {"duration", "radiant_win", "y", "match_id"}
    assert not (set(feats) & banned), f"запрещённые признаки: {set(feats) & banned}"
    pool_fit_ids, val_ids = split_ids(pool_ids, 0.15 / 0.80, seed + 1)
    fit_ids, calib_ids = split_ids(pool_fit_ids, config.CALIB_FRACTION, seed + 2)

    def sel(ids_):
        return df[df["match_id"].isin(set(ids_.tolist()))]

    d_fit, d_val, d_cal, d_test = sel(fit_ids), sel(val_ids), sel(calib_ids), sel(test_ids)
    assert set(d_fit["match_id"]) & set(d_test["match_id"]) == set(), "пересечение fit/test"
    assert set(d_cal["match_id"]) & set(d_test["match_id"]) == set(), "пересечение calib/test"

    ds_fit = lgb.Dataset(d_fit[feats], label=d_fit["y"], free_raw_data=False)
    ds_val = lgb.Dataset(d_val[feats], label=d_val["y"], reference=ds_fit, free_raw_data=False)
    # Гиперпараметры берутся из config, но могут быть переопределены. Зачем:
    # ансамбль из моделей с РАЗНЫМИ параметрами обычно разнообразнее, чем из
    # одинаковых с разными сидами, — но «обычно» здесь не аргумент, и проверять
    # это надо тем же bench_ensemble. Сид LightGBM привязан к сиду разбиения:
    # иначе две модели с одним --seed отличались бы бэггингом, и «разброс от
    # сида разбиения» мерился бы вместе с разбросом от бэггинга.
    params = dict(config.LGBM_PARAMS)
    params.update({k: v for k, v in (lgbm_params or {}).items() if v is not None})
    params.setdefault("seed", seed)
    params.setdefault("bagging_seed", seed + 101)
    params.setdefault("feature_fraction_seed", seed + 202)
    if lgbm_params:
        print("\n  гиперпараметры переопределены: "
              + ", ".join(f"{k}={v}" for k, v in sorted(lgbm_params.items())
                          if v is not None))
    booster = lgb.train(
        params, ds_fit, num_boost_round=config.LGBM_ROUNDS,
        valid_sets=[ds_val], valid_names=["val"],
        callbacks=[lgb.early_stopping(config.LGBM_EARLY_STOPPING, verbose=False),
                   lgb.log_evaluation(0)],
    )

    # Калибратор и уровень обрезки хвостов выбираются кросс-валидацией
    # ВНУТРИ калибровочной выборки, по фолдам, разбитым по match_id.
    #
    # Почему не одна отложенная половина, как было раньше. На реальных
    # про-данных при 1500 матчах калибровочная выборка — около 300
    # матчей; половина от неё это ~150, и выбор по ней оказался шумным:
    # изотоника выглядела лучшей (смещение 0.068), а на тесте дала 0.116
    # и сломала середину диапазона, починив хвост. Кросс-валидация
    # оценивает каждый метод на всех 300 матчах вместо 150.
    p_cal_raw = booster.predict(d_cal[feats], num_iteration=booster.best_iteration)
    y_cal = d_cal["y"].to_numpy()
    g_cal = d_cal["match_id"].to_numpy()
    n_cal_matches = int(d_cal["match_id"].nunique())

    eps_grid = [0.0, 1e-3, 3e-3, 1e-2, 2e-2, 5e-2, 8e-2, 0.12]
    n_folds = min(5, max(2, n_cal_matches // 40))
    cal_folds = list(GroupKFold(n_splits=n_folds).split(p_cal_raw, y_cal, groups=g_cal))

    probe_scores: list[tuple[str, float, float]] = []
    probe_gap: dict[tuple[str, float], float] = {}
    probe_ece: dict[tuple[str, float], float] = {}
    for method in ("none", "sigmoid", "isotonic"):
        oof_cal = np.empty(len(y_cal), dtype=np.float64)
        for tr, te in cal_folds:
            cal = fit_calibrator(method, p_cal_raw[tr], y_cal[tr])
            oof_cal[te] = apply_calibrator(cal, p_cal_raw[te])
        for e in eps_grid:
            q = np.clip(oof_cal, max(e, EPS), 1 - max(e, EPS))
            probe_scores.append((method, e,
                                 float(log_loss(y_cal, q, labels=[0, 1]))))
            rel = reliability_table(y_cal, q, g_cal)
            probe_gap[(method, e)] = (float(rel["gap"].abs().max())
                                      if len(rel) else float("nan"))
            probe_ece[(method, e)] = ece(y_cal, q, g_cal)
    def _key(t):
        return t[2] if calib_criterion == "logloss" else probe_ece[(t[0], t[1])]

    if state_calib == "auto":
        chosen_state, calib_eps, _ = min(probe_scores, key=_key)
    else:
        chosen_state = state_calib
        sub = [t for t in probe_scores if t[0] == chosen_state]
        _, calib_eps, _ = min(sub, key=_key)
    best_per_method = {m: min((t for t in probe_scores if t[0] == m), key=_key)
                       for m in ("none", "sigmoid", "isotonic")}

    iso = fit_calibrator(chosen_state, p_cal_raw, d_cal["y"].to_numpy())

    y_test = d_test["y"].to_numpy()
    p_test_raw = booster.predict(d_test[feats], num_iteration=booster.best_iteration)
    p_test = np.clip(apply_calibrator(iso, p_test_raw),
                     max(calib_eps, EPS), 1 - max(calib_eps, EPS))
    g_test = d_test["match_id"].to_numpy()

    base_rate = float(d_fit["y"].mean())
    m_raw = core_metrics(y_test, p_test_raw)
    m_cal = core_metrics(y_test, p_test)
    m_base = core_metrics(y_test, np.full(len(y_test), base_rate))
    m_draft_only = core_metrics(
        y_test, 1 / (1 + np.exp(-d_test["draft_logit"].to_numpy())))

    print("\n--- Стейт-модель на тесте ---")
    print(f"  n_test_rows = {len(y_test)}   n_test_matches = {d_test['match_id'].nunique()}"
          f"   (эффективный размер выборки ближе ко второму)")
    print(f"  деревьев: {booster.best_iteration}")
    print(f"  калибратор выбран {n_folds}-фолдовой кросс-валидацией по match_id "
          f"внутри калибровочной выборки ({n_cal_matches} матчей):")
    print(f"    критерий выбора: {calib_criterion}")
    print(f"    {'метод':<10}{'eps':<8}{'log_loss':>10}{'ECE':>9}{'макс.смещ.':>12}")
    for m, (mm, ee, ss) in best_per_method.items():
        mark = "  <-- выбран" if m == chosen_state else ""
        print(f"    {m:<10}{ee:<8g}{ss:>10.4f}{probe_ece[(m, ee)]:>9.3f}"
              f"{probe_gap[(m, ee)]:>12.3f}{mark}")
    print("    (ECE — средневзвешенное смещение по децилям, оно и есть «насколько"
          "\n     честно число на экране». Выбрать по нему: --calib-criterion ece)")
    if calib_eps >= eps_grid[-1]:
        print(f"    ВНИМАНИЕ: eps={calib_eps:g} на границе сетки, оптимум может "
              f"лежать дальше")
    hdr = f"  {'модель':<28}{'log_loss':>10}{'brier':>9}{'acc':>8}{'auc':>8}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for name, m in (("константа (base rate)", m_base),
                    ("только драфт-prior", m_draft_only),
                    ("стейт, сырая", m_raw),
                    ("стейт, калиброванная", m_cal)):
        print(f"  {name:<28}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
              f"{m['acc']:>8.4f}{m['auc']:>8.4f}")

    print("\n--- Reliability: предсказано против фактического (калиброванная) ---")
    rel_cal = reliability_table(y_test, p_test, g_test)
    print(_fmt(rel_cal))
    worst = float(rel_cal["gap"].abs().max()) if len(rel_cal) else float("nan")
    print(f"  ECE (средневзвешенное смещение): {ece(y_test, p_test, g_test):.3f}"
          f"   |   сырая: {ece(y_test, p_test_raw, g_test):.3f}")
    print(f"  максимальное смещение по децилям: {worst:.3f}"
          + ("  — для оверлея приемлемо" if worst <= 0.05 else
             "  — ВЕЛИКО. Число на экране будет систематически врать."))
    print("\n--- Reliability: то же для СЫРОЙ модели (для сравнения) ---")
    print(_fmt(reliability_table(y_test, p_test_raw, g_test)))

    print("\n--- Точность по десятиминутным отрезкам (калиброванная) ---")
    print(_fmt(by_minute_bucket(d_test, p_test, y_test, base_rate)))

    imp = pd.DataFrame({
        "feature": feats,
        "gain": booster.feature_importance("gain"),
    }).sort_values("gain", ascending=False)
    imp["gain_share"] = imp["gain"] / imp["gain"].sum()
    print("\n--- Вклад признаков (gain) ---")
    print(_fmt(imp[["feature", "gain_share"]].head(12)))

    artefact = {
        "hero_ids": hero_ids,
        "id2idx": id2idx,
        "hero_names": hero_names,
        "draft_model": draft_model,
        "draft_calib": chosen,
        "state_calib": chosen_state,
        "draft_coef": coefs,
        "draft_intercept": intercept,
        "draft_feature_names": dnames,
        "booster": booster,
        "iso": iso,
        "calib_eps": calib_eps,
        "state_features": feats,
        "use_xp": use_xp,
        "live_features": live_features,
        "extra_features": extra_features,
        "exact_features": exact_features,
        "gold_norm": gold_norm,
        # Обучалась ли модель со слепым холдаутом. Без этого флага
        # `blindtest` не может отличить настоящую слепую выборку от
        # обычного теста — и молча выдал бы второе за первое.
        "holdout": bool(use_holdout),
        "holdout_permille": HO.HOLDOUT_PERMILLE if use_holdout else 0,
        "elo_pre": elo_pre,
        "elo_final": elo_final,
        # Какой рейтинг подавать в лайве. Без этого ключа артефакт от
        # старого обучения читается как командный — то есть старые модели
        # продолжают работать, а не начинают молча считать не то.
        "rating_kind": rating,
        "player_elo_final": player_final,
        "player_elo_k": config.PLAYER_ELO_K,
        "base_rate": base_rate,
        "test_match_ids": test_ids.tolist(),
        "metrics": {"draft_test": dm_test, "state_test_calibrated": m_cal,
                    "state_test_raw": m_raw, "baseline": m_base,
                    "truth_corr": truth_corr},
        "source_dir": str(source_dir),
        "trained_at": time.time(),
        "pool_frac": float(pool_frac),
        "n_pool_matches": int(len(pool_ids)),
        # Чем эта модель отличается от соседней по ансамблю. Без записи в
        # артефакт «разнородный ансамбль» через неделю не отличить от
        # обычного, а разница между ними — это как раз то, что мерилось.
        "lgbm_overrides": {k: v for k, v in (lgbm_params or {}).items()
                           if v is not None},
    }
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("wb") as fh:
            pickle.dump(artefact, fh)
        print(f"\nМодель сохранена: {out_path}")
    print(f"Всего секунд: {time.time() - t0:.1f}")
    print("=" * 74)
    return artefact if return_artefact else 0



def _safe_stdout() -> None:
    """Не ронять программу из-за символа, которого нет в кодовой странице.

    Windows-консоль по умолчанию бывает cp866 или cp1251: первая не знает
    длинного тире и стрелок, вторая — ещё и блочной графики. Падать из-за
    оформления недопустимо, поэтому непечатаемое заменяется на '?'.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(description="Обучение драфт- и стейт-модели.")
    ap.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    ap.add_argument("--no-xp", action="store_true",
                    help="убрать признаки по опыту: GetRealtimeStats не отдаёт XP, "
                         "только уровни, а достраивать XP из уровней нельзя")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--seed", type=int, default=config.SPLIT_SEED)
    ap.add_argument("--draft-calib", choices=["isotonic", "sigmoid", "none", "auto"],
                    default="isotonic",
                    help="калибратор драфт-модели; auto выбирает по OOF log loss")
    ap.add_argument("--state-calib", choices=["isotonic", "sigmoid", "none", "auto"],
                    default="auto",
                    help="калибратор стейт-модели; auto выбирает на пробной половине")
    ap.add_argument("--extra-features", action="store_true",
                    help="добавить фраги, распределение нетворса и байбэки "
                         "(замерьте прирост через dwp.compare)")
    ap.add_argument("--exact-features", action="store_true",
                    help="добавить признаки, у которых определение в обучении "
                         "и в лайве совпадает точно: уровни, ластхиты, денаи, "
                         "вышки третьего яруса (замерьте прирост через compare)")
    ap.add_argument("--live-features", action="store_true",
                    help="обучать только на признаках, доступных в лайве "
                         "(подразумевает --no-xp); модель для live.py")
    ap.add_argument("--gold-norm", choices=list(F.GOLD_NORM_MODES), default="off",
                    help="перевес по золоту как доля суммарного нетворса: "
                         "add добавляет к абсолютному, replace вытесняет его")
    ap.add_argument("--calib-criterion", choices=["logloss", "ece"], default="logloss",
                    help="по чему выбирать калибратор; ece — честность числа на экране")
    ap.add_argument("--no-holdout", dest="holdout", action="store_false",
                    help="обучаться и на слепом холдауте тоже. Тогда слепой "
                         "оценки не останется вовсе — только для разовых "
                         "экспериментов, не для боевой модели")
    ap.set_defaults(holdout=True)
    # Гиперпараметры LightGBM. Не для «покрутить и посмотреть»: подбирать их
    # по тесту значит расходовать тест, а разброс от сида разбиения (0.0070 на
    # холдауте) больше типичной разницы между разумными наборами. Заведены
    # ради РАЗНООБРАЗИЯ в ансамбле — и это тоже надо мерить, а не полагать.
    ap.add_argument("--leaves", type=int, default=None,
                    help=f"num_leaves (по умолчанию {config.LGBM_PARAMS['num_leaves']})")
    ap.add_argument("--lr", type=float, default=None,
                    help=f"learning_rate (по умолчанию "
                         f"{config.LGBM_PARAMS['learning_rate']})")
    ap.add_argument("--min-leaf", type=int, default=None,
                    help=f"min_data_in_leaf (по умолчанию "
                         f"{config.LGBM_PARAMS['min_data_in_leaf']}); мельче — "
                         f"листья начинают запоминать отдельные матчи, строки "
                         f"внутри матча скоррелированы")
    ap.add_argument("--l2", type=float, default=None,
                    help=f"lambda_l2 (по умолчанию {config.LGBM_PARAMS['lambda_l2']})")
    ap.add_argument("--pool-frac", type=float, default=1.0,
                    help="обучаться на доле обучающего пула (тест не "
                         "урезается). Для кривой обучения: сколько стоит "
                         "объём выборки — `python -m dwp.learning_curve`")
    ap.add_argument("--rating", choices=["player", "team"], default="player",
                    help="какой рейтинг подавать драфт-модели. player "
                         "(умолчание) замерен лучше командного: AUC 0.6138 "
                         "против 0.6001, log loss −0.0049 [−0.0072, −0.0026]")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    src = config.SYNTH_MATCHES_DIR if args.source == "synthetic" else config.RAW_MATCHES_DIR
    # --live-features без --no-xp бессмысленно: XP в лайве тоже нет.
    use_xp = not (args.no_xp or args.live_features)
    suffix = "_live" if args.live_features else ("_noxp" if args.no_xp else "")
    if args.extra_features:
        suffix += "_extra"
    if args.exact_features:
        suffix += "_exact"
    if args.gold_norm != "off":
        suffix += f"_gn{args.gold_norm}"
    out = args.out or (config.MODELS_DIR / f"model_{args.source}{suffix}.pkl")
    return run(src, use_xp=use_xp, limit=args.limit, seed=args.seed,
               out_path=out, draft_calib=args.draft_calib,
               state_calib=args.state_calib, calib_criterion=args.calib_criterion,
               live_features=args.live_features,
               extra_features=args.extra_features, gold_norm=args.gold_norm,
               exact_features=args.exact_features, use_holdout=args.holdout,
               pool_frac=args.pool_frac, rating=args.rating,
               lgbm_params={"num_leaves": args.leaves,
                            "learning_rate": args.lr,
                            "min_data_in_leaf": args.min_leaf,
                            "lambda_l2": args.l2})


if __name__ == "__main__":
    sys.exit(main())
