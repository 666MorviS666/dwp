"""Сводка точности в одном месте — для страницы «точность» в панели.

Числа этого проекта разбросаны по семи модулям и трём файлам, и это не
случайность: они меряют РАЗНОЕ, и складывать их в одну цифру нельзя.
Здесь они собираются рядом, но не смешиваются, а каждое подписано тем,
что оно значит и на какой выборке получено:

  офлайн-тест     метрики из артефакта модели: её собственный тест
  слепой холдаут  15% матчей, спрятанных от обучения; реестр вскрытий
  лайв            калибровка по фактически показанным на экране числам

Ни одно из них не «главное». Офлайн-метрика меряет модель, лайв меряет
то, что видел зритель, и они расходятся: на восьми матчах ECE лайва был
0.126 против обещанных офлайн 0.021-0.034. Показывать надо оба.

Кривая обучения и коридор неопределённости отсюда убраны: на панели их
блоков больше нет, а сами замеры живут в своих модулях
(`dwp.learning_curve`, `dwp.forecast --check`) и печатаются там.
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, holdout as HO
from .train import by_minute_bucket, core_metrics, ece, reliability_table


def _read_artefact(path: Path) -> dict | None:
    try:
        with path.open("rb") as fh:
            return pickle.load(fh)
    except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
            ModuleNotFoundError):
        return None


def models() -> list[dict]:
    """Что обучено и с какими офлайн-метриками."""
    out = []
    for p in sorted(config.MODELS_DIR.glob("*.pkl")):
        a = _read_artefact(p)
        if a is None:
            out.append({"name": p.name, "error": "артефакт не читается"})
            continue
        m = a.get("metrics") or {}
        cal = m.get("state_test_calibrated") or {}
        base = m.get("baseline") or {}
        draft = m.get("draft_test") or {}
        out.append({
            "name": p.name,
            "features": len(a.get("state_features") or []),
            "feature_names": list(a.get("state_features") or []),
            "live": bool(a.get("live_features")),
            "exact": bool(a.get("exact_features")),
            "extra": bool(a.get("extra_features")),
            "gold_norm": a.get("gold_norm"),
            "use_xp": bool(a.get("use_xp")),
            "holdout": a.get("holdout"),
            "pool_frac": a.get("pool_frac"),
            "n_pool": a.get("n_pool_matches"),
            "trained_at": a.get("trained_at"),
            "state_calib": a.get("state_calib"),
            "calib_eps": a.get("calib_eps"),
            "n_test_matches": len(a.get("test_match_ids") or []),
            "log_loss": cal.get("log_loss"),
            "brier": cal.get("brier"),
            "acc": cal.get("acc"),
            "auc": cal.get("auc"),
            "baseline": base.get("log_loss"),
            "draft_log_loss": draft.get("log_loss"),
            "draft_auc": draft.get("auc"),
            "size": p.stat().st_size,
        })
    return out


def reveals() -> list[dict]:
    """Реестр вскрытий холдаута: чем их больше, тем меньше в нём слепоты."""
    return HO.registry_rows()


def live_calibration(log_dir: Path | None = None) -> dict:
    """Калибровка по ФАКТИЧЕСКИ ПОКАЗАННЫМ числам, отдельно по моделям.

    Считается только по уже добранным исходам (`--offline`-режим
    `livecheck`): ходить в сеть из веб-обработчика нельзя — страница
    повиснет на минуту, а виноватой будет выглядеть панель.
    """
    from .livecheck import load_logs, resolve

    log_dir = log_dir or config.LIVE_LOG_DIR
    out: dict = {"models": [], "note": "", "n_rows": 0, "n_matches": 0}
    df = load_logs(log_dir)
    if df.empty:
        out["note"] = ("лайв-лога нет. Он пишется сам, когда панель смотрит "
                       "матч: data/live_log/<match_id>.csv")
        return out
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df[df["match_id"].notna()].copy()
    df["match_id"] = df["match_id"].astype(np.int64)
    out["n_rows_total"] = int(len(df))
    out["n_matches_total"] = int(df["match_id"].nunique())
    matches = resolve(sorted(df["match_id"].unique().tolist()),
                      config.LIVE_RESOLVED_DIR, offline=True)
    if not matches:
        out["note"] = ("исходы записанных матчей ещё не добраны. "
                       "Запустите задачу «Калибровка лайва» — она сходит "
                       "в OpenDota и посчитает.")
        return out
    df = df[df["match_id"].isin(matches)].copy()
    df["y"] = df["match_id"].map(
        lambda m: 1 if matches[int(m)].get("radiant_win") else 0)
    out["n_rows"] = int(len(df))
    out["n_matches"] = int(df["match_id"].nunique())
    out["pending"] = out["n_matches_total"] - out["n_matches"]

    for name in sorted(df["model"].dropna().unique()) if "model" in df else []:
        d = df[df["model"] == name]
        y = d["y"].to_numpy()
        p = d["p"].to_numpy(dtype=float)
        g = d["match_id"].to_numpy()
        item = {"model": str(name), "n_rows": int(len(d)),
                "n_matches": int(d["match_id"].nunique()),
                "wins_radiant": int(d.groupby("match_id")["y"].first().sum())}
        if len(np.unique(y)) < 2:
            item["note"] = ("все матчи с одним исходом — калибровку считать "
                            "не на чем")
            out["models"].append(item)
            continue
        m = core_metrics(y, p)
        item.update({"log_loss": m["log_loss"], "brier": m["brier"],
                     "acc": m["acc"], "auc": m["auc"],
                     "ece": float(ece(y, p, g))})
        rel = reliability_table(y, p, g)
        item["reliability"] = json.loads(rel.to_json(orient="records"))
        if "minute" in d.columns:
            bym = by_minute_bucket(d, p, y, float(np.mean(y)))
            item["by_minute"] = json.loads(bym.to_json(orient="records"))
        out["models"].append(item)
    if not out["models"]:
        out["note"] = "в логе нет колонки model — старый формат"
    return out


def bands() -> list[dict]:
    """Обещанная честность числа по отрезкам матча (config)."""
    return [{"from": a, "to": b, "ece": e} for a, b, e in config.RELIABILITY_BANDS]


def headline() -> dict | None:
    """Две строки, ради которых раздел и открывают: точность лайв-модели и
    точность драфт-модели — обе на слепом холдауте, обе с выборкой.

    Берётся из `data/holdout_preds.npz` и `data/verdict.json`, то есть из
    ОДНОГО вскрытия холдаута, а не из собственных тестов моделей: у
    моделей с разным сидом тесты разные, и сравнивать их между собой
    нельзя. Нет файлов — возвращаем None, и раздел честно говорит, что
    считать не на чем, вместо того чтобы показать метрики из артефактов
    и выдать их за холдаут.
    """
    cache = config.DATA_DIR / "holdout_preds.npz"
    if not cache.exists():
        return None
    try:
        d = np.load(cache, allow_pickle=True)
    except (OSError, ValueError):
        return None
    if "p__ENSEMBLE" not in d.files:
        return None
    y = d["y"].astype(int)
    mid = d["match_id"]
    p = d["p__ENSEMBLE"].astype(float)
    out: dict = {"n_rows": int(len(y)), "n_matches": int(len(np.unique(mid)))}

    # «Точность» — доля строк, где сторона угадана. Именно она понятна без
    # объяснений, в отличие от log loss; log loss оставлен рядом, потому
    # что по нему принимаются решения.
    m = core_metrics(y, p)
    out["live"] = {
        "acc": float(((p >= 0.5).astype(int) == y).mean()),
        "log_loss": m["log_loss"], "brier": m["brier"], "auc": m["auc"],
        "ece": float(ece(y, p, mid)),
    }
    # Драфт-модель: одна строка на матч, оценка до начала игры.
    if "draft_logit" in d.files:
        dl = d["draft_logit"].astype(float)
        first = np.unique(mid, return_index=True)[1]
        z = dl[first]
        yy = y[first]
        ok = np.isfinite(z)
        if ok.sum() > 50:
            pd_ = 1.0 / (1.0 + np.exp(-z[ok]))
            md = core_metrics(yy[ok], pd_)
            out["draft"] = {
                "acc": float(((pd_ >= 0.5).astype(int) == yy[ok]).mean()),
                "auc": md["auc"], "log_loss": md["log_loss"],
                "n_matches": int(ok.sum()),
            }
    v = None
    vp = config.DATA_DIR / "verdict.json"
    if vp.exists():
        try:
            v = (json.loads(vp.read_text(encoding="utf-8")) or {}).get("measured")
        except (OSError, json.JSONDecodeError):
            v = None
    out["verdict"] = v
    return out


def everything(log_dir: Path | None = None) -> dict:
    return {
        "headline": headline(),
        "models": models(),
        "reveals": reveals(),
        "live": live_calibration(log_dir),
        "holdout_permille": HO.HOLDOUT_PERMILLE,
        "when": time.time(),
    }
