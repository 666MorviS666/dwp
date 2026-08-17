"""Взгляд вперёд: насколько игра ещё не решена и куда её ведёт темп.

ПОСТАНОВКА ЗАДАЧИ, ради которой модуль и заведён. Стейт-модель отвечает на
вопрос «кто выиграет при таком положении», и отвечает честно. Но на экране
это выглядит как аналитика вслед событиям: команда отыгралась — число
выросло, отдала — упало. Инструмент, чья задача предсказывать, обязан
говорить и то, чего ещё не случилось: сколько в этой позиции осталось
неопределённости и в какую сторону она смещена.

Выдумывать для этого второй «прогнозный» процент нельзя: калиброванная
вероятность и так есть лучшая оценка исхода, а любое число «на
перспективу» поверх неё будет либо той же величиной, либо хуже неё. Что
можно сделать честно — три вещи, и все три здесь измеримы:

1. КОРИДОР. Взять историю и посмотреть, чем такие позиции кончались:
   где окажется оценка через N минут, как часто лидер меняется, как часто
   матч вообще успевает закончиться. Это не прогноз поверх модели, это
   распределение будущего той же модели, замеренное на матчах, которых
   она не видела.

2. ТЕМП. Продлить текущие производные на N минут вперёд и пересчитать
   моделью: «если темп сохранится». Это ЯВНОЕ допущение, а не прогноз, и
   подписано оно должно быть именно так. Полезно ли оно вообще — вопрос
   замера (`--check`), а не веры.

3. ЧТО ЕЩЁ НЕ РАЗЫГРАНО. Стоящие t3 и бараки, свободное золото на выкупы.
   Никакой модели, просто перечень ресурсов, которые ещё в игре: отставание
   в 20 тысяч при целых казармах и при снесённых — разные позиции, и это
   видно без всякой статистики.

    python -m dwp.forecast --build      # собрать коридор по холдауту
    python -m dwp.forecast --check      # бьёт ли экстраполяция темпа «p не меняется»
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features as F, holdout as HO, live

TABLE_PATH = config.DATA_DIR / "horizon.json"

# Горизонты, на которые считается коридор. 10 минут — это заметный кусок
# матча и при этом не «до конца»: на 30-й минуте до конца может быть и пять
# минут, и сорок.
HORIZONS = (5.0, 10.0)
DEFAULT_HORIZON = 10.0

# Границы отрезков по минуте. Мелко в начале, где всё меняется быстро.
MINUTE_BINS = (0, 10, 15, 20, 25, 30, 35, 40, 50, 200)
# Границы по оценке ЛИДЕРА (то есть по max(p, 1-p), всегда >= 0.5).
# Разбиение неравномерное: разница между 0.90 и 0.97 для зрителя больше,
# чем между 0.55 и 0.62.
LEAD_BINS = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99, 1.01)

# Меньше — доля неотличима от чего угодно, клетку не показываем. То же
# правило, что у таблицы камбэков: число без выборки на экран не идёт.
MIN_CELL = 40


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _bin(value: float, edges) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= value < edges[i + 1]:
            return i
    return None


def _cell_key(mi: int, li: int) -> str:
    return f"{mi}:{li}"


# --- Сбор коридора по истории -------------------------------------------

def collect_rows(art: dict, matches: list[dict],
                 horizons=HORIZONS) -> pd.DataFrame:
    """Поминутные (минута, p, p через N минут, исход) по каждому матчу.

    Если матч кончился раньше горизонта, «оценка через N минут» — это сам
    исход: 0 или 1. Так честнее, чем выбрасывать такие строки, и заодно
    даёт полезное число «как часто игра просто успевает закончиться».
    """
    from .compare import _predict_single

    out = []
    for m in matches:
        mid = int(m["match_id"])
        parsed = F.parse_objectives(m)
        fr = F.match_state_frame(m, parsed)
        ps = []
        for a in live.members(art):
            elo = a["elo_pre"].get(mid, a["elo_pre"].get(str(mid), 0.0))
            Xd, _, _ = F.draft_matrix([m], a["id2idx"], {mid: elo})
            pd_ = float(a["draft_model"].predict_proba(Xd)[0, 1])
            d = fr.copy()
            d["draft_logit"] = float(np.log(pd_ / (1 - pd_)))
            ps.append(_predict_single(a, d))
        p = np.mean(ps, axis=0)
        minute = fr["minute"].to_numpy(dtype=float)
        y = int(fr["y"].iloc[0])
        end = float(minute.max())
        by_minute = {int(round(mn)): float(pp) for mn, pp in zip(minute, p)}
        for mn, pp in zip(minute, p):
            row = {"match_id": mid, "minute": float(mn), "p": float(pp), "y": y}
            for h in horizons:
                t = int(round(mn + h))
                if t <= end and t in by_minute:
                    row[f"p{h:g}"] = by_minute[t]
                    row[f"ended{h:g}"] = 0
                else:
                    row[f"p{h:g}"] = float(y)
                    row[f"ended{h:g}"] = 1
            out.append(row)
    return pd.DataFrame(out)


def build_table(rows: pd.DataFrame, horizons=HORIZONS) -> dict:
    """Клетки (отрезок минут) x (оценка лидера) -> распределение будущего.

    Всё считается ОТ ЛИЦА ТЕКУЩЕГО ЛИДЕРА: p_lead = max(p, 1-p), а
    будущая оценка берётся у той же стороны. Так удваивается выборка
    (стороны складываются) и вопрос формулируется так, как его задают
    вслух: «удержит ли тот, кто ведёт».
    """
    lead = np.maximum(rows["p"].to_numpy(), 1.0 - rows["p"].to_numpy())
    is_radiant_lead = rows["p"].to_numpy() >= 0.5
    y = rows["y"].to_numpy()
    lead_wins = np.where(is_radiant_lead, y, 1 - y)
    mn = rows["minute"].to_numpy()

    cells: dict[str, dict] = {}
    for mi in range(len(MINUTE_BINS) - 1):
        in_m = (mn >= MINUTE_BINS[mi]) & (mn < MINUTE_BINS[mi + 1])
        for li in range(len(LEAD_BINS) - 1):
            sel = in_m & (lead >= LEAD_BINS[li]) & (lead < LEAD_BINS[li + 1])
            n = int(sel.sum())
            if n < MIN_CELL:
                continue
            cell = {
                "n": n,
                "n_matches": int(rows["match_id"].to_numpy()[sel].size and
                                 len(set(rows["match_id"].to_numpy()[sel]))),
                "minute": [MINUTE_BINS[mi], MINUTE_BINS[mi + 1]],
                "lead": [LEAD_BINS[li], min(1.0, LEAD_BINS[li + 1])],
                "p_lead_mean": float(lead[sel].mean()),
                "lead_wins": float(lead_wins[sel].mean()),
            }
            for h in horizons:
                fut = rows[f"p{h:g}"].to_numpy()[sel]
                fut_lead = np.where(is_radiant_lead[sel], fut, 1.0 - fut)
                q = np.percentile(fut_lead, [10, 25, 50, 75, 90])
                cell[f"h{h:g}"] = {
                    "q10": float(q[0]), "q25": float(q[1]), "q50": float(q[2]),
                    "q75": float(q[3]), "q90": float(q[4]),
                    "flip": float(np.mean(fut_lead < 0.5)),
                    "ended": float(rows[f"ended{h:g}"].to_numpy()[sel].mean()),
                    "move": float(np.mean(np.abs(fut_lead - lead[sel]))),
                }
            cells[_cell_key(mi, li)] = cell
    return {
        "cells": cells, "minute_bins": list(MINUTE_BINS),
        "lead_bins": list(LEAD_BINS), "horizons": [float(h) for h in horizons],
        "min_cell": MIN_CELL, "n_rows": int(len(rows)),
        "n_matches": int(rows["match_id"].nunique()),
        "built_at": time.time(),
    }


def load_table(path: Path | None = None) -> dict | None:
    p = path or TABLE_PATH
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) and d.get("cells") else None


def band(table: dict | None, minute: float, p: float,
         horizon: float = DEFAULT_HORIZON) -> dict | None:
    """Коридор для текущей позиции. None — клетки нет, и врать нечем."""
    if not table or minute is None or p is None:
        return None
    if minute != minute or p != p:
        return None
    mi = _bin(float(minute), table.get("minute_bins") or MINUTE_BINS)
    lead = max(float(p), 1.0 - float(p))
    li = _bin(lead, table.get("lead_bins") or LEAD_BINS)
    if mi is None or li is None:
        return None
    cell = (table.get("cells") or {}).get(_cell_key(mi, li))
    if not cell:
        return None
    h = cell.get(f"h{horizon:g}")
    if not h:
        return None
    radiant_leads = float(p) >= 0.5
    return {
        "horizon": float(horizon),
        "leader": "radiant" if radiant_leads else "dire",
        "p_lead": lead,
        "n": cell["n"], "n_matches": cell.get("n_matches"),
        "minute": cell["minute"], "lead_bin": cell["lead"],
        # Доли, замеренные на матчах, которых модель не видела.
        "lead_wins": cell["lead_wins"],
        "flip": h["flip"], "ended": h["ended"], "move": h["move"],
        "q10": h["q10"], "q25": h["q25"], "q50": h["q50"],
        "q75": h["q75"], "q90": h["q90"],
    }


# --- Экстраполяция темпа -------------------------------------------------

# Что продлевается вперёд и чем. Всё остальное держится постоянным
# НАМЕРЕННО: продлевать потерю вышек или уровни значило бы выдумывать
# события, а не темп. Экстраполируется только то, у чего производная и
# так есть в признаках.
PROJECT_RULES = {
    "gold_adv": ("gold_adv_slope5", 1.0),
    "gold_adv_norm": ("gold_adv_norm_slope5", 1.0),
    "kills_adv": ("kills_adv_d5", 0.2),        # d5 — прирост за 5 минут
}


# ЗАМЕР, из-за которого проекция НЕ показывается на панели. Прогнать
# заново: `python -m dwp.forecast --check [--horizon 5]`.
# 42843 строки, 1167 матчей слепого холдаута, ансамбль из пяти сидов:
#
#   горизонт   ошибка по будущей оценке      log loss против исхода
#              наивно      по темпу         наивно      по темпу
#   5 минут    0.1267      0.1345           0.5236      0.5341
#   10 минут   0.1796      0.1879           0.5236      0.5542
#
# То есть «если темп сохранится» ХУЖЕ, чем «оценка не изменится», и по
# обоим вопросам сразу. Перевес по золоту за последние пять минут не
# продлевается: он сам по себе возвращается к среднему. Показывать такое
# число на экране значило бы показывать заведомо худшую оценку рядом с
# лучшей — ровно та беда, ради которой этот блок и заводился.
#
# Код проекции оставлен: он нужен, чтобы замер можно было повторить, и
# чтобы следующая попытка начиналась не с нуля. Полезной эта величина
# может стать, если продлевать не золото, а что-то, у чего инерция выше.
PROJECTION_MEASURED = {
    "5": {"mae_now": 0.1267, "mae_proj": 0.1345,
          "ll_now": 0.5236, "ll_proj": 0.5341},
    "10": {"mae_now": 0.1796, "mae_proj": 0.1879,
           "ll_now": 0.5236, "ll_proj": 0.5542},
    "n_rows": 42843, "n_matches": 1167,
}


def project(art: dict, df: pd.DataFrame, horizon: float = DEFAULT_HORIZON
            ) -> dict | None:
    """«Если темп сохранится»: продлить производные и пересчитать моделью.

    ИЗМЕРЕНО И ОТКЛОНЕНО ДЛЯ ЭКРАНА — см. PROJECTION_MEASURED выше.
    Функция оставлена ради воспроизводимости замера.
    """
    row = df.iloc[0]
    minute = row.get("minute", np.nan)
    if minute != minute:
        return None
    d = df.copy()
    changed = []
    for target, (src, k) in PROJECT_RULES.items():
        if target not in d.columns or src not in d.columns:
            continue
        base, slope = row.get(target, np.nan), row.get(src, np.nan)
        if base != base or slope != slope:
            continue
        d.loc[d.index[0], target] = float(base) + float(slope) * k * horizon
        changed.append(target)
    if not changed:
        return None
    d.loc[d.index[0], "minute"] = float(minute) + horizon
    d.attrs = dict(df.attrs)
    p, _bd = live.predict(art, d)
    return {
        "horizon": float(horizon),
        "p": float(p),
        "minute": float(minute) + horizon,
        "changed": changed,
        "gold_adv": (float(d.iloc[0]["gold_adv"])
                     if "gold_adv" in d.columns else None),
    }


def tempo(df: pd.DataFrame) -> dict | None:
    """Темп как ФАКТ, а не как прогноз.

    Проекция вероятности по темпу замерена и оказалась хуже наивной
    (PROJECTION_MEASURED). Но сам темп — не прогноз, а величина из
    признаков: столько золота команда прибавила за последние пять минут,
    столько фрагов набрала. Это ровно то, чего не видно в проценте:
    отстающая команда, которая последние пять минут зарабатывает
    быстрее, и отстающая, которая продолжает проваливаться, дают на
    экране одно и то же число.

    None — окно ещё не наполнилось. Окна считаются в ИГРОВЫХ минутах от
    момента подключения, а не в опросах.
    """
    row = df.iloc[0]

    def val(name):
        v = row.get(name, np.nan)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return None
        return None if v != v else v

    g5, k5, g1 = val("gold_adv_slope5"), val("kills_adv_d5"), val("gold_adv_d1")
    if g5 is None and k5 is None and g1 is None:
        return None
    return {
        "gold_per_min": g5,          # разница золота: сколько в минуту
        "gold_last_min": g1,
        "kills_5": k5,
        "minute": val("minute"),
    }


# --- Что ещё не разыграно ------------------------------------------------

def resources(st: dict, builds: dict | None = None) -> dict:
    """Ресурсы, которые ещё в игре. Никакой модели — перечень фактов.

    Зачем на экране. Отставание в 20 тысяч при целых казармах и при
    снесённых — совершенно разные позиции, и разницу видно без всякой
    статистики. Модель это отчасти знает (вышки и бараки у неё в
    признаках), но на экране число одно, а из чего оно сложилось —
    не видно.
    """
    tow = st.get("towers_lost") or {}
    rax = st.get("rax_lost") or {}
    t3 = st.get("t3_lost") or {}
    out: dict = {"sides": []}
    for tn in (config.TEAM_RADIANT, config.TEAM_DIRE):
        side = {
            "towers_standing": (None if tn not in tow
                                else config.N_TOWERS_PER_SIDE - int(tow[tn])),
            "t3_standing": (None if tn not in t3
                            else config.N_T3_PER_SIDE - int(t3[tn])),
            "rax_standing": (None if tn not in rax
                             else config.N_RAX_PER_SIDE - int(rax[tn])),
        }
        out["sides"].append(side)
    if builds and builds.get("teams") and len(builds["teams"]) == 2:
        for i, t in enumerate(builds["teams"]):
            out["sides"][i]["unspent"] = t.get("unspent")
            out["sides"][i]["item_value"] = t.get("value")
    return out


# --- Всё вместе для панели ----------------------------------------------

def outlook(art: dict, df: pd.DataFrame, st: dict, p: float,
            table: dict | None, comeback: dict | None = None,
            builds: dict | None = None,
            horizon: float = DEFAULT_HORIZON) -> dict:
    """Блок «перспектива» целиком. Каждая часть может быть None."""
    row = df.iloc[0]
    minute = row.get("minute", np.nan)
    out: dict = {
        "horizon": float(horizon),
        "band": band(table, minute, p, horizon),
        "tempo": tempo(df),
        "resources": resources(st, builds),
        "comeback": None,
        # Чтобы вопрос «а почему не показываете прогноз по темпу» имел
        # ответ прямо на экране, а не только в README.
        "projection_measured": PROJECTION_MEASURED,
    }
    if comeback is not None:
        out["comeback"] = comeback
    # Насколько «решена» игра: доля будущих оценок, оставшихся за лидером.
    b = out["band"]
    if b:
        out["settled"] = 1.0 - float(b["flip"])
        out["model_vs_history"] = float(b["p_lead"]) - float(b["lead_wins"])
    return out


# --- Замер: помогает ли экстраполяция темпа ------------------------------

def check_projection(rows: pd.DataFrame,
                     horizon: float = DEFAULT_HORIZON) -> dict:
    """Бьёт ли «если темп сохранится» наивное «оценка не изменится».

    Меряется двумя разными вопросами, и их нельзя путать:

    * КУДА ПОЕДЕТ ЧИСЛО — сравнение с фактической оценкой той же модели
      через N минут. Тут экстраполяция может выиграть: она пользуется
      производной, а «p не меняется» — нет.
    * КТО ВЫИГРАЕТ — log loss против исхода. Тут выиграть почти нельзя:
      калиброванная p и так лучшая оценка исхода. Если экстраполяция
      всё-таки выигрывает, это значит, что модель НЕ ДОБИРАЕТ из темпа,
      и такой результат надо записать, а не отпраздновать.
    """
    ok = rows[f"proj{horizon:g}"].notna() & rows[f"p{horizon:g}"].notna()
    d = rows[ok]
    fut = d[f"p{horizon:g}"].to_numpy()
    now = d["p"].to_numpy()
    prj = d[f"proj{horizon:g}"].to_numpy()
    y = d["y"].to_numpy()
    eps = 1e-6

    def ll(p_):
        p_ = np.clip(p_, eps, 1 - eps)
        return float(-np.mean(y * np.log(p_) + (1 - y) * np.log(1 - p_)))

    return {
        "n": int(len(d)),
        "n_matches": int(d["match_id"].nunique()),
        "mae_now": float(np.mean(np.abs(now - fut))),
        "mae_proj": float(np.mean(np.abs(prj - fut))),
        "ll_now": ll(now),
        "ll_proj": ll(prj),
        "moved": float(np.mean(np.abs(prj - now))),
    }


# --- CLI ------------------------------------------------------------------

def _load_holdout(limit: int | None) -> list[dict]:
    from .bench_ensemble import load_holdout
    return load_holdout(limit, verbose=True)


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Коридор неопределённости и проверка экстраполяции темпа.")
    ap.add_argument("--build", action="store_true",
                    help="собрать таблицу коридора по слепому холдауту")
    ap.add_argument("--check", action="store_true",
                    help="замерить, помогает ли экстраполяция темпа")
    ap.add_argument("--show", action="store_true",
                    help="напечатать собранную таблицу")
    ap.add_argument("--model", type=Path, nargs="+", default=None)
    ap.add_argument("--n", type=int, default=None,
                    help="ограничить число матчей холдаута")
    ap.add_argument("--horizon", type=float, default=DEFAULT_HORIZON)
    ap.add_argument("--out", type=Path, default=TABLE_PATH)
    args = ap.parse_args(argv)

    if args.show:
        t = load_table(args.out)
        if not t:
            print(f"Таблицы нет: {args.out}\nЧто делать: python -m dwp.forecast "
                  "--build", file=sys.stderr)
            return 2
        print_table(t, args.horizon)
        return 0
    if not (args.build or args.check):
        ap.print_help()
        return 0

    try:
        art = live.load_models(args.model or live.default_models())
    except live.LiveError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    for a in live.members(art):
        if not a.get("holdout"):
            print("ОШИБКА: модель обучена БЕЗ слепого холдаута — считать по "
                  "нему коридор нельзя, эти матчи она видела.\nЧто делать: "
                  "обучите набор без флага --no-holdout.", file=sys.stderr)
            return 2

    print("=" * 74)
    print(f"Модель: {art.get('name')}")
    matches = _load_holdout(args.n)
    if not matches:
        print("ОШИБКА: холдаут пуст.", file=sys.stderr)
        return 2
    print(f"Слепой холдаут: {len(matches)} матчей")
    t0 = time.time()
    rows = collect_rows(art, matches)
    print(f"Строк: {len(rows)}   разбор {time.time() - t0:.0f} с")
    print("=" * 74)

    if args.build:
        table = build_table(rows)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(table, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print(f"\nКлеток заполнено: {len(table['cells'])} "
              f"(порог {MIN_CELL} наблюдений)")
        print(f"Таблица сохранена: {args.out}")
        print_table(table, args.horizon)
        sha = hashlib.sha256(
            np.ascontiguousarray(rows["p"].to_numpy()).tobytes()).hexdigest()
        n = HO.note_reveal(
            (art.get("name") or "модель")[:60], "holdout",
            int(rows["match_id"].nunique()), int(len(rows)), sha,
            {"log_loss": float("nan"), "brier": float("nan"),
             "ece": float("nan")},
            note="forecast --build: описательная таблица, модель не менялась")
        print(f"\nОбращение к холдауту записано в реестр (по этой модели их "
              f"уже {n}).")
        print("Это описательная таблица, а не выбор модели: по ней ничего не "
              "подгоняется.\nНо запись всё равно нужна — чтобы число взглядов "
              "не терялось.")

    if args.check:
        h = args.horizon
        print(f"\n--- Экстраполяция темпа на {h:g} минут ---")
        col = f"proj{h:g}"
        rows = _project_rows(art, matches, rows, h)
        if col not in rows.columns:
            print("ОШИБКА: проекция не посчиталась.", file=sys.stderr)
            return 2
        res = check_projection(rows, h)
        print(f"  строк {res['n']} из {res['n_matches']} матчей   "
              f"средний сдвиг проекции {res['moved']:.3f}")
        print(f"\n  {'что сравниваем':<38}{'наивно':>10}{'по темпу':>11}")
        print("  " + "-" * 59)
        print(f"  {'ошибка по будущей оценке (|dp|)':<38}"
              f"{res['mae_now']:>10.4f}{res['mae_proj']:>11.4f}")
        print(f"  {'log loss против ИСХОДА':<38}"
              f"{res['ll_now']:>10.4f}{res['ll_proj']:>11.4f}")
        print()
        if res["mae_proj"] < res["mae_now"]:
            print("  Куда поедет число: экстраполяция ближе к фактической "
                  "будущей оценке.")
        else:
            print("  Куда поедет число: экстраполяция НЕ ближе наивного "
                  "«останется как есть».")
        if res["ll_proj"] < res["ll_now"]:
            print("  Против исхода она тоже лучше — это значит, что модель "
                  "НЕ ДОБИРАЕТ из темпа.")
            print("  Такой результат надо не праздновать, а проверять: скорее "
                  "всего, признакам не хватает производных.")
        else:
            print("  Против исхода она хуже — и так и должно быть: "
                  "калиброванная p уже")
            print("  лучшая оценка исхода. Значит проекцию можно показывать "
                  "как сценарий,")
            print("  но нельзя показывать вместо вероятности.")
    print("=" * 74)
    return 0


def _project_rows(art: dict, matches: list[dict], rows: pd.DataFrame,
                  horizon: float) -> pd.DataFrame:
    """Проекция по каждой строке выгрузки. Отдельно, потому что дорого."""
    from .compare import _predict_single

    col = f"proj{horizon:g}"
    got: list[tuple[int, float, float]] = []
    for m in matches:
        mid = int(m["match_id"])
        parsed = F.parse_objectives(m)
        fr = F.match_state_frame(m, parsed)
        ps = []
        for a in live.members(art):
            elo = a["elo_pre"].get(mid, a["elo_pre"].get(str(mid), 0.0))
            Xd, _, _ = F.draft_matrix([m], a["id2idx"], {mid: elo})
            pdr = float(a["draft_model"].predict_proba(Xd)[0, 1])
            d = fr.copy()
            d["draft_logit"] = float(np.log(pdr / (1 - pdr)))
            for target, (src, k) in PROJECT_RULES.items():
                if target in d.columns and src in d.columns:
                    d[target] = d[target] + d[src] * k * horizon
            d["minute"] = d["minute"] + horizon
            ps.append(_predict_single(a, d))
        p = np.mean(ps, axis=0)
        for mn, pp in zip(fr["minute"].to_numpy(dtype=float), p):
            got.append((mid, float(mn), float(pp)))
    proj = pd.DataFrame(got, columns=["match_id", "minute", col])
    return rows.merge(proj, on=["match_id", "minute"], how="left")


def print_table(t: dict, horizon: float = DEFAULT_HORIZON) -> None:
    key = f"h{horizon:g}"
    print(f"\n--- Коридор на {horizon:g} минут вперёд "
          f"({t.get('n_matches')} матчей, {t.get('n_rows')} строк) ---")
    print(f"  {'минуты':<10}{'оценка лидера':<16}{'n':>6}{'смена лидера':>14}"
          f"{'матч кончится':>15}{'лидер выиграл':>15}")
    print("  " + "-" * 74)
    for k in sorted(t["cells"], key=lambda s: (int(s.split(":")[0]),
                                               int(s.split(":")[1]))):
        c = t["cells"][k]
        h = c.get(key)
        if not h:
            continue
        mrange = f"{c['minute'][0]}-{c['minute'][1]}"
        lrange = f"{c['lead'][0]:.2f}-{c['lead'][1]:.2f}"
        print(f"  {mrange:<10}{lrange:<16}{c['n']:>6}{h['flip']:>13.1%}"
              f"{h['ended']:>15.1%}{c['lead_wins']:>15.1%}")
    print("\n  «смена лидера» — доля случаев, когда через горизонт впереди")
    print("  оказалась другая сторона. «лидер выиграл» — чем такие позиции")
    print("  кончались на самом деле; сравнивать его надо с самой оценкой.")


if __name__ == "__main__":
    sys.exit(main())
