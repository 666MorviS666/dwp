"""Свести обучение и лайв к одной величине золота: восстановить нетворс.

САМАЯ ДОРОГАЯ ИЗВЕСТНАЯ НЕТОЧНОСТЬ ПРОЕКТА. Обучение подаёт в признак
`gold_adv` разницу ДОБЫТОГО золота (`radiant_gold_adv` из OpenDota), а
лайв подаёт разницу НЕТВОРСА (Σ `net_worth` из GetRealtimeStats). Это
разные величины, и цена расхождения замерена: в среднем 6.3 п.п. на
экране, до 22.5 на реальном матче, в 81% строк в пользу лидера.

ЧТО УЖЕ ПРОБОВАЛИ И ПОЧЕМУ НЕ ВЫШЛО (README, «Что доработать», п. 0):

    подгонка множителем ×1.36        ошибка не в масштабе, корреляция 0.39
    graph_gold вместо нетворса       та же величина другой природы
    восстановление по purchase_log   |медиана| 1052 на игроке, но по
                                     десяти игрокам копится до 7546 на
                                     командной разнице — БОЛЬШЕ, чем
                                     устраняемый зазор

ЧТО ЗДЕСЬ ДЕЛАЕТСЯ ИНАЧЕ. Не восстанавливается нетворс каждого игрока.
Восстанавливается сразу КОМАНДНАЯ РАЗНИЦА — та единственная величина,
которая идёт в признак. Ошибки отдельных игроков при этом не копятся, а
частично гасятся, и подгонка идёт прямо под нужный ответ.

    нетворс_adv(t) ≈ добытое_adv(t) + b0 + b1·t + b2·Δсмертей(t)
                                    + b3·Δбуйбэков(t) + b4·Δ(буйбэк×уровень)(t)
                                    + b5·Δуровней(t)

Все шесть величин есть ПОМИНУТНО: смерти — из `kills_log` противников
(восстанавливаются с потерей 1.5%: смерти от крипов и вышек туда не
попадают), буйбэки — из `buyback_log` со временем, уровни — из `xp_t`
через `config.XP_TO_LEVEL`.

ГДЕ ПОДГОНЯЕТСЯ И ГДЕ ПРОВЕРЯЕТСЯ — РАЗНЫЕ ВЕЩИ, И ЭТО ЗДЕСЬ ГЛАВНОЕ.
Коэффициенты подгоняются на КОНЦЕ матчей, где нетворс известен точно
(`players[].net_worth`). Но применять их придётся ПОСРЕДИ матча, где
проверить их этим способом нечем. Поэтому есть вторая, независимая
проверка: записанные живьём матчи, у которых есть и поминутный нетворс
из лайв-лога, и разобранный офлайн-матч (`data/live_log/resolved/`). На
них видно, держится ли подгонка не только в конечной точке.

    python -m dwp.goldrecon --fit            # подогнать на концах матчей
    python -m dwp.goldrecon --check-live     # проверить посреди матча
    python -m dwp.goldrecon --fit --check-live --n 4000
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config, features as F, holdout as HO

OUT_JSON = config.DATA_DIR / "gold_recon.json"

# Порядок обязан совпадать с порядком в подогнанном векторе — иначе
# коэффициенты молча уедут не к тем величинам.
#
# СВОБОДНОГО ЧЛЕНА ЗДЕСЬ НЕТ, И ЭТО НЕ МЕЛОЧЬ. На нулевой минуте зазор
# между разницей добытого золота и разницей нетворса равен НУЛЮ по
# определению: обе команды стартуют с нуля и того, и другого. Все
# оставшиеся слагаемые в нуле тоже обращаются в ноль (минута, накопленные
# смерти, буйбэки, разница уровней), то есть поправка проходит через
# начало координат так же, как настоящий зазор.
#
# С свободным членом она через ноль не проходит, и это было замерено:
# подгонка на концах матчей давала const = +1207 золота, то есть на
# первой минуте поправка утверждала, что одна команда богаче другой на
# тысячу с лишним просто так. На концах это незаметно (там его гасит
# `minute`), а посреди матча ломает всё: проверка на живых логах дала
# |ошибку| 3269 -> 3659, то есть ХУЖЕ, чем без поправки.
TERMS = ("minute", "d_adv", "bb_adv", "bblv_adv", "lv_adv")
TERMS_WITH_CONST = ("const",) + TERMS


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _levels(xp_t: list) -> np.ndarray:
    return np.array([F.level_from_xp(float(x)) for x in (xp_t or [])])


def series(m: dict) -> dict | None:
    """Поминутные величины одного матча: то, что доступно и офлайн, и в лайве.

    Возвращает None, если матч разобран не полностью. Молча подставлять
    нули нельзя: ноль смертей и «смерти не разобраны» — разные вещи, и
    вторая перекосила бы подгонку в пользу длинных матчей.
    """
    pls = m.get("players") or []
    if len(pls) != 10:
        return None
    ga = m.get("radiant_gold_adv") or []
    if len(ga) < 5:
        return None
    n = len(ga)
    mn = np.arange(n, dtype=float)
    rad = [p for p in pls if p.get("isRadiant")]
    dire = [p for p in pls if not p.get("isRadiant")]

    def deaths_of(side, killers) -> np.ndarray:
        """Смерти стороны `side` — это фраги её противников со временем."""
        out = np.zeros(n)
        for p in killers:
            for k in (p.get("kills_log") or []):
                t = k.get("time")
                if t is None:
                    continue
                i = int(t) // 60
                if 0 <= i < n:
                    out[i] += 1
        return np.cumsum(out)

    def buybacks_of(side) -> tuple[np.ndarray, np.ndarray]:
        cnt = np.zeros(n)
        lvl = np.zeros(n)
        for p in side:
            lv = _levels(p.get("xp_t") or [])
            for b in (p.get("buyback_log") or []):
                t = b.get("time")
                if t is None:
                    continue
                i = int(t) // 60
                if 0 <= i < n:
                    cnt[i] += 1
                    lvl[i] += float(lv[i]) if i < len(lv) else float(p.get("level") or 0)
        return np.cumsum(cnt), np.cumsum(lvl)

    def levels_of(side) -> np.ndarray:
        acc = np.zeros(n)
        for p in side:
            lv = _levels(p.get("xp_t") or [])
            if len(lv) == 0:
                return None
            if len(lv) < n:
                lv = np.concatenate([lv, np.full(n - len(lv), lv[-1])])
            acc += lv[:n]
        return acc

    lv_r, lv_d = levels_of(rad), levels_of(dire)
    if lv_r is None or lv_d is None:
        return None
    bb_r, bblv_r = buybacks_of(rad)
    bb_d, bblv_d = buybacks_of(dire)
    return {
        "match_id": int(m["match_id"]),
        "minute": mn,
        "g_adv": np.asarray(ga, dtype=float),
        "d_adv": deaths_of(rad, dire) - deaths_of(dire, rad),
        "bb_adv": bb_r - bb_d,
        "bblv_adv": bblv_r - bblv_d,
        "lv_adv": lv_r - lv_d,
        # Известна только конечная точка — на ней и подгоняем.
        "nw_end": (sum(float(p.get("net_worth") or 0) for p in rad)
                   - sum(float(p.get("net_worth") or 0) for p in dire)),
    }


def design(s: dict, idx, use_const: bool = False) -> np.ndarray:
    """Матрица признаков поправки на выбранных минутах."""
    cols = [np.asarray(s[k])[idx] for k in TERMS]
    if use_const:
        cols.insert(0, np.ones_like(np.asarray(s["minute"])[idx]))
    return np.column_stack(cols)


def fit(rows: list[dict], ridge: float = 1e-3,
        use_const: bool = False) -> dict:
    """Подгонка на КОНЦАХ матчей: там нетворс известен точно."""
    X, y = [], []
    for s in rows:
        last = len(s["minute"]) - 1
        X.append(design(s, [last], use_const)[0])
        y.append(s["nw_end"] - s["g_adv"][last])
    X = np.asarray(X)
    y = np.asarray(y)
    A = X.T @ X + ridge * np.eye(X.shape[1]) * len(X)
    beta = np.linalg.solve(A, X.T @ y)
    pred = X @ beta
    res = y - pred
    ss = 1.0 - float(np.sum(res ** 2) / np.sum((y - y.mean()) ** 2))
    return {"beta": [float(b) for b in beta],
            "terms": list(TERMS_WITH_CONST if use_const else TERMS),
            "use_const": bool(use_const),
            "r2": ss, "n": int(len(y)),
            "gap_abs_median": float(np.median(np.abs(y))),
            "res_abs_median": float(np.median(np.abs(res))),
            "res_abs_p90": float(np.percentile(np.abs(res), 90))}


def correct(beta, s: dict, idx=None, use_const: bool = False) -> np.ndarray:
    """Восстановленная разница нетворса на выбранных минутах."""
    if idx is None:
        idx = np.arange(len(s["minute"]))
    return (np.asarray(s["g_adv"])[idx]
            + design(s, idx, use_const) @ np.asarray(beta))


# --- проверка посреди матча ----------------------------------------------

def live_pairs(log_dir: Path | None = None) -> list[tuple[Path, Path]]:
    """Матчи, у которых есть и живой лог, и разобранный офлайн-разбор."""
    ld = log_dir or (config.DATA_DIR / "live_log")
    res = ld / "resolved"
    out = []
    for csv_path in sorted(ld.glob("*.csv")):
        mid = csv_path.stem.split("__")[0]
        j = res / f"{mid}.json"
        if j.exists():
            out.append((csv_path, j))
    return out


def read_live(path: Path) -> list[tuple[float, float]]:
    """(минута, нетворс_adv) из живого лога. Пустые строки пропускаем."""
    raw = path.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    out = []
    for r in csv.DictReader(raw.decode(enc).splitlines()):
        try:
            mn = float(r.get("minute") or "")
            nw = float(r.get("nw_adv") or "")
        except ValueError:
            continue
        out.append((mn, nw))
    return out


def live_rows(verbose: bool = True) -> list[dict]:
    """Строки с НАСТОЯЩИМ нетворсом посреди матча.

    Единственный источник такой правды: матчи, записанные живьём (нетворс
    поминутно из GetRealtimeStats) и потом добранные с OpenDota (добытое
    золото поминутно). Их мало — но подгонять поправку, которая работает
    в середине матча, больше не на чем: в офлайн-разборе нетворс есть
    только в конечной точке.
    """
    out = []
    for csv_path, jpath in live_pairs():
        try:
            m = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        s = series(m)
        if s is None:
            continue
        n = len(s["minute"])
        seen = set()
        rows = []
        for mn, nw in read_live(csv_path):
            i = int(round(mn))
            if not (0 <= i < n) or i in seen:
                continue                       # одна строка на минуту
            seen.add(i)
            rows.append({"i": i, "nw": nw})
        if len(rows) < 5:
            continue
        out.append({"s": s, "rows": rows, "file": csv_path.name})
        if verbose:
            print(f"    {csv_path.name:<34} минут {len(rows):>4}")
    return out


def sanity_endpoint(groups, verbose: bool = True) -> None:
    """Одна ли это величина вообще: лайв-нетворс и офлайн-нетворс.

    Если живой `nw_adv` в конце записи и офлайновая разница `net_worth`
    расходятся на порядок, то сравнивать их поминутно бессмысленно, и
    выяснить это надо ДО подгонки, а не после.
    """
    if not verbose:
        return
    print("\n  Одна ли это величина. Сверять можно ТОЛЬКО те записи, что")
    print("  дошли до конца матча: у оборванных лайв показывает середину, а")
    print("  офлайн — финал, и расхождение там ничего не значит.")
    print(f"    {'матч':>12}{'посл. мин.':>12}{'мин. в матче':>14}"
          f"{'лайв nw_adv':>14}{'офлайн финал':>14}{'':>10}")
    ok, tot = 0, 0
    for g in groups:
        last = max(g["rows"], key=lambda r: r["i"])
        end = len(g["s"]["minute"]) - 1
        full = last["i"] >= end - 1
        mark = ""
        if full:
            tot += 1
            d = abs(last["nw"] - g["s"]["nw_end"])
            ok += d < 0.15 * max(1.0, abs(g["s"]["nw_end"]))
            mark = f"расх. {d:.0f}"
        else:
            mark = "оборвана"
        print(f"    {g['s']['match_id']:>12}{last['i']:>12}{end:>14}"
              f"{last['nw']:>14.0f}{g['s']['nw_end']:>14.0f}  {mark}")
    print(f"    записей до конца матча: {tot}, из них сходятся в пределах "
          f"15%: {ok}")


def fit_live_loo(groups, verbose: bool = True) -> dict:
    """Подгонка на живых строках с проверкой «выкинь один матч».

    Пять коэффициентов на полторы тысячи строк — параметров хватает с
    запасом, но строки внутри матча скоррелированы, поэтому проверка идёт
    ЦЕЛЫМИ МАТЧАМИ: обучаемся на тринадцати, проверяем на четырнадцатом, и
    так по кругу. Иначе вышла бы обычная подгонка под себя.
    """
    per = []
    raw_all, cor_all = [], []
    for k, g in enumerate(groups):
        X, y = [], []
        for other in groups:
            if other is g:
                continue
            idx = [r["i"] for r in other["rows"]]
            X.append(design(other["s"], idx))
            y.append(np.array([r["nw"] for r in other["rows"]])
                     - np.asarray(other["s"]["g_adv"])[idx])
        X = np.vstack(X)
        y = np.concatenate(y)
        A = X.T @ X + 1e-3 * np.eye(X.shape[1]) * len(X)
        beta = np.linalg.solve(A, X.T @ y)
        idx = [r["i"] for r in g["rows"]]
        nw = np.array([r["nw"] for r in g["rows"]])
        raw = np.asarray(g["s"]["g_adv"])[idx] - nw
        cor = correct(beta, g["s"], idx) - nw
        per.append({"match_id": g["s"]["match_id"], "n": len(idx),
                    "raw": float(np.median(np.abs(raw))),
                    "cor": float(np.median(np.abs(cor)))})
        raw_all.append(raw)
        cor_all.append(cor)
    raw_all = np.abs(np.concatenate(raw_all))
    cor_all = np.abs(np.concatenate(cor_all))
    # Итоговые коэффициенты — на всех матчах сразу; проверка выше уже
    # сказала, чего они стоят на невиданном матче.
    X, y = [], []
    for g in groups:
        idx = [r["i"] for r in g["rows"]]
        X.append(design(g["s"], idx))
        y.append(np.array([r["nw"] for r in g["rows"]])
                 - np.asarray(g["s"]["g_adv"])[idx])
    X, y = np.vstack(X), np.concatenate(y)
    A = X.T @ X + 1e-3 * np.eye(X.shape[1]) * len(X)
    beta = np.linalg.solve(A, X.T @ y)
    if verbose:
        print(f"\n  {'матч':>12}{'минут':>7}{'|ошибка| как есть':>20}"
              f"{'с поправкой':>14}")
        print("  " + "-" * 55)
        for r in per:
            mark = "лучше" if r["cor"] < r["raw"] else "ХУЖЕ"
            print(f"  {r['match_id']:>12}{r['n']:>7}{r['raw']:>20.0f}"
                  f"{r['cor']:>14.0f}  {mark}")
    return {
        "beta": [float(b) for b in beta], "terms": list(TERMS),
        "use_const": False,
        "n_matches": len(groups), "n_rows": int(len(raw_all)),
        "raw_median": float(np.median(raw_all)),
        "cor_median": float(np.median(cor_all)),
        "raw_p90": float(np.percentile(raw_all, 90)),
        "cor_p90": float(np.percentile(cor_all, 90)),
        "better_matches": sum(1 for r in per if r["cor"] < r["raw"]),
        "per_match": per,
    }


def check_live(beta, verbose: bool = True, use_const: bool = False) -> dict:
    """Держится ли поправка ПОСРЕДИ матча, а не только в конце.

    Это единственная независимая проверка: коэффициенты подогнаны на
    конечных точках, а здесь сверяются с настоящим нетворсом на каждой
    минуте записанных живьём матчей.
    """
    pairs = live_pairs()
    per_match = []
    raw_all, cor_all = [], []
    for csv_path, jpath in pairs:
        try:
            m = json.loads(jpath.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        s = series(m)
        if s is None:
            continue
        live = read_live(csv_path)
        if len(live) < 5:
            continue
        n = len(s["minute"])
        raw_e, cor_e = [], []
        for mn, nw in live:
            i = int(round(mn))
            if not (0 <= i < n):
                continue
            raw_e.append(s["g_adv"][i] - nw)
            cor_e.append(float(correct(beta, s, [i], use_const)[0]) - nw)
        if len(raw_e) < 5:
            continue
        per_match.append({
            "match_id": s["match_id"], "n": len(raw_e),
            "raw": float(np.median(np.abs(raw_e))),
            "cor": float(np.median(np.abs(cor_e))),
        })
        raw_all += raw_e
        cor_all += cor_e
    if not raw_all:
        return {"n_matches": 0}
    raw_all = np.abs(np.asarray(raw_all))
    cor_all = np.abs(np.asarray(cor_all))
    if verbose:
        print(f"  {'матч':>12}{'строк':>7}{'|ошибка| как есть':>20}"
              f"{'с поправкой':>14}{'':>4}")
        print("  " + "-" * 58)
        for r in per_match:
            mark = "лучше" if r["cor"] < r["raw"] else "ХУЖЕ"
            print(f"  {r['match_id']:>12}{r['n']:>7}{r['raw']:>20.0f}"
                  f"{r['cor']:>14.0f}  {mark}")
    better = sum(1 for r in per_match if r["cor"] < r["raw"])
    return {
        "n_matches": len(per_match), "n_rows": int(len(raw_all)),
        "raw_median": float(np.median(raw_all)),
        "cor_median": float(np.median(cor_all)),
        "raw_p90": float(np.percentile(raw_all, 90)),
        "cor_p90": float(np.percentile(cor_all, 90)),
        "better_matches": better,
        "per_match": per_match,
    }


# --- сбор ----------------------------------------------------------------

def load_series(limit: int | None, holdout: bool = False,
                verbose: bool = True) -> list[dict]:
    files = sorted(config.RAW_MATCHES_DIR.glob("*.json"))
    files = [f for f in files if f.stem.isdigit()
             and (HO.is_holdout(f.stem) == holdout)]
    if limit and limit < len(files):
        step = len(files) / limit
        files = [files[int(i * step)] for i in range(limit)]
    out, bad = [], 0
    t0 = time.time()
    for i, f in enumerate(files):
        try:
            s = series(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError, ValueError, KeyError):
            s = None
        if s is None:
            bad += 1
        else:
            out.append(s)
        if verbose and (i + 1) % 500 == 0:
            print(f"    прочитано {i + 1}/{len(files)}  "
                  f"({time.time() - t0:.0f} с)")
    if verbose:
        print(f"  разобрано матчей: {len(out)}, пропущено {bad}, "
              f"{time.time() - t0:.0f} с")
    return out


def load(path: Path | None = None) -> dict | None:
    p = path or OUT_JSON
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) and d.get("beta") else None


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Восстановление командной разницы нетворса из добытого золота.")
    ap.add_argument("--fit", action="store_true", help="подогнать коэффициенты")
    ap.add_argument("--check-live", action="store_true",
                    help="проверить поправку посреди матча по живым логам")
    ap.add_argument("--fit-live", action="store_true",
                    help="подогнать ПО живым логам, проверка «выкинь матч»")
    ap.add_argument("--n", type=int, default=3000,
                    help="сколько матчей читать для подгонки")
    ap.add_argument("--out", type=Path, default=OUT_JSON)
    args = ap.parse_args(argv)
    if not (args.fit or args.check_live or args.fit_live):
        args.fit = args.check_live = True

    got = load(args.out)
    print("=" * 78)
    if args.fit_live:
        print("ПОДГОНКА ПО ЖИВЫМ ЛОГАМ — там нетворс известен на каждой минуте")
        print("=" * 78)
        groups = live_rows()
        if len(groups) < 5:
            print(f"ОШИБКА: пар «живой лог + разбор» всего {len(groups)}, "
                  f"подгонять не на чем.", file=sys.stderr)
            return 2
        sanity_endpoint(groups)
        r = fit_live_loo(groups)
        print(f"\n  Всего: {r['n_matches']} матчей, {r['n_rows']} минут")
        print(f"  |ошибка| медиана : {r['raw_median']:.0f} -> "
              f"{r['cor_median']:.0f} золота")
        print(f"  |ошибка| p90     : {r['raw_p90']:.0f} -> {r['cor_p90']:.0f}")
        print(f"  матчей, где стало лучше: {r['better_matches']} из "
              f"{r['n_matches']}")
        print("\n  коэффициенты (на всех матчах):")
        for t, b in zip(r["terms"], r["beta"]):
            print(f"    {t:<10}{b:>12.2f}")
        win = (r["cor_median"] < r["raw_median"]
               and r["better_matches"] * 2 > r["n_matches"])
        print("\n  ВЫВОД: " + (
            f"поправка работает на невиданном матче. Но матчей всего "
            f"{r['n_matches']}, и это\n  паблики: переносить на про-игры без "
            f"новой записи нельзя."
            if win else
            "поправка НЕ переносится на невиданный матч."))
        got = dict(got or {}, live_fit=r)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(got, ensure_ascii=False, indent=1,
                                       default=float), encoding="utf-8")
        print(f"\n  записано: {args.out}")
        print("=" * 78)
        if not (args.fit or args.check_live):
            return 0
    if args.fit:
        print("ПОДГОНКА на концах матчей (нетворс там известен точно)")
        print("=" * 78)
        rows = load_series(args.n)
        if len(rows) < 200:
            print("ОШИБКА: матчей слишком мало.", file=sys.stderr)
            return 2
        # Половина на подгонку, половина на замер — иначе R² будет про
        # то, как хорошо мы запомнили, а не про то, как хорошо считаем.
        half = len(rows) // 2
        yte = np.asarray([s["nw_end"] - s["g_adv"][-1] for s in rows[half:]])
        print(f"\n  матчей: {len(rows)}  (подгонка {half}, замер {len(rows) - half})")
        print(f"\n  зазор без поправки: |медиана| {np.median(np.abs(yte)):.0f}"
              f"  p90 {np.percentile(np.abs(yte), 90):.0f}")
        print(f"\n  {'вариант':<22}{'R² на замере':>14}{'|медиана| ост.':>16}"
              f"{'p90 ост.':>11}")
        print("  " + "-" * 63)
        variants = {}
        for uc, label in ((False, "через ноль (в бою)"),
                          (True, "со свободным членом")):
            f_tr = fit(rows[:half], use_const=uc)
            Xte = np.asarray([design(s, [len(s["minute"]) - 1], uc)[0]
                              for s in rows[half:]])
            res = yte - Xte @ np.asarray(f_tr["beta"])
            r2 = 1.0 - float(np.sum(res ** 2) / np.sum((yte - yte.mean()) ** 2))
            print(f"  {label:<22}{r2:>14.3f}{np.median(np.abs(res)):>16.0f}"
                  f"{np.percentile(np.abs(res), 90):>11.0f}")
            variants[uc] = {"r2": r2, "res_median": float(np.median(np.abs(res))),
                            "fit": fit(rows, use_const=uc)}
        print("\n  «Со свободным членом» подогнан лучше в конечной точке, но на\n"
              "  нулевой минуте утверждает, что зазор уже ненулевой, — а он там\n"
              "  ровно ноль по определению. В бой идёт вариант через ноль;\n"
              "  второй оставлен, чтобы разницу было видно, а не помнить.")
        f_all = variants[False]["fit"]
        print(f"\n  коэффициенты варианта через ноль (на всех {len(rows)}):")
        for t, b in zip(f_all["terms"], f_all["beta"]):
            print(f"    {t:<10}{b:>12.2f}")
        got = {"beta": f_all["beta"], "terms": f_all["terms"],
               "use_const": False, "fit": f_all,
               "r2_holdout_half": variants[False]["r2"],
               "r2_with_const": variants[True]["r2"],
               "res_abs_median_te": variants[False]["res_median"],
               "gap_abs_median_te": float(np.median(np.abs(yte))),
               "n_matches": len(rows), "built_at": time.time()}

    if args.check_live:
        if not got:
            print("ОШИБКА: коэффициентов нет, сначала --fit", file=sys.stderr)
            return 2
        print("\n" + "=" * 78)
        print("ПРОВЕРКА ПОСРЕДИ МАТЧА — записанные живьём матчи")
        print("=" * 78)
        print("  Коэффициенты подогнаны на КОНЦАХ матчей. Здесь они сверяются")
        print("  с настоящим нетворсом на каждой минуте — это единственная")
        print("  независимая проверка, что поправка держится не только в конце.\n")
        ch = check_live(got["beta"], use_const=bool(got.get("use_const")))
        if not ch.get("n_matches"):
            print("  Пар «живой лог + разобранный матч» не нашлось.")
        else:
            print(f"\n  Всего: {ch['n_matches']} матчей, {ch['n_rows']} строк")
            print(f"  |ошибка| медиана : {ch['raw_median']:.0f} -> "
                  f"{ch['cor_median']:.0f} золота")
            print(f"  |ошибка| p90     : {ch['raw_p90']:.0f} -> "
                  f"{ch['cor_p90']:.0f}")
            print(f"  матчей, где стало лучше: {ch['better_matches']} из "
                  f"{ch['n_matches']}")
            win = ch["cor_median"] < ch["raw_median"]
            print("\n  ВЫВОД: " + (
                "поправка держится и посреди матча." if win else
                "поправка НЕ держится посреди матча — подгонка на концах "
                "туда не переносится."))
            got["live_check"] = ch

    if got:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(got, ensure_ascii=False, indent=1,
                                       default=float), encoding="utf-8")
        print(f"\n  записано: {args.out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
