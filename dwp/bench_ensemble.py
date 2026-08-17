"""Замер ансамбля: несколько сидов против одиночной модели.

ЗАЧЕМ. Разброс log loss между полностью переобученными моделями,
отличающимися ТОЛЬКО сидом разбиения, — 0.027. Весь драфт даёт 0.0133,
сборки 0.0012, урон 0.0003. То есть одиночная модель — это розыгрыш
лотереи с размахом больше любого измеренного эффекта, и усреднение
убирает розыгрыш, ничего не зная про предметную область.

КРИТЕРИЙ ПРИЁМКИ — не «ансамбль лучше лучшей модели», а **лучше СРЕДНЕЙ**.
Лучшую заранее выбрать нельзя: в этом и смысл. Поэтому здесь считается
разность «log loss ансамбля минус СРЕДНИЙ log loss участников», и
интервал к ней — парным бутстрапом по матчам, на одних и тех же
пересэмплированных матчах для всех моделей сразу.

ГДЕ МЕРЯЕТСЯ. На слепом холдауте (15% матчей, спрятанных от обучения по
хэшу match_id). Он один и тот же у всех сидов — от сида зависит только
разбиение ОСТАВШИХСЯ матчей, — поэтому сравнение честное. Каждый запуск
дописывается в реестр вскрытий `data/blind/registry.csv`: холдаут тем
слепее, чем реже туда смотрят.

    python -m dwp.bench_ensemble --models models\\ens_*.pkl
    python -m dwp.bench_ensemble --models models\\ens_*.pkl --n 400
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
from .compare import _predict_single, EPS
from .train import ece, logit


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def load_holdout(limit: int | None = None, verbose: bool = True) -> list[dict]:
    """Только спрятанные матчи. Читать все 7734 файла ради 15% незачем."""
    files = sorted(p for p in config.RAW_MATCHES_DIR.glob("*.json")
                   if p.stem.isdigit() and HO.is_holdout(p.stem))
    if limit:
        files = files[:limit]
    out, bad = [], 0
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            bad += 1
    if verbose and bad:
        print(f"  битых файлов пропущено: {bad}")
    return F.usable_matches(out, verbose=verbose)


def frames_once(matches: list[dict]) -> tuple[pd.DataFrame, dict]:
    """Кадры БЕЗ draft_logit плюс разобранные objectives.

    Кадр от модели не зависит — зависит только `draft_logit`. Считать его
    заново на каждую модель значило бы платить пятикратно за одно и то же:
    на 1160 матчах это минуты.
    """
    frames, parsed = [], {}
    for m in matches:
        mid = int(m["match_id"])
        parsed[mid] = F.parse_objectives(m)
        frames.append(F.match_state_frame(m, parsed[mid]))
    return pd.concat(frames, ignore_index=True), parsed


def draft_logit_by_match(art: dict, matches: list[dict]) -> dict[int, float]:
    out = {}
    for m in matches:
        mid = int(m["match_id"])
        elo = art["elo_pre"].get(mid, art["elo_pre"].get(str(mid), 0.0))
        Xd, _, _ = F.draft_matrix([m], art["id2idx"], {mid: elo})
        p = float(art["draft_model"].predict_proba(Xd)[0, 1])
        out[mid] = float(logit(np.array([p]))[0])
    return out


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def bootstrap_gap(y, ps: list[np.ndarray], p_ens: np.ndarray, groups,
                  n_boot: int = 2000, seed: int = 0) -> dict:
    """Интервал разности «ансамбль минус средний участник» по матчам."""
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    deltas = np.empty(n_boot)
    best = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        singles = [_ll(y[rows], p[rows]) for p in ps]
        e = _ll(y[rows], p_ens[rows])
        deltas[b] = e - float(np.mean(singles))
        best[b] = e - float(np.min(singles))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    blo, bhi = np.percentile(best, [2.5, 97.5])
    return {
        "delta": float(np.mean(deltas)), "lo": float(lo), "hi": float(hi),
        "p_better": float(np.mean(deltas < 0)),
        "vs_best_delta": float(np.mean(best)), "vs_best_lo": float(blo),
        "vs_best_hi": float(bhi), "vs_best_p": float(np.mean(best < 0)),
    }


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Ансамбль по сидам против одиночной модели на слепом холдауте.")
    ap.add_argument("--models", type=Path, nargs="+",
                    default=[config.MODELS_DIR / "ens_*.pkl"],
                    help="артефакты ансамбля; шаблоны раскрываются сами")
    ap.add_argument("--n", type=int, default=None,
                    help="ограничить число матчей холдаута (для быстрой проверки)")
    ap.add_argument("--baseline", type=Path, nargs="+", default=None,
                    help="второй набор моделей: тогда считается ещё и парное "
                         "сравнение двух АНСАМБЛЕЙ на одних и тех же матчах "
                         "(например, разнородный против набора по сидам)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-registry", action="store_true",
                    help="не записывать вскрытие в реестр (только для отладки)")
    args = ap.parse_args(argv)

    try:
        paths = live.resolve_models(args.models)
        arts = [live.load_model(p) for p in paths]
    except live.LiveError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    if len(arts) < 2:
        print("ОШИБКА: нужно минимум две модели — иначе усреднять нечего.\n"
              "Что делать: обучите набор, меняя только --seed и --out:\n"
              "  python -m dwp.train --source real --live-features "
              "--extra-features --exact-features --gold-norm add "
              "--seed 3 --out models\\ens_03.pkl", file=sys.stderr)
        return 2
    for p, a in zip(paths, arts):
        if not a.get("holdout"):
            print(f"ОШИБКА: {p.name} обучена БЕЗ слепого холдаута "
                  f"(holdout={a.get('holdout')!r}).\nМерить её на холдауте "
                  f"нельзя: эти матчи она видела.\nЧто делать: переобучите без "
                  f"флага --no-holdout.", file=sys.stderr)
            return 2
    feats = arts[0]["state_features"]
    for p, a in zip(paths[1:], arts[1:]):
        if a["state_features"] != feats:
            print(f"ОШИБКА: у {p.name} другой набор признаков — усреднять "
                  f"нельзя.", file=sys.stderr)
            return 2

    t0 = time.time()
    print("=" * 74)
    print(f"Моделей: {len(arts)} — {', '.join(p.stem for p in paths)}")
    print(f"Признаков: {len(feats)}")
    matches = load_holdout(args.n, verbose=True)
    if not matches:
        print("ОШИБКА: холдаут пуст. `python -m dwp.holdout` покажет, сколько "
              "матчей спрятано.", file=sys.stderr)
        return 2
    print(f"Слепой холдаут: {len(matches)} матчей "
          f"(правило sha1(match_id) mod 1000 < {HO.HOLDOUT_PERMILLE})")
    df, _parsed = frames_once(matches)
    y = df["y"].to_numpy()
    g = df["match_id"].to_numpy()
    print(f"Строк: {len(df)}   разбор занял {time.time() - t0:.0f} с")
    print("=" * 74)

    ps = []
    for p, a in zip(paths, arts):
        dl = draft_logit_by_match(a, matches)
        d = df.copy()
        d["draft_logit"] = d["match_id"].map(dl).astype(float)
        ps.append(_predict_single(a, d))
    p_ens = np.mean(ps, axis=0)

    print(f"\n  {'модель':<18}{'log_loss':>10}{'brier':>9}{'ECE':>8}")
    print("  " + "-" * 43)
    for p, pi in zip(paths, ps):
        print(f"  {p.stem[:17]:<18}{_ll(y, pi):>10.4f}{_brier(y, pi):>9.4f}"
              f"{ece(y, pi, g):>8.4f}")
    mean_ll = float(np.mean([_ll(y, pi) for pi in ps]))
    mean_br = float(np.mean([_brier(y, pi) for pi in ps]))
    print("  " + "-" * 43)
    print(f"  {'среднее по одной':<18}{mean_ll:>10.4f}{mean_br:>9.4f}"
          f"{'':>8}")
    print(f"  {'АНСАМБЛЬ':<18}{_ll(y, p_ens):>10.4f}{_brier(y, p_ens):>9.4f}"
          f"{ece(y, p_ens, g):>8.4f}")

    spread = float(np.max([_ll(y, pi) for pi in ps])
                   - np.min([_ll(y, pi) for pi in ps]))
    print(f"\n  Разброс log loss между сидами: {spread:.4f}")
    print("  Для масштаба: весь драфт даёт 0.0133 над базой, сборки 0.0012.")

    r = bootstrap_gap(y, ps, p_ens, g, n_boot=args.n_boot, seed=args.seed)
    print(f"\n  Ансамбль минус СРЕДНЯЯ одиночная: {r['delta']:+.4f}   "
          f"95% [{r['lo']:+.4f}, {r['hi']:+.4f}]   P(лучше) {r['p_better']:.3f}")
    print(f"  Ансамбль минус ЛУЧШАЯ одиночная:  {r['vs_best_delta']:+.4f}   "
          f"95% [{r['vs_best_lo']:+.4f}, {r['vs_best_hi']:+.4f}]   "
          f"P(лучше) {r['vs_best_p']:.3f}")
    print()
    if r["hi"] < 0:
        print("  ВЫВОД: ансамбль лучше средней одиночной ЗНАЧИМО — весь "
              "интервал ниже нуля.")
        print("  Это и есть критерий приёмки: лучшую одиночную заранее выбрать "
              "нельзя.")
    elif r["lo"] <= 0 <= r["hi"]:
        print("  ВЫВОД: разница НЕ значима — интервал накрывает ноль.")
    else:
        print("  ВЫВОД: ансамбль ХУЖЕ средней одиночной значимо. Это "
              "неожиданный результат,")
        print("  и записать его надо именно так, а не спрятать.")

    if args.baseline:
        print("\n--- Ансамбль против ансамбля ---")
        try:
            b_paths = live.resolve_models(args.baseline)
            b_arts = [live.load_model(p) for p in b_paths]
        except live.LiveError as e:
            print(f"  базовый набор не загрузился: {e}", file=sys.stderr)
            return 2
        b_ps = []
        for p, a in zip(b_paths, b_arts):
            if a["state_features"] != feats:
                print(f"  ОШИБКА: у {p.name} другой набор признаков — "
                      f"сравнивать нельзя.", file=sys.stderr)
                return 2
            dl = draft_logit_by_match(a, matches)
            d = df.copy()
            d["draft_logit"] = d["match_id"].map(dl).astype(float)
            b_ps.append(_predict_single(a, d))
        p_base = np.mean(b_ps, axis=0)
        print(f"  база:    {len(b_arts)} моделей "
              f"({', '.join(p.stem for p in b_paths)})")
        print(f"  проверка: {len(arts)} моделей "
              f"({', '.join(p.stem for p in paths)})")
        print(f"\n  {'набор':<12}{'log_loss':>10}{'brier':>9}{'ECE':>8}")
        print("  " + "-" * 39)
        print(f"  {'база':<12}{_ll(y, p_base):>10.4f}{_brier(y, p_base):>9.4f}"
              f"{ece(y, p_base, g):>8.4f}")
        print(f"  {'проверка':<12}{_ll(y, p_ens):>10.4f}{_brier(y, p_ens):>9.4f}"
              f"{ece(y, p_ens, g):>8.4f}")
        pair = bootstrap_gap(y, [p_base], p_ens, g, n_boot=args.n_boot,
                             seed=args.seed)
        print(f"\n  проверка минус база: {pair['delta']:+.4f}   "
              f"95% [{pair['lo']:+.4f}, {pair['hi']:+.4f}]   "
              f"P(лучше) {pair['p_better']:.3f}")
        if pair["hi"] < 0:
            print("  ВЫВОД: проверяемый набор лучше значимо.")
        elif pair["lo"] <= 0 <= pair["hi"]:
            print("  ВЫВОД: разница НЕ значима — интервал накрывает ноль.")
            print("  Значит усложнение набора себя не окупает, и это надо "
                  "записать.")
        else:
            print("  ВЫВОД: проверяемый набор ХУЖЕ значимо.")

    sha = hashlib.sha256(np.ascontiguousarray(p_ens).tobytes()).hexdigest()
    if not args.no_registry:
        name = "ансамбль(" + "+".join(p.stem for p in paths) + ")"
        n = HO.note_reveal(
            name[:60], "holdout", len(np.unique(g)), len(y), sha,
            {"log_loss": _ll(y, p_ens), "brier": _brier(y, p_ens),
             "ece": ece(y, p_ens, g)},
            note=f"bench_ensemble, {len(arts)} моделей")
        print(f"\n  Вскрытие холдаута записано в реестр (по этой модели их "
              f"уже {n}).")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
