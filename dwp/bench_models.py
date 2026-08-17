"""Все модели из `models/` на слепом холдауте: какие держать, какие убрать.

ЗАЧЕМ ОТДЕЛЬНО ОТ `bench_ensemble`. Тот отвечает на один вопрос — «лучше
ли ансамбль средней своей одиночной». Здесь вопрос другой и приземлённее:
в папке лежит четырнадцать артефактов, часть из них — остатки замеров, и
надо знать, какие из них вообще чего-то стоят, чтобы остальные убрать.

ЧТО ЗДЕСЬ ВАЖНО НЕ ПЕРЕПУТАТЬ. Не всякую модель можно померить на
холдауте. Артефакт, обученный БЕЗ `--holdout` (ключ `holdout` не True),
эти матчи видел, и его log loss на них — не оценка качества, а оценка
памяти. Такие модели считаются отдельным списком и подписываются
«видела эти матчи», а в общий рейтинг не попадают. Молча смешать их с
остальными значило бы объявить лучшей ту, что просто запомнила ответы.

ОДИН ВЗГЛЯД НА ВСЕХ. Холдаут тем слепее, чем реже туда смотрят
(`dwp.holdout`). Поэтому здесь все модели считаются за один проход и
пишется ОДНА строка в реестр вскрытий, а не по строке на модель.

ПОБОЧНЫЙ ПРОДУКТ — кэш предсказаний `data/holdout_preds.npz`. Поминутные
вероятности каждой модели на каждом матче холдаута. Он нужен затем, чтобы
настройку вердикта (`dwp.verdict`) не пришлось оплачивать новым проходом
инференса: развилок там много, а вероятности от них не зависят.

    python -m dwp.bench_models                 # все модели из models/
    python -m dwp.bench_models --n 200         # быстрая проверка
    python -m dwp.bench_models --models models\\ens_*.pkl models\\div_*.pkl
    python -m dwp.bench_models --no-registry   # не тратить вскрытие (отладка)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path

import numpy as np

from . import config, holdout as HO, live
from .bench_ensemble import (draft_logit_by_match, load_holdout, frames_once,
                             _ll, _brier)
from .compare import _predict_single
from .train import ece

PRED_CACHE = config.DATA_DIR / "holdout_preds.npz"

# Отрезки минут для разбора точности по фазам. Те же границы, что у
# коридора (`forecast.MINUTE_BINS`), укрупнённые до трёх: до 15-й минуты
# лежит половина всей ошибки, и мельчить внутри неё нечем.
PHASE_BINS = ((0, 10), (10, 15), (15, 20), (20, 30), (30, 200))


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _acc(y: np.ndarray, p: np.ndarray) -> float:
    """Доля угаданных победителей. Ничьих в Dota нет, порог 0.5.

    Печатается ТРЕТЬЕЙ колонкой, после log loss и Brier, — по требованию
    README: accuracy не видит разницы между «уверен на 51%» и «уверен на
    99%», и выбирать модель по ней нельзя. Но именно её человек читает
    как «сколько раз модель угадала», поэтому прятать её тоже неправильно.
    """
    return float(np.mean((p >= 0.5) == (y == 1)))


def _phase_acc(y: np.ndarray, p: np.ndarray, minute: np.ndarray) -> dict:
    out = {}
    for lo, hi in PHASE_BINS:
        sel = (minute >= lo) & (minute < hi)
        n = int(sel.sum())
        out[f"{lo}-{hi}"] = (_acc(y[sel], p[sel]) if n else float("nan"), n)
    return out


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал для доли. Нужен затем, что «68%» без интервала — не число.

    Не бутстрап: здесь доля по НЕЗАВИСИМЫМ наблюдениям (один матч — одно
    наблюдение), и Вилсон точнее нормального приближения на краях.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = k / n
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / d
    return (float(max(0.0, c - half)), float(min(1.0, c + half)))


def paired_vs_reference(y: np.ndarray, groups: np.ndarray,
                        p_ref: np.ndarray, ps: dict[str, np.ndarray],
                        n_boot: int = 2000, seed: int = 0) -> dict[str, dict]:
    """Разность «модель минус эталон» по log loss, бутстрапом ПО МАТЧАМ.

    Строки внутри матча скоррелированы: сорок минут одного матча — это не
    сорок независимых наблюдений. Пересэмплирование идёт по match_id,
    целыми матчами, и одни и те же пересэмплированные матчи используются
    для ВСЕХ моделей сразу — иначе разности между ними окажутся шумом
    разных розыгрышей, а не разностью моделей.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    idx = {g: np.flatnonzero(groups == g) for g in uniq}
    names = list(ps)
    deltas = {k: np.empty(n_boot) for k in names}
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx[g] for g in pick])
        ref = _ll(y[rows], p_ref[rows])
        for k in names:
            deltas[k][b] = _ll(y[rows], ps[k][rows]) - ref
    out = {}
    for k in names:
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        out[k] = {"delta": float(_ll(y, ps[k]) - _ll(y, p_ref)),
                  "lo": float(lo), "hi": float(hi),
                  "p_worse": float(np.mean(deltas[k] > 0))}
    return out


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Все модели из models/ на слепом холдауте: рейтинг и что убрать.")
    ap.add_argument("--models", type=Path, nargs="+", default=None,
                    help="что мерить; по умолчанию все models/*.pkl")
    ap.add_argument("--n", type=int, default=None,
                    help="ограничить число матчей холдаута (быстрая проверка)")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-registry", action="store_true",
                    help="не записывать вскрытие в реестр (только отладка)")
    ap.add_argument("--no-cache", action="store_true",
                    help="не сохранять кэш предсказаний для dwp.verdict")
    args = ap.parse_args(argv)

    paths = (live.resolve_models(args.models) if args.models
             else sorted(config.MODELS_DIR.glob("*.pkl")))
    if not paths:
        print("ОШИБКА: в models/ нет ни одного .pkl.\n"
              "Что делать: `python -m dwp.train --source real --live-features "
              "--extra-features --exact-features --gold-norm add "
              "--seed 3 --out models\\ens_03.pkl`", file=sys.stderr)
        return 2

    clean: list[tuple[Path, dict]] = []
    dirty: list[tuple[Path, dict]] = []
    broken: list[tuple[Path, str]] = []
    offline: list[str] = []
    for p in paths:
        try:
            art = live.load_model(p)
        except live.LiveError as e:
            # `load_model` — боевой загрузчик, и он отвергает модели, которые
            # в лайве работать не могут (например обученные по опыту:
            # GetRealtimeStats отдаёт только уровни). Здесь же мы считаем
            # ОФЛАЙН, на разобранных матчах, где эти признаки есть, — поэтому
            # такая модель мерится, но помечается: в бою она непригодна.
            try:
                import pickle
                art = pickle.loads(p.read_bytes())
                if not isinstance(art, dict) or "state_features" not in art:
                    raise ValueError("не артефакт модели")
                art.setdefault("name", p.name)
                art.setdefault("paths", [str(p)])
                offline.append(p.stem)
            except Exception:                                 # noqa: BLE001
                broken.append((p, str(e).splitlines()[0]))
                continue
        (clean if art.get("holdout") else dirty).append((p, art))

    print("=" * 78)
    print(f"Артефактов в {config.MODELS_DIR}: {len(paths)}")
    print(f"  меряемых на холдауте : {len(clean)}")
    print(f"  видели холдаут       : {len(dirty)}  (их числа — не оценка качества)")
    if offline:
        print(f"  только офлайн        : {len(offline)}  "
              f"({', '.join(offline)}) — боевой загрузчик их отвергает, "
              f"здесь считаются на разобранных матчах")
    if broken:
        print(f"  не грузятся          : {len(broken)}")
        for p, why in broken:
            print(f"      {p.name}: {why}")

    t0 = time.time()
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
    mn = df["minute"].to_numpy(dtype=float)
    print(f"Строк: {len(df)}   разбор занял {time.time() - t0:.0f} с")
    print("=" * 78)

    # --- предсказания -----------------------------------------------------
    def predict_all(items: list[tuple[Path, dict]]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for p, art in items:
            # draft_logit в общий кадр не кладётся намеренно: он у каждой
            # модели свой и подставляется ниже. Проверять его наличие здесь
            # значило бы отвергнуть все модели разом.
            missing = [f for f in art["state_features"]
                       if f not in df.columns and f != "draft_logit"]
            if missing:
                print(f"  {p.stem}: пропущена — в кадре нет колонок "
                      f"{', '.join(missing[:4])}")
                continue
            t = time.time()
            dl = draft_logit_by_match(art, matches)
            d = df.copy()
            d["draft_logit"] = d["match_id"].map(dl).astype(float)
            out[p.stem] = _predict_single(art, d)
            print(f"  {p.stem:<18} посчитана за {time.time() - t:.0f} с")
        return out

    print("\nСчитаю предсказания:")
    ps_clean = predict_all(clean)
    ps_dirty = predict_all(dirty)
    if not ps_clean:
        print("ОШИБКА: ни одной модели, обученной с холдаутом. Мерить нечем.\n"
              "Что делать: переобучите без флага --no-holdout.", file=sys.stderr)
        return 2

    # Боевой ансамбль — эталон сравнения: именно он сейчас в бою по
    # умолчанию (`live.default_models`), и вопрос «убрать ли модель»
    # означает «хуже ли она того, что уже работает».
    ens_names = sorted(k for k in ps_clean if k.startswith("ens_"))
    p_ens = (np.mean([ps_clean[k] for k in ens_names], axis=0)
             if len(ens_names) >= 2 else None)

    # --- таблица ----------------------------------------------------------
    def row(name: str, p: np.ndarray) -> str:
        ph = _phase_acc(y, p, mn)
        return (f"  {name[:22]:<23}{_ll(y, p):>9.4f}{_brier(y, p):>8.4f}"
                f"{ece(y, p, g):>8.4f}{_acc(y, p):>8.3f}"
                f"{ph['0-10'][0]:>9.3f}{ph['15-20'][0]:>9.3f}{ph['30-200'][0]:>9.3f}")

    head = (f"  {'модель':<23}{'log_loss':>9}{'brier':>8}{'ECE':>8}{'acc':>8}"
            f"{'acc 0-10':>9}{'acc15-20':>9}{'acc 30+':>9}")
    print("\n" + "=" * 78)
    print("МЕРЯЕМЫЕ МОДЕЛИ — обучены без этих матчей, число честное")
    print("=" * 78)
    print(head)
    print("  " + "-" * 74)
    order = sorted(ps_clean, key=lambda k: _ll(y, ps_clean[k]))
    for k in order:
        print(row(k, ps_clean[k]))
    if p_ens is not None:
        print("  " + "-" * 74)
        print(row(f"АНСАМБЛЬ({len(ens_names)}) в бою", p_ens))

    if ps_dirty:
        print("\n" + "=" * 78)
        print("ВИДЕЛИ ЭТИ МАТЧИ ПРИ ОБУЧЕНИИ — числа завышены, не сравнивать")
        print("=" * 78)
        print(head)
        print("  " + "-" * 74)
        for k in sorted(ps_dirty, key=lambda k: _ll(y, ps_dirty[k])):
            print(row(k, ps_dirty[k]))
        print("\n  Эти строки показаны только чтобы их НЕ приняли за рейтинг:\n"
              "  модель, видевшая матч при обучении, на нём выглядит лучше, чем\n"
              "  она есть. Сравнивать их с верхней таблицей нельзя.")

    # --- парное сравнение с боевым ансамблем ------------------------------
    if p_ens is not None:
        print("\n" + "=" * 78)
        print("КАЖДАЯ ПРОТИВ БОЕВОГО АНСАМБЛЯ — парный бутстрап по матчам")
        print("=" * 78)
        cmp_ = paired_vs_reference(y, g, p_ens, ps_clean,
                                   n_boot=args.n_boot, seed=args.seed)
        print(f"  {'модель':<23}{'Δ log_loss':>12}{'95% интервал':>24}"
              f"{'P(хуже)':>10}")
        print("  " + "-" * 74)
        for k in order:
            c = cmp_[k]
            print(f"  {k[:22]:<23}{c['delta']:>+12.4f}"
                  f"   [{c['lo']:+.4f}, {c['hi']:+.4f}]{c['p_worse']:>10.3f}")
        print("\n  Плюс значит «хуже ансамбля»: log loss меньше — лучше.")

        print("\n" + "=" * 78)
        print("ЧТО С ЭТИМ ДЕЛАТЬ")
        print("=" * 78)
        keep, retire, unclear = [], [], []
        for k in order:
            c = cmp_[k]
            if k in ens_names:
                keep.append((k, "участник боевого ансамбля"))
            elif c["lo"] > 0:
                retire.append((k, f"хуже ансамбля значимо ({c['delta']:+.4f}, "
                                  f"весь интервал выше нуля)"))
            elif c["hi"] < 0:
                keep.append((k, f"ЛУЧШЕ ансамбля значимо ({c['delta']:+.4f}) — "
                                f"разобраться, чем"))
            else:
                unclear.append((k, f"не отличается от ансамбля "
                                   f"({c['delta']:+.4f}, интервал накрывает ноль)"))
        for title, items in (("держать", keep), ("убрать", retire),
                             ("не отличается — держать нечем, но и вреда нет",
                              unclear)):
            if items:
                print(f"\n  {title}:")
                for k, why in items:
                    print(f"      {k:<20} {why}")
        if dirty:
            print("\n  особый случай — обучены БЕЗ холдаута:")
            for p, _a in dirty:
                print(f"      {p.stem:<20} померить на слепых матчах нельзя; "
                      f"держать только если нужна как запасная")

    # --- кэш для dwp.verdict ---------------------------------------------
    if not args.no_cache:
        store = {"y": y, "match_id": g, "minute": mn}
        for k, v in ps_clean.items():
            store[f"p__{k}"] = v
        if p_ens is not None:
            store["p__ENSEMBLE"] = p_ens
            store["ens_members"] = np.array(ens_names, dtype=object)
            # Оценка ДО начала игры, по одной на матч. Кладём сюда, чтобы
            # раздел «точность» брал её из того же вскрытия холдаута, что
            # и лайв-модель: собственные тесты моделей между собой
            # несравнимы (у разных сидов разные тестовые выборки).
            members = [a for p_, a in clean if p_.stem in set(ens_names)]
            if members:
                per = [draft_logit_by_match(a, matches) for a in members]
                keys = set().union(*(set(d_.keys()) for d_ in per))
                avg = {k_: float(np.mean([d_[k_] for d_ in per if k_ in d_]))
                       for k_ in keys}
                store["draft_logit"] = np.array(
                    [avg.get(int(m_), np.nan) for m_ in g], dtype=float)
        PRED_CACHE.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(PRED_CACHE, **store)
        print(f"\n  Кэш предсказаний: {PRED_CACHE} "
              f"({PRED_CACHE.stat().st_size / 1024:.0f} КБ) — его читает "
              f"`python -m dwp.verdict --tune`.")

    # --- реестр -----------------------------------------------------------
    if not args.no_registry:
        p_reg = p_ens if p_ens is not None else ps_clean[order[0]]
        sha = hashlib.sha256(np.ascontiguousarray(p_reg).tobytes()).hexdigest()
        n = HO.note_reveal(
            "bench_models(все)", "holdout", len(np.unique(g)), len(y), sha,
            {"log_loss": _ll(y, p_reg), "brier": _brier(y, p_reg),
             "ece": ece(y, p_reg, g)},
            note=f"рейтинг {len(ps_clean)} моделей за один проход")
        print(f"\n  Вскрытие холдаута записано в реестр (всего по этой "
              f"подписи: {n}).")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
