"""Сравнение двух моделей с доверительным интервалом.

Зачем отдельный модуль. При 588 тестовых матчах разница в log loss
порядка 0.005 неотличима от шума, а по одному числу этого не видно.
Без интервала любая правка выглядит как улучшение примерно в половине
случаев, и модель «улучшается» бесконечно, никуда не двигаясь.

Бутстрап идёт по match_id, а не по строкам. Это важно не всегда, и
стоит понимать когда именно. Замерено на искусственных данных
(600 матчей по 40 строк):

  * если модели различаются НЕЗАВИСИМЫМ шумом на каждой строке —
    интервалы по матчам и по строкам совпадают (0.0024 против 0.0023);
  * если различие СКОРРЕЛИРОВАНО внутри матча — интервал по строкам
    выходит 0.0018 против 0.0120 по матчам, вшестеро уже настоящего,
    и объявляет значимой разницу, которой нет.

Второй случай и есть реальный: смена признака или калибратора сдвигает
кривую целого матча, а не отдельные строки.

Сравнение ПАРНОЕ: на каждой пересэмплированной выборке считается
разность метрик двух моделей. Так уходит общая дисперсия «попались
лёгкие матчи», и остаётся только то, чем модели отличаются.
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features as F
from .train import apply_calibrator, ece, logit

EPS = 1e-6


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def load_artefact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"нет файла модели {path}")
    with path.open("rb") as fh:
        return pickle.load(fh)


def _frames_for(art: dict, matches: list[dict],
                parsed_cache: dict | None = None) -> pd.DataFrame:
    """Поминутные кадры с `draft_logit` ИМЕННО ЭТОЙ модели.

    Драфт-модель у каждого артефакта своя (она обучена на своём
    разбиении), поэтому кадр нельзя посчитать один раз на всех.
    """
    frames = []
    for m in matches:
        mid = int(m["match_id"])
        if parsed_cache is None:
            parsed = F.parse_objectives(m)
        else:
            parsed = parsed_cache.get(mid)
            if parsed is None:
                parsed = parsed_cache[mid] = F.parse_objectives(m)
        elo = art["elo_pre"].get(mid, art["elo_pre"].get(str(mid), 0.0))
        Xd, _, _ = F.draft_matrix([m], art["id2idx"], {mid: elo})
        p_draft = float(art["draft_model"].predict_proba(Xd)[0, 1])
        fr = F.match_state_frame(m, parsed)
        fr["draft_logit"] = logit(np.array([p_draft]))[0]
        frames.append(fr)
    return pd.concat(frames, ignore_index=True)


def _predict_single(art: dict, df: pd.DataFrame) -> np.ndarray:
    feats = art["state_features"]
    raw = art["booster"].predict(df[feats], num_iteration=art["booster"].best_iteration)
    e = max(art.get("calib_eps", 0.0), EPS)
    return np.clip(apply_calibrator(art["iso"], raw), e, 1 - e)


def predict_on(art: dict, matches: list[dict],
               parsed_cache: dict | None = None
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(вероятности, лейблы, match_id) на переданных матчах.

    Ансамбль (артефакт с ключом `members`) считается так же, как в бою:
    у каждой модели свой `draft_logit`, свой калибратор и своя обрезка
    хвостов, а усредняются уже готовые вероятности.
    """
    ms = art.get("members") or [art]
    ps = []
    df0 = None
    for m in ms:
        df = _frames_for(m, matches, parsed_cache)
        if df0 is None:
            df0 = df
        ps.append(_predict_single(m, df))
    p = np.mean(ps, axis=0)
    return p, df0["y"].to_numpy(), df0["match_id"].to_numpy()


def _ll(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, EPS, 1 - EPS)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def _brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


METRICS = {"log_loss": _ll, "brier": _brier}


def paired_bootstrap(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                     groups: np.ndarray, n_boot: int = 2000,
                     seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    # Индексы строк по матчам считаем один раз: пересэмплирование по
    # матчам должно тянуть за собой все строки матча целиком.
    idx_by_match = {g: np.flatnonzero(groups == g) for g in uniq}
    out: dict[str, dict] = {}
    for name, fn in METRICS.items():
        base = fn(y, pb) - fn(y, pa)
        deltas = np.empty(n_boot)
        for b in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            rows = np.concatenate([idx_by_match[g] for g in pick])
            deltas[b] = fn(y[rows], pb[rows]) - fn(y[rows], pa[rows])
        lo, hi = np.percentile(deltas, [2.5, 97.5])
        out[name] = {
            "delta": base,
            "lo": float(lo),
            "hi": float(hi),
            "p_better": float(np.mean(deltas < 0)),
        }
    return out


def verdict(ll: dict, identical: bool) -> list[str]:
    """Текст вывода по log loss.

    Отдельной функцией, потому что здесь был баг: цепочка сравнений
    строгая с обеих сторон, и интервал, КАСАЮЩИЙСЯ нуля, проваливался
    в ветку «B хуже значимо». В вырожденном случае (одинаковые модели)
    интервал равен ровно [0, 0], и compare объявлял модель значимо хуже
    самой себя.
    """
    if identical:
        return ["  ВЫВОД: модели дают побитово одинаковые предсказания.",
                "  Сравнивать нечего: либо это один и тот же артефакт, либо обе",
                "  обучены одними флагами с одним --seed. Чтобы сравнение имело",
                "  смысл, у B должно отличаться что-то, кроме имени файла."]
    if ll["lo"] <= 0 <= ll["hi"]:
        word = "хуже" if ll["delta"] > 0 else "лучше"
        return ["  ВЫВОД: разница по log loss НЕ значима — интервал накрывает ноль.",
                f"  B выглядит {word} на {abs(ll['delta']):.4f}, но это объясняется "
                f"тем, какие\n  матчи попали в тест. Нужна либо большая выборка, "
                f"либо более крупное изменение."]
    if ll["hi"] < 0:
        return [f"  ВЫВОД: B лучше значимо. Весь интервал ниже нуля "
                f"({ll['lo']:+.4f}..{ll['hi']:+.4f})."]
    return [f"  ВЫВОД: B ХУЖЕ значимо. Весь интервал выше нуля "
            f"({ll['lo']:+.4f}..{ll['hi']:+.4f})."]


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Парное сравнение двух моделей с доверительным интервалом.")
    ap.add_argument("model_a", type=Path, help="базовая модель")
    ap.add_argument("model_b", type=Path, help="модель с изменением")
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        a, b = load_artefact(args.model_a), load_artefact(args.model_b)
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}\nЧто делать: обучите обе модели через "
              f"`python -m dwp.train --out <файл>`.", file=sys.stderr)
        return 2

    ta, tb = set(a.get("test_match_ids", [])), set(b.get("test_match_ids", []))
    if ta != tb:
        print(f"ОШИБКА: у моделей РАЗНЫЕ тестовые выборки "
              f"({len(ta)} и {len(tb)} матчей, общих {len(ta & tb)}).\n"
              f"Сравнивать их нельзя: разница будет отражать разбиение, а не "
              f"модели.\nЧто делать: обучите обе с одинаковым --seed и на одном "
              f"наборе данных.", file=sys.stderr)
        return 2
    if not ta:
        print("ОШИБКА: в артефактах нет списка тестовых матчей.", file=sys.stderr)
        return 2

    directory = (config.RAW_MATCHES_DIR if args.source == "real"
                 else config.SYNTH_MATCHES_DIR)
    matches = [m for m in F.usable_matches(F.load_matches(directory), verbose=False)
               if int(m["match_id"]) in ta]
    if not matches:
        print(f"ОШИБКА: тестовых матчей не найдено в {directory}.", file=sys.stderr)
        return 2

    pa, y, g = predict_on(a, matches)
    pb, y2, g2 = predict_on(b, matches)
    assert np.array_equal(y, y2) and np.array_equal(g, g2), "выборки разъехались"

    print("=" * 74)
    print(f"A = {args.model_a.name}   признаков {len(a['state_features'])}, "
          f"калибратор {a.get('state_calib', '?')}")
    print(f"B = {args.model_b.name}   признаков {len(b['state_features'])}, "
          f"калибратор {b.get('state_calib', '?')}")
    print(f"Тест: {len(np.unique(g))} матчей, {len(y)} строк "
          f"(интервал считается по матчам)")
    print("=" * 74)

    print(f"\n  {'метрика':<12}{'A':>10}{'B':>10}{'B - A':>10}"
          f"{'95% интервал':>22}{'P(B лучше)':>12}")
    res = paired_bootstrap(y, pa, pb, g, n_boot=args.n_boot, seed=args.seed)
    for name, fn in METRICS.items():
        r = res[name]
        ci = f"[{r['lo']:+.4f}, {r['hi']:+.4f}]"
        print(f"  {name:<12}{fn(y, pa):>10.4f}{fn(y, pb):>10.4f}{r['delta']:>+10.4f}"
              f"{ci:>22}{r['p_better']:>12.3f}")
    ea, eb = ece(y, pa, g), ece(y, pb, g)
    print(f"  {'ECE':<12}{ea:>10.4f}{eb:>10.4f}{eb - ea:>+10.4f}"
          f"{'(без интервала)':>22}")

    ll = res["log_loss"]
    identical = bool(np.array_equal(pa, pb))
    print()
    for line in verdict(ll, identical):
        print(line)
    print("\n  Интервал получен пересэмплированием МАТЧЕЙ. По строкам он вышел бы")
    print("  вшестеро уже: изменение модели сдвигает кривую матча целиком, а не")
    print("  отдельные строки, и строки внутри матча не являются независимыми.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
