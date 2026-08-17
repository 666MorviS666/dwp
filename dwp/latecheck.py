"""Разбор одного вопроса: провал после 50-й минуты — это ошибка модели
или честная неопределённость поздней игры.

Два исхода различаются по reliability ВНУТРИ отрезка минут:

  * модель уверена и ошибается (в крайних децилях pred_mean далеко
    от actual) — виновато представление признаков: абсолютный перевес
    по золоту на 60-й минуте значит не то же, что на 20-й, а строк
    там мало, чтобы дерево выучило это взаимодействие;

  * модель не уверена, actual в каждом дециле около 0.5, log loss
    высокий просто потому, что задача такая — лечить нечего,
    матч длиной 70 минут по определению тот, где никто не закрыл.

Печатается также доля строк, где модель дала больше 0.8 или меньше 0.2:
если уверенных предсказаний в поздней игре почти нет, разговор про
«хуже базовой линии» вообще не о модели, а о восьми строках.

Запуск:
    python -m dwp.latecheck models\\B_extra.pkl --source real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features as F
from .compare import load_artefact, predict_on, _ll
from .train import _safe_stdout

BANDS = [(0, 30), (30, 50), (50, 60), (60, 10_000)]


def minute_of_rows(matches: list[dict]) -> np.ndarray:
    """Минуты в том же порядке строк, что возвращает predict_on."""
    out = []
    for m in matches:
        fr = F.match_state_frame(m)
        out.append(fr["minute"].to_numpy())
    return np.concatenate(out)


def band_report(y: np.ndarray, p: np.ndarray, mins: np.ndarray, groups: np.ndarray,
                base_rate: float, n_bins: int = 5) -> None:
    """n_matches печатается рядом с n_rows не для красоты. Строки одного
    матча делят лейбл и почти дублируют друг друга: двадцать строк из трёх
    матчей — это три наблюдения, а не двадцать, и смещение в 0.5 на таком
    дециле может быть одним неудачным матчем."""
    for lo, hi in BANDS:
        sel = (mins >= lo) & (mins < hi)
        if sel.sum() < 20:
            print(f"\n=== минуты {lo}-{hi if hi < 10_000 else '+'}: строк {int(sel.sum())}, "
                  f"мало для разбора ===")
            continue
        yy, pp, gg = y[sel], p[sel], groups[sel]
        ll = _ll(yy, pp)
        ll_base = _ll(yy, np.full(len(yy), base_rate))
        conf = float(((pp > 0.8) | (pp < 0.2)).mean())
        print(f"\n=== минуты {lo}-{hi if hi < 10_000 else '+'} ===")
        print(f"  строк {int(sel.sum())}   матчей {len(np.unique(gg))}   "
              f"log_loss {ll:.4f}   база {ll_base:.4f}   "
              f"доля уверенных (p вне 0.2..0.8): {conf:.3f}")
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(pp, edges[1:-1], right=False), 0, n_bins - 1)
        rows = []
        for b in range(n_bins):
            m = idx == b
            if not m.any():
                continue
            rows.append({
                "bin": f"[{edges[b]:.1f},{edges[b + 1]:.1f})",
                "n_rows": int(m.sum()),
                "n_matches": int(len(np.unique(gg[m]))),
                "pred_mean": float(pp[m].mean()),
                "actual": float(yy[m].mean()),
                "gap": float(pp[m].mean() - yy[m].mean()),
            })
        print(pd.DataFrame(rows).to_string(
            index=False, float_format=lambda v: f"{v:.4f}"))


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Reliability по отрезкам минут: ошибка модели или неопределённость.")
    ap.add_argument("model", type=Path)
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    args = ap.parse_args(argv)

    art = load_artefact(args.model)
    test_ids = set(art.get("test_match_ids") or [])
    if not test_ids:
        print("ОШИБКА: в артефакте нет test_match_ids.", file=sys.stderr)
        return 2
    directory = (config.RAW_MATCHES_DIR if args.source == "real"
                 else config.SYNTH_MATCHES_DIR)
    matches = [m for m in F.usable_matches(F.load_matches(directory), verbose=False)
               if int(m["match_id"]) in test_ids]
    if not matches:
        print(f"ОШИБКА: тестовых матчей не найдено в {directory}.", file=sys.stderr)
        return 2

    p, y, g = predict_on(art, matches)
    mins = minute_of_rows(matches)
    assert len(mins) == len(y), f"минуты и строки разъехались: {len(mins)} и {len(y)}"
    print("=" * 74)
    print(f"{args.model.name}: {len(np.unique(g))} матчей, {len(y)} строк, "
          f"base_rate {art['base_rate']:.4f}")
    print("=" * 74)
    band_report(y, p, mins, g, float(art["base_rate"]))
    print("\nКак читать: если в крайних децилях gap большой при заметном"
          "\nn_matches — модель уверенно ошибается, дело в признаках. Если"
          "\nactual везде около 0.5 и доля уверенных мала — неопределённость"
          "\nчестная, чинить нечего. Если n_matches в дециле однозначное —"
          "\nне читать вовсе, это один-два матча.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
