"""Кривая обучения: сколько точности стоит объём выгрузки.

ВОПРОС. «Влияет ли число матчей на точность предсказания, и стоит ли
качать ещё?» Ответ на него нельзя дать рассуждением: у любой модели
кривая обучения рано или поздно выходит на полку, и вопрос ровно в том,
дошли мы до неё или нет.

КАК МЕРЯЕТСЯ. Урезается ТОЛЬКО обучающий пул; тест остаётся целиком и не
меняется внутри одного сида. Иначе разница между точками кривой мерила бы
заодно и то, что тест стал меньше, — а это совсем другая величина.
Отброшенные матчи выбрасываются и из истории Elo: Elo это признак, и
оставить в нём исходы «невиденных» матчей значило бы протечку, которая
занижает эффект объёма.

ПОЧЕМУ НЕСКОЛЬКО СИДОВ. Разброс log loss между полностью переобученными
моделями, отличающимися только сидом разбиения, сопоставим с самим
эффектом. По одному прогону на точку кривая получилась бы пилой, и
любой её зуб можно было бы прочитать как «полка». Поэтому каждая точка —
среднее по нескольким сидам, и рядом печатается разброс.

ЧЕГО ЭТОТ ЗАМЕР НЕ ГОВОРИТ. Он про объём ПРИ НЕИЗМЕННОЙ МЕТЕ. Реальный
добор данных приносит ещё и свежие патчи, а старые матчи при этом
устаревают; здесь и то и другое перемешано, потому что подвыборка берётся
случайно по всей истории, а не по времени.

    python -m dwp.learning_curve                       # 5 точек x 3 сида
    python -m dwp.learning_curve --fracs 0.25 0.5 1.0 --seeds 7
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import config
from .train import run as train_run

OUT_PATH = config.DATA_DIR / "learning_curve.json"


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def point(frac: float, seed: int, **kw) -> dict | None:
    """Один прогон обучения. Возвращает метрики на ЕГО тесте."""
    art = train_run(
        config.RAW_MATCHES_DIR, use_xp=False, limit=None, seed=seed,
        out_path=None, draft_calib="isotonic", state_calib="auto",
        calib_criterion="logloss", live_features=True, extra_features=True,
        gold_norm="add", exact_features=True, use_holdout=True,
        pool_frac=frac, quiet=True, return_artefact=True, **kw)
    if not isinstance(art, dict):
        return None
    m = art["metrics"]
    return {
        "frac": frac, "seed": seed,
        "n_pool": art.get("n_pool_matches"),
        "log_loss": m["state_test_calibrated"]["log_loss"],
        "brier": m["state_test_calibrated"]["brier"],
        "auc": m["state_test_calibrated"]["auc"],
        "baseline": m["baseline"]["log_loss"],
        "draft_log_loss": m["draft_test"]["log_loss"],
        "draft_auc": m["draft_test"]["auc"],
        "n_test_matches": len(art.get("test_match_ids") or []),
    }


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for frac in sorted({r["frac"] for r in rows}):
        got = [r for r in rows if r["frac"] == frac]
        ll = np.array([r["log_loss"] for r in got], dtype=float)
        out.append({
            "frac": frac,
            "n_pool": int(np.mean([r["n_pool"] for r in got])),
            "runs": len(got),
            "log_loss": float(ll.mean()),
            "sd": float(ll.std(ddof=1)) if len(ll) > 1 else float("nan"),
            "se": (float(ll.std(ddof=1) / np.sqrt(len(ll)))
                   if len(ll) > 1 else float("nan")),
            "brier": float(np.mean([r["brier"] for r in got])),
            "auc": float(np.mean([r["auc"] for r in got])),
            "draft_auc": float(np.mean([r["draft_auc"] for r in got])),
            "baseline": float(np.mean([r["baseline"] for r in got])),
        })
    return out


def verdict(summary: list[dict]) -> list[str]:
    """Что означает кривая. Решает ПОСЛЕДНИЙ шаг, а не размах целиком.

    Вопрос звучит «стоит ли качать ещё», а не «помогли ли данные вообще».
    Ответ на первый даёт предельная отдача В ТОЧКЕ, где мы стоим: разница
    между двумя последними долями. Размах от 20% до 100% может быть
    большим и при полностью выродившейся отдаче в конце — именно так
    выглядит любая насыщающаяся кривая.
    """
    lines: list[str] = []
    full, prev = summary[-1], summary[-2] if len(summary) > 1 else summary[-1]
    gain_last = prev["log_loss"] - full["log_loss"]
    se = [s["se"] if s["se"] == s["se"] else 0.0 for s in (prev, full)]
    se_comb = float(np.hypot(se[0], se[1]))
    gain_all = summary[0]["log_loss"] - full["log_loss"]
    lines.append("")
    lines.append(f"  Весь размах ({summary[0]['n_pool']} -> {full['n_pool']} "
                 f"матчей): {gain_all:+.4f} log loss.")
    lines.append(f"  ПОСЛЕДНИЙ шаг ({prev['n_pool']} -> {full['n_pool']}): "
                 f"{gain_last:+.4f} при двух ошибках среднего ±{2 * se_comb:.4f}.")
    if gain_last > 2 * max(se_comb, 1e-9):
        lines.append("  ВЫВОД: объём ещё РАБОТАЕТ. Столько же примерно даст и")
        lines.append("  следующий такой же прирост выгрузки — качать имеет смысл.")
    else:
        # Где кривая вышла на полку: первая доля, чей log loss не хуже
        # лучшего больше чем на две ошибки среднего.
        best = min(s["log_loss"] for s in summary)
        tol = 2 * max(se_comb, 1e-9)
        knee = next((s for s in summary if s["log_loss"] - best <= tol), full)
        lines.append("  ВЫВОД: кривая ВЫШЛА НА ПОЛКУ. Последний прирост данных не")
        lines.append("  дал ничего сверх разброса по сидам, значит добор матчей")
        lines.append("  сам по себе точность уже не поднимет.")
        lines.append(f"  Полка начинается примерно с {knee['n_pool']} матчей "
                     f"обучающего пула;")
        lines.append(f"  сейчас их {full['n_pool']}, то есть запас по объёму "
                     f"уже выбран.")
        lines.append("  Резерв надо искать в другом: ансамбль по сидам, "
                     "эмбеддинги героев,")
        lines.append("  веса по свежести патчей.")
    # Отдельно про драфт: у стейт-модели полка может быть уже достигнута, а
    # у драфт-модели нет — она линейная, коэффициентов 127, и данных ей не
    # хватает дольше. Это важно, потому что половина всей ошибки лежит в
    # первых пятнадцати минутах, где драфт — единственный сигнал.
    d_gain = full["draft_auc"] - summary[0]["draft_auc"]
    d_last = full["draft_auc"] - prev["draft_auc"]
    lines.append("")
    lines.append(f"  Драфт-модель отдельно: AUC {summary[0]['draft_auc']:.4f} -> "
                 f"{full['draft_auc']:.4f} ({d_gain:+.4f}),")
    lines.append(f"  на последнем шаге {d_last:+.4f}.")
    if d_last > 0.002:
        lines.append("  У НЕЁ полка ещё НЕ достигнута — 127 коэффициентов на "
                     "несколько тысяч")
        lines.append("  матчей, данных ей не хватает дольше. А половина всей "
                     "ошибки лежит в")
        lines.append("  первых пятнадцати минутах, где драфт — единственный "
                     "доступный сигнал.")
    for ln in lines:
        print(ln)
    return lines


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Кривая обучения: точность против объёма выгрузки.")
    ap.add_argument("--fracs", type=float, nargs="+",
                    default=[0.2, 0.4, 0.6, 0.8, 1.0])
    ap.add_argument("--seeds", type=int, nargs="+", default=[3, 7, 11])
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    args = ap.parse_args(argv)

    total = len(args.fracs) * len(args.seeds)
    print("=" * 74)
    print(f"Кривая обучения: {len(args.fracs)} точек x {len(args.seeds)} сидов "
          f"= {total} обучений, примерно по минуте каждое.")
    print("Урезается только обучающий пул; тест внутри сида не меняется.")
    print("=" * 74, flush=True)

    rows: list[dict] = []
    t0 = time.time()
    for i, frac in enumerate(sorted(args.fracs)):
        for seed in args.seeds:
            r = point(frac, seed)
            if r is None:
                print(f"  доля {frac:.2f} сид {seed}: обучение не удалось",
                      flush=True)
                continue
            rows.append(r)
            print(f"  доля {frac:.2f}  сид {seed:>3}  матчей {r['n_pool']:>5}  "
                  f"log_loss {r['log_loss']:.4f}  auc {r['auc']:.4f}  "
                  f"драфт auc {r['draft_auc']:.4f}   "
                  f"[{len(rows)}/{total}, {time.time() - t0:.0f} с]", flush=True)
    if not rows:
        print("ОШИБКА: ни одного удачного обучения.", file=sys.stderr)
        return 2

    summary = summarise(rows)
    print("\n" + "=" * 74)
    print(f"  {'доля':>6}{'матчей':>9}{'log_loss':>11}{'±sd':>9}{'brier':>9}"
          f"{'auc':>8}{'драфт auc':>11}")
    print("  " + "-" * 62)
    for s in summary:
        sd = "-" if s["sd"] != s["sd"] else f"{s['sd']:.4f}"
        print(f"  {s['frac']:>6.2f}{s['n_pool']:>9}{s['log_loss']:>11.4f}"
              f"{sd:>9}{s['brier']:>9.4f}{s['auc']:>8.4f}{s['draft_auc']:>11.4f}")

    verdict(summary)
    print("\n  Оговорка: подвыборка случайна по всей истории, поэтому замер про")
    print("  ОБЪЁМ при неизменной мете. Настоящий добор приносит ещё и свежие")
    print("  патчи, а это отдельный эффект, здесь не отделённый.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(
        {"runs": rows, "summary": summary, "when": time.time(),
         "seeds": args.seeds, "fracs": sorted(args.fracs)},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  Числа сохранены: {args.out}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
