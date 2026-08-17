"""Замер: стоят ли чего-нибудь признаки по сборкам и по урону.

Порядок здесь такой же, как везде в проекте: сначала измерить, потом
решать. Оба набора признаков придуманы под конкретную гипотезу —

  сборки: «команда отстаёт по золоту, но у их кора шесть слотов, значит
          шансы есть» (проверка самой гипотезы отдельно, `dwp.builds --table`);
  урон:   «урон в матче тоже должен что-то весить»,

— и обе гипотезы разумны настолько, что их легко принять на веру. Поэтому
здесь A/B на одном разбиении и с парным бутстрапом ПО МАТЧАМ: разница
меньше 0.008 log loss на этих данных вообще неотличима от выбора сида
(замерено в `dwp.bench multi`: ст.откл 0.0079 по семи разбиениям).

Оговорка, которую надо держать в голове при чтении вывода: **урона в лайве
нет вовсе** (`dwp.damage --check`). Даже если он здесь поможет, в боевую
модель он не пойдёт — это будет мерой того, чего лайв-источник лишает, а
не поводом добавить признак.

Запуск:
    python -m dwp.export --what extras          # один раз, ~25 минут
    python -m dwp.bench_extras
    python -m dwp.bench_extras --group builds --seeds 3,7,11
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .bench import _paired, run
from .builds import BUILD_COLUMNS
from .damage import DAMAGE_COLUMNS
from .features import EXACT_STATE_FEATURES

# Свободное золото команды, отдельно от остальных сборок.
#
# ЗАЧЕМ ОТДЕЛЬНО. Набор `builds` целиком уже мерили, и он не прошёл
# (−0.0012 при разбросе по сидам 0.0079). Но внутри него `unspent_adv`
# стоит особняком: это единственная величина блока, которая в лайве
# ТОЧНА — источник отдаёт `players[].gold` прямо, ничего восстанавливать
# не нужно. Остальные признаки сборок в лайве собираются из инвентаря по
# справочнику, то есть с погрешностью. Мерить их одной группой значит
# топить точный признак в шести приблизительных.
#
# И вторая причина: свободное золото — это выкупы, то есть ровно та часть
# расхождения `gold_adv`, которую лайв всё-таки видит.
UNSPENT_COLUMNS = ["unspent_adv"]

GROUPS = {
    "unspent": UNSPENT_COLUMNS,
    "builds": BUILD_COLUMNS,
    "damage": DAMAGE_COLUMNS,
    "both": BUILD_COLUMNS + DAMAGE_COLUMNS,
}


def load(frames: Path, draft: Path, extras: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    fr = pd.read_csv(frames)
    dr = pd.read_csv(draft)
    ex = pd.read_csv(extras)
    before = len(fr)
    fr = fr.merge(ex, on=["match_id", "minute"], how="left")
    assert len(fr) == before, "слияние размножило строки — проверьте ключи"
    cov = fr[BUILD_COLUMNS[0]].notna().mean()
    covd = fr[DAMAGE_COLUMNS[0]].notna().mean()
    print(f"Строк: {len(fr)}, матчей: {fr.match_id.nunique()}")
    print(f"  сборки известны у {cov * 100:.1f}% строк, урон у {covd * 100:.1f}%")
    print("  (где неизвестно — NaN, LightGBM обрабатывает пропуск сам; ноль "
          "тут был бы\n   ложью: «предметов нет» и «мы не знаем» — разные вещи)")
    return fr, dr


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="A/B по сборкам и урону.")
    ap.add_argument("--frames", type=Path,
                    default=config.DATA_DIR / "export_frames.csv.gz")
    ap.add_argument("--draft", type=Path,
                    default=config.DATA_DIR / "export_draft.csv.gz")
    ap.add_argument("--extras", type=Path,
                    default=config.DATA_DIR / "export_extras.csv.gz")
    ap.add_argument("--group", choices=[*GROUPS, "all"], default="all")
    ap.add_argument("--seeds", default="7",
                    help="через запятую; несколько сидов — потому что разброс "
                         "по разбиениям больше типичного эффекта")
    args = ap.parse_args(argv)

    for p in (args.frames, args.draft, args.extras):
        if not p.exists():
            print(f"ОШИБКА: нет файла {p}.\nЧто делать: python -m dwp.export "
                  f"--what {'extras' if 'extras' in p.name else p.stem.split('_')[-1]}",
                  file=sys.stderr)
            return 2
    fr, dr = load(args.frames, args.draft, args.extras)

    # База — набор лайв-модели: без XP, с «точными» признаками. Именно её
    # мы и хотим улучшить, и сравнивать надо с ней, а не с офлайн-моделью.
    base_extra = list(EXACT_STATE_FEATURES)
    seeds = [int(s) for s in args.seeds.split(",")]
    groups = list(GROUPS) if args.group == "all" else [args.group]

    print(f"\nБаза: признаки лайв-модели (--no-xp + точные), "
          f"{len(base_extra)} сверх основного набора")
    print("=" * 74)
    for seed in seeds:
        oa, pa, y, g, _ = run(fr, dr, seed=seed, elo_scale=10.0, draft_C=0.02,
                              use_xp=False, feats_extra=True,
                              extra_cols=base_extra, verbose=False, tag="A")
        print(f"\n[сид {seed}]  база: ll {oa['ll']:.4f}  ECE {oa['ece']:.3f}  "
              f"auc {oa['auc']:.4f}  тест {oa['n_test_matches']} матчей")
        for gname in groups:
            cols = [c for c in GROUPS[gname] if c in fr.columns]
            ob, pb, y2, g2, _ = run(fr, dr, seed=seed, elo_scale=10.0,
                                    draft_C=0.02, use_xp=False, feats_extra=True,
                                    extra_cols=base_extra + cols,
                                    verbose=False, tag=gname)
            assert np.array_equal(y, y2) and np.array_equal(g, g2), \
                "выборки разъехались"
            d, lo, hi, pw = _paired(y.astype(float), pa, pb, g)
            verdict = ("значимо лучше" if hi < 0 else
                       "значимо ХУЖЕ" if lo > 0 else "не значимо")
            print(f"  + {gname:<8} ({len(cols)} призн.): ll {ob['ll']:.4f}  "
                  f"ECE {ob['ece']:.3f}  разница {d:+.4f}  "
                  f"95% [{lo:+.4f}, {hi:+.4f}]  {verdict}")
    print("=" * 74)
    print("Как читать. Интервал накрывает ноль — прирост не доказан, и "
          "добавлять\nпризнаки на этом основании нельзя: шесть лишних "
          "признаков это шесть\nновых способов разъехаться между обучением и "
          "лайвом (README, gold_adv).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
