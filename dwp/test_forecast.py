"""Проверка блока «перспектива» без сети и без обучения.

Что именно проверяется. Не то, что коридор «правильный» — правильность
его чисел обеспечивает выборка, а не код, — а то, что он МОЛЧИТ там, где
данных нет, и не переворачивает стороны там, где они есть. Обе ошибки
незаметны глазом: перевёрнутая сторона выглядит как осмысленное число,
а выдуманная клетка — как честная статистика.

    python -m dwp.test_forecast
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from . import config, forecast as FC

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'ПРОВАЛ'} {name}" + (f": {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def fake_rows() -> pd.DataFrame:
    """Две сотни матчей с известным исходом и известной динамикой оценки.

    Половина — Radiant ведёт и удерживает, половина — Dire ведёт и
    удерживает. Значит смена лидера должна выйти нулевой, а «лидер
    выиграл» — стопроцентной. Любое другое число означает, что стороны
    где-то перепутаны.
    """
    rows = []
    for mid in range(200):
        rad_leads = mid % 2 == 0
        y = 1 if rad_leads else 0
        for mn in range(0, 40):
            p = 0.75 if rad_leads else 0.25
            rows.append({"match_id": mid, "minute": float(mn), "p": p, "y": y,
                         "p10": p, "ended10": 0, "p5": p, "ended5": 0})
    return pd.DataFrame(rows)


def main() -> int:
    print("\n1. Коридор: стороны не переворачиваются")
    t = FC.build_table(fake_rows())
    check("клетки заполнены", len(t["cells"]) > 0, f"{len(t['cells'])} клеток")
    b = FC.band(t, 20.0, 0.75, 10.0)
    check("клетка для Radiant нашлась", b is not None)
    if b:
        check("лидер — Radiant", b["leader"] == "radiant", b["leader"])
        check("смены лидера нет", abs(b["flip"]) < 1e-9, f"{b['flip']}")
        check("лидер выиграл в 100% случаев", abs(b["lead_wins"] - 1.0) < 1e-9,
              f"{b['lead_wins']:.3f}")
    b2 = FC.band(t, 20.0, 0.25, 10.0)
    check("зеркальная клетка для Dire нашлась", b2 is not None)
    if b and b2:
        check("оценка лидера одинакова с обеих сторон",
              abs(b["p_lead"] - b2["p_lead"]) < 1e-9)
        check("лидер — Dire", b2["leader"] == "dire", b2["leader"])
        check("исход тот же", abs(b["lead_wins"] - b2["lead_wins"]) < 1e-9)

    print("\n2. Коридор молчит там, где данных нет")
    check("нет таблицы -> None", FC.band(None, 20.0, 0.7) is None)
    check("минута вне сетки -> None", FC.band(t, 500.0, 0.7) is None)
    check("пустая клетка -> None", FC.band(t, 20.0, 0.999) is None,
          "у 0.95-1.00 наблюдений нет")
    check("NaN на входе -> None", FC.band(t, float("nan"), 0.7) is None)

    print("\n3. Порог по числу наблюдений соблюдается")
    small = fake_rows().head(FC.MIN_CELL - 1)
    ts = FC.build_table(small)
    check("клеток меньше порога не создано", len(ts["cells"]) == 0,
          f"{len(ts['cells'])} клеток при {len(small)} строках")

    print("\n4. Темп: факт из признаков, а не прогноз")
    df = pd.DataFrame([{"minute": 22.0, "gold_adv_slope5": 310.0,
                        "gold_adv_d1": 120.0, "kills_adv_d5": -3.0}])
    tp = FC.tempo(df)
    check("темп посчитан", tp is not None)
    if tp:
        check("золото в минуту прочитано", tp["gold_per_min"] == 310.0)
        check("фраги за 5 минут прочитаны", tp["kills_5"] == -3.0)
    empty = pd.DataFrame([{"minute": 3.0, "gold_adv_slope5": np.nan,
                           "gold_adv_d1": np.nan, "kills_adv_d5": np.nan}])
    check("окно не наполнилось -> None", FC.tempo(empty) is None)

    print("\n5. Ресурсы: пропуск остаётся пропуском, а не нулём")
    st = {"towers_lost": {config.TEAM_RADIANT: 3, config.TEAM_DIRE: 7},
          "rax_lost": {config.TEAM_RADIANT: 0, config.TEAM_DIRE: 2},
          "t3_lost": {config.TEAM_RADIANT: 0, config.TEAM_DIRE: 2}}
    r = FC.resources(st)
    check("стоящих вышек у Radiant 8",
          r["sides"][0]["towers_standing"] == config.N_TOWERS_PER_SIDE - 3,
          str(r["sides"][0]["towers_standing"]))
    check("бараков у Dire осталось 4",
          r["sides"][1]["rax_standing"] == config.N_RAX_PER_SIDE - 2,
          str(r["sides"][1]["rax_standing"]))
    r2 = FC.resources({})
    check("нет данных о зданиях -> None, а не ноль",
          r2["sides"][0]["towers_standing"] is None,
          repr(r2["sides"][0]["towers_standing"]))

    print("\n6. Замер проекции записан в коде, а не только в README")
    m = FC.PROJECTION_MEASURED
    check("есть оба горизонта", "5" in m and "10" in m)
    check("проекция ХУЖЕ наивной на обоих горизонтах",
          all(m[h]["mae_proj"] > m[h]["mae_now"] for h in ("5", "10")),
          "иначе её стоило бы вернуть на экран")

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО ПРОВЕРОК: {len(FAILED)}")
        for n in FAILED:
            print(f"  - {n}")
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
