"""Проверка вердикта и признака «убит» без сети и без обучения.

Что проверяется. Не то, что вердикт «правильный» — правильность его доли
обеспечивает выборка, а не код, — а пять свойств, каждое из которых
ломается незаметно:

1. Вердикт НЕ МЕНЯЕТСЯ. Ни при каком развитии матча, включая полный
   разворот и 95% у противоположной стороны. Это главное требование, и
   держаться оно должно структурно, а не за счёт удачно выбранного
   порога.
2. Раньше срока вердикта нет, и до порога уверенности — тоже нет.
3. Крайний срок работает: матч, где уверенность так и не набралась, всё
   равно получает вердикт, а не остаётся без него навсегда.
4. Сглаживание не зависит от частоты опроса. Панель опрашивает раз в 2 с,
   холдаут даёт строку раз в минуту; разойдись они — правило на экране
   будет не тем, которое мерили.
5. Число без выборки не показывается: клетка меньше порога не создаётся,
   а `hitrate` на пустое место отвечает None.

Плюс шестое, из соседнего модуля: «убит» ставится по замеренному признаку
и снимается движением, а не таймером.

    python -m dwp.test_verdict
"""

from __future__ import annotations

import sys

import numpy as np

from . import deaths as DTH, verdict as VD

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'ПРОВАЛ'} {name}" + (f": {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def ramp(n: int = 40, lo: float = 0.45, hi: float = 0.95):
    """Матч, где одна сторона уверенно и монотонно забирает игру."""
    mn = np.arange(n, dtype=float)
    return mn, np.linspace(lo, hi, n)


def reversal(n: int = 40, early: float = 0.85, late: float = 0.05):
    """Матч с полным разворотом: ранний фаворит проваливается.

    Ровно тот случай, про который заказчик сказал «даже если
    противоположная сторона имеет 90%». Вердикт обязан остаться прежним:
    камбэк — это ошибка вердикта, а не повод его переписать.
    """
    mn = np.arange(n, dtype=float)
    p = np.concatenate([np.full(12, early),
                        np.linspace(early, late, n - 12)])
    return mn, p


def fake_players(coords, deaths, team_a: int = 2, team_b: int = 3) -> dict:
    """Ответ вида GetRealtimeStats из готовых координат и счётчиков."""
    teams = []
    for ti, tn in enumerate((team_a, team_b)):
        pl = []
        for i in range(5):
            k = ti * 5 + i
            x, y = coords[k]
            pl.append({"team_slot": i, "heroid": 100 + k, "name": f"p{k}",
                       "kill_count": 0, "death_count": deaths[k],
                       "assists_count": 0, "x": x, "y": y})
        teams.append({"team_number": tn, "players": pl})
    return {"teams": teams}


def main() -> int:
    print("\n1. Вердикт не меняется НИ ПРИ КАКОМ развитии матча")
    mn, p = reversal()
    naive = int((np.diff(np.where(p[mn >= 7] >= 0.5, 1, -1)) != 0).sum())
    got = VD.run_rule(mn, p, VD.Rule(t_open=7, half_life=1.0, theta_p=0.65))
    seen = set(int(s) for s in got["side"] if s != 0)
    check("сырой процент в этом матче сторону меняет", naive >= 1,
          f"{naive} смен")
    check("вердикт назвал раннего фаворита", got["commit_side"] == 1,
          str(got["commit_side"]))
    check("и не сменил его ни разу, хотя оценка ушла к 0.05",
          seen == {1}, f"встречавшиеся стороны: {sorted(seen)}")
    check("flips равен нулю по построению", got["flips"] == 0)
    # То же самое, но порогом, при котором вердикт выносится сразу: смены
    # нет и здесь — она структурно отсутствует, а не подавлена порогом.
    low = VD.run_rule(mn, p, VD.Rule(t_open=7, half_life=0.25, theta_p=0.50))
    check("при пороге 0.50 смены тоже нет",
          set(int(s) for s in low["side"] if s != 0) == {1})
    # И на дрожащей около половины оценке — том случае, ради которого
    # прежняя редакция заводила гистерезис.
    rng = np.random.default_rng(1)
    wob = np.clip(0.5 + 0.06 * rng.standard_normal(40), 0.02, 0.98)
    w = VD.run_rule(np.arange(40.0), wob, VD.Rule(7, 0.25, 0.50))
    check("на дрожащей оценке сторона тоже одна",
          len(set(int(s) for s in w["side"] if s != 0)) == 1)

    print("\n2. До срока вердикта нет, до уверенности — тоже нет")
    mn2, p2 = ramp(lo=0.50, hi=0.95)
    r = VD.Rule(t_open=7, half_life=0.5, theta_p=0.70, t_force_after=1e9)
    got2 = VD.run_rule(mn2, p2, r)
    check("до 7-й минуты стороны нет",
          bool((got2["side"][mn2 < 7] == 0).all()))
    check("вердикт вынесен не на 7-й минуте, а когда набралась уверенность",
          got2["commit_minute"] > 7.0, f"минута {got2['commit_minute']:.1f}")
    check("в момент фиксации уверенность не ниже порога",
          got2["commit_conf"] >= 0.70, f"{got2['commit_conf']:.3f}")
    # Матч, который так и не ушёл от половины, без крайнего срока остаётся
    # вообще без вердикта — и это должно быть видно, а не подменяться.
    flat = VD.run_rule(np.arange(40.0), np.full(40, 0.52),
                       VD.Rule(7, 1.0, 0.80, t_force_after=1e9))
    check("без крайнего срока слабый матч остаётся без вердикта",
          not flat["committed"])
    late = VD.run_rule(np.arange(30.0, 40.0), np.full(10, 0.9),
                       VD.Rule(t_open=7, half_life=1.0, theta_p=0.6),
                       first_minute=30.0)
    check("подключились в середине -> ждём окно наблюдения, а не минуту 7",
          abs(late["open_at"] - (30.0 + VD.WARMUP_IF_LATE)) < 1e-9,
          f"{late['open_at']}")

    print("\n3. Крайний срок доводит вердикт до конца")
    forced = VD.run_rule(np.arange(40.0), np.full(40, 0.52),
                         VD.Rule(7, 1.0, 0.80, t_force_after=6.0))
    check("уверенности нет, но на крайнем сроке сторона названа",
          forced["committed"])
    check("названа ровно на крайнем сроке",
          abs(forced["commit_minute"] - 13.0) < 1e-9,
          f"минута {forced['commit_minute']}")
    check("названа та сторона, что впереди", forced["commit_side"] == 1)

    print("\n4. Сглаживание не зависит от частоты опроса")
    # Тот же матч, опрошенный раз в минуту и раз в 2 секунды. Значения
    # между минутами берутся линейной интерполяцией — то есть это ОДИН
    # И ТОТ ЖЕ ряд, показанный с разной частотой.
    mn_c = np.arange(0, 39.0, 2.0 / 60.0)
    p_c = np.interp(mn_c, mn2, p2)
    z_min = VD.smooth(mn2, p2, 1.0)
    z_fine = VD.smooth(mn_c, p_c, 1.0)
    at = np.searchsorted(mn_c, mn2[-1]) - 1
    gap = abs(float(z_min[-1]) - float(z_fine[at]))
    check("сглаженный логит совпадает на разной частоте", gap < 0.05,
          f"разница {gap:.4f} в логитах")
    r_min = VD.run_rule(mn2, p2, VD.Rule(7, 1.0, 0.70))
    r_fine = VD.run_rule(mn_c, p_c, VD.Rule(7, 1.0, 0.70))
    check("сторона одна и та же",
          r_min["commit_side"] == r_fine["commit_side"])
    check("и минута вердикта совпадает с точностью до шага опроса",
          abs(r_min["commit_minute"] - r_fine["commit_minute"]) <= 1.0,
          f"{r_min['commit_minute']:.2f} против {r_fine['commit_minute']:.2f}")

    print("\n5. Число без выборки не показывается")
    rows = []
    for i in range(400):
        m_, p_ = ramp()
        rows.append({"match_id": i, "minute": m_, "p": p_, "y": 1, "half": "B"})
    rule = VD.Rule(7, 0.5, 0.65)
    tbl = VD.build_hitrate(rows, rule)
    check("клетки заполнены", len(tbl["cells"]) > 0, f"{len(tbl['cells'])}")
    check("все клетки не меньше порога",
          all(c["n_matches"] >= VD.MIN_CELL for c in tbl["cells"].values()),
          f"порог {VD.MIN_CELL} матчей")
    full = {"rule": {"t_open": 7, "half_life": 0.5, "theta_p": 0.65,
                     "t_force_after": 6.0},
            "hitrate": tbl}
    check("нет таблицы -> None", VD.hitrate(None, 20.0, 0.8) is None)
    check("минута вне сетки -> None", VD.hitrate(tbl, 500.0, 0.8) is None)
    check("NaN на входе -> None", VD.hitrate(tbl, float("nan"), 0.8) is None)
    # Два набора клеток: смотрели с начала и подключились в середине. Это
    # разные популяции, и подставлять одной долю другой нельзя.
    check("наборы клеток помечены режимом",
          all(c.split(":")[0] in ("s", "j") for c in tbl["cells"]),
          f"ключи: {sorted(tbl['cells'])[:4]}")
    modes = {c.split(":")[0] for c in tbl["cells"]}
    check("в замере есть и подключение в середине", "j" in modes,
          f"режимы: {sorted(modes)}")
    check("боевой вызов без таблицы -> None",
          VD.live_verdict(None, [1.0], [0.6]) is None)
    lv = VD.live_verdict(full, list(mn2), list(p2), names=("Рад", "Дир"))
    check("боевой вызов вернул сторону", lv and lv["side"] in ("radiant", "dire"),
          str(lv and lv["side"]))
    check("имя стороны подставлено", lv and lv["name"] in ("Рад", "Дир"),
          str(lv and lv["name"]))
    check("уверенность в момент фиксации в (0.5, 1]",
          lv and 0.5 <= lv["commit_conf"] <= 1.0,
          f"{lv and lv['commit_conf']:.3f}")
    # Клетка ОБЯЗАНА найтись: позиция взята из тех же матчей, на которых
    # таблица и построена. Если тут None — значит боевой вызов ищет клетку
    # не там, где она лежит. Ровно это однажды и случилось: в поиск уходил
    # весь файл вместо поддерева `hitrate`, бины совпадали, клетка не
    # находилась никогда, а панель писала «выборки нет» — и выглядело это
    # как честная оговорка, а не как ошибка.
    check("доля попаданий подтянулась к боевому вызову", bool(lv and lv["hit"]),
          "None означает, что клетку ищут не в том поддереве")
    if lv and lv["hit"]:
        check("рядом с долей стоит выборка", lv["hit"]["n_matches"] > 0,
              f"{lv['hit']['n_matches']} матчей")
    # Подпись под вердиктом обязана быть НЕПОДВИЖНОЙ: клетка ищется по
    # моменту фиксации, значит на более поздней минуте она та же самая.
    half = len(mn2) // 2 + 8
    lv2 = VD.live_verdict(full, list(mn2[:half]), list(p2[:half]))
    if lv and lv2 and lv["hit"] and lv2["hit"]:
        check("доля не меняется по ходу матча",
              abs(lv["hit"]["hit"] - lv2["hit"]["hit"]) < 1e-12,
              f"{lv2['hit']['hit']:.4f} -> {lv['hit']['hit']:.4f}")
        check("и минута вердикта тоже не меняется",
              abs(lv["commit_minute"] - lv2["commit_minute"]) < 1e-12)

    print("\n6. «Убит» ставится счётчиком и снимается движением")
    d = DTH.Deaths()
    base = [(0.1 * i, 0.1 * i) for i in range(10)]
    d.update(fake_players(base, [0] * 10), 10.0)
    d.update(fake_players(base, [0] * 10), 10.1)
    check("пока никто не умирал — мёртвых нет", d.n_dead == 0, str(d.n_dead))
    check("до первой смерти состояние помечено как неизвестное",
          not d.state(2, 0)["known"])
    dd = [0] * 10
    dd[3] = 1
    d.update(fake_players(base, dd), 10.2)
    check("счётчик вырос -> герой мёртв", d.state(2, 3)["dead"])
    check("остальные живы", d.n_dead == 1, str(d.n_dead))
    d.update(fake_players(base, dd), 10.4)
    check("координаты стоят -> всё ещё мёртв", d.state(2, 3)["dead"])
    moved = list(base)
    moved[3] = (base[3][0] + 0.02, base[3][1])
    d.update(fake_players(moved, dd), 10.6)
    check("координаты пошли -> ожил", not d.state(2, 3)["dead"])
    # Дрожание в последнем разряде движением считаться не должно. На
    # замере его нет вовсе (координата повторяется бит в бит), но порог
    # обязан пережить его появление.
    dd2 = list(dd)
    dd2[3] = 2
    d.update(fake_players(moved, dd2), 10.8)
    jitter = list(moved)
    jitter[3] = (moved[3][0] + DTH.MOVE_EPS / 10.0, moved[3][1])
    d.update(fake_players(jitter, dd2), 11.0)
    check("дрожание меньше порога не воскрешает", d.state(2, 3)["dead"],
          f"порог {DTH.MOVE_EPS:g}")

    print("\n7. Замер записан в коде, а не только в отчёте")
    t = VD.load_table()
    m = (t or {}).get("measured") or {}
    if t is None:
        check("data/verdict.json ещё не собран", True,
              "это не провал: `python -m dwp.verdict --tune`")
    elif m.get("acc_commit") is None:
        # Таблица осталась от прежнего правила (с гистерезисом): ключей
        # нового замера в ней нет. Падать на этом нельзя — иначе после
        # каждой смены правила тест валится не по делу, — но и молчать
        # тоже: пока таблицу не пересобрали, панель показывает старое.
        check("таблица осталась от прежнего правила", True,
              "пересобрать: `python -m dwp.bench_models` "
              "и `python -m dwp.verdict --tune --frontier`")
    else:
        check("смен стороны ноль", m.get("flips_mean", 9) == 0.0,
              f"{m.get('flips_mean')}")
        check("сырой процент при этом сторону менял",
              m.get("naive_flips_mean", 0) > 0.5,
              f"{m.get('naive_flips_mean'):.2f} смен за матч")
        check("подбор и замер шли на РАЗНЫХ матчах",
              m.get("half_a_matches", 0) > 0 and m.get("half_b_matches", 0) > 0,
              f"A {m.get('half_a_matches')}, B {m.get('half_b_matches')}")
        check("доля сбывшихся вердиктов записана и она скромная",
              0.5 < m.get("acc_commit", 0) < 0.95,
              f"{m.get('acc_commit'):.4f} — столько стоит ранний вердикт")
        check("вердикт выносится почти во всех матчах",
              m.get("commit_rate", 0) >= 0.95,
              f"{m.get('commit_rate'):.3f}")
        check("медиана минуты вердикта в пределах разумного",
              m.get("commit_minute_p50", 99) <= 15.0,
              f"{m.get('commit_minute_p50'):.1f} мин")

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
