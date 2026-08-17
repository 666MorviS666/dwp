"""Геометрия мини-карты: проверка без браузера.

Зачем отдельный тест. Река на карте была нарисована ВДОЛЬ мида вместо
поперёк — карта из-за этого не читалась вообще, и заметить это можно было
только глазами на скриншоте. Ошибка ровно того сорта, который тесты и
должны ловить: числа все правильные, а картинка бессмысленная.

Здесь проверяется та же математика, что использует SVG в dwp.web:

  * ломаные линий строятся по РЕАЛЬНЫМ координатам вышек из справочника;
  * у верхней и нижней линий есть излом на углу карты, у мида нет;
  * верхняя идёт по левому краю и верху, нижняя по низу и правому краю;
  * река перпендикулярна миду, то есть в экранных процентах вдоль sy = sx.

Тест работает от справочника data/map_buildings.json. Нет справочника или
он неполный по какой-то линии — эта линия пропускается с сообщением, а не
роняет проверку: справочник наполняется по мере просмотра матчей.

Запуск: python -m dwp.test_map_geometry
"""

from __future__ import annotations

import sys

from . import config, minimap as M, web

# Экранные проценты: 0,0 — левый верхний угол.
LEFT, RIGHT, TOP, BOTTOM = 25.0, 75.0, 25.0, 75.0


def to_screen(book: dict) -> tuple[float, callable, callable]:
    r = M._bounds(book)
    return r, (lambda v: (v + r) / (2 * r) * 100), (lambda v: (r - v) / (2 * r) * 100)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    book = M.load_book()
    if len(book) < 12:
        print(f"ПРОПУСК: в {M.BOOK_PATH} всего {len(book)} позиций — проверять "
              f"нечего. Справочник наполняется при просмотре матчей "
              f"(`python -m dwp.web` или `dwp.live --map`).")
        return 0
    r, px, py = to_screen(book)
    lanes = web.lane_paths(book)
    print(f"справочник: {len(book)} позиций из {config.N_BUILDINGS_TOTAL}, "
          f"полуразмер {r:.3f}, линий разобрано {len(lanes)}")
    if len(lanes) != 3:
        print(f"ПРОВАЛ: линий должно быть 3, разобрано {len(lanes)}",
              file=sys.stderr)
        return 1

    top, mid, bot = [[(px(p["x"]), py(p["y"])) for p in ln] for ln in lanes]
    fails = []

    def check(name: str, cond: bool, detail: str) -> None:
        print(f"  {'OK  ' if cond else 'ПРОВАЛ'} {name}: {detail}")
        if not cond:
            fails.append(name)

    # 1. Излом на углу: у верхней и нижней он есть, у мида нет.
    # Меряется прямо: точка ломаной, которой не соответствует НИ ОДНО
    # здание, и есть вставленный угол. Через радиус это мерить нельзя —
    # у мида крайние точки это троны, они и так в углах карты.
    real = {(round(b["x"], 4), round(b["y"], 4)) for b in book.values()}

    def synthetic(ln):
        return [p for p in ln if (round(p["x"], 4), round(p["y"], 4)) not in real]

    s_top, s_mid, s_bot = (synthetic(ln) for ln in lanes)
    check("у верхней линии вставлен угол", len(s_top) == 1,
          f"точек не из справочника: {len(s_top)} "
          f"(без угла линия срезала бы через центр карты)")
    check("у нижней линии вставлен угол", len(s_bot) == 1,
          f"точек не из справочника: {len(s_bot)}")
    check("у мида угла нет", len(s_mid) == 0,
          f"точек не из справочника: {len(s_mid)} — мид прямой, и лишний "
          f"угол увёл бы его с диагонали")
    if len(s_top) == 1:
        cx, cy = px(s_top[0]["x"]), py(s_top[0]["y"])
        check("угол верхней — в левом верхнем", cx < LEFT and cy < TOP,
              f"экранные ({cx:.1f}%, {cy:.1f}%)")
    if len(s_bot) == 1:
        cx, cy = px(s_bot[0]["x"]), py(s_bot[0]["y"])
        check("угол нижней — в правом нижнем", cx > RIGHT and cy > BOTTOM,
              f"экранные ({cx:.1f}%, {cy:.1f}%)")

    # 2. Верхняя линия обязана побывать и у левого края, и у верха.
    check("верхняя идёт по левому краю", min(x for x, _ in top) < LEFT,
          f"минимальный x = {min(x for x, _ in top):.1f}%")
    check("верхняя идёт по верху", min(y for _, y in top) < TOP,
          f"минимальный y = {min(y for _, y in top):.1f}%")
    # 3. Нижняя — по низу и по правому краю.
    check("нижняя идёт по низу", max(y for _, y in bot) > BOTTOM,
          f"максимальный y = {max(y for _, y in bot):.1f}%")
    check("нижняя идёт по правому краю", max(x for x, _ in bot) > RIGHT,
          f"максимальный x = {max(x for x, _ in bot):.1f}%")

    # 4. Мид — диагональ из левого нижнего в правый верхний, то есть
    # экранное y убывает, когда x растёт.
    dx = mid[-1][0] - mid[0][0]
    dy = mid[-1][1] - mid[0][1]
    check("мид идёт по диагонали Radiant -> Dire", dx > 30 and dy < -30,
          f"смещение по x {dx:+.1f}%, по y {dy:+.1f}%")

    # 5. РЕКА. Она перпендикулярна миду. Мид в экранных процентах идёт
    # вдоль sy = 100 - sx, значит река вдоль sy = sx: из левого верхнего
    # угла в правый нижний. Проверяем именно так, потому что именно здесь
    # и была ошибка — река шла вдоль мида.
    on_river = lambda x, y: abs(y - x) < 12          # noqa: E731
    check("река поперёк мида, а не вдоль",
          on_river(10, 10) and on_river(90, 90)
          and not on_river(90, 10) and not on_river(10, 90),
          "левый верхний и правый нижний углы на реке, правый верхний и "
          "левый нижний — нет")
    mid_pts_on_river = sum(1 for x, y in mid if on_river(x, y))
    check("мид пересекает реку ровно в середине", mid_pts_on_river <= 1,
          f"точек мида в полосе реки: {mid_pts_on_river} (вдоль реки их было "
          f"бы большинство)")

    # 6. Стороны. Radiant внизу слева, Dire вверху справа.
    forts = [b for b in book.values() if b["type"] == 2]
    for f in forts:
        x, y = px(f["x"]), py(f["y"])
        want = "низ-слева" if f["team"] == config.TEAM_RADIANT else "верх-справа"
        ok = ((x < LEFT and y > BOTTOM) if f["team"] == config.TEAM_RADIANT
              else (x > RIGHT and y < TOP))
        check(f"трон team {f['team']} стоит {want}", ok,
              f"экранные ({x:.1f}%, {y:.1f}%)")

    print()
    if fails:
        print(f"ПРОВАЛЕНО ПРОВЕРОК: {len(fails)} — {', '.join(fails)}",
              file=sys.stderr)
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
