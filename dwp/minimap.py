"""Мини-карта в консоли: здания, снесённые здания, скопления героев.

Данные для неё в `GetRealtimeStats` уже есть и ничего не стоят: у каждого
здания и каждого игрока лежат `x` и `y`. Проверено на слепках — координаты в
диапазоне примерно ±0.34, Radiant в левом нижнем углу, Dire в правом верхнем.

ГЛАВНАЯ СЛОЖНОСТЬ. Снесённое здание из списка не пропадает, но обнуляется
ЦЕЛИКОМ: `team`, `type`, `tier`, `lane`, `x`, `y`, `heading` — всё становится
нулём. То есть по ответу нельзя сказать, какая именно вышка пала: видно только
сколько записей обнулилось.

Спасает то, что позиции зданий на карте статичны. Проверено на двух слепках
РАЗНЫХ матчей: 21 общее стоящее здание совпало по `(team, type, tier, lane,
x, y)` до четвёртого знака. Поэтому здесь ведётся справочник позиций
`data/map_buildings.json`: каждый опрос дописывает в него стоящие здания, а
снесённые определяются как «есть в справочнике, нет среди стоящих».

Справочник наполняется сам. Пока в нём не все 36 позиций, часть снесённых
зданий нарисовать нельзя — и об этом печатается строка под картой, а не
подставляется пустое место. Полным справочник становится после первого же
опроса матча, в котором ещё ничего не снесено.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import config

try:
    from rich.text import Text
    HAVE_RICH = True
except ImportError:
    HAVE_RICH = False
    Text = None                                    # type: ignore

BOOK_PATH = config.DATA_DIR / "map_buildings.json"

# Ширина вдвое больше высоты: знакоместо в терминале примерно вдвое выше,
# чем шире, а карта Dota квадратная.
W, H = 35, 17

RAD, DIRE = "green", "red"
TYPE_TOWER, TYPE_RAX, TYPE_FORT = 0, 1, 2

# (глиф юникод, глиф ascii, приоритет). Приоритет решает, что видно, когда в
# одну клетку попало несколько объектов.
GLYPH = {
    TYPE_TOWER: ("▲", "^", 1),
    TYPE_RAX: ("■", "#", 2),
    TYPE_FORT: ("★", "A", 3),
}
DEAD = ("✕", "x", 0)
LANE_DOT = ("·", ".", -1)


def _key(x: float, y: float) -> str:
    return f"{x:.4f},{y:.4f}"


def load_book(path: Path = BOOK_PATH) -> dict[str, dict]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, list):
        return {}
    return {_key(b["x"], b["y"]): b for b in raw
            if isinstance(b, dict) and "x" in b and "y" in b}


def save_book(book: dict[str, dict], path: Path = BOOK_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sorted(book.values(), key=lambda b: (b["team"], b["type"],
                                                                   b["tier"], b["lane"])),
                               ensure_ascii=False, indent=1), encoding="utf-8")


def standing(payload: dict) -> dict[str, dict]:
    """Стоящие здания из ответа. Снесённые сюда не попадают: у них всё по нулям,
    и отличить их от «здание в точке (0,0)» нельзя иначе как по флагу."""
    out = {}
    for b in (payload.get("buildings") or []):
        if not isinstance(b, dict) or b.get("destroyed"):
            continue
        if b.get("x") is None or b.get("y") is None:
            continue
        out[_key(b["x"], b["y"])] = {
            "team": int(b.get("team", 0)), "type": int(b.get("type", 0)),
            "tier": int(b.get("tier", 0)), "lane": int(b.get("lane", 0)),
            "x": float(b["x"]), "y": float(b["y"])}
    return out


def learn(payload: dict, book: dict[str, dict]) -> int:
    """Дописать в справочник позиции, которых там ещё не было."""
    new = 0
    for k, b in standing(payload).items():
        if k not in book:
            book[k] = b
            new += 1
    return new


def _bounds(book: dict[str, dict]) -> float:
    """Полуразмер квадрата карты. Считается ТОЛЬКО по зданиям: они стоят на
    месте, и картинка не будет дёргаться от того, что герой ушёл в угол."""
    r = max((max(abs(b["x"]), abs(b["y"])) for b in book.values()), default=0.35)
    return r * 1.04


def _cell(x: float, y: float, r: float) -> tuple[int, int]:
    """Экранные (строка, колонка). y переворачивается: на карте Radiant внизу,
    а строки в терминале растут сверху вниз."""
    col = int(round((x + r) / (2 * r) * (W - 1)))
    row = int(round((r - y) / (2 * r) * (H - 1)))
    return max(0, min(H - 1, row)), max(0, min(W - 1, col))


def _line(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    """Отрезок по клеткам, обычный Брезенхэм."""
    (r0, c0), (r1, c1) = a, b
    dr, dc = abs(r1 - r0), abs(c1 - c0)
    sr, sc = (1 if r1 > r0 else -1), (1 if c1 > c0 else -1)
    err = dc - dr
    out = []
    while True:
        out.append((r0, c0))
        if (r0, c0) == (r1, c1):
            return out
        e2 = 2 * err
        if e2 > -dr:
            err -= dr
            c0 += sc
        if e2 < dc:
            err += dc
            r0 += sr


def _lane_hint(book: dict[str, dict], r: float) -> list[tuple[int, int]]:
    """Клетки, по которым идут линии. Ломаная строится по вышкам одной линии:
    t3 -> t2 -> t1 своей стороны, затем t1 -> t2 -> t3 чужой. Реальная линия
    изогнута сильнее, но по трём точкам на сторону изгиб виден, и карта
    перестаёт быть россыпью значков."""
    cells: list[tuple[int, int]] = []
    for lane in (1, 2, 3):
        pts = []
        for team, tiers in ((config.TEAM_RADIANT, (3, 2, 1)),
                            (config.TEAM_DIRE, (1, 2, 3))):
            for tier in tiers:
                for b in book.values():
                    if (b["type"] == TYPE_TOWER and b["team"] == team
                            and b["lane"] == lane and b["tier"] == tier):
                        pts.append(_cell(b["x"], b["y"], r))
                        break
        for a, c in zip(pts, pts[1:]):
            cells.extend(_line(a, c))
    return cells


def render(payload: dict, book: dict[str, dict], uni: bool = True):
    """Карта построчно. Возвращает список rich.Text (или строк без rich)."""
    gi = 0 if uni else 1
    st = standing(payload)
    r = _bounds(book or st)

    grid: list[list[tuple[str, str, int]]] = [
        [(" ", "", -9) for _ in range(W)] for _ in range(H)]

    def put(row: int, col: int, glyph: str, style: str, prio: int) -> None:
        if prio >= grid[row][col][2]:
            grid[row][col] = (glyph, style, prio)

    for row, col in _lane_hint(book, r):
        put(row, col, LANE_DOT[gi], "grey35", LANE_DOT[2])

    # Снесённые: те позиции справочника, которых нет среди стоящих.
    unknown_dead = max(0, len(payload.get("buildings") or []) - len(st)
                       - sum(1 for k in book if k not in st))
    for k, b in book.items():
        if k in st:
            continue
        row, col = _cell(b["x"], b["y"], r)
        side = RAD if b["team"] == config.TEAM_RADIANT else DIRE
        put(row, col, DEAD[gi], f"dim {side}", DEAD[2])

    for b in st.values():
        row, col = _cell(b["x"], b["y"], r)
        g = GLYPH.get(b["type"])
        if g is None:
            continue
        side = RAD if b["team"] == config.TEAM_RADIANT else DIRE
        put(row, col, g[gi], side, g[2])

    # Герои: в клетку их попадает сколько угодно, поэтому рисуется ЧИСЛО, а не
    # значок каждого. Смешанная клетка — это замес, и она отмечается отдельно.
    heads: dict[tuple[int, int], list[int]] = {}
    for t in (payload.get("teams") or []):
        for p in (t.get("players") or []):
            if p.get("x") is None or p.get("y") is None:
                continue
            heads.setdefault(_cell(float(p["x"]), float(p["y"]), r), []).append(
                int(p.get("team", t.get("team_number", 0))))
    for (row, col), teams in heads.items():
        rad = sum(1 for t in teams if t == config.TEAM_RADIANT)
        dire = len(teams) - rad
        if rad and dire:
            put(row, col, "*", "bold yellow", 9)
        else:
            put(row, col, str(min(9, len(teams))),
                f"bold {RAD if rad else DIRE}", 8)

    lines = []
    for grow in grid:
        if HAVE_RICH:
            t = Text()
            for glyph, style, _ in grow:
                t.append(glyph, style=style or None)
            lines.append(t)
        else:
            lines.append("".join(g for g, _, _ in grow))
    return lines, len(book), unknown_dead


def legend(n_book: int, uni: bool = True) -> str:
    gi = 0 if uni else 1
    s = (f"{GLYPH[TYPE_TOWER][gi]} вышка  {GLYPH[TYPE_RAX][gi]} барак  "
         f"{GLYPH[TYPE_FORT][gi]} трон  {DEAD[gi]} снесено  "
         f"1-5 героев в клетке, * замес  (зелёный Radiant, красный Dire)")
    if n_book < config.N_BUILDINGS_TOTAL:
        s += (f"\nсправочник позиций: {n_book} из {config.N_BUILDINGS_TOTAL} — "
              f"снесённые вне справочника не нарисованы; он дополнится сам, "
              f"когда попадётся опрос с целой картой")
    return s
