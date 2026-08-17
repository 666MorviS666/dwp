"""Отрисовка лайв-панели.

Отдельный модуль, потому что live.py отвечает за данные, и мешать туда
разметку — значит каждый раз при правке цвета рисковать признаками.

rich не обязателен. Если его нет, вызывающий код печатает прежний
текстовый вывод: смотреть матч важнее, чем смотреть красиво.
    pip install rich
"""

from __future__ import annotations

import numpy as np

from . import config

try:
    from rich.align import Align
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich import box
    HAVE_RICH = True
except ImportError:                        # rich не установлен — не беда
    HAVE_RICH = False
    Align = Console = Group = Panel = Table = Text = box = None   # type: ignore

RAD = "green"
DIRE = "red"
SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 60) -> str:
    """История вероятности одной строкой. Шкала жёстко 0..1, а не по
    минимуму и максимуму: иначе колебание в один процент нарисуется во
    весь экран и будет выглядеть как перелом матча."""
    if not values:
        return ""
    v = values[-width:]
    return "".join(SPARK[min(7, max(0, int(p * 8)))] for p in v)


def prob_bar(p: float, width: int = 54) -> "Text":
    k = int(round(p * width))
    t = Text()
    t.append("█" * k, style=RAD)
    t.append("█" * (width - k), style=DIRE)
    return t


def clock(minute: float) -> str:
    """Игровые часы из дробной минуты.

    Через `f"{minute:.0f}"` было нельзя: это округление, а не отбрасывание, и
    на 31:40 панель показывала 32:40 — то есть врала полминуты из каждой
    минуты. NaN тоже сюда доходит (ответ без game_time, но с нетворсом), и
    раньше `int(nan)` ронял всю программу необработанным ValueError.
    """
    if minute is None or (isinstance(minute, float) and np.isnan(minute)):
        return "--:--"
    total = int(round(float(minute) * 60))
    sign = "-" if total < 0 else ""
    total = abs(total)
    return f"{sign}{total // 60}:{total % 60:02d}"


def reliability_note(minute: float) -> str:
    """Подпись о том, насколько число на экране честно ИМЕННО СЕЙЧАС.

    Общий ECE 0.011-0.016 — среднее по матчу. Человек смотрит одну
    минуту, а на 55-й расхождение впятеро больше, чем на 20-й. Молчать
    об этом хуже, чем показать грубую оценку: иначе 82% на 55-й минуте
    читается с той же уверенностью, что 82% на 20-й.
    """
    if minute is None or (isinstance(minute, float) and np.isnan(minute)):
        return ""
    for lo, hi, ece in config.RELIABILITY_BANDS:
        if minute >= lo and (hi is None or minute < hi):
            return (f"на этой минуте предсказанная доля расходится "
                    f"с фактической в среднем на {ece * 100:.0f} п.п.")
    return ""


def _fmt(v: float) -> str:
    if isinstance(v, float) and np.isnan(v):
        return "—"
    return f"{v:+,.0f}".replace(",", " ") if abs(float(v)) >= 100 else f"{float(v):+.2f}"


def _cnt(v: float) -> str:
    """Счётчик для строки со счётом: NaN — это «не знаем», а не ноль."""
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.0f}"


# Ширины колонок таблицы вкладов. Сумма плюс отступы обязана влезать в 80
# колонок вместе с рамкой панели: при 25+13+7+26 не влезала, и rich резал
# самую правую колонку — ту самую цифру вклада, ради которой таблица и нужна
# ("-0.812" превращалось в "-0.8…").
COL_LABEL, COL_VALUE, COL_CONTRIB, COL_BAR = 24, 12, 8, 20


def render(p: float, minute: float, gold: float, rows: list[dict],
           names: tuple[str, str], kills: tuple[float, float],
           towers: tuple[float, float], phist: list[float],
           warns: list[str], mmap: list | None = None,
           mmap_note: str = "") -> "Group":
    """rows: [{'label','value','contrib','pending'}], уже отсортированные.
    mmap: готовые строки мини-карты (dwp.minimap.render), либо None."""
    rn, dn = (names[0] or "Radiant"), (names[1] or "Dire")

    head = Table.grid(expand=True)
    head.add_column(justify="left")
    head.add_column(justify="center")
    head.add_column(justify="right")
    head.add_row(
        Text(f"{rn}", style=f"bold {RAD}"),
        Text(clock(minute), style="bold white"),
        Text(f"{dn}", style=f"bold {DIRE}"))
    head.add_row(
        Text(f"{p * 100:.1f}%", style=f"bold {RAD}"),
        Text(f"{gold:+,.0f} золота".replace(",", " ") if not np.isnan(gold) else "—",
             style="yellow"),
        Text(f"{(1 - p) * 100:.1f}%", style=f"bold {DIRE}"))

    score = Table.grid(expand=True)
    score.add_column(justify="left")
    score.add_column(justify="center")
    score.add_column(justify="right")
    score.add_row(Text(f"фраги {_cnt(kills[0])}", style="dim"),
                  Text(f"вышек потеряно {_cnt(towers[0])} : {_cnt(towers[1])}",
                       style="dim"),
                  Text(f"{_cnt(kills[1])} фраги", style="dim"))

    tbl = Table(box=box.SIMPLE_HEAD, pad_edge=False, header_style="dim",
                show_edge=False, expand=False)
    tbl.add_column("что двигает число", width=COL_LABEL, no_wrap=True)
    tbl.add_column("значение", justify="right", width=COL_VALUE, no_wrap=True)
    tbl.add_column("вклад", justify="right", width=COL_CONTRIB, no_wrap=True)
    tbl.add_column("", justify="left", width=COL_BAR, no_wrap=True)
    scale = max(0.3, max((abs(r["contrib"]) for r in rows), default=0.3))
    half = (COL_BAR - 1) // 2
    for r in rows:
        c = float(r["contrib"])
        n = int(round(abs(c) / scale * half))
        if c >= 0:
            bar = Text(" " * half + "│", style="dim")
            bar.append("█" * n, style=RAD)
        else:
            bar = Text(" " * (half - n), style="dim")
            bar.append("█" * n, style=DIRE)
            bar.append("│", style="dim")
        label = Text(r["label"], style="yellow" if r["pending"] else "")
        val = Text(r["value"] if isinstance(r["value"], str) else _fmt(r["value"]),
                   style="yellow" if r["pending"] else "dim")
        tbl.add_row(label, val, Text(f"{c:+.3f}", style="dim"), bar)

    parts = [head, score, Text(""), prob_bar(p)]
    note = reliability_note(minute)
    if note:
        parts.append(Text(note, style="dim"))
    if len(phist) > 1:
        parts.append(Text(sparkline(phist), style="cyan"))
    if mmap:
        parts.append(Text(""))
        parts.append(Align.center(Group(*mmap)))
        if mmap_note:
            parts.append(Text(mmap_note, style="dim"))
    parts += [Text(""), tbl]
    if warns:
        parts.append(Text("\n".join("· " + w for w in warns), style="dim yellow"))
    return Group(*parts)


def panel(*args, **kw) -> "Panel":
    return Panel(render(*args, **kw), box=box.ROUNDED, border_style="dim",
                 title="dwp", title_align="left")
