"""Запись каждого лайв-опроса на диск.

Зачем. Все метрики этого проекта офлайновые: они меряют модель на выгрузке
OpenDota. Точность того, что реально показано на экране во время матча, не
измерялась никогда — и измерить её нечем, потому что опросы никуда не
сохранялись. Лог закрывает эту дыру: `dwp.livecheck` потом добирает исход
матча и считает калибровку по фактически показанным числам.

Заодно в лог пишется сырьё для двух открытых вопросов, ответить на которые из
завершённых матчей нельзя:

  * `graph_gold_last` против `nw_adv` — какая из двух величин совпадает с
    `radiant_gold_adv`, на котором обучен признак `gold_adv`;
  * `game_time` — включает ли он 90 секунд до горна.

Формат — CSV на матч, дописывание строк. Не JSON: лог должен переживать
падение процесса посреди матча, а недописанный JSON не прочитать.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Колонки, которые пишутся всегда, до признаков модели. Порядок фиксирован:
# файл дописывается между запусками, и переставлять колонки нельзя.
HEAD = ["ts", "match_id", "server_steam_id", "game_state", "game_time", "p"]
# Диагностика для открытых вопросов — после признаков.
TAIL = ["graph_gold_last", "graph_gold_len", "nw_adv", "nw_total",
        "team_id_radiant", "team_id_dire",
        # Вероятность Stratz по тому же матчу и та же минута по их часам.
        # Ради этой пары колонок всё и заводилось: только так можно
        # посчитать log loss двух моделей на ОДНИХ И ТЕХ ЖЕ минутах, а не
        # сравнивать наши числа с чужими по памяти.
        "p_stratz", "stratz_time",
        "model"]


def _num(v) -> str:
    """Пустая клетка вместо nan: в CSV «nan» и «неизвестно» лучше не путать."""
    if v is None:
        return ""
    if isinstance(v, float) and np.isnan(v):
        return ""
    return f"{v:.6g}" if isinstance(v, float) else str(v)


class LiveLog:
    """Один файл на матч. Открывается лениво: пока матч не опознан — не пишем."""

    def __init__(self, directory: Path, model: str, features: list[str]):
        self.dir = directory
        self.model = model
        self.features = list(features)
        self.cols = HEAD + self.features + TAIL
        self.path: Path | None = None
        self.rows = 0

    def _open(self, st: dict) -> Path | None:
        """Файл с ПОДХОДЯЩЕЙ шапкой, иначе новый.

        Иначе случается вот что: два `--watch` с разными моделями на один и
        тот же матч пишут в один файл, у моделей разный набор признаков — и
        в CSV оказываются строки на 28 и на 34 колонки вперемешку. Такой
        файл не читается вообще ничем, то есть теряются обе записи сразу.
        Случилось на живом матче, а не в теории.
        """
        mid = st.get("match_id")
        sid = st.get("server_steam_id")
        if not mid and not sid:
            return None
        base = str(mid) if mid else f"server_{sid}"
        self.dir.mkdir(parents=True, exist_ok=True)
        stem = Path(self.model).stem
        for name in (f"{base}.csv", f"{base}__{stem}.csv",
                     *(f"{base}__{stem}_{i}.csv" for i in range(2, 20))):
            path = self.dir / name
            if not path.exists():
                with path.open("w", encoding="utf-8", newline="") as fh:
                    csv.writer(fh).writerow(self.cols)
                return path
            if self._header_of(path) == self.cols:
                return path
        return None

    @staticmethod
    def _header_of(path: Path) -> list[str] | None:
        try:
            with path.open(encoding="utf-8", newline="") as fh:
                return next(csv.reader(fh), None)
        except OSError:
            return None

    def write(self, st: dict, row: pd.Series, p: float,
              stratz: dict | None = None) -> None:
        if self.path is None:
            self.path = self._open(st)
            if self.path is None:
                return
        nws = st.get("player_nw") or {}
        if len(nws) == 2:
            nw_r = float(np.sum(nws[config.TEAM_RADIANT]))
            nw_d = float(np.sum(nws[config.TEAM_DIRE]))
            nw_adv, nw_total = nw_r - nw_d, nw_r + nw_d
        else:
            nw_adv = nw_total = np.nan
        tids = st.get("team_ids") or {}
        values = (
            [f"{time.time():.0f}", st.get("match_id") or "",
             st.get("server_steam_id") or "", _num(st.get("game_state")),
             _num(st.get("game_time")), f"{p:.6f}"]
            + [_num(row.get(f)) for f in self.features]
            + [_num(st.get("graph_gold_last")), _num(st.get("graph_gold_len")),
               _num(nw_adv), _num(nw_total),
               _num(tids.get(config.TEAM_RADIANT)), _num(tids.get(config.TEAM_DIRE)),
               _num((stratz or {}).get("p")), _num((stratz or {}).get("time")),
               self.model]
        )
        with self.path.open("a", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(values)
        self.rows += 1

    def note(self) -> str:
        if self.path is None:
            return "лог: матч не опознан (нет match_id и server_steam_id)"
        return f"лог: {self.rows} строк -> {self.path}"
