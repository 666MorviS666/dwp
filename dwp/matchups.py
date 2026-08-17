"""Кто против кого хорош: статистика противостояний героев с OpenDota.

Зачем отдельно от модели. Драфт-модель здесь ЛИНЕЙНАЯ: у неё по одному
коэффициенту на героя и ни одного парного члена. Она физически не знает, кто
кого контрит и кто с кем сочетается, и рисовать такие стрелки «по модели»
значило бы выдумывать. Синергии и контрпики честно вынесены в «Что
доработать» как незакрытая задача.

Но показать зрителю противостояния можно — из данных, а не из модели.
OpenDota отдаёт `/heroes/{id}/matchups`: по каждому сопернику число сыгранных
матчей и число побед, посчитанное на всей их выгрузке. Числа большие
(десятки и сотни тысяч игр на пару), поэтому в отличие от наших 7734 матчей
это не шум.

ГРАНИЦА, которую нельзя размывать: эти числа НЕ участвуют в предсказании.
Панель показывает их отдельным блоком и подписывает источник. Иначе зритель
решит, что модель их учла, а она нет.

Кэш на диске: пары меняются от патча к патчу, но не за час. Повторный запуск
в сеть не ходит.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from . import config
from .collect import ApiError, OpenDotaClient

CACHE_TTL_DAYS = 14


def _path(hero_id: int, directory: Path | None = None) -> Path:
    return (directory or config.MATCHUPS_DIR) / f"{hero_id}.json"


def load(hero_id: int, directory: Path | None = None) -> list[dict] | None:
    """Кэш с диска, если он не протух. Иначе None."""
    p = _path(hero_id, directory)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > CACHE_TTL_DAYS * 86400:
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, list) else None


def fetch(hero_id: int, client: OpenDotaClient,
          directory: Path | None = None) -> list[dict] | None:
    """Добрать с OpenDota и положить в кэш. None — не получилось."""
    try:
        data = client.get(f"/heroes/{hero_id}/matchups")
    except ApiError:
        return None
    if not isinstance(data, list):
        return None
    d = directory or config.MATCHUPS_DIR
    d.mkdir(parents=True, exist_ok=True)
    _path(hero_id, d).write_text(json.dumps(data, ensure_ascii=False),
                                 encoding="utf-8")
    return data


def table(hero_ids: list[int], client: OpenDotaClient | None = None,
          directory: Path | None = None) -> dict[int, dict[int, tuple[int, int]]]:
    """hero_id -> {соперник: (сыграно, побед этого героя)}.

    Сеть трогается только для тех героев, у кого кэш пуст или протух.
    """
    out: dict[int, dict[int, tuple[int, int]]] = {}
    need = []
    for hid in hero_ids:
        data = load(hid, directory)
        if data is None:
            need.append(hid)
        else:
            out[hid] = _index(data)
    if need and client is not None:
        for hid in need:
            data = fetch(hid, client, directory)
            if data is not None:
                out[hid] = _index(data)
    return out


def _index(data: list[dict]) -> dict[int, tuple[int, int]]:
    idx = {}
    for row in data:
        try:
            idx[int(row["hero_id"])] = (int(row["games_played"]), int(row["wins"]))
        except (KeyError, TypeError, ValueError):
            continue
    return idx


MIN_GAMES = 30          # меньше — считать долю бессмысленно вовсе
SIGMAS = 2.0            # во сколько стандартных ошибок должен уложиться перевес


def _se(games: int) -> float:
    """Стандартная ошибка доли побед. Худший случай p=0.5, он же и типичный."""
    return (0.25 / games) ** 0.5 if games > 0 else float("inf")


def grid(radiant: list[int], dire: list[int],
         tbl: dict[int, dict[int, tuple[int, int]]]) -> dict:
    """Сетка 5x5: перевес героя Radiant против героя Dire.

    ГЛАВНОЕ ЗДЕСЬ — НЕ ДОЛЯ, А ЕЁ ОШИБКА. Замерено на живом составе: у
    OpenDota на пару героев приходится порядка 240 матчей, то есть
    стандартная ошибка около 3.2 п.п. Разница в 1-2 п.п. на такой выборке
    неотличима от нуля, и печатать её как «контрит» — врать числом.

    Поэтому у каждой клетки есть `sig`: перевес больше SIGMAS стандартных
    ошибок. Панель показывает число только у значимых, остальные — «≈».
    """
    rows = []
    for r in radiant:
        cells = []
        for d in dire:
            games, wins = tbl.get(r, {}).get(d, (0, 0))
            if games < MIN_GAMES:
                cells.append({"vs": d, "games": games, "wr": None,
                              "se": None, "sig": False})
                continue
            wr = wins / games
            se = _se(games)
            cells.append({"vs": d, "games": games, "wr": wr, "se": se,
                          "sig": abs(wr - 0.5) > SIGMAS * se})
        known = [c for c in cells if c["wr"] is not None]
        # Средний перевес против всего состава, взвешенный по числу игр:
        # редкие пары не должны тянуть итог наравне с частыми. Ошибка
        # среднего меньше, поэтому итог бывает значим там, где ни одна
        # отдельная клетка не значима.
        tot = sum(c["games"] for c in known)
        avg = (sum(c["wr"] * c["games"] for c in known) / tot) if tot else None
        avg_se = _se(tot) if tot else None
        rows.append({"hero": r, "cells": cells, "avg": avg, "games": tot,
                     "se": avg_se,
                     "sig": bool(avg is not None
                                 and abs(avg - 0.5) > SIGMAS * avg_se)})
    all_games = [c["games"] for row in rows for c in row["cells"] if c["games"]]
    med = sorted(all_games)[len(all_games) // 2] if all_games else 0

    # Состав против состава. Ради этого числа всё и затевалось: отдельная
    # пара героев набирает у OpenDota порядка сотни матчей, а все 25 пар
    # вместе — тысячи, и ошибка падает в пять раз. То есть на уровне пары
    # сказать почти нечего, а на уровне состава — уже есть что.
    known = [c for row in rows for c in row["cells"] if c["wr"] is not None]
    tot = sum(c["games"] for c in known)
    team = sum(c["wr"] * c["games"] for c in known) / tot if tot else None
    team_se = _se(tot) if tot else None
    return {"radiant": radiant, "dire": dire, "rows": rows,
            "min_games": MIN_GAMES, "sigmas": SIGMAS, "median_games": med,
            "median_se": _se(med) if med else None,
            "team": {"wr": team, "games": tot, "se": team_se, "pairs": len(known),
                     "sig": bool(team is not None
                                 and abs(team - 0.5) > SIGMAS * team_se)}}
