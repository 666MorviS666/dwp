"""Парные данные по героям из Stratz: контры и синергии.

Зачем понадобился второй источник. У OpenDota на пару героев приходится
порядка 80-240 матчей — стандартная ошибка доли побед 3-6 п.п., и разница в
пару процентов неотличима от нуля. Замерено на живом составе: значимых пар
оказалось 2 из 25. То есть вопрос «кто кого контрит» на тех данных просто не
имеет ответа.

У Stratz на ту же пару приходится порядка 7000 матчей (замерено: медиана
6906 у `vs`, 5353 у `with`), ошибка падает до 0.6 п.п. — и вопрос ответ
получает. Плюс там есть `with`, то есть синергия, которой у OpenDota нет
вовсе.

СЕМАНТИКА, проверенная на данных, а не по документации: `winCount` — победы
ГЕРОЯ `heroId` против (или вместе с) `heroId2`. Сверка: Spectre против
Bloodseeker 53.86%, Bloodseeker против Spectre 46.13%, сумма 99.99%.
`matchCount` у зеркальных строк расходится на десятые доли процента — это
шум сбора, не ошибка.

ТОКЕН берётся только из переменной окружения STRATZ_TOKEN и никогда не
пишется в файлы проекта: он личный, а data/ уходит в бэкапы и репозитории.
    setx STRATZ_TOKEN ваш_токен      (после setx откройте окно заново)

Лимиты (из заголовков ответа): секунда ~8, минута 150, час 1500, сутки 15000.
Справочник из 127 героев добирается тринадцатью запросами и кэшируется на
диск, так что в сеть модуль ходит редко.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

from . import config

TOKEN_ENV = "STRATZ_TOKEN"
URL = "https://api.stratz.com/graphql"
CACHE_TTL_DAYS = 14
BATCH = 8                 # героев в одном запросе
MIN_INTERVAL = 0.4        # с, лимит по секунде маленький


class StratzError(RuntimeError):
    """Сетевая или протокольная ошибка с готовым советом."""


def token() -> str | None:
    return os.environ.get(TOKEN_ENV) or None


def _path(hero_id: int, directory: Path | None = None) -> Path:
    return (directory or config.STRATZ_DIR) / f"{hero_id}.json"


def load(hero_id: int, directory: Path | None = None) -> dict | None:
    p = _path(hero_id, directory)
    if not p.exists():
        return None
    if (time.time() - p.stat().st_mtime) > CACHE_TTL_DAYS * 86400:
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) and "vs" in d else None


QUERY = """
{ heroStats { matchUp(heroIds: [%s], take: 200) {
    heroId
    vs   { heroId2 matchCount winCount }
    with { heroId2 matchCount winCount }
} } }"""


class Client:
    def __init__(self, tok: str | None = None, verbose: bool = False):
        self.token = tok or token()
        if not self.token:
            raise StratzError(
                f"Нет токена Stratz.\nЧто делать: получите его на "
                f"https://stratz.com/api и выполните\n"
                f"    setx {TOKEN_ENV} ваш_токен\n"
                f"затем откройте окно заново. Без токена панель показывает "
                f"пары по данным OpenDota — их на порядок меньше.")
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.token}",
            # Stratz отклоняет запросы с дефолтным User-Agent библиотеки.
            "User-Agent": "STRATZ_API",
            "Content-Type": "application/json"})
        self.verbose = verbose
        self._last = 0.0

    def _throttle(self) -> None:
        wait = MIN_INTERVAL - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    def raw(self, query: str) -> dict:
        """Произвольный запрос с той же политикой лимитов и ошибок."""
        self._throttle()
        try:
            r = self.session.post(URL, json={"query": query}, timeout=60)
        except requests.RequestException as e:
            raise StratzError(f"Stratz недоступен: {type(e).__name__}: {e}") from e
        if r.status_code == 429:
            time.sleep(float(r.headers.get("Retry-After") or 5))
            return self.raw(query)
        if r.status_code in (401, 403):
            raise StratzError(
                f"Stratz вернул {r.status_code}: токен не принят или истёк.\n"
                f"Что делать: перевыпустите его на https://stratz.com/api и "
                f"обновите переменную {TOKEN_ENV}.")
        if r.status_code != 200:
            raise StratzError(f"Stratz вернул {r.status_code}: {r.text[:200]}")
        d = r.json()
        if "errors" in d:
            raise StratzError(f"Stratz: {json.dumps(d['errors'])[:300]}")
        return d

    def query(self, hero_ids: list[int]) -> list[dict]:
        self._throttle()
        body = {"query": QUERY % ", ".join(str(int(h)) for h in hero_ids)}
        try:
            r = self.session.post(URL, json=body, timeout=60)
        except requests.RequestException as e:
            raise StratzError(f"Stratz недоступен: {type(e).__name__}: {e}") from e
        if r.status_code == 429:
            wait = float(r.headers.get("Retry-After") or 5)
            if self.verbose:
                print(f"  [лимит Stratz] ждём {wait:.0f} с")
            time.sleep(wait)
            return self.query(hero_ids)
        if r.status_code in (401, 403):
            raise StratzError(
                f"Stratz вернул {r.status_code}: токен не принят или истёк.\n"
                f"Что делать: перевыпустите его на https://stratz.com/api и "
                f"обновите переменную {TOKEN_ENV}.")
        if r.status_code != 200:
            raise StratzError(f"Stratz вернул {r.status_code}: {r.text[:200]}")
        d = r.json()
        if "errors" in d:
            raise StratzError(f"Stratz: {json.dumps(d['errors'])[:300]}")
        rows = (((d.get("data") or {}).get("heroStats") or {}).get("matchUp"))
        if rows is None:
            raise StratzError(
                f"В ответе Stratz нет heroStats.matchUp. Верхние ключи: "
                f"{sorted(d)}\nЧто делать: схема GraphQL изменилась, "
                f"поправьте QUERY в dwp/stratz.py.")
        return rows


def _index(rows: list[dict], key: str) -> dict[int, tuple[int, int]]:
    out = {}
    for v in (rows or []):
        try:
            out[int(v["heroId2"])] = (int(v["matchCount"]), int(v["winCount"]))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def table(hero_ids: list[int], client: Client | None = None,
          directory: Path | None = None,
          verbose: bool = False) -> dict[int, dict]:
    """hero_id -> {"vs": {другой: (сыграно, побед)}, "with": то же}.

    Сначала кэш; в сеть только за тем, чего нет. client=None — работаем
    исключительно по кэшу, ни одного запроса.
    """
    out: dict[int, dict] = {}
    need = []
    for hid in dict.fromkeys(int(h) for h in hero_ids):
        d = load(hid, directory)
        if d is None:
            need.append(hid)
        else:
            out[hid] = {"vs": {int(k): tuple(v) for k, v in d["vs"].items()},
                        "with": {int(k): tuple(v) for k, v in d["with"].items()}}
    if not need or client is None:
        return out
    dirp = directory or config.STRATZ_DIR
    dirp.mkdir(parents=True, exist_ok=True)
    for i in range(0, len(need), BATCH):
        chunk = need[i:i + BATCH]
        if verbose:
            print(f"  Stratz: {i + len(chunk)}/{len(need)} героев")
        for row in client.query(chunk):
            hid = int(row["heroId"])
            rec = {"vs": _index(row.get("vs"), "vs"),
                   "with": _index(row.get("with"), "with")}
            _path(hid, dirp).write_text(
                json.dumps({"vs": {str(k): list(v) for k, v in rec["vs"].items()},
                            "with": {str(k): list(v) for k, v in rec["with"].items()}}),
                encoding="utf-8")
            out[hid] = rec
    return out


# --- Вероятность Stratz по идущему матчу --------------------------------

LIVE_QUERY = """
{ live { match(id: %d) {
    matchId gameTime gameMinute radiantScore direScore completed
    winRateValues
    liveWinRateValues { time winRate }
} } }"""

# ДОПУЩЕНИЕ, НЕ ПРОВЕРЕННОЕ НА ДАННЫХ: winRateValues — вероятность победы
# RADIANT. Так это подписано в интерфейсе Stratz, но в схеме GraphQL стороны
# нет, и перепутать её ничего не мешает. Проверяется эмпирически в
# dwp.livecheck: у досмотренных матчей последнее значение должно быть выше
# 0.5 там, где выиграл Radiant. Пока проверка не набрала матчей, число
# показывается с пометкой.
LIVE_SIDE_IS_RADIANT = True


def live_win_rate(match_id: int, client: Client) -> dict | None:
    """Последняя вероятность Stratz по идущему матчу.

    None означает «Stratz этот матч не ведёт» — обычная ситуация: живьём он
    отслеживает лиговые матчи и высокий рейтинг, а рядовой паблик может не
    попасть вовсе. Это не ошибка и не должно ничего ронять.
    """
    rows = client.raw(LIVE_QUERY % int(match_id))
    m = ((rows.get("data") or {}).get("live") or {}).get("match")
    if not isinstance(m, dict):
        return None
    series = m.get("liveWinRateValues") or []
    if series:
        last = series[-1]
        p = last.get("winRate")
        t = last.get("time")
        src = "liveWinRateValues"
    else:
        vals = m.get("winRateValues")
        if not isinstance(vals, list) or not vals:
            return None
        p, t, src = vals[-1], m.get("gameTime"), "winRateValues"
    try:
        p = float(p)
    except (TypeError, ValueError):
        return None
    # Встречается и доля, и проценты — приводим к доле, но только явный
    # случай, чтобы не «починить» верное число.
    if p > 1.0:
        p /= 100.0
    if not (0.0 <= p <= 1.0):
        return None
    return {"p": p, "time": t, "points": len(series), "source": src,
            "game_time": m.get("gameTime"), "minute": m.get("gameMinute"),
            "completed": m.get("completed")}


# --- Сведение к числам, которые показывает панель -----------------------

SIGMAS = 2.0
MIN_GAMES = 300           # у Stratz это заведомо выполняется почти везде


def _se(n: int) -> float:
    return (0.25 / n) ** 0.5 if n > 0 else float("inf")


def pair(tbl: dict, kind: str, a: int, b: int) -> dict:
    """Одна клетка: доля побед a против/вместе с b, с ошибкой и значимостью."""
    n, w = (tbl.get(a, {}).get(kind, {}) or {}).get(b, (0, 0))
    if n < MIN_GAMES:
        return {"vs": b, "games": n, "wr": None, "se": None, "sig": False}
    wr, se = w / n, _se(n)
    return {"vs": b, "games": n, "wr": wr, "se": se,
            "sig": abs(wr - 0.5) > SIGMAS * se}


def grid(radiant: list[int], dire: list[int], tbl: dict) -> dict:
    """Сетка контров 5x5 плюс синергии внутри каждой команды."""
    rows = []
    for r in radiant:
        cells = [pair(tbl, "vs", r, d) for d in dire]
        known = [c for c in cells if c["wr"] is not None]
        tot = sum(c["games"] for c in known)
        avg = sum(c["wr"] * c["games"] for c in known) / tot if tot else None
        se = _se(tot) if tot else None
        rows.append({"hero": r, "cells": cells, "avg": avg, "games": tot,
                     "se": se,
                     "sig": bool(avg is not None and abs(avg - 0.5) > SIGMAS * se)})
    known = [c for row in rows for c in row["cells"] if c["wr"] is not None]
    tot = sum(c["games"] for c in known)
    team = sum(c["wr"] * c["games"] for c in known) / tot if tot else None
    team_se = _se(tot) if tot else None

    def syn(side: list[int]) -> dict:
        ps = [pair(tbl, "with", a, b)
              for i, a in enumerate(side) for b in side[i + 1:]]
        ok = [p for p in ps if p["wr"] is not None]
        n = sum(p["games"] for p in ok)
        wr = sum(p["wr"] * p["games"] for p in ok) / n if n else None
        s = _se(n) if n else None
        return {"wr": wr, "games": n, "se": s, "pairs": len(ok),
                "sig": bool(wr is not None and abs(wr - 0.5) > SIGMAS * s),
                "cells": ps}

    all_games = [c["games"] for row in rows for c in row["cells"] if c["games"]]
    med = sorted(all_games)[len(all_games) // 2] if all_games else 0
    return {"radiant": radiant, "dire": dire, "rows": rows,
            "min_games": MIN_GAMES, "sigmas": SIGMAS, "median_games": med,
            "median_se": _se(med) if med else None,
            "team": {"wr": team, "games": tot, "se": team_se,
                     "pairs": len(known),
                     "sig": bool(team is not None
                                 and abs(team - 0.5) > SIGMAS * team_se)},
            "synergy": {"radiant": syn(radiant), "dire": syn(dire)},
            "source": "Stratz"}
