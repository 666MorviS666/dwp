"""Лента событий матча, собранная из последовательных опросов.

ЗАЧЕМ ОТДЕЛЬНЫЙ МОДУЛЬ. В `GetRealtimeStats` журнала событий нет вовсе:
ни строчки «A убил B», ни времени последнего фрага. Есть только счётчики
`kill_count` / `death_count` / `assists_count` у каждого игрока и список
зданий. Значит лента не читается, а ВЫВОДИТСЯ из разницы между соседними
опросами — и вместе с ней выводится ровно та неопределённость, которой в
настоящем журнале не было бы.

ЧТО ЗДЕСЬ ЧЕСТНО, А ЧТО НЕТ.

* Смерть героя — факт: `death_count` вырос. Спорить не с чем.
* Фраг у противника — факт: `kill_count` вырос.
* А вот КТО КОГО убил — вывод, и он однозначен только когда в окне
  ровно одна смерть и ровно один фраг. Две смерти и два фрага в одном
  замесе разложить по парам нечем: таких полей в источнике нет.
  Поэтому событие помечается `certain=False`, и панель показывает его
  как замес «трое погибли, фраги у двоих», а не выдумывает пары.
* Смерть без единого фрага у противника — это вышка, крипы, Рошан или
  денай. Своей команде фраг за такое не идёт, и подставлять «наверное,
  убил кто-то рядом» нельзя. Пишем «не от героя».
* Расстояние до ближайшего врага СЧИТАЕТСЯ и отдаётся отдельным полем
  `guess`, но помечено как догадка и в текст события не идёт. Координата
  мёртвого героя — это, судя по слепкам, место смерти, но проверить это
  нечем, а «судя по» в подпись к фрагу ставить нельзя.

Здания разбираются точнее: позиция здания статична (см. `minimap`), и
исчезновение позиции из числа стоящих однозначно называет и сторону, и
ярус, и линию. Тут догадок нет.

Окно склейки. Опрос идёт раз в 1-2 с, а замес длится дольше; если
разбирать каждый опрос отдельно, один бой распадётся на четыре события с
разной атрибуцией. Поэтому изменения копятся `WINDOW` игровых секунд и
разбираются вместе — так атрибуция получается на всём бое сразу, а не на
случайном его срезе.
"""

from __future__ import annotations

import math

from . import config, minimap as M

# Сколько ИГРОВЫХ секунд копить изменения перед разбором. Меньше секунды
# бессмысленно: источник обновляется примерно раз в 1.2 с. Больше пяти —
# лента начнёт отставать от картинки настолько, что зритель заметит.
WINDOW = 3.0

# Сколько событий держать. Лента живёт в памяти панели и уходит в браузер
# целиком на каждом опросе, поэтому длина ограничена.
MAX_EVENTS = 80

TYPE_TOWER, TYPE_RAX, TYPE_FORT = 0, 1, 2
LANE_NAME = {1: "низ", 2: "мид", 3: "верх"}
# Винительный падеж: строка собирается как «Radiant потерял ...».
TYPE_NAME = {TYPE_TOWER: "вышку", TYPE_RAX: "барак", TYPE_FORT: "трон"}


def _players(payload: dict) -> dict:
    """Счётчики и позиции игроков по устойчивому ключу.

    Ключ — (команда, слот): `accountid` в паблике бывает нулевым у всех
    сразу, а `heroid` до конца драфта равен нулю. Слот же не меняется.
    """
    out = {}
    for t in (payload.get("teams") or []):
        tn = t.get("team_number")
        if tn is None:
            continue
        for i, p in enumerate(t.get("players") or []):
            slot = p.get("team_slot")
            key = (int(tn), int(slot) if slot is not None else i)
            out[key] = {
                "team": int(tn),
                "hero": int(p.get("heroid") or p.get("hero_id") or 0),
                "name": str(p.get("name") or ""),
                "k": int(p.get("kill_count") or 0),
                "d": int(p.get("death_count") or 0),
                "a": int(p.get("assists_count") or 0),
                "x": p.get("x"), "y": p.get("y"),
            }
    return out


def _dist(a: dict, b: dict) -> float | None:
    if None in (a.get("x"), a.get("y"), b.get("x"), b.get("y")):
        return None
    return math.hypot(float(a["x"]) - float(b["x"]), float(a["y"]) - float(b["y"]))


class Feed:
    """Лента одного матча. При смене матча заводится новая.

    Счётчики предыдущего матча к новому отношения не имеют, а разница
    между ними дала бы залп выдуманных событий на первом же опросе.
    """

    def __init__(self, max_events: int = MAX_EVENTS) -> None:
        self.prev: dict | None = None
        self.prev_book: dict[str, dict] | None = None
        self.prev_counts: dict[tuple, int] | None = None
        self.prev_rosh: float | None = None
        self.events: list[dict] = []
        self.max_events = max_events
        self._seq = 0
        # Копилка изменений: ключ игрока -> накопленные дельты.
        self._pending: dict = {}
        self._pending_at: float | None = None
        self._pending_snap: dict = {}
        self.started_at: float | None = None

    # --- вход ----------------------------------------------------------

    def update(self, payload: dict, minute: float) -> list[dict]:
        """Разобрать очередной опрос. Возвращает НОВЫЕ события."""
        if minute is None or minute != minute:            # NaN
            return []
        if self.started_at is None:
            self.started_at = float(minute)
        new: list[dict] = []
        new += self._buildings(payload, minute)
        new += self._roshan(payload, minute)
        new += self._fights(payload, minute)
        for e in new:
            self._seq += 1
            e["id"] = self._seq
        self.events.extend(new)
        if len(self.events) > self.max_events:
            del self.events[: len(self.events) - self.max_events]
        return new

    # --- бои -----------------------------------------------------------

    def _fights(self, payload: dict, minute: float) -> list[dict]:
        cur = _players(payload)
        prev, self.prev = self.prev, cur
        out: list[dict] = []
        if prev is not None:
            for key, p in cur.items():
                q = prev.get(key)
                if q is None:
                    continue
                dk, dd, da = p["k"] - q["k"], p["d"] - q["d"], p["a"] - q["a"]
                # Счётчики только растут. Уменьшились — это другой матч или
                # переподключение к другому серверу: копилку надо выбросить,
                # иначе первая же разница выдаст залп выдуманных событий.
                if dk < 0 or dd < 0 or da < 0:
                    self._pending, self._pending_at = {}, None
                    return []
                if dk or dd or da:
                    acc = self._pending.setdefault(key, {"k": 0, "d": 0, "a": 0})
                    acc["k"] += dk
                    acc["d"] += dd
                    acc["a"] += da
                    self._pending_snap[key] = dict(p)
                    if self._pending_at is None:
                        self._pending_at = float(minute)
        if self._pending_at is not None and (
                (minute - self._pending_at) * 60.0 >= WINDOW):
            out += self._flush(self._pending_at)
        return out

    def _flush(self, minute: float) -> list[dict]:
        pend, snap = self._pending, self._pending_snap
        self._pending, self._pending_at, self._pending_snap = {}, None, {}
        victims = [(k, v["d"]) for k, v in pend.items() if v["d"] > 0]
        killers = [(k, v["k"]) for k, v in pend.items() if v["k"] > 0]
        assists = [k for k, v in pend.items() if v["a"] > 0]
        if not victims and not killers:
            return []
        n_deaths = sum(n for _, n in victims)
        n_kills = sum(n for _, n in killers)

        def face(key) -> dict:
            p = snap.get(key) or {}
            return {"hero": p.get("hero") or 0, "team": p.get("team"),
                    "name": p.get("name") or ""}

        # Однозначный случай: одна смерть, один фраг, и фраг у противника.
        if n_deaths == 1 and n_kills == 1:
            vk, kk = victims[0][0], killers[0][0]
            if snap.get(vk, {}).get("team") != snap.get(kk, {}).get("team"):
                return [{
                    "kind": "kill", "minute": float(minute), "certain": True,
                    "victims": [face(vk)], "killers": [face(kk)],
                    "assists": [face(a) for a in assists if a not in (vk, kk)],
                    "note": "",
                }]
        # Смерть без единого фрага у противника: вышка, крипы, Рошан, денай.
        if n_deaths and not n_kills:
            return [{
                "kind": "death", "minute": float(minute), "certain": True,
                "victims": [face(k) for k, _ in victims], "killers": [],
                "assists": [], "note": "не от героя: вышка, крипы или денай",
            }]
        # Всё остальное — замес. Пары не восстанавливаются: полей нет.
        ev = {
            "kind": "fight", "minute": float(minute), "certain": False,
            "victims": [face(k) for k, _ in victims],
            "killers": [face(k) for k, _ in killers],
            "assists": [face(a) for a in assists],
            "note": f"замес: погибло {n_deaths}, фрагов {n_kills} — "
                    f"кто кого, в источнике не сказано",
        }
        # Догадка по расстоянию: отдельным полем, в текст не идёт.
        guess = []
        for vkey, _ in victims:
            v = snap.get(vkey) or {}
            best, bd = None, None
            for kkey, _ in killers:
                kp = snap.get(kkey) or {}
                if kp.get("team") == v.get("team"):
                    continue
                d = _dist(v, kp)
                if d is not None and (bd is None or d < bd):
                    best, bd = kkey, d
            if best is not None:
                guess.append({"victim": face(vkey), "killer": face(best),
                              "dist": round(float(bd), 4)})
        if guess:
            ev["guess"] = guess
        return [ev]

    # --- здания --------------------------------------------------------

    @staticmethod
    def _counts(payload: dict) -> dict[tuple, int]:
        """Сколько стоит зданий каждого вида, БЕЗ привязки к позиции.

        Запасной путь на случай, если в ответе не окажется координат:
        тогда «какая именно вышка» не сказать, а «сторона, тип и ярус» —
        можно. Молча не показать ничего было бы хуже.
        """
        out: dict[tuple, int] = {}
        for b in (payload.get("buildings") or []):
            if not isinstance(b, dict) or b.get("destroyed"):
                continue
            key = (int(b.get("team") or 0), int(b.get("type") or 0),
                   int(b.get("tier") or 0))
            out[key] = out.get(key, 0) + 1
        return out

    def _event(self, team: int, typ: int, tier: int, lane: int,
               minute: float, exact: bool) -> dict:
        kind = {TYPE_TOWER: "tower", TYPE_RAX: "rax",
                TYPE_FORT: "fort"}.get(typ, "building")
        side = "Radiant" if team == config.TEAM_RADIANT else "Dire"
        what = TYPE_NAME.get(typ, "здание")
        lane_name = LANE_NAME.get(lane) if exact else None
        tier_s = f" t{tier}" if tier else ""
        return {
            "kind": kind, "minute": float(minute), "certain": True,
            "team": team, "tier": tier, "lane": lane if exact else 0,
            "note": f"{side} потерял {what}{tier_s}"
                    + (f" на линии {lane_name}" if lane_name else ""),
            "victims": [], "killers": [], "assists": [],
        }

    def _buildings(self, payload: dict, minute: float) -> list[dict]:
        cur = M.standing(payload)
        cur_n = self._counts(payload)
        prev, self.prev_book = self.prev_book, cur
        prev_n, self.prev_counts = self.prev_counts, cur_n
        if prev is None and prev_n is None:
            return []
        # Позиции точнее: они называют и линию тоже. Считаем по ним, когда
        # координаты есть с обеих сторон сравнения.
        if prev and cur:
            return [self._event(b["team"], b["type"], b["tier"], b["lane"],
                                minute, exact=True)
                    for k in sorted(set(prev) - set(cur))
                    for b in (prev[k],)]
        if prev_n is None:
            return []
        out = []
        for key, n in sorted(prev_n.items()):
            gone = n - cur_n.get(key, 0)
            team, typ, tier = key
            if gone <= 0 or not team:
                continue
            out += [self._event(team, typ, tier, 0, minute, exact=False)
                    for _ in range(gone)]
        return out

    # --- Рошан ---------------------------------------------------------

    def _roshan(self, payload: dict, minute: float) -> list[dict]:
        """Только если Valve добавит таймер: сейчас его в ответе нет.

        Проверено на живых слепках — слов roshan и aegis в ответе нет
        вовсе. Ветка оставлена, потому что стоит три строки, а гадать
        «наверное, Рошана взял лидер» нельзя ни при каких обстоятельствах.
        """
        rt = ((payload.get("match") or {}).get("roshan_respawn_timer"))
        if rt is None:
            return []
        rt = float(rt)
        prev, self.prev_rosh = self.prev_rosh, rt
        if prev is not None and prev <= 0 < rt:
            return [{"kind": "roshan", "minute": float(minute), "certain": True,
                     "note": "Рошан убит — кто именно, источник не говорит",
                     "victims": [], "killers": [], "assists": []}]
        return []

    # --- наружу --------------------------------------------------------

    def recent(self, limit: int = 24) -> list[dict]:
        return self.events[-limit:][::-1]
