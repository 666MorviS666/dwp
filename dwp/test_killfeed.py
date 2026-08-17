"""Проверка ленты событий на подставных опросах.

Лента ВЫВОДИТСЯ из разницы счётчиков, а не читается из журнала (журнала в
`GetRealtimeStats` нет). Значит проверять надо не то, что она что-то
показывает, а то, что она НЕ показывает: не называет убийцу там, где его
из данных не следует, и не выдумывает событий на смене матча.

    python -m dwp.test_killfeed
"""

from __future__ import annotations

import sys

from . import config
from .killfeed import Feed, WINDOW

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'ПРОВАЛ'} {name}" + (f": {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def payload(counts: dict[tuple[int, int], tuple[int, int, int]],
            buildings: list[dict] | None = None,
            game_time: float = 600.0, pos: dict | None = None) -> dict:
    """counts: (команда, слот) -> (фраги, смерти, ассисты)."""
    teams = []
    for tn in (config.TEAM_RADIANT, config.TEAM_DIRE):
        ps = []
        for slot in range(5):
            k, d, a = counts.get((tn, slot), (0, 0, 0))
            x, y = (pos or {}).get((tn, slot), (0.1 * slot, 0.1 * tn))
            ps.append({"team_slot": slot, "team": tn, "heroid": tn * 10 + slot,
                       "name": f"p{tn}{slot}", "kill_count": k, "death_count": d,
                       "assists_count": a, "x": x, "y": y, "net_worth": 5000,
                       "level": 10, "gold": 300, "lh_count": 50,
                       "denies_count": 5})
        teams.append({"team_number": tn, "players": ps, "net_worth": 25000})
    return {"match": {"game_time": game_time, "game_state": 5,
                      "match_id": "1", "server_steam_id": "1"},
            "teams": teams,
            "buildings": buildings if buildings is not None else []}


def bld(team: int, typ: int, tier: int, lane: int, x: float, y: float,
        destroyed: bool = False) -> dict:
    return {"team": team, "type": typ, "tier": tier, "lane": lane,
            "x": x, "y": y, "destroyed": destroyed}


def feed_at(f: Feed, counts, minute: float, buildings=None, pos=None):
    return f.update(payload(counts, buildings, game_time=minute * 60.0, pos=pos),
                    minute)


def main() -> int:
    R, D = config.TEAM_RADIANT, config.TEAM_DIRE
    gap = WINDOW / 60.0 + 0.01                 # чуть больше окна склейки

    print("\n1. Один фраг: жертва и убийца названы точно")
    f = Feed()
    feed_at(f, {}, 10.0)
    feed_at(f, {(R, 0): (1, 0, 0), (D, 2): (0, 1, 0)}, 10.02)
    ev = feed_at(f, {(R, 0): (1, 0, 0), (D, 2): (0, 1, 0)}, 10.02 + gap)
    check("одно событие", len(ev) == 1, f"получено {len(ev)}")
    if ev:
        e = ev[0]
        check("это фраг", e["kind"] == "kill", e["kind"])
        check("атрибуция однозначна", e["certain"] is True)
        check("убийца — герой Radiant",
              e["killers"] and e["killers"][0]["hero"] == R * 10 + 0)
        check("жертва — герой Dire",
              e["victims"] and e["victims"][0]["hero"] == D * 10 + 2)

    print("\n2. Смерть без фрага у противника: убийцу НЕ подставляем")
    f = Feed()
    feed_at(f, {}, 12.0)
    feed_at(f, {(R, 1): (0, 1, 0)}, 12.02)
    ev = feed_at(f, {(R, 1): (0, 1, 0)}, 12.02 + gap)
    check("одно событие", len(ev) == 1, f"получено {len(ev)}")
    if ev:
        check("вид — смерть, а не фраг", ev[0]["kind"] == "death", ev[0]["kind"])
        check("убийц нет", ev[0]["killers"] == [])
        check("сказано, что не от героя", "не от героя" in ev[0]["note"],
              ev[0]["note"])

    print("\n3. Замес 2 на 2: пары НЕ выдумываются")
    f = Feed()
    feed_at(f, {}, 20.0)
    feed_at(f, {(R, 0): (1, 0, 0), (R, 3): (1, 0, 0),
                (D, 1): (0, 1, 0), (D, 4): (0, 1, 0)}, 20.02)
    ev = feed_at(f, {(R, 0): (1, 0, 0), (R, 3): (1, 0, 0),
                     (D, 1): (0, 1, 0), (D, 4): (0, 1, 0)}, 20.02 + gap)
    check("одно событие на весь замес", len(ev) == 1, f"получено {len(ev)}")
    if ev:
        e = ev[0]
        check("помечено как неоднозначное", e["certain"] is False)
        check("вид — замес", e["kind"] == "fight", e["kind"])
        check("жертв двое", len(e["victims"]) == 2, str(len(e["victims"])))
        check("убийц двое", len(e["killers"]) == 2, str(len(e["killers"])))
        check("в тексте сказано, что пар нет", "кто кого" in e["note"], e["note"])
        check("догадка по расстоянию есть, но отдельным полем",
              "guess" in e and e["guess"] and "dist" in e["guess"][0])

    print("\n4. События одного боя склеиваются, а разных — нет")
    f = Feed()
    feed_at(f, {}, 30.0)
    feed_at(f, {(R, 0): (1, 0, 0), (D, 0): (0, 1, 0)}, 30.01)
    mid = feed_at(f, {(R, 0): (1, 0, 0), (R, 1): (1, 0, 0),
                      (D, 0): (0, 1, 0), (D, 1): (0, 1, 0)}, 30.02)
    check("внутри окна ничего не выдано", mid == [], f"выдано {len(mid)}")
    ev = feed_at(f, {(R, 0): (1, 0, 0), (R, 1): (1, 0, 0),
                     (D, 0): (0, 1, 0), (D, 1): (0, 1, 0)}, 30.02 + gap)
    check("после окна выдан один замес", len(ev) == 1 and ev[0]["kind"] == "fight",
          f"{[e['kind'] for e in ev]}")

    print("\n5. Здание: сторона, ярус и линия — из справочника позиций")
    f = Feed()
    alive = [bld(R, 0, 1, 2, -0.20, -0.20), bld(D, 0, 1, 2, 0.20, 0.20)]
    feed_at(f, {}, 15.0, buildings=alive)
    gone = [bld(0, 0, 0, 0, 0.0, 0.0, destroyed=True), alive[1]]
    ev = feed_at(f, {}, 15.1, buildings=gone)
    check("одно событие по зданию", len(ev) == 1, f"получено {len(ev)}")
    if ev:
        e = ev[0]
        check("это вышка", e["kind"] == "tower", e["kind"])
        check("сторона Radiant", e["team"] == R, str(e["team"]))
        check("ярус 1 и линия мид", e["tier"] == 1 and e["lane"] == 2,
              f"t{e['tier']} lane {e['lane']}")

    print("\n6. Смена матча: счётчики упали — событий НЕ выдумываем")
    f = Feed()
    feed_at(f, {(R, 0): (8, 2, 5), (D, 0): (3, 7, 4)}, 40.0)
    ev = feed_at(f, {(R, 0): (0, 0, 0), (D, 0): (0, 0, 0)}, 0.5)
    check("ни одного события", ev == [], f"выдано {len(ev)}")
    ev = feed_at(f, {(R, 0): (0, 0, 0), (D, 0): (0, 0, 0)}, 0.5 + gap)
    check("и на следующем опросе тоже", ev == [], f"выдано {len(ev)}")

    print("\n7. Первый опрос матча: разницы ещё не с чем брать")
    f = Feed()
    ev = feed_at(f, {(R, 0): (5, 1, 2)}, 25.0,
                 buildings=[bld(R, 0, 1, 2, -0.2, -0.2)])
    check("событий нет", ev == [], f"выдано {len(ev)}")

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
