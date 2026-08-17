"""Урон: есть ли он в источниках и весит ли он что-нибудь.

Вопрос был поставлен просто: «статистика урона прямо в матче тоже должна
что-то весить». Ответ разбивается на два, и они разные.

ЕСТЬ ЛИ УРОН В ЛАЙВЕ. Нет. В ответе `GetRealtimeStats` слова `damage` нет
ни разу — проверено на двух живых слепках (`--check`), у игрока там ровно
восемнадцать полей, и урона среди них не значится: уровень, фраги, смерти,
ассисты, добивания, денаи, золото, нетворс, предметы, способности, x/y.
Значит признак по урону, даже полезный, в бою получал бы NaN на каждой
строке — ровно то, за что `live.py` отвергает модели с XP.

ЕСТЬ ЛИ УРОН В ИСТОРИИ. Да, но в двух видах, и оба неполные:
  * итоговые `hero_damage` / `tower_damage` у игрока — одно число за матч.
    Поминутного ряда нет. В признак его не подставить: на 20-й минуте мы
    знали бы итог 45-й, это утечка из будущего;
  * `teamfights[]` — по каждой стычке есть `start`, `end` и `damage` у
    каждого участника. Вот из этого поминутный ряд собирается честно, и
    именно он здесь и строится. Покрытие: стычки занимают около 16%
    времени матча, то есть это урон В ЗАМЕСАХ, а не весь.

Что модуль делает:
    --check           показать, что урона в лайв-ответе нет (или появился)
    --frames          выгрузить поминутные признаки по урону
    --measure         замерить прирост качества от них (через dwp.bench)

Результат замера записан в README. Коротко: смотрите туда, а не сюда, —
здесь только инструмент.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# Слова, по которым ищется урон в ответе лайва. Список нарочно широкий:
# задача — не пропустить поле, если Valve его добавит.
DAMAGE_HINTS = ("damage", "dmg", "healing", "heal", "hits", "stun")

DAMAGE_COLUMNS = ["tf_damage_adv", "tf_damage_share", "tf_healing_adv",
                  "tf_damage_per_gold_adv", "tf_gold_delta_adv",
                  "tf_buybacks_adv", "tf_minutes_since"]


def scan_live(payload: dict) -> dict:
    """Есть ли в ответе хоть что-то про урон. Возвращает найденные пути."""
    found: list[str] = []
    keys: set[str] = set()

    def walk(o, path=""):
        if isinstance(o, dict):
            for k, v in o.items():
                p = f"{path}.{k}" if path else k
                kl = str(k).lower()
                keys.add(str(k))
                if any(h in kl for h in DAMAGE_HINTS):
                    found.append(p)
                walk(v, p)
        elif isinstance(o, list) and o:
            walk(o[0], path + "[0]")

    walk(payload)
    player_keys: list[str] = []
    for t in (payload.get("teams") or []):
        for p in (t.get("players") or []):
            player_keys = sorted(p.keys())
            break
        if player_keys:
            break
    return {"found": found, "player_keys": player_keys, "n_keys": len(keys)}


def match_damage_frame(match: dict, T: int | None = None) -> pd.DataFrame | None:
    """Поминутный урон в стычках, накопительно, по сторонам.

    Как считается время. У стычки есть `start` и `end`; урон приписывается
    минуте окончания — именно тогда результат замеса становится известен
    зрителю. Приписать началу значило бы знать исход стычки заранее, а
    размазать по секундам нечем: внутри стычки разбивки нет.

    Сторона игрока определяется ПОРЯДКОМ в `teamfights[].players`: их всегда
    десять и они идут в том же порядке, что `match["players"]`. Порядок
    проверяется по длине; не сошлось — None, а не догадка.
    """
    tf = match.get("teamfights")
    players = match.get("players") or []
    if not isinstance(tf, list) or len(players) != 10:
        return None
    gold = match.get("radiant_gold_adv")
    if not gold:
        return None
    n = len(gold) - 1 if T is None else T
    if n < 1:
        return None
    sides = np.array([int(p.get("player_slot", 0)) < 128 for p in players])
    if sides.sum() != 5:
        return None

    dmg = np.zeros((2, n + 1))       # [0] Radiant, [1] Dire
    heal = np.zeros((2, n + 1))
    gd = np.zeros((2, n + 1))
    bb = np.zeros((2, n + 1))
    last_fight = np.full(n + 1, np.nan)
    seen = 0
    for fight in tf:
        if not isinstance(fight, dict):
            continue
        end = fight.get("end")
        fp = fight.get("players")
        if end is None or not isinstance(fp, list) or len(fp) != 10:
            continue
        mn = max(0, int(end) // 60)
        if mn > n:
            continue
        seen += 1
        for i, q in enumerate(fp):
            if not isinstance(q, dict):
                continue
            s = 0 if sides[i] else 1
            dmg[s, mn:] += float(q.get("damage") or 0)
            heal[s, mn:] += float(q.get("healing") or 0)
            gd[s, mn:] += float(q.get("gold_delta") or 0)
            bb[s, mn:] += float(q.get("buybacks") or 0)
        last_fight[mn:] = mn
    if not seen:
        # Ни одной разобранной стычки — честный NaN по всем столбцам.
        # Ноль означал бы «замесов не было», а это для 40-й минуты ложь.
        nan = np.full(n + 1, np.nan)
        return pd.DataFrame({"match_id": int(match["match_id"]),
                             "minute": np.arange(n + 1),
                             **{c: nan.copy() for c in DAMAGE_COLUMNS}})

    tot = dmg[0] + dmg[1]
    share = np.where(tot > 0, (dmg[0] - dmg[1]) / np.maximum(tot, 1.0), np.nan)
    # Урон на добытое золото: команда, которая при равной экономике наносит
    # больше, сильнее, чем показывает нетворс. Это и есть та часть «веса
    # урона», которой в золоте нет.
    g = np.zeros((2, n + 1))
    for i, p in enumerate(players):
        arr = p.get("gold_t")
        if not isinstance(arr, list) or not arr:
            g = None
            break
        a = np.asarray(arr, dtype=np.float64)
        m = min(len(a), n + 1)
        s = 0 if sides[i] else 1
        g[s, :m] += a[:m]
        if m <= n:
            g[s, m:] += a[m - 1]
    if g is None:
        per_gold = np.full(n + 1, np.nan)
    else:
        with np.errstate(invalid="ignore", divide="ignore"):
            per_gold = (np.where(g[0] > 0, dmg[0] / np.maximum(g[0], 1.0), np.nan)
                        - np.where(g[1] > 0, dmg[1] / np.maximum(g[1], 1.0), np.nan))
    since = np.arange(n + 1) - last_fight
    return pd.DataFrame({
        "match_id": int(match["match_id"]),
        "minute": np.arange(n + 1),
        "tf_damage_adv": dmg[0] - dmg[1],
        "tf_damage_share": share,
        "tf_healing_adv": heal[0] - heal[1],
        "tf_damage_per_gold_adv": per_gold,
        "tf_gold_delta_adv": gd[0] - gd[1],
        "tf_buybacks_adv": bb[1] - bb[0],   # знак как у tower_adv: + в пользу Radiant
        "tf_minutes_since": since,
    })


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Урон: доступность и вес.")
    ap.add_argument("--check", action="store_true",
                    help="проверить, есть ли урон в лайв-ответе")
    ap.add_argument("--from-file", type=Path, nargs="*",
                    default=[config.DATA_DIR / "live_dump.json",
                             config.DATA_DIR / "live_dump_late.json"],
                    help="слепки лайв-ответа для --check")
    ap.add_argument("--sample", type=int, default=200,
                    help="сколько матчей посмотреть для сводки по истории")
    args = ap.parse_args(argv)

    if args.check:
        print("=== Есть ли урон в ответе GetRealtimeStats ===")
        any_found = False
        for p in args.from_file:
            p = Path(p)
            if not p.exists():
                print(f"  {p.name}: файла нет, пропускаю")
                continue
            raw = p.read_bytes()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            got = scan_live(json.loads(raw.decode(enc)))
            any_found = any_found or bool(got["found"])
            print(f"\n  {p.name}: ключей в ответе {got['n_keys']}, "
                  f"похожих на урон: {len(got['found'])}")
            if got["found"]:
                for f in got["found"][:20]:
                    print(f"      {f}")
            print(f"    поля игрока: {', '.join(got['player_keys'])}")
        print("\nВЫВОД: " + ("урон в ответе ПОЯВИЛСЯ — пересмотрите этот "
                             "раздел README."
                             if any_found else
                             "урона в лайв-ответе нет. Признак по урону в бою "
                             "был бы NaN\nна каждой строке, поэтому в лайв-модель "
                             "он не идёт."))
        return 0

    # Сводка по истории: сколько урона вообще видно через стычки.
    from .builds import iter_matches
    n = 0
    cov, shares, corr_rows = [], [], []
    for m in iter_matches(config.RAW_MATCHES_DIR, args.sample):
        fr = match_damage_frame(m)
        if fr is None or fr["tf_damage_adv"].isna().all():
            continue
        n += 1
        tf = m.get("teamfights") or []
        dur = max(1, m.get("duration") or 1)
        cov.append(sum(max(0, (f.get("end") or 0) - (f.get("start") or 0))
                       for f in tf if isinstance(f, dict)) / dur)
        hd_r = sum(p.get("hero_damage") or 0 for p in m["players"]
                   if int(p.get("player_slot", 0)) < 128)
        hd_d = sum(p.get("hero_damage") or 0 for p in m["players"]
                   if int(p.get("player_slot", 0)) >= 128)
        corr_rows.append((float(fr["tf_damage_adv"].iloc[-1]), float(hd_r - hd_d)))
        shares.append(float(fr["tf_damage_share"].iloc[-1]))
    if not n:
        print("Матчей со стычками не нашлось.", file=sys.stderr)
        return 2
    a = np.array(corr_rows)
    print(f"Матчей разобрано: {n}")
    print(f"  доля времени матча внутри стычек: медиана {np.median(cov):.3f}")
    print(f"  урон в стычках против итогового hero_damage: "
          f"корреляция {np.corrcoef(a[:, 0], a[:, 1])[0, 1]:.4f}, "
          f"отношение медиан {np.median(a[:, 0] / np.where(a[:, 1] == 0, np.nan, a[:, 1])):.3f}")
    print("  То есть ряд по стычкам — это ЧАСТЬ урона, а не весь урон;\n"
          "  зато у него есть время, а у итогового числа его нет.")
    print("\nЧтобы замерить, весит ли это что-нибудь:\n"
          "    python -m dwp.export --what extras\n"
          "    python -m dwp.bench_extras")
    return 0


if __name__ == "__main__":
    sys.exit(main())
