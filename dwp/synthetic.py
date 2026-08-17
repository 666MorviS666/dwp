"""Генератор синтетических матчей в схеме OpenDota.

Зачем: до того как трогать сеть, нужен датасет с ИЗВЕСТНОЙ истинной силой
героев и команд. Тогда по метрикам видно, восстанавливает пайплайн сигнал
или в нём баг. На реальных данных отличить «модель плохая» от «код
сломан» невозможно.

Что специально воспроизводится из реальных данных:
  * оба формата `objectives` (новый building_kill и старый CHAT_MESSAGE_*);
  * матчи с неизвестным третьим типом события — парсер не должен падать;
  * поминутные массивы radiant_gold_adv / radiant_xp_adv;
  * битовые маски tower_status_* / barracks_status_*.

Чего НЕ воспроизводится (и на что опираться нельзя): реальные id героев,
реальные имена зданий во всех вариантах, реальная семантика поля `team`
в старых событиях. Последнее — допущение, см. config.OLD_TOWER_KILL_*.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys

import numpy as np

from . import config

LANES = ("top", "mid", "bot")


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def make_heroes(n_heroes: int) -> list[dict]:
    """Список героев в схеме /heroes. id намеренно НЕ подряд:

    в реальном OpenDota id героев разрежены (есть дыры), и код, который
    молча считает hero_id индексом массива, ломается именно на этом.
    """
    heroes = []
    hid = 1
    for i in range(n_heroes):
        heroes.append(
            {
                "id": hid,
                "name": f"npc_dota_hero_synth_{i:03d}",
                "localized_name": f"Synth {i:03d}",
                "primary_attr": ("str", "agi", "int")[i % 3],
                "attack_type": "Melee" if i % 2 else "Ranged",
                "roles": ["Carry"],
            }
        )
        hid += 1
        if i % 7 == 3:          # дыра в нумерации
            hid += 1
    return heroes


def _tower_event(new_format: bool, time_s: int, victim_is_radiant: bool,
                 lane: str, tier: int, player_slot: int) -> dict:
    """Одно событие сноса вышки в одном из двух форматов."""
    if new_format:
        side = "goodguys" if victim_is_radiant else "badguys"
        return {
            "time": time_s,
            "type": "building_kill",
            "key": f"npc_dota_{side}_tower{tier}_{lane}",
            "slot": player_slot % 5,
            "player_slot": player_slot,
        }
    # Старый формат. Поле team трактуется согласно
    # config.OLD_TOWER_KILL_TEAM_MEANS; здесь генерим под "victim".
    victim_team = config.TEAM_RADIANT if victim_is_radiant else config.TEAM_DIRE
    killer_team = config.TEAM_DIRE if victim_is_radiant else config.TEAM_RADIANT
    team = victim_team if config.OLD_TOWER_KILL_TEAM_MEANS == "victim" else killer_team
    return {
        "time": time_s,
        "type": "CHAT_MESSAGE_TOWER_KILL",
        "team": team,
        "slot": player_slot % 5,
        "player_slot": player_slot,
    }


def _rax_event(new_format: bool, time_s: int, victim_is_radiant: bool,
               lane: str, melee: bool, player_slot: int) -> dict:
    if new_format:
        side = "goodguys" if victim_is_radiant else "badguys"
        kind = "melee" if melee else "range"
        return {
            "time": time_s,
            "type": "building_kill",
            "key": f"npc_dota_{side}_{kind}_rax_{lane}",
            "slot": player_slot % 5,
            "player_slot": player_slot,
        }
    victim_team = config.TEAM_RADIANT if victim_is_radiant else config.TEAM_DIRE
    killer_team = config.TEAM_DIRE if victim_is_radiant else config.TEAM_RADIANT
    team = victim_team if config.OLD_TOWER_KILL_TEAM_MEANS == "victim" else killer_team
    return {
        "time": time_s,
        "type": "CHAT_MESSAGE_BARRACKS_KILL",
        "team": team,
        "key": str(1 << (LANES.index(lane) * 2 + (0 if melee else 1))),
        "player_slot": player_slot,
    }


def generate_match(rng: np.random.Generator, match_id: int, hero_ids: np.ndarray,
                   hero_strength: np.ndarray, team_strength: np.ndarray,
                   start_time: int) -> dict:
    n_heroes = len(hero_ids)
    n_teams = len(team_strength)

    r_team, d_team = rng.choice(n_teams, size=2, replace=False)
    picks = rng.choice(n_heroes, size=10, replace=False)
    r_idx, d_idx = picks[:5], picks[5:]

    s = float(hero_strength[r_idx].sum() - hero_strength[d_idx].sum())
    s += float(team_strength[r_team] - team_strength[d_team])

    p_radiant = _sigmoid(s)
    radiant_win = bool(rng.random() < p_radiant)
    sign = 1.0 if radiant_win else -1.0

    # Разгромы короче: длительность зависит от того, насколько исход
    # предопределён. Это же создаёт корреляцию duration с исходом —
    # ровно та ловушка, из-за которой duration запрещено брать в признаки.
    base_len = 44.0 - 9.0 * abs(math.tanh(s))
    dur_min = int(np.clip(rng.normal(base_len, 8.0), 22, 72))
    duration = dur_min * 60 + int(rng.integers(0, 60))

    # --- траектория золота: случайное блуждание со сносом к победителю ---
    # Камбэки обязательны: если знак финального перевеса всегда совпадает с
    # победителем, стейт-модель получает вырожденно лёгкую задачу и по
    # метрикам нельзя отличить рабочий пайплайн от протечки.
    comeback = rng.random() < 0.18
    cb_until = rng.uniform(0.35, 0.6) if comeback else 0.0
    gold = np.zeros(dur_min + 1)
    xp = np.zeros(dur_min + 1)
    k_ramp = 21000.0 / dur_min          # интеграл сноса ≈ 10.5k к концу
    for t in range(1, dur_min + 1):
        frac = t / dur_min
        local_sign = -sign if frac < cb_until else sign
        drift = 200.0 * math.tanh(s) + k_ramp * local_sign * frac ** 0.8
        gold[t] = gold[t - 1] + drift + rng.normal(0, 950)
        xp[t] = 1.15 * gold[t] + rng.normal(0, 1400)
    gold_adv = [int(v) for v in np.round(gold)]
    xp_adv = [int(v) for v in np.round(xp)]

    # --- здания: вероятность растёт от перевеса по золоту ---
    new_format = bool(rng.random() < 0.6)
    objectives: list[dict] = []
    towers_lost = {True: 0, False: 0}     # ключ: victim_is_radiant
    rax_lost = {True: 0, False: 0}

    # Победитель не может потерять всё: падение обеих башен трона и всех
    # бараков почти всегда заканчивается поражением. Без этого потолка
    # генератор выдаёт 2% матчей «снесли весь хайграунд и выиграли», и
    # признаки по зданиям выглядят слабее, чем они есть в реальности.
    cap_tow = {True: 10 if radiant_win else config.N_TOWERS_PER_SIDE,
               False: config.N_TOWERS_PER_SIDE if radiant_win else 10}
    cap_rax = {True: 5 if radiant_win else config.N_RAX_PER_SIDE,
               False: config.N_RAX_PER_SIDE if radiant_win else 5}

    for t in range(5, dur_min + 1):
        g = gold[t]
        for victim_is_radiant in (True, False):
            # Radiant сносит вышки Dire (victim_is_radiant=False) тем чаще,
            # чем больше его перевес по золоту.
            adv = -g if victim_is_radiant else g
            p = 0.42 * _sigmoid(adv / 4000.0) * (0.4 + 0.6 * t / dur_min)
            if towers_lost[victim_is_radiant] < cap_tow[victim_is_radiant] and rng.random() < p:
                towers_lost[victim_is_radiant] += 1
                tier = min(4, 1 + towers_lost[victim_is_radiant] // 3)
                slot = int(rng.integers(0, 5)) + (128 if victim_is_radiant else 0)
                objectives.append(
                    _tower_event(new_format, t * 60 + int(rng.integers(0, 60)),
                                 victim_is_radiant, LANES[int(rng.integers(0, 3))],
                                 tier, slot)
                )
            if (towers_lost[victim_is_radiant] >= 6
                    and rax_lost[victim_is_radiant] < cap_rax[victim_is_radiant]
                    and rng.random() < 0.18 * _sigmoid(adv / 4000.0)):
                rax_lost[victim_is_radiant] += 1
                slot = int(rng.integers(0, 5)) + (128 if victim_is_radiant else 0)
                objectives.append(
                    _rax_event(new_format, t * 60 + int(rng.integers(0, 60)),
                               victim_is_radiant, LANES[int(rng.integers(0, 3))],
                               bool(rng.integers(0, 2)), slot)
                )

    # Победитель обязан снести минимум 5 вышек и 2 барака — иначе трон
    # не падает. Досыпаем на хайграунде в последние минуты.
    loser_is_radiant = not radiant_win
    while towers_lost[loser_is_radiant] < 5:
        towers_lost[loser_is_radiant] += 1
        t = dur_min - int(rng.integers(0, 4))
        objectives.append(
            _tower_event(new_format, max(60, t * 60), loser_is_radiant,
                         LANES[int(rng.integers(0, 3))], 3,
                         int(rng.integers(0, 5)) + (128 if loser_is_radiant else 0))
        )
    while rax_lost[loser_is_radiant] < 2:
        rax_lost[loser_is_radiant] += 1
        t = dur_min - int(rng.integers(0, 3))
        objectives.append(
            _rax_event(new_format, max(60, t * 60), loser_is_radiant,
                       LANES[int(rng.integers(0, 3))], bool(rng.integers(0, 2)),
                       int(rng.integers(0, 5)) + (128 if loser_is_radiant else 0))
        )

    # --- Рошан ---
    next_rosh_ok = 12
    while next_rosh_ok <= dur_min:
        if rng.random() < 0.35:
            t = next_rosh_ok + int(rng.integers(0, 5))
            if t > dur_min:
                break
            g = gold[min(t, dur_min)]
            killer_radiant = rng.random() < _sigmoid(g / 3000.0)
            team = config.TEAM_RADIANT if killer_radiant else config.TEAM_DIRE
            if config.OLD_ROSHAN_KILL_TEAM_MEANS != "killer":
                team = config.TEAM_DIRE if killer_radiant else config.TEAM_RADIANT
            objectives.append(
                {
                    "time": t * 60 + int(rng.integers(0, 60)),
                    "type": "CHAT_MESSAGE_ROSHAN_KILL",
                    "team": team,
                    "player_slot": int(rng.integers(0, 5)) + (0 if killer_radiant else 128),
                }
            )
            next_rosh_ok = t + 9
        else:
            next_rosh_ok += 3

    # Терзатель (Торментор). ПРАВИЛА ПАТЧА 7.38: на карте ОДИН
    # Терзатель, первый спавн на 15-й минуте, дальше респавн раз в 10
    # минут; он перемещается между углами карты, а не существует в двух
    # экземплярах. До 7.38 их было двое — отсюда путаница.
    # По числу убийств за матч 1 и 2 не различаются: при спавне на 15-й
    # минуте четыре убийства укладываются в матч 45+ минут.
    t_next = 15
    while t_next <= dur_min:
        if rng.random() < 0.72:
            t = t_next + int(rng.integers(0, 4))
            if t > dur_min:
                break
            killer_radiant = rng.random() < _sigmoid(gold[min(t, dur_min)] / 3500.0)
            objectives.append({
                "time": t * 60 + int(rng.integers(0, 60)),
                "type": "CHAT_MESSAGE_MINIBOSS_KILL",
                "team": config.TEAM_RADIANT if killer_radiant else config.TEAM_DIRE,
                "slot": int(rng.integers(0, 5)),
                "player_slot": int(rng.integers(0, 5)) + (0 if killer_radiant else 128),
            })
            t_next = t + 10
        else:
            t_next += 3

    # Шум: события, которых парсер не знает. Не должны ронять его и не
    # должны молча учитываться как ноль.
    if rng.random() < 0.10:
        objectives.append(
            {"time": int(rng.integers(300, duration)),
             "type": "CHAT_MESSAGE_FIRSTBLOOD", "player_slot": 2}
        )
    if rng.random() < 0.06:
        objectives.append(
            {"time": int(rng.integers(300, duration)),
             "type": "building_kill", "key": "npc_dota_neutral_hideout_7"}
        )

    objectives.sort(key=lambda o: o["time"])

    # Битовые маски. ВНИМАНИЕ: здесь установлены просто младшие биты по
    # числу выживших зданий. Реальная семантика — конкретное здание на
    # конкретный бит; совпадает только popcount, и только на него
    # опирается check_schema.
    def _mask(alive: int, width: int) -> int:
        return (1 << alive) - 1

    # Поигроковые ряды: gold_t, kills_log, buyback_log. Нужны, чтобы
    # признаки из --extra-features проверялись на синтетике, а не только
    # на реальных данных, где отличить баг от слабого сигнала нельзя.
    # Доли нетворса внутри команды: керри забирает больше саппортов.
    # Доли рисуются на каждую команду отдельно: если задать один и тот же
    # набор обеим, признак концентрации нетворса окажется константой и
    # проверить его на синтетике будет нельзя.
    def _shares() -> np.ndarray:
        w = rng.dirichlet(np.array([2.6, 2.0, 1.6, 1.3, 1.0]) * rng.uniform(1.0, 6.0))
        return np.sort(w)[::-1]
    total_r = 300.0 + np.arange(dur_min + 1) * 620.0
    players = []
    for side, idxs, base_slot in ((True, r_idx, 0), (False, d_idx, 128)):
        # Половина перевеса по золоту приходится на сторону, вторая
        # половина вычитается у соперника — так сумма сходится с gold.
        team_total = total_r + (gold / 2.0 if side else -gold / 2.0)
        team_total = np.maximum(team_total, 100.0)
        shares = _shares()
        perm = rng.permutation(5)
        for i, h in enumerate(idxs):
            g = team_total * shares[perm[i]]
            kills_log = []
            # Фраги коррелируют с перевесом, но не повторяют его: команда
            # может выигрывать замесы и не конвертировать это в золото.
            for t in range(3, dur_min + 1):
                lam = 0.055 * _sigmoid((gold[t] if side else -gold[t]) / 5000.0) * 2
                if rng.random() < lam:
                    kills_log.append({"time": t * 60 + int(rng.integers(0, 60)),
                                      "key": "npc_dota_hero_x"})
            buyback_log = []
            for t in range(20, dur_min + 1):
                if rng.random() < 0.012 * _sigmoid((-gold[t] if side else gold[t]) / 5000.0) * 2:
                    buyback_log.append({"time": t * 60, "slot": i,
                                        "player_slot": base_slot + i})
            players.append({
                "hero_id": int(hero_ids[h]),
                "player_slot": base_slot + i,
                "isRadiant": side,
                "account_id": (1000 if side else 2000) + i,
                "times": [t * 60 for t in range(dur_min + 1)],
                "gold_t": [int(v) for v in np.round(g)],
                "xp_t": [int(v) for v in np.round(g * 1.15)],
                "kills_log": kills_log,
                "buyback_log": buyback_log,
            })

    return {
        "match_id": match_id,
        "radiant_win": radiant_win,
        "duration": duration,
        "start_time": start_time,
        "leagueid": 9999,
        "patch": 55,
        "version": 21,
        "radiant_team_id": int(r_team) + 10000,
        "dire_team_id": int(d_team) + 10000,
        "radiant_gold_adv": gold_adv,
        "radiant_xp_adv": xp_adv,
        "tower_status_radiant": _mask(config.N_TOWERS_PER_SIDE - towers_lost[True], 11),
        "tower_status_dire": _mask(config.N_TOWERS_PER_SIDE - towers_lost[False], 11),
        "barracks_status_radiant": _mask(config.N_RAX_PER_SIDE - rax_lost[True], 6),
        "barracks_status_dire": _mask(config.N_RAX_PER_SIDE - rax_lost[False], 6),
        "objectives": objectives,
        "players": players,
        "_synthetic": True,
        "_true_logit": s,
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Генератор синтетических матчей.")
    # 4000 ≈ порядок реального объёма про-матчей за пару лет. На 1200
    # изотоническая калибровка драфт-модели измеримо ухудшает log loss:
    # на ~190 точках фолда она даёт грубые ступени. Это не баг генератора,
    # а свойство метода при малом n, см. README.
    ap.add_argument("--n", type=int, default=4000, help="сколько матчей")
    ap.add_argument("--seed", type=int, default=config.SYNTH_SEED)
    ap.add_argument("--clean", action="store_true", help="очистить каталог перед генерацией")
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)
    heroes = make_heroes(config.SYNTH_N_HEROES)
    hero_ids = np.array([h["id"] for h in heroes], dtype=np.int64)

    # Истинная сила. Часть героев сильно перекошена — иначе на паре тысяч
    # матчей вообще нечего восстанавливать.
    hero_strength = rng.normal(0.0, 0.16, size=len(heroes))
    hero_strength[:8] += 0.55
    hero_strength[8:16] -= 0.55
    team_strength = rng.normal(0.0, 0.45, size=config.SYNTH_N_TEAMS)

    out = config.SYNTH_MATCHES_DIR
    if args.clean and out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    config.HEROES_PATH.write_text(json.dumps(heroes, ensure_ascii=False), encoding="utf-8")

    start = 1_700_000_000
    for i in range(args.n):
        mid = 8_000_000_000 + i * 137
        m = generate_match(rng, mid, hero_ids, hero_strength, team_strength,
                           start + i * 3600)
        (out / f"{mid}.json").write_text(json.dumps(m), encoding="utf-8")

    truth = {
        "hero_strength": {int(hid): float(v) for hid, v in zip(hero_ids, hero_strength)},
        "team_strength": {int(i) + 10000: float(v) for i, v in enumerate(team_strength)},
        "seed": args.seed,
        "n_matches": args.n,
    }
    (config.DATA_DIR / "synthetic_truth.json").write_text(
        json.dumps(truth, ensure_ascii=False), encoding="utf-8")

    print(f"Сгенерировано матчей: {args.n} -> {out}")
    print(f"Героев: {len(heroes)} -> {config.HEROES_PATH}")
    print(f"Истинные силы -> {config.DATA_DIR / 'synthetic_truth.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
