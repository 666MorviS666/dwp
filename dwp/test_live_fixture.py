"""Проверка live.py на фикстурах вместо сети.

Фикстура построена по СЛЕПКУ ЖИВОГО ОТВЕТА GetRealtimeStats
(про-матч, game_state=5, 36 зданий). Проверенные на нём факты, каждый
из которых раньше был догадкой и оказался частично неверным:

  * верхние ключи: match / teams / buildings / graph_data / delta_frame;
  * match содержит game_time, game_state, picks, bans — но НЕ содержит
    roshan_respawn_timer; слов roshan и aegis в ответе нет вовсе;
  * teams[].net_worth и teams[].players[].heroid есть; team_id в паблике
    приходит нулём;
  * buildings: 22 вышки + 12 бараков + 2 трона = 36 записей,
    type 0/1/2 соответственно;
  * У СНЕСЁННОГО здания team ОБНУЛЯЕТСЯ. Поэтому считать потери по
    флагу destroyed нельзя — только по недостаче стоящих;
  * в стадии драфта ответ структурно полный, но все значения нулевые.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from dwp import config, live  # noqa: E402


def pick_model(prefer: tuple[str, ...] = (), **want) -> Path | None:
    """Модель по СВОЙСТВАМ, а не по имени файла.

    Раньше имена были зашиты (`model_synthetic_noxp.pkl`), и тест не запускался
    нигде, кроме машины, где перед этим прогнали синтетику: падал на загрузке
    ещё до первой проверки. Свойства лежат в самом артефакте, поэтому подойдёт
    любая подходящая модель, а привычные имена просто идут первыми.
    """
    names = list(config.MODELS_DIR.glob("*.pkl"))
    names.sort(key=lambda p: (0, prefer.index(p.name)) if p.name in prefer
               else (1, 0))
    for p in names:
        try:
            with p.open("rb") as fh:
                art = pickle.load(fh)
        except Exception:
            continue
        if all(bool(art.get(k)) == v for k, v in want.items()):
            return p
    return None


def make_payload(game_time: int, nw_r: int, nw_d: int, roshan_timer=None,
                 tow_r: int = 0, tow_d: int = 0, hero_ids=None,
                 team_ids=(0, 0), game_state: int = 5) -> dict:
    """Слепок живого ответа. roshan_timer оставлен параметром, но по
    умолчанию не проставляется: в реальном ответе такого поля нет."""
    hero_ids = hero_ids or list(range(1, 11))
    buildings = []
    for team, lost_t, lost_r in ((config.TEAM_RADIANT, tow_r, 0),
                                 (config.TEAM_DIRE, tow_d, 0)):
        for i in range(config.N_TOWERS_PER_SIDE):
            if i < lost_t:
                # У снесённого здания team обнуляется — так в живом ответе.
                buildings.append({"team": 0, "type": 0, "lane": 0, "tier": 0,
                                  "x": 0, "y": 0, "destroyed": True})
            else:
                buildings.append({"team": team, "type": 0, "lane": i % 3,
                                  "tier": i % 3 + 1, "destroyed": False})
        for i in range(config.N_RAX_PER_SIDE):
            buildings.append({"team": team, "type": 1, "lane": i % 3,
                              "destroyed": False})
        buildings.append({"team": team, "type": 2, "lane": 0, "destroyed": False})
    match = {"server_steam_id": "90290402786683905", "match_id": "8123456789",
             "timestamp": 1700000000, "game_time": game_time, "game_mode": 2,
             "league_id": 0, "game_state": game_state, "picks": [], "bans": [],
             "lobby_type": 1, "start_timestamp": 1700000000}
    if roshan_timer is not None:
        match["roshan_respawn_timer"] = roshan_timer
    return {
        "match": match,
        "teams": [
            {"team_number": config.TEAM_RADIANT, "team_id": team_ids[0],
             "team_name": "R", "team_tag": "R", "score": 10, "net_worth": nw_r,
             "players": [{"accountid": 1, "playerid": i, "name": "p", "team": 2,
                          "heroid": h, "level": 10, "gold": 500,
                          "kill_count": 3 - i % 3, "death_count": 2,
                          "assists_count": 4,
                          "net_worth": int(nw_r * (0.30 - 0.045 * i)),
                          "team_slot": i}
                         for i, h in enumerate(hero_ids[:5])]},
            {"team_number": config.TEAM_DIRE, "team_id": team_ids[1],
             "team_name": "D", "team_tag": "D", "score": 8, "net_worth": nw_d,
             "players": [{"accountid": 2, "playerid": 5 + i, "name": "q", "team": 3,
                          "heroid": h, "level": 10, "gold": 400,
                          "kill_count": 1 + i % 2, "death_count": 3,
                          "assists_count": 2,
                          "net_worth": int(nw_d * (0.22 - 0.01 * i)),
                          "team_slot": i}
                         for i, h in enumerate(hero_ids[5:])]},
        ],
        "buildings": buildings,
        "graph_data": {"graph_gold": [-43, -50, -671]},
        "delta_frame": True,
    }


def main() -> int:
    model = pick_model(prefer=("model_synthetic_noxp.pkl",), use_xp=False)
    if model is None:
        print(f"ОШИБКА: в {config.MODELS_DIR} нет ни одной модели без признаков "
              f"по опыту.\nЧто делать: `python -m dwp.train --no-xp`.",
              file=sys.stderr)
        return 2
    art = live.load_model(model)
    hero_ids = art["hero_ids"][:10]
    print(f"Модель: {model.name}")

    # 1. Идущий матч: считаем потери по недостаче стоящих зданий.
    st, rep = live.extract_state(
        make_payload(1500, 30000, 24000, tow_r=3, tow_d=5, hero_ids=hero_ids))
    assert st["gold_adv"] == 6000, st["gold_adv"]
    assert st["towers_lost"][config.TEAM_RADIANT] == 3, st["towers_lost"]
    assert st["towers_lost"][config.TEAM_DIRE] == 5, st["towers_lost"]
    assert st["in_progress"]
    assert "roshan_respawn_timer" not in rep.found
    print(f"1. Идущий матч: потери вышек 3/5 посчитаны по недостаче стоящих; "
          f"{rep.found['здания']}")

    # 2. Стадия драфта: всё нулевое -> модуль обязан отказаться.
    st_d, _ = live.extract_state(
        make_payload(0, 0, 0, hero_ids=[0] * 10, game_state=2))
    assert not st_d["in_progress"], "драфт принят за идущий матч"
    print("2. Драфт (heroid=0, net_worth=0): in_progress=False, предсказание "
          "будет отклонено")

    # 3. Серия опросов: производные золота.
    tr = live.LiveTracker()
    for gt, r, d in [(600, 20000, 18000), (660, 20500, 18200), (720, 21500, 18300),
                     (780, 22000, 18500), (840, 23000, 18600), (900, 25000, 18700)]:
        s_, _ = live.extract_state(make_payload(gt, r, d, tow_r=2, tow_d=4,
                                                hero_ids=hero_ids))
        tr.update(gt / 60, s_["gold_adv"], s_.get("roshan_respawn_timer"))
        last_df, warns = live.build_row(art, s_, tr)
    row = last_df.iloc[0]
    assert row["minute"] == 15.0
    assert row["gold_adv_d1"] == 6300 - 4400, row["gold_adv_d1"]
    assert np.isclose(row["gold_adv_slope5"], (6300 - 2000) / 5)
    assert np.isnan(row["minutes_since_roshan"]), "Рошан не может быть известен"
    assert np.isnan(row["roshan_radiant"]) and np.isnan(row["roshan_dire"])
    assert row["tower_adv"] == 4 - 2, row["tower_adv"]
    p, bd = live.predict(art, last_df)
    assert 0.0 < p < 1.0
    print(f"3. Серия опросов: d1={row['gold_adv_d1']:.0f}, "
          f"slope5={row['gold_adv_slope5']:.0f}, tower_adv={row['tower_adv']:.0f}, "
          f"p={p * 100:.1f}%")

    # 4. Рейтинг: подхвачен из истории и подписан.
    #
    # Каким он будет, зависит от артефакта: у моделей с rating_kind ==
    # "player" это рейтинг игроков по account_id, у прежних — командный
    # Elo по team_id. Проверяем не формулировку, а то, что рейтинг вообще
    # объявлен в предупреждениях: молча подставленный ноль — ровно та
    # ошибка, ради которой эти строки и печатаются.
    by_player = art.get("rating_kind") == "player"
    mark = "рейтинг игроков" if by_player else "Elo из истории"
    known = sorted(art.get("elo_final") or {})[:2]
    if len(known) == 2:
        s_, rep_ = live.extract_state(
            make_payload(1500, 30000, 24000, tow_r=1, tow_d=1,
                         hero_ids=hero_ids, team_ids=(known[0], known[1])))
        assert s_["team_ids"][config.TEAM_RADIANT] == known[0]
        _, warns = live.build_row(art, s_, live.LiveTracker())
        assert any(mark in w for w in warns), warns
        print(f"4. рейтинг подхвачен и подписан: "
              f"{[w for w in warns if mark in w][0]}")
    s_, _ = live.extract_state(make_payload(1500, 30000, 24000, hero_ids=hero_ids))
    _, warns = live.build_row(art, s_, live.LiveTracker())
    if by_player:
        # В фикстуре нет account_id вовсе, значит знакомых игроков ноль, и
        # рейтинг обязан быть нулевым — но сказанным вслух.
        assert any("0 из 10 знакомы" in w for w in warns), warns
        print("   без account_id рейтинг обнуляется с явным предупреждением")
    else:
        assert any("team_id = 0" in w for w in warns), warns
        print("   при team_id=0 (паблик) Elo обнуляется с явным предупреждением")

    # 5. Деградация: нет buildings -> NaN, а не ноль.
    pl = make_payload(1500, 30000, 24000, hero_ids=hero_ids)
    del pl["buildings"]
    s_, rep_ = live.extract_state(pl)
    assert "buildings" in rep_.missing
    df, warns = live.build_row(art, s_, live.LiveTracker())
    assert np.isnan(df.iloc[0]["radiant_towers_lost"]), "молчаливый ноль вместо NaN"
    print("5. Без buildings: признаки по вышкам = NaN, не 0")

    # 5b. Фраги и распределение нетворса из лайв-ответа.
    st9, rep9 = live.extract_state(
        make_payload(1500, 30000, 24000, tow_r=1, tow_d=1, hero_ids=hero_ids))
    assert st9["kills"][config.TEAM_RADIANT] == 3 + 2 + 1 + 3 + 2, st9["kills"]
    assert len(st9["player_nw"][config.TEAM_RADIANT]) == 5
    pth = pick_model(prefer=("model_synthetic_live_extra.pkl",
                             "model_synthetic_live.pkl"),
                     use_xp=False, live_features=True, extra_features=True)
    art_x = live.load_model(pth) if pth is not None else None
    if art_x is not None and "kills_adv" in art_x["state_features"]:
        dfx, _ = live.build_row(art_x, st9, live.LiveTracker())
        assert not np.isnan(dfx.iloc[0]["kills_adv"]), "фраги не доехали до признаков"
        assert not np.isnan(dfx.iloc[0]["nw_top_adv"]), "нетворс не доехал"
        print(f"5b. Фраги {st9['kills']} и нетворс по игрокам разобраны; "
              f"kills_adv={dfx.iloc[0]['kills_adv']:+.0f}, "
              f"nw_top_adv={dfx.iloc[0]['nw_top_adv']:+.0f}")
    else:
        print(f"5b. Фраги {st9['kills']} и нетворс разобраны "
              f"(модель без --extra-features, признаки не проверялись)")

    # 6. Чужая нумерация сторон в зданиях -> сообщить, а не обнулить.
    pl = make_payload(1500, 30000, 24000, tow_r=3, tow_d=5, hero_ids=hero_ids)
    for b in pl["buildings"]:
        if b["team"] == config.TEAM_RADIANT:
            b["team"] = 77
    s_, rep_ = live.extract_state(pl)
    assert any("здания по сторонам" in m for m in rep_.missing), rep_.missing
    print("6. Чужая нумерация team в buildings: сообщено, а не посчитано как ноль")

    # 7. Совсем чужая схема.
    s_, rep_ = live.extract_state({"result": {"status": 1, "msg": "not found"}})
    assert "game_time" in rep_.missing and "teams" in rep_.missing
    print(f"7. Чужая схема: не найдено {len(rep_.missing)} полей")

    # 8. Модель с XP должна быть отвергнута.
    xp_model = pick_model(prefer=("model_synthetic.pkl",), use_xp=True)
    if xp_model is None:
        print("8. ПРОПУЩЕНО: нет ни одной модели с признаками по опыту")
    else:
        try:
            live.load_model(xp_model)
        except live.LiveError as e:
            assert "XP" in str(e)
            print(f"8. Модель с признаками по опыту ({xp_model.name}) отвергнута "
                  f"с объяснением")
        else:
            raise AssertionError("модель с XP должна быть отвергнута")

    fx = config.DATA_DIR / "live_fixture.json"
    fx.write_text(json.dumps(make_payload(1500, 32000, 25000, tow_r=4, tow_d=7,
                                          hero_ids=hero_ids),
                             ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nФикстура (слепок живой схемы): {fx}")
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
