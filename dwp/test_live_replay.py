"""Реплей: завершённые матчи прогоняются через ЛАЙВ-путь и сверяются с офлайном.

Зачем нужен отдельно от test_live_fixture. Фикстуры проверяют, что экстрактор
понимает схему ответа и корректно деградирует. Они НЕ проверяют главного: что
лайв-путь считает те же признаки, что и `features.match_state_frame`, на котором
модель училась. Разойтись эти два пути могут молча — на метриках обучения такое
не видно вообще, потому что обучение лайв-кода не касается.

Как устроено. Для каждой минуты завершённого матча собирается ответ в схеме
GetRealtimeStats: `teams[].players[].net_worth` из `gold_t`, `kill_count` из
`kills_log`, `buildings` из разобранных `tower_kills`/`rax_kills`. Дальше он
идёт обычным боевым путём `extract_state -> LiveTracker -> build_row -> предикт`
и сравнивается с офлайн-кадром той же минуты.

ЧТО ЭТОТ СТЕНД ДОКАЗЫВАЕТ, А ЧТО НЕТ. Числитель `gold_adv` здесь одинаков по
построению (обе стороны берут `gold_t`), поэтому расхождение — это ровно цена
кода лайва. Стенд НЕ проверяет, что в бою `teams[].net_worth` равен тому, на чём
модель училась: это другая величина, и меряется она в `dwp.livecheck`.

`draft_logit` из сверки исключён: офлайн берёт ПРЕД-матчевый рейтинг из
истории, а лайв — финальный, накопленный обучением (по `account_id`, а у
старых моделей по `team_id`). Это разные числа по замыслу, а не расхождение
кода. Для сверки вероятностей офлайну подставляется лайвовое значение, чтобы
сравнивались именно поминутные признаки.

Запуск:
    python -m dwp.test_live_replay
    python -m dwp.test_live_replay --n 60 --model models\\live.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from dwp import config, features as F, live  # noqa: E402
from dwp.train import apply_calibrator  # noqa: E402


# Требование стенда — ТОЧНОЕ совпадение, без допусков на «прогрев окон».
# Подключение с нулевой минуты даёт лайву ровно ту же историю, что есть у
# офлайна, поэтому и результат обязан быть тот же. Раньше здесь был список
# исключений (slope5 и kills_adv_d5 первые пять минут отдавали NaN, а офлайн —
# усечённое окно); вместо того чтобы узаконить расхождение допуском, конвенция
# усечённого окна перенесена в LiveTracker._lookback.
#
# Подключение В СЕРЕДИНЕ матча этот стенд не проверяет и проверить не может:
# ранней истории там нет ни у кого. Что в этом случае показывается на экране,
# помечает live.ready_at.

# Признаки, которых лайв-источник не отдаёт вовсе: в бою они всегда NaN.
# Сверять их с офлайном бессмысленно, это ограничение источника, а не кода.
SKIP = set(F.LIVE_UNAVAILABLE) | set(F.EXTRA_LIVE_UNAVAILABLE) | {
    "draft_logit", "xp_adv", "xp_adv_d1"}


def _deaths_by_hero(match: dict) -> dict[int, list[int]]:
    """Минуты смертей каждого героя, восстановленные из фрагов противников.

    В `players[]` смерть есть только итоговым числом, зато у каждого фрага
    в `kills_log` записаны и время, и КОГО убили (`npc_dota_hero_...`).
    Значит поминутные смерти восстанавливаются точно тем же способом, что
    и поминутные фраги, — и раньше здесь стоял жёсткий `death_count: 0`,
    из-за чего реплей не мог показать ни точного фрага в ленте (только
    «замес»), ни мёртвого героя на карте.

    Смерти от крипов и вышек сюда не попадают: их в `kills_log` нет ни у
    кого. На замере это 1.5% смертей — и пусть лучше стенд знает о
    полутора процентах, чем о ста.
    """
    name2id = {}
    try:
        for h in json.loads(config.HEROES_PATH.read_text(encoding="utf-8")):
            nm = h.get("name")
            if nm:
                name2id[str(nm)] = int(h["id"])
    except (OSError, json.JSONDecodeError, KeyError, TypeError):
        return {}
    out: dict[int, list[int]] = {}
    for p in match.get("players") or []:
        for e in (p.get("kills_log") or []):
            if not isinstance(e, dict):
                continue
            t_, key = e.get("time"), e.get("key")
            hid = name2id.get(str(key))
            if t_ is None or hid is None:
                continue
            out.setdefault(hid, []).append(int(t_) // 60)
    return out


def payload_at(match: dict, t: int, parsed: F.ParsedObjectives,
               deaths: dict[int, list[int]] | None = None) -> dict:
    """Ответ GetRealtimeStats, собранный из завершённого матча на минуте t."""
    if deaths is None:
        deaths = _deaths_by_hero(match)
    teams = []
    for tn, want_radiant in ((config.TEAM_RADIANT, True), (config.TEAM_DIRE, False)):
        players = []
        for p in match["players"]:
            if (int(p["player_slot"]) < 128) != want_radiant:
                continue
            g = p["gold_t"]
            kills = sum(1 for e in (p.get("kills_log") or [])
                        if isinstance(e, dict) and e.get("time") is not None
                        and int(e["time"]) // 60 <= t)

            def at(key: str, default=0.0) -> float:
                v = p.get(key)
                if not isinstance(v, list) or not v:
                    return default
                return float(v[min(t, len(v) - 1)])

            # Инвентарь и рюкзак в схеме живого ответа: шесть слотов
            # инвентаря, дальше рюкзак, пустой слот -1. Берём КОНЕЧНЫЙ
            # состав — поминутного в выгрузке нет. Для сверки признаков это
            # безразлично (предметов в модели нет вовсе), а вот блок сборок
            # без них не рисуется вообще, и стенд молча его не проверял.
            inv = [int(p.get(f"item_{k}") or 0) or -1 for k in range(6)]
            inv += [int(p.get(f"backpack_{k}") or 0) or -1 for k in range(3)]
            players.append({"accountid": int(p.get("account_id") or 0),
                            "playerid": 0, "name": "",
                            "gold": 0.0, "items": inv,
                            "team": tn, "heroid": int(p["hero_id"]),
                            # Уровень тот же, что офлайн считает из xp_t: так
                            # проверяется проводка признака, а не таблица —
                            # таблица выведена и проверена отдельно (config).
                            "level": F.level_from_xp(at("xp_t")),
                            "lh_count": at("lh_t"), "denies_count": at("dn_t"),
                            "kill_count": kills,
                            "death_count": sum(
                                1 for mn in deaths.get(int(p["hero_id"]), ())
                                if mn <= t),
                            "net_worth": float(g[min(t, len(g) - 1)])})
        teams.append({"team_number": tn, "team_id": 0, "team_name": "",
                      "net_worth": sum(x["net_worth"] for x in players),
                      "players": players})

    buildings = []
    for tn, is_radiant in ((config.TEAM_RADIANT, True), (config.TEAM_DIRE, False)):
        lost_r = sum(1 for mn, v in parsed.rax_kills if v == is_radiant and mn <= t)
        # Ярусы расставляются по РЕАЛЬНЫМ событиям, а не по порядку: раньше
        # стоящим вышкам приписывался ярус i%3+1, и проверить признак по
        # третьему ярусу этим было нельзя. Состав стороны: 3+3+3+2 = 11.
        lost_by_tier = {k: sum(1 for mn, v, tier in parsed.tower_kills_tier
                               if v == is_radiant and tier == k and mn <= t)
                        for k in (1, 2, 3, 4)}
        stand = [{"team": tn, "type": 0, "lane": j % 3 + 1, "tier": k,
                  "destroyed": False}
                 for k, n in ((1, 3), (2, 3), (3, 3), (4, 2))
                 for j in range(max(0, n - lost_by_tier[k]))]
        # Список зданий обязан быть ровно N_BUILDINGS_TOTAL: на этом держится
        # проверка полноты в extract_state. Поэтому снесённых ровно столько,
        # сколько не стоит, а не столько, сколько событий (они могли разойтись,
        # если у какой-то вышки не разобрался ярус).
        # У снесённого здания в живом ответе обнуляются все поля разом.
        buildings += [{"team": 0, "type": 0, "lane": 0, "tier": 0,
                       "destroyed": True}] * (config.N_TOWERS_PER_SIDE - len(stand))
        buildings += stand
        for i in range(config.N_RAX_PER_SIDE):
            buildings.append({"team": 0, "type": 0, "lane": 0, "tier": 0,
                              "destroyed": True} if i < lost_r else
                             {"team": tn, "type": 1, "lane": i % 3,
                              "destroyed": False})
        buildings.append({"team": tn, "type": 2, "lane": 0, "destroyed": False})

    return {"match": {"server_steam_id": "0", "match_id": str(match["match_id"]),
                      "game_time": t * 60, "game_state": 5,
                      "game_mode": 2, "league_id": 0},
            "teams": teams, "buildings": buildings,
            "graph_data": {"graph_gold": []}, "delta_frame": True}


def predict_rows(art: dict, df: pd.DataFrame) -> np.ndarray:
    """Офлайн-путь. Для ансамбля усредняются вероятности — как и в бою.

    `draft_logit` тут уже подставлен вызывающим и одинаков для всех
    участников (стенд уравнивает его специально, чтобы сверялись
    поминутные признаки, а не разница драфт-моделей).
    """
    feats = art["state_features"]
    ps = []
    for m in live.members(art):
        raw = m["booster"].predict(df[feats], num_iteration=m["booster"].best_iteration)
        eps = max(m.get("calib_eps", 0.0), 1e-6)
        ps.append(np.clip(apply_calibrator(m["iso"], raw), eps, 1 - eps))
    return np.mean(ps, axis=0)


def replay_match(art: dict, match: dict, parsed: F.ParsedObjectives,
                 offline: pd.DataFrame) -> pd.DataFrame:
    """Поминутный прогон одного матча боевым путём."""
    tracker = live.LiveTracker()
    rows = []
    for t in range(int(offline["minute"].max()) + 1):
        st, _ = live.extract_state(payload_at(match, t, parsed))
        minute = float(st["game_time"]) / 60.0
        tracker.update(minute, st.get("gold_adv", np.nan),
                       st.get("roshan_respawn_timer"))
        row, _ = live.build_row(art, st, tracker)
        rows.append(row.iloc[0])
    return pd.DataFrame(rows).reset_index(drop=True)


def usable_for_replay(match: dict) -> tuple[F.ParsedObjectives | None, str]:
    """Матч годится, если по нему вообще есть с чем сверяться."""
    if not match.get("radiant_gold_adv"):
        return None, "нет radiant_gold_adv"
    if len(match.get("players") or []) != 10:
        return None, "игроков не 10"
    if any(not isinstance(p.get("gold_t"), list) or not p["gold_t"]
           for p in match["players"]):
        return None, "нет gold_t"
    parsed = F.parse_objectives(match)
    if not (parsed.tower_ok and parsed.rax_ok):
        return None, "вышки или бараки не разобраны"
    if not parsed.tower_tier_ok:
        # Без яруса у каждой вышки собрать честный buildings[] нельзя:
        # стенд не имеет права угадывать, какая именно вышка пала.
        return None, "ярус известен не у всех вышек"
    for k, n in ((1, 3), (2, 3), (3, 3), (4, 2)):
        for side in (True, False):
            if sum(1 for _, v, tier in parsed.tower_kills_tier
                   if v == side and tier == k) > n:
                return None, f"снесённых вышек яруса {k} больше {n}"
    for side in (True, False):
        if sum(1 for _, v in parsed.tower_kills if v == side) > config.N_TOWERS_PER_SIDE:
            return None, "снесённых вышек больше 11"
        if sum(1 for _, v in parsed.rax_kills if v == side) > config.N_RAX_PER_SIDE:
            return None, "снесённых бараков больше 6"
    return parsed, ""


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Сверка лайв-пути с офлайн-путём.")
    ap.add_argument("--n", type=int, default=40, help="сколько матчей прогнать")
    ap.add_argument("--model", type=Path, nargs="+",
                    default=[config.MODELS_DIR / "live.pkl"],
                    help="модель или несколько (ансамбль): шаблоны раскрываются")
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--max-dp", type=float, default=1e-6,
                    help="допустимое расхождение вероятности (пути обязаны "
                         "совпадать точно, допуск только на шум float)")
    args = ap.parse_args(argv)

    src = config.RAW_MATCHES_DIR if args.source == "real" else config.SYNTH_MATCHES_DIR
    files = sorted(src.glob("*.json"))
    if not files:
        print(f"ОШИБКА: в {src} нет матчей.", file=sys.stderr)
        return 2
    art = live.load_models(args.model)
    feats = [f for f in art["state_features"] if f not in SKIP]
    print(f"Модель: {art.get('name')}  признаков сверяется: {len(feats)} из "
          f"{len(art['state_features'])}")
    print(f"Матчей в {src}: {len(files)}, берём {args.n}")

    picked, megas, skipped = [], [], 0
    for f in files:
        if len(picked) >= args.n and megas:
            break
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            skipped += 1
            continue
        parsed, why = usable_for_replay(m)
        if parsed is None:
            skipped += 1
            continue
        mega = any(sum(1 for _, v in parsed.tower_kills if v == side)
                   == config.N_TOWERS_PER_SIDE for side in (True, False))
        if len(picked) < args.n:
            picked.append((m, parsed))
            if mega:
                megas.append(int(m["match_id"]))
        elif mega:
            # Матч с потерей всех вышек обязателен: именно на нём ломался
            # подсчёт зданий, и без него стенд не покрывает главный случай.
            picked.append((m, parsed))
            megas.append(int(m["match_id"]))
    print(f"Отобрано {len(picked)} матчей (пропущено {skipped}), из них с потерей "
          f"всех {config.N_TOWERS_PER_SIDE} вышек: {len(megas)}")
    if not megas:
        print("ОШИБКА: не нашлось ни одного матча, где сторона потеряла все вышки.\n"
              "Без него стенд не проверяет главный случай — увеличьте --n.",
              file=sys.stderr)
        return 2

    mismatches: dict[str, list] = {}
    dp_all, dp_mega, worst = [], [], (-1.0, None, None)
    for i, (m, parsed) in enumerate(picked, 1):
        offline = F.match_state_frame(m, parsed)
        lv = replay_match(art, m, parsed, offline)
        # Elo офлайн и лайв берут из разных источников по замыслу — уравниваем,
        # чтобы сверялись поминутные признаки, а не разница в Elo.
        offline = offline.copy()
        offline["draft_logit"] = lv["draft_logit"].to_numpy()
        p_off, p_lv = predict_rows(art, offline), predict_rows(art, lv)
        dp = np.abs(p_lv - p_off)
        dp_all.append(dp)
        is_mega = int(m["match_id"]) in megas
        if is_mega:
            dp_mega.append(dp)
        if dp.max() > worst[0]:
            worst = (float(dp.max()), int(m["match_id"]), int(np.argmax(dp)))

        minute = offline["minute"].to_numpy()
        for f_ in feats:
            a, b = offline[f_].to_numpy(dtype=float), lv[f_].to_numpy(dtype=float)
            bad = ~((np.isnan(a) & np.isnan(b)) | np.isclose(a, b, equal_nan=False))
            if not bad.any():
                continue
            k = int(np.argmax(bad))
            mismatches.setdefault(f_, []).append(
                (int(m["match_id"]), int(minute[k]), float(a[k]), float(b[k]),
                 int(bad.sum())))
        if i % 10 == 0 or i == len(picked):
            print(f"  прогнано {i}/{len(picked)}")

    dp_all = np.concatenate(dp_all)
    print()
    print("=" * 74)
    print(f"строк сверено: {len(dp_all)}")
    print(f"|p_лайв - p_офлайн|: среднее {dp_all.mean():.4f}  "
          f"p90 {np.percentile(dp_all, 90):.4f}  max {dp_all.max():.4f}")
    if dp_mega:
        d = np.concatenate(dp_mega)
        print(f"  из них матчи с потерей всех вышек: строк {len(d)}, "
              f"среднее {d.mean():.4f}, max {d.max():.4f}")
    if worst[0] > 0:
        print(f"худшая строка: матч {worst[1]}, минута {worst[2]}, "
              f"расхождение {worst[0]:.4f}")

    ok = True
    if mismatches:
        ok = False
        print("\nПРИЗНАКИ РАСХОДЯТСЯ ПОСЛЕ ПРОГРЕВА:")
        for f_, items in sorted(mismatches.items()):
            total = sum(x[4] for x in items)
            mid, mn, a, b = items[0][:4]
            print(f"    {f_:<24} {total} строк в {len(items)} матчах; пример: "
                  f"матч {mid}, минута {mn}: офлайн {a}, лайв {b}")
    if dp_all.max() > args.max_dp:
        ok = False
        print(f"\nВЕРОЯТНОСТЬ РАСХОДИТСЯ: max {dp_all.max():.4f} > "
              f"допуска {args.max_dp:.4f}")
    if not ok:
        print("\nСТЕНД НЕ ПРОЙДЕН")
        return 1
    print("\nВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
