"""Проверка допущений парсера на реальных данных.

Модуль ничего не додумывает. Каждое допущение либо подтверждается
данными, либо помечается как непроверяемое. Особый случай — поле `team`
в старых событиях CHAT_MESSAGE_TOWER_KILL: его смысл (кто снёс или кто
потерял) документацией не закреплён, поэтому он определяется
эмпирически — сверкой числа событий с popcount битовой маски
tower_status_* в конце матча. Обе гипотезы проверяются одинаково, и
печатается та, что сходится.

Коды возврата: 0 — всё сошлось, 1 — есть проваленные проверки,
2 — нечего проверять (нет данных).
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from . import config, features as F

OK, WARN, FAIL = "OK    ", "ВНИМАНИЕ", "ПРОВАЛ"


class Report:
    def __init__(self) -> None:
        self.failed = 0
        self.warned = 0

    def say(self, status: str, name: str, detail: str = "") -> None:
        if status == FAIL:
            self.failed += 1
        if status == WARN:
            self.warned += 1
        print(f"[{status}] {name}")
        for line in detail.splitlines():
            if line.strip():
                print(f"         {line}")


def check_fields(matches: list[dict], rep: Report) -> None:
    required = ["match_id", "radiant_win", "duration", "players", "objectives",
                "radiant_gold_adv"]
    optional = ["radiant_xp_adv", "tower_status_radiant", "tower_status_dire",
                "barracks_status_radiant", "barracks_status_dire",
                "radiant_team_id", "dire_team_id", "start_time"]
    n = len(matches)
    miss = {f: sum(1 for m in matches if m.get(f) is None) for f in required}
    bad = {f: c for f, c in miss.items() if c}
    if bad:
        rep.say(FAIL, "Обязательные поля матча",
                "\n".join(f"{f}: отсутствует в {c} из {n} матчей" for f, c in bad.items())
                + "\nЧто делать: такие матчи надо отсеивать (features.usable_matches "
                  "это делает), но если доля велика — изменилась схема /matches/{id}.")
    else:
        rep.say(OK, "Обязательные поля матча", f"все {len(required)} присутствуют в {n} матчах")

    opt = {f: sum(1 for m in matches if m.get(f) is None) for f in optional}
    absent = {f: c for f, c in opt.items() if c}
    rep.say(WARN if absent else OK, "Необязательные поля",
            ("\n".join(f"{f}: нет в {c} из {n}" for f, c in absent.items())
             + "\nПризнаки на их основе будут NaN, а не нулём.") if absent
            else f"все {len(optional)} присутствуют")


def check_gold_length(matches: list[dict], rep: Report) -> None:
    diffs = []
    for m in matches:
        g = m.get("radiant_gold_adv")
        if g and m.get("duration"):
            diffs.append(len(g) - (int(m["duration"]) // 60 + 1))
    if not diffs:
        rep.say(FAIL, "Длина radiant_gold_adv", "нечего сверять")
        return
    c = Counter(diffs)
    worst = max(abs(d) for d in diffs)
    rep.say(OK if worst <= 1 else WARN, "Длина radiant_gold_adv против duration",
            f"len(gold) - (duration//60 + 1): {dict(c.most_common(5))}\n"
            f"Допущение «индекс массива = минута матча» "
            f"{'подтверждается' if worst <= 1 else 'НЕ подтверждается, максимум расхождения '
                                                   f'{worst} — проверьте вручную'}")


def check_players(matches: list[dict], rep: Report) -> None:
    has_slot = has_isr = both_agree = disagree = 0
    bad_sides = 0
    for m in matches:
        ps = m.get("players") or []
        for p in ps:
            if p.get("player_slot") is not None:
                has_slot += 1
            if "isRadiant" in p:
                has_isr += 1
            if p.get("player_slot") is not None and "isRadiant" in p:
                if (int(p["player_slot"]) < 128) == bool(p["isRadiant"]):
                    both_agree += 1
                else:
                    disagree += 1
        try:
            F.match_sides(m)
        except ValueError:
            bad_sides += 1
    rep.say(FAIL if disagree else OK, "Определение стороны игрока",
            f"player_slot есть у {has_slot} записей, isRadiant у {has_isr}\n"
            f"согласуются: {both_agree}, противоречат: {disagree}\n"
            f"матчей, где стороны не 5/5: {bad_sides}\n"
            f"Парсер берёт player_slot<128, isRadiant только как запасной вариант.")


def check_heroes(matches: list[dict], rep: Report) -> None:
    try:
        hero_ids, id2idx, _ = F.load_heroes()
    except FileNotFoundError as e:
        rep.say(FAIL, "Справочник героев", str(e))
        return
    seen, unknown = Counter(), Counter()
    for m in matches:
        for p in m.get("players") or []:
            h = p.get("hero_id")
            if h:
                seen[int(h)] += 1
                if int(h) not in id2idx:
                    unknown[int(h)] += 1
    cov = len(seen) / max(len(hero_ids), 1)
    rep.say(FAIL if unknown else (WARN if cov < 0.5 else OK), "hero_id против /heroes",
            f"в справочнике {len(hero_ids)} героев, в матчах встретилось {len(seen)} "
            f"({cov * 100:.0f}%)\n"
            + (f"НЕИЗВЕСТНЫЕ hero_id: {dict(unknown.most_common(10))}\n"
               f"Что делать: обновите справочник (`python -m dwp.collect --heroes`). "
               f"Неизвестный герой сейчас даёт нулевой вклад — это тихая потеря сигнала."
               if unknown else
               "Неизвестных id нет. Индексация идёт по отсортированному id, "
               "дыры в нумерации безопасны."))


def check_objectives(matches: list[dict], rep: Report) -> None:
    types, keys, unknown_keys = Counter(), Counter(), Counter()
    team_values = Counter()
    fmts = Counter()
    no_obj = 0
    for m in matches:
        objs = m.get("objectives")
        if not objs:
            no_obj += 1
            continue
        for o in objs:
            if not isinstance(o, dict):
                types["<не словарь>"] += 1
                continue
            t = str(o.get("type"))
            types[t] += 1
            if t == "building_kill":
                k = str(o.get("key", ""))
                if "goodguys" in k or "badguys" in k:
                    keys[k.replace("goodguys", "<side>").replace("badguys", "<side>")] += 1
                else:
                    unknown_keys[k] += 1
            if "team" in o:
                team_values[repr(o["team"])] += 1
        fmts[F.parse_objectives(m).fmt] += 1

    rep.say(WARN if no_obj else OK, "Наличие objectives",
            f"матчей без objectives: {no_obj} из {len(matches)}")
    print(f"         типы событий: {dict(types.most_common(12))}")
    print(f"         форматы по матчам: {dict(fmts)}")

    known_types = {"building_kill", "CHAT_MESSAGE_TOWER_KILL",
                   "CHAT_MESSAGE_BARRACKS_KILL", "CHAT_MESSAGE_ROSHAN_KILL",
                   "CHAT_MESSAGE_MINIBOSS_KILL"}
    unknown_types = {t: c for t, c in types.items() if t not in known_types}
    rep.say(WARN if unknown_types else OK, "Нераспознанные типы событий",
            (f"{dict(Counter(unknown_types).most_common(10))}\n"
             "Они не падают и не считаются как ноль — просто игнорируются. "
             "Если среди них есть события по зданиям, парсер надо дополнить."
             if unknown_types else "нет"))

    if keys:
        print(f"         шаблоны building_kill: {dict(keys.most_common(10))}")
    rep.say(WARN if unknown_keys else OK, "building_kill без goodguys/badguys в key",
            (f"{dict(unknown_keys.most_common(10))}\n"
             "Сторону из такого ключа не определить, событие пропущено."
             if unknown_keys else "нет"))

    allowed = {repr(config.TEAM_RADIANT), repr(config.TEAM_DIRE)}
    other = {v: c for v, c in team_values.items() if v not in allowed}
    rep.say(WARN if other else OK,
            f"Допущение TEAM_RADIANT={config.TEAM_RADIANT}, TEAM_DIRE={config.TEAM_DIRE}",
            f"встреченные значения team: {dict(team_values.most_common(8))}"
            + ("\nЕсть значения вне {2,3} — проверьте, что они означают."
               if other else ""))


def resolve_tower_team_meaning(matches: list[dict], rep: Report) -> None:
    """Эмпирически определяет смысл поля `team` в старом формате.

    Сверка: число событий сноса вышек стороны X должно совпасть с
    (11 − popcount(tower_status_X)). Гипотеза, при которой сходится
    большинство матчей, и есть верная.
    """
    old = [m for m in matches
           if any(isinstance(o, dict) and o.get("type") == "CHAT_MESSAGE_TOWER_KILL"
                  for o in (m.get("objectives") or []))
           and isinstance(m.get("tower_status_radiant"), int)
           and isinstance(m.get("tower_status_dire"), int)]
    if not old:
        rep.say(WARN, "Смысл поля team в CHAT_MESSAGE_TOWER_KILL",
                "матчей со старым форматом и битовой маской не найдено — "
                "проверить нечем, допущение остаётся непроверенным")
        return
    scores = {}
    for meaning in ("victim", "killer"):
        ok = sum(1 for m in old
                 if F.parse_objectives(m, tower_team_means=meaning).tower_consistent)
        scores[meaning] = ok / len(old)
    best = max(scores, key=scores.get)
    margin = scores[best] - min(scores.values())
    detail = (f"матчей со старым форматом: {len(old)}\n"
              f"доля сошедшихся с tower_status: "
              + ", ".join(f"{k}={v:.3f}" for k, v in scores.items()) + "\n")
    if scores[best] < 0.8:
        rep.say(FAIL, "Смысл поля team в CHAT_MESSAGE_TOWER_KILL",
                detail + "НИ ОДНА гипотеза не сходится. Не считайте признаки по "
                         "вышкам из старого формата, пока не разберётесь: они "
                         "будут неверны, а по метрикам это почти не видно.")
    elif margin < 0.2:
        rep.say(WARN, "Смысл поля team в CHAT_MESSAGE_TOWER_KILL",
                detail + f"гипотезы различаются слабо (разрыв {margin:.3f}), "
                         f"вывод ненадёжен")
    else:
        cur = config.OLD_TOWER_KILL_TEAM_MEANS
        rep.say(OK if best == cur else FAIL,
                "Смысл поля team в CHAT_MESSAGE_TOWER_KILL",
                detail + f"данные говорят: team = {best}\n"
                + (f"config.OLD_TOWER_KILL_TEAM_MEANS = {cur!r} — совпадает"
                   if best == cur else
                   f"config.OLD_TOWER_KILL_TEAM_MEANS = {cur!r} — НЕ СОВПАДАЕТ.\n"
                   f"Что делать: поставьте OLD_TOWER_KILL_TEAM_MEANS = {best!r} "
                   f"и переобучите модель."))


def check_towers_all_formats(matches: list[dict], rep: Report) -> None:
    """Сверка вышек с tower_status для ВСЕХ форматов, не только старого.

    Раньше эта сверка жила внутри разрешения неоднозначности `team` и
    поэтому запускалась только на старом формате. На данных, где весь
    массив в новом формате, самый важный признак оставался
    непроверенным, а модуль писал «проверить нечем» — и это выглядело
    как «всё в порядке».
    """
    ok = bad = none = 0
    for m in matches:
        c = F.parse_objectives(m).tower_consistent
        if c is None:
            none += 1
        elif c:
            ok += 1
        else:
            bad += 1
    total = ok + bad
    share = ok / total if total else 0.0
    rep.say(OK if total and share >= 0.95 else (FAIL if total else WARN),
            "Вышки: сверка с tower_status_* (все форматы)",
            f"сошлось {ok}, разошлось {bad}, сверить нечем {none}\n"
            + ("проверить нечем: нет tower_status или событий по вышкам"
               if not total else
               f"доля совпадений {share:.3f}\n"
               + ("" if share >= 0.95 else
                  "Парсер вышек считает не то. Обучаться нельзя: tower_adv — "
                  "один из сильнейших сигналов, и ошибка в нём по метрикам "
                  "почти не видна.")))


def check_team_vs_slot(matches: list[dict], rep: Report) -> None:
    """Жёсткая сверка поля `team` с `player_slot` в одном и том же событии.

    Там, где есть оба поля, конвенция определяется однозначно и без
    гипотез. Там, где player_slot отсутствует, — так и пишем.
    """
    for otype in ("CHAT_MESSAGE_MINIBOSS_KILL", "CHAT_MESSAGE_ROSHAN_KILL",
                  "CHAT_MESSAGE_AEGIS"):
        evs = [o for m in matches for o in (m.get("objectives") or [])
               if isinstance(o, dict) and o.get("type") == otype]
        if not evs:
            continue
        both = [o for o in evs if o.get("player_slot") is not None and "team" in o]
        if not both:
            rep.say(WARN, f"team против player_slot: {otype}",
                    f"событий {len(evs)}, но player_slot нет ни у одного — "
                    f"жёстко проверить нельзя, остаётся косвенная проверка")
            continue
        agree = sum(1 for o in both
                    if (int(o["player_slot"]) < 128) == (F.team_is_radiant(o["team"]) is True))
        share = agree / len(both)
        if share >= 0.99:
            verdict, note = OK, "team = команда УБИЙЦЫ, подтверждено жёстко"
        elif share <= 0.01:
            verdict, note = FAIL, ("team = команда ЖЕРТВЫ. Парсер трактует "
                                   "наоборот, признак будет с обратным знаком.")
        else:
            verdict, note = FAIL, ("поле неоднозначно — стройте признак от "
                                   "player_slot, а не от team")
        rep.say(verdict, f"team против player_slot: {otype}",
                f"событий {len(evs)}, из них с обоими полями {len(both)}\n"
                f"совпало {agree} ({share:.3f})\n{note}")


def check_player_series(matches: list[dict], rep: Report) -> None:
    """Проверяет поигроковые ряды, на которых стоят --extra-features.

    Ключевая сверка: сумма gold_t по Radiant минус сумма по Dire должна
    совпадать с radiant_gold_adv. Если не совпадает — gold_t это не то,
    чем я его считаю, и признаки по распределению нетворса будут врать.
    """
    have = {"gold_t": 0, "kills_log": 0, "buyback_log": 0, "times": 0}
    errs, checked = [], 0
    for m in matches:
        ps = m.get("players") or []
        for k in have:
            if any(isinstance(p.get(k), list) and p.get(k) for p in ps):
                have[k] += 1
        g = m.get("radiant_gold_adv")
        if not g or len(ps) != 10:
            continue
        ser = F.parse_player_series(m, len(g) - 1)
        if ser.gold is None or ser.is_radiant is None:
            continue
        calc = ser.gold[ser.is_radiant].sum(0) - ser.gold[~ser.is_radiant].sum(0)
        ref = np.asarray(g, dtype=float)[:len(calc)]
        rel = np.nanmax(np.abs(calc - ref)) / max(1.0, float(np.nanmax(np.abs(ref))))
        errs.append(rel)
        checked += 1
    n = len(matches)
    rep.say(OK if have["gold_t"] else WARN, "Поигроковые ряды",
            f"матчей с gold_t: {have['gold_t']}/{n}, kills_log: {have['kills_log']}, "
            f"buyback_log: {have['buyback_log']}, times: {have['times']}\n"
            + ("Признаки --extra-features строятся на них; где их нет, "
               "соответствующие признаки станут NaN."
               if have["gold_t"] else
               "Без gold_t флаг --extra-features даст NaN в половине признаков."))
    if checked:
        med = float(np.median(errs))
        worst = float(np.max(errs))
        ok = med < 0.05
        rep.say(OK if ok else FAIL,
                "sum(gold_t) против radiant_gold_adv",
                f"сверено матчей: {checked}\n"
                f"относительное расхождение: медиана {med:.3f}, максимум {worst:.3f}\n"
                + ("Допущение «gold_t = нетворс игрока по минутам» "
                   "подтверждается." if ok else
                   "НЕ СХОДИТСЯ. gold_t означает не то, что предполагает парсер, "
                   "и признаки nw_top_adv / nw_conc_adv будут неверны. "
                   "Не используйте --extra-features, пока не разберётесь."))


def check_roshan_team(matches: list[dict], rep: Report) -> None:
    """Слабая проверка: независимой сверки для Рошана нет.

    Единственное, что можно измерить, — правдоподобие: под верной
    трактовкой победитель матча должен убивать Рошана чаще проигравшего.
    Это не доказательство, и так и написано в выводе.
    """
    diffs = {"killer": [], "victim": []}
    n = 0
    for m in matches:
        if m.get("radiant_win") is None:
            continue
        objs = [o for o in (m.get("objectives") or [])
                if isinstance(o, dict) and o.get("type") == "CHAT_MESSAGE_ROSHAN_KILL"]
        if not objs:
            continue
        n += 1
        sides = [F.team_is_radiant(o.get("team")) for o in objs]
        sides = [s_ for s_ in sides if s_ is not None]
        for meaning in ("killer", "victim"):
            # meaning="killer": team — это убийца, значит Radiant убил там,
            # где сторона == Radiant. meaning="victim": наоборот.
            killers = sides if meaning == "killer" else [not s_ for s_ in sides]
            r = sum(1 for s_ in killers if s_)
            d = len(killers) - r
            diffs[meaning].append((r - d) if m["radiant_win"] else (d - r))
    if not n:
        rep.say(WARN, "Смысл team в CHAT_MESSAGE_ROSHAN_KILL",
                "событий по Рошану не найдено")
        return
    means = {k: float(np.mean(v)) for k, v in diffs.items()}
    best = max(means, key=means.get)
    rep.say(WARN, "Смысл team в CHAT_MESSAGE_ROSHAN_KILL (проверка косвенная)",
            f"матчей с Рошаном: {n}\n"
            f"средний перевес победителя по Рошанам: "
            + ", ".join(f"{k}={v:+.3f}" for k, v in means.items()) + "\n"
            f"правдоподобнее: {best}; сейчас в config: "
            f"{config.OLD_ROSHAN_KILL_TEAM_MEANS!r}\n"
            "Это НЕ доказательство: сверять число Рошанов не с чем, "
            "битовой маски для них нет. Признак по Рошану слабый, "
            "и ошибка в его знаке дорого не стоит — но и полагаться нельзя.")


def check_rax(matches: list[dict], rep: Report) -> None:
    n_ok = n_incons = n_none = 0
    for m in matches:
        p = F.parse_objectives(m)
        if p.rax_consistent is None:
            n_none += 1
        elif p.rax_consistent:
            n_ok += 1
        else:
            n_incons += 1
    total = n_ok + n_incons
    share = n_ok / total if total else 0.0
    rep.say(OK if share >= 0.8 else (WARN if total else WARN),
            "Бараки: сверка с barracks_status_*",
            f"сошлось {n_ok}, разошлось {n_incons}, сверить нечем {n_none}\n"
            + ("" if share >= 0.8 else
               "Низкая доля совпадений. Признак rax_adv станет NaN там, где "
               "события не разобраны, — это лучше, чем молчаливый ноль, но "
               "сигнал теряется."))



def _safe_stdout() -> None:
    """Не ронять программу из-за символа, которого нет в кодовой странице.

    Windows-консоль по умолчанию бывает cp866 или cp1251: первая не знает
    длинного тире и стрелок, вторая — ещё и блочной графики. Падать из-за
    оформления недопустимо, поэтому непечатаемое заменяется на '?'.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(description="Валидация допущений парсера на данных.")
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--dir", type=Path, default=None)
    ap.add_argument("--limit", type=int, default=1500)
    args = ap.parse_args(argv)

    directory = args.dir or (config.RAW_MATCHES_DIR if args.source == "real"
                             else config.SYNTH_MATCHES_DIR)
    matches = F.load_matches(directory, limit=args.limit)
    if not matches:
        print(f"ОШИБКА: в {directory} нет матчей.\n"
              f"Что делать: `python -m dwp.collect --n 500` для реальных данных "
              f"или `python -m dwp.synthetic` для синтетики, затем повторите.",
              file=sys.stderr)
        return 2

    print("=" * 78)
    print(f"Проверка схемы: {directory}, матчей {len(matches)}")
    print("=" * 78)
    rep = Report()
    check_fields(matches, rep)
    check_gold_length(matches, rep)
    check_players(matches, rep)
    check_heroes(matches, rep)
    check_objectives(matches, rep)
    check_towers_all_formats(matches, rep)
    resolve_tower_team_meaning(matches, rep)
    check_team_vs_slot(matches, rep)
    check_rax(matches, rep)
    check_player_series(matches, rep)
    check_roshan_team(matches, rep)
    print("=" * 78)
    print(f"Провалов: {rep.failed}, предупреждений: {rep.warned}")
    if rep.failed:
        print("Формат данных НЕ соответствует допущениям парсера. "
              "Обучать модель на этих данных нельзя, пока не разберётесь выше.")
    print("=" * 78)
    return 1 if rep.failed else 0


if __name__ == "__main__":
    sys.exit(main())
