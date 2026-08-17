"""Сквозная проверка веб-пути на НАСТОЯЩЕМ матче, без сети.

Зачем отдельно от `test_live_replay`. Тот стенд сверяет числа: лайв-путь
против офлайн-пути, признак за признаком. А панель — это ещё и всё
вокруг числа: лента событий, коридор, темп, сборки, карта, разбор драфта.
Каждый из этих блоков собирается в `Poller.snapshot`, и уронить панель
может любой из них, ничего не сделав с самим числом.

Правило проекта: инвариант, проверенный только на фикстурах, которые вы
сами и придумали, не проверен. Поэтому здесь берётся завершённый матч из
`data/matches` и из него поминутно собираются ответы в схеме
GetRealtimeStats — тем же кодом, что и в реплей-стенде.

Сеть отрезана намеренно: Stratz и OpenDota подменяются заглушками. Их
доступность — не предмет этой проверки, а падать из-за неё стенд не
должен.

    python -m dwp.test_web_snapshot
    python -m dwp.test_web_snapshot --model models\\ens_*.pkl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

from . import config, features as F, live, web
from .test_live_replay import payload_at, usable_for_replay

FAILED: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if cond else 'ПРОВАЛ'} {name}" + (f": {detail}" if detail else ""))
    if not cond:
        FAILED.append(name)


def pick_match(limit: int = 400) -> tuple[dict, F.ParsedObjectives] | None:
    """Первый матч, который годится для реплея и в котором были фраги."""
    for f in sorted(config.RAW_MATCHES_DIR.glob("*.json"))[:limit]:
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        parsed, _why = usable_for_replay(m)
        if parsed is None:
            continue
        kills = sum(len(p.get("kills_log") or []) for p in m["players"])
        if kills >= 10:
            return m, parsed
    return None


def check_page() -> None:
    """Разметка и скрипт панели согласованы между собой.

    Ловит целый класс поломок, который снимок поймать не может: панель
    рисуется браузером, и опечатка в id или лишняя скобка в скрипте не
    роняет ни один питоновский стенд — просто половина экрана остаётся
    пустой. Поэтому здесь проверяется, что каждый id, к которому лезет
    скрипт, в разметке есть, что окна раскладки существуют, и что скрипт
    вообще разбирается парсером JS.
    """
    page = web.PAGE
    body = page[page.find("</style>"):]
    js = re.search(r"<script>(.*)</script>", page, re.S).group(1)
    ids = set(re.findall(r'id="([\w-]+)"', body))

    missing = sorted({n for n in re.findall(r'getElementById\("([\w-]+)"\)', js)
                      if n not in ids})
    check("скрипт не лезет в несуществующие id", not missing,
          ", ".join(missing))

    cards = re.search(r"const CARDS=\[([^\]]*)\]", js)
    got = re.findall(r'"([\w-]+)"', cards.group(1)) if cards else []
    check("список окон раскладки найден в скрипте", bool(got), str(got))
    lost = [c for c in got if c not in ids]
    check("каждое окно раскладки есть в разметке", not lost, ", ".join(lost))
    # Значения по умолчанию должны покрывать ровно тот же набор окон:
    # окно без ширины уехало бы на всю строку, окно без масштаба — исчезло.
    dflt = re.search(r"const LAY_DEF=\{(.*?)\};", js, re.S)
    keys = re.findall(r"(\w+):\[", dflt.group(1)) if dflt else []
    check("у каждого окна есть размер по умолчанию", sorted(keys) == sorted(got),
          f"{sorted(keys)} против {sorted(got)}")

    # Свободный режим: маркер без своего правила в CSS существует в DOM, но
    # имеет нулевой размер — схватиться за него нельзя, а на глаз разницы
    # никакой. Ровно тот класс поломок, ради которого этот тест и написан.
    css = page[:page.find("</style>")]
    kinds = re.search(r'for\(const k of \[([^\]]*)\]\)\s*\n?\s*add\("hd "', js)
    marks = re.findall(r'"(\w+)"', kinds.group(1)) if kinds else []
    check("маркеры размера перечислены в скрипте", len(marks) == 8, str(marks))
    lost = [k for k in marks + ["mv"]
            if f"#panel.free>.card>.hd.{k}" not in css]
    check("у каждого маркера есть правило в CSS", not lost, ", ".join(lost))
    # Обёртки создаёт layWrap, а размеры и прокрутку им задаёт CSS. Порознь
    # они бессмысленны: без правил обёртка просто съест высоту окна.
    wraps = [c for c in ("cbody", "czoom")
             if f'className="{c}"' not in js.replace(" ", "")
             or f">.card>.cbody" not in css]
    check("обёртки окна заведены и в скрипте, и в CSS", not wraps,
          ", ".join(wraps))

    try:
        import esprima
    except ImportError:
        print("  (esprima не установлен — синтаксис скрипта не проверен; "
              "pip install esprima)")
        return
    # esprima 4 — это ES2017: `??` и `?.` она не знает, а в скрипте они
    # законны. Подменяем их на равные по грамматике, синтаксис от этого
    # не меняется.
    probe = js.replace("??", "||").replace("?.", ".")
    try:
        esprima.parseScript(probe)
        check("скрипт разбирается парсером JS", True)
    except Exception as e:                                  # noqa: BLE001
        check("скрипт разбирается парсером JS", False, str(e))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Сквозная проверка веб-пути.")
    ap.add_argument("--model", type=Path, nargs="+", default=None)
    ap.add_argument("--minutes", type=int, default=45)
    args = ap.parse_args(argv)

    got = pick_match()
    if got is None:
        print("ОШИБКА: не нашлось пригодного матча в data/matches.",
              file=sys.stderr)
        return 2
    match, parsed = got
    art = live.load_models(args.model or live.default_models())
    print(f"Матч {match['match_id']}, модель {art.get('name')}")

    poller = web.Poller(art, live.log_model_name(art), key=None, interval=2.0,
                        from_file=None, log_dir=config.LIVE_LOG_DIR,
                        do_log=False)
    # Сеть отрезаем: их доступность не предмет этой проверки.
    poller.stratz_live = lambda mid: None
    poller._fetch_matchups = lambda key: None
    poller.watch("test")

    offline = F.match_state_frame(match, parsed)
    n = min(args.minutes, int(offline["minute"].max()))
    snaps = []
    for t in range(n + 1):
        payload = payload_at(match, t, parsed)
        try:
            snaps.append(poller.snapshot(payload))
        except Exception as e:                              # noqa: BLE001
            check(f"снимок на минуте {t} не упал", False,
                  f"{type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return 1
    print(f"Собрано снимков: {len(snaps)} (минуты 0-{n})")

    print("\n1. Снимок собирается целиком")
    ok = [s for s in snaps if s.get("ok")]
    check("все снимки посчитаны", len(ok) == len(snaps),
          f"{len(ok)} из {len(snaps)}")
    last = ok[-1]
    for key in ("p", "minute", "builds", "draft", "map", "feed",
                "names", "kills", "towers_lost", "verdict"):
        check(f"есть блок {key}", key in last and last[key] is not None)
    check("вероятность в (0,1)", 0.0 < last["p"] < 1.0, f"{last['p']:.4f}")
    # В снимке не должно быть того, что панель не читает: каждый лишний
    # ключ — это данные, которые считаются на каждый опрос и выбрасываются.
    dead = [k for k in ("contribs", "phist", "mhist", "stratz", "reliability")
            if k in last]
    check("выброшенных полей в снимке нет", not dead, ", ".join(dead))

    print("\n2. Лента событий: события есть и не выдуманы")
    ev = last["feed"]["events"]
    kinds = {e["kind"] for e in ev}
    check("события появились", last["feed"]["n"] > 0,
          f"{last['feed']['n']} за матч")
    check("есть фраги или замесы", bool(kinds & {"kill", "fight", "death"}),
          str(sorted(kinds)))
    check("есть события по зданиям", "tower" in kinds or "rax" in kinds,
          str(sorted(kinds)))
    bad = [e for e in ev if e["kind"] == "kill"
           and (len(e["victims"]) != 1 or len(e["killers"]) != 1)]
    check("у точного фрага ровно один убийца и одна жертва", not bad,
          f"нарушений {len(bad)}")
    mixed = [e for e in ev if e["kind"] == "kill" and e["victims"]
             and e["killers"]
             and e["victims"][0]["team"] == e["killers"][0]["team"]]
    check("убийца и жертва из разных команд", not mixed,
          f"нарушений {len(mixed)}")
    faces = [f for e in ev for f in e["victims"] + e["killers"]]
    named = [f for f in faces if f["name"] and f["name"] != "?"]
    check("у героев в ленте есть имена", len(named) == len(faces),
          f"{len(named)} из {len(faces)}")

    print("\n3. Сборки: то, во что превратилось золото")
    b = last["builds"]
    if b is None:
        print("  (справочник предметов не загружен — блок не рисуется)")
    else:
        check("две команды", len(b["teams"]) == 2)
        for t in b["teams"]:
            check("пятеро в команде", len(t["players"]) == 5,
                  f"{len(t['players'])}")
            check("нетворс есть у всех",
                  all(p["nw"] is not None for p in t["players"]))
            check("шесть слотов у каждого",
                  all(len(p["items"]) == 6 for p in t["players"]))
        # Панель рисует полосу нетворса относительно богатейшего героя
        # МАТЧА. Ноль в знаменателе схлопнул бы её у всех сразу.
        mx = max(p["nw"] or 0 for t in b["teams"] for p in t["players"])
        check("богатейший герой не нулевой", mx > 0, f"{mx:.0f}")

    print("\n4. Разбор драфта и «почему»")
    d = last["draft"]
    check("драфт разобран", bool(d.get("ok")))
    check("по пять героев на сторону",
          len(d.get("radiant") or []) == 5 and len(d.get("dire") or []) == 5)
    why = d.get("why") or []
    check("причины собраны", bool(why), f"{len(why)} строк")
    counted = {w["counted"] for w in why}
    check("у каждой причины сказано, входит ли она в процент",
          counted <= {True, False} and len(why) > 0)

    print("\n5. Вердикт: назван один раз и НИ РАЗУ не сменился")
    v = last["verdict"]
    if v is None:
        print("  (правило не собрано — python -m dwp.verdict --tune)")
    else:
        opened = float(v["open_at"])
        early = [s for s in ok if s["minute"] is not None and s["minute"] < opened]
        check("до срока вердикта нет",
              all(s["verdict"]["side"] is None for s in early),
              f"проверено снимков: {len(early)}")
        # ГЛАВНАЯ ПРОВЕРКА ПАНЕЛИ. Правило смен не допускает по построению
        # (это держит test_verdict), но панель зовёт его заново на каждом
        # опросе, с историей, которая растёт. Здесь проверяется, что при
        # таком вызове сторона всё равно одна и та же от опроса к опросу —
        # и это ровно то, что увидит зритель.
        named = [s["verdict"] for s in ok if s["verdict"]
                 and s["verdict"]["side"] is not None]
        sides = {x["side"] for x in named}
        raw = [("radiant" if s["p"] >= 0.5 else "dire") for s in ok
               if s["minute"] is not None and s["minute"] >= opened]
        raw_flips = sum(1 for a, b in zip(raw, raw[1:]) if a != b)
        check("сторона на панели одна на весь матч", len(sides) <= 1,
              f"встречались: {sorted(sides)} на {len(named)} опросах")
        check("а сырой процент за это время сторону менял", raw_flips > 0,
              f"{raw_flips} смен — вот от чего вердикт и спасает")
        if named:
            mins = {round(x["commit_minute"], 6) for x in named}
            check("минута вердикта тоже не переезжает", len(mins) == 1,
                  f"{sorted(mins)}")
            check("уверенность в момент вердикта не ниже половины",
                  named[-1]["commit_conf"] >= 0.5,
                  f"{named[-1]['commit_conf']:.4f}")
        if v.get("hit"):
            check("рядом с долей попаданий стоит выборка",
                  v["hit"]["n_matches"] > 0, f"{v['hit']['n_matches']} матчей")

    print("\n6. Мёртвые герои")
    # В офлайн-реплее координат у героев нет — их нет и в разобранном
    # матче, — поэтому список heroes здесь пуст, и проверять на нём
    # нечего. Но сам счётчик мёртвых считается по death_count, а он в
    # реплее есть: его и проверяем. Признак «убит» у ИКОНКИ на карте
    # проверяется отдельно, на синтетических ответах, — dwp.test_verdict,
    # раздел 6, там же снимается воскрешение движением.
    hs = last["map"]["heroes"]
    if hs:
        check("у каждого героя есть признак «убит»",
              all("dead" in h for h in hs), f"{len(hs)} героев")
    else:
        print("  (координат в реплее нет — иконок на карте не будет, "
              "проверяем только счётчик)")
    seen = last["map"].get("n_deaths_seen", 0)
    # Смерти НУЛЕВОЙ минуты сюда не входят намеренно: смерть видна только
    # как прирост счётчика, а на первом опросе прироста не с чем сравнивать.
    # Это то же ограничение, что и «подключились в середине матча»: про то,
    # что случилось до первого опроса, мы не знаем ничего и не выдумываем.
    real = sum(1 for p in match["players"] for e in (p.get("kills_log") or [])
               if isinstance(e, dict) and e.get("time") is not None
               and 0 < int(e["time"]) // 60 <= n)
    check("панель увидела смерти на этом матче", seen > 0, f"{seen} смертей")
    check("увидела все, кроме случившихся до первого опроса", seen == real,
          f"{seen} против {real}")
    check("одновременно мёртвых не больше десяти",
          all(s["map"].get("n_dead", 0) <= 10 for s in ok))
    check("без координат «мёртв» не выставляется",
          all(s["map"].get("n_dead", 0) == 0 for s in ok) if not hs else True,
          "иначе герой остался бы мёртвым навсегда — снять признак нечем")

    print("\n7. Число на панели — то же, что даёт консольный путь")
    st, _rep = live.extract_state(payload_at(match, n, parsed))
    tr = live.LiveTracker()
    for t in range(n + 1):
        s2, _ = live.extract_state(payload_at(match, t, parsed))
        gt = s2.get("game_time")
        tr.update(float(gt) / 60.0 if gt is not None else np.nan,
                  s2.get("gold_adv", np.nan), s2.get("roshan_respawn_timer"))
        df2, _w = live.build_row(art, s2, tr)
    p2, _bd = live.predict(art, df2)
    check("вероятности совпадают до 1e-9", abs(p2 - last["p"]) < 1e-9,
          f"панель {last['p']:.9f}, консоль {p2:.9f}")

    print("\n8. Страница панели цела")
    check_page()

    print("\n" + "=" * 70)
    if FAILED:
        print(f"ПРОВАЛЕНО ПРОВЕРОК: {len(FAILED)}")
        for x in FAILED:
            print(f"  - {x}")
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
