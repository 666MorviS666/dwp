"""Проверки сборок и урона: справочник, лайв-разбор, восстановление истории.

Что здесь доказывается и чего этот стенд НЕ доказывает.

ДОКАЗЫВАЕТСЯ:
  * лайв-массив `players[].items` разобран верно — длина девять, первые
    шесть слотов инвентарь; нейтральных предметов в нём нет;
  * стоимость сборки, посчитанная по справочнику, сходится с независимым
    счётом самой Valve (`net_worth − gold`) в пределах цены нейтрала;
  * восстановление инвентаря по `purchase_log` не хуже, чем было замерено
    (набор крупных предметов совпадает с концом матча у 77.8% игроков) —
    если правка ухудшит цифру, стенд упадёт;
  * стоимость сборки по построению не убывает со временем;
  * урона в лайв-ответе нет (и стенд скажет, если он появится).

НЕ ДОКАЗЫВАЕТСЯ: что сборки помогают предсказывать. Это меряется отдельно
(`dwp.bench_extras`), и результат замера записан в README.

Запуск:
    python -m dwp.test_builds
    python -m dwp.test_builds --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from dwp import builds as B, config, damage as D, items as I  # noqa: E402

# Пороги — не круглые числа с потолка, а замеренные значения с запасом.
# Рядом стоит то, что получилось на 400 матчах при написании стенда.
MIN_BIG_MATCH = 0.70        # замерено 0.778
MIN_SHARD_AGREE = 0.80      # замерено 0.888
MAX_VALUE_MEDIAN_ERR = 900  # замерено +450 золота на игрока


def _dumps() -> list[Path]:
    return [p for p in (config.DATA_DIR / "live_dump.json",
                        config.DATA_DIR / "live_dump_late.json") if p.exists()]


def _load_json(p: Path) -> dict:
    raw = p.read_bytes()
    enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
    return json.loads(raw.decode(enc))


def check_book(book: I.ItemBook) -> list[str]:
    bad = []
    print("\n--- справочник предметов ---")
    if not book.ok:
        return ["справочник не загружен"]
    print(f"  предметов {len(book.by_id)}, крупных "
          f"{sum(1 for it in book.by_name.values() if it.big)}, "
          f"консьюмаблов {sum(1 for it in book.by_name.values() if it.consumable)}")
    for name in ("black_king_bar", "blink", "tango", "ward_observer"):
        if name not in book.by_name:
            bad.append(f"в справочнике нет {name}")
    bkb = book.by_name.get("black_king_bar")
    if bkb and not bkb.big:
        bad.append("BKB не считается крупным предметом")
    tango = book.by_name.get("tango")
    if tango and not tango.consumable:
        bad.append("танго не помечено консьюмаблом")
    return bad


def check_live(book: I.ItemBook) -> list[str]:
    bad = []
    print("\n--- разбор живого ответа ---")
    dumps = _dumps()
    if not dumps:
        print("  слепков нет — пропускаю")
        return bad
    for p in dumps:
        payload = _load_json(p)
        n_players = 0
        diffs = []
        for t in (payload.get("teams") or []):
            for pl in (t.get("players") or []):
                arr = pl.get("items")
                if arr is None:
                    continue
                n_players += 1
                if len(arr) != I.N_ITEM_SLOTS:
                    bad.append(f"{p.name}: длина items {len(arr)}, "
                               f"ожидалось {I.N_ITEM_SLOTS}")
                b = I.build_of(arr, book)
                if b.kit > b.slots + b.backpack:
                    bad.append(f"{p.name}: предметов сборки {b.kit} больше, "
                               f"чем занятых слотов {b.slots}+{b.backpack}")
                for iid in arr:
                    it = book.get(iid)
                    # Нейтральные предметы в этот массив не попадают —
                    # допущение, на котором держится счёт слотов.
                    if it is not None and it.neutral:
                        bad.append(f"{p.name}: в items нейтральный предмет "
                                   f"{it.name} — счёт слотов надо пересмотреть")
                nw, g = pl.get("net_worth"), pl.get("gold")
                if nw is not None and g is not None:
                    diffs.append(b.value - (float(nw) - float(g)))
        if n_players and n_players != 10:
            bad.append(f"{p.name}: игроков с items {n_players}, ожидалось 10")
        if diffs:
            med = float(np.median(diffs))
            print(f"  {p.name}: игроков {n_players}, стоимость по справочнику "
                  f"минус (net_worth-gold): медиана {med:+.0f}")
            # Расхождение ожидаемо и объяснимо: в net_worth входят нейтрал,
            # рюкзак и консьюмаблы, которых в нашей сумме нет. Но если оно
            # станет ОГРОМНЫМ, значит разъехались единицы измерения.
            if abs(med) > 4000:
                bad.append(f"{p.name}: расхождение со счётом Valve {med:+.0f} — "
                           f"это уже не рюкзак с нейтралом")
    return bad


def check_damage_absent() -> list[str]:
    print("\n--- урон в лайв-ответе ---")
    bad = []
    for p in _dumps():
        got = D.scan_live(_load_json(p))
        print(f"  {p.name}: полей про урон {len(got['found'])}")
        if got["found"]:
            print(f"    ПОЯВИЛИСЬ: {', '.join(got['found'][:10])}")
            print("    Это не поломка стенда: значит, источник расширили и "
                  "раздел README про урон надо перечитать.")
    return bad


def check_reconstruction(book: I.ItemBook, n: int) -> list[str]:
    bad = []
    print(f"\n--- восстановление инвентаря по purchase_log ({n} матчей) ---")
    exact = tot = shard_ok = shard_n = 0
    vdiff = []
    nonmono = 0
    for m in B.iter_matches(config.RAW_MATCHES_DIR, n):
        dur = m.get("duration")
        players = m.get("players") or []
        if not dur or len(players) != 10:
            continue
        for p in players:
            log = p.get("purchase_log")
            if not isinstance(log, list) or not log:
                continue
            inv = I.inventory_at(log, dur, book)
            rec = Counter({k: v for k, v in inv.items()
                           if (it := book.by_name.get(k)) and it.big})
            fin: Counter = Counter()
            for key in ([f"item_{i}" for i in range(6)]
                        + [f"backpack_{i}" for i in range(3)]):
                it = book.get(p.get(key))
                if it is not None and it.big:
                    fin[it.name] += 1
            tot += 1
            if rec == fin:
                exact += 1
            nw, g = p.get("net_worth"), p.get("gold")
            if nw is not None and g is not None:
                _, value, _ = I.build_from_inventory(inv, book)
                vdiff.append(value - (float(nw) - float(g)))
            if p.get("aghanims_shard") is not None:
                shard_n += 1
                bought = any(str(e.get("key", "")).startswith("aghanims_shard")
                             for e in log if isinstance(e, dict))
                if bought == bool(p.get("aghanims_shard")):
                    shard_ok += 1
        # СТОИМОСТЬ сборки игрока не может убывать: покупка либо добавляет
        # предмет, либо съедает части, а части дешевле собранного.
        # Исключение ровно одно и оно известно — съедаемые предметы
        # (CONSUMED_ON_USE): скипетр-2 поглощает скипетр и исчезает сам.
        #
        # Считать этот инвариант по КОРУ команды нельзя, и это выяснилось
        # стендом: «кор» — это игрок с наибольшей стоимостью предметов, а он
        # меняется по ходу матча, так что его ряд обязан прыгать. Число
        # крупных предметов не монотонно и у отдельного игрока: Abyssal
        # Blade съедает Basher и Sange, оба крупные, — было два, стало один.
        T = int(dur // 60)
        for p in players:
            log = p.get("purchase_log")
            if not isinstance(log, list) or not log:
                continue
            _, val, _, _ = B._player_series(p, book, T)
            drops = np.flatnonzero(np.diff(val) < -1e-9)
            for k in drops:
                mn = int(k) + 1
                eaten = any(
                    str(e.get("key", "")) in I.CONSUMED_ON_USE
                    and mn * 60 <= float(e.get("time", -1)) <= mn * 60 + 59
                    for e in log if isinstance(e, dict) and e.get("time") is not None)
                if not eaten:
                    nonmono += 1
    if not tot:
        return ["ни одного матча с журналом покупок"]
    rate = exact / tot
    med = float(np.median(vdiff)) if vdiff else float("nan")
    print(f"  игроков сверено: {tot}")
    print(f"  набор КРУПНЫХ предметов совпал с концом матча: {rate * 100:.1f}% "
          f"(порог {MIN_BIG_MATCH * 100:.0f}%, при написании было 77.8%)")
    print(f"  стоимость минус (net_worth-gold): медиана {med:+.0f} "
          f"(порог |{MAX_VALUE_MEDIAN_ERR}|, при написании было +450)")
    if shard_n:
        sr = shard_ok / shard_n
        print(f"  флаг aghanims_shard согласуется с журналом: {sr * 100:.1f}% "
              f"(порог {MIN_SHARD_AGREE * 100:.0f}%, при написании 88.8%)")
        if sr < MIN_SHARD_AGREE:
            bad.append(f"шард согласуется лишь в {sr * 100:.1f}% — список "
                       f"CONSUMED_ON_USE устарел?")
    if rate < MIN_BIG_MATCH:
        bad.append(f"восстановление ухудшилось: {rate * 100:.1f}% < "
                   f"{MIN_BIG_MATCH * 100:.0f}%")
    if not np.isnan(med) and abs(med) > MAX_VALUE_MEDIAN_ERR:
        bad.append(f"стоимость разъехалась со счётом Valve: медиана {med:+.0f}")
    print(f"  необъяснённых падений стоимости сборки: {nonmono} (должно быть 0)")
    if nonmono:
        bad.append(f"{nonmono} раз стоимость сборки игрока упала БЕЗ покупки "
                   f"съедаемого предмета — покупка не может отнимать золото "
                   f"из инвентаря")
    return bad


def check_table() -> list[str]:
    print("\n--- таблица шансов ---")
    t = B.load_table()
    if t is None:
        print("  таблицы нет (python -m dwp.builds --table) — пропускаю")
        return []
    bad = []
    m = t.get("model")
    print(f"  матчей в таблице: {t['n_matches']}, клеток {len(t['cells'])}, "
          f"непрерывная модель: {'есть' if m else 'нет'}")
    # ВНЕ ОБЛАСТИ ОПРЕДЕЛЕНИЯ ответа быть не должно — ни клеточного, ни
    # гладкого. Логистическая регрессия сама по себе продолжится куда
    # угодно, и на третьей минуте выдаст осмысленно выглядящее число там,
    # где отставания ещё не существует как явления.
    lo_m = (m or {}).get("minute_min", 10.0)
    lo_d = (m or {}).get("deficit_min", 2000.0)
    if B.lookup(t, lo_m - 5, 12000, 5) is not None:
        bad.append(f"на {lo_m - 5:.0f}-й минуте таблица отвечает, а наблюдений "
                   f"раньше {lo_m:.0f}-й в ней нет")
    if B.lookup(t, 32.0, lo_d / 2, 5) is not None:
        bad.append(f"при отставании {lo_d / 2:.0f} таблица отвечает, а "
                   f"наблюдений меньше {lo_d:.0f} в ней нет")
    got = B.lookup(t, 32.0, 12000, 6, 6)
    if got is None:
        bad.append("на 32-й минуте при отставании 12к таблица молчит")
    else:
        print(f"  пример: 32-я минута, отставание 12к, кор 6 крупных -> "
              f"по когорте {got['rate'] * 100:.1f}% ± "
              f"{(got['se'] or 0) * 100:.1f} на n={got['n']}"
              + (f", гладкая {got['smooth'] * 100:.1f}%"
                 if got.get("smooth") is not None else ""))
        for k in ("rate", "smooth"):
            v = got.get(k)
            if v is not None and not (0.0 <= v <= 1.0):
                bad.append(f"{k} вне [0,1]")
    # Гладкая оценка обязана работать там, где клетка пуста, — ради этого
    # она и заводилась: раньше панель молчала на 15-й минуте.
    mid = B.lookup(t, 15.0, 3000, 4, 4)
    if mid is None or mid.get("smooth") is None:
        bad.append("на 15-й минуте при отставании 3к нет даже гладкой оценки")
    else:
        print(f"  на 15-й минуте, отставание 3к, кор 4 -> гладкая "
              f"{mid['smooth'] * 100:.1f}%")
    if m and m["log_loss"] >= m["log_loss_base"]:
        bad.append(f"непрерывная модель не лучше константы: "
                   f"{m['log_loss']:.4f} против {m['log_loss_base']:.4f}")
    elif m:
        print(f"  модель против константы: {m['log_loss']:.4f} против "
              f"{m['log_loss_base']:.4f} на {m['n_test_matches']} отложенных")
    return bad


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Проверки сборок и урона.")
    ap.add_argument("--n", type=int, default=60,
                    help="сколько матчей брать для проверки восстановления")
    args = ap.parse_args(argv)

    book = I.load()
    bad: list[str] = []
    bad += check_book(book)
    bad += check_live(book)
    bad += check_damage_absent()
    bad += check_reconstruction(book, args.n)
    bad += check_table()

    print()
    print("=" * 74)
    if bad:
        print("СТЕНД НЕ ПРОЙДЕН:")
        for b in bad:
            print(f"  * {b}")
        return 1
    print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
