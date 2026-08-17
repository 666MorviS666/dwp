"""Проверка пригодности account_id для Elo по игрокам.

Отвечает на один вопрос: хватит ли истории на игроках, чтобы Elo по
пятёрке был точнее Elo по team_id. Ничего не обучает и не пишет,
только считает по data\\matches.

Запуск: python -m dwp.check_accounts
        python -m dwp.check_accounts --limit 500
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _read(path: Path) -> dict | None:
    # Читаем как utf-8, при отказе — cp1251: бриф утверждает cp1251,
    # а features.load_matches читает utf-8. Здесь важнее не потерять файл.
    for enc in ("utf-8", "cp1251"):
        try:
            return json.loads(path.read_text(encoding=enc))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
    return None


def main(argv: list[str] | None = None) -> int:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description="Пригодность account_id для Elo по игрокам.")
    ap.add_argument("--dir", type=Path, default=Path("data") / "matches")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    files = sorted(args.dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"ОШИБКА: в {args.dir} нет json.", file=sys.stderr)
        return 2

    n_match = n_bad_file = 0
    n_player = n_missing = 0
    per_acc: Counter = Counter()
    per_team: Counter = Counter()
    matches_full = 0          # все 10 игроков опознаны
    by_patch: Counter = Counter()
    order = []                # (start_time, [acc...], radiant_team_id, dire_team_id)

    for f in files:
        m = _read(f)
        if m is None:
            n_bad_file += 1
            continue
        if not isinstance(m, dict) or "match_id" not in m:
            # В каталоге может лежать посторонний json (справочник героев —
            # список, а не объект). Молча пропускать нельзя, но и падать не на чем.
            n_bad_file += 1
            continue
        players = m.get("players") or []
        if not players:
            continue
        n_match += 1
        by_patch[m.get("patch")] += 1
        accs = []
        for p in players:
            n_player += 1
            a = p.get("account_id")
            if not a:
                n_missing += 1
            else:
                accs.append(int(a))
                per_acc[int(a)] += 1
        if len(accs) == 10:
            matches_full += 1
        for t in (m.get("radiant_team_id"), m.get("dire_team_id")):
            if t:
                per_team[int(t)] += 1
        order.append((m.get("start_time") or 0, accs))

    print(f"файлов {len(files)}, нечитаемых {n_bad_file}, матчей с игроками {n_match}")
    print(f"патчи: {dict(by_patch.most_common())}")
    print(f"игроков всего {n_player}, без account_id {n_missing} "
          f"({100 * n_missing / max(n_player, 1):.2f}%)")
    print(f"матчей, где опознаны все 10: {matches_full} "
          f"({100 * matches_full / max(n_match, 1):.1f}%)")
    print(f"уникальных account_id {len(per_acc)}, уникальных team_id {len(per_team)}")

    for label, c in (("игрок", per_acc), ("команда", per_team)):
        v = sorted(c.values())
        if not v:
            continue
        med = v[len(v) // 2]
        print(f"  матчей на одного {label}: медиана {med}, "
              f"минимум {v[0]}, максимум {v[-1]}, "
              f"доля с <4 матчей {100 * sum(1 for x in v if x < 4) / len(v):.1f}%")

    # Главное число: у скольких матчей на момент игры уже есть история
    # у всех десяти. Именно это, а не общее покрытие, ограничивает Elo.
    order.sort(key=lambda t: t[0])
    seen: Counter = Counter()
    cold_any = cold_all = 0
    thin_any = 0
    for _, accs in order:
        if not accs:
            cold_all += 1
        else:
            if any(seen[a] == 0 for a in accs):
                cold_any += 1
            if any(seen[a] < 4 for a in accs):
                thin_any += 1
        for a in accs:
            seen[a] += 1
    print("\nхронологически, до матча:")
    print(f"  хотя бы один игрок без истории вообще: {cold_any} "
          f"({100 * cold_any / max(n_match, 1):.1f}%)")
    print(f"  хотя бы один игрок с историей <4 матчей: {thin_any} "
          f"({100 * thin_any / max(n_match, 1):.1f}%)")
    print("Сравните с командными 9.5% и 26.6% из брифа: если эти числа "
          "заметно ниже — Elo по игрокам имеет смысл, если такие же — нет.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
