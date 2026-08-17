"""Выгрузка составов: одна строка на матч, десять account_id по сторонам.

Нужна ровно для одного: посчитать Elo по игрокам и сравнить его с Elo
по team_id на одних и тех же матчах. Ни одного из существующих экспортов
для этого не хватает — в draft есть team_id, но нет игроков.

Ничего не обучает, ничего не пишет кроме одного csv.

Запуск: python -m dwp.export_rosters
        python -m dwp.export_rosters --out data\\export_rosters.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


def _read(path: Path) -> dict | None:
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

    ap = argparse.ArgumentParser(description="Выгрузка составов по матчам.")
    ap.add_argument("--dir", type=Path, default=Path("data") / "matches")
    ap.add_argument("--out", type=Path, default=Path("data") / "export_rosters.csv")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    files = sorted(args.dir.glob("*.json"))
    if args.limit:
        files = files[: args.limit]
    if not files:
        print(f"ОШИБКА: в {args.dir} нет json.", file=sys.stderr)
        return 2

    cols = (["match_id", "start_time", "patch", "y",
             "radiant_team_id", "dire_team_id"]
            + [f"r{i}" for i in range(1, 6)] + [f"d{i}" for i in range(1, 6)])

    rows, skipped = [], {"не матч": 0, "нет исхода": 0, "не 5+5": 0}
    for f in files:
        m = _read(f)
        if not isinstance(m, dict) or "match_id" not in m:
            skipped["не матч"] += 1
            continue
        if m.get("radiant_win") is None:
            skipped["нет исхода"] += 1
            continue
        rad, dire = [], []
        for p in m.get("players") or []:
            # Сторона берётся из player_slot, а не из isRadiant: слот есть
            # всегда, isRadiant — производное поле OpenDota и может отсутствовать.
            slot = p.get("player_slot")
            if slot is None:
                continue
            a = p.get("account_id")
            (rad if slot < 128 else dire).append(int(a) if a else 0)
        if len(rad) != 5 or len(dire) != 5:
            skipped["не 5+5"] += 1
            continue
        rows.append([int(m["match_id"]), m.get("start_time"), m.get("patch"),
                     int(bool(m["radiant_win"])),
                     m.get("radiant_team_id") or 0, m.get("dire_team_id") or 0]
                    + rad + dire)

    rows.sort(key=lambda r: r[1] or 0)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        w.writerows(rows)

    print(f"rosters: {len(rows)} матчей -> {args.out}")
    bad = {k: v for k, v in skipped.items() if v}
    if bad:
        print(f"  пропущено: {bad}")
    zero = sum(1 for r in rows for a in r[6:] if a == 0)
    print(f"  слотов с нулевым account_id: {zero} из {len(rows) * 10}")
    print(f"  размер файла: {args.out.stat().st_size / 1e6:.2f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
