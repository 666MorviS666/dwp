"""Жёсткая проверка допущения OLD_ROSHAN_KILL_TEAM_MEANS.

У CHAT_MESSAGE_ROSHAN_KILL нет player_slot, поэтому сверить трактовку
поля `team` напрямую нельзя — так считалось раньше, и допущение
«team = команда убийцы» держалось на косвенном перевесе победителя
по числу Рошанов.

Но у CHAT_MESSAGE_AEGIS player_slot ЕСТЬ. Аегис падает с Рошана и
подбирается в ту же секунду, почти всегда командой, которая его убила.
Значит сторону убийцы можно восстановить из player_slot аегиса и сверить
с `team` соседнего события. Это уже жёсткая проверка, а не косвенная.

Что означает результат:
  * доля совпадений около единицы -> team = убийца, допущение верно;
  * доля около нуля -> team = потерпевшая сторона, знак признаков по
    Рошану перевёрнут;
  * доля около половины -> поле `team` вообще не про сторону, и
    признаки roshan_radiant/roshan_dire сейчас шум.

Несовпадения не обязательно ошибка парсера: аегис можно украсть или
денайнуть. Их доля и есть оценка того, как часто это бывает.

Запуск:
    python -m dwp.check_aegis
    python -m dwp.check_aegis --window 30
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter

import numpy as np

from . import config, features as F
from .train import _safe_stdout

ROSHAN = "CHAT_MESSAGE_ROSHAN_KILL"
AEGIS = "CHAT_MESSAGE_AEGIS"


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Сверка team у ROSHAN_KILL со стороной подобравшего аегис.")
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--window", type=int, default=90,
                    help="сколько секунд между убийством и подбором считать одним событием")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)

    directory = (config.RAW_MATCHES_DIR if args.source == "real"
                 else config.SYNTH_MATCHES_DIR)
    matches = F.load_matches(directory, limit=args.limit)
    if not matches:
        print(f"ОШИБКА: в {directory} нет матчей.", file=sys.stderr)
        return 2

    n_rosh = n_aegis = 0
    aegis_with_slot = 0
    paired = 0
    agree = 0
    dt_all: list[int] = []
    rosh_team_values: Counter = Counter()
    unpaired_reasons: Counter = Counter()

    for m in matches:
        obj = m.get("objectives") or []
        rosh = [o for o in obj if o.get("type") == ROSHAN]
        aeg = [o for o in obj if o.get("type") == AEGIS]
        n_rosh += len(rosh)
        n_aegis += len(aeg)
        aegis_with_slot += sum(1 for o in aeg if o.get("player_slot") is not None)
        for r in rosh:
            rt = r.get("team")
            rosh_team_values[rt] += 1
            rtime = r.get("time")
            if rtime is None:
                unpaired_reasons["у рошана нет time"] += 1
                continue
            near = [a for a in aeg
                    if a.get("time") is not None
                    and a.get("player_slot") is not None
                    and abs(int(a["time"]) - int(rtime)) <= args.window]
            if not near:
                unpaired_reasons["нет аегиса в окне"] += 1
                continue
            a = min(near, key=lambda a: abs(int(a["time"]) - int(rtime)))
            dt_all.append(int(a["time"]) - int(rtime))
            picked_radiant = int(a["player_slot"]) < 128
            side = F.team_is_radiant(rt)
            if side is None:
                unpaired_reasons["team не распознан"] += 1
                continue
            paired += 1
            agree += int(side == picked_radiant)

    print("=" * 74)
    print(f"матчей {len(matches)} | событий ROSHAN_KILL {n_rosh} | AEGIS {n_aegis} "
          f"(из них с player_slot {aegis_with_slot})")
    print(f"значения поля team у ROSHAN_KILL: {dict(rosh_team_values)}")
    if dt_all:
        d = np.array(dt_all)
        print(f"задержка подбора аегиса, секунд: медиана {np.median(d):.0f}, "
              f"90-й процентиль {np.percentile(d, 90):.0f}, максимум {d.max()}")
    if not paired:
        print("Сопоставить не удалось ни одного события.")
        print(f"причины: {dict(unpaired_reasons)}")
        print("=" * 74)
        return 1
    frac = agree / paired
    print(f"\nсопоставлено пар {paired} из {n_rosh} ROSHAN_KILL"
          f"   (не сошлось: {dict(unpaired_reasons)})")
    print(f"team трактуется как УБИЙЦА и совпадает: {agree} из {paired} = {frac:.4f}")
    # Интервал Вилсона, а не нормальный: при доле ровно 1 или 0 нормальный
    # схлопывается в точку и обещает уверенность, которой нет.
    z = 1.96
    den = 1 + z * z / paired
    centre = (frac + z * z / (2 * paired)) / den
    half = z * np.sqrt(frac * (1 - frac) / paired
                       + z * z / (4 * paired * paired)) / den
    print(f"95% интервал доли (Вилсон): [{centre - half:.4f}, "
          f"{min(1.0, centre + half):.4f}]")
    if frac > 0.9:
        print("ВЫВОД: OLD_ROSHAN_KILL_TEAM_MEANS = 'killer' подтверждён жёстко.")
    elif frac < 0.1:
        print("ВЫВОД: трактовка ПЕРЕВЁРНУТА, в config должно стоять 'victim'.")
    else:
        print("ВЫВОД: поле team не определяет сторону однозначно. Признаки по "
              "Рошану в текущем виде частично шум.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
