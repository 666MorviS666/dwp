"""Слепой холдаут: матчи, которых обучение не видит никогда.

ЗАЧЕМ ЭТО НУЖНО СВЕРХ ОБЫЧНОГО ТЕСТА. В `train.py` уже есть тестовая
выборка, и она честная — ровно один раз. Дальше начинается настоящая
работа: поменяли признак, посмотрели на тест, поменяли калибратор,
посмотрели на тест, выбрали `--exact-features` по тесту, выбрали модель для
лайва по тесту. К десятому такому кругу тест перестаёт быть независимым:
через глаза разработчика исход каждого матча уже просочился в решения. Это
не гипотетическая беда — весь раздел «Какую модель брать» построен на
сравнении по тесту, и именно так тест и расходуется.

Лечится это единственным способом: **часть матчей не показывать вообще
никому**, включая себя. Здесь такая часть выделяется — и не сидом, а
хэшем.

ПОЧЕМУ ХЭШ, А НЕ SEED. `GroupShuffleSplit(random_state=7)` даёт устойчивое
разбиение только при неизменном наборе матчей. Добрали двести новых — и
разбиение переехало целиком: матч, лежавший в холдауте, оказался в
обучении, а «слепая» оценка стала оценкой на виденных матчах, причём молча.
Хэш от `match_id` этим не страдает: попадание матча в холдаут зависит
только от него самого, и добор данных ничего не двигает.

    доля холдаута       15% матчей
    правило             sha1(str(match_id)) mod 1000 < 150

Как это связано с `dwp.blindtest`: холдаут — это ЧТО прячем, blindtest —
КАК смотрим (запечатать оценки, потом вскрыть). Каждое вскрытие
дописывается в реестр `data/blind/registry.csv`, чтобы было видно, сколько
раз в холдаут уже заглядывали: третий взгляд подряд после трёх правок — это
уже не слепая оценка, и пусть это будет написано, а не забыто.

Запуск:
    python -m dwp.holdout                 # сколько матчей спрятано и какие
    python -m dwp.holdout --check <id>    # этот матч слепой или нет
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
import time

from . import config

# Доля матчей, спрятанных от обучения. 15% от 7734 — это около 1160 матчей,
# то есть достаточно, чтобы log loss на них имел смысл (интервал по матчам
# порядка 0.01), и не настолько много, чтобы обучению стало не на чем расти.
HOLDOUT_PERMILLE = 150
REGISTRY = config.DATA_DIR / "blind" / "registry.csv"
REGISTRY_COLUMNS = ["ts", "when", "model", "source", "n_matches", "n_rows",
                    "sha256", "log_loss", "brier", "ece", "note"]


def is_holdout(match_id) -> bool:
    """Спрятан ли матч от обучения. Зависит только от самого match_id."""
    try:
        mid = int(match_id)
    except (TypeError, ValueError):
        return False
    h = hashlib.sha1(str(mid).encode("ascii")).hexdigest()
    return int(h[:8], 16) % 1000 < HOLDOUT_PERMILLE


def split(matches: list[dict]) -> tuple[list[dict], list[dict]]:
    """(что можно обучать, что спрятано)."""
    train, hold = [], []
    for m in matches:
        (hold if is_holdout(m.get("match_id")) else train).append(m)
    return train, hold


def note_reveal(model: str, source: str, n_matches: int, n_rows: int,
                sha: str, metrics: dict, note: str = "") -> int:
    """Записать факт вскрытия. Возвращает, каким по счёту он оказался.

    Смысл не в бухгалтерии, а в том, что число взглядов — это ЕДИНСТВЕННОЕ,
    что отличает слепую оценку от подогнанной, и держать его в голове
    нельзя.
    """
    REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    new = not REGISTRY.exists()
    row = {
        "ts": f"{time.time():.0f}",
        "when": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model": model, "source": source,
        "n_matches": n_matches, "n_rows": n_rows, "sha256": sha[:16],
        "log_loss": f"{metrics.get('log_loss', float('nan')):.4f}",
        "brier": f"{metrics.get('brier', float('nan')):.4f}",
        "ece": f"{metrics.get('ece', float('nan')):.4f}",
        "note": note,
    }
    with REGISTRY.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=REGISTRY_COLUMNS)
        if new:
            w.writeheader()
        w.writerow(row)
    return count_reveals(model)


def count_reveals(model: str | None = None) -> int:
    if not REGISTRY.exists():
        return 0
    try:
        with REGISTRY.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))
    except OSError:
        return 0
    if model is None:
        return len(rows)
    return sum(1 for r in rows if r.get("model") == model)


def registry_rows() -> list[dict]:
    if not REGISTRY.exists():
        return []
    try:
        with REGISTRY.open(encoding="utf-8", newline="") as fh:
            return list(csv.DictReader(fh))
    except OSError:
        return []


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(description="Слепой холдаут: что спрятано.")
    ap.add_argument("--check", type=int, nargs="*",
                    help="проверить конкретные match_id")
    args = ap.parse_args(argv)

    if args.check:
        for mid in args.check:
            print(f"  {mid}  {'СЛЕПОЙ (обучение его не видит)' if is_holdout(mid) else 'обучаемый'}")
        return 0

    files = sorted(config.RAW_MATCHES_DIR.glob("*.json"))
    ids = [int(f.stem) for f in files if f.stem.isdigit()]
    hold = [i for i in ids if is_holdout(i)]
    print(f"Матчей в выгрузке: {len(ids)}")
    print(f"Спрятано в холдаут: {len(hold)} ({len(hold) / max(1, len(ids)) * 100:.1f}%, "
          f"правило: sha1(match_id) mod 1000 < {HOLDOUT_PERMILLE})")
    print(f"Обучению доступно:  {len(ids) - len(hold)}")
    print(f"\nПервые пять спрятанных: {', '.join(str(i) for i in hold[:5])}")
    rows = registry_rows()
    print(f"\nВскрытий холдаута записано: {len(rows)}")
    if rows:
        print(f"  {'когда':<21}{'модель':<22}{'матчей':>8}{'log_loss':>10}{'ECE':>8}")
        for r in rows[-10:]:
            print(f"  {r['when']:<21}{r['model'][:21]:<22}{r['n_matches']:>8}"
                  f"{r['log_loss']:>10}{r['ece']:>8}")
        print("\n  Каждая строка — это взгляд в слепую выборку. Чем их больше,\n"
              "  тем меньше в ней слепоты: выбирая правку по этим числам, вы\n"
              "  подгоняете модель под холдаут ровно так же, как раньше под тест.")
    else:
        print("  Ни разу. Так и держать: смотреть туда стоит на крупных\n"
              "  развилках, а не после каждой правки.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
