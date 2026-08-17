"""Разбор завершённого матча: кривая вероятности, переломы, разложение.

Elo берётся ПРЕД-матчевый из сохранённой таблицы match_id -> разница.
Финальный рейтинг команды для разбора прошлого матча — утечка из
будущего (в него уже вошёл исход этого самого матча) и вдобавок выход за
диапазон, на котором училась модель. Если матча нет в таблице (новый),
это печатается явно.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

from . import config, explain as E, features as F
from .train import logit


def load_artefact(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Нет обученной модели {path}. Запустите `python -m dwp.train`.")
    with path.open("rb") as fh:
        return pickle.load(fh)


def _elo_for_match(art: dict, match: dict) -> tuple[float, str]:
    mid = int(match["match_id"])
    pre = art["elo_pre"]
    if mid in pre:
        return float(pre[mid]), "пред-матчевый из таблицы"
    if isinstance(pre, dict) and str(mid) in pre:
        return float(pre[str(mid)]), "пред-матчевый из таблицы"
    fin = art["elo_final"]
    r, d = match.get("radiant_team_id"), match.get("dire_team_id")
    if r and d and int(r) in fin and int(d) in fin:
        return float(fin[int(r)] - fin[int(d)]), "ФИНАЛЬНЫЙ рейтинг (матч не в таблице)"
    return 0.0, "неизвестен, подставлен 0 — предсказание хуже, чем могло бы быть"


def analyze_match(art: dict, match: dict, window: int, verbose: bool = True) -> dict:
    mid = int(match["match_id"])
    rad, dire = F.match_sides(match)
    elo_diff, elo_src = _elo_for_match(art, match)

    Xd, _, _ = F.draft_matrix([match], art["id2idx"], {mid: elo_diff})
    p_draft = float(art["draft_model"].predict_proba(Xd)[0, 1])

    parsed = F.parse_objectives(match)
    df = F.match_state_frame(match, parsed)
    df["draft_logit"] = logit(np.array([p_draft]))[0]
    feats = art["state_features"]
    missing = [f for f in feats if f not in df.columns]
    if missing:
        raise ValueError(f"в срезах нет признаков {missing} — модель и парсер рассинхронизированы")

    p_raw = art["booster"].predict(df[feats], num_iteration=art["booster"].best_iteration)
    eps = max(art.get("calib_eps", 0.0), 1e-6)
    from .train import apply_calibrator
    p = np.clip(apply_calibrator(art["iso"], p_raw), eps, 1 - eps)
    contrib, base = E.state_contributions(art, df)
    minutes = df["minute"].to_numpy()

    if not verbose:
        return {"match_id": mid, "p": p, "minutes": minutes}

    in_test = mid in set(art.get("test_match_ids", []))
    print("=" * 78)
    print(f"Матч {mid}   исход: {'Radiant' if match['radiant_win'] else 'Dire'}"
          f"   длительность {match['duration'] // 60}:{match['duration'] % 60:02d}")
    print(f"  матч в тестовой выборке: {'да' if in_test else 'НЕТ — модель его видела, '
                                                            'числа оптимистичны'}")
    print(f"  формат objectives: {parsed.fmt}; вышки разобраны: "
          f"{'да' if parsed.tower_ok else 'НЕТ (признаки = NaN)'}; "
          f"сверка с tower_status: {parsed.tower_consistent}")
    if parsed.unparsed:
        print(f"  неразобранные события: "
              f"{', '.join(f'{k}×{v}' for k, v in parsed.unparsed.most_common(4))}")
    print(f"  Elo diff = {elo_diff:+.1f} ({elo_src})")
    print(f"  драфт-модель до старта: Radiant {p_draft * 100:.1f}%")

    tps = E.turning_points(minutes, p, contrib, feats, window=window)
    marks = {tp["minute_to"]: chr(ord("1") + i) for i, tp in enumerate(tps[:9])}

    print("\n--- Вероятность победы Radiant "
          "(заливка вверх от 50% = ведёт Radiant, вниз = Dire) ---")
    print(E.sparkline(minutes, p, marks=marks))

    print(f"\n--- Переломные моменты "
          f"(изменение вероятности за {window} мин, атрибуция = разность вкладов) ---")
    if not tps:
        print("  не найдено: вероятность менялась плавно, ни одного скачка "
              "больше 6 п.п. за окно")
    for i, tp in enumerate(tps):
        drv = ", ".join(f"{n} {v:+.2f}" for n, v in tp["drivers"])
        tag = chr(ord("1") + i) if i < 9 else " "
        arrow = "\u2197" if tp["swing"] > 0 else "\u2198"
        try:
            arrow.encode(getattr(sys.stdout, "encoding", None) or "ascii")
        except (UnicodeEncodeError, LookupError):
            arrow = "^" if tp["swing"] > 0 else "v"
        print(f"  [{tag}] мин {tp['minute_from']:>3}-{tp['minute_to']:<3} {arrow} "
              f"{tp['p_from'] * 100:5.1f}% -> {tp['p_to'] * 100:5.1f}%  "
              f"({tp['swing'] * 100:+5.1f} п.п.)\n      {drv}")

    last = int(minutes[-1])
    bd, base_val, total = E.final_breakdown(art, df, last)
    print(f"\n--- Разложение оценки на минуте {last} (сырые логиты, ДО калибровки) ---")
    print(f"  базовое значение модели: {base_val:+.3f}")
    scale = max(0.2, float(bd["вклад_логит"].abs().max()))
    print(f"  {'признак':<21}{'значение':>13}  {'вклад':>7}  "
          f"{'Dire':<12}{'Radiant':>12}")
    for _, r in bd.iterrows():
        val = r["значение"]
        if isinstance(val, float) and np.isnan(val):
            vs = "NaN"
        else:
            vs = f"{val:+.0f}" if abs(float(val)) >= 100 else f"{float(val):+.2f}"
        print(f"  {r['признак']:<21}{vs:>13}  {r['вклад_логит']:>+7.3f}  "
              f"{E.bar(r['вклад_логит'], 24, -scale, scale)}")
    print(f"  сумма = {total:+.3f}  ->  сырая p = {1 / (1 + np.exp(-total)) * 100:.1f}%"
          f"  ->  после калибровки {p[-1] * 100:.1f}%")
    print("  (разница между двумя последними числами — это и есть работа изотоники;"
          "\n   разложить её по признакам нельзя, она нелинейна)")
    print("  ВНИМАНИЕ: вклады УСЛОВНЫЕ, при фиксированных остальных признаках, а не"
          "\n  маргинальные. Пример: 'Radiant потерял 9 вышек' может давать ПЛЮС к его"
          "\n  шансам — при равном золоте это значит, что он отыгрался, а не что терять"
          "\n  вышки полезно. Для комментатора такую строку нужно переформулировать.")
    print("=" * 78)
    return {"match_id": mid, "p": p, "minutes": minutes, "turning_points": tps}



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
    ap = argparse.ArgumentParser(description="Разбор завершённого матча.")
    ap.add_argument("--model", type=Path, default=config.MODELS_DIR / "model_synthetic.pkl")
    ap.add_argument("--source", choices=["synthetic", "real"], default=None,
                    help="по умолчанию — тот источник, на котором обучалась модель")
    ap.add_argument("--random", type=int, default=0, help="сколько случайных матчей разобрать")
    ap.add_argument("--match-id", type=int, default=None)
    ap.add_argument("--window", type=int, default=3, help="окно детектора переломов, минут")
    ap.add_argument("--any", action="store_true",
                    help="брать матчи из всей выборки, а не только из тестовой")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    try:
        art = load_artefact(args.model)
    except FileNotFoundError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2

    src = args.source or ("real" if "matches" in art.get("source_dir", "")
                          and "synthetic" not in art.get("source_dir", "") else "synthetic")
    directory = config.SYNTH_MATCHES_DIR if src == "synthetic" else config.RAW_MATCHES_DIR

    if args.match_id is not None:
        path = directory / f"{args.match_id}.json"
        if not path.exists():
            print(f"ОШИБКА: нет файла {path}. Матч не скачан.", file=sys.stderr)
            return 2
        analyze_match(art, json.loads(path.read_text(encoding="utf-8")), args.window)
        return 0

    n = args.random or 1
    matches = F.usable_matches(F.load_matches(directory), verbose=False)
    if not matches:
        print(f"ОШИБКА: в {directory} нет матчей.", file=sys.stderr)
        return 2
    # По умолчанию берём только тестовые матчи: разбирать матч, на котором
    # модель училась, и показывать это как демонстрацию качества — обман.
    test_ids = set(art.get("test_match_ids", []))
    pool = [m for m in matches if int(m["match_id"]) in test_ids] if not args.any else matches
    if not pool:
        print("ВНИМАНИЕ: тестовых матчей в каталоге не найдено, беру любые "
              "(числа будут оптимистичны).")
        pool = matches
    rng = np.random.default_rng(args.seed)
    for i in rng.choice(len(pool), size=min(n, len(pool)), replace=False):
        analyze_match(art, pool[int(i)], args.window)
    return 0


if __name__ == "__main__":
    sys.exit(main())
