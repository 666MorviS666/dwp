"""Выгрузка данных проекта в плоские файлы — чтобы их можно было
разбирать вне этой машины, не таская гигабайты сырых JSON.

Ничего не считает и ничего не решает: только достаёт то, что уже умеют
features.py и train.py, и кладёт в csv.gz. Все режимы читают только
данные, ни один не пишет в data/matches и не трогает модели.

Режимы:
  draft    одна строка на матч: вектор героев, elo_pre, исход, патч, время
  frames   поминутные срезы всех матчей — вход стейт-модели целиком
  meta     одна строка на матч: длительность, патч, лига, флаги парсера
  events   перепись типов событий в objectives и набора ключей у каждого
  testpred предсказания заданной модели на её тестовой выборке
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, features as F
from .train import _safe_stdout


def _load(source: str, limit: int | None, usable_only: bool = True) -> list[dict]:
    directory = (config.RAW_MATCHES_DIR if source == "real"
                 else config.SYNTH_MATCHES_DIR)
    ms = F.load_matches(directory, limit=limit)
    if not ms:
        raise SystemExit(f"ОШИБКА: в {directory} нет матчей.")
    return F.usable_matches(ms, verbose=True) if usable_only else ms


def _meta_row(m: dict) -> dict:
    p = F.parse_objectives(m)
    return {
        "match_id": int(m["match_id"]),
        "radiant_win": int(bool(m.get("radiant_win"))),
        "duration": m.get("duration"),
        "start_time": m.get("start_time"),
        "patch": m.get("patch"),
        "leagueid": m.get("leagueid"),
        "lobby_type": m.get("lobby_type"),
        "game_mode": m.get("game_mode"),
        "radiant_team_id": m.get("radiant_team_id"),
        "dire_team_id": m.get("dire_team_id"),
        # Урезанный матч (slim_stream) приносит уже посчитанные длины: сами
        # списки в нём не хранятся, чтобы не держать в памяти всю выгрузку.
        "n_teamfights": m.get("teamfights_n", len(m.get("teamfights") or [])),
        "n_picks_bans": m.get("picks_bans_n", len(m.get("picks_bans") or [])),
        "obj_fmt": p.fmt,
        "tower_ok": int(p.tower_ok),
        "rax_ok": int(p.rax_ok),
        "roshan_ok": int(p.roshan_ok),
        "tower_consistent": p.tower_consistent,
        "rax_consistent": p.rax_consistent,
        "n_unparsed": int(sum(p.unparsed.values())),
    }


def slim_stream(source: str, limit: int | None) -> list[dict]:
    """Пригодные матчи в урезанном виде: только то, что нужно драфту и мете.

    Читает по одному и сразу выбрасывает всё тяжёлое (поминутные ряды,
    журналы покупок, чат). Иначе `data/matches` — два гигабайта JSON, а
    разобранные словари занимают в разы больше и в память не влезают.
    Отбор — тот же `usable_matches`, что и раньше, и делается ДО урезания:
    ему нужен `radiant_gold_adv`.
    """
    from .builds import iter_matches
    directory = (config.RAW_MATCHES_DIR if source == "real"
                 else config.SYNTH_MATCHES_DIR)
    keep = ("match_id", "start_time", "radiant_win", "radiant_team_id",
            "dire_team_id", "patch", "leagueid", "lobby_type", "game_mode",
            "duration", "tower_status_radiant", "tower_status_dire",
            "barracks_status_radiant", "barracks_status_dire")
    out, n = [], 0
    for m in iter_matches(directory, limit):
        n += 1
        if not F.usable_matches([m], verbose=False):
            continue
        slim = {k: m.get(k) for k in keep}
        slim["players"] = [{"hero_id": p.get("hero_id"),
                            "player_slot": p.get("player_slot")}
                           for p in (m.get("players") or [])]
        slim["objectives"] = m.get("objectives")
        slim["teamfights_n"] = len(m.get("teamfights") or [])
        slim["picks_bans_n"] = len(m.get("picks_bans") or [])
        out.append(slim)
        if n % 1000 == 0:
            print(f"  прочитано {n}, пригодных {len(out)}", flush=True)
    print(f"[export] прочитано {n} матчей, пригодных {len(out)}")
    return out


def do_draft(matches: list[dict], out: Path) -> None:
    hero_ids, id2idx, names = F.load_heroes()
    elo_pre, _ = F.build_elo(matches)
    X, y, mids = F.draft_matrix(matches, id2idx, elo_pre)
    cols = [f"h{h}" for h in hero_ids] + ["elo_div400"]
    df = pd.DataFrame(X, columns=cols)
    # Герои — это ровно -1/0/+1, int8 сжимает файл в разы против float.
    df[cols[:-1]] = df[cols[:-1]].astype(np.int8)
    by_id = {int(m["match_id"]): m for m in matches}
    df.insert(0, "match_id", mids)
    df.insert(1, "y", y.astype(np.int8))
    df.insert(2, "start_time", [by_id[i].get("start_time") for i in mids])
    df.insert(3, "patch", [by_id[i].get("patch") for i in mids])
    df.insert(4, "radiant_team_id", [by_id[i].get("radiant_team_id") for i in mids])
    df.insert(5, "dire_team_id", [by_id[i].get("dire_team_id") for i in mids])
    df.to_csv(out, index=False, compression="gzip")
    print(f"draft: {len(df)} матчей, {len(cols)} признаков -> {out}")
    print(f"  имена героев по столбцам hNNN: "
          f"{', '.join(f'h{h}={names.get(h, h)}' for h in hero_ids[:4])}, ...")


def do_frames(matches, out: Path, round_to: int = 4) -> None:
    frames, skipped = [], 0
    for m in matches:
        try:
            frames.append(F.match_state_frame(m))
        except ValueError:
            skipped += 1
    df = pd.concat(frames, ignore_index=True)
    # Округление до 4 знаков: доли нетворса имеют смысл до третьего,
    # а несжатый float64 раздувает файл вдвое без всякой пользы.
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(round_to)
    df.to_csv(out, index=False, compression="gzip")
    print(f"frames: {len(df)} строк из {df['match_id'].nunique()} матчей, "
          f"{len(df.columns)} столбцов -> {out}"
          + (f"   (пропущено без поминутных данных: {skipped})" if skipped else ""))
    print(f"  столбцы: {', '.join(df.columns)}")


def do_extras(out: Path, limit: int | None) -> None:
    """Поминутные признаки, которых нет в основном кадре: сборки и урон.

    Отдельным файлом, а не столбцами в `frames`, по двум причинам. Первая:
    их источники (`purchase_log`, `teamfights`) есть не у всех матчей, и
    подмешивать NaN в основную выгрузку незачем. Вторая: они пока НЕ
    признаки модели — их прирост меряется в `dwp.bench_extras`, и пока
    замер не сделан, файл существует ровно для замера.

    Читает матчи по одному: в `data/matches` два гигабайта, целиком в
    память они не влезут.
    """
    from . import builds as BLD, damage as DMG, items as I
    book = I.load()
    if not book.ok:
        raise SystemExit("ОШИБКА: нет справочника предметов (data/items.json).")
    frames, n, no_build, no_dmg = [], 0, 0, 0
    for m in BLD.iter_matches(config.RAW_MATCHES_DIR, limit):
        if m.get("radiant_win") is None:
            continue
        b = BLD.match_build_frame(m, book)
        d = DMG.match_damage_frame(m)
        if b is None and d is None:
            continue
        if b is None:
            no_build += 1
        if d is None:
            no_dmg += 1
        if b is not None and d is not None:
            fr = b.merge(d, on=["match_id", "minute"], how="outer")
        else:
            fr = b if b is not None else d
        frames.append(fr)
        n += 1
        if n % 500 == 0:
            print(f"  разобрано матчей: {n}", flush=True)
    if not frames:
        raise SystemExit("ОШИБКА: ни одного матча с нужными полями.")
    df = pd.concat(frames, ignore_index=True)
    for c in df.columns:
        if df[c].dtype.kind == "f":
            df[c] = df[c].round(4)
    df.to_csv(out, index=False, compression="gzip")
    print(f"extras: {len(df)} строк из {df['match_id'].nunique()} матчей, "
          f"{len(df.columns)} столбцов -> {out}")
    print(f"  без журнала покупок: {no_build} матчей, без стычек: {no_dmg}")
    print(f"  столбцы: {', '.join(c for c in df.columns if c not in ('match_id', 'minute'))}")


def do_meta(matches: list[dict], out: Path) -> None:
    df = pd.DataFrame([_meta_row(m) for m in matches])
    df.to_csv(out, index=False)
    print(f"meta: {len(df)} матчей -> {out}")
    print(f"  патчи: {dict(df['patch'].value_counts().head(10))}")
    print(f"  формат objectives: {dict(df['obj_fmt'].value_counts())}")


def do_events(matches: list[dict], out: Path) -> None:
    """Перепись типов и ключей. Нужна, чтобы не гадать, какие поля бывают:
    отсутствие типа в одном матче ничего не значит, а доля по всей базе —
    значит."""
    types: Counter = Counter()
    keys: dict[str, Counter] = defaultdict(Counter)
    team_vals: dict[str, Counter] = defaultdict(Counter)
    n_with_obj = 0
    for m in matches:
        obj = m.get("objectives")
        if obj:
            n_with_obj += 1
        for e in obj or []:
            t = str(e.get("type"))
            types[t] += 1
            for k in e:
                keys[t][k] += 1
            if "team" in e:
                team_vals[t][e["team"]] += 1
    lines = [f"матчей {len(matches)}, из них с непустым objectives {n_with_obj}", ""]
    for t, n in types.most_common():
        present = ", ".join(f"{k}:{c}" for k, c in keys[t].most_common())
        lines.append(f"{t}  всего {n}")
        lines.append(f"    ключи (сколько событий имеют ключ): {present}")
        if team_vals[t]:
            lines.append(f"    значения team: {dict(team_vals[t])}")
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nevents -> {out}")


def do_testpred(matches: list[dict], model: Path, out: Path) -> None:
    from .compare import load_artefact, predict_on
    art = load_artefact(model)
    test_ids = set(art.get("test_match_ids") or [])
    if not test_ids:
        raise SystemExit("ОШИБКА: в артефакте нет test_match_ids.")
    sel = [m for m in matches if int(m["match_id"]) in test_ids]
    if not sel:
        raise SystemExit("ОШИБКА: тестовые матчи не найдены в источнике.")
    p, y, g = predict_on(art, sel)
    minutes = np.concatenate([F.match_state_frame(m)["minute"].to_numpy() for m in sel])
    if len(minutes) != len(y):
        raise SystemExit(f"ОШИБКА: минут {len(minutes)}, строк {len(y)} — не сходится.")
    df = pd.DataFrame({"match_id": g, "minute": minutes, "y": y, "p": np.round(p, 6)})
    df.to_csv(out, index=False, compression="gzip")
    print(f"testpred: {len(df)} строк из {df['match_id'].nunique()} матчей "
          f"по модели {model.name} -> {out}")
    print(f"  калибратор {art.get('state_calib')}, eps {art.get('calib_eps')}, "
          f"признаков {len(art['state_features'])}")


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(description="Выгрузка данных проекта в плоские файлы.")
    ap.add_argument("--what", required=True,
                    choices=["draft", "frames", "meta", "events", "testpred",
                             "extras"])
    ap.add_argument("--source", choices=["real", "synthetic"], default="real")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", type=Path, default=None,
                    help="артефакт модели, нужен только для --what testpred")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)

    default_ext = ".txt" if args.what == "events" else (
        ".csv" if args.what == "meta" else ".csv.gz")
    out = args.out or (config.DATA_DIR / f"export_{args.what}{default_ext}")
    out.parent.mkdir(parents=True, exist_ok=True)

    # extras и frames читают матчи по одному: в data/matches два гигабайта
    # JSON, и разобранные в словари они в память этой машины не влезают.
    # Остальным режимам нужен весь список сразу (Elo — хронологический).
    if args.what == "extras":
        do_extras(out, args.limit)
        print(f"размер файла: {out.stat().st_size / 1e6:.2f} МБ")
        return 0
    if args.what == "frames":
        from .builds import iter_matches
        # Отбор матч за матчем тем же `usable_matches`, что и раньше. Свой,
        # «похожий» фильтр здесь уже стоил получаса: он пропустил матчи, у
        # которых нет пригодного состава, выгрузка frames стала шире
        # выгрузки draft, и bench упал на их несовпадении.
        stream = (m for m in iter_matches(
            config.RAW_MATCHES_DIR if args.source == "real"
            else config.SYNTH_MATCHES_DIR, args.limit)
            if F.usable_matches([m], verbose=False))
        do_frames(stream, out)
        print(f"размер файла: {out.stat().st_size / 1e6:.2f} МБ")
        return 0

    # draft и meta обходятся урезанными матчами и читают их потоком.
    if args.what in ("draft", "meta"):
        matches = slim_stream(args.source, args.limit)
        if args.what == "draft":
            do_draft(matches, out)
        else:
            do_meta(matches, out)
        print(f"размер файла: {out.stat().st_size / 1e6:.2f} МБ")
        return 0

    # events смотрит на все матчи, включая непригодные для обучения:
    # именно у них интереснее всего, чем они непригодны.
    matches = _load(args.source, args.limit, usable_only=(args.what != "events"))

    if args.what == "draft":
        do_draft(matches, out)
    elif args.what == "frames":
        do_frames(matches, out)
    elif args.what == "meta":
        do_meta(matches, out)
    elif args.what == "events":
        do_events(matches, out)
    else:
        if args.model is None:
            raise SystemExit("ОШИБКА: для --what testpred нужен --model.")
        do_testpred(matches, args.model, out)
    print(f"размер файла: {out.stat().st_size / 1e6:.2f} МБ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
