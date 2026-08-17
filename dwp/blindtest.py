"""Слепой тест: оценка матча без знания исхода, с запечатанным ответом.

ЗАЧЕМ ЭТО ОТДЕЛЬНО ОТ ОБЫЧНЫХ МЕТРИК. Все замеры проекта устроены так:
взяли модель, прогнали на тесте, посмотрели число, поправили, прогнали
снова. Это честно ровно один раз. К пятому кругу тестовая выборка слепой
быть перестаёт: исход каждого матча уже подсмотрен через метрику, и
«улучшение» запросто оказывается подгонкой под конкретные 1547 матчей.
Поймать такое изнутри нельзя — метрика при этом растёт по-настоящему.

Поэтому здесь два ХОДА, и порядок между ними обеспечен механически:

    --seal     прогнать матчи ЛАЙВ-путём поминутно и записать оценки в файл.
               Исход в этот момент не читается вообще: ни для отбора матчей,
               ни для метрик. Файл закрывается контрольной суммой.
    --reveal   вскрыть: проверить сумму, подтянуть исходы, посчитать.

Пересчитать `--reveal` можно сколько угодно раз — он ничего не меняет. А вот
поправить модель и пересчитать по СТАРОМУ конверту не выйдет: сумма не
сойдётся, и модуль об этом скажет.

ЧТО МЕРЯЕТСЯ ПРИ ВСКРЫТИИ — и это ответ на вопрос «умеет ли модель
анализировать матч, которого не видела»:

  * честность числа: log loss, Brier, ECE, reliability по децилям;
  * РАЗЛОЖЕНИЕ МЁРФИ: Brier = ненадёжность − разрешение + неопределённость.
    Отдельно «число честное» и отдельно «модель различает матчи». Модель,
    которая всегда говорит 50%, идеально надёжна и бесполезна: у неё
    разрешение равно нулю, и видно это только так;
  * сравнение с правилами, которые тоже не знают исхода и ничему не учились:
    константа, «впереди по золоту», «впереди по вышкам», один драфт;
  * когда модель определяется и как часто она уверенно ошибается — то есть
    сколько камбэков она проспала;
  * `--quiz`: человек против модели вслепую. Показывается положение на
    случайной минуте без исхода И БЕЗ ЧИСЛА МОДЕЛИ, вы называете свою
    оценку, потом вскрывается и то и другое.

Запуск:

    python -m dwp.blindtest --seal --n 200
    python -m dwp.blindtest --reveal
    python -m dwp.blindtest --reveal --worst 10        # где ошиблась крупно
    python -m dwp.blindtest --quiz --n 10              # сыграть самому
    python -m dwp.blindtest --seal --source livelog    # по записанным лайвам
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import builds as BLD, config, holdout as HO, items as I, live
from . import test_live_replay as TR
from .livecheck import load_logs, resolve
from .train import _fmt, by_minute_bucket, core_metrics, ece, reliability_table

SEAL_DIR = config.DATA_DIR / "blind"
EPS = 1e-6

# Столбцы конверта. Кроме оценки пишется то, что видно на экране в тот же
# момент: без этого вскрытие показывает «модель ошиблась», но не даёт
# посмотреть, на чём именно.
SEAL_COLUMNS = ["match_id", "minute", "p", "gold_adv", "tower_adv",
                "kills_adv", "draft_logit", "in_test"]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- Ход первый: запечатать -------------------------------------------------

def seal_from_matches(art: dict, model_name: str, n: int, out: Path,
                      only_test: bool = True, limit_scan: int | None = None,
                      verbose: bool = True, use_holdout: bool = False) -> dict:
    """Прогон завершённых матчей ЛАЙВ-путём. Исход не читается.

    Слепота обеспечивается тем, что модуль не открывает `radiant_win` вовсе:
    отбор идёт по пригодности к реплею, а оценка — по тому же коду, что и в
    бою (`extract_state -> LiveTracker -> build_row -> predict`).

    Три степени слепоты, и разница между ними существенна:

      --holdout   матчи, спрятанные от обучения НАВСЕГДА (`dwp.holdout`).
                  Единственная по-настоящему слепая выборка: её не видели
                  ни обучение, ни подбор признаков, ни выбор калибратора.
      по умолчанию  тестовая выборка модели. Обучение её не видело, но по
                  ней принимались решения — а значит она уже частично
                  израсходована.
      --all-matches  что угодно. Слепым это не является, и об этом будет
                  написано при вскрытии.
    """
    test_ids = set(int(x) for x in (art.get("test_match_ids") or []))
    if use_holdout and not art.get("holdout"):
        print("ВНИМАНИЕ: модель обучена БЕЗ слепого холдаута "
              "(`--no-holdout` или старый артефакт).\n"
              "  Матчи холдаута она, возможно, видела при обучении — тогда "
              "оценка ниже\n  завышена. Переобучите: python -m dwp.train "
              "--source real ...", file=sys.stderr)
    if only_test and not use_holdout and not test_ids:
        raise SystemExit("ОШИБКА: в артефакте нет test_match_ids, а без них "
                         "нельзя отличить слепой матч от виденного при "
                         "обучении. Обучите модель заново или снимите "
                         "--only-test.")
    files = sorted(config.RAW_MATCHES_DIR.glob("*.json"))
    if limit_scan:
        files = files[:limit_scan]
    rows: list[dict] = []
    used: list[int] = []
    skipped = 0
    for f in files:
        if len(used) >= n:
            break
        mid_guess = f.stem
        # Отсев по имени файла, до чтения: разбирать двухмегабайтный JSON
        # ради того, чтобы выбросить его по match_id, незачем.
        if mid_guess.isdigit():
            mid_i = int(mid_guess)
            if use_holdout and not HO.is_holdout(mid_i):
                continue
            if not use_holdout and only_test and mid_i not in test_ids:
                continue
        try:
            m = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            skipped += 1
            continue
        mid = int(m["match_id"])
        if use_holdout and not HO.is_holdout(mid):
            continue
        if not use_holdout and only_test and mid not in test_ids:
            continue
        parsed, _why = TR.usable_for_replay(m)
        if parsed is None:
            skipped += 1
            continue
        try:
            df = TR.replay_match(art, m, parsed, _minutes_frame(m))
        except Exception as e:                              # noqa: BLE001
            print(f"  [{mid}] реплей не удался: {type(e).__name__}: {e}",
                  file=sys.stderr)
            skipped += 1
            continue
        p = TR.predict_rows(art, df)
        for i in range(len(df)):
            rows.append({
                "match_id": mid,
                "minute": int(df.iloc[i]["minute"]),
                "p": round(float(p[i]), 6),
                "gold_adv": _num(df.iloc[i].get("gold_adv")),
                "tower_adv": _num(df.iloc[i].get("tower_adv")),
                "kills_adv": _num(df.iloc[i].get("kills_adv")),
                "draft_logit": _num(df.iloc[i].get("draft_logit")),
                "in_test": int(HO.is_holdout(mid) or mid in test_ids),
            })
        used.append(mid)
        if verbose and len(used) % 20 == 0:
            print(f"  запечатано матчей: {len(used)}", flush=True)
    if not rows:
        raise SystemExit(
            "ОШИБКА: ни одного пригодного матча не нашлось."
            + ("\nПри --holdout это значит, что спрятанные матчи не годятся "
               "для реплея\n(нет ярусов вышек или поминутных рядов). "
               "Увеличьте --scan." if use_holdout else ""))
    return _write_seal(rows, used, out, model_name, art,
                       "holdout" if use_holdout else "replay", skipped)


def _minutes_frame(match: dict) -> pd.DataFrame:
    """Сколько минут прогонять. Отдельной функцией, чтобы НЕ звать
    `features.match_state_frame`: та кладёт в кадр столбец `y`, то есть
    исход, а конверт обязан быть слепым даже случайно."""
    g = match.get("radiant_gold_adv") or []
    return pd.DataFrame({"minute": np.arange(max(len(g) - 1, 0) + 1)})


def _num(v) -> str:
    if v is None:
        return ""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    return "" if np.isnan(f) else f"{f:.6g}"


def seal_from_livelog(log_dir: Path, out: Path, model_name: str) -> dict:
    """Конверт из УЖЕ ЗАПИСАННОГО лайва.

    Это самый честный источник, какой есть: числа не пересчитываются, а
    берутся ровно те, что стояли на экране во время матча. Модель их
    выдавала, не зная ничего о будущем, — слепее не бывает.
    """
    df = load_logs(log_dir)
    if df.empty:
        raise SystemExit(f"ОШИБКА: в {log_dir} нет пригодных логов.")
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df[df["match_id"].notna()].copy()
    df["match_id"] = df["match_id"].astype(np.int64)
    rows = []
    for _, r in df.iterrows():
        mn = pd.to_numeric(r.get("minute"), errors="coerce")
        rows.append({
            "match_id": int(r["match_id"]),
            "minute": (None if pd.isna(mn) else int(mn)),
            "p": float(r["p"]),
            "gold_adv": _num(r.get("gold_adv")),
            "tower_adv": _num(r.get("tower_adv")),
            "kills_adv": _num(r.get("kills_adv")),
            "draft_logit": _num(r.get("draft_logit")),
            "in_test": 1,          # лайв-матч моделью не виден по построению
        })
    used = sorted(df["match_id"].unique().tolist())
    return _write_seal(rows, used, out, model_name, None, "livelog", 0)


def _write_seal(rows: list[dict], used: list[int], out: Path, model_name: str,
                art: dict | None, source: str, skipped: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows)[SEAL_COLUMNS].to_csv(out, index=False)
    meta = {
        "source": source,
        "model": model_name,
        "features": list((art or {}).get("state_features") or []),
        "sealed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_matches": len(used),
        "n_rows": len(rows),
        "match_ids": used,
        "skipped": skipped,
        "base_rate_from_training": (None if art is None
                                    else float(art.get("base_rate", float("nan")))),
        "sha256": _sha256(out),
        "note": ("Исходы при записи этого файла не читались. Проверить можно "
                 "так: sha256 совпадает — файл не менялся; match_ids входят в "
                 "test_match_ids модели — матчи не участвовали в обучении."),
    }
    out.with_suffix(".json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    return meta


# --- Ход второй: вскрыть ----------------------------------------------------

def murphy(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Разложение Brier: ненадёжность − разрешение + неопределённость.

    Зачем оно нужно, если есть log loss. Log loss одним числом смешивает две
    разные способности: «не врать» и «различать». Модель, которая всегда
    говорит базовую ставку, идеально надёжна (ненадёжность 0) и совершенно
    бесполезна (разрешение 0). Разложение показывает их порознь, а
    неопределённость — это Brier у той самой константы, то есть планка, от
    которой отсчитывается вся польза.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    o = float(y.mean())
    rel = res = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        w = m.sum() / len(y)
        rel += w * (p[m].mean() - y[m].mean()) ** 2
        res += w * (y[m].mean() - o) ** 2
    unc = o * (1 - o)
    return {"reliability": rel, "resolution": res, "uncertainty": unc,
            "brier": float(np.mean((p - y) ** 2)),
            "skill": (res - rel) / unc if unc > 0 else float("nan")}


def _load_outcomes(meta: dict, ids: list[int], offline: bool
                   ) -> tuple[dict[int, int], dict[int, dict]]:
    """Исходы. Читаются ТОЛЬКО здесь — в запечатывании их нет."""
    out: dict[int, int] = {}
    full: dict[int, dict] = {}
    if meta.get("source") == "livelog":
        got = resolve(ids, config.LIVE_RESOLVED_DIR, offline)
        for mid, m in got.items():
            if m.get("radiant_win") is not None:
                out[int(mid)] = 1 if m["radiant_win"] else 0
                full[int(mid)] = m
        return out, full
    for mid in ids:
        p = config.RAW_MATCHES_DIR / f"{mid}.json"
        if not p.exists():
            continue
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if m.get("radiant_win") is None:
            continue
        out[int(mid)] = 1 if m["radiant_win"] else 0
        full[int(mid)] = m
    return out, full


def reveal(seal_path: Path, worst: int = 0, offline: bool = False,
           with_builds: bool = True) -> int:
    meta_path = seal_path.with_suffix(".json")
    if not seal_path.exists() or not meta_path.exists():
        print(f"ОШИБКА: нет конверта {seal_path} или его описания.\n"
              f"Что делать: python -m dwp.blindtest --seal", file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    now = _sha256(seal_path)
    print("=" * 74)
    print(f"Конверт: {seal_path.name}   источник: {meta['source']}   "
          f"модель: {meta['model']}")
    print(f"Запечатан: {meta['sealed_at']}   матчей {meta['n_matches']}, "
          f"строк {meta['n_rows']}")
    if now != meta["sha256"]:
        print("\nВНИМАНИЕ: контрольная сумма НЕ совпала. Файл изменился после\n"
              "запечатывания, и слепым этот тест считать нельзя. Перезапечатайте.",
              file=sys.stderr)
    else:
        print("Контрольная сумма совпала: файл не менялся с момента записи.")

    df = pd.read_csv(seal_path)
    ids = sorted(df["match_id"].unique().tolist())
    y_by_match, matches = _load_outcomes(meta, ids, offline)
    if not y_by_match:
        print("\nОШИБКА: ни одного исхода не нашлось.", file=sys.stderr)
        return 2
    df = df[df["match_id"].isin(y_by_match)].copy()
    df["y"] = df["match_id"].map(y_by_match).astype(int)
    y = df["y"].to_numpy()
    p = np.clip(df["p"].to_numpy(dtype=float), EPS, 1 - EPS)
    g = df["match_id"].to_numpy()
    n_m = df["match_id"].nunique()
    print(f"Вскрыто: {n_m} матчей, {len(df)} строк. "
          f"Побед Radiant: {int(df.groupby('match_id')['y'].first().sum())} "
          f"из {n_m}")
    not_blind = int((df["in_test"] == 0).sum())
    if not_blind:
        print(f"ВНИМАНИЕ: {not_blind} строк из матчей, которых нет ни в "
              f"холдауте, ни в тестовой выборке модели — они не слепые.")
    if meta.get("source") == "replay":
        print("Источник — ТЕСТОВАЯ выборка: обучение её не видело, но по ней "
              "принимались\nрешения (какие признаки, какой калибратор, какую "
              "модель брать в лайв).\nПо-настоящему слепая оценка — "
              "`--seal --holdout`.")
    if len(np.unique(y)) < 2:
        print("\nВСЕ МАТЧИ КОНЧИЛИСЬ ОДИНАКОВО. Так уже было с лайв-логом: "
              "все\nзаписанные матчи выиграл Radiant, и калибровку считать "
              "не на чем.\nlog loss ниже мерит уверенность, а не различение; "
              "AUC не определён.")

    print("\n--- Насколько честно число ---")
    m = core_metrics(y, p)
    ece_v = ece(y, p, g)
    print(f"  log_loss {m['log_loss']:.4f}   brier {m['brier']:.4f}   "
          f"acc {m['acc']:.4f}   auc {m['auc']:.4f}")
    print(f"  ECE (средневзвешенное смещение по децилям): {ece_v:.4f}")
    if meta.get("source") == "holdout":
        # Каждое вскрытие холдаута записывается: число взглядов — это
        # единственное, что отличает слепую оценку от подогнанной.
        k = HO.note_reveal(meta.get("model", "?"), "holdout", n_m, len(df),
                           meta.get("sha256", ""),
                           {**m, "ece": ece_v})
        print(f"\n  Это вскрытие слепого холдаута номер {k} для модели "
              f"{meta.get('model')}.")
        if k >= 3:
            print("  Три и больше — холдаут уже не вполне слеп: выбирая "
                  "правки по этим\n  числам, вы подгоняете модель под него. "
                  "Смотрите сюда на развилках,\n  а не после каждой правки "
                  "(python -m dwp.holdout — весь журнал).")

    print("\n--- Разложение Мёрфи: честность отдельно, различение отдельно ---")
    mu = murphy(y, p)
    if mu["uncertainty"] <= 0:
        print("  не определено: все матчи кончились одинаково, различать "
              "нечего.\n  Нужны оба исхода — тогда разложение и появится.")
    else:
        print(f"  brier              {mu['brier']:.4f}")
        print(f"  ненадёжность       {mu['reliability']:.4f}   "
              f"(0 — идеально калибровано)")
        print(f"  разрешение         {mu['resolution']:.4f}   "
              f"(0 — модель не отличает матчи друг от друга)")
        print(f"  неопределённость   {mu['uncertainty']:.4f}   "
              f"(brier у константы; отсюда и считается польза)")
        print(f"  навык (res-rel)/unc {mu['skill']:+.4f}   "
              f"{'модель полезнее константы' if mu['skill'] > 0 else 'КОНСТАНТА НЕ ХУЖЕ'}")

    print("\n--- Правила, которые тоже не знают исхода ---")
    base = meta.get("base_rate_from_training")
    rows = []
    if base is not None and not (isinstance(base, float) and np.isnan(base)):
        pb = np.full(len(y), float(base))
        rows.append(("константа (base rate обучения)", pb, None))
    ga = pd.to_numeric(df["gold_adv"], errors="coerce").to_numpy()
    ok = ~np.isnan(ga)
    acc_gold = (float(np.mean((ga[ok] > 0).astype(int) == y[ok]))
                if ok.any() else float("nan"))
    ta = pd.to_numeric(df["tower_adv"], errors="coerce").to_numpy()
    okt = ~np.isnan(ta) & (ta != 0)
    acc_tow = (float(np.mean((ta[okt] > 0).astype(int) == y[okt]))
               if okt.any() else float("nan"))
    dl = pd.to_numeric(df["draft_logit"], errors="coerce").to_numpy()
    okd = ~np.isnan(dl)
    if okd.sum() > 0:
        pd_ = np.clip(1 / (1 + np.exp(-dl[okd])), EPS, 1 - EPS)
        rows.append(("один драфт (без карты)", pd_, okd))
    print(f"  {'правило':<34}{'log_loss':>10}{'brier':>9}{'acc':>8}")
    for name, pp, mask in rows:
        yy = y if mask is None else y[mask]
        mm = core_metrics(yy, pp)
        print(f"  {name:<34}{mm['log_loss']:>10.4f}{mm['brier']:>9.4f}"
              f"{mm['acc']:>8.4f}")
    print(f"  {'модель':<34}{m['log_loss']:>10.4f}{m['brier']:>9.4f}"
          f"{m['acc']:>8.4f}")
    print(f"\n  «впереди по золоту — тот и выиграет»: угадывает "
          f"{acc_gold:.4f} строк ({int(ok.sum())})")
    print(f"  «впереди по вышкам — тот и выиграет»: угадывает "
          f"{acc_tow:.4f} строк ({int(okt.sum())}, ничьи по вышкам не в счёт)")
    print("  Это не модели, а правила без обучения: они показывают, сколько "
          "стоит\n  сама позиция, чтобы прирост модели не приписали ей "
          "целиком.")

    print("\n--- Reliability по децилям ---")
    print(_fmt(reliability_table(y, p, g)))
    if "minute" in df.columns and df["minute"].notna().any():
        print("\n--- По десятиминутным отрезкам ---")
        d = df.dropna(subset=["minute"]).copy()
        print(_fmt(by_minute_bucket(d, np.clip(d["p"].to_numpy(float), EPS, 1 - EPS),
                                    d["y"].to_numpy(), float(np.mean(y)))))

    print("\n--- Когда модель определяется и когда ошибается уверенно ---")
    decided, wrong90 = [], []
    per_match = []
    for mid, d in df.groupby("match_id"):
        d = d.sort_values("minute")
        yy = int(d["y"].iloc[0])
        pp = np.clip(d["p"].to_numpy(dtype=float), EPS, 1 - EPS)
        conf_win = pp if yy == 1 else 1 - pp          # уверенность в победителе
        end = float(d["minute"].max()) if d["minute"].notna().any() else np.nan
        # Минута, после которой модель уже не сомневалась.
        idx = None
        for i in range(len(pp)):
            if np.all(conf_win[i:] >= 0.8):
                idx = i
                break
        settle = float(d["minute"].iloc[idx]) if idx is not None else np.nan
        worst_i = int(np.argmin(conf_win))
        per_match.append({
            "match_id": int(mid), "y": yy, "minutes": end,
            "settle": settle,
            "left": (end - settle) if not np.isnan(settle) else np.nan,
            "min_conf": float(conf_win.min()),
            "min_conf_minute": float(d["minute"].iloc[worst_i]),
            "final_conf": float(conf_win[-1]),
            "mean_conf": float(conf_win.mean()),
        })
        if not np.isnan(settle):
            decided.append(settle)
        if conf_win.min() <= 0.10:
            wrong90.append(int(mid))
    pm = pd.DataFrame(per_match)
    got = pm["settle"].notna()
    print(f"  определилась (и больше не колебалась ниже 80% за победителя) в "
          f"{int(got.sum())} матчах из {len(pm)}")
    if got.any():
        print(f"    медиана: на {pm.loc[got, 'settle'].median():.0f}-й минуте, "
              f"за {pm.loc[got, 'left'].median():.0f} минут до конца")
    print(f"  средняя уверенность в будущем победителе за весь матч: "
          f"{pm['mean_conf'].mean():.3f}")
    print(f"  матчей, где модель хоть раз давала победителю меньше 10%: "
          f"{len(wrong90)} ({len(wrong90) / len(pm) * 100:.1f}%)")
    print("    Это и есть проспанные камбэки. Ниже видно, чем они пахли.")

    if with_builds and len(wrong90):
        _comeback_builds(wrong90, matches, pm)

    if worst:
        print(f"\n--- {worst} матчей, где модель ошибалась крупнее всего ---")
        w = pm.sort_values("min_conf").head(worst)
        print(f"  {'match_id':<12}{'исход':<10}{'мин':>5}{'худшая оценка':>15}"
              f"{'на минуте':>11}{'в конце':>9}")
        for _, r in w.iterrows():
            print(f"  {int(r['match_id']):<12}"
                  f"{'Radiant' if r['y'] == 1 else 'Dire':<10}"
                  f"{r['minutes']:>5.0f}{r['min_conf'] * 100:>14.1f}%"
                  f"{r['min_conf_minute']:>11.0f}{r['final_conf'] * 100:>8.1f}%")
        print("  «худшая оценка» — сколько процентов модель давала будущему "
              "победителю\n  в самый неудачный для себя момент.")
    print("=" * 74)
    return 0


def _comeback_builds(wrong_ids: list[int], matches: dict[int, dict],
                     pm: pd.DataFrame) -> None:
    """Что было со сборками в матчах, где модель уверенно ошиблась.

    Связка с `dwp.builds`: если у победителя, которого модель хоронила, кор
    был собран не хуже, чем у лидера, — значит на экране было видно то, чего
    модель не знает. Если нет — камбэк объясняется чем-то другим, и сборки
    тут ни при чём. Оба ответа полезны, поэтому печатается тот, что вышел.
    """
    book = I.load(download=False, verbose=False)
    if not book.ok:
        return
    ge = lt = 0
    seen = 0
    for mid in wrong_ids:
        m = matches.get(int(mid))
        if not m:
            continue
        fr = BLD.match_build_frame(m, book)
        if fr is None:
            continue
        row = pm[pm.match_id == int(mid)]
        if row.empty or np.isnan(row["min_conf_minute"].iloc[0]):
            continue
        t = int(row["min_conf_minute"].iloc[0])
        if t >= len(fr):
            t = len(fr) - 1
        winner_is_radiant = bool(row["y"].iloc[0] == 1)
        w = fr.iloc[t][f"core_big_{'radiant' if winner_is_radiant else 'dire'}"]
        l = fr.iloc[t][f"core_big_{'dire' if winner_is_radiant else 'radiant'}"]
        seen += 1
        if w >= l:
            ge += 1
        else:
            lt += 1
    if not seen:
        return
    print(f"\n  Сборки в этих матчах (на минуте худшей оценки), разобрано "
          f"{seen}:")
    print(f"    кор будущего победителя собран НЕ ХУЖЕ кора лидера: {ge} "
          f"({ge / seen * 100:.0f}%)")
    print(f"    хуже: {lt} ({lt / seen * 100:.0f}%)")
    print("    Модель ни того, ни другого не видит: предметов в её признаках "
          "нет.")


# --- Ход третий: человек против модели --------------------------------------

def quiz(seal_path: Path, n: int, seed: int, offline: bool,
         answers_path: Path) -> int:
    """Слепой тест для человека: положение без исхода и без числа модели."""
    meta_path = seal_path.with_suffix(".json")
    if not seal_path.exists() or not meta_path.exists():
        print(f"ОШИБКА: нет конверта {seal_path}. Сначала --seal.",
              file=sys.stderr)
        return 2
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    df = pd.read_csv(seal_path)
    ids = sorted(df["match_id"].unique().tolist())
    y_by_match, matches = _load_outcomes(meta, ids, offline)
    df = df[df["match_id"].isin(y_by_match)]
    if df.empty:
        print("ОШИБКА: исходы не найдены, вскрывать нечего.", file=sys.stderr)
        return 2
    rng = np.random.default_rng(seed)
    # По одной позиции из матча: две минуты одного матча — это одно
    # наблюдение, а не два, и подряд они ещё и почти одинаковы.
    pick_ids = list(rng.permutation(sorted(df["match_id"].unique())))[:n]
    book = I.load(download=False, verbose=False)
    rows = []
    print("=" * 74)
    print("СЛЕПОЙ ТЕСТ. Показано положение на карте. Исход и число модели "
          "скрыты.")
    print("Введите свою оценку шансов RADIANT в процентах (0-100), Enter — "
          "пропустить.")
    print("=" * 74)
    for k, mid in enumerate(pick_ids, 1):
        d = df[df.match_id == mid].sort_values("minute")
        if d["minute"].notna().sum() < 3:
            continue
        # Минуту берём не с конца: под занавес всё уже понятно и по счёту.
        lo, hi = 8, max(9, int(d["minute"].max()) - 3)
        want = int(rng.integers(lo, hi))
        r = d.iloc[int((d["minute"] - want).abs().argmin())]
        m = matches.get(int(mid), {})
        print(f"\n--- {k}/{len(pick_ids)} · матч {mid} · минута "
              f"{int(r['minute'])} ---")
        _show_position(r, m, book)
        try:
            # Метка порядка байт: ответы иногда подают файлом, а PowerShell
            # пишет UTF-8 с BOM, и первая строка приезжает с невидимым
            # символом впереди — «не число» на ровном месте.
            raw = input("  ваша оценка за Radiant, %: ").strip().lstrip("﻿")
        except EOFError:
            print("  (ввод кончился)")
            break
        if not raw:
            continue
        try:
            guess = float(raw.replace(",", ".").rstrip("%")) / 100.0
        except ValueError:
            print(f"  {raw!r} — не число, пропускаю")
            continue
        guess = float(np.clip(guess, 0.01, 0.99))
        y = y_by_match[int(mid)]
        pm = float(r["p"])
        print(f"  ВСКРЫВАЕМ: выиграл {'Radiant' if y else 'Dire'}. "
              f"Модель говорила {pm * 100:.1f}%, вы — {guess * 100:.1f}%")
        rows.append({"match_id": int(mid), "minute": int(r["minute"]),
                     "y": y, "p_model": pm, "p_human": guess,
                     "ts": time.strftime("%Y-%m-%d %H:%M:%S")})
    if not rows:
        print("\nНи одного ответа — считать нечего.")
        return 0
    a = pd.DataFrame(rows)
    y = a["y"].to_numpy()
    print("\n" + "=" * 74)
    print(f"Ответов: {len(a)} (матчей столько же: по одной позиции на матч)")
    print(f"  {'кто':<10}{'log_loss':>10}{'brier':>9}{'угадал сторону':>17}")
    for name, col in (("вы", "p_human"), ("модель", "p_model")):
        p = np.clip(a[col].to_numpy(dtype=float), EPS, 1 - EPS)
        ll = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        br = float(np.mean((p - y) ** 2))
        acc = float(np.mean((p >= 0.5).astype(int) == y))
        print(f"  {name:<10}{ll:>10.4f}{br:>9.4f}{acc:>17.2f}")
    print(f"  {'константа 50%':<10}{np.log(2):>10.4f}{0.25:>9.4f}{'-':>17}")
    if len(a) < 30:
        print(f"\n  ВНИМАНИЕ: {len(a)} наблюдений — это мало. Разница в log "
              f"loss меньше\n  0.1 на такой выборке ничего не значит; чтобы "
              f"сравнение стало\n  осмысленным, нужны десятки позиций.")
    answers_path.parent.mkdir(parents=True, exist_ok=True)
    old = (pd.read_csv(answers_path) if answers_path.exists()
           else pd.DataFrame())
    pd.concat([old, a], ignore_index=True).to_csv(answers_path, index=False)
    print(f"  ответы дописаны: {answers_path}")
    print("=" * 74)
    return 0


def score_guesses(path: Path, offline: bool) -> int:
    """Догадки, оставленные в слепом режиме панели, против модели.

    Отличие от `--quiz` в том, что здесь человек смотрел ЖИВОЙ матч и не мог
    подсмотреть исход даже случайно: его тогда ещё не существовало. Это самый
    чистый вид сравнения, какой вообще можно устроить.
    """
    if not path.exists():
        print(f"Догадок нет ({path}).\nЧто делать: откройте панель, включите "
              f"«слепой режим» и назовите своё число.", file=sys.stderr)
        return 2
    df = pd.read_csv(path)
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df[df["match_id"].notna()].copy()
    if df.empty:
        print("В файле нет догадок с опознанным матчем.", file=sys.stderr)
        return 2
    df["match_id"] = df["match_id"].astype(np.int64)
    ids = sorted(df["match_id"].unique().tolist())
    got = resolve(ids, config.LIVE_RESOLVED_DIR, offline)
    y = {int(k): (1 if v.get("radiant_win") else 0) for k, v in got.items()
         if v.get("radiant_win") is not None}
    print(f"Догадок в файле: {len(df)} по {len(ids)} матчам; исход известен у "
          f"{len(y)}")
    d = df[df["match_id"].isin(y)].copy()
    if d.empty:
        print("Ни один матч ещё не досмотрен — считать нечего. Повторите "
              "позже: OpenDota разбирает матч не сразу.")
        return 0
    d["y"] = d["match_id"].map(y)
    yy = d["y"].to_numpy()
    print(f"\n  {'кто':<10}{'log_loss':>10}{'brier':>9}{'угадал сторону':>17}"
          f"{'наблюдений':>12}")
    for name, col in (("вы", "p_human"), ("модель", "p_model")):
        p = np.clip(d[col].to_numpy(dtype=float), EPS, 1 - EPS)
        ll = float(-np.mean(yy * np.log(p) + (1 - yy) * np.log(1 - p)))
        print(f"  {name:<10}{ll:>10.4f}"
              f"{float(np.mean((p - yy) ** 2)):>9.4f}"
              f"{float(np.mean((p >= 0.5).astype(int) == yy)):>17.2f}"
              f"{len(d):>12}")
    print(f"  {'константа':<10}{float(np.log(2)):>10.4f}{0.25:>9.4f}")
    n_m = d["match_id"].nunique()
    print(f"\n  Матчей: {n_m}. Считать надо по ним, а не по строкам: две "
          f"догадки\n  в одном матче — это не два независимых наблюдения.")
    if n_m < 20:
        print("  На такой выборке сравнение ничего не решает — нужны десятки "
              "матчей.")
    return 0


def _show_position(r: pd.Series, match: dict, book: I.ItemBook) -> None:
    def f(x, suffix=""):
        v = pd.to_numeric(x, errors="coerce")
        return "—" if pd.isna(v) else f"{v:+.0f}{suffix}"

    print(f"  золото Radiant−Dire: {f(r.get('gold_adv'))}    "
          f"вышки: {f(r.get('tower_adv'))}    фраги: {f(r.get('kills_adv'))}")
    dl = pd.to_numeric(r.get("draft_logit"), errors="coerce")
    if not pd.isna(dl):
        print(f"  драфт до старта давал Radiant "
              f"{100 / (1 + np.exp(-float(dl))):.1f}%")
    if not match or not book.ok:
        return
    fr = BLD.match_build_frame(match, book)
    if fr is None:
        return
    t = min(int(r["minute"]), len(fr) - 1)
    row = fr.iloc[t]
    print(f"  предметы: перевес {row['item_value_adv']:+.0f} золота; "
          f"крупных у коров {int(row['core_big_radiant'])} против "
          f"{int(row['core_big_dire'])}")


# --- CLI --------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Слепой тест: оценка без знания исхода, с запечатыванием.")
    ap.add_argument("--seal", action="store_true", help="запечатать оценки")
    ap.add_argument("--reveal", action="store_true", help="вскрыть и посчитать")
    ap.add_argument("--quiz", action="store_true",
                    help="сыграть самому против модели вслепую")
    ap.add_argument("--guesses", action="store_true",
                    help="посчитать догадки, оставленные в слепом режиме "
                         "веб-панели (data/blind/live_guesses.csv)")
    ap.add_argument("--source", choices=["replay", "livelog"], default="replay",
                    help="откуда брать оценки: реплей завершённых матчей "
                         "лайв-путём или уже записанный лайв-лог")
    ap.add_argument("--n", type=int, default=100,
                    help="сколько матчей запечатать (или позиций в --quiz)")
    ap.add_argument("--model", type=Path, nargs="+", default=None,
                    help="модель или несколько (ансамбль): вероятности "
                         "усредняются. По умолчанию — то же, что в бою: "
                         "models\\ens_*.pkl, если он обучен")
    ap.add_argument("--file", type=Path, default=None,
                    help="файл конверта (по умолчанию data/blind/<модель>.csv)")
    ap.add_argument("--log-dir", type=Path, default=config.LIVE_LOG_DIR)
    ap.add_argument("--holdout", action="store_true",
                    help="брать матчи из слепого холдаута (dwp.holdout) — "
                         "единственная выборка, которой не видели ни "
                         "обучение, ни подбор признаков")
    ap.add_argument("--all-matches", action="store_true",
                    help="не ограничиваться тестовой выборкой модели "
                         "(тогда тест НЕ слепой, и это будет написано)")
    ap.add_argument("--scan", type=int, default=None,
                    help="сколько файлов просмотреть при отборе")
    ap.add_argument("--worst", type=int, default=0,
                    help="показать N матчей с самой крупной ошибкой")
    ap.add_argument("--offline", action="store_true",
                    help="не ходить в сеть за исходами лайв-лога")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    if not (args.seal or args.reveal or args.quiz or args.guesses):
        ap.print_help()
        return 0
    if args.guesses:
        return score_guesses(SEAL_DIR / "live_guesses.csv", args.offline)

    kind = "holdout" if (args.holdout and args.source != "livelog") else args.source
    try:
        model_paths = live.resolve_models(args.model or live.default_models())
    except live.LiveError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    # Имя конверта. У ансамбля оно должно отличаться от имени одиночной
    # модели: конверт, запечатанный одной моделью и вскрытый другой, — это
    # не слепой тест, а сравнение двух разных вещей под одним именем.
    model_stem = (model_paths[0].stem if len(model_paths) == 1 else
                  f"ens{len(model_paths)}_{model_paths[0].stem}"
                  f"..{model_paths[-1].stem}")
    name = (args.file or SEAL_DIR / f"{model_stem}_{kind}.csv")
    if args.seal:
        if args.source == "livelog":
            meta = seal_from_livelog(args.log_dir, name, model_stem)
        else:
            try:
                art = live.load_models(model_paths)
            except live.LiveError as e:
                print(f"ОШИБКА: {e}", file=sys.stderr)
                return 2
            meta = seal_from_matches(art, live.log_model_name(art), args.n, name,
                                     only_test=not args.all_matches,
                                     limit_scan=args.scan,
                                     use_holdout=args.holdout)
        print(f"\nЗапечатано: {meta['n_matches']} матчей, {meta['n_rows']} "
              f"строк -> {name}")
        print(f"  sha256 {meta['sha256'][:16]}…  описание: "
              f"{name.with_suffix('.json').name}")
        print("  Исходы при записи не читались. Вскрыть: "
              "python -m dwp.blindtest --reveal"
              + ("" if args.source == "replay" else " --source livelog"))
    if args.quiz:
        return quiz(name, args.n, args.seed, args.offline,
                    SEAL_DIR / "quiz_answers.csv")
    if args.reveal:
        if not name.exists() and args.holdout:
            print(f"ОШИБКА: нет конверта {name}.\nСначала: python -m "
                  f"dwp.blindtest --seal --holdout", file=sys.stderr)
            return 2
        return reveal(name, worst=args.worst, offline=args.offline)
    return 0


if __name__ == "__main__":
    sys.exit(main())
