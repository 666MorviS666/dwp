"""Сверка лайв-лога с завершёнными матчами: три ответа, которых не было.

Всё, что этот проект знал о своей точности, было померено офлайн — на выгрузке
OpenDota, тем же кодом, что строит обучающую выборку. Лайв-путь не измерялся
вообще. Здесь он измеряется: `dwp.livelog` записал, что было показано на
экране, а OpenDota знает, чем матч кончился.

1. КАЛИБРОВКА ЛАЙВА. Reliability по децилям и по десятиминутным отрезкам на
   фактически показанных числах. Именно с ней надо сверять обещания
   `config.RELIABILITY_BANDS`: они посчитаны офлайн и на лайв перенесены на
   веру.

2. ЧТО ТАКОЕ gold_adv В ЛАЙВЕ. Признак обучен на `radiant_gold_adv` из
   OpenDota — это разница ДОБЫТОГО золота (сумма `gold_t`), а не нетворса.
   Лайв подаёт разницу `teams[].net_worth`. На завершённых матчах разница
   между этими величинами измерена: множитель около 1.36 в пользу лидера.
   Здесь сверяются обе величины из лога, `nw_adv` и `graph_gold_last`, с
   `radiant_gold_adv` той же минуты — и становится видно, какую из них надо
   подавать в признак.

3. СМЕЩЕНИЕ game_time. Включает ли он 90 секунд до горна. Сверяется последний
   записанный `game_time` с `duration` и `pre_game_duration` матча.

Запуск:
    python -m dwp.livecheck                       # добрать исходы и посчитать
    python -m dwp.livecheck --offline             # только по уже добранным
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .collect import ApiError, OpenDotaClient
from .train import _fmt, by_minute_bucket, core_metrics, ece, reliability_table


def load_logs(log_dir: Path) -> pd.DataFrame:
    """Все CSV лайв-лога одной таблицей. Файлы без match_id пропускаются."""
    files = sorted(p for p in log_dir.glob("*.csv") if p.is_file())
    if not files:
        return pd.DataFrame()
    frames = []
    for f in files:
        try:
            d = pd.read_csv(f)
        except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as e:
            print(f"  [пропущен] {f.name}: {type(e).__name__}: {e}")
            continue
        if "match_id" not in d.columns or d["match_id"].isna().all():
            print(f"  [пропущен] {f.name}: матч не опознан, сверять не с чем")
            continue
        d["log_file"] = f.name
        frames.append(d)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def resolve(match_ids: list[int], out_dir: Path, offline: bool) -> dict[int, dict]:
    """Завершённые матчи по id. Кэш на диске, чтобы не дёргать API повторно.

    ВАЖНО: кэш отдельный от config.RAW_MATCHES_DIR. Смотрят обычно паблики, а
    RAW_MATCHES_DIR — обучающая выборка про-матчей; подмешать туда паблики
    значит молча испортить следующее обучение.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    got: dict[int, dict] = {}
    need = []
    for mid in match_ids:
        p = out_dir / f"{mid}.json"
        if p.exists():
            try:
                got[mid] = json.loads(p.read_text(encoding="utf-8"))
                continue
            except json.JSONDecodeError:
                pass
        need.append(mid)
    if need and offline:
        print(f"  --offline: {len(need)} матчей не добрано, они пропущены "
              f"({', '.join(str(m) for m in need[:5])}...)")
        return got
    if need:
        client = OpenDotaClient()
        print(f"  добираю {len(need)} матчей с OpenDota...")
        for mid in need:
            try:
                m = client.get(f"/matches/{mid}")
            except ApiError as e:
                print(f"  [{mid}] не добран: {e}")
                continue
            if not isinstance(m, dict) or m.get("radiant_win") is None:
                print(f"  [{mid}] ещё не обработан OpenDota — повторите позже")
                continue
            (out_dir / f"{mid}.json").write_text(
                json.dumps(m, ensure_ascii=False), encoding="utf-8")
            got[mid] = m
    return got


def calibration_by_model(df: pd.DataFrame) -> None:
    """Если один матч смотрели несколькими моделями — считать общей кучей
    нельзя: строки не независимы, и смешение выдаёт среднее двух кривых за
    одну. Зато разложенные по моделям, они дают прямое сравнение на одном
    и том же матче — ровно то, ради чего вторую модель и включают."""
    models = sorted(df["model"].dropna().unique()) if "model" in df else []
    if len(models) <= 1:
        calibration(df)
        return
    print(f"\nВ логе {len(models)} модели: {', '.join(models)}. "
          f"Считаю каждую отдельно — это и есть сравнение на живом матче.")
    for m in models:
        d = df[df["model"] == m]
        print(f"\n===== модель {m} =====")
        calibration(d)


def calibration(df: pd.DataFrame) -> None:
    y = df["y"].to_numpy()
    p = df["p"].to_numpy(dtype=float)
    g = df["match_id"].to_numpy()
    print("\n--- 1. Калибровка ЛАЙВА (по фактически показанным числам) ---")
    print(f"  строк {len(y)}, матчей {df['match_id'].nunique()} "
          f"(эффективный размер выборки ближе ко второму)")
    if len(np.unique(y)) < 2:
        print("  все матчи с одним исходом — считать калибровку не на чем")
        return
    m = core_metrics(y, p)
    print(f"  log_loss={m['log_loss']:.4f}  brier={m['brier']:.4f}  "
          f"acc={m['acc']:.4f}  auc={m['auc']:.4f}")
    print(f"  ECE (средневзвешенное смещение по децилям): {ece(y, p, g):.3f}")
    print(_fmt(reliability_table(y, p, g)))
    if "minute" in df.columns:
        print("\n  по десятиминутным отрезкам:")
        d = df.rename(columns={"minute": "minute"})
        print(_fmt(by_minute_bucket(d, p, y, float(np.mean(y)))))
    print("\n  Сверьте ECE по отрезкам с config.RELIABILITY_BANDS "
          f"{config.RELIABILITY_BANDS}:\n  они померены офлайн и на лайв "
          f"перенесены на веру.")


def gold_question(df: pd.DataFrame, matches: dict[int, dict]) -> None:
    """Какая величина из лога совпадает с radiant_gold_adv."""
    print("\n--- 2. Чем должен быть gold_adv в лайве ---")
    # nw_adv и graph_gold_last не зависят от модели: это сырьё из ответа.
    # Если матч писали двумя моделями, строки задвоятся и «строк сверено»
    # соврёт вдвое, поэтому берём одну модель.
    if "model" in df.columns and df["model"].nunique() > 1:
        keep = sorted(df["model"].dropna().unique())[0]
        df = df[df["model"] == keep]
        print(f"  (несколько моделей в логе; сырьё одинаково, беру {keep})")
    rows = []
    for mid, d in df.groupby("match_id"):
        ga = matches[int(mid)].get("radiant_gold_adv") or []
        if not ga:
            continue
        for _, r in d.iterrows():
            mn = r.get("minute")
            if pd.isna(mn):
                continue
            i = int(mn)
            if not (0 <= i < len(ga)):
                continue
            rows.append({"ref": float(ga[i]),
                         "nw_adv": float(r["nw_adv"]) if pd.notna(r.get("nw_adv")) else np.nan,
                         "graph": float(r["graph_gold_last"])
                                  if pd.notna(r.get("graph_gold_last")) else np.nan})
    if not rows:
        print("  не с чем сверять: в логе нет минут, попадающих в "
              "radiant_gold_adv")
        return
    a = pd.DataFrame(rows)
    print(f"  строк сверено: {len(a)}")
    print(f"  {'величина':<22}{'медиана отн.':>14}{'медиана |разн.|':>17}"
          f"{'корреляция':>12}")
    corrs = {}
    for name, col in (("нетворс (сейчас)", "nw_adv"),
                      ("graph_gold[-1]", "graph")):
        v = a[col].to_numpy()
        ok = ~np.isnan(v) & (np.abs(a["ref"].to_numpy()) > 500)
        if ok.sum() < 10:
            print(f"  {name:<22}{'нет данных':>14}")
            continue
        ratio = np.median(v[ok] / a["ref"].to_numpy()[ok])
        diff = np.median(np.abs(v[ok] - a["ref"].to_numpy()[ok]))
        corr = np.corrcoef(v[ok], a["ref"].to_numpy()[ok])[0, 1]
        corrs[name] = corr
        print(f"  {name:<22}{ratio:>14.3f}{diff:>17.0f}{corr:>12.4f}")
    print("\n  Как читать. Сначала КОРРЕЛЯЦИЯ, потом отношение.")
    print("    корр. > 0.95 — та же величина, отличается лишь масштабом;\n"
          "                   тогда отношение и есть множитель, и его можно\n"
          "                   внести в формулу;\n"
          "    корр. < 0.9  — величина ДРУГОЙ ПРИРОДЫ, и подгонять множитель\n"
          "                   бессмысленно: ошибка не в масштабе.")
    if corrs and max(corrs.values()) < 0.9:
        print("\n  ЗДЕСЬ КОРРЕЛЯЦИЯ НИЗКАЯ. Причина известна и измерена:\n"
              "  нетворс = предметы + несвязанное золото, а radiant_gold_adv =\n"
              "  ДОБЫТОЕ золото. Расходятся они на всём, что куплено и\n"
              "  исчезло, и главное там — БУЙБЭКИ. На 1200 командах зазор\n"
              "  «добытое минус нетворс» объясняется на R2=0.81 парой\n"
              "  (длительность, число буйбэков x уровень): около 255 золота\n"
              "  за минуту на расходники и около 119 x уровень за буйбэк.\n"
              "  Буйбэки идут в концовке и почти всегда у отстающей стороны —\n"
              "  поэтому расхождение растёт именно там, где смотрят.")


def clock_question(df: pd.DataFrame, matches: dict[int, dict]) -> None:
    print("\n--- 3. Что такое game_time ---")
    rows = []
    for mid, d in df.groupby("match_id"):
        m = matches[int(mid)]
        gt = pd.to_numeric(d["game_time"], errors="coerce").max()
        dur, pre = m.get("duration"), m.get("pre_game_duration")
        if gt is None or pd.isna(gt) or not dur:
            continue
        rows.append({"match_id": int(mid), "last_game_time": float(gt),
                     "duration": float(dur),
                     "pre_game": float(pre) if pre is not None else np.nan,
                     "gt-dur": float(gt) - float(dur)})
    if not rows:
        print("  не с чем сверять")
        return
    a = pd.DataFrame(rows)
    print(_fmt(a))
    d = a["gt-dur"].to_numpy()
    print(f"\n  медиана (последний game_time - duration): {np.median(d):+.0f} с")
    print("  Матч смотрели до конца и разница около 0 -> game_time это часы от\n"
          "  горна, как минуты обучающей выборки. Около +90 -> в него входит\n"
          "  предматчевая пауза, и признак `minute` в лайве сдвинут.\n"
          "  (Замерено отдельно: сдвиг даже на 2 минуты стоит менее 0.001\n"
          "  средней вероятности, так что это вопрос аккуратности, не точности.)")


def stratz_question(df: pd.DataFrame) -> None:
    """Наше число против числа Stratz на ОДНИХ И ТЕХ ЖЕ минутах.

    Это единственный корректный способ сравнить две модели: обе оценивают
    один матч в один момент, исход общий. Сравнивать по памяти («у них было
    67, у нас 13») нельзя — минуты разные и выборка одна.
    """
    print("\n--- 4. Наше число против Stratz ---")
    if "p_stratz" not in df.columns:
        print("  в логе нет колонки p_stratz — записи делались до того, как\n"
              "  Stratz подключили. Досмотрите матч заново.")
        return
    d = df[pd.to_numeric(df["p_stratz"], errors="coerce").notna()].copy()
    if d.empty:
        print("  Stratz не вёл ни один из записанных матчей. Живьём он\n"
              "  отслеживает лиговые матчи и высокий рейтинг; рядовой паблик\n"
              "  к ним может не попасть вовсе. Это не сбой.")
        return
    d["p_stratz"] = pd.to_numeric(d["p_stratz"], errors="coerce")
    y = d["y"].to_numpy()
    ours = np.clip(d["p"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    theirs = np.clip(d["p_stratz"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    print(f"  строк с обоими числами: {len(d)}, матчей "
          f"{d['match_id'].nunique()}")

    # ПРОВЕРКА ДОПУЩЕНИЯ: winRateValues — за Radiant. В схеме стороны нет,
    # и перепутать её ничего не мешает. Если сторона наша, то на последних
    # минутах число должно тянуться к исходу; если чужая — к обратному.
    late = d.sort_values("minute").groupby("match_id").tail(3)
    ly = late["y"].to_numpy()
    lp = np.clip(late["p_stratz"].to_numpy(dtype=float), 1e-6, 1 - 1e-6)
    agree = float(np.mean((lp > 0.5) == (ly == 1)))
    print(f"  на последних минутах число Stratz указывает на победителя в "
          f"{agree:.0%} строк")
    if agree < 0.5:
        print("  ВНИМАНИЕ: меньше половины — похоже, это вероятность DIRE,\n"
              "  а не Radiant. Поправьте stratz.LIVE_SIDE_IS_RADIANT и\n"
              "  пересчитайте; до этого сравнение ниже читать нельзя.")
        return
    if len(np.unique(y)) < 2:
        print("  все матчи с одним исходом — log loss сравнивать можно, но\n"
              "  осторожно: он тут мерит уверенность, а не различение.")
    print(f"\n  {'модель':<12}{'log_loss':>10}{'brier':>9}{'ср. p за победителя':>22}")
    for name, p in (("наша", ours), ("Stratz", theirs)):
        ll = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
        br = np.mean((p - y) ** 2)
        conf = np.mean(np.where(y == 1, p, 1 - p))
        print(f"  {name:<12}{ll:>10.4f}{br:>9.4f}{conf:>22.3f}")
    diff = np.abs(ours - theirs)
    print(f"\n  расхождение |наше - Stratz|: среднее {diff.mean():.3f}, "
          f"медиана {np.median(diff):.3f}, p90 {np.percentile(diff, 90):.3f}")
    print("  Помните про размер выборки: один матч — это не сравнение\n"
          "  моделей, а одно наблюдение. Смотрите на число матчей выше.")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Сверка лайв-лога с завершёнными матчами.")
    ap.add_argument("--log-dir", type=Path, default=config.LIVE_LOG_DIR)
    ap.add_argument("--resolved-dir", type=Path, default=config.LIVE_RESOLVED_DIR)
    ap.add_argument("--offline", action="store_true",
                    help="не ходить в сеть, считать только по добранным матчам")
    args = ap.parse_args(argv)

    print(f"Лог: {args.log_dir}")
    df = load_logs(args.log_dir)
    if df.empty:
        print("Лога нет или он пуст.\nЧто делать: посмотрите матч с записью —\n"
              "    python -m dwp.live --watch --server-steam-id <id> "
              "--model models\\live.pkl --log", file=sys.stderr)
        return 2
    df["match_id"] = pd.to_numeric(df["match_id"], errors="coerce")
    df = df[df["match_id"].notna()].copy()
    df["match_id"] = df["match_id"].astype(np.int64)
    print(f"строк в логе: {len(df)}, матчей: {df['match_id'].nunique()}")

    matches = resolve(sorted(df["match_id"].unique().tolist()),
                      args.resolved_dir, args.offline)
    if not matches:
        print("Ни один матч не добран — считать нечего.\nЧто делать: уберите "
              "--offline, либо подождите: OpenDota обрабатывает матч не сразу.",
              file=sys.stderr)
        return 2
    df = df[df["match_id"].isin(matches)].copy()
    df["y"] = df["match_id"].map(
        lambda m: 1 if matches[int(m)].get("radiant_win") else 0)
    print(f"добрано матчей: {len(matches)}, строк под сверку: {len(df)}")

    # Матч без radiant_gold_adv OpenDota не разобрала: исход у него есть, а
    # поминутных рядов нет. Для калибровки он годится, для вопроса про золото —
    # нет, и молчать об этом нельзя: иначе «строк сверено» окажется втрое
    # меньше числа строк в логе, и непонятно почему.
    raw = [m for m, v in matches.items() if not v.get("radiant_gold_adv")]
    if raw:
        n_rows = int(df["match_id"].isin(raw).sum())
        print(f"  из них НЕ РАЗОБРАНЫ OpenDota (нет radiant_gold_adv): "
              f"{len(raw)} шт, {n_rows} строк — {', '.join(str(m) for m in raw)}\n"
              f"  Они идут в калибровку (исход известен), но не в раздел 2.\n"
              f"  Разбор паблик-матчей OpenDota делает не всегда; запросить "
              f"можно кнопкой Parse на opendota.com/matches/<id>.")

    calibration_by_model(df)
    gold_question(df, matches)
    clock_question(df, matches)
    stratz_question(df)
    return 0


if __name__ == "__main__":
    sys.exit(main())
