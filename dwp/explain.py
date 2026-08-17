"""Разложение предсказания по вкладам и поиск переломных моментов.

Все вклады считаются в СЫРЫХ ЛОГИТАХ, до калибровки. Это важно: сумма
вкладов даёт сырой логит модели, а показанная пользователю вероятность —
результат изотоники поверх него. Поэтому «вклад в процентных пунктах»
здесь принципиально не считается: изотоника нелинейна и немонотонна по
отношению к отдельному признаку, разложить её по слагаемым нельзя.

Для стейт-модели используется pred_contrib=True — это точный разбор
деревьев (TreeSHAP внутри LightGBM), а не приближение поверх модели.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd


# --- Драфт-модель -------------------------------------------------------

def explain_draft(artefact: dict, radiant: list[int], dire: list[int],
                  elo_diff: float) -> pd.DataFrame:
    """Вклад каждого героя и Elo в ЛИНЕЙНЫЙ логит драфт-модели.

    Внимание: это логит логистической регрессии, а не логит вероятности,
    которую отдаёт CalibratedClassifierCV. После калибровки числа
    отличаются; линейное разложение объясняет модель до калибровки.
    """
    id2idx = artefact["id2idx"]
    names = artefact["hero_names"]
    coefs = artefact["draft_coef"]
    n = len(artefact["hero_ids"])
    rows = []
    for hid in radiant:
        i = id2idx.get(int(hid))
        if i is None:
            rows.append({"сторона": "Radiant", "фактор": f"hero_id={hid} (нет в справочнике)",
                         "вклад": 0.0, "известен": False})
            continue
        rows.append({"сторона": "Radiant", "фактор": names.get(int(hid), str(hid)),
                     "вклад": float(coefs[i]), "известен": True})
    for hid in dire:
        i = id2idx.get(int(hid))
        if i is None:
            rows.append({"сторона": "Dire", "фактор": f"hero_id={hid} (нет в справочнике)",
                         "вклад": 0.0, "известен": False})
            continue
        rows.append({"сторона": "Dire", "фактор": names.get(int(hid), str(hid)),
                     "вклад": float(-coefs[i]), "известен": True})
    rows.append({"сторона": "—", "фактор": f"Elo (разница {elo_diff:+.0f})",
                 "вклад": float(coefs[n] * elo_diff / 400.0), "известен": True})
    rows.append({"сторона": "—", "фактор": "свободный член",
                 "вклад": float(artefact["draft_intercept"]), "известен": True})
    df = pd.DataFrame(rows)
    return df.reindex(df["вклад"].abs().sort_values(ascending=False).index)


# --- Стейт-модель -------------------------------------------------------

def state_contributions(artefact: dict, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(вклады [n, n_features], базовое значение [n]) в сырых логитах."""
    booster = artefact["booster"]
    feats = artefact["state_features"]
    contrib = booster.predict(X[feats], num_iteration=booster.best_iteration,
                              pred_contrib=True)
    contrib = np.asarray(contrib)
    return contrib[:, :-1], contrib[:, -1]


def final_breakdown(artefact: dict, X: pd.DataFrame, minute: int) -> pd.DataFrame:
    feats = artefact["state_features"]
    row = X[X["minute"] == minute]
    if row.empty:
        row = X.iloc[[-1]]
    contrib, base = state_contributions(artefact, row)
    df = pd.DataFrame({
        "признак": feats,
        "значение": [row.iloc[0][f] for f in feats],
        "вклад_логит": contrib[0],
    })
    df = df.reindex(df["вклад_логит"].abs().sort_values(ascending=False).index)
    total = float(contrib[0].sum() + base[0])
    return df, float(base[0]), total


# --- Переломные моменты -------------------------------------------------

def turning_points(minutes: np.ndarray, prob: np.ndarray, contrib: np.ndarray,
                   feature_names: list[str], window: int = 3, top_k: int = 4,
                   min_swing: float = 0.06) -> list[dict]:
    """Минуты с наибольшим изменением вероятности за окно `window`.

    Атрибуция — разность вкладов между началом и концом окна. Именно
    разность, а не абсолютный вклад: постоянный перевес по золоту даёт
    большой вклад каждую минуту, но переломом не является.
    """
    n = len(minutes)
    if n <= window:
        return []
    swing = np.full(n, 0.0)
    swing[window:] = prob[window:] - prob[:-window]

    picked: list[int] = []
    order = np.argsort(-np.abs(swing))
    for t in order:
        if abs(swing[t]) < min_swing:
            break
        if any(abs(t - q) < window for q in picked):
            continue      # немаксимальное подавление: одно событие — одна точка
        picked.append(int(t))
        if len(picked) >= top_k:
            break

    out = []
    for t in sorted(picked):
        d = contrib[t] - contrib[t - window]
        idx = np.argsort(-np.abs(d))[:3]
        out.append({
            "minute_from": int(minutes[t - window]),
            "minute_to": int(minutes[t]),
            "p_from": float(prob[t - window]),
            "p_to": float(prob[t]),
            "swing": float(swing[t]),
            "drivers": [(feature_names[i], float(d[i])) for i in idx],
        })
    return out


# --- Кривая в текст -----------------------------------------------------

def _unicode_ok() -> bool:
    """Умеет ли консоль в блочные символы. Windows с кодовой страницей
    866 не умеет, и вместо графика получится каша из знаков вопроса."""
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "\u2588\u2591\u2500".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


def sparkline(minutes: np.ndarray, prob: np.ndarray, height: int = 13,
              width: int = 78, marks: dict[int, str] | None = None) -> str:
    """Кривая вероятности победы Radiant.

    Заливка от линии 50% к значению: сразу видно, кто ведёт и насколько,
    без чтения оси. Переломные моменты помечаются на отдельной строке.
    """
    n = len(prob)
    if n == 0:
        return "(нет данных)"
    uni = _unicode_ok()
    CH_LINE, CH_FILL, CH_MID = ("\u2588", "\u2591", "\u00b7") if uni else ("#", ":", "-")
    BAR_V, BAR_H, COR = ("\u2502", "\u2500", "\u2514") if uni else ("|", "-", "+")

    cols = min(width, n)
    idx = np.linspace(0, n - 1, cols).round().astype(int)
    p = np.clip(prob[idx], 0.0, 1.0)
    m = minutes[idx]
    mid = height // 2

    grid = [[" "] * cols for _ in range(height)]
    for c, v in enumerate(p):
        r = int(round((1.0 - float(v)) * (height - 1)))
        lo, hi = (r, mid) if r < mid else (mid, r)
        for rr in range(lo, hi + 1):
            grid[rr][c] = CH_FILL
        grid[r][c] = CH_LINE
    for c in range(cols):
        if grid[mid][c] == " ":
            grid[mid][c] = CH_MID

    lines = []
    for r in range(height):
        pct = int(round((1 - r / (height - 1)) * 100))
        label = f"{pct:3d}%" if r % 3 == 0 or r == mid else "    "
        lines.append(f"{label} {BAR_V}" + "".join(grid[r]))

    # Метки переломов под графиком.
    if marks:
        row = [" "] * cols
        for minute, ch in marks.items():
            pos = int(np.argmin(np.abs(m - minute)))
            row[pos] = ch
        lines.append("     " + BAR_V + "".join(row))

    axis = [" "] * cols
    labels = [" "] * (cols + 6)
    step = max(6, cols // 9)
    for c in range(0, cols, step):
        axis[c] = BAR_H
        for k, ch in enumerate(str(int(m[c]))):
            if c + k < len(labels):
                labels[c + k] = ch
    lines.append("     " + COR + "".join(BAR_H if a == BAR_H else BAR_H for a in axis))
    lines.append("      " + "".join(labels).rstrip() + "   мин")
    return "\n".join(lines)


def bar(value: float, width: int = 24, lo: float = -1.0, hi: float = 1.0) -> str:
    """Горизонтальная полоса от центра — для таблицы вкладов."""
    uni = _unicode_ok()
    fill, axis = ("\u2588", "\u2502") if uni else ("#", "|")
    half = width // 2
    v = float(np.clip(value / max(abs(lo), abs(hi)), -1.0, 1.0))
    k = int(round(abs(v) * half))
    if v >= 0:
        return " " * half + axis + fill * k + " " * (half - k)
    return " " * (half - k) + fill * k + axis + " " * half
