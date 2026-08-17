"""Сборки героев: во что превращён нетворс и что это меняет.

Откуда взялся модуль. Пользовательская постановка: «команда проигрывает,
но у них кор с шестью слотами — шансы есть». Проверяемая её форма звучит
так: **при равном отставании по золоту меняется ли доля отыгранных матчей
в зависимости от того, собран ли кор отстающих**. На этот вопрос можно
ответить числом, и модуль отвечает — `python -m dwp.builds --table`.

Что здесь есть:

  live_builds()        разбор `players[].items` живого ответа: слоты,
                       стоимость предметов, крупные предметы, свободное золото
  match_build_frame()  то же поминутно для ЗАВЕРШЁННОГО матча, восстановлением
                       по `purchase_log` (см. items.inventory_at)
  comeback_table()     эмпирические шансы отстающих, по когортам, с выборкой
  lookup()             достать шанс для текущей ситуации — это и рисует панель

ОПРЕДЕЛЕНИЯ СОВПАДАЮТ НАМЕРЕННО. Главный урок этого проекта (см. README про
`gold_adv`) — признак, определённый в обучении иначе, чем в лайве, врёт
именно тогда, когда на него смотрят. Поэтому «стоимость предметов» и здесь,
и там считается ОДИНАКОВО: сумма цен предметов в инвентаре по справочнику,
без консьюмаблов и нейтралов. Есть и второй, независимый способ — у Valve
это `net_worth − gold`; он считается только в лайве и служит сверкой, а не
источником (расхождение печатает `--check`).

ЧЕГО ЭТОТ МОДУЛЬ НЕ ДЕЛАЕТ. Он не добавляет признаков в модель. Прирост от
них надо мерить отдельно (`dwp.export --what extras` плюс `dwp.bench`), и
результат замера записан в README — вместе с тем, что из этого вышло.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config, items as I

# Когорты для таблицы шансов. Границы выбраны так, чтобы в каждой клетке
# осталась выборка, о которой можно говорить: сотни матчей, а не десятки.
#
# Сетка расширена вниз (с 10-й минуты и с 2к отставания) после вопроса «а
# разве иначе нельзя»: раньше панель молчала на 19-й минуте при отставании
# 4900, хотя положение уже читаемое. Клеток стало шестнадцать, и они же —
# наблюдения для непрерывной модели (fit_comeback_model), которая отвечает
# уже в любой точке, а не только в центре клетки.
MINUTE_BINS = ((10, 20), (20, 30), (30, 40), (40, 200))
DEFICIT_BINS = ((2000, 5000), (5000, 10000), (10000, 20000),
                (20000, 10 ** 9))
# «Кор собран» — столько предметов в инвентаре (без консьюмаблов) у самого
# богатого предметами героя стороны. Шесть — это полный инвентарь, ровно та
# формулировка, которую и надо проверить.
FULL_SLOTS = 6

TABLE_PATH = config.DATA_DIR / "comeback.json"


# --- Лайв ----------------------------------------------------------------

@dataclass
class PlayerBuild:
    hero_id: int
    name: str
    slots: int            # занято слотов инвентаря — то, что видно в игре
    backpack: int
    kit: int              # предметов сборки (без консьюмаблов, с рюкзаком)
    value: int            # золота в предметах (по справочнику)
    big: int
    net_worth: float | None
    gold: float | None    # свободное золото: его ещё не превратили в предметы
    level: int | None
    items: list[dict] = field(default_factory=list)

    @property
    def valve_value(self) -> float | None:
        """Стоимость имущества по счёту самой Valve. Сверка, не источник."""
        if self.net_worth is None or self.gold is None:
            return None
        return float(self.net_worth) - float(self.gold)


def live_builds(payload: dict, book: I.ItemBook,
                hero_names: dict[int, str] | None = None) -> dict:
    """Сборки обеих команд из ответа GetRealtimeStats.

    Возвращает {team_number: {"players": [...], "value":, "unspent":,
    "big":, "core": PlayerBuild|None}}. Пустой словарь, если справочник не
    загружен или в ответе нет `items` — молча показывать нули нельзя, панель
    в этом случае просто не рисует блок.
    """
    if not book.ok:
        return {}
    out: dict[int, dict] = {}
    saw_items = False
    for t in (payload.get("teams") or []):
        tn = t.get("team_number")
        if tn is None:
            continue
        players: list[PlayerBuild] = []
        for p in (t.get("players") or []):
            hid = int(p.get("heroid") or p.get("hero_id") or 0)
            b = I.build_of(p.get("items"), book)
            if b.known:
                saw_items = True
            players.append(PlayerBuild(
                hero_id=hid,
                name=(hero_names or {}).get(hid, str(hid)),
                slots=b.slots, backpack=b.backpack, kit=b.kit,
                value=b.value, big=b.big,
                net_worth=p.get("net_worth"), gold=p.get("gold"),
                level=p.get("level"),
                items=[{"id": it.id, "name": it.dname, "cost": it.cost,
                        "img": it.img, "big": it.big,
                        "consumable": it.consumable} if it else None
                       for it in b.items],
            ))
        if not players:
            continue
        core = max(players, key=lambda q: (q.value, q.net_worth or 0))
        out[int(tn)] = {
            "players": players,
            "value": sum(q.value for q in players),
            "big": sum(q.big for q in players),
            "unspent": (None if any(q.gold is None for q in players)
                        else float(sum(q.gold for q in players))),
            "core": core,
        }
    return out if saw_items else {}


# --- Офлайн: поминутно по завершённому матчу -----------------------------

BUILD_COLUMNS = ["item_value_adv", "big_items_adv", "core_big_radiant",
                 "core_big_dire", "core_big_adv", "core_value_adv",
                 "core_kit_radiant", "core_kit_dire", "unspent_adv"]


def _player_series(p: dict, book: I.ItemBook, T: int
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(слоты, стоимость, крупных, потрачено) по минутам 0..T для игрока.

    Один проход по журналу покупок с накопительными суммами — пересобирать
    инвентарь на каждую минуту незачем, а матчей 7734.
    """
    log = [e for e in (p.get("purchase_log") or [])
           if isinstance(e, dict) and e.get("time") is not None and e.get("key")]
    log.sort(key=lambda e: float(e["time"]))
    slots = np.zeros(T + 1)
    value = np.zeros(T + 1)
    big = np.zeros(T + 1)
    spent = np.zeros(T + 1)
    inv: dict[str, int] = {}
    cur_slots = cur_value = cur_big = cur_spent = 0.0
    k = 0
    for t in range(T + 1):
        limit = t * 60 + 59
        while k < len(log) and float(log[k]["time"]) <= limit:
            it = book.by_name.get(str(log[k]["key"]))
            k += 1
            if it is None:
                continue
            paid = it.cost or 0
            for c in it.components:
                if inv.get(c, 0) > 0:
                    inv[c] -= 1
                    if inv[c] <= 0:
                        del inv[c]
                    ci = book.by_name.get(c)
                    if ci is None:
                        continue
                    # За части уже платили — иначе рецепт стоил бы дважды.
                    paid -= ci.cost or 0
                    if not ci.consumable and not ci.neutral:
                        cur_slots -= 1
                        cur_value -= ci.cost or 0
                        if ci.big:
                            cur_big -= 1
            cur_spent += max(0.0, float(paid))
            if it.name in I.CONSUMED_ON_USE:
                continue
            inv[it.name] = inv.get(it.name, 0) + 1
            if not it.consumable and not it.neutral:
                cur_slots += 1
                cur_value += it.cost or 0
                if it.big:
                    cur_big += 1
        slots[t], value[t], big[t], spent[t] = (cur_slots, cur_value, cur_big,
                                                cur_spent)
    return slots, value, big, spent


def match_build_frame(match: dict, book: I.ItemBook,
                      T: int | None = None) -> pd.DataFrame | None:
    """Поминутные признаки сборок одного завершённого матча.

    None, если журнала покупок нет хоть у одного из десяти игроков: перевес,
    посчитанный по девяти, молча занижен, а по метрикам это не видно — та же
    ошибка, что однажды перевернула знак признака по вышкам.
    """
    players = match.get("players") or []
    if len(players) != 10 or not book.ok:
        return None
    gold = match.get("radiant_gold_adv")
    if not gold:
        return None
    n = len(gold) - 1 if T is None else T
    if n < 1:
        return None
    if any(not isinstance(p.get("purchase_log"), list) or not p["purchase_log"]
           for p in players):
        return None
    if any(not isinstance(p.get("gold_t"), list) or not p["gold_t"]
           for p in players):
        return None

    sides = [int(p.get("player_slot", 0)) < 128 for p in players]
    if sum(sides) != 5:
        return None
    S = np.zeros((10, n + 1))
    V = np.zeros((10, n + 1))
    B = np.zeros((10, n + 1))
    SP = np.zeros((10, n + 1))
    G = np.zeros((10, n + 1))
    for i, p in enumerate(players):
        S[i], V[i], B[i], SP[i] = _player_series(p, book, n)
        g = np.asarray(p["gold_t"], dtype=np.float64)
        m = min(len(g), n + 1)
        G[i, :m] = g[:m]
        if m <= n:
            G[i, m:] = g[m - 1]
    r = np.array(sides)
    # «Кор» — не по золоту, а по стоимости предметов: в лайве определение
    # ровно то же, и подменять его нетворсом значило бы завести ещё одну
    # пару расходящихся величин.
    def core_of(mat: np.ndarray, side: np.ndarray) -> np.ndarray:
        return np.take_along_axis(mat[side], np.argmax(V[side], axis=0)[None, :], 0)[0]

    core_b_r, core_b_d = core_of(B, r), core_of(B, ~r)
    core_k_r, core_k_d = core_of(S, r), core_of(S, ~r)
    core_v_r, core_v_d = V[r].max(axis=0), V[~r].max(axis=0)
    unspent = G - SP
    return pd.DataFrame({
        "match_id": int(match["match_id"]),
        "minute": np.arange(n + 1),
        "item_value_adv": V[r].sum(axis=0) - V[~r].sum(axis=0),
        "big_items_adv": B[r].sum(axis=0) - B[~r].sum(axis=0),
        "core_big_radiant": core_b_r,
        "core_big_dire": core_b_d,
        "core_big_adv": core_b_r - core_b_d,
        "core_value_adv": core_v_r - core_v_d,
        "core_kit_radiant": core_k_r,
        "core_kit_dire": core_k_d,
        "unspent_adv": unspent[r].sum(axis=0) - unspent[~r].sum(axis=0),
    })


# --- Шансы отстающих -----------------------------------------------------

def _bin(value: float, bins) -> int | None:
    for i, (lo, hi) in enumerate(bins):
        if lo <= value < hi:
            return i
    return None


def big_group(core_big: float | None) -> str | None:
    """Группа сборки кора. Три ступени, а не порог: где именно проходит
    граница «собран», заранее неизвестно, и выдумывать её незачем —
    таблица показывает всю лестницу, а панель берёт нужную ступень."""
    if core_big is None or (isinstance(core_big, float) and np.isnan(core_big)):
        return None
    n = int(core_big)
    if n <= 2:
        return "0-2"
    if n <= 4:
        return "3-4"
    return "5+"


BIG_GROUPS = ("0-2", "3-4", "5+")


def comeback_rows(match: dict, book: I.ItemBook) -> list[dict]:
    """По одной строке на когорту, в которую матч ПОПАЛ ВПЕРВЫЕ.

    Считать каждую минуту отдельным наблюдением нельзя: сорок минут одного
    матча — это одно наблюдение, а не сорок, и выборка раздулась бы в
    десятки раз (та же ошибка, о которой README пишет в разделе про ставки).
    """
    fr = match_build_frame(match, book)
    if fr is None:
        return []
    gold = np.asarray(match["radiant_gold_adv"], dtype=np.float64)[: len(fr)]
    win_r = bool(match.get("radiant_win"))
    seen: set[tuple[int, int]] = set()
    out = []
    for t in range(len(fr)):
        mb = _bin(t, MINUTE_BINS)
        if mb is None:
            continue
        g = gold[t]
        db = _bin(abs(g), DEFICIT_BINS)
        if db is None:
            continue
        if (mb, db) in seen:
            continue
        seen.add((mb, db))
        trailing_is_radiant = g < 0
        row = fr.iloc[t]
        side = "radiant" if trailing_is_radiant else "dire"
        other = "dire" if trailing_is_radiant else "radiant"
        out.append({
            "match_id": int(match["match_id"]),
            "minute_bin": mb, "deficit_bin": db, "minute": t,
            "deficit": float(abs(g)),
            "core_big": float(row[f"core_big_{side}"]),
            "lead_core_big": float(row[f"core_big_{other}"]),
            "core_kit": float(row[f"core_kit_{side}"]),
            "group": big_group(row[f"core_big_{side}"]),
            "comeback": int(win_r == trailing_is_radiant),
        })
    return out


def iter_matches(directory: Path, limit: int | None = None):
    """Матчи по одному, не загружая выгрузку целиком.

    В `data/matches` два гигабайта JSON; разобранные в словари, они не
    поместятся в память этой машины. Поэтому здесь генератор, а не
    `features.load_matches`.
    """
    files = sorted(directory.glob("*.json"))
    if limit:
        files = files[:limit]
    bad = 0
    for f in files:
        try:
            yield json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            bad += 1
    if bad:
        print(f"[builds] нечитаемых файлов пропущено: {bad}", file=sys.stderr)


def comeback_table(matches, book: I.ItemBook, verbose: bool = True,
                   rows_out: Path | None = None) -> dict:
    rows = []
    skipped = 0
    seen = 0
    for m in matches:
        seen += 1
        if m.get("radiant_win") is None:
            skipped += 1
            continue
        r = comeback_rows(m, book)
        if not r:
            skipped += 1
        rows.extend(r)
        if verbose and seen % 500 == 0:
            print(f"  разобрано матчей: {seen}", flush=True)
    if not rows:
        return {"cells": [], "n_matches": 0, "skipped": skipped}
    df = pd.DataFrame(rows)
    # Сырые наблюдения на диск. Пересчёт таблицы по всей выгрузке занимает
    # около получаса, а любой следующий вопрос («не спрятана ли разница в
    # минуте входа?») — это два действия над этими же строками.
    if rows_out is not None:
        df.to_csv(rows_out, index=False)
        if verbose:
            print(f"  сырые наблюдения -> {rows_out}")

    def stat(s: pd.DataFrame) -> dict:
        if not len(s):
            return {"n": 0, "rate": None, "se": None}
        r = float(s.comeback.mean())
        # Стандартная ошибка доли. Печатается всегда рядом с самой долей:
        # 25% на восьми матчах и 25% на восьмистах — разные утверждения.
        return {"n": int(len(s)), "rate": r,
                "se": float(np.sqrt(max(r * (1 - r), 1e-9) / len(s)))}

    cells = []
    for mb in range(len(MINUTE_BINS)):
        for db in range(len(DEFICIT_BINS)):
            sub = df[(df.minute_bin == mb) & (df.deficit_bin == db)]
            if not len(sub):
                continue
            cell = {"minute_bin": mb, "deficit_bin": db, **stat(sub)}
            cell["by_big"] = {g: stat(sub[sub.group == g]) for g in BIG_GROUPS}
            # Третий разрез, и он важнее двух первых. «Кор собран» само по
            # себе путается с длиной матча: на 40-й минуте собраны обе
            # стороны. Поэтому — сборка кора отстающих ОТНОСИТЕЛЬНО кора
            # лидера: не отстал ли он ещё и по предметам.
            rel = sub.core_big - sub.lead_core_big
            cell["rel_ge"] = stat(sub[rel >= 0])
            cell["rel_lt"] = stat(sub[rel < 0])
            # Средняя минута входа по группам. Без неё разрез по сборке
            # можно спутать с разрезом по времени: внутри отрезка 20-30
            # положение на 21-й минуте и на 29-й — разные вещи, а собранный
            # кор к 29-й минуте бывает чаще.
            cell["minute_mean"] = {
                "all": float(sub.minute.mean()),
                "kit6": (float(sub[sub.core_kit >= FULL_SLOTS].minute.mean())
                         if (sub.core_kit >= FULL_SLOTS).any() else None),
                "kit_less": (float(sub[sub.core_kit < FULL_SLOTS].minute.mean())
                             if (sub.core_kit < FULL_SLOTS).any() else None),
            }
            # Второй разрез — буквально «шесть слотов»: столько предметов
            # сборки у кора, считая рюкзак. Он почти всегда набирается к
            # 25-й минуте, поэтому в заголовок не идёт, но записан.
            cell["kit6"] = stat(sub[sub.core_kit >= FULL_SLOTS])
            cell["kit_less"] = stat(sub[sub.core_kit < FULL_SLOTS])
            cells.append(cell)
    return {
        "cells": cells,
        "model": fit_comeback_model(df),
        "n_matches": int(df.match_id.nunique()),
        "n_rows": int(len(df)),
        "skipped": skipped,
        "minute_bins": [list(b) for b in MINUTE_BINS],
        "deficit_bins": [list(b) for b in DEFICIT_BINS],
        "big_groups": list(BIG_GROUPS),
        "full_slots": FULL_SLOTS,
        "big_item_cost": I.BIG_ITEM_COST,
    }


def load_table(path: Path | None = None) -> dict | None:
    path = path or TABLE_PATH
    if not path.exists():
        return None
    try:
        t = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    # Границы области определения. В файлах, посчитанных до их появления,
    # их нет — тогда берём из сетки когорт: наблюдения по построению не
    # выходят за неё, так что это верная нижняя граница, а не догадка.
    m = t.get("model")
    if m is not None:
        m.setdefault("minute_min", float(t["minute_bins"][0][0]))
        m.setdefault("deficit_min", float(t["deficit_bins"][0][0]))
    return t


MIN_CELL = 30      # меньше — доля неотличима от чего угодно, не показываем

# --- Непрерывная оценка вместо девяти клеток -----------------------------
#
# Таблица по когортам честна, но груба: она молчит на 19-й минуте и при
# отставании 4900, а на границе 10к прыгает ступенькой. Логистическая
# регрессия по тем же наблюдениям даёт число в любой точке и гладко.
#
# ПОЧЕМУ ЭТО НЕ ДУБЛИРУЕТ ОСНОВНУЮ МОДЕЛЬ. Основная модель отвечает на
# вопрос «кто выиграет» по двадцати одному признаку и обучена на минутных
# срезах. Здесь другой вопрос и другая выборка: «как часто отыгрывались
# ИЗ ТАКОГО ЖЕ ПОЛОЖЕНИЯ», одно наблюдение на матч, четыре признака, и
# ни один из них модель не видит целиком (сборка кора ей вообще
# недоступна). Это второй голос, а не пересказ первого.
COMEBACK_FEATURES = ("minute", "deficit_k", "core_big", "core_big_rel")


def fit_comeback_model(df: pd.DataFrame) -> dict | None:
    """Логистическая регрессия «отыграется ли отстающий».

    Возвращает коэффициенты и оценку качества на отложенных матчах — без
    неё число было бы просто нарисованной кривой.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import GroupShuffleSplit
        from sklearn.metrics import log_loss, roc_auc_score
    except ImportError:
        return None
    d = df.dropna(subset=["minute", "deficit", "core_big", "lead_core_big"])
    if len(d) < 500:
        return None
    X = np.c_[d.minute.to_numpy(float),
              d.deficit.to_numpy(float) / 1000.0,
              d.core_big.to_numpy(float),
              (d.core_big - d.lead_core_big).to_numpy(float)]
    y = d.comeback.to_numpy(int)
    g = d.match_id.to_numpy()
    # Сплит по матчам: один матч даёт до трёх наблюдений, и разнести их
    # между обучением и проверкой значило бы подсмотреть исход.
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25,
                                    random_state=7).split(X, y, groups=g))
    m = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    p_te = m.predict_proba(X[te])[:, 1]
    base = float(y[tr].mean())
    out = {
        "coef": [float(v) for v in m.coef_[0]],
        "intercept": float(m.intercept_[0]),
        "features": list(COMEBACK_FEATURES),
        "n_fit": int(len(tr)), "n_test": int(len(te)),
        "n_test_matches": int(len(np.unique(g[te]))),
        "log_loss": float(log_loss(y[te], p_te, labels=[0, 1])),
        "log_loss_base": float(log_loss(y[te], np.full(len(te), base),
                                        labels=[0, 1])),
        "auc": (float(roc_auc_score(y[te], p_te))
                if len(np.unique(y[te])) > 1 else None),
        "base_rate": base,
    }
    # Модель на весь набор — её и сохраняем для панели.
    m_all = LogisticRegression(max_iter=2000).fit(X, y)
    out["coef"] = [float(v) for v in m_all.coef_[0]]
    out["intercept"] = float(m_all.intercept_[0])
    # Область определения. Логистическая регрессия с радостью продолжит
    # прямую куда угодно и выдаст «шанс отыграться» на третьей минуте при
    # отставании в 500 золота — там ещё никто не отстаёт, и наблюдений
    # такого рода в выборке нет ни одного. За границами модуль молчит.
    out["minute_min"] = float(d.minute.min())
    out["deficit_min"] = float(d.deficit.min())
    return out


def comeback_chance(model: dict | None, minute: float, deficit: float,
                    core_big: float | None,
                    lead_core_big: float | None) -> float | None:
    """Непрерывная оценка шанса отыграться.

    None означает «сказать нечего»: нет модели, неизвестна сборка, или
    положение лежит ВНЕ области, на которой модель обучалась. Последнее —
    не формальность: экстраполяция логистической регрессии на третью минуту
    даёт бодрое число там, где отставания ещё не существует как явления.
    """
    if not model or core_big is None:
        return None
    if minute is None or deficit is None:
        return None
    if (minute < model.get("minute_min", 0.0)
            or abs(float(deficit)) < model.get("deficit_min", 0.0)):
        return None
    rel = 0.0 if lead_core_big is None else float(core_big) - float(lead_core_big)
    x = [float(minute), abs(float(deficit)) / 1000.0, float(core_big), rel]
    z = model["intercept"] + sum(c * v for c, v in zip(model["coef"], x))
    return float(1.0 / (1.0 + np.exp(-z)))


def lookup(table: dict | None, minute: float, deficit: float,
           core_big: int | None, lead_core_big: int | None = None) -> dict | None:
    """Шанс отыграться в текущей ситуации — то, что показывает панель.

    None означает «сказать нечего»: либо таблицы нет, либо ситуация вне
    когорт (отставание меньше 5к, минута меньше 20-й), либо в клетке
    слишком мало матчей. Пустое место честнее выдуманного процента.

    ОБ ОДНОЙ НЕТОЧНОСТИ ПРЯМО ЗДЕСЬ. Ось отставания на истории — это
    `radiant_gold_adv`, то есть ДОБЫТОЕ золото, а панель подаёт разницу
    нетворса: та самая пара расходящихся величин, о которой README пишет в
    разделе про `gold_adv`. Нетворс-разница систематически крупнее (медиана
    отношения 1.39), поэтому панель попадает скорее в более тяжёлую клетку,
    чем в более лёгкую, и показанный шанс скорее занижен. Подгонять
    множителем не стали: README отдельно показывает, что ошибка не в
    масштабе.
    """
    if not table or not table.get("cells"):
        return None
    # Непрерывная оценка считается всегда, когда есть модель и сборка: она
    # не знает про границы клеток и потому отвечает и на 12-й минуте, и при
    # отставании 3400.
    smooth = comeback_chance(table.get("model"), minute, deficit,
                             core_big, lead_core_big)
    mb = _bin(minute, [tuple(b) for b in table["minute_bins"]])
    db = _bin(abs(deficit), [tuple(b) for b in table["deficit_bins"]])
    if mb is None or db is None:
        # Вне сетки когорт когортного числа нет, но гладкое — есть.
        return ({"rate": None, "n": 0, "se": None, "group": None,
                 "smooth": smooth, "minute_bin": None, "deficit_bin": None}
                if smooth is not None else None)
    for c in table["cells"]:
        if c["minute_bin"] != mb or c["deficit_bin"] != db:
            continue
        base = {"minute_bin": mb, "deficit_bin": db, "smooth": smooth,
                "all_rate": c["rate"], "all_n": c["n"]}
        g = big_group(core_big)
        got = (c.get("by_big") or {}).get(g) if g else None
        if got and got.get("n", 0) >= MIN_CELL:
            return {**base, "rate": got["rate"], "n": got["n"],
                    "se": got["se"], "group": g}
        if c["n"] >= MIN_CELL:
            return {**base, "rate": c["rate"], "n": c["n"],
                    "se": float(np.sqrt(max(c["rate"] * (1 - c["rate"]), 1e-9)
                                        / c["n"])),
                    "group": None}
        return {**base, "rate": None, "n": c["n"], "se": None, "group": None}
    return ({"rate": None, "n": 0, "se": None, "group": None,
             "smooth": smooth, "minute_bin": mb, "deficit_bin": db}
            if smooth is not None else None)


# --- CLI -----------------------------------------------------------------

def _fmt_bin(b) -> str:
    lo, hi = b
    return f"{lo // 1000}-{hi // 1000}к" if hi < 10 ** 8 else f"{lo // 1000}к+"


def _cellstr(x: dict | None) -> str:
    """Доля без выборки не печатается — это соглашение проекта."""
    if not x or not x.get("n"):
        return f"{'—':>20}"
    if x["n"] < MIN_CELL:
        return f"{'мало, n=' + str(x['n']):>20}"
    return f"{x['rate'] * 100:>8.1f}% ±{x['se'] * 100:>4.1f} n={x['n']:<4}"


def print_table(t: dict) -> None:
    print(f"\nМатчей в выборке: {t['n_matches']}  "
          f"(наблюдений {t['n_rows']}, пропущено матчей без журнала покупок: "
          f"{t['skipped']})")
    print("Одно наблюдение = один матч в одной когорте, взятый на ПЕРВОЙ минуте "
          "попадания.\nСтолбцы — сколько КРУПНЫХ предметов (дороже "
          f"{t.get('big_item_cost', I.BIG_ITEM_COST)}) у самого богатого "
          "предметами героя ОТСТАЮЩЕЙ стороны.\n")
    hdr = (f"  {'минуты':<8}{'отставание':<11}{'матчей':>7}{'отыгрались':>11}"
           + "".join(f"{'кор ' + g:>20}" for g in t["big_groups"]))
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for c in t["cells"]:
        mb = t["minute_bins"][c["minute_bin"]]
        db = t["deficit_bins"][c["deficit_bin"]]
        mtxt = f"{mb[0]}-{mb[1]}" if mb[1] < 100 else f"{mb[0]}+"
        print(f"  {mtxt:<8}{_fmt_bin(db):<11}{c['n']:>7}{c['rate'] * 100:>10.1f}%"
              + "".join(_cellstr((c.get("by_big") or {}).get(g))
                        for g in t["big_groups"]))
    m = t.get("model")
    if m:
        print("\nНепрерывная оценка (та же выборка, логистическая регрессия):")
        print(f"  на отложенных {m['n_test_matches']} матчах: log loss "
              f"{m['log_loss']:.4f} против {m['log_loss_base']:.4f} у "
              f"константы"
              + (f", AUC {m['auc']:.3f}" if m.get("auc") else ""))
        names = {"minute": "минута", "deficit_k": "отставание, тысяч",
                 "core_big": "крупных предметов у кора",
                 "core_big_rel": "их перевес над кором лидера"}
        for f, c in zip(m["features"], m["coef"]):
            print(f"    {names.get(f, f):<32}{c:+.4f}")
        print("  Она и рисуется на панели: работает в любой точке, а не "
              "только\n  в центре клетки, и не прыгает ступенькой на границе.")

    print("\nГлавный разрез — сборка кора ОТСТАЮЩИХ против кора лидера:")
    print(f"  {'минуты':<8}{'отставание':<11}"
          f"{'не хуже лидера':>22}{'хуже лидера':>22}")
    for c in t["cells"]:
        mb = t["minute_bins"][c["minute_bin"]]
        db = t["deficit_bins"][c["deficit_bin"]]
        mtxt = f"{mb[0]}-{mb[1]}" if mb[1] < 100 else f"{mb[0]}+"
        print(f"  {mtxt:<8}{_fmt_bin(db):<11}"
              f"{_cellstr(c.get('rel_ge')):>22}{_cellstr(c.get('rel_lt')):>22}")
    print("\nТо же самое по «шести слотам» (предметы сборки, включая рюкзак):")
    print(f"  {'минуты':<8}{'отставание':<11}"
          f"{'кор 6+ предметов':>22}{'меньше шести':>22}")
    for c in t["cells"]:
        mb = t["minute_bins"][c["minute_bin"]]
        db = t["deficit_bins"][c["deficit_bin"]]
        mtxt = f"{mb[0]}-{mb[1]}" if mb[1] < 100 else f"{mb[0]}+"
        print(f"  {mtxt:<8}{_fmt_bin(db):<11}"
              f"{_cellstr(c.get('kit6')):>22}{_cellstr(c.get('kit_less')):>22}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Сборки героев: разбор лайва и шансы отстающих по истории.")
    ap.add_argument("--table", action="store_true",
                    help="пересчитать таблицу шансов по data/matches и сохранить")
    ap.add_argument("--show", action="store_true",
                    help="показать сохранённую таблицу")
    ap.add_argument("--from-file", type=Path,
                    help="разобрать сборки из сохранённого ответа лайва")
    ap.add_argument("--limit", type=int, default=None,
                    help="сколько матчей брать для --table")
    ap.add_argument("--out", type=Path, default=TABLE_PATH)
    ap.add_argument("--rows-out", type=Path,
                    default=config.DATA_DIR / "comeback_rows.csv",
                    help="куда положить сырые наблюдения для перепроверок")
    args = ap.parse_args(argv)

    book = I.load()
    if not book.ok:
        print("ОШИБКА: нет справочника предметов.", file=sys.stderr)
        return 2

    if args.from_file:
        raw = args.from_file.read_bytes()
        enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
        payload = json.loads(raw.decode(enc))
        from . import features as F
        try:
            _, _, names = F.load_heroes()
        except FileNotFoundError:
            names = {}
        got = live_builds(payload, book, names)
        if not got:
            print("В ответе нет players[].items — сборки не разобрать.")
            return 2
        for tn, side in sorted(got.items()):
            print(f"\n--- team_number {tn}: предметов на {side['value']} золота, "
                  f"крупных {side['big']}, свободного золота "
                  f"{side['unspent'] if side['unspent'] is not None else '—'} ---")
            for q in sorted(side["players"], key=lambda x: -x.value):
                names_ = ", ".join(i["name"] for i in q.items if i)
                vv = q.valve_value
                print(f"  {q.name:<18} ур.{q.level or '?':>3}  слотов {q.slots}"
                      f"  сборка {q.kit}  предметов на {q.value:>6}"
                      f"  крупных {q.big}"
                      f"  свободно {q.gold if q.gold is not None else '—':>6}")
                print(f"      сверка с net_worth-gold: "
                      f"{'—' if vv is None else f'{vv:.0f}'} "
                      f"(разница {'—' if vv is None else f'{q.value - vv:+.0f}'}; "
                      f"в неё входят рюкзак, нейтрал и консьюмаблы)")
                print(f"      {names_}")
        return 0

    if args.table:
        n_files = len(list(config.RAW_MATCHES_DIR.glob("*.json")))
        print(f"Матчей в выгрузке: {n_files}"
              + (f", берём {args.limit}" if args.limit else ""))
        t = comeback_table(iter_matches(config.RAW_MATCHES_DIR, args.limit), book,
                           rows_out=args.rows_out)
        args.out.write_text(json.dumps(t, ensure_ascii=False, indent=1),
                            encoding="utf-8")
        print_table(t)
        print(f"\nТаблица сохранена: {args.out}")
        return 0

    t = load_table(args.out)
    if t is None:
        print(f"Таблицы нет ({args.out}). Посчитать: python -m dwp.builds --table",
              file=sys.stderr)
        return 2
    print_table(t)
    return 0


if __name__ == "__main__":
    sys.exit(main())
