"""Вердикт: одна сторона на весь матч, названная один раз и навсегда.

ЗАЧЕМ. Калиброванная вероятность — правильное число, но читать её глазами
тяжело: на реальных данных она меняется между соседними минутами в среднем
на 4.9 п.п., в худшем случае на 61.4 (README). Стоит перевесу по нетворсу
качнуться туда-сюда около нуля — и «кто побеждает» на экране меняется
каждую минуту, хотя в матче ничего не произошло.

ЧЕМ ЭТО ОТЛИЧАЕТСЯ ОТ ПРЕЖНЕЙ РЕДАКЦИИ. Раньше здесь стоял гистерезис:
вердикт мог поменяться, если сглаженная оценка уходила за порог. Смен
выходило 0.88 за матч вместо 2.10 у сырого процента — меньше, но не ноль.
Требование изменилось и стало жёстче: **ни одной смены**, даже если
противоположная сторона наберёт 90%. Формулировка заказчика:

    «он не должен ни разу колебаться, даже если противоположная сторона
    имеет 90%; то что он поспешил — это будет вина вердикта, то что он не
    предусмотрел камбек той команды»

То есть цена ошибки принята сознательно, и правило теперь такое:

    до T_OPEN            вердикта нет, панель говорит «смотрю»
    с T_OPEN             как только сглаженная уверенность достигает THETA,
                         сторона называется и ФИКСИРУЕТСЯ НАВСЕГДА
    на T_FORCE           если уверенность так и не набралась — называем всё
                         равно, по той стороне, что есть

Смен стороны нет ВООБЩЕ, и не потому, что порог высокий, а потому, что в
правиле нет ветки, которая её меняет. Это структурное свойство, а не
настройка: сломать его правкой констант нельзя.

ЧЕГО ЭТО НЕ ДЕЛАЕТ, И ЭТО ГЛАВНОЕ. Неподвижный вердикт не становится
верным оттого, что он неподвижен. Информации о победителе на седьмой
минуте мало — коридор (`data/horizon.json`) говорит прямо: на минутах
0-10 при оценке лидера 0.50-0.60 лидер выигрывает примерно в 55% случаев.
Ожидание уверенности эту беду частью лечит (ждём, пока сигнал появится),
но потолок ставит не правило, а данные. Поэтому рядом с вердиктом всегда
идёт доля, с которой ТАКИЕ ЖЕ вердикты сбывались, и число матчей, на
которых она посчитана.

ПОЧЕМУ ДОЛЯ ТЕПЕРЬ ЧЕСТНЕЕ, ЧЕМ БЫЛА. Прежняя таблица считала попадания
по МИНУТАМ: сорок минут одного матча давали сорок наблюдений, хотя
независимое наблюдение там одно. Долю приходилось считать по строкам, а
интервал — по матчам, и это оговаривалось отдельно. Теперь решение
принимается один раз за матч, поэтому и наблюдение ровно одно: доля и
интервал считаются на одной и той же выборке независимых матчей.

ГДЕ ПОДБИРАЛОСЬ И ГДЕ МЕРИЛОСЬ — РАЗНЫЕ МАТЧИ. Слепой холдаут делится
пополам по второму хэшу от match_id:

    половина A (настройка)  перебор сетки, выбор THETA, T_FORCE, полураспада
    половина B (замер)      всё, что печатается и показывается на панели

Деление по хэшу, а не по сиду, — по той же причине, что и сам холдаут
(`dwp.holdout`): добор данных не должен двигать матчи между половинами.

    python -m dwp.bench_models          # сначала: он готовит кэш вероятностей
    python -m dwp.verdict --tune        # подобрать правило и померить
    python -m dwp.verdict --show        # что подобрано, без пересчёта
    python -m dwp.verdict --frontier    # чем платит терпение и чем спешка
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np

from . import config, holdout as HO

TABLE_PATH = config.DATA_DIR / "verdict.json"
PRED_CACHE = config.DATA_DIR / "holdout_preds.npz"

EPS = 1e-6

# Оси таблицы «как часто ТАКОЙ вердикт сбывался».
#
# Минут всего две группы, и это не лень. Вердикт выносится один раз, в
# узком окне около T_OPEN, поэтому дробить минуты дальше нечем: клетки
# станут по десятку матчей, а доля на десятке матчей неотличима от чего
# угодно. Второй интервал нужен для случая «подключились к матчу в
# середине» — там вердикт выносится на 30-й минуте, и это совсем другая
# позиция, смешивать её с обычной нельзя.
COMMIT_MINUTE_BINS = (0.0, 15.0, 200.0)
CONF_BINS = (0.50, 0.60, 0.70, 0.80, 0.90, 1.01)

# Меньше этого числа МАТЧЕЙ в клетке — доля не печатается вовсе. Здесь
# именно матчи, а не строки: наблюдение на матч ровно одно.
MIN_CELL = 40

# Сколько минут наблюдения нужно, если подключились к матчу в середине.
# Сглаженный логит по одной точке — это сама точка, и выносить вердикт по
# ней значило бы выносить его по мгновенному значению.
WARMUP_IF_LATE = 2.0

# Сетка перебора.
#
# T_OPEN закреплён снаружи (умолчание 7): «пусть смотрит семь минут» —
# это требование к панели, а не величина для подбора. Цену другого
# значения показывает --frontier.
#
# THETA — уверенность, при которой вердикт выносится. 0.50 означает
# «назвать сразу на T_OPEN, какая бы сторона ни была» (уверенность не
# бывает меньше половины), поэтому этот вариант в сетке есть и служит
# точкой отсчёта.
#
# T_FORCE — крайний срок. Без него матч, где оценка так и не ушла от
# половины, остался бы вообще без вердикта.
GRID_T_OPEN = (5.0, 6.0, 7.0, 8.0, 10.0)
GRID_HALF_LIFE = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0)
GRID_THETA_P = (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90)
GRID_T_FORCE = (0.0, 2.0, 4.0, 6.0, 8.0, 13.0, 1e9)   # +T_OPEN, см. _force_at

# Доля матчей, в которых вердикт обязан появиться. Правило, которое молчит
# на каждом десятом матче, — это не вердикт, а отговорка.
MIN_COMMIT_RATE = 0.98

# Крайняя медиана минуты вердикта при подборе. Заказчик просил семь минут;
# ждать уверенности разрешено, но не до тридцатой минуты.
DEFAULT_BUDGET = 10.0


def _force_str(rule) -> str:
    """Крайний срок для печати. «нет» вместо миллиарда минут.

    Бесконечность в сетке задана большим числом, и напечатанное
    `1000000007` не только нечитаемо, но и сливается с соседней колонкой.
    """
    d = rule["t_force_after"] if isinstance(rule, dict) else rule.t_force_after
    t0 = rule["t_open"] if isinstance(rule, dict) else rule.t_open
    return "нет" if float(d) > 1e6 else f"{float(t0) + float(d):.0f}"


def _safe_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# --- правило -------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """Правило вердикта. Смены стороны в нём нет по построению.

    `t_force` задаётся как ОТСТУП от `t_open`, а не абсолютной минутой:
    иначе при переборе t_open появлялись бы пары, где крайний срок раньше
    открытия, и их пришлось бы отсеивать руками.
    """
    t_open: float = 7.0
    half_life: float = 1.0
    theta_p: float = 0.65
    t_force_after: float = 6.0

    @property
    def t_force(self) -> float:
        return self.t_open + self.t_force_after


def smooth(minutes: np.ndarray, p: np.ndarray, half_life: float) -> np.ndarray:
    """Экспоненциальное окно по ИГРОВОМУ времени, точное на наклоне.

    ЗАЧЕМ ТОЧНОЕ. Правило подбиралось на холдауте, где строка раз в
    минуту, а работает на панели, где опрос раз в две секунды. Если
    сглаживание зависит от частоты, то на экране окажется не то правило,
    которое мерили, — и разойдутся они молча, потому что оба ряда
    выглядят осмысленно.

    Наивная запись `z = w*z + (1-w)*x` с `w = 0.5**(dt/hl)` от частоты
    ЗАВИСИТ: на постоянном входе она верна, а на растущем отстаёт на
    `a*dt*w/(1-w)` вместо непрерывного `a*tau`. Разница поймана
    тестом — 0.12 в логитах на обычном разгоне оценки, то есть до трёх
    процентных пунктов на экране.

    Здесь взято точное решение dz/dt = (x - z)/tau при входе, ЛИНЕЙНОМ
    между отсчётами (ramp-invariant дискретизация):

        k  = exp(-dt/tau) = 0.5**(dt/half_life)
        z1 = z0*k + x0*(1-k) + (x1-x0)*(1 - (tau/dt)*(1-k))

    На постоянном входе она сводится к обычной, на наклоне даёт ровно
    непрерывное отставание `a*tau` при ЛЮБОМ dt, а при dt -> 0 стоит на
    месте. Проверяется `dwp.test_verdict`, раздел 3.
    """
    z = _logit(p)
    out = np.empty_like(z)
    acc = z[0]
    out[0] = acc
    if half_life <= 0:
        return z.copy()
    tau = half_life / np.log(2.0)
    for i in range(1, len(z)):
        dt = float(minutes[i] - minutes[i - 1])
        if not (dt == dt) or dt <= 0:
            out[i] = acc
            continue
        k = 0.5 ** (dt / half_life)
        acc = (acc * k + z[i - 1] * (1.0 - k)
               + (z[i] - z[i - 1]) * (1.0 - (tau / dt) * (1.0 - k)))
        out[i] = acc
    return out


def run_rule(minutes: np.ndarray, p: np.ndarray, rule: Rule,
             first_minute: float | None = None,
             z: np.ndarray | None = None) -> dict:
    """Прогнать правило по одному матчу.

    Возвращает поминутный ряд стороны (+1 Radiant, -1 Dire, 0 «ещё
    смотрю») и момент фиксации. Ряд той же длины, что и вход.

    ГЛАВНОЕ СВОЙСТВО: как только `cur` стал ненулевым, он не меняется —
    ветки, которая его меняет, в цикле нет. Проверять надо именно это, а
    не величину порога.

    `z` можно передать готовым: сглаживание зависит ТОЛЬКО от полураспада,
    а перебор гоняет по девять порогов и семь крайних сроков на каждое
    значение. Пересчитывать один и тот же ряд шестьдесят три раза —
    сорок минут вместо одной.
    """
    if z is None:
        z = smooth(minutes, p, rule.half_life)
    fm = float(minutes[0]) if first_minute is None else float(first_minute)
    # Подключились в середине — ждём окно наблюдения, а не «минуту 7»,
    # которая давно прошла и про которую мы ничего не знаем.
    open_at = max(rule.t_open, fm + WARMUP_IF_LATE) if fm > 0.5 else rule.t_open
    force_at = open_at + rule.t_force_after

    conf = _sigmoid(np.abs(z))
    side = np.zeros(len(z), dtype=int)
    cur = 0
    at = float("nan")
    at_conf = float("nan")
    for i in range(len(z)):
        if cur == 0 and minutes[i] >= open_at and (
                conf[i] >= rule.theta_p or minutes[i] >= force_at):
            cur = 1 if z[i] >= 0 else -1
            at = float(minutes[i])
            at_conf = float(conf[i])
        side[i] = cur
    return {"side": side, "z": z, "conf": conf,
            "committed": cur != 0, "commit_side": cur,
            "commit_minute": at, "commit_conf": at_conf,
            "open_at": open_at, "force_at": force_at,
            # Смен нет по построению. Ключ оставлен, чтобы вызывающий код
            # мог это утверждение проверить, а не поверить в него.
            "flips": 0}


# --- разбор холдаута -----------------------------------------------------

def _half(match_id) -> str:
    """A или B. Второй хэш, чтобы деление не совпало с делением холдаута."""
    h = hashlib.sha1(f"verdict:{int(match_id)}".encode("ascii")).hexdigest()
    return "A" if int(h[:8], 16) % 2 == 0 else "B"


def load_cache(path: Path | None = None, key: str = "p__ENSEMBLE") -> list[dict]:
    """Матчи холдаута как список рядов (минуты, p, исход).

    Кэш готовит `dwp.bench_models`: считать инференс заново на каждой
    развилке перебора незачем, вероятности от правила не зависят.
    """
    p = path or PRED_CACHE
    if not p.exists():
        raise FileNotFoundError(
            f"нет кэша вероятностей {p}.\n"
            f"Что делать: `python -m dwp.bench_models` — он его и пишет.")
    d = np.load(p, allow_pickle=True)
    if key not in d:
        have = [k for k in d.files if k.startswith("p__")]
        raise KeyError(f"в кэше нет {key}; есть: {', '.join(have)}")
    pv = d[key]
    y = d["y"]
    mid = d["match_id"]
    mn = d["minute"]
    out = []
    for m in np.unique(mid):
        sel = mid == m
        order = np.argsort(mn[sel], kind="stable")
        out.append({
            "match_id": int(m),
            "minute": mn[sel][order].astype(float),
            "p": pv[sel][order].astype(float),
            "y": int(y[sel][order][0]),
            "half": _half(m),
        })
    return out


# --- метрики -------------------------------------------------------------

def smooth_cache(rows: list[dict], half_lives) -> dict:
    """Сглаженные ряды на все полураспады разом: {(match_id, hl): z}."""
    out = {}
    for r in rows:
        for hl in half_lives:
            out[(r["match_id"], float(hl))] = smooth(r["minute"], r["p"], hl)
    return out


def evaluate(rows: list[dict], rule: Rule, zc: dict | None = None) -> dict:
    """Что даёт правило.

    Главное число — `acc_commit`: доля МАТЧЕЙ, в которых названная
    сторона оказалась победителем. Наблюдение на матч ровно одно, они
    независимы, поэтому интервал считается прямо здесь и без оговорок.

    `acc_hold` (доля верных МИНУТ на экране) оставлена для сравнения с
    сырым процентом: у неподвижного вердикта она равна доле верных
    матчей, взвешенной длиной показа, и одна она картину не описывает.
    """
    tot = right = 0
    ok_commit = []
    minutes_at = []
    confs_at = []
    n_no_commit = 0
    per_bin: dict[int, list[int]] = {}
    for r in rows:
        z = zc.get((r["match_id"], float(rule.half_life))) if zc else None
        got = run_rule(r["minute"], r["p"], rule, z=z)
        want = 1 if r["y"] == 1 else -1
        if not got["committed"]:
            n_no_commit += 1
            continue
        hit = int(got["commit_side"] == want)
        ok_commit.append(hit)
        minutes_at.append(got["commit_minute"])
        confs_at.append(got["commit_conf"])
        live_sel = got["side"] != 0
        n = int(live_sel.sum())
        tot += n
        right += n if hit else 0
        b = int(got["commit_minute"] // 2) * 2
        per_bin.setdefault(b, []).append(hit)
    n_com = len(ok_commit)
    acc = float(np.mean(ok_commit)) if n_com else float("nan")
    lo, hi = _wilson(acc, n_com) if n_com else (float("nan"), float("nan"))
    return {
        "rule": asdict(rule),
        "acc_commit": acc,
        "acc_commit_lo": lo, "acc_commit_hi": hi,
        "n_commit": n_com,
        "commit_rate": n_com / len(rows) if rows else float("nan"),
        "n_no_commit": n_no_commit,
        "commit_minute_p50": float(np.median(minutes_at)) if n_com else float("nan"),
        "commit_minute_mean": float(np.mean(minutes_at)) if n_com else float("nan"),
        "commit_conf_mean": float(np.mean(confs_at)) if n_com else float("nan"),
        "acc_hold": right / tot if tot else float("nan"),
        "n_rows": tot,
        "n_matches": len(rows),
        "flips_mean": 0.0,          # по построению
        "by_commit_minute": {str(k): [float(np.mean(v)), len(v)]
                             for k, v in sorted(per_bin.items())},
    }


def naive_baseline(rows: list[dict], t_lock: float = 0.0) -> dict:
    """С чем сравнивать: «сторона по сырому проценту, без сглаживания».

    Это то, что панель показывала бы без вердикта — кто крупнее написан.
    Число смен у него и есть та самая дёрганость.
    """
    tot = right = 0
    flips = []
    for r in rows:
        sel = r["minute"] >= t_lock
        if not sel.any():
            continue
        side = np.where(r["p"][sel] >= 0.5, 1, -1)
        want = 1 if r["y"] == 1 else -1
        tot += len(side)
        right += int((side == want).sum())
        flips.append(int((np.diff(side) != 0).sum()))
    return {
        "acc_hold": right / tot if tot else float("nan"),
        "n_rows": tot, "n_matches": len(flips),
        "flips_mean": float(np.mean(flips)) if flips else float("nan"),
        "flips_zero": float(np.mean(np.asarray(flips) == 0)) if flips else float("nan"),
    }


# --- таблица «как часто такой вердикт сбывался» --------------------------

def _bin(v: float, edges) -> int | None:
    for i in range(len(edges) - 1):
        if edges[i] <= v < edges[i + 1]:
            return i
    return None


def _wilson(ph: float, n: int, z: float = 1.96) -> tuple[float, float]:
    """Интервал Вилсона для доли `ph` на `n` независимых наблюдениях."""
    if n <= 0:
        return (float("nan"), float("nan"))
    ph = min(1.0, max(0.0, float(ph)))
    d = 1 + z * z / n
    c = (ph + z * z / (2 * n)) / d
    half = z * float(np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n))) / d
    return (float(max(0.0, c - half)), float(min(1.0, c + half)))


def _join_minute(match_id: int, minutes: np.ndarray) -> float | None:
    """Минута, с которой «подключились» к матчу, — для замера середины.

    Выбирается детерминированно по match_id: замер должен повторяться, а
    не зависеть от прогона. Берём точку между четвертью и тремя четвертями
    матча — раньше это почти начало, позже вердикт уже не нужен.
    """
    last = float(minutes[-1])
    if last < 12.0:
        return None
    h = int(hashlib.sha1(f"join:{int(match_id)}".encode("ascii")).hexdigest()[:8], 16)
    lo, hi = 0.25 * last, 0.75 * last
    return lo + (hi - lo) * ((h % 1000) / 999.0)


def _commit_of(r: dict, rule: Rule, join: float | None) -> tuple | None:
    """(минута, уверенность, попал ли) для одного матча. None — вердикта нет."""
    mn, p = r["minute"], r["p"]
    if join is not None:
        sel = mn >= join
        if int(sel.sum()) < 3:
            return None
        mn, p = mn[sel], p[sel]
    got = run_rule(mn, p, rule, first_minute=float(mn[0]))
    if not got["committed"]:
        return None
    want = 1 if r["y"] == 1 else -1
    return (float(got["commit_minute"]), float(got["commit_conf"]),
            int(got["commit_side"] == want))


def build_hitrate(rows: list[dict], rule: Rule) -> dict:
    """Клетки (минута вердикта) x (уверенность) -> доля сбывшихся.

    ДВЕ ТАБЛИЦЫ, И ЭТО НЕ ИЗБЫТОЧНОСТЬ. Панель включают не только с
    нулевой минуты: чаще всего матч выбирают из списка уже идущим. Тогда
    сглаженной истории нет, окно наблюдения отсчитывается от подключения,
    и вердикт выносится позже и на куда более высокой уверенности — на
    живом матче, подключившись на 12-й минуте, правило назвало сторону на
    14-й с уверенностью 82%. В таблице «с начала» такой клетки нет вовсе,
    и панель честно писала «выборки нет» — на самой обычной для себя
    ситуации.

    Поэтому клеток два набора: `s` — матч смотрели с начала, `j` — к
    матчу подключились в середине. Каждый матч даёт в каждый набор РОВНО
    ОДНО наблюдение, поэтому наблюдения независимы и интервал Вилсона
    считается без оговорок. Смешивать наборы нельзя: это разные
    популяции, и доля в них разная.

    Считается ТОЛЬКО на половине B, которую перебор не видел. Клетки
    меньше MIN_CELL МАТЧЕЙ выбрасываются целиком: показать «100% на семи
    матчах» хуже, чем не показать ничего.
    """
    acc: dict[str, list[int]] = {}
    for r in rows:
        for mode, join in (("s", None),
                           ("j", _join_minute(r["match_id"], r["minute"]))):
            if mode == "j" and join is None:
                continue
            got = _commit_of(r, rule, join)
            if got is None:
                continue
            at, conf, hit = got
            ci = _bin(conf, CONF_BINS)
            if ci is None:
                continue
            if mode == "s":
                mi = _bin(at, COMMIT_MINUTE_BINS)
                if mi is None:
                    continue
                key = f"s:{mi}:{ci}"
            else:
                # У подключения в середине минута вердикта — это почти
                # ровно минута подключения плюс окно наблюдения, то есть
                # свойство зрителя, а не матча. Делить по ней клетки
                # значит дробить выборку по шуму; условие здесь — только
                # уверенность.
                key = f"j:*:{ci}"
            acc.setdefault(key, []).append(hit)
    cells = {}
    for k, v in acc.items():
        if len(v) < MIN_CELL:
            continue
        n = len(v)
        ph = float(sum(v)) / n
        lo, hi = _wilson(ph, n)
        cells[k] = {"hit": ph, "n_matches": n, "lo": lo, "hi": hi}
    return {
        "cells": cells,
        "minute_bins": list(COMMIT_MINUTE_BINS),
        "conf_bins": list(CONF_BINS),
        "min_cell": MIN_CELL,
    }


def hitrate(table: dict | None, minute: float, conf: float,
            joined_late: bool = False) -> dict | None:
    """Клетка таблицы под ТОТ момент, когда вердикт был вынесен.

    Именно под момент фиксации, а не под текущую минуту: вердикт больше
    не меняется, значит и подпись под ним меняться не должна.

    `joined_late` выбирает набор клеток: подключились к идущему матчу или
    смотрели с начала. Это разные популяции (см. build_hitrate), и брать
    для одной долю, посчитанную на другой, значило бы подписать вердикт
    чужим числом. None — клетки нет, врать нечем.
    """
    if not table or minute is None or conf is None:
        return None
    if minute != minute or conf != conf:
        return None
    mi = _bin(float(minute), table.get("minute_bins") or COMMIT_MINUTE_BINS)
    ci = _bin(float(conf), table.get("conf_bins") or CONF_BINS)
    if mi is None or ci is None:
        return None
    cells = table.get("cells") or {}
    if joined_late:
        cell = cells.get(f"j:*:{ci}")
        mode, mrange = "j", None
    else:
        cell = cells.get(f"s:{mi}:{ci}")
        mode = "s"
        mrange = [table["minute_bins"][mi], table["minute_bins"][mi + 1]]
    if not cell:
        return None
    return dict(cell, mode=mode, minute=mrange,
                conf=[table["conf_bins"][ci], min(1.0, table["conf_bins"][ci + 1])])


# --- боевой путь ---------------------------------------------------------

def load_table(path: Path | None = None) -> dict | None:
    p = path or TABLE_PATH
    if not p.exists():
        return None
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return d if isinstance(d, dict) and d.get("rule") else None


def rule_from(table: dict | None) -> Rule:
    r = (table or {}).get("rule") or {}
    return Rule(t_open=float(r.get("t_open", 7.0)),
                half_life=float(r.get("half_life", 1.0)),
                theta_p=float(r.get("theta_p", 0.65)),
                t_force_after=float(r.get("t_force_after", 6.0)))


def live_verdict(table: dict | None, minutes, ps,
                 names: tuple[str, str] | None = None) -> dict | None:
    """Вердикт для панели по истории опросов.

    На вход идут ровно те `mhist`/`phist`, что копит `live.LiveTracker`:
    второй реализации инференса быть не должно, и второй истории тоже.

    Всё, что возвращается после фиксации, ПОСТОЯННО в пределах матча:
    сторона, минута, уверенность в момент фиксации и доля попаданий для
    такой позиции. Панели нечего перерисовывать, и это не украшение —
    это и есть требование «ни разу не колебаться».
    """
    if table is None:
        return None
    mn = np.asarray([m for m, p in zip(minutes, ps)
                     if m == m and p == p], dtype=float)
    pv = np.asarray([p for m, p in zip(minutes, ps)
                     if m == m and p == p], dtype=float)
    if len(mn) == 0:
        return None
    rule = rule_from(table)
    got = run_rule(mn, pv, rule, first_minute=float(mn[0]))
    side = int(got["commit_side"])
    # Клетка берётся по МОМЕНТУ ФИКСАЦИИ. Передавать сюда `table` целиком
    # было ошибкой прежней редакции: сетка бинов совпадала с умолчаниями
    # модуля, поэтому бины считались верно, а клетка не находилась
    # НИКОГДА, и панель молча писала «выборки нет».
    # Подключились к идущему матчу или смотрели с начала — разные
    # популяции, и доля у них разная. Признак тот же, по которому
    # сдвигается окно наблюдения: первая минута истории.
    joined_late = float(mn[0]) > 0.5
    hr = (hitrate(table.get("hitrate"), got["commit_minute"],
                  got["commit_conf"], joined_late=joined_late)
          if got["committed"] else None)
    out = {
        "side": ("radiant" if side > 0 else "dire") if side else None,
        "name": None,
        "committed": bool(got["committed"]),
        "commit_minute": (None if got["commit_minute"] != got["commit_minute"]
                          else float(got["commit_minute"])),
        "commit_conf": (None if got["commit_conf"] != got["commit_conf"]
                        else float(got["commit_conf"])),
        "minute": float(mn[-1]),
        # Насколько близко к порогу прямо сейчас — только пока вердикта
        # нет. После фиксации это число не показывается вовсе: оно
        # шевелится, а вердикт — нет, и рядом они бы спорили.
        "progress": (None if got["committed"] else
                     max(0.0, min(1.0, (float(got["conf"][-1]) - 0.5)
                                  / max(1e-9, rule.theta_p - 0.5)))),
        "open_at": float(got["open_at"]),
        "force_at": float(got["force_at"]),
        "hit": hr,
        "joined_late": joined_late,
        "rule": asdict(rule),
        "measured": table.get("measured"),
        "min_cell": (table.get("hitrate") or {}).get("min_cell", MIN_CELL),
    }
    if side and names:
        out["name"] = names[0] if side > 0 else names[1]
    return out


# --- подбор --------------------------------------------------------------

def tune(rows_a: list[dict], t_open: float | None = 7.0,
         budget: float = DEFAULT_BUDGET,
         min_commit_rate: float = MIN_COMMIT_RATE,
         verbose: bool = True) -> tuple[Rule, list[dict], dict]:
    """Перебор сетки на половине A.

    КРИТЕРИЙ. Смен стороны больше нет, значит и оптимизировать по ним
    нечего — прежний двухступенчатый критерий («не хуже лучшего на
    tolerance, дальше самый неподвижный») потерял смысл целиком. Теперь
    оптимизируется прямо то, что заказчик и просил: **доля матчей, в
    которых названная сторона выиграла**.

    Но у этой величины есть вырожденный максимум: подними порог до 0.95 и
    крайний срок до тридцатой минуты — вердикт будет выноситься, когда
    матч уже решён, и сбываться в 95% случаев. Пользы от такого вердикта
    ноль. Поэтому максимум берётся с двумя ограничениями:

        commit_rate >= min_commit_rate   вердикт обязан появляться почти
                                         всегда, а не когда удобно
        медиана минуты <= budget         «посмотреть семь минут» — это
                                         требование, а не пожелание

    Оба ограничения — договор, а не измерение, поэтому вынесены в ключи
    `--budget` и `--min-commit-rate`. Цену любого выбора показывает
    `--frontier`.
    """
    zc = smooth_cache(rows_a, GRID_HALF_LIFE)
    t_grid = (float(t_open),) if t_open is not None else GRID_T_OPEN
    got = []
    for t in t_grid:
        for hl in GRID_HALF_LIFE:
            for tp in GRID_THETA_P:
                for tf in GRID_T_FORCE:
                    r = Rule(t_open=t, half_life=hl, theta_p=tp,
                             t_force_after=tf)
                    got.append(evaluate(rows_a, r, zc))
    ok = [m for m in got
          if m["commit_rate"] >= min_commit_rate
          and m["commit_minute_p50"] <= budget]
    if not ok:
        # Ограничения несовместимы с сеткой — молча подсунуть «лучшее из
        # оставшегося» нельзя, это будет не то правило, о котором просили.
        raise SystemExit(
            f"ОШИБКА: ни один набор не укладывается в ограничения "
            f"(вердикт хотя бы в {min_commit_rate:.0%} матчей, медиана "
            f"минуты <= {budget:g}). Ослабьте --budget или "
            f"--min-commit-rate.")
    ok.sort(key=lambda m: (-m["acc_commit"], m["commit_minute_p50"]))
    best = Rule(**ok[0]["rule"])
    by_acc = sorted(got, key=lambda m: -m["acc_commit"])
    info = {
        "n_grid": len(got), "budget": budget,
        "min_commit_rate": min_commit_rate,
        "n_within": len(ok),
        "unconstrained_acc": by_acc[0]["acc_commit"],
        "unconstrained_rule": by_acc[0]["rule"],
        "unconstrained_p50": by_acc[0]["commit_minute_p50"],
        "chosen_acc": ok[0]["acc_commit"],
        "chosen_p50": ok[0]["commit_minute_p50"],
    }
    if verbose:
        print(f"  перебрано наборов: {len(got)}")
        print(f"  в ограничениях    : {len(ok)}")
        u = by_acc[0]
        print(f"  БЕЗ ограничений   : сбывается {u['acc_commit']:.4f}, "
              f"но медиана минуты {u['commit_minute_p50']:.1f} "
              f"(порог {u['rule']['theta_p']:.2f}) — вердикт к тому времени "
              f"уже не нужен")
        print(f"  выбрано           : сбывается {ok[0]['acc_commit']:.4f}, "
              f"медиана минуты {ok[0]['commit_minute_p50']:.1f}")
    return best, ok, info


def main(argv: list[str] | None = None) -> int:
    _safe_stdout()
    ap = argparse.ArgumentParser(
        description="Вердикт: одна сторона на весь матч. Подбор и замер.")
    ap.add_argument("--tune", action="store_true",
                    help="подобрать правило на половине A и померить на B")
    ap.add_argument("--show", action="store_true",
                    help="что подобрано (без пересчёта)")
    ap.add_argument("--frontier", action="store_true",
                    help="чем платит терпение и чем спешка")
    ap.add_argument("--t-open", type=float, default=7.0,
                    help="с какой минуты вердикт вообще возможен "
                         "(по умолчанию 7)")
    ap.add_argument("--tune-t-open", action="store_true",
                    help="подбирать и момент открытия тоже")
    ap.add_argument("--budget", type=float, default=DEFAULT_BUDGET,
                    help="крайняя МЕДИАНА минуты вердикта при подборе")
    ap.add_argument("--min-commit-rate", type=float, default=MIN_COMMIT_RATE,
                    help="в какой доле матчей вердикт обязан появиться")
    ap.add_argument("--model", default="p__ENSEMBLE",
                    help="какой ряд вероятностей из кэша брать")
    ap.add_argument("--out", type=Path, default=TABLE_PATH)
    args = ap.parse_args(argv)

    if args.show and not (args.tune or args.frontier):
        t = load_table(args.out)
        if not t:
            print(f"Таблицы {args.out} нет. `python -m dwp.verdict --tune`")
            return 1
        print(json.dumps(t.get("rule"), ensure_ascii=False, indent=2))
        print(json.dumps(t.get("measured"), ensure_ascii=False, indent=2))
        print(f"клеток в таблице попаданий: "
              f"{len(t.get('hitrate', {}).get('cells', {}))}")
        return 0

    try:
        rows = load_cache(key=args.model)
    except (FileNotFoundError, KeyError) as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    a = [r for r in rows if r["half"] == "A"]
    b = [r for r in rows if r["half"] == "B"]
    print("=" * 78)
    print(f"Слепой холдаут, ряд {args.model}: {len(rows)} матчей")
    print(f"  половина A (подбор правила): {len(a)} матчей")
    print(f"  половина B (замер и таблица): {len(b)} матчей")
    print("=" * 78)
    if len(a) < 50 or len(b) < 50:
        print("ОШИБКА: половин слишком мало, подбирать не на чем.", file=sys.stderr)
        return 2

    print("\nПодбор на половине A:")
    t0 = time.time()
    best, grid, info = tune(a, t_open=None if args.tune_t_open else args.t_open,
                            budget=args.budget,
                            min_commit_rate=args.min_commit_rate)
    print(f"  занял {time.time() - t0:.0f} с")
    print(f"\n  ВЫБРАНО: ждать с {best.t_open:g}-й минуты, полураспад "
          f"{best.half_life:g} мин, называть при уверенности "
          f"{best.theta_p:.2f}, крайний срок "
          + (f"{best.t_force:g} мин" if best.t_force_after <= 1e6
             else "не назначен (ждём уверенности сколько нужно)"))

    print("\n  Пять лучших наборов в ограничениях (на A):")
    print(f"    {'ждать с':>8}{'полурасп':>10}{'порог':>7}{'крайний':>9}"
          f"{'сбылось':>9}{'медиана':>9}{'доля вердиктов':>16}")
    for m in grid[:5]:
        r = m["rule"]
        print(f"    {r['t_open']:>8.0f}{r['half_life']:>10.2f}"
              f"{r['theta_p']:>7.2f}{_force_str(r):>9}"
              f"{m['acc_commit']:>9.4f}{m['commit_minute_p50']:>9.1f}"
              f"{m['commit_rate']:>16.3f}")

    print("\n" + "=" * 78)
    print("ЗАМЕР НА ПОЛОВИНЕ B — эти матчи перебор не видел")
    print("=" * 78)
    mb = evaluate(b, best)
    nb = naive_baseline(b, t_lock=best.t_open)
    print(f"  {'':<30}{'вердикт':>12}{'сырой процент':>16}")
    print("  " + "-" * 58)
    print(f"  {'СБЫЛСЯ, доля матчей':<30}{mb['acc_commit']:>12.4f}{'':>16}")
    print(f"  {'  95% интервал':<30}"
          f"{'[' + format(mb['acc_commit_lo'], '.3f') + ', '
             + format(mb['acc_commit_hi'], '.3f') + ']':>12}{'':>16}")
    print(f"  {'смен стороны за матч':<30}{0.0:>12.2f}{nb['flips_mean']:>16.2f}")
    print(f"  {'верных минут на экране':<30}{mb['acc_hold']:>12.4f}"
          f"{nb['acc_hold']:>16.4f}")
    print(f"  {'вердикт вынесен, доля матчей':<30}{mb['commit_rate']:>12.3f}{'':>16}")
    print(f"  {'медиана минуты вердикта':<30}{mb['commit_minute_p50']:>12.1f}{'':>16}")
    print(f"  {'средняя минута вердикта':<30}{mb['commit_minute_mean']:>12.1f}{'':>16}")
    print(f"\n  матчей {mb['n_matches']}, из них с вердиктом {mb['n_commit']}, "
          f"без вердикта {mb['n_no_commit']}")

    print("\n  Сбываемость по минуте вердикта (половина B):")
    print(f"    {'минуты':>9}{'сбылось':>9}{'матчей':>9}")
    for k, (v, n) in mb["by_commit_minute"].items():
        print(f"    {k + '-' + str(int(k) + 2):>9}{v:>9.3f}{n:>9}")

    if args.frontier:
        print("\n" + "=" * 78)
        print("ЧЕМ ПЛАТИТ ТЕРПЕНИЕ (замер на B, момент открытия и полураспад "
              "выбранные)")
        print("=" * 78)
        print(f"    {'порог':>7}{'крайний':>9}{'сбылось':>9}{'медиана мин':>13}"
              f"{'вердиктов':>11}")
        for tp in GRID_THETA_P:
            for tf in (0.0, 6.0, 13.0):
                r = Rule(t_open=best.t_open, half_life=best.half_life,
                         theta_p=tp, t_force_after=tf)
                m = evaluate(b, r)
                print(f"    {tp:>7.2f}{r.t_force:>9.0f}{m['acc_commit']:>9.4f}"
                      f"{m['commit_minute_p50']:>13.1f}"
                      f"{m['commit_rate']:>11.3f}")
        print("\n  Порог 0.50 с крайним сроком, равным открытию, — это\n"
              "  «назвать сторону ровно на седьмой минуте, какая бы она ни\n"
              "  была». Всё, что ниже, — плата за ожидание уверенности:\n"
              "  вердикт точнее, но появляется позже и не в каждом матче.")

        # Главная развилка проекта, и в таблице выше она НЕ ВИДНА: там у
        # каждого варианта стоит крайний срок, то есть на трудном матче
        # вердикт всё равно выдавливается силой. А если крайнего срока нет
        # и ждать уверенности столько, сколько нужно, — вердикт становится
        # заметно вернее ценой того, что в части матчей его не будет вовсе.
        print("\n" + "=" * 78)
        print("ЕСЛИ ЖДАТЬ УВЕРЕННОСТИ СКОЛЬКО НУЖНО (замер на B, без крайнего "
              "срока)")
        print("=" * 78)
        print(f"    {'порог':>7}{'сбылось':>9}{'95% интервал':>18}"
              f"{'вердиктов':>11}{'медиана мин':>13}{'матчей':>9}")
        for tp in GRID_THETA_P:
            if tp < 0.60:
                continue
            r = Rule(t_open=best.t_open, half_life=best.half_life,
                     theta_p=tp, t_force_after=1e9)
            m = evaluate(b, r)
            n = int(m["n_commit"])
            lo, hi = _wilson(m["acc_commit"], n) if n else (float("nan"),) * 2
            print(f"    {tp:>7.2f}{m['acc_commit']:>9.4f}"
                  f"{f'[{lo:.3f}, {hi:.3f}]':>18}{m['commit_rate']:>11.3f}"
                  f"{m['commit_minute_p50']:>13.1f}{n:>9}")
        print("\n  Читать так: порог — это НЕ точность, а требование к\n"
              "  сглаженной уверенности. Чем он выше, тем вернее вердикт и\n"
              "  тем в большей доле матчей его не будет вовсе. Нынешний\n"
              "  выбор (0.60) продиктован бюджетом медианной минуты\n"
              f"  (--budget {DEFAULT_BUDGET:g}); поднять планку — это\n"
              "      python -m dwp.verdict --tune --budget 20 "
              "--min-commit-rate 0.95")

        print("\n" + "=" * 78)
        print("ЧЕМ ПЛАТИТ СПЕШКА (замер на B, порог и крайний срок выбранные)")
        print("=" * 78)
        print(f"    {'ждать с':>8}{'сбылось':>9}{'медиана мин':>13}"
              f"{'вердиктов':>11}")
        for t in GRID_T_OPEN:
            r = Rule(t_open=t, half_life=best.half_life,
                     theta_p=best.theta_p, t_force_after=best.t_force_after)
            m = evaluate(b, r)
            print(f"    {t:>8.0f}{m['acc_commit']:>9.4f}"
                  f"{m['commit_minute_p50']:>13.1f}{m['commit_rate']:>11.3f}")
        print("\n  Это прямой ответ на «пусть посмотрит семь минут и скажет\n"
              "  точно»: раньше — не точнее, а просто раньше.")

    print("\nСтрою таблицу попаданий на половине B…")
    hr = build_hitrate(b, best)
    print(f"  клеток заполнено: {len(hr['cells'])} "
          f"(порог {MIN_CELL} матчей)")
    for k, c in sorted(hr["cells"].items()):
        mode, mi, ci = k.split(":")
        ci = int(ci)
        where = ("с начала, минуты "
                 f"{COMMIT_MINUTE_BINS[int(mi)]:g}-"
                 f"{COMMIT_MINUTE_BINS[int(mi)+1]:g}" if mode == "s"
                 else "подключились в середине      ")
        print(f"    {where:<28} уверенность "
              f"{CONF_BINS[ci]:.2f}-{min(1.0, CONF_BINS[ci+1]):.2f}: "
              f"сбылось {c['hit']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}] "
              f"на {c['n_matches']} матчах")

    out = {
        "rule": asdict(best),
        "hitrate": hr,
        "measured": {
            "half_a_matches": len(a), "half_b_matches": len(b),
            "acc_commit": mb["acc_commit"],
            "acc_commit_lo": mb["acc_commit_lo"],
            "acc_commit_hi": mb["acc_commit_hi"],
            "n_commit": mb["n_commit"],
            "commit_rate": mb["commit_rate"],
            "commit_minute_p50": mb["commit_minute_p50"],
            "commit_minute_mean": mb["commit_minute_mean"],
            "acc_hold": mb["acc_hold"],
            "flips_mean": 0.0,
            "naive_acc_hold": nb["acc_hold"],
            "naive_flips_mean": nb["flips_mean"],
            "n_rows": mb["n_rows"], "n_matches": mb["n_matches"],
            "by_commit_minute": mb["by_commit_minute"],
            "model": args.model,
        },
        "tuning": info,
        "built_at": time.time(),
        "holdout_permille": HO.HOLDOUT_PERMILLE,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"  записано: {args.out}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
