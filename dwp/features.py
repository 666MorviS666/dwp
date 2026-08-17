"""Парсинг матча OpenDota и построение признаков.

Главный принцип модуля: если поле не найдено или событие не распознано —
это попадает в отчёт (`ParsedObjectives.unparsed`, флаги `*_ok`), а не
превращается в тихий ноль. Ноль в признаке по вышкам неотличим от
«вышек не теряли» и стоит модели одного из сильнейших сигналов, а по
метрикам это почти не видно.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

# --- Признаки стейт-модели ----------------------------------------------

BASE_STATE_FEATURES = [
    "minute",
    "gold_adv",
    "gold_adv_d1",
    "gold_adv_slope5",
    "radiant_towers_lost",
    "dire_towers_lost",
    "tower_adv",
    "rax_adv",
    "roshan_radiant",
    "roshan_dire",
    "minutes_since_roshan",
    "tormentor_radiant",
    "tormentor_dire",
    "draft_logit",
]
XP_STATE_FEATURES = ["xp_adv", "xp_adv_d1"]

# Признаки сверх базовых. Включаются флагом --extra-features, чтобы
# прирост можно было замерить через dwp.compare, а не принять на веру.
EXTRA_STATE_FEATURES = [
    "kills_adv",        # разница по фрагам: команда может быть ровна по золоту
    "kills_adv_d5",     # и при этом выигрывать все замесы
    "nw_top_adv",       # нетворс богатейшего игрока: 10к на керри и 10к,
    "nw_conc_adv",      # размазанные по саппортам — разные ситуации
    "bb_used_radiant",  # байбэки: в поздней игре наличие выкупа решает
    "bb_used_dire",     # больше, чем пара тысяч золота
    "bb_oncd_adv",
]

# Из них в GetRealtimeStats доступны kill_count и net_worth на игрока,
# то есть фраги и распределение нетворса считаются и в лайве.
# Байбэков в лайв-ответе нет.
EXTRA_LIVE_UNAVAILABLE = ["bb_used_radiant", "bb_used_dire", "bb_oncd_adv"]

# Перевес по золоту как ДОЛЯ от суммарного нетворса обеих команд.
# Зачем: 5000 золота на 20-й минуте это 6% общего нетворса, а на 70-й —
# около 1%. Дерево могло бы выучить это взаимодействие с `minute` само,
# но при min_data_in_leaf=200 и нескольких сотнях строк после 60-й минуты
# на такой лист данных не хватает. Нормировка вносит это знание руками.
# В лайве знаменатель есть: сумма teams[].net_worth.
NORM_STATE_FEATURES = ["gold_adv_norm", "gold_adv_norm_slope5"]

# Признаки, у которых определение в обучении и в лайве совпадает ТОЧНО,
# до последней единицы. Заведены после того, как выяснилось (см. README,
# раздел про gold_adv), что сильнейший признак модели — единственный, у
# которого определения расходятся: обучение берёт добытое золото, лайв
# подаёт нетворс, и в концовке с буйбэками это стоит до 22 п.п. на экране.
# Здесь расхождению взяться неоткуда:
#   уровни  xp_t -> уровень по config.XP_TO_LEVEL  <->  players[].level
#   ластхиты          lh_t                         <->  players[].lh_count
#   денаи             dn_t                         <->  players[].denies_count
#   вышки t3   имя здания в building_kill          <->  tier у стоящих зданий
# Включаются флагом --exact-features, чтобы прирост меряли, а не верили.
EXACT_STATE_FEATURES = [
    "radiant_t3_lost",   # t3 открывает бараки; tower_adv считает t1 и t3
    "dire_t3_lost",      # одинаково, а это разные позиции
    "t3_adv",
    "level_adv",         # опыт, которого у лайв-модели не было вовсе
    "lh_adv",
    "dn_adv",
]

BUYBACK_COOLDOWN_MIN = 8.0

# Нижняя граница знаменателя: на нулевой минуте сумма gold_t бывает ровно
# нулём (стартовое золото в неё не входит), и деление уходит в бесконечность.
# Порядок величины — стартовое золото десяти героев. Точное число неважно,
# это сглаживание знаменателя, а не физика.
NW_FLOOR = 6000.0


# Признаки, которых GetRealtimeStats не отдаёт (проверено на живом
# ответе): Рошана в нём нет вовсе — ни таймера, ни событий; Торментора
# тоже; XP заменён уровнями, а достраивать его из уровней нельзя.
# Модель, обученная с ними, в бою получает по ним NaN на каждой строке —
# то есть учится опираться на то, чего в проде никогда не будет.
LIVE_UNAVAILABLE = [
    "roshan_radiant", "roshan_dire", "minutes_since_roshan",
    "tormentor_radiant", "tormentor_dire",
]


GOLD_NORM_MODES = ("off", "add", "replace")
# Что вытесняется в режиме replace: абсолютный перевес и его наклон.
# gold_adv_d1 остаётся — минутный прирост в абсолютных числах читается
# одинаково на любой стадии, нормировать его не за чем.
GOLD_NORM_REPLACES = ["gold_adv", "gold_adv_slope5"]


def state_features(use_xp: bool, live_only: bool = False,
                   extra: bool = False, gold_norm: str = "off",
                   exact: bool = False) -> list[str]:
    # Порядок фиксирован: LightGBM запоминает имена, и рассинхрон между
    # обучением и лайвом даёт молча неправильные предсказания.
    if gold_norm not in GOLD_NORM_MODES:
        raise ValueError(f"gold_norm={gold_norm!r}, ожидалось одно из {GOLD_NORM_MODES}")
    feats = BASE_STATE_FEATURES + (XP_STATE_FEATURES if use_xp else [])
    if extra:
        feats = feats + EXTRA_STATE_FEATURES
    if exact:
        # level_adv и xp_adv — одно и то же с точностью до округления,
        # и вместе они только дробят сплиты. При use_xp побеждает xp_adv:
        # он точнее и в офлайн-модели уже проверен.
        feats = feats + [f for f in EXACT_STATE_FEATURES
                         if not (use_xp and f == "level_adv")]
    if gold_norm == "replace":
        feats = [f for f in feats if f not in set(GOLD_NORM_REPLACES)]
    if gold_norm in ("add", "replace"):
        feats = feats + NORM_STATE_FEATURES
    if live_only:
        drop = set(LIVE_UNAVAILABLE) | set(EXTRA_LIVE_UNAVAILABLE)
        feats = [f for f in feats if f not in drop]
    return feats


# --- Герои --------------------------------------------------------------

def load_heroes(path: Path | None = None) -> tuple[list[int], dict[int, int], dict[int, str]]:
    """Возвращает (упорядоченный список hero_id, id->индекс, id->имя).

    Индексация идёт по отсортированному id, а не по позиции в файле:
    порядок в ответе /heroes не гарантирован, а индексы должны быть
    стабильны между обучением и инференсом.
    """
    path = path or config.HEROES_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Нет файла героев {path}. Запустите `python -m dwp.collect --heroes` "
            f"(нужна сеть) или `python -m dwp.synthetic`."
        )
    heroes = json.loads(path.read_text(encoding="utf-8"))
    ids = sorted(int(h["id"]) for h in heroes)
    names = {int(h["id"]): h.get("localized_name") or h.get("name") or str(h["id"])
             for h in heroes}
    return ids, {hid: i for i, hid in enumerate(ids)}, names


# --- Разбор objectives --------------------------------------------------

@dataclass
class ParsedObjectives:
    tower_kills: list[tuple[int, bool]] = field(default_factory=list)   # (минута, victim_is_radiant)
    # То же, но с ярусом: (минута, victim_is_radiant, tier 1..4). Отдельным
    # списком, а не третьим элементом в tower_kills, потому что ярус есть
    # только в новом формате событий (`building_kill` с именем здания), а
    # старый CHAT_MESSAGE_TOWER_KILL отдаёт одну лишь сторону. Замеряно на
    # 1500 матчах: новый формат в 99.9%, событий с ярусом 16728, без — ни
    # одного. Но полагаться на это нельзя, поэтому есть флаг tower_tier_ok.
    tower_kills_tier: list[tuple[int, bool, int]] = field(default_factory=list)
    rax_kills: list[tuple[int, bool]] = field(default_factory=list)
    roshan_kills: list[tuple[int, bool | None]] = field(default_factory=list)  # (минута, killer_is_radiant)
    tormentor_kills: list[tuple[int, bool | None]] = field(default_factory=list)
    unparsed: Counter = field(default_factory=Counter)
    fmt: str = "none"          # new / old / mixed / none
    objectives_present: bool = False   # список objectives вообще был
    tower_ok: bool = False     # вышки удалось разобрать
    tower_tier_ok: bool = False   # и у КАЖДОЙ известен ярус
    rax_ok: bool = False
    roshan_ok: bool = False
    tower_consistent: bool | None = None   # сверка с tower_status_*
    rax_consistent: bool | None = None


def _tier_from_key(key: str) -> int:
    """Ярус вышки из имени здания: npc_dota_goodguys_tower3_mid -> 3.
    Ноль означает «в имени яруса нет» — тогда признаки по ярусам станут NaN,
    а не сделают вид, что это первый ярус."""
    for tier in (1, 2, 3, 4):
        if f"tower{tier}" in key:
            return tier
    return 0


def _side_from_building_key(key: str) -> bool | None:
    """True = разрушено здание Radiant, False = Dire, None = не поняли."""
    if "goodguys" in key:
        return True
    if "badguys" in key:
        return False
    return None


def team_is_radiant(team_value) -> bool | None:
    """Декодирует поле `team` в сторону: True = Radiant, False = Dire.

    Функция НЕ знает, кого обозначает эта сторона — жертву или убийцу.
    Это решает вызывающий по соответствующей константе в config. Раньше
    трактовка была зашита сюда параметром `meaning`, и на Рошане это дало
    двойное отрицание: счётчики Рошанов у сторон менялись местами, а по
    метрикам это было не видно (вклад признака ~0.5%).
    """
    try:
        team = int(team_value)
    except (TypeError, ValueError):
        return None
    if team == config.TEAM_RADIANT:
        return True
    if team == config.TEAM_DIRE:
        return False
    return None


def parse_objectives(match: dict,
                     tower_team_means: str | None = None) -> ParsedObjectives:
    res = ParsedObjectives()
    tower_team_means = tower_team_means or config.OLD_TOWER_KILL_TEAM_MEANS
    objectives = match.get("objectives")
    res.objectives_present = bool(objectives)
    if not objectives:
        res.unparsed["<objectives отсутствует или пуст>"] += 1
        return res

    saw_new = saw_old = False
    for o in objectives:
        if not isinstance(o, dict):
            res.unparsed["<элемент objectives не словарь>"] += 1
            continue
        otype = o.get("type")
        time_s = o.get("time")
        if otype is None or time_s is None:
            res.unparsed["<нет type или time>"] += 1
            continue
        minute = max(0, int(time_s) // 60)

        if otype == "building_kill":
            key = str(o.get("key", ""))
            side = _side_from_building_key(key)
            if side is None:
                res.unparsed[f"building_kill/{key}"] += 1
                continue
            saw_new = True
            if "rax" in key:
                res.rax_kills.append((minute, side))
            elif "tower" in key:
                res.tower_kills.append((minute, side))
                res.tower_kills_tier.append((minute, side, _tier_from_key(key)))
            elif "fort" in key:
                pass                      # трон — исход уже известен, в признаки не идёт
            else:
                res.unparsed[f"building_kill/{key}"] += 1

        elif otype == "CHAT_MESSAGE_TOWER_KILL":
            side = team_is_radiant(o.get("team"))
            if side is not None and tower_team_means != "victim":
                side = not side          # поле обозначает убийцу -> жертва наоборот
            if side is None:
                res.unparsed[f"CHAT_MESSAGE_TOWER_KILL/team={o.get('team')!r}"] += 1
                continue
            saw_old = True
            res.tower_kills.append((minute, side))

        elif otype == "CHAT_MESSAGE_BARRACKS_KILL":
            side = team_is_radiant(o.get("team"))
            if side is not None and tower_team_means != "victim":
                side = not side
            if side is None:
                # Встречается вариант, где стороны нет вовсе, а key —
                # битовая маска. Её раскладку я не проверял, поэтому
                # честно помечаем как неразобранное.
                res.unparsed[f"CHAT_MESSAGE_BARRACKS_KILL/key={o.get('key')!r}"] += 1
                continue
            saw_old = True
            res.rax_kills.append((minute, side))

        elif otype == "CHAT_MESSAGE_ROSHAN_KILL":
            killer = team_is_radiant(o.get("team"))
            if killer is not None and config.OLD_ROSHAN_KILL_TEAM_MEANS != "killer":
                killer = not killer      # поле обозначает жертву -> убийца наоборот
            res.roshan_kills.append((minute, killer))
            if killer is None:
                res.unparsed[f"CHAT_MESSAGE_ROSHAN_KILL/team={o.get('team')!r}"] += 1

        elif otype == "CHAT_MESSAGE_MINIBOSS_KILL":
            # Торментор. Золото за него уже сидит в radiant_gold_adv, но
            # аганимов шард в нём не отражается — это отдельный сигнал.
            # Конвенция team = убийца подтверждена ЖЁСТКО: на реальных
            # про-матчах team согласуется с player_slot в 137 из 137
            # событий, где слот присутствует.
            killer = team_is_radiant(o.get("team"))
            slot = o.get("player_slot")
            if slot is not None:
                by_slot = int(slot) < 128
                if killer is not None and by_slot != killer:
                    res.unparsed["MINIBOSS_KILL/team противоречит player_slot"] += 1
                killer = by_slot          # слот надёжнее: он однозначен
            res.tormentor_kills.append((minute, killer))
            if killer is None:
                res.unparsed[f"CHAT_MESSAGE_MINIBOSS_KILL/team={o.get('team')!r}"] += 1

        else:
            res.unparsed[str(otype)] += 1

    res.fmt = "mixed" if (saw_new and saw_old) else ("new" if saw_new else
                                                    ("old" if saw_old else "none"))
    res.tower_ok = len(res.tower_kills) > 0
    # Ярус нужен у ВСЕХ снесённых вышек: одна вышка с неизвестным ярусом
    # означает, что счётчик по ярусам занижен, а по какому именно ярусу —
    # неизвестно. Частично посчитанный признак хуже честного NaN.
    res.tower_tier_ok = (res.tower_ok
                         and len(res.tower_kills_tier) == len(res.tower_kills)
                         and all(t > 0 for _, _, t in res.tower_kills_tier))
    res.rax_ok = len(res.rax_kills) > 0
    res.roshan_ok = all(k is not None for _, k in res.roshan_kills)

    # Сверка с финальной битовой маской. Ровно эта проверка ловит
    # неверную трактовку поля `team` в старом формате: при перепутанных
    # сторонах счётчики не сойдутся почти во всех матчах.
    ts_r, ts_d = match.get("tower_status_radiant"), match.get("tower_status_dire")
    if res.tower_ok and isinstance(ts_r, int) and isinstance(ts_d, int):
        lost_r = sum(1 for _, v in res.tower_kills if v)
        lost_d = sum(1 for _, v in res.tower_kills if not v)
        exp_r = config.N_TOWERS_PER_SIDE - bin(ts_r).count("1")
        exp_d = config.N_TOWERS_PER_SIDE - bin(ts_d).count("1")
        res.tower_consistent = (lost_r == exp_r) and (lost_d == exp_d)

    bs_r, bs_d = match.get("barracks_status_radiant"), match.get("barracks_status_dire")
    if res.rax_ok and isinstance(bs_r, int) and isinstance(bs_d, int):
        lost_r = sum(1 for _, v in res.rax_kills if v)
        lost_d = sum(1 for _, v in res.rax_kills if not v)
        res.rax_consistent = (
            lost_r == config.N_RAX_PER_SIDE - bin(bs_r).count("1")
            and lost_d == config.N_RAX_PER_SIDE - bin(bs_d).count("1")
        )
    return res


# --- Стороны и драфт ----------------------------------------------------

def match_sides(match: dict) -> tuple[list[int], list[int]]:
    """(hero_id Radiant, hero_id Dire). Бросает ValueError, если сторону
    определить нечем — молча делить 10 игроков пополам нельзя."""
    players = match.get("players")
    if not players:
        raise ValueError(f"match {match.get('match_id')}: нет поля players")
    rad, dire = [], []
    for p in players:
        hid = p.get("hero_id")
        if not hid:
            raise ValueError(f"match {match.get('match_id')}: у игрока нет hero_id")
        slot = p.get("player_slot")
        if slot is not None:
            is_rad = int(slot) < 128
        elif "isRadiant" in p:
            is_rad = bool(p["isRadiant"])
        else:
            raise ValueError(
                f"match {match.get('match_id')}: нет ни player_slot, ни isRadiant")
        (rad if is_rad else dire).append(int(hid))
    if len(rad) != 5 or len(dire) != 5:
        raise ValueError(
            f"match {match.get('match_id')}: сторон {len(rad)}/{len(dire)}, ожидалось 5/5")
    return rad, dire


# --- Elo ----------------------------------------------------------------

def build_elo(matches: list[dict], k: float = 32.0, start: float = 1500.0
              ) -> tuple[dict[int, float], dict[int, float]]:
    """Хронологический Elo по team_id.

    Возвращает (match_id -> ПРЕД-матчевая разница рейтингов, team_id ->
    финальный рейтинг). Пред-матчевую разницу обязательно брать из первого
    словаря: подставить финальный рейтинг в разбор прошлого матча — это
    утечка из будущего и выход за диапазон, на котором училась модель.
    """
    order = sorted(matches, key=lambda m: (m.get("start_time") or 0, m["match_id"]))
    rating: dict[int, float] = {}
    pre: dict[int, float] = {}
    for m in order:
        r_id, d_id = m.get("radiant_team_id"), m.get("dire_team_id")
        r_id = int(r_id) if r_id else -1
        d_id = int(d_id) if d_id else -2
        rr = rating.get(r_id, start)
        dr = rating.get(d_id, start)
        pre[int(m["match_id"])] = rr - dr
        if m.get("radiant_win") is None or r_id < 0 or d_id < 0:
            continue                      # без team_id обновлять нечего
        exp_r = 1.0 / (1.0 + 10 ** ((dr - rr) / 400.0))
        outcome = 1.0 if m["radiant_win"] else 0.0
        rating[r_id] = rr + k * (outcome - exp_r)
        rating[d_id] = dr - k * (outcome - exp_r)
    return pre, rating


def match_accounts(match: dict) -> tuple[list[int], list[int]]:
    """account_id пятёрок Radiant и Dire. Пустые списки, если состав неполон.

    Неполный состав возвращается пустым намеренно: посчитать среднее по
    четверым значит молча занизить силу команды, а по метрикам такое не
    видно (соглашение «молчаливого нуля быть не должно»).
    """
    rad, dire = [], []
    for p in (match.get("players") or []):
        a = p.get("account_id")
        if not a:
            continue
        (rad if p.get("isRadiant") else dire).append(int(a))
    if len(rad) != 5 or len(dire) != 5:
        return [], []
    return rad, dire


def build_player_elo(matches: list[dict], k: float | None = None,
                     ) -> tuple[dict[int, float], dict[int, float]]:
    """Хронологический Elo по ИГРОКАМ (account_id).

    Возвращает (match_id -> ПРЕД-матчевая разница сил, account_id ->
    финальный рейтинг), в тех же единицах, что и `build_elo`, — чтобы
    `draft_matrix` не пришлось менять.

    ЗАЧЕМ ОН ВМЕСТО КОМАНДНОГО. Замер (шагающий хронологический протокол,
    10636 матчей вне холдаута) показал, что предматчевый сигнал почти
    весь сидит в рейтинге, а не в героях: только герои дают AUC 0.5465,
    герои с командным Elo — 0.6001, герои с рейтингом игроков — 0.6138.
    Разница по log loss −0.0049, 95% [−0.0072, −0.0026], то есть
    значимая. Командный рейтинг ПОВЕРХ игрокового не добавляет ничего
    (0.6131 против 0.6138), поэтому он и заменяется, а не дополняется.

    Почему по игрокам выходит лучше, видно из самих данных: 5080 игроков
    примерно по 25 матчей против 1090 команд примерно по 11; состав
    меняется, а team_id остаётся тем же; и account_id есть у 100% строк
    выгрузки, тогда как оба team_id — только у 95.4%.

    Сила команды — СРЕДНЕЕ рейтингов пятёрки (сумма замерена и хуже).
    Обновление общее на команду и одинаковое для всех пятерых: вклад
    отдельного игрока в исход из этих данных не виден, и приписывать его
    было бы выдумкой.
    """
    k = float(config.PLAYER_ELO_K if k is None else k)
    order = sorted(matches, key=lambda m: (m.get("start_time") or 0, m["match_id"]))
    rating: dict[int, float] = {}
    pre: dict[int, float] = {}
    for m in order:
        rad, dire = match_accounts(m)
        if not rad:
            pre[int(m["match_id"])] = 0.0
            continue
        sr = float(np.mean([rating.get(a, 0.0) for a in rad]))
        sd = float(np.mean([rating.get(a, 0.0) for a in dire]))
        pre[int(m["match_id"])] = sr - sd
        if m.get("radiant_win") is None:
            continue
        exp_r = 1.0 / (1.0 + 10 ** ((sd - sr) / 400.0))
        upd = k * ((1.0 if m["radiant_win"] else 0.0) - exp_r)
        for a in rad:
            rating[a] = rating.get(a, 0.0) + upd
        for a in dire:
            rating[a] = rating.get(a, 0.0) - upd
    return pre, rating


def player_elo_diff(ratings: dict, rad: list[int], dire: list[int]) -> float:
    """Разница сил по готовой таблице рейтингов. Единая точка для лайва.

    Ключи таблицы могут приехать из JSON строками — артефакт пишется
    пиклом, но кто-нибудь однажды сохранит его иначе, и молча получить
    ноль вместо рейтинга здесь дороже, чем лишний int().
    """
    if len(rad) != 5 or len(dire) != 5:
        return float("nan")

    def look(a: int) -> float:
        if a in ratings:
            return float(ratings[a])
        return float(ratings.get(str(a), 0.0))

    return float(np.mean([look(a) for a in rad])
                 - np.mean([look(a) for a in dire]))


def draft_matrix(matches: list[dict], id2idx: dict[int, int],
                 elo_pre: dict[int, float]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X = [+1 за героя Radiant, −1 за героя Dire ... , elo_diff/400]."""
    n = len(id2idx)
    X = np.zeros((len(matches), n + 1), dtype=np.float32)
    y = np.zeros(len(matches), dtype=np.int8)
    mids = np.zeros(len(matches), dtype=np.int64)
    for i, m in enumerate(matches):
        rad, dire = match_sides(m)
        for h in rad:
            if h in id2idx:
                X[i, id2idx[h]] += 1.0
        for h in dire:
            if h in id2idx:
                X[i, id2idx[h]] -= 1.0
        X[i, n] = (elo_pre.get(int(m["match_id"]), 0.0) / 400.0
                   * config.DRAFT_ELO_SCALE)
        y[i] = 1 if m["radiant_win"] else 0
        mids[i] = int(m["match_id"])
    return X, y, mids


def draft_feature_names(hero_ids: list[int], names: dict[int, str]) -> list[str]:
    return [names.get(h, str(h)) for h in hero_ids] + [
        f"elo_diff/400*{config.DRAFT_ELO_SCALE:g}"]


# --- Поигроковые ряды ---------------------------------------------------

@dataclass
class PlayerSeries:
    """Поминутные ряды по игрокам, разобранные из players[].

    Каждое поле либо разобрано, либо остаётся None, и тогда признаки на
    его основе становятся NaN. Ноль здесь означал бы «фрагов не было» или
    «нетворс нулевой», что для 30-й минуты ложь.
    """
    gold: np.ndarray | None = None        # [10, T+1] нетворс по игрокам
    is_radiant: np.ndarray | None = None  # [10] bool
    kills_r: np.ndarray | None = None     # [T+1] накопленные фраги Radiant
    kills_d: np.ndarray | None = None
    bb_r: np.ndarray | None = None        # [T+1] накопленные байбэки
    bb_d: np.ndarray | None = None
    bb_oncd_r: np.ndarray | None = None   # [T+1] игроков с выкупом на КД
    bb_oncd_d: np.ndarray | None = None
    # Ряды, у которых определение СОВПАДАЕТ с лайвом до последней единицы:
    # уровень из xp_t против players[].level, lh_t против lh_count,
    # dn_t против denies_count. Ради них всё это и заводится — см. README,
    # раздел про gold_adv: сильнейший признак модели единственный, у кого
    # определения расходятся, и вес стоит перенести на те, у кого не расходятся.
    level: np.ndarray | None = None       # [10, T+1] уровень по игрокам
    lh: np.ndarray | None = None          # [10, T+1] последние удары
    dn: np.ndarray | None = None          # [10, T+1] денаи
    notes: list[str] = field(default_factory=list)


def parse_player_series(match: dict, T: int) -> PlayerSeries:
    """Разбирает gold_t, kills_log и buyback_log из players[]."""
    res = PlayerSeries()
    players = match.get("players") or []
    if len(players) != 10:
        res.notes.append(f"игроков {len(players)}, ожидалось 10")
        return res

    sides = []
    for p in players:
        slot = p.get("player_slot")
        if slot is None:
            res.notes.append("нет player_slot")
            return res
        sides.append(int(slot) < 128)
    res.is_radiant = np.array(sides, dtype=bool)

    # --- нетворс по игрокам ---
    golds = []
    for p in players:
        g = p.get("gold_t")
        if not isinstance(g, list) or len(g) < 2:
            golds = []
            break
        arr = np.full(T + 1, np.nan)
        n = min(len(g), T + 1)
        arr[:n] = np.asarray(g[:n], dtype=np.float64)
        if n <= T:                       # ряд короче матча — тянем последнее
            arr[n:] = arr[n - 1]
        golds.append(arr)
    if len(golds) == 10:
        res.gold = np.vstack(golds)
    else:
        res.notes.append("gold_t отсутствует или короче двух точек")

    # --- фраги ---
    kr, kd = np.zeros(T + 1), np.zeros(T + 1)
    seen_log = False
    for p, is_rad in zip(players, sides):
        log = p.get("kills_log")
        if not isinstance(log, list):
            continue
        seen_log = True
        for e in log:
            t = e.get("time") if isinstance(e, dict) else None
            if t is None:
                continue
            mn = max(0, int(t) // 60)
            if mn <= T:
                (kr if is_rad else kd)[mn:] += 1
    if seen_log:
        res.kills_r, res.kills_d = kr, kd
    else:
        res.notes.append("kills_log отсутствует у всех игроков")

    # --- байбэки ---
    br, bd = np.zeros(T + 1), np.zeros(T + 1)
    cr, cd = np.zeros(T + 1), np.zeros(T + 1)
    seen_bb = False
    for p, is_rad in zip(players, sides):
        log = p.get("buyback_log")
        if not isinstance(log, list):
            continue
        seen_bb = True
        for e in log:
            t = e.get("time") if isinstance(e, dict) else None
            if t is None:
                continue
            mn = max(0, int(t) // 60)
            if mn > T:
                continue
            (br if is_rad else bd)[mn:] += 1
            # Выкуп на перезарядке восемь минут: именно это, а не сам
            # факт выкупа, определяет, сможет ли команда защитить трон.
            hi = min(T + 1, mn + int(BUYBACK_COOLDOWN_MIN) + 1)
            (cr if is_rad else cd)[mn:hi] += 1
    if seen_bb:
        res.bb_r, res.bb_d = br, bd
        res.bb_oncd_r, res.bb_oncd_d = cr, cd
    else:
        res.notes.append("buyback_log отсутствует у всех игроков")

    # --- ряды, совпадающие с лайвом по определению ---
    res.lh = _series(players, "lh_t", T, res.notes)
    res.dn = _series(players, "dn_t", T, res.notes)
    xp = _series(players, "xp_t", T, res.notes)
    if xp is not None:
        res.level = np.vectorize(level_from_xp)(xp).astype(np.float64)
    return res


def _series(players: list[dict], key: str, T: int,
            notes: list[str]) -> np.ndarray | None:
    """Поминутный ряд по десяти игрокам в матрицу [10, T+1].

    Тянет последнее значение, если ряд короче матча — как это уже делается
    для gold_t. Нет ряда хотя бы у одного игрока — None, и признак станет
    NaN: посчитанный по девяти игрокам перевес молча занижен."""
    out = []
    for p in players:
        v = p.get(key)
        if not isinstance(v, list) or len(v) < 2:
            notes.append(f"{key} отсутствует или короче двух точек")
            return None
        arr = np.full(T + 1, np.nan)
        n = min(len(v), T + 1)
        arr[:n] = np.asarray(v[:n], dtype=np.float64)
        if n <= T:
            arr[n:] = arr[n - 1]
        out.append(arr)
    return np.vstack(out)


def level_from_xp(xp: float) -> float:
    """Уровень по накопленному опыту. Таблица выведена из данных, см.
    config.XP_TO_LEVEL — справочная из интернета на этих матчах неверна."""
    if xp is None or (isinstance(xp, float) and np.isnan(xp)):
        return np.nan
    lo, hi = 0, len(config.XP_TO_LEVEL) - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if xp >= config.XP_TO_LEVEL[mid]:
            lo = mid
        else:
            hi = mid - 1
    return float(lo + 1)


def _nw_features(ser: PlayerSeries, T: int) -> tuple[np.ndarray, np.ndarray]:
    """(разница нетворса богатейших, разница концентрации)."""
    if ser.gold is None or ser.is_radiant is None:
        nan = np.full(T + 1, np.nan)
        return nan, nan.copy()
    r, d = ser.gold[ser.is_radiant], ser.gold[~ser.is_radiant]
    top = np.nanmax(r, axis=0) - np.nanmax(d, axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        # Доля богатейшего в нетворсе команды: показывает, собран ли
        # перевес на керри или размазан по составу.
        sr, sd = np.nansum(r, axis=0), np.nansum(d, axis=0)
        conc = np.where(sr > 0, np.nanmax(r, axis=0) / sr, np.nan) - \
               np.where(sd > 0, np.nanmax(d, axis=0) / sd, np.nan)
    return top, conc


# --- Поминутные признаки ------------------------------------------------

def match_state_frame(match: dict, parsed: ParsedObjectives | None = None,
                      max_minute: int | None = None) -> pd.DataFrame:
    """Поминутные срезы одного матча. duration в признаки не попадает —
    он коррелирует с исходом и в лайве неизвестен."""
    parsed = parsed or parse_objectives(match)
    gold = match.get("radiant_gold_adv")
    if not gold:
        raise ValueError(f"match {match.get('match_id')}: пустой radiant_gold_adv "
                         f"(матч не распарсен OpenDota)")
    gold = np.asarray(gold, dtype=np.float64)
    T = len(gold) - 1
    if max_minute is not None:
        T = min(T, max_minute)
    minutes = np.arange(T + 1)

    xp_raw = match.get("radiant_xp_adv")
    if xp_raw and len(xp_raw) >= T + 1:
        xp = np.asarray(xp_raw[: T + 1], dtype=np.float64)
    else:
        xp = np.full(T + 1, np.nan)       # NaN, а не ноль: «неизвестно» ≠ «поровну»

    g = gold[: T + 1]
    d1 = np.full(T + 1, np.nan)
    d1[1:] = np.diff(g)
    slope5 = np.full(T + 1, np.nan)
    for t in range(1, T + 1):
        lo = max(0, t - 5)
        slope5[t] = (g[t] - g[lo]) / (t - lo)
    xd1 = np.full(T + 1, np.nan)
    if not np.all(np.isnan(xp)):
        xd1[1:] = np.diff(xp)

    def cum(events: list[tuple[int, bool]], want_radiant: bool) -> np.ndarray:
        out = np.zeros(T + 1)
        for minute, victim_is_radiant in events:
            if victim_is_radiant == want_radiant and minute <= T:
                out[minute:] += 1
        return out

    if parsed.tower_ok:
        tow_r = cum(parsed.tower_kills, True)
        tow_d = cum(parsed.tower_kills, False)
    else:
        # Вышки не разобрались — честный NaN. Ноль здесь означал бы
        # «никто не потерял ни одной вышки», что для 40-й минуты ложь.
        tow_r = np.full(T + 1, np.nan)
        tow_d = np.full(T + 1, np.nan)

    # Вышки третьего яруса отдельно от общего счёта. Причина: tower_adv
    # считает три первых яруса и три третьих одинаково, а это совсем разные
    # позиции — t3 открывает бараки. В лайве это считается точно: у СТОЯЩИХ
    # зданий tier и lane на месте всегда (проверено на слепках).
    if parsed.tower_tier_ok:
        t3 = [(mn, v) for mn, v, tier in parsed.tower_kills_tier if tier == 3]
        t3_r = cum(t3, True)
        t3_d = cum(t3, False)
    else:
        t3_r = np.full(T + 1, np.nan)
        t3_d = np.full(T + 1, np.nan)

    if parsed.rax_ok:
        rax_r = cum(parsed.rax_kills, True)
        rax_d = cum(parsed.rax_kills, False)
        # Знак как у tower_adv: положительное = Radiant впереди. Обратный
        # знак у соседнего признака делает разложение нечитаемым, а поймать
        # это по метрикам нельзя — модель выучит любой знак одинаково.
        rax_adv = rax_d - rax_r
    else:
        rax_adv = np.full(T + 1, np.nan)

    rosh_r = np.zeros(T + 1)
    rosh_d = np.zeros(T + 1)
    since = np.full(T + 1, np.nan)
    known_rosh = [(mn, k) for mn, k in parsed.roshan_kills if k is not None]
    for minute, killer_is_radiant in known_rosh:
        if minute <= T:
            if killer_is_radiant:
                rosh_r[minute:] += 1
            else:
                rosh_d[minute:] += 1
    if known_rosh:
        last = None
        for t in range(T + 1):
            for minute, _ in known_rosh:
                if minute == t:
                    last = t
            if last is not None:
                since[t] = t - last
    torm_r = np.zeros(T + 1)
    torm_d = np.zeros(T + 1)
    for minute, killer_is_radiant in parsed.tormentor_kills:
        if killer_is_radiant is None or minute > T:
            continue
        if killer_is_radiant:
            torm_r[minute:] += 1
        else:
            torm_d[minute:] += 1

    if not parsed.objectives_present:
        # Ни одного события вообще: ноль Рошанов здесь означал бы
        # «Рошана не убивали», а на деле мы просто не знаем.
        rosh_r[:] = np.nan
        rosh_d[:] = np.nan
        torm_r[:] = np.nan
        torm_d[:] = np.nan

    ser = parse_player_series(match, T)
    nw_top, nw_conc = _nw_features(ser, T)
    nan = np.full(T + 1, np.nan)

    # Доля перевеса в общем нетворсе. Знаменатель — сумма gold_t по всем
    # десяти игрокам; сверено на реальных матчах: разность командных сумм
    # совпадает с radiant_gold_adv точно. Нет gold_t — честный NaN.
    if ser.gold is None:
        g_norm = nan.copy()
    else:
        tot = np.nansum(ser.gold, axis=0)
        g_norm = g / np.maximum(tot, NW_FLOOR)
        g_norm[np.isnan(tot)] = np.nan
    gn_slope5 = np.full(T + 1, np.nan)
    if not np.all(np.isnan(g_norm)):
        for t in range(1, T + 1):
            lo = max(0, t - 5)
            gn_slope5[t] = (g_norm[t] - g_norm[lo]) / (t - lo)
    def side_diff(mat: np.ndarray | None) -> np.ndarray:
        """Сумма по Radiant минус сумма по Dire для ряда [10, T+1]."""
        if mat is None or ser.is_radiant is None:
            return nan.copy()
        r = np.nansum(mat[ser.is_radiant], axis=0)
        d = np.nansum(mat[~ser.is_radiant], axis=0)
        out = r - d
        out[np.all(np.isnan(mat), axis=0)] = np.nan
        return out

    level_adv = side_diff(ser.level)
    lh_adv = side_diff(ser.lh)
    dn_adv = side_diff(ser.dn)

    kills_r = ser.kills_r if ser.kills_r is not None else nan
    kills_d = ser.kills_d if ser.kills_d is not None else nan
    kills_adv = kills_r - kills_d
    kills_d5 = np.full(T + 1, np.nan)
    if ser.kills_r is not None:
        for t in range(T + 1):
            lo = max(0, t - 5)
            kills_d5[t] = kills_adv[t] - kills_adv[lo]

    return pd.DataFrame({
        "match_id": int(match["match_id"]),
        "minute": minutes,
        "gold_adv": g,
        "gold_adv_d1": d1,
        "gold_adv_slope5": slope5,
        "gold_adv_norm": g_norm,
        "gold_adv_norm_slope5": gn_slope5,
        "xp_adv": xp,
        "xp_adv_d1": xd1,
        "radiant_towers_lost": tow_r,
        "dire_towers_lost": tow_d,
        "tower_adv": tow_d - tow_r,
        "radiant_t3_lost": t3_r,
        "dire_t3_lost": t3_d,
        "t3_adv": t3_d - t3_r,
        "level_adv": level_adv,
        "lh_adv": lh_adv,
        "dn_adv": dn_adv,
        "rax_adv": rax_adv,
        "roshan_radiant": rosh_r,
        "roshan_dire": rosh_d,
        "minutes_since_roshan": since,
        "tormentor_radiant": torm_r,
        "tormentor_dire": torm_d,
        "kills_adv": kills_adv,
        "kills_adv_d5": kills_d5,
        "nw_top_adv": nw_top,
        "nw_conc_adv": nw_conc,
        "bb_used_radiant": ser.bb_r if ser.bb_r is not None else nan,
        "bb_used_dire": ser.bb_d if ser.bb_d is not None else nan,
        "bb_oncd_adv": ((ser.bb_oncd_d - ser.bb_oncd_r)
                        if ser.bb_oncd_r is not None else nan),
        "y": (1 if match.get("radiant_win") else 0),
    })


# --- Загрузка матчей ----------------------------------------------------

def load_matches(directory: Path, limit: int | None = None) -> list[dict]:
    files = sorted(directory.glob("*.json"))
    if limit:
        files = files[:limit]
    out = []
    bad = 0
    for f in files:
        try:
            out.append(json.loads(f.read_text(encoding="utf-8")))
        except json.JSONDecodeError:
            bad += 1
    if bad:
        print(f"[features] пропущено битых файлов: {bad} из {len(files)}")
    return out


def usable_matches(matches: list[dict], verbose: bool = True) -> list[dict]:
    """Оставляет только пригодные матчи и печатает, сколько и почему отсеяно."""
    ok, reasons = [], Counter()
    for m in matches:
        if m.get("radiant_win") is None:
            reasons["нет radiant_win"] += 1
            continue
        if not m.get("radiant_gold_adv"):
            reasons["пустой radiant_gold_adv (не распарсен)"] += 1
            continue
        try:
            match_sides(m)
        except ValueError as e:
            reasons[str(e).split(": ", 1)[-1]] += 1
            continue
        ok.append(m)
    if verbose and reasons:
        print("[features] отсеяно матчей:")
        for r, c in reasons.most_common():
            print(f"    {c:5d}  {r}")
    return ok
