"""Лайв-инференс по Steam Web API.

Схема GetRealtimeStats не документирована. Поэтому весь разбор построен
на принципе «нашёл — скажи где, не нашёл — скажи что не нашёл». Пути, по
которым модуль ищет поля, взяты ПО ПАМЯТИ и не проверены на живом
ответе; именно для этого есть `--dump`. Сверьте вывод `--dump` с таблицей
из `--once --explain-fields` до того, как доверять числам.

XP здесь нет — GetRealtimeStats отдаёт только уровни. Достраивать XP из
уровней запрещено: таблица опыта меняется патчами, и подставленная по
памяти константа тихо испортит модель. Поэтому модуль требует модель,
обученную с `--no-xp`.

Читать память клиента Dota 2 нельзя ни при каких условиях. Единственный
легальный источник реального времени для своей игры — Game State
Integration.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from . import config
from . import dash as D, explain as E, features as F, http as H, minimap as M
from .livelog import LiveLog
from .train import apply_calibrator, logit


class LiveError(RuntimeError):
    pass


class NotStarted(LiveError):
    """Матч ещё не начался: драфт, выбор героев, экран стратегии.

    Отдельный класс, потому что для `--once` это ошибка («предсказывать не по
    чему»), а для `--watch` — обычное состояние, которое надо переждать. Раньше
    и то и другое было LiveError, и запуск `--watch` за минуту до начала матча
    завершал программу вместо того, чтобы дождаться горна.
    """


# Сколько опросов подряд должно провалиться, чтобы `--watch` сдался. Один
# потерянный пакет за матч — норма, и прекращать из-за него сессию нельзя:
# при 15 с между опросами это пять минут непрерывных отказов.
WATCH_MAX_FAILS = 20
WATCH_MIN_INTERVAL = 5.0

# История трекера прореживается по ИГРОВОМУ времени: не чаще одной точки в
# HIST_MIN_GAP секунд. Иначе частый опрос ломает признаки, а не улучшает их.
#
# ЗАМЕРЕНО НА ЖИВОМ МАТЧЕ: GetRealtimeStats обновляется примерно раз в 1.2 с
# (40 опросов с шагом 1 с дали 33 различных game_time, шаг 1-2 с), ответ
# приходит за 0.29 с. То есть опрашивать раз в 1-2 с осмысленно — карта
# оживает. Но тогда за час матча набежит под три тысячи точек, при
# maxlen=400 хвост уедет, и окно «пять минут назад» станет считаться от
# начала дека, а не от начала матча: усечённое окно молча превратилось бы в
# выдуманное. Прореживание до 5 с даёт 12 точек на минуту — этого хватает
# всем окнам (минимальное окно у признаков — минута), а погрешность окна
# не превышает 5 с против прежних 15 с при редком опросе.
HIST_MIN_GAP = 5.0 / 60.0      # в игровых минутах
HIST_MAX = 1500                # 1500 x 5 с = чуть больше двух часов матча


def get_key() -> str:
    key = os.environ.get(config.STEAM_API_KEY_ENV)
    if not key:
        raise LiveError(
            f"Не задан ключ Steam Web API.\n"
            f"Что делать: получите ключ на https://steamcommunity.com/dev/apikey "
            f"и выполните:\n"
            f"    Windows:  setx {config.STEAM_API_KEY_ENV} ваш_ключ\n"
            f"              (после setx закройте и откройте окно заново)\n"
            f"    Linux/Mac: export {config.STEAM_API_KEY_ENV}=ваш_ключ\n"
            f"Без ключа доступны только `--dump --from-file` и `--once --from-file`.")
    return key


_SESSION: requests.Session | None = None


def _session() -> requests.Session:
    """Одна сессия на процесс: --watch стучится раз в 15 с весь матч."""
    global _SESSION
    if _SESSION is None:
        _SESSION = H.make_session()
    return _SESSION


def _fatal(status: int, r: requests.Response) -> Exception | None:
    """Коды, которые повторять бессмысленно."""
    if status in (401, 403):
        deny = r.headers.get("x-deny-reason")
        return LiveError(
            f"{status} от {r.url}: ключ отклонён или доступ закрыт"
            + (f" (x-deny-reason: {deny})" if deny else "") + ".\n"
            f"Что делать: проверьте тот же запрос напрямую —\n"
            f"    python -c \"import os,requests;"
            f"print(requests.get('{r.url.split('?')[0]}',params={{'key':os.environ['"
            f"{config.STEAM_API_KEY_ENV}']}},timeout=15).status_code)\"\n"
            f"Если там 200, а здесь 403 — ошибка в dwp, а не в ключе.")
    return None


def _get(url: str, params: dict, *, timeout: float | None = None,
         retries: int | None = None) -> dict:
    # requests молча выбрасывает параметры со значением None. Запрос
    # уходит без ключа, Steam отвечает 403, и сообщение выглядит как
    # «ключ отклонён» — хотя ключ в глаза не видели. Проверяем явно.
    missing = [k for k, v in params.items() if v is None]
    if missing:
        raise LiveError(
            f"Внутренняя ошибка: параметры {missing} равны None, "
            f"запрос к {url} ушёл бы без них.\n"
            f"Что делать: это баг в dwp, а не в вашей конфигурации.")
    extra = {}
    if timeout is not None:
        extra["timeout"] = timeout
    if retries is not None:
        extra["retries"] = retries
    return H.get_json(
        _session(), url, params, err=LiveError, fatal=_fatal,
        give_up_hint="проверьте сеть и доступность api.steampowered.com. "
                     "В режиме --watch отдельные неудачные опросы "
                     "пропускаются, сессия продолжается.",
        **extra)


# Ключи, под которыми в game_list может лежать название команды. Какой
# из них реально приходит — не проверено на живом ответе, поэтому
# перебираем все и честно показываем прочерк, если ни один не подошёл.
TEAM_NAME_KEYS = (
    ("team_name_radiant", "team_name_dire"),
    ("radiant_team_name", "dire_team_name"),
)
TEAM_OBJ_KEYS = (("radiant_team", "dire_team"), ("team_radiant", "team_dire"))


def game_team_names(g: dict) -> tuple[str, str]:
    """Название команды, а если его нет — team_id, а если и его нет — прочерк.

    У паблик-игр team_name_* приходят пустыми строками; это не поломка
    схемы, а отсутствие команд. Раньше модуль писал «названия не найдены»
    и предлагал чинить парсер, что вводило в заблуждение.
    """
    for kr, kd in TEAM_NAME_KEYS:
        if g.get(kr) or g.get(kd):
            return str(g.get(kr) or "?"), str(g.get(kd) or "?")
    for kr, kd in TEAM_OBJ_KEYS:
        r, d = g.get(kr), g.get(kd)
        if isinstance(r, dict) or isinstance(d, dict):
            return (str((r or {}).get("team_name") or (r or {}).get("name") or "?"),
                    str((d or {}).get("team_name") or (d or {}).get("name") or "?"))
    ri, di = g.get("team_id_radiant"), g.get("team_id_dire")
    if ri or di:
        return f"id {ri or '?'}", f"id {di or '?'}"
    return "-", "-"


def top_live_games(key: str, partner: int = 0, *, timeout: float | None = None,
                   retries: int | None = None) -> list[dict]:
    """Список идущих игр.

    timeout/retries вынесены наружу ради веб-панели. По умолчанию политика
    та же, что у остальных запросов (4 попытки по 20 с), и для консоли это
    правильно: там ждут одного ответа. А страницу браузер опрашивает раз в
    две секунды, и полторы минуты ожидания в обработчике — это уже не
    «медленно», это «панель зависла».
    """
    data = _get(config.STEAM_TOP_LIVE, {"key": key, "partner": partner},
                timeout=timeout, retries=retries)
    games = data.get("game_list")
    if games is None:
        # ПУСТОЙ объект — это не смена схемы, а «сейчас показывать нечего».
        # GetTopLiveGame так и отвечает: {} вместо {"game_list": []}. Раньше
        # здесь была ошибка, и веб-панель из-за неё оставалась без интерфейса
        # в совершенно штатной ситуации — между матчами.
        if not data:
            return []
        raise LiveError(
            f"В ответе GetTopLiveGame нет ключа 'game_list'. Верхние ключи: "
            f"{sorted(data)[:10]}\nЧто делать: схема изменилась, посмотрите "
            f"сырой ответ и поправьте top_live_games().")
    return games


def realtime_stats(key: str, server_steam_id: str) -> dict:
    return _get(config.STEAM_REALTIME, {"key": key, "server_steam_id": server_steam_id})


# --- Разбор недокументированного ответа ---------------------------------

def _dig(obj, *path):
    for p in path:
        if isinstance(obj, dict) and p in obj:
            obj = obj[p]
        else:
            return None
    return obj


class FieldReport:
    def __init__(self) -> None:
        self.found: dict[str, str] = {}
        self.missing: list[str] = []

    def ok(self, name: str, where: str) -> None:
        self.found[name] = where

    def miss(self, name: str) -> None:
        self.missing.append(name)

    def render(self) -> str:
        lines = ["  найдено:"]
        for k, v in self.found.items():
            lines.append(f"    {k:<24} <- {v}")
        if self.missing:
            lines.append("  НЕ НАЙДЕНО (соответствующие признаки = NaN, не ноль):")
            for k in self.missing:
                lines.append(f"    {k}")
        return "\n".join(lines)


def extract_state(payload: dict) -> tuple[dict, FieldReport]:
    """Достаёт то, что нужно модели. Ничего не додумывает."""
    rep = FieldReport()
    st: dict = {}

    gt = _dig(payload, "match", "game_time")
    if gt is None:
        gt = _dig(payload, "match", "timestamp")
        if gt is not None:
            rep.ok("game_time", "match.timestamp (ЗАПАСНОЙ путь, не то же самое!)")
    else:
        rep.ok("game_time", "match.game_time")
    if gt is None:
        rep.miss("game_time")
    st["game_time"] = gt

    teams = payload.get("teams")
    nw = {}
    heroes = {2: [], 3: []}
    if isinstance(teams, list) and teams:
        for t in teams:
            tn = t.get("team_number")
            if tn is None:
                continue
            if t.get("net_worth") is not None:
                nw[int(tn)] = float(t["net_worth"])
            else:
                players = t.get("players") or []
                vals = [p.get("net_worth") for p in players if p.get("net_worth") is not None]
                if vals:
                    nw[int(tn)] = float(sum(vals))
            for p in (t.get("players") or []):
                h = p.get("heroid", p.get("hero_id"))
                if h:
                    heroes.setdefault(int(tn), []).append(int(h))
        if nw:
            rep.ok("net_worth по командам", "teams[].net_worth или сумма players[].net_worth")
        else:
            rep.miss("net_worth по командам")
    else:
        rep.miss("teams")

    lead = _dig(payload, "match", "radiant_lead")
    if config.TEAM_RADIANT in nw and config.TEAM_DIRE in nw:
        st["gold_adv"] = nw[config.TEAM_RADIANT] - nw[config.TEAM_DIRE]
        rep.ok("gold_adv", "разница net_worth команд")
    elif lead is not None:
        st["gold_adv"] = float(lead)
        rep.ok("gold_adv", "match.radiant_lead (ЗАПАСНОЙ путь)")
    else:
        st["gold_adv"] = np.nan
        rep.miss("gold_adv")

    # Фраги и нетворс по игрокам: в GetRealtimeStats они есть
    # (players[].kill_count и players[].net_worth), значит признаки
    # kills_adv, nw_top_adv и nw_conc_adv считаются и в лайве.
    kills, nws = {}, {}
    # Уровни, ластхиты и денаи. Ради них заведён EXACT_STATE_FEATURES:
    # определение этих величин в обучающей выгрузке и здесь совпадает точно
    # (lh_t <-> lh_count, dn_t <-> denies_count, xp_t через таблицу уровней
    # <-> level), в отличие от золота, где лайв отдаёт нетворс, а обучение
    # училось на добытом. Сумма по пятерым, а не отдельные игроки: признак
    # командный.
    sums: dict[str, dict[int, float]] = {"level": {}, "lh": {}, "dn": {}}
    SRC = {"level": "level", "lh": "lh_count", "dn": "denies_count"}
    for t in (teams if isinstance(teams, list) else []):
        tn = t.get("team_number")
        if tn is None:
            continue
        ps = t.get("players") or []
        kc = [p.get("kill_count") for p in ps if p.get("kill_count") is not None]
        nw = [p.get("net_worth") for p in ps if p.get("net_worth") is not None]
        if kc:
            kills[int(tn)] = float(sum(kc))
        if len(nw) == 5:
            nws[int(tn)] = [float(v) for v in nw]
        for name, src in SRC.items():
            v = [p.get(src) for p in ps if p.get(src) is not None]
            # Ровно пятеро: сумма по четверым молча занижает перевес, а по
            # метрикам такое не видно.
            if len(v) == 5:
                sums[name][int(tn)] = float(sum(v))
    st["kills"] = kills
    st["player_nw"] = nws
    # account_id пятёрок: на них стоит рейтинг игроков, заменивший
    # командный Elo. В GetRealtimeStats поле называется `accountid`
    # (без подчёркивания), в выгрузке OpenDota — `account_id`; берём оба,
    # чтобы расхождение имён не превратилось в молчаливый ноль.
    accs: dict[int, list[int]] = {}
    for t in (teams if isinstance(teams, list) else []):
        tn = t.get("team_number")
        if tn is None:
            continue
        got = [p.get("accountid", p.get("account_id")) for p in (t.get("players") or [])]
        got = [int(a) for a in got if a]
        if len(got) == 5:
            accs[int(tn)] = got
    st["accounts"] = accs
    if len(accs) == 2:
        rep.ok("состав по account_id", "players[].accountid")
    else:
        rep.miss("accountid (нужны все 5 игроков в каждой команде)")
    for name in SRC:
        st[f"{name}_sum"] = sums[name]
    got = [n for n in SRC if len(sums[n]) == 2]
    if got:
        rep.ok("уровни/ластхиты/денаи",
               ", ".join(f"players[].{SRC[n]}" for n in got))
    for n in SRC:
        if len(sums[n]) != 2:
            rep.miss(f"{SRC[n]} (нужны все 5 игроков в каждой команде)")
    st["team_names"] = {int(t["team_number"]): str(t.get("team_name") or "")
                        for t in (teams if isinstance(teams, list) else [])
                        if t.get("team_number") is not None}
    if len(kills) == 2:
        rep.ok("фраги по командам", "сумма players[].kill_count")
    else:
        rep.miss("фраги (players[].kill_count)")
    if len(nws) == 2:
        rep.ok("нетворс по игрокам", "players[].net_worth (5 на команду)")
    else:
        rep.miss("нетворс по игрокам (players[].net_worth)")

    st["heroes_radiant"] = heroes.get(config.TEAM_RADIANT, [])
    st["heroes_dire"] = heroes.get(config.TEAM_DIRE, [])
    if len(st["heroes_radiant"]) == 5 and len(st["heroes_dire"]) == 5:
        rep.ok("составы", "teams[].players[].heroid")
    else:
        rep.miss(f"составы (получено {len(st['heroes_radiant'])}/"
                 f"{len(st['heroes_dire'])} героев вместо 5/5)")

    # Стадия матча. В драфте ответ структурно полный, но все значения
    # нулевые: heroid=0, net_worth=0, зданий два. Раньше модуль принимал
    # это за настоящие данные и выдавал 55% ни на чём.
    st["game_state"] = _dig(payload, "match", "game_state")
    # game_state == 5 у идущего матча — единственное значение, подтверждённое на
    # живом ответе, поэтому оно только ДОБАВЛЯЕТ уверенности, а не заменяет
    # прежнюю эвристику. Если константа окажется неверной для какого-то режима,
    # поведение останется прежним, а не сломается в другую сторону.
    st["in_progress"] = bool(
        st["game_state"] == config.GAME_STATE_IN_PROGRESS
        or (st.get("gold_adv") is not None and not np.isnan(st.get("gold_adv", np.nan))
            and st["gold_adv"] != 0)
        or any(h for h in st["heroes_radiant"] + st["heroes_dire"]))

    buildings = payload.get("buildings")
    st["towers_lost"] = {}
    st["rax_lost"] = {}
    if not isinstance(buildings, list) or not buildings:
        rep.miss("buildings")
    else:
        # ПРОВЕРЕНО НА ЖИВОМ ОТВЕТЕ: у снесённого здания поле team
        # обнуляется (team=0), поэтому по флагу destroyed нельзя понять,
        # чьё оно было. Считаем по НЕДОСТАЧЕ стоящих: 11 минус то,
        # сколько осталось у стороны. Сверка на реальном матче: Radiant 8
        # из 11 и Dire 9 из 11 при пяти снесённых — сходится.
        # type: 0 = вышка, 1 = барак, 2 = трон. Подтверждено раскладом
        # 22/12/2 на полном списке из 36 зданий.
        standing: dict[tuple, int] = {}
        for b in buildings:
            if b.get("destroyed"):
                continue
            key = (b.get("team"), b.get("type"))
            standing[key] = standing.get(key, 0) + 1
        st["standing"] = standing

        # Раньше отсутствие стороны среди стоящих означало «не разобрали», и
        # сторона, потерявшая ВСЕ вышки, получала ноль потерянных: tower_adv
        # переворачивал знак ровно в тот момент, когда счёт на экране важнее
        # всего. Отличить «не осталось ни одной» от «схема другая» можно так:
        # список зданий приходит целиком и не укорачивается, а трон стоит,
        # пока идёт игра. Полная длина плюс оба трона на месте — значит
        # нумерация та, и пустая клетка это честный ноль стоящих.
        full = len(buildings) == config.N_BUILDINGS_TOTAL
        thrones = all(standing.get((t, 2)) == 1
                      for t in (config.TEAM_RADIANT, config.TEAM_DIRE))
        n_tow = {t: standing.get((t, 0), 0)
                 for t in (config.TEAM_RADIANT, config.TEAM_DIRE)}
        n_rax = {t: standing.get((t, 1), 0)
                 for t in (config.TEAM_RADIANT, config.TEAM_DIRE)}
        too_many = (max(n_tow.values()) > config.N_TOWERS_PER_SIDE
                    or max(n_rax.values()) > config.N_RAX_PER_SIDE)
        if not full:
            rep.miss(f"здания: записей {len(buildings)}, ожидалось "
                     f"{config.N_BUILDINGS_TOTAL} — список неполный, считать по "
                     f"недостаче стоящих нельзя")
        elif not thrones:
            rep.miss(f"здания по сторонам: тронов (type 2) у team "
                     f"{config.TEAM_RADIANT}/{config.TEAM_DIRE} не нашлось "
                     f"(ключи team/type: {sorted(standing)}) — похоже, сменилась "
                     f"нумерация сторон")
        elif too_many:
            rep.miss(f"здания по сторонам: стоящих больше, чем бывает — вышки "
                     f"{n_tow}, бараки {n_rax}")
        else:
            st["towers_lost"] = {t: config.N_TOWERS_PER_SIDE - n_tow[t]
                                 for t in n_tow}
            st["rax_lost"] = {t: config.N_RAX_PER_SIDE - n_rax[t] for t in n_rax}
            rep.ok("здания", f"buildings[]: считаем по недостаче стоящих "
                             f"(осталось вышек {n_tow[config.TEAM_RADIANT]}"
                             f"/{n_tow[config.TEAM_DIRE]} из "
                             f"{config.N_TOWERS_PER_SIDE})")
            # Ярусы. У СТОЯЩЕГО здания tier на месте всегда (у снесённого
            # обнуляется вместе со всем остальным), поэтому потери третьего
            # яруса считаются той же недостачей: три t3 на сторону.
            n_t3 = {t: sum(1 for b in buildings
                           if not b.get("destroyed") and b.get("team") == t
                           and b.get("type") == 0 and b.get("tier") == 3)
                    for t in (config.TEAM_RADIANT, config.TEAM_DIRE)}
            if max(n_t3.values()) > config.N_T3_PER_SIDE:
                rep.miss(f"вышки третьего яруса: стоящих больше трёх {n_t3}")
            else:
                st["t3_lost"] = {t: config.N_T3_PER_SIDE - n_t3[t] for t in n_t3}
                rep.ok("вышки по ярусам",
                       f"buildings[].tier у стоящих (t3 осталось "
                       f"{n_t3[config.TEAM_RADIANT]}/{n_t3[config.TEAM_DIRE]} "
                       f"из {config.N_T3_PER_SIDE})")

    # ПРОВЕРЕНО НА ЖИВОМ ОТВЕТЕ: слов roshan и aegis в ответе нет вовсе.
    # Таймер Рошана в этом источнике отсутствует, а не «не нашёлся».
    # Путь оставлен на случай, если Valve его добавит.
    st["roshan_respawn_timer"] = _dig(payload, "match", "roshan_respawn_timer")

    # team_id для Elo. В паблике приходит 0, у турнирных матчей — реальный.
    tids = {}
    for t in (payload.get("teams") or []):
        tn, ti = t.get("team_number"), t.get("team_id")
        if tn is not None and ti:
            tids[int(tn)] = int(ti)
    st["team_ids"] = tids
    if len(tids) == 2:
        rep.ok("team_id команд", f"teams[].team_id {tids}")

    # Опознавательные поля: нужны, чтобы потом связать лог с завершённым
    # матчем в OpenDota (см. dwp.livecheck).
    st["match_id"] = _dig(payload, "match", "match_id")
    st["server_steam_id"] = _dig(payload, "match", "server_steam_id")

    gg = _dig(payload, "graph_data", "graph_gold")
    if isinstance(gg, list) and gg:
        # В ПРИЗНАК НЕ ИДЁТ, но пишется в лог. Прежний довод «это не разница
        # нетворса, значит подставлять нельзя» был перевёрнут: признак
        # gold_adv обучен на radiant_gold_adv из OpenDota, а это разница
        # ДОБЫТОГО золота, не нетворса. То есть кандидатом на расхождение
        # является как раз нетворс, а graph_gold может оказаться ровно той
        # величиной, на которой модель училась. Проверяется это сверкой лога
        # с завершённым матчем, а не рассуждением.
        st["graph_gold_len"] = len(gg)
        st["graph_gold_last"] = float(gg[-1])
    return st, rep


class LiveTracker:
    """Хранит историю опросов: производные золота и таймер Рошана.

    Рошан в лайве определяется по переходу таймера респавна из нуля в
    положительное значение. Кто именно убил — из этого источника не
    следует, и подставлять «наверное, лидер» нельзя. Поэтому счётчики
    Рошанов по сторонам остаются NaN, а заполняется только время с
    последнего Рошана. Если подключились к игре в середине, ранние
    Рошаны потеряны — это видно по nan в minutes_since_roshan.
    """

    def __init__(self) -> None:
        self.hist: deque[tuple[float, float]] = deque(maxlen=HIST_MAX)   # (минута, gold_adv)
        self.khist: deque[tuple[float, float]] = deque(maxlen=HIST_MAX)  # (минута, kills_adv)
        self.nhist: deque[tuple[float, float]] = deque(maxlen=HIST_MAX)  # (минута, gold_adv_norm)
        self.first_minute: float | None = None
        self.phist: deque[float] = deque(maxlen=180)   # история вероятности
        self.mhist: deque[float] = deque(maxlen=180)   # и минуты к ней
        self.last_rt: float | None = None
        self.last_rosh_minute: float | None = None
        self.rosh_events = 0

    @staticmethod
    def _push(hist: deque, minute: float, value: float) -> None:
        """Добавить точку, если она не слишком близко к предыдущей.

        Прореживание по ИГРОВОМУ времени, а не по числу опросов: частота
        опроса — свойство панели, а окна признаков заданы в минутах матча, и
        зависеть от первого они не должны. Первая точка добавляется всегда:
        на ней держится `from_match_start`.
        """
        if hist and (float(minute) - hist[-1][0]) < HIST_MIN_GAP:
            return
        hist.append((float(minute), float(value)))

    def update(self, minute: float, gold_adv: float, rt: float | None) -> None:
        if not (np.isnan(minute) if isinstance(minute, float) else False):
            if self.first_minute is None:
                self.first_minute = float(minute)
            self._push(self.hist, minute, gold_adv)
        if rt is not None:
            rt = float(rt)
            if self.last_rt is not None and self.last_rt <= 0 < rt:
                self.rosh_events += 1
                self.last_rosh_minute = minute
            self.last_rt = rt

    @property
    def from_match_start(self) -> bool:
        """Подключились ли мы к матчу с самого горна.

        Только тогда усечённое окно означает ровно то же, что при обучении:
        «от начала матча». Подключившись на 31-й минуте, мы про предыдущие
        пять не знаем ничего, и «ноль фрагов за пять минут» было бы выдумкой,
        а не укороченным окном.
        """
        return self.first_minute is not None and self.first_minute <= 0.5

    def _lookback(self, hist, minute: float, back: float) -> tuple[float, float] | None:
        """Точка `back` минут назад, а если её нет — самая ранняя из истории.

        Возвращает (значение, ФАКТИЧЕСКИЙ сдвиг в минутах) или None.

        Почему усечённое окно, а не NaN. При обучении окно в начале матча тоже
        усечено, но не выброшено: `match_state_frame` считает slope5[t] по
        (t − max(0, t−5)) минутам, поэтому пропуска там не было НИ РАЗУ, кроме
        нулевой минуты. Отдавая в эти минуты NaN, лайв уводил строку в ветку,
        которую LightGBM из данных не выучивал — измерено реплеем: до 14 п.п.
        расхождения с офлайном на четвёртой минуте.

        Но усечение законно ТОЛЬКО от начала матча (см. from_match_start).
        Подключились в середине — честный NaN, и `ready_at` показывает, с
        какой минуты признак наполнится.

        Фактический сдвиг возвращается и для полного окна: при опросе раз в
        15 с точка «пять минут назад» на самом деле старше на 0-15 с, и делить
        на ровно 5.0 значит завышать наклон.
        """
        best = None
        for mn, v in hist:
            if mn <= minute - back:
                best = (v, minute - mn)
        if best is not None:
            return best
        if not hist or not self.from_match_start:
            return None
        mn, v = hist[0]
        return v, minute - mn

    def norm_slope(self, minute: float, gold_norm: float) -> float:
        """Наклон доли за 5 игровых минут, той же конвенции, что и обучение."""
        if isinstance(minute, float) and np.isnan(minute):
            return np.nan
        if isinstance(gold_norm, float) and np.isnan(gold_norm):
            return np.nan
        self._push(self.nhist, minute, gold_norm)
        got = self._lookback(self.nhist, minute, 5.0)
        if got is None or got[1] <= 0:
            return np.nan
        return (gold_norm - got[0]) / got[1]

    def kills_delta(self, minute: float, kills_adv: float) -> float:
        """Изменение разницы фрагов за пять минут: темп замесов.

        Без деления на длину окна — при обучении это тоже разность счётчиков,
        а не скорость.
        """
        if kills_adv is None or (isinstance(kills_adv, float) and np.isnan(kills_adv)):
            return np.nan
        self._push(self.khist, minute, kills_adv)
        got = self._lookback(self.khist, minute, 5.0)
        return kills_adv - got[0] if got is not None else np.nan

    def _at(self, minute: float, back: float) -> float | None:
        target = minute - back
        best = None
        for mn, g in self.hist:
            if mn <= target:
                best = g
        return best

    def derivatives(self, minute: float, gold_adv: float) -> tuple[float, float]:
        g1 = self._at(minute, 1.0)
        d1 = gold_adv - g1 if g1 is not None else np.nan
        got = self._lookback(self.hist, minute, 5.0)
        slope5 = (gold_adv - got[0]) / got[1] if got is not None and got[1] > 0 else np.nan
        return d1, slope5

    def note_p(self, minute: float, p: float) -> None:
        """Запомнить показанное число вместе с игровой минутой.

        Минута нужна графику: опрос идёт по стенным часам, и если по оси
        отложить номер опроса, то пауза в матче или пропущенный пакет
        растянут кривую в месте, где на карте ничего не происходило.
        """
        self.phist.append(float(p))
        self.mhist.append(float(minute) if minute == minute else float("nan"))

    def minutes_since_roshan(self, minute: float) -> float:
        if self.last_rosh_minute is None:
            return np.nan
        return minute - self.last_rosh_minute


def build_row(art: dict, st: dict, tracker: LiveTracker) -> tuple[pd.DataFrame, list[str]]:
    warns: list[str] = []
    gt = st.get("game_time")
    minute = float(gt) / 60.0 if gt is not None else np.nan
    gold = st.get("gold_adv", np.nan)
    d1, slope5 = tracker.derivatives(minute, gold)
    if np.isnan(d1) or np.isnan(slope5):
        # Не «несколько опросов»: окно считается в ИГРОВЫХ минутах. Если
        # подключились на 13-й, темп за 5 минут появится на 18-й, сколько
        # бы опросов за это время ни прошло.
        warns.append("окна считаются в игровых минутах от момента подключения, "
                     "а не в опросах: за минуту — через 1 мин, за пять — через 5 мин")

    # Обе стороны или ни одной. Частичный словарь молча превращался в ноль по
    # недостающей стороне, а ноль в признаке по вышкам неотличим от «вышек не
    # теряли» — цена одного из сильнейших сигналов.
    tow, rax = st.get("towers_lost") or {}, st.get("rax_lost") or {}
    if len(tow) == 2:
        tow_r = float(tow[config.TEAM_RADIANT])
        tow_d = float(tow[config.TEAM_DIRE])
    else:
        tow_r = tow_d = np.nan
        warns.append("здания не разобраны -> признаки по вышкам = NaN")
    if len(rax) == 2:
        rax_adv = float(rax[config.TEAM_DIRE]) - float(rax[config.TEAM_RADIANT])
    else:
        rax_adv = np.nan
    t3 = st.get("t3_lost") or {}
    if len(t3) == 2:
        t3_r, t3_d = float(t3[config.TEAM_RADIANT]), float(t3[config.TEAM_DIRE])
    else:
        t3_r = t3_d = np.nan

    def team_diff(name: str) -> float:
        s = st.get(f"{name}_sum") or {}
        if len(s) != 2:
            return np.nan
        return float(s[config.TEAM_RADIANT]) - float(s[config.TEAM_DIRE])

    # Требование 4: финальный рейтинг корректен именно для НОВЫХ матчей,
    # а лайв-матч всегда новый — утечки из будущего здесь нет.
    tids = st.get("team_ids") or {}
    if art.get("rating_kind") == "player":
        # Рейтинг игроков. Заменил командный Elo по замеру: AUC драфта
        # 0.6138 против 0.6001, и account_id есть там, где team_id нет.
        pf = art.get("player_elo_final") or {}
        accs = st.get("accounts") or {}
        rad, dire = accs.get(config.TEAM_RADIANT, []), accs.get(config.TEAM_DIRE, [])
        elo_diff = F.player_elo_diff(pf, rad, dire)
        if elo_diff != elo_diff:
            elo_diff = 0.0
            elo_note = "составы по account_id неполные -> рейтинг 0"
        else:
            known = sum(1 for a in list(rad) + list(dire)
                        if a in pf or str(a) in pf)
            elo_note = (f"рейтинг игроков: {elo_diff:+.0f} "
                        f"({known} из 10 знакомы по истории)")
    else:
        # Старые артефакты без ключа rating_kind: командный Elo, как было.
        fin = art.get("elo_final") or {}
        r_id, d_id = tids.get(config.TEAM_RADIANT), tids.get(config.TEAM_DIRE)
        if r_id in fin and d_id in fin:
            elo_diff = float(fin[r_id] - fin[d_id])
            elo_note = f"Elo из истории: {elo_diff:+.0f} (team_id {r_id} против {d_id})"
        elif r_id or d_id:
            elo_diff = 0.0
            elo_note = (f"team_id есть ({r_id}/{d_id}), но этих команд нет в обучающей "
                        f"истории -> Elo 0")
        else:
            elo_diff = 0.0
            elo_note = "team_id = 0 (паблик или ответ без команд) -> Elo 0"
    # Наружу для разбора драфта в dwp.web: без Elo разложение по героям
    # читается так, будто героями всё и решается.
    st["elo_diff"] = elo_diff
    kills = st.get("kills") or {}
    nws = st.get("player_nw") or {}
    if len(kills) == 2:
        kills_adv = kills[config.TEAM_RADIANT] - kills[config.TEAM_DIRE]
    else:
        kills_adv = np.nan
    if len(nws) == 2:
        r, d = np.array(nws[config.TEAM_RADIANT]), np.array(nws[config.TEAM_DIRE])
        nw_top = float(r.max() - d.max())
        nw_conc = float((r.max() / r.sum() if r.sum() > 0 else np.nan)
                        - (d.max() / d.sum() if d.sum() > 0 else np.nan))
    else:
        nw_top = nw_conc = np.nan
    kd5 = tracker.kills_delta(minute, kills_adv)

    # Знаменатель для gold_adv_norm. При обучении это сумма gold_t по десяти
    # игрокам, в лайве — сумма players[].net_worth. ЭТО РАЗНЫЕ ВЕЛИЧИНЫ:
    # gold_t — добытое золото, net_worth — оно же минус потраченное плюс
    # стоимость предметов. Сверен был только порядок величины (медиана суммы
    # на 25-29-й минуте: 117866 в выгрузке против 110577 в слепке на 25:41),
    # а не определение. Замер на 1500 завершённых матчах: сумма нетворса
    # составляет 0.941 от суммы gold_t, то есть знаменатель систематически
    # занижен примерно на 6%. Вместе с числителем (см. gold_adv выше) это
    # даёт перекос доли в пользу лидера; величина открыта, её меряет
    # dwp.livecheck на реальном лайв-логе.
    if len(nws) == 2:
        total_nw = float(np.sum(nws[config.TEAM_RADIANT])
                         + np.sum(nws[config.TEAM_DIRE]))
    else:
        total_nw = np.nan
    if np.isnan(total_nw) or np.isnan(gold):
        gold_norm = np.nan
    else:
        gold_norm = gold / max(total_nw, F.NW_FLOOR)
    gn_slope5 = tracker.norm_slope(minute, gold_norm)

    row = {
        "minute": minute,
        "kills_adv": kills_adv,
        "kills_adv_d5": kd5,
        "nw_top_adv": nw_top,
        "nw_conc_adv": nw_conc,
        "gold_adv": gold,
        "gold_adv_d1": d1,
        "gold_adv_slope5": slope5,
        "gold_adv_norm": gold_norm,
        "gold_adv_norm_slope5": gn_slope5,
        "radiant_towers_lost": tow_r,
        "dire_towers_lost": tow_d,
        "tower_adv": (tow_d - tow_r) if not np.isnan(tow_r) else np.nan,
        "radiant_t3_lost": t3_r,
        "dire_t3_lost": t3_d,
        "t3_adv": (t3_d - t3_r) if not np.isnan(t3_r) else np.nan,
        "level_adv": team_diff("level"),
        "lh_adv": team_diff("lh"),
        "dn_adv": team_diff("dn"),
        "rax_adv": rax_adv,
        # Кто убил Рошана, из GetRealtimeStats не следует. NaN, а не догадка.
        "roshan_radiant": np.nan,
        "roshan_dire": np.nan,
        "minutes_since_roshan": tracker.minutes_since_roshan(minute),
    }
    # Предупреждать только если модель на эти признаки опирается. Лайв-модель
    # обучена без них, и постоянная строка про Рошана была чистым шумом.
    if set(art.get("state_features") or ()) & set(F.LIVE_UNAVAILABLE):
        warns.append("Рошана и Терзателя в GetRealtimeStats нет вовсе (ни "
                     "таймеров, ни событий) -> эти признаки = NaN")

    rad, dire = st.get("heroes_radiant") or [], st.get("heroes_dire") or []
    dls: list[float] = []
    if len(rad) == 5 and len(dire) == 5:
        fake = {"match_id": 0, "radiant_win": True,
                "players": [{"hero_id": h, "player_slot": i} for i, h in enumerate(rad)]
                           + [{"hero_id": h, "player_slot": 128 + i}
                              for i, h in enumerate(dire)]}
        Xd, _, _ = F.draft_matrix([fake], art["id2idx"], {0: elo_diff})
        # У каждого участника ансамбля СВОЯ драфт-модель: она обучена на своём
        # разбиении, и подставлять всем один логит значило бы усреднять не то,
        # что замерялось. Поэтому логитов столько же, сколько моделей; в саму
        # строку идёт их среднее (его видит лог и разбор), а каждая модель
        # считает по своему — см. predict().
        for m in members(art):
            p_draft = float(m["draft_model"].predict_proba(Xd)[0, 1])
            dls.append(float(logit(np.array([p_draft]))[0]))
        row["draft_logit"] = float(np.mean(dls))
        warns.append(elo_note)
    else:
        row["draft_logit"] = np.nan
        warns.append("составы неполные -> draft_logit = NaN, оценка заметно хуже")

    df = pd.DataFrame([row])
    for f in art["state_features"]:
        if f not in df.columns:
            df[f] = np.nan
            warns.append(f"признак {f} не заполняется в лайве -> NaN")
    if dls:
        df.attrs["draft_logits"] = dls
    return df, warns


# Подписи для экрана. Сырые имена признаков нужны при отладке, но во
# время матча читать "nw_conc_adv" некогда. Второй элемент — сколько
# игровых минут от подключения нужно, чтобы признак наполнился данными.
HUMAN: dict[str, tuple[str, float]] = {
    "gold_adv": ("перевес по золоту", 0.0),
    "gold_adv_d1": ("золото: за минуту", 1.0),
    "gold_adv_slope5": ("золото: темп за 5 мин", 5.0),
    "gold_adv_norm": ("перевес в долях от банка", 0.0),
    "gold_adv_norm_slope5": ("доля: темп за 5 мин", 5.0),
    "xp_adv": ("перевес по опыту", 0.0),
    "xp_adv_d1": ("опыт: за минуту", 1.0),
    # Не «рейтинг команд»: моделей две породы, и какая именно рейтинговая
    # колонка внутри — видно из art["rating_kind"], а не отсюда. Подпись
    # тут общая для обеих, а какой это рейтинг, пишет строка-замечание
    # ниже («рейтинг игроков: +68, 10 из 10 знакомы по истории»).
    "draft_logit": ("драфт + рейтинг", 0.0),
    "tower_adv": ("вышки: перевес", 0.0),
    "radiant_towers_lost": ("вышек потерял Radiant", 0.0),
    "dire_towers_lost": ("вышек потерял Dire", 0.0),
    "rax_adv": ("бараки: перевес", 0.0),
    "kills_adv": ("перевес по фрагам", 0.0),
    "kills_adv_d5": ("фраги: за 5 мин", 5.0),
    "nw_top_adv": ("богатейшие герои: разрыв", 0.0),
    "nw_conc_adv": ("перекос золота в одного", 0.0),
    "minute": ("минута матча", 0.0),
    "roshan_radiant": ("Рошанов у Radiant", 0.0),
    "roshan_dire": ("Рошанов у Dire", 0.0),
    "minutes_since_roshan": ("минут с Рошана", 0.0),
    "tormentor_radiant": ("Терзателей у Radiant", 0.0),
    "tormentor_dire": ("Терзателей у Dire", 0.0),
    "bb_used_radiant": ("выкупов у Radiant", 0.0),
    "bb_used_dire": ("выкупов у Dire", 0.0),
    "bb_oncd_adv": ("выкуп на перезарядке", 0.0),
    "radiant_t3_lost": ("t3-вышек потерял Radiant", 0.0),
    "dire_t3_lost": ("t3-вышек потерял Dire", 0.0),
    "t3_adv": ("вышки t3: перевес", 0.0),
    "level_adv": ("перевес по уровням", 0.0),
    "lh_adv": ("перевес по добиваниям", 0.0),
    "dn_adv": ("перевес по денаям", 0.0),
}


def human(feat: str) -> str:
    return HUMAN.get(feat, (feat, 0.0))[0]


def ready_at(feat: str, tracker: "LiveTracker") -> float | None:
    """С какой игровой минуты признак перестанет быть пустым.

    Нужно, потому что NaN в бою и NaN при обучении означают разное.
    kills_adv_d5 при обучении не был пустым НИ РАЗУ (0 строк из 183036),
    gold_adv_slope5 — только на нулевой минуте. Значит ветка, по которой
    LightGBM уводит пропуск, для этих признаков не выучена из данных, и
    вклад рядом с пустым значением брать всерьёз нельзя.
    """
    need = HUMAN.get(feat, (feat, 0.0))[1]
    if need <= 0 or tracker.first_minute is None:
        return None
    return tracker.first_minute + need


def members(art: dict) -> list[dict]:
    """Участники ансамбля. Одиночная модель — ансамбль из одной."""
    return art.get("members") or [art]


def log_model_name(art: dict) -> str:
    """Как модель подписывается в лайв-логе и в именах его файлов.

    Подпись должна быть РАЗНОЙ у разных наборов моделей: `dwp.livecheck`
    считает калибровку отдельно по каждой модели, и смешать в одну кучу
    строки ансамбля и одиночной модели значило бы посчитать среднее по
    двум разным величинам. Длину приходится ограничивать: имя уходит в
    имя файла.
    """
    paths = [Path(p).stem for p in (art.get("paths") or [])]
    if len(paths) <= 1:
        return art.get("name") or (paths[0] + ".pkl" if paths else "model.pkl")
    joined = "+".join(paths)
    if len(joined) > 48:
        joined = f"ens{len(paths)}_{paths[0]}..{paths[-1]}"
    return joined + ".pkl"


def predict(art: dict, df: pd.DataFrame) -> tuple[float, pd.DataFrame]:
    """Вероятность и разложение по вкладам.

    Для ансамбля усредняются ВЕРОЯТНОСТИ, а не логиты: на замере разницы
    между ними не было (0.5567 против 0.5567), а вероятности проще —
    у каждой модели свой калибратор и своя обрезка хвостов, и усреднять
    уже калиброванные вероятности законно. Калибровать среднее заново
    было бы нельзя: для этого нужна отдельная выборка.

    Вклады усредняются тоже. Иначе сумма вкладов перестанет сходиться с
    числом на экране: у каждой модели разбор свой.
    """
    feats = art["state_features"]
    ms = members(art)
    dls = df.attrs.get("draft_logits") or []
    ps: list[float] = []
    contribs: list[np.ndarray] = []
    for i, m in enumerate(ms):
        d = df
        if len(dls) == len(ms) and len(ms) > 1:
            # Своя драфт-модель у каждого участника, см. build_row.
            d = df.copy()
            d.loc[d.index[0], "draft_logit"] = dls[i]
        raw = m["booster"].predict(d[feats], num_iteration=m["booster"].best_iteration)
        eps = max(m.get("calib_eps", 0.0), 1e-6)
        ps.append(float(np.clip(apply_calibrator(m["iso"], raw), eps, 1 - eps)[0]))
        contrib, _base = E.state_contributions(m, d)
        contribs.append(np.asarray(contrib[0], dtype=float))
    p = float(np.mean(ps))
    bd = pd.DataFrame({"признак": feats, "значение": [df.iloc[0][f] for f in feats],
                       "вклад_логит": np.mean(contribs, axis=0)})
    bd = bd.reindex(bd["вклад_логит"].abs().sort_values(ascending=False).index)
    if len(ms) > 1:
        bd.attrs["members"] = [float(x) for x in ps]
    return p, bd


def load_model(path: Path) -> dict:
    if not path.exists():
        raise LiveError(
            f"Нет модели {path}.\n"
            f"Что делать: `python -m dwp.train --no-xp` — лайв-источник не отдаёт XP.")
    with path.open("rb") as fh:
        art = pickle.load(fh)
    missing = [f for f in art.get("state_features", []) if f in F.LIVE_UNAVAILABLE]
    if missing and not art.get("live_features"):
        print(f"ВНИМАНИЕ: модель обучена с признаками, которых нет в "
              f"GetRealtimeStats:\n    {', '.join(missing)}\n"
              f"  В бою они всегда NaN, а модель училась на них опираться. "
              f"Обучите так:\n    python -m dwp.train --source real --live-features",
              file=sys.stderr)
    if art.get("use_xp"):
        raise LiveError(
            f"Модель {path} обучена С признаками по опыту, а GetRealtimeStats "
            f"отдаёт только уровни. Достраивать XP из уровней нельзя: таблица "
            f"опыта меняется патчами.\n"
            f"Что делать: `python -m dwp.train --no-xp` и укажите полученный файл.")
    art.setdefault("name", path.name)
    art.setdefault("paths", [str(path)])
    return art


ENSEMBLE_GLOB = "ens_*.pkl"


def default_models() -> list[Path]:
    """Что брать, если модель не указана.

    Ансамбль, если он обучен, иначе одиночная `live_exact.pkl`. Такой
    порядок — не «новее значит лучше», а замер: на слепом холдауте из
    1167 матчей ансамбль из пяти сидов дал log loss 0.5233 против 0.5285
    у средней одиночной (интервал по матчам [−0.0057, −0.0047]) и ECE
    0.0082 против 0.0155–0.0315. Для лайва важнее второе: ECE — это то,
    насколько врёт число на экране.

    Собрать ансамбль: `python -m dwp.train --source real --live-features
    --extra-features --exact-features --gold-norm add --seed N
    --out models\\ens_0N.pkl` для N = 3, 5, 7, 11, 13.
    """
    got = sorted(config.MODELS_DIR.glob(ENSEMBLE_GLOB))
    if len(got) >= 2:
        return got
    return [config.MODELS_DIR / "live_exact.pkl"]


def resolve_models(values) -> list[Path]:
    """Раскрыть шаблоны в списке путей к моделям.

    `--model models\\ens_*.pkl` в PowerShell не раскрывается оболочкой, а
    писать пять путей руками — приглашение ошибиться в одном. Порядок
    сортируется, чтобы имя ансамбля не зависело от порядка файлов на диске.
    """
    if isinstance(values, (str, Path)):
        values = [values]
    out: list[Path] = []
    for v in values:
        p = Path(v)
        if any(ch in p.name for ch in "*?["):
            got = sorted((p.parent if str(p.parent) else Path(".")).glob(p.name))
            if not got:
                raise LiveError(
                    f"Шаблон {p} не нашёл ни одного файла.\n"
                    f"Что делать: обучите набор — `python -m dwp.train --source real "
                    f"--live-features --extra-features --exact-features --seed N "
                    f"--out models\\ens_0N.pkl`.")
            out.extend(got)
        else:
            out.append(p)
    seen, uniq = set(), []
    for p in out:
        key = str(p.resolve()) if p.exists() else str(p)
        if key not in seen:
            seen.add(key)
            uniq.append(p)
    return uniq


# Поля, которые у участников ансамбля обязаны совпадать. Разойдутся — числа
# будут усредняться по моделям, говорящим о разном, и это не увидеть по
# метрикам: среднее двух осмысленных чисел выглядит осмысленно.
ENSEMBLE_MUST_MATCH = ("state_features", "use_xp", "live_features",
                       "extra_features", "exact_features", "gold_norm")


def load_models(paths) -> dict:
    """Ансамбль из нескольких артефактов, снаружи неотличимый от одного.

    ЗАЧЕМ. Разброс log loss между полностью переобученными моделями,
    отличающимися ТОЛЬКО сидом разбиения, — 0.027; весь драфт даёт 0.0133.
    То есть одиночная модель это розыгрыш лотереи, и усреднение убирает
    розыгрыш. Замерено: −0.0050 log loss, интервал по матчам
    [−0.0063, −0.0037] — больше, чем дало любое добавление признаков.

    Возвращается обычный артефакт: те же ключи, плюс `members`. Всё, что
    читает `state_features`, `id2idx` или `draft_coef`, продолжает работать;
    `predict` и `build_row` замечают `members` и усредняют.
    """
    paths = resolve_models(paths)
    arts = [load_model(p) for p in paths]
    if len(arts) == 1:
        return arts[0]
    first = arts[0]
    for p, a in zip(paths[1:], arts[1:]):
        for key in ENSEMBLE_MUST_MATCH:
            if a.get(key) != first.get(key):
                raise LiveError(
                    f"Модели ансамбля обучены по-разному: у {paths[0].name} "
                    f"{key}={first.get(key)!r}, у {p.name} {key}={a.get(key)!r}.\n"
                    f"Что делать: обучите набор одной командой, меняя только "
                    f"--seed и --out.")
        if a.get("id2idx") != first.get("id2idx"):
            raise LiveError(
                f"У {p.name} другой справочник героев, чем у {paths[0].name}: "
                f"усреднять коэффициенты по разным индексам нельзя.\n"
                f"Что делать: переобучите набор на одном data/heroes.json.")
    # Коэффициенты драфта усредняются ТОЛЬКО ради экранного разбора по героям.
    # В само предсказание идут не они, а логиты каждой модели по отдельности.
    coefs = np.mean([np.asarray(a["draft_coef"], dtype=float) for a in arts], axis=0)
    art = dict(first)
    art.update({
        "members": arts,
        "draft_coef": coefs,
        "draft_intercept": float(np.mean([float(a.get("draft_intercept", 0.0))
                                          for a in arts])),
        "calib_eps": float(max(a.get("calib_eps", 0.0) for a in arts)),
        "name": f"ансамбль({len(arts)}): " + "+".join(p.stem for p in paths),
        "paths": [str(p) for p in paths],
        # Слепым холдаут остаётся, только если он слепой у ВСЕХ участников:
        # достаточно одной модели, видевшей эти матчи, чтобы ансамбль их видел.
        "holdout": all(bool(a.get("holdout")) for a in arts),
        "trained_at": max(float(a.get("trained_at") or 0.0) for a in arts),
        "metrics": {"members": [a.get("metrics") for a in arts]},
    })
    return art



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
    ap = argparse.ArgumentParser(description="Лайв-инференс по Steam Web API.")
    ap.add_argument("--list", action="store_true", help="список топовых live-игр")
    ap.add_argument("--dump", action="store_true", help="напечатать сырой JSON и выйти")
    ap.add_argument("--once", action="store_true", help="один опрос и предсказание")
    ap.add_argument("--watch", action="store_true", help="опрашивать в цикле")
    ap.add_argument("--server-steam-id", type=str, default=None)
    ap.add_argument("--from-file", type=Path, default=None,
                    help="взять ответ из файла вместо сети (для отладки после --dump)")
    ap.add_argument("--interval", type=float, default=15.0)
    ap.add_argument("--model", type=Path, nargs="+", default=None,
                    help="модель или НЕСКОЛЬКО моделей: тогда вероятности "
                         "усредняются (замерено −0.0052 log loss и ECE вдвое "
                         "лучше). Шаблоны раскрываются сами: "
                         "--model models\\ens_*.pkl. По умолчанию — ансамбль "
                         "models\\ens_*.pkl, если он обучен, иначе live_exact.pkl")
    ap.add_argument("--explain-fields", action="store_true",
                    help="показать, какие поля найдены и где")
    ap.add_argument("--log", dest="log", action="store_true", default=None,
                    help="писать каждый опрос в data/live_log (в --watch по "
                         "умолчанию включено); потом `python -m dwp.livecheck`")
    ap.add_argument("--no-log", dest="log", action="store_false",
                    help="не писать лог опросов")
    ap.add_argument("--log-dir", type=Path, default=config.LIVE_LOG_DIR)
    ap.add_argument("--out", type=Path,
                    help="куда записать --dump (UTF-8). Без него дамп идёт в "
                         "stdout, а `> файл` в PowerShell даст UTF-16")
    ap.add_argument("--map", action="store_true",
                    help="рисовать мини-карту: здания, снесённые здания, "
                         "скопления героев (координаты есть в ответе)")
    args = ap.parse_args(argv)
    # В наблюдении лог нужен всегда: без него точность лайва так и останется
    # неизмеренной. Для разовых запусков он бессмысленен — одна строка.
    if args.log is None:
        args.log = bool(args.watch)

    logger: LiveLog | None = None

    def log_note() -> None:
        if logger is not None and logger.rows:
            print(f"\n{logger.note()}\nДальше: python -m dwp.livecheck "
                  f"--log-dir {args.log_dir}")

    try:
        if args.list:
            key = get_key()
            games = top_live_games(key)
            if not games:
                print("Сейчас нет игр в GetTopLiveGame. Это нормально между "
                      "матчами; метод показывает не все игры.")
                return 0
            rows = []
            for g in games:
                r, d = game_team_names(g)
                mmr = g.get("average_mmr") or g.get("avg_mmr")
                rows.append((str(g.get("server_steam_id", "?")),
                             f"{r} vs {d}", str(g.get("league_id") or "-"),
                             str(g.get("spectators", "-")), str(mmr or "-")))
            rows.sort(key=lambda t: -int(t[3]) if t[3].isdigit() else 0)
            w = max(18, min(38, max(len(t[1]) for t in rows) + 1))
            print(f"{'server_steam_id':<20}{'команды':<{w}}"
                  f"{'лига':<8}{'зрителей':>9}{'MMR':>7}")
            print("-" * (20 + w + 24))
            for sid, teams, lg, sp, mmr in rows:
                print(f"{sid:<20}{teams[:w - 1]:<{w}}{lg:<8}{sp:>9}{mmr:>7}")
            if all(t[1].startswith("-") for t in rows):
                print("\nНи у одной игры нет названий команд и team_id — обычно "
                      "это значит,\nчто в списке только паблики. У турнирных "
                      "матчей имена появятся.")
            print(f"\nВсего игр: {len(games)}. Сортировка по числу зрителей.")
            print(f"Дальше: python -m dwp.live --once --server-steam-id "
                  f"{rows[0][0]} --model models\\live.pkl")
            return 0

        payload = None
        if args.from_file is not None:
            # Кодировку определяем по метке порядка байт: `dwp.live --dump >
            # файл` в PowerShell пишет UTF-16, и жёсткий encoding="utf-8" ронял
            # чтение собственного же дампа на первом байте.
            raw = args.from_file.read_bytes()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            payload = json.loads(raw.decode(enc))
        else:
            if not (args.dump or args.once or args.watch):
                ap.print_help()
                return 0
            if not args.server_steam_id:
                print("ОШИБКА: нужен --server-steam-id. Возьмите его из "
                      "`python -m dwp.live --list`.", file=sys.stderr)
                return 2
            key = get_key()
            # Для --watch запрос здесь не нужен: цикл всё равно опрашивает сам.
            # Раньше этот опрос делался всегда, и в режиме наблюдения он был
            # лишним — а главное, он шёл ВНЕ цикла, поэтому первая же сетевая
            # ошибка завершала сессию, минуя всю защиту от обрывов.
            if args.dump or not args.watch:
                payload = realtime_stats(key, args.server_steam_id)

        if args.dump:
            text = json.dumps(payload, ensure_ascii=False, indent=2)
            if args.out:
                # Именно записью в файл, а не через `> файл`: PowerShell
                # перенаправлением пишет UTF-16, и получившийся json потом не
                # открывается ничем, что ждёт UTF-8.
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(text, encoding="utf-8")
                print(f"Сырой ответ -> {args.out} ({len(text)} байт, UTF-8)")
            else:
                print(text[:200000])
            print("\n# Сверьте эти поля с таблицей `--once --explain-fields`: "
                  "пути в live.extract_state взяты по памяти и не проверены "
                  "на живом ответе.", file=sys.stderr)
            return 0

        art = load_models(args.model or default_models())
        tracker = LiveTracker()
        live_view = None
        shown_fields = None
        # Справочник позиций зданий: снесённое здание в ответе обнуляется
        # целиком, включая координаты, поэтому нарисовать его можно только по
        # ранее виденной позиции. Позиции статичны между матчами (проверено).
        mm_book = M.load_book() if args.map else {}
        if args.log:
            logger = LiveLog(args.log_dir, log_model_name(art),
                             art["state_features"])

        def emit(text: str) -> None:
            """Строка поверх живой панели, не ломая её рамку.

            Обычный print внутри контекста rich.live.Live ложится прямо на
            панель: сообщение и рамка перемешивались в одну строку. У Live для
            этого есть своя консоль, она печатает НАД живой областью.
            """
            if live_view is not None:
                live_view.console.print(text)
            else:
                print(text)

        def step(pl: dict) -> None:
            nonlocal shown_fields
            st, rep = extract_state(pl)
            if args.explain_fields:
                # В цикле таблица полей одинакова от опроса к опросу — печатаем
                # её только когда набор найденного изменился.
                seen = frozenset(rep.found) | frozenset(rep.missing)
                if seen != shown_fields:
                    emit(rep.render())
                    shown_fields = seen
            if not st.get("in_progress"):
                msg = (f"Матч ещё не идёт: game_state={st.get('game_state')!r}, "
                       f"нетворс обеих команд нулевой, героев не выбрано.\n"
                       f"Так выглядит стадия драфта — ответ структурно полный, "
                       f"но все значения нули. Предсказывать не по чему.")
                if args.watch:
                    raise NotStarted(msg)
                raise LiveError(msg + "\nЧто делать: дождитесь начала игры "
                                      "и повторите.")
            tracker.update(float(st["game_time"]) / 60.0 if st.get("game_time") is not None
                           else np.nan, st.get("gold_adv", np.nan),
                           st.get("roshan_respawn_timer"))
            df, warns = build_row(art, st, tracker)
            if np.isnan(df.iloc[0]["gold_adv"]) and np.isnan(df.iloc[0]["minute"]):
                raise LiveError(
                    "Не найдено ни игрового времени, ни разницы нетворса — "
                    "предсказывать не по чему.\nЧто делать: `--dump` и сверьте "
                    "структуру ответа с live.extract_state().")
            p, bd = predict(art, df)
            mn = df.iloc[0]["minute"]
            gold = df.iloc[0]["gold_adv"]
            if rep.missing:
                warns.insert(0, f"поля не найдены: {', '.join(rep.missing)}")
            if isinstance(mn, float) and np.isnan(mn):
                warns.insert(0, "игровое время не найдено -> часы и признак "
                                "«минута матча» пусты")
            width = 46
            k = int(round(p * width))
            uni = E._unicode_ok()
            fill, empty, sep = ("\u2588", "\u2591", "\u2502") if uni else ("#", ".", "|")
            rows = []
            mn_known = not (isinstance(mn, float) and np.isnan(mn))
            waiting = shortened = False
            for _, r in bd.head(7).iterrows():
                f = r["признак"]
                v = r["значение"]
                nan = isinstance(v, float) and np.isnan(v)
                at = ready_at(f, tracker)
                # Окно признака бывает уже не пустым, но ещё не полным: значение
                # показываем, а пометку «считано по укороченному» оставляем.
                warming = at is not None and mn_known and mn < at
                if nan and at is not None:
                    val, pending = f"ждём {at:.0f} мин", True
                    waiting = True
                elif nan:
                    val, pending = "нет данных", True
                else:
                    val, pending = float(v), warming
                    shortened = shortened or warming
                rows.append({"label": human(f), "value": val,
                             "contrib": float(r["вклад_логит"]), "pending": pending})
            # Две разные жёлтые ситуации, и путать их нельзя: «ждём N мин» —
            # признак ПУСТ, число на экране посчитано без него; «считано по
            # укороченному» — значение есть, но окно короче обученного.
            if waiting:
                warns.append("жёлтым «ждём N мин» — окна от момента подключения "
                             "ещё нет, признак пуст и в число не вошёл")
            if shortened:
                warns.append("жёлтым — окно признака от момента подключения ещё "
                             "не набралось, значение считано по укороченному")
            tracker.note_p(mn, p)
            if logger is not None:
                logger.write(st, df.iloc[0], p)
            kl = st.get("kills") or {}
            tw = st.get("towers_lost") or {}
            nm = st.get("team_names") or {}
            mmap, mnote = None, ""
            if args.map:
                if M.learn(pl, mm_book):
                    M.save_book(mm_book)
                mmap, n_book, _unk = M.render(pl, mm_book, uni)
                mnote = M.legend(n_book, uni)
            args_ = (p, mn, gold, rows,
                     (nm.get(config.TEAM_RADIANT, ""), nm.get(config.TEAM_DIRE, "")),
                     (kl.get(config.TEAM_RADIANT, float("nan")),
                      kl.get(config.TEAM_DIRE, float("nan"))),
                     (tw.get(config.TEAM_RADIANT, float("nan")),
                      tw.get(config.TEAM_DIRE, float("nan"))),
                     list(tracker.phist), list(dict.fromkeys(warns)), mmap, mnote)
            if D.HAVE_RICH:
                if live_view is not None:
                    live_view.update(D.panel(*args_))
                else:
                    D.Console().print(D.panel(*args_))
                return
            # Запасной путь без rich: тот же смысл, только текстом.
            print()
            print(f"  {D.clock(mn):>7}   нетворс {D._fmt(gold):>8}")
            print(f"  Radiant {p * 100:5.1f}% {fill * k}{empty * (width - k)}"
                  f" {(1 - p) * 100:5.1f}% Dire")
            note = D.reliability_note(mn)
            if note:
                print(f"  {note}")
            if mmap:
                print()
                for line in mmap:
                    print("  " + (line if isinstance(line, str) else line.plain))
                print(f"  {mnote}")
            print(f"\n  {'что двигает число':<26}{'значение':>13}  {'вклад':>7}")
            for r in rows:
                val = r["value"] if isinstance(r["value"], str) else D._fmt(r["value"])
                print(f"  {r['label']:<26}{val:>13}{' ?' if r['pending'] else '  '}"
                      f"{r['contrib']:>+7.3f}")
            for w in dict.fromkeys(warns):
                print(f"  ! {w}")

        if args.watch and args.from_file is None:
            interval = max(WATCH_MIN_INTERVAL, args.interval)
            if interval != args.interval:
                print(f"--interval {args.interval:g} слишком мал, поднят до "
                      f"{interval:g} с: чаще опрашивать Steam незачем, ответ "
                      f"всё равно обновляется реже.")
            key = get_key()
            print(f"Опрос каждые {interval:.0f} с. Ctrl+C для выхода. "
                  f"GetRealtimeStats отстаёт от эфира на задержку трансляции.")
            fails = 0

            def poll_once() -> None:
                """Один опрос. Сессия не должна умирать от одного пакета.

                Раньше любая ошибка внутри цикла улетала наружу и завершала
                программу: один обрыв связи или старт во время драфта — и
                смотреть матч больше нечем.
                """
                nonlocal fails
                try:
                    step(realtime_stats(key, args.server_steam_id))
                    fails = 0
                    return
                except NotStarted as e:
                    fails = 0
                    emit(f"[ждём начала матча] {str(e).splitlines()[0]}")
                    return
                except LiveError as e:
                    fails += 1
                    emit(f"[опрос {fails}/{WATCH_MAX_FAILS} не удался] {e}")
                except Exception as e:                # noqa: BLE001
                    fails += 1
                    emit(f"[сбой {fails}/{WATCH_MAX_FAILS}] {type(e).__name__}: {e}\n"
                         f"Это баг dwp: сохраните ответ через --dump, по нему "
                         f"воспроизводится.")
                if fails >= WATCH_MAX_FAILS:
                    raise LiveError(
                        f"{WATCH_MAX_FAILS} опросов подряд не удались "
                        f"({fails * interval:.0f} с без данных).\n"
                        f"Что делать: проверьте сеть и что матч ещё идёт — "
                        f"после его конца server_steam_id перестаёт отвечать.")

            def loop() -> None:
                while True:
                    poll_once()
                    time.sleep(interval)

            if D.HAVE_RICH:
                from rich.live import Live
                with Live(refresh_per_second=4, screen=False) as lv:
                    live_view = lv
                    loop()
            else:
                loop()
        else:
            step(payload)
        return 0

    except LiveError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nОстановлено.")
        return 130
    except Exception as e:                            # noqa: BLE001
        # Схема недокументирована и может смениться в любой момент. Трейсбек
        # в лицо посреди матча ничем не помогает, а сохранённый ответ помогает.
        print(f"ОШИБКА: неожиданный сбой в dwp: {type(e).__name__}: {e}\n"
              f"Что делать: это баг dwp, а не ваша конфигурация. Сохраните "
              f"ответ (`python -m dwp.live --dump --server-steam-id ...`) — "
              f"по нему воспроизводится.", file=sys.stderr)
        return 3
    finally:
        # Лог полезен ровно тогда, когда сессия оборвалась, поэтому путь к нему
        # печатается на любом выходе, включая Ctrl+C и сбой.
        log_note()


if __name__ == "__main__":
    sys.exit(main())
