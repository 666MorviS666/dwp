"""Веб-панель лайва: список матчей, выбор, аналитика, автозапись.

Одной командой:

    python -m dwp.web

Дальше всё в браузере на http://127.0.0.1:8765/ — список идущих игр,
клик по строке, панель. Ключ Steam берётся из STEAM_API_KEY.

Зачем отдельно от live.py. В консоли герой на мини-карте — цифра, а цифра
не говорит, кто там стоит; в браузере на том же месте стоит иконка. Плюс
появляется место для разбора драфта по героям, который в 80 колонок не
влезал, и для карты, на которой видно лес и линии.

Что тут ВАЖНО не сделать: вторую реализацию инференса. Числа считает ровно
тот же путь, что и консоль, — `live.extract_state` -> `live.build_row` ->
`live.predict`, и пишет их тот же `livelog.LiveLog`. Этот модуль только
опрашивает, складывает результат в JSON и отдаёт страницу.

Зависимостей не добавляет: сервер на http.server из стандартной библиотеки.

Без сети, на сохранённом ответе:
    python -m dwp.web --from-file data\\live_dump_late.json

ИКОНКИ ГЕРОЕВ грузятся с CDN Valve. Нет сети или адрес сменился — на их
месте останется короткое имя, страница не ломается.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import threading
import time
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import accuracy as ACC, builds as BLD, config, items as I, live
from . import jobs as JOBS, killfeed as KF, matchups as MU, minimap as M, stratz as ST
from . import deaths as DTH, verdict as VD
from .collect import OpenDotaClient
from .livelog import LiveLog

HERO_IMG = ("https://cdn.cloudflare.steamstatic.com/apps/dota2/images/"
            "dota_react/heroes/{slug}.png")
GAMES_TTL = 20.0          # список игр меняется медленно, чаще дёргать незачем
# Бюджет одного похода за списком игр. Обычная политика проекта — 4 попытки
# по 20 с — для страницы не годится: ЗАМЕРЕНО на молчащем Steam, обработчик
# держал ответ 96 секунд, а браузер за это время присылал ещё полсотни таких
# же запросов. Здесь лучше быстро сказать «Steam не отвечает».
GAMES_TIMEOUT = 6.0
GAMES_RETRIES = 2
STRATZ_LIVE_TTL = 30.0    # их вероятность обновляется не чаще, лимит мал
GUESS_PATH = config.DATA_DIR / "blind" / "live_guesses.csv"

# Как часто опрашивать Steam. ЗАМЕРЕНО НА ЖИВОМ МАТЧЕ: GetRealtimeStats
# обновляется примерно раз в 1.2 с (40 опросов с шагом 1 с дали 33 различных
# game_time), ответ приходит за 0.29 с. Прежние 15 с выбрасывали двенадцать
# обновлений из тринадцати — карта дёргалась не потому, что герои прыгают, а
# потому что мы почти не смотрели.
#
# Про лимит. У Steam Web API это 100 000 запросов в сутки на ключ, то есть
# 1.16 в секунду В СРЕДНЕМ ЗА СУТКИ. Опрос раз в 2 с за 45-минутный матч —
# 1350 запросов, полтора процента суточной квоты; даже круглосуточная
# слежка (43 200 в сутки) вдвое ниже лимита. Ниже секунды опускаться
# бессмысленно: источник всё равно не обновится.
MIN_POLL = 1.0
DEFAULT_POLL = 2.0
# А вот В ЛОГ чаще писать незачем: строки внутри матча почти дублируют друг
# друга, эффективный размер выборки — число матчей, а не строк. 15 с дают
# ~200 строк на матч, 2 с дали бы 1350 при той же ценности.
DEFAULT_LOG_EVERY = 15.0


def hero_pick_counts() -> dict[int, int]:
    """Сколько раз каждый герой встречался в обучающей выборке.

    Нужно, чтобы к коэффициенту можно было приписать его ошибку. Без этого
    «+0.284 у Lone Druid» читается как «модель считает героя сильным», а на
    деле это 384 матча из 7434, то есть примерно 2.8 стандартной ошибки на
    одном из 127 проверенных героев — столько выдаёт и случайность.

    Считается по выгрузке `data/export_draft.csv.gz` (столбцы hNNN, ±1 за
    героя): сумма модулей по столбцу и есть число матчей с этим героем. Нет
    выгрузки — вернём пусто, панель просто не покажет выборку.
    """
    src = config.DATA_DIR / "export_draft.csv.gz"
    if not src.exists():
        return {}
    try:
        import pandas as pd
        d = pd.read_csv(src)
    except Exception:                                       # noqa: BLE001
        return {}
    out = {}
    for c in d.columns:
        if c.startswith("h") and c[1:].isdigit():
            out[int(c[1:])] = int(d[c].abs().sum())
    return out


def hero_slugs() -> dict[int, tuple[str, str]]:
    """hero_id -> (короткое имя для картинки, читаемое имя)."""
    out: dict[int, tuple[str, str]] = {}
    if not config.HEROES_PATH.exists():
        return out
    try:
        data = json.loads(config.HEROES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return out
    for h in data:
        try:
            hid = int(h["id"])
        except (KeyError, TypeError, ValueError):
            continue
        slug = str(h.get("name", "")).replace("npc_dota_hero_", "")
        out[hid] = (slug, h.get("localized_name") or slug or str(hid))
    return out


# Во сколько раз угол должен быть дальше от центра, чтобы его вставили.
# ЗАМЕРЕНО на двух состояниях справочника, и это важно: с неполным
# справочником крайние вышки далеко друг от друга и выигрыш большой, с
# полным они стоят у самого угла и выигрыш падает почти вдвое.
#     справочник 31/36:  верх 1.25   низ 1.31   мид 0.96
#     справочник 36/36:  верх 1.09   низ 1.13   мид 0.93
# Порог 1.15 работал только на неполном; на полном линии распрямлялись и
# срезали угол. 1.05 разделяет оба замера с запасом с обеих сторон.
CORNER_MIN_GAIN = 1.05


def lane_paths(book: dict[str, dict]) -> list[list[dict]]:
    """Ломаные линий по РЕАЛЬНЫМ позициям вышек из справочника.

    Порядок: трон Radiant -> его t4/t3/t2/t1 -> t1/t2/t3/t4 Dire -> трон Dire.

    ИЗГИБ НА УГЛУ. Верхняя и нижняя линии идут не по прямой: верхняя
    поднимается по левому краю и сворачивает вправо у верхнего левого угла,
    нижняя идёт по низу и сворачивает вверх у правого нижнего. Прямая между
    крайними вышками срезала бы угол через центр карты — так и было, и карта
    из-за этого не читалась.

    Угол не рисуется от руки, а выводится из тех же координат: у пары
    (последняя вышка Radiant, первая вышка Dire) два кандидата — (ax, by) и
    (bx, ay). Настоящий угол тот, что ДАЛЬШЕ от центра, потому что линии
    загибаются наружу. Вставляется он только если выигрыш по радиусу больше
    CORNER_MIN_GAIN — у мида оба кандидата не дальше самих вышек, и туда
    ничего не вставляется (см. замеры у константы).
    """
    def find(team: int, typ: int, tier: int, lane: int) -> dict | None:
        for b in book.values():
            if (b["team"] == team and b["type"] == typ
                    and b["tier"] == tier and b["lane"] == lane):
                return b
        return None

    def rad(p) -> float:
        return (p["x"] ** 2 + p["y"] ** 2) ** 0.5

    out = []
    for lane in (1, 2, 3):
        side: dict[int, list[dict]] = {}
        for team, tiers in ((config.TEAM_RADIANT, (4, 3, 2, 1)),
                            (config.TEAM_DIRE, (1, 2, 3, 4))):
            got = []
            fort = find(team, 2, 0, 0)
            if fort is not None and team == config.TEAM_RADIANT:
                got.append(fort)
            for tier in tiers:
                b = find(team, 0, tier, lane)
                if b is not None:
                    got.append(b)
            if fort is not None and team == config.TEAM_DIRE:
                got.append(fort)
            side[team] = got
        r, d = side[config.TEAM_RADIANT], side[config.TEAM_DIRE]
        pts = list(r)
        if r and d:
            a, b = r[-1], d[0]
            best = max(({"x": a["x"], "y": b["y"]}, {"x": b["x"], "y": a["y"]}),
                       key=rad)
            if rad(best) > CORNER_MIN_GAIN * max(rad(a), rad(b)):
                pts.append(best)
        pts += d
        if len(pts) >= 2:
            out.append([{"x": p["x"], "y": p["y"]} for p in pts])
    return out


class Poller(threading.Thread):
    """Опрос в фоне. Цель можно менять на ходу — из браузера."""

    daemon = True

    def __init__(self, art: dict, model_name: str, key: str | None,
                 interval: float, from_file: Path | None, log_dir: Path,
                 do_log: bool, log_every: float = DEFAULT_LOG_EVERY):
        super().__init__(name="dwp-poller")
        self.art = art
        self.model_name = model_name
        self.key = key
        self.interval = interval
        self.log_every = log_every
        self.last_log = 0.0
        self.from_file = from_file
        self.log_dir = log_dir
        self.do_log = do_log
        self.heroes = hero_slugs()
        self.picks = hero_pick_counts()
        self.book = M.load_book()
        self.lanes = lane_paths(self.book)
        # Справочник предметов и таблица шансов отыграться. Оба
        # необязательны: нет — блок сборок просто не рисуется. Показывать
        # вместо них нули или выдуманные проценты нельзя.
        self.items = I.load()
        self.comeback = BLD.load_table()
        # Коридора неопределённости здесь больше нет: блок «перспектива»
        # с панели убран, а таблицу для раздела «данные и обучение»
        # проверяет сам `jobs.state()` по наличию файла. Держать её
        # загруженной в опросчике значило бы читать 38 КБ на старте ради
        # никого. Сама таблица жива — её считает и показывает
        # `python -m dwp.forecast`.
        # Правило вердикта и таблица его попаданий. Подобрано на половине A
        # слепого холдаута, померено на половине B (`dwp.verdict --tune`).
        # Нет файла — блок вердикта не рисуется вовсе: показать сторону без
        # доли, с которой такие вердикты сбывались, значит показать
        # уверенность, которой никто не мерил.
        self.vtable = VD.load_table()

        self.lock = threading.Lock()
        self.sid: str | None = None
        # Номер поколения: растёт на каждом переключении матча. Опрос
        # начинается со снятия номера и в конце публикует снимок ТОЛЬКО
        # если номер не сменился. Без этого переключение, попавшее внутрь
        # опроса (а он длится ~0.3 с), публиковало снимок ПРОШЛОГО матча
        # уже с новым sid — то есть панель молча показывала чужие числа,
        # выглядящие свежими.
        self.gen = 0
        self.tracker = live.LiveTracker()
        self.feed = KF.Feed()
        self.dead = DTH.Deaths()
        self.logger: LiveLog | None = None
        self.state: dict = {"ok": False, "idle": True,
                            "error": "матч не выбран"}
        self.games: list[dict] = []
        self.games_at = 0.0
        self.games_err: str | None = None
        self.games_busy = False
        self.stop = threading.Event()
        self.wake = threading.Event()
        self.fails = 0
        # Противостояния героев (OpenDota). Считаются в отдельном потоке:
        # десять запросов с троттлингом занимают секунды, а панель обязана
        # рисоваться сразу и без них.
        self.mu_key: tuple = ()
        self.mu: dict | None = None
        self.mu_state = "нет составов"
        # Вероятность Stratz по тому же матчу. Спрашивается реже нашего
        # опроса: их число обновляется не чаще, а лимит по секунде мал.
        # На экран не выводится — уходит колонкой в лайв-лог, чтобы потом
        # можно было сравнить, кто врал больше (`dwp.livecheck`).
        self.sz: dict | None = None
        self.sz_at = 0.0
        self.sz_mid: int | None = None
        self.sz_state = "не спрашивали"
        self.sz_client: ST.Client | None = None

    # --- выбор матча ---------------------------------------------------

    def watch(self, sid: str | None) -> None:
        """Переключиться на матч. Трекер и лог начинаются заново: окна
        производных считаются от момента подключения, и тащить их с
        прошлого матча значило бы показать выдуманный темп."""
        with self.lock:
            self.sid = sid
            self.gen += 1
            self.tracker = live.LiveTracker()
            # Лента событий тоже начинается заново: её события — это разница
            # счётчиков, и разница между двумя РАЗНЫМИ матчами дала бы залп
            # выдуманных фрагов на первом же опросе.
            self.feed = KF.Feed()
            # Кто мёртв — тоже считается разницей счётчиков и потому
            # начинается заново вместе с лентой.
            self.dead = DTH.Deaths()
            self.logger = (LiveLog(self.log_dir, self.model_name,
                                   self.art["state_features"])
                           if (self.do_log and sid) else None)
            self.fails = 0
            self.state = {"ok": False, "idle": sid is None,
                          "error": ("матч не выбран" if sid is None
                                    else "первый опрос ещё не прошёл"),
                          "sid": sid}
        self.wake.set()

    # --- список игр ----------------------------------------------------

    def game_list(self) -> tuple[list[dict], str | None]:
        now = time.time()
        with self.lock:
            # Свежесть по времени, а НЕ по непустоте: пустой список — штатный
            # ответ между матчами, и проверка `if self.games` заставляла
            # ходить в Steam на каждый опрос страницы, то есть раз в 2 с.
            fresh = self.games_at > 0 and (now - self.games_at) < GAMES_TTL
            if fresh:
                return list(self.games), self.games_err
            # За списком идёт РОВНО ОДИН запрос за раз. Без этого медленный
            # Steam превращался в лавину: страница опрашивает раз в 2 с,
            # каждый опрос видел кэш протухшим и заводил свой поход в сеть.
            if self.games_busy:
                return list(self.games), self.games_err
            self.games_busy = True
        if self.key is None:
            with self.lock:
                self.games_busy = False
            return [], ("нет ключа Steam: список игр недоступен. "
                        "Работает только --from-file.")
        try:
            raw = live.top_live_games(self.key, timeout=GAMES_TIMEOUT,
                                      retries=GAMES_RETRIES)
            err = None
        except live.LiveError as e:
            raw, err = [], str(e)
        except Exception as e:                              # noqa: BLE001
            # Обработчик страницы не имеет права падать из-за списка игр:
            # без него панель ещё работает, а без страницы — уже нет.
            raw, err = [], f"{type(e).__name__}: {e}"
        finally:
            with self.lock:
                self.games_busy = False
        rows = []
        for g in raw:
            r, d = live.game_team_names(g)
            hero = [self.heroes.get(int(p.get("heroid") or 0), ("", ""))
                    for p in (g.get("players") or [])]
            rows.append({
                "sid": str(g.get("server_steam_id", "")),
                "radiant": r, "dire": d,
                "league": g.get("league_id") or 0,
                "spectators": g.get("spectators") or 0,
                "mmr": g.get("average_mmr") or g.get("avg_mmr") or 0,
                "heroes": [{"slug": s, "name": n,
                            "img": HERO_IMG.format(slug=s) if s else ""}
                           for s, n in hero],
            })
        rows.sort(key=lambda r: -int(r["spectators"] or 0))
        with self.lock:
            self.games, self.games_at, self.games_err = rows, now, err
        return rows, err

    # --- сбор снимка ---------------------------------------------------

    def snapshot(self, payload: dict) -> dict:
        art = self.art
        st, rep = live.extract_state(payload)
        if not st.get("in_progress"):
            return {"ok": False, "waiting": True, "sid": self.sid,
                    "error": f"матч ещё не идёт (game_state="
                             f"{st.get('game_state')!r}) — похоже на драфт, ждём"}
        gt = st.get("game_time")
        self.tracker.update(float(gt) / 60.0 if gt is not None else np.nan,
                            st.get("gold_adv", np.nan),
                            st.get("roshan_respawn_timer"))
        df, warns = live.build_row(art, st, self.tracker)
        # Разбор вкладов (второе значение) панели больше не нужен: блок «что
        # двигает число» снят. Считать его на каждый опрос ради выброшенного
        # JSON незачем — сам разбор жив в `python -m dwp.live --once`.
        p, _bd = live.predict(art, df)
        mn = df.iloc[0]["minute"]
        mn_known = not (isinstance(mn, float) and np.isnan(mn))
        self.tracker.note_p(mn, p)

        sz = self.stratz_live(st.get("match_id"))
        # Опрашиваем часто (ради карты), а пишем редко: строки внутри матча
        # почти дублируют друг друга, и раздувать лог в семь раз значит
        # только сделать вид, что выборка выросла.
        now = time.time()
        if self.logger is not None and (now - self.last_log) >= self.log_every:
            self.last_log = now
            self.logger.write(st, df.iloc[0], p, stratz=sz)

        if M.learn(payload, self.book):
            M.save_book(self.book)
            self.lanes = lane_paths(self.book)
        if rep.missing:
            warns.insert(0, f"поля не найдены: {', '.join(rep.missing)}")

        kl = st.get("kills") or {}
        tw = st.get("towers_lost") or {}
        nm = st.get("team_names") or {}

        def pair(d, default=None):
            return [d.get(config.TEAM_RADIANT, default),
                    d.get(config.TEAM_DIRE, default)]

        self.feed.update(payload, mn if mn_known else float("nan"))
        # Вердикт считается из ТОЙ ЖЕ истории, что рисует кривую: второй
        # реализации ни инференса, ни истории быть не должно. Правило
        # работает по игровому времени, поэтому частота опроса на него не
        # влияет — то же правило прогонялось поминутно на холдауте.
        vd = VD.live_verdict(
            self.vtable, list(self.tracker.mhist), list(self.tracker.phist),
            names=(nm.get(config.TEAM_RADIANT) or "Radiant",
                   nm.get(config.TEAM_DIRE) or "Dire"))
        builds = self.builds_block(payload, df)

        return {
            "ok": True, "ts": time.time(), "sid": self.sid,
            "match_id": st.get("match_id"), "p": float(p),
            "minute": None if not mn_known else float(mn),
            "clock": live.D.clock(mn),
            "names": [nm.get(config.TEAM_RADIANT) or "Radiant",
                      nm.get(config.TEAM_DIRE) or "Dire"],
            "kills": pair(kl), "towers_lost": pair(tw),
            "rax_lost": pair(st.get("rax_lost") or {}),
            "gold_adv": (None if np.isnan(df.iloc[0]["gold_adv"])
                         else float(df.iloc[0]["gold_adv"])),
            # История вероятности в снимок больше не кладётся: кривую с
            # панели убрали, а вердикт считается из той же истории ЗДЕСЬ,
            # на сервере. Отдавать браузеру четыреста чисел на каждый опрос
            # ради того, чтобы он их выбросил, незачем.
            "warns": list(dict.fromkeys(warns)),
            "verdict": vd,
            "builds": builds,
            "draft": self.draft_block(st, df),
            "map": self.map_block(payload, mn if mn_known else float("nan")),
            "feed": self.feed_block(),
            "model": self.model_name,
            "n_models": len(live.members(art)),
            "log": (self.logger.note() if self.logger is not None
                    else "запись выключена"),
            # Числа Stratz в снимке нет намеренно: второй процент рядом с
            # нашим — это два спорящих числа на экране. Спрашивать их всё
            # равно продолжаем: колонка p_stratz в лайв-логе нужна, чтобы
            # потом сравнить, кто врал больше (`dwp.livecheck`).
        }

    def face(self, hero_id: int) -> dict:
        slug, name = self.heroes.get(int(hero_id or 0), ("", ""))
        return {"id": int(hero_id or 0), "slug": slug,
                "name": name or (str(hero_id) if hero_id else "?"),
                "img": HERO_IMG.format(slug=slug) if slug else ""}

    def feed_block(self) -> dict:
        """Лента событий портретами.

        Журнала событий в GetRealtimeStats нет — лента ВЫВОДИТСЯ из разницы
        счётчиков между опросами (см. dwp/killfeed.py). Поэтому каждое
        событие несёт флаг `certain`: точная атрибуция бывает только когда
        в окне одна смерть и один фраг. Замес показывается замесом, а не
        выдуманными парами.
        """
        out = []
        for e in self.feed.recent(24):
            item = {k: e.get(k) for k in
                    ("id", "kind", "minute", "certain", "note", "team",
                     "tier", "lane")}
            for side in ("victims", "killers", "assists"):
                item[side] = [{**self.face(f.get("hero")),
                               "team": f.get("team")}
                              for f in (e.get(side) or [])]
            out.append(item)
        return {"events": out, "n": len(self.feed.events)}

    def builds_block(self, payload: dict, df) -> dict | None:
        """Сборки обеих команд и эмпирические шансы отстающих.

        Модель этого не знает: предметов в её признаках нет вовсе. Поэтому
        блок отделён от «что двигает число» и подписан — иначе зритель
        решит, что шесть слотов уже учтены в проценте.

        Шансы берутся из таблицы, посчитанной по 7483 матчам
        (`python -m dwp.builds --table`), а не из модели. Нет таблицы или
        ситуация вне когорт — блок про шансы просто не рисуется.
        """
        if not self.items.ok:
            return None
        names = {hid: nm for hid, (_slug, nm) in self.heroes.items()}
        got = BLD.live_builds(payload, self.items, names)
        if not got:
            return None
        teams = []
        for tn in (config.TEAM_RADIANT, config.TEAM_DIRE):
            side = got.get(tn)
            if side is None:
                return None
            players = []
            for q in sorted(side["players"], key=lambda x: -(x.net_worth or 0)):
                slug, _ = self.heroes.get(q.hero_id, ("", ""))
                players.append({
                    "id": q.hero_id, "name": q.name,
                    "img": HERO_IMG.format(slug=slug) if slug else "",
                    "level": q.level, "slots": q.slots, "kit": q.kit,
                    "value": q.value, "big": q.big,
                    "nw": q.net_worth, "gold": q.gold,
                    "core": q is side["core"],
                    "items": [None if it is None else
                              {"name": it["name"], "img": it["img"],
                               "cost": it["cost"], "big": it["big"],
                               "consumable": it["consumable"]}
                              for it in q.items],
                })
            teams.append({
                "value": side["value"], "big": side["big"],
                "unspent": side["unspent"],
                "core_big": side["core"].big, "core_kit": side["core"].kit,
                "core_name": side["core"].name,
                "players": players,
            })
        gold = df.iloc[0].get("gold_adv", np.nan)
        minute = df.iloc[0].get("minute", np.nan)
        chance = None
        if (not np.isnan(gold) and not np.isnan(minute)
                and self.comeback is not None):
            trailing = 0 if gold < 0 else 1        # 0 = Radiant отстаёт
            got_ch = BLD.lookup(self.comeback, float(minute), float(gold),
                                teams[trailing]["core_big"],
                                teams[1 - trailing]["core_big"])
            if got_ch:
                mdl = self.comeback.get("model") or {}
                chance = {**got_ch, "trailing": trailing,
                          "deficit": abs(float(gold)),
                          "core_big": teams[trailing]["core_big"],
                          "lead_core_big": teams[1 - trailing]["core_big"],
                          "core_name": teams[trailing]["core_name"],
                          "n_matches": self.comeback.get("n_matches"),
                          "model_auc": mdl.get("auc"),
                          "model_n": mdl.get("n_test_matches")}
        return {"teams": teams, "chance": chance,
                "value_adv": teams[0]["value"] - teams[1]["value"]}

    def draft_block(self, st: dict, df) -> dict:
        """Разбор драфта по героям.

        ВНИМАНИЕ на честность подписи. Коэффициенты — ЛИНЕЙНЫЙ логит
        логистической регрессии, до калибровки; число, которое панель
        показывает крупно (`logit`), берётся ПОСЛЕ CalibratedClassifierCV.
        Сумма коэффициентов ему НЕ равна, и складывать их зрителю нельзя —
        та же оговорка в explain.explain_draft. Поэтому крупно стоит
        калиброванное число, а коэффициенты лежат под раскрытием как
        разбор, а не как слагаемые.
        """
        art = self.art
        rad = [int(h) for h in (st.get("heroes_radiant") or []) if h]
        dire = [int(h) for h in (st.get("heroes_dire") or []) if h]
        dl = df.iloc[0].get("draft_logit", np.nan)
        out = {"logit": None if np.isnan(dl) else float(dl),
               "radiant": [], "dire": [], "elo": None, "intercept": None,
               "ok": False}
        if not (rad and dire) or "draft_coef" not in art:
            return out
        id2idx = art.get("id2idx") or {}
        coefs = art["draft_coef"]
        for side, ids, sign in (("radiant", rad, 1.0), ("dire", dire, -1.0)):
            for hid in ids:
                i = id2idx.get(int(hid))
                slug, name = self.heroes.get(hid, ("", str(hid)))
                c = None if i is None else float(coefs[i]) * sign
                n = self.picks.get(hid, 0)
                # Грубая оценка ошибки коэффициента при признаке ±1,
                # встреченном в n матчах: информация ~ n/4, значит se ~ 2/vn.
                # Регуляризация коэффициенты поджимает, так что это верхняя
                # оценка — но порядок величины она даёт верный, а без неё
                # число читается как факт.
                se = (2.0 / n ** 0.5) if n > 0 else None
                out[side].append({
                    "id": hid, "slug": slug, "name": name,
                    "img": HERO_IMG.format(slug=slug) if slug else "",
                    "coef": c, "known": i is not None,
                    "picks": n, "se": se,
                    "sig": bool(c is not None and se and abs(c) > 2 * se),
                })
        elo = st.get("elo_diff")
        if elo is not None and len(coefs) > len(id2idx):
            e = float(coefs[len(id2idx)]) * float(elo) / 400.0
            out["elo"] = {
                "value": float(elo), "coef": e,
                # Что именно за рейтинг — зависит от артефакта. Подписать
                # рейтинг игроков «Elo команд» значило бы соврать в
                # единственном месте, где виден его вклад.
                "label": ("рейтинг игроков"
                          if art.get("rating_kind") == "player"
                          else "Elo команд"),
            }
        out["intercept"] = float(art.get("draft_intercept", 0.0))
        out["ok"] = True
        # Сама сетка 5x5 в браузер не уезжает: блок «кто против кого» снят,
        # а от неё нужна одна строка — состав против состава, — и её
        # собирает draft_why. Это ~4 КБ JSON на каждый опрос экономии.
        out["why"] = self.draft_why(out, self.matchups(rad, dire))
        return out

    def draft_why(self, d: dict, mu: dict | None = None) -> list[dict]:
        """ПОЧЕМУ драфт сильнее — списком причин, каждая со своим источником.

        Вердикт «у Radiant сильнее на 4%» без «почему» читается как оракул.
        А «почему» тут собирается из трёх РАЗНЫХ источников, и смешивать их
        в одно число нельзя:

          модель   одиночные коэффициенты героев, без парных членов;
          Stratz   статистика пар — контры и синергии, миллионы пабликов;
          рейтинг  история побед — по составу игроков либо по team_id,
                   смотря чем обучена модель (art["rating_kind"]).

        Причём вклад Stratz в ЧИСЛО МОДЕЛИ равен нулю: парные данные в неё
        не входят (замерено — AUC признака 0.4964, статистика пабликов на
        про не переносится). Поэтому у каждой строки стоит, учтена она в
        проценте или это второй, независимый голос.
        """
        why: list[dict] = []
        if d.get("elo") and abs(float(d["elo"]["value"])) >= 15:
            e = d["elo"]
            # Подпись берётся из артефакта, а не пишется здесь руками:
            # моделей две породы, и назвать рейтинг игроков «Elo команд»
            # значило бы соврать в единственной строке, где он виден.
            player = self.art.get("rating_kind") == "player"
            why.append({
                "text": f"{e.get('label') or 'рейтинг'}: {e['value']:+.0f}",
                "weight": float(e["coef"]), "source": "модель",
                "counted": True, "sig": True,
                "note": ("среднее Elo пяти игроков по account_id — состав, "
                         "а не тег команды" if player else
                         "Elo по team_id, а не по составу игроков")})
        heroes = [(h, "Radiant") for h in d.get("radiant") or []]
        heroes += [(h, "Dire") for h in d.get("dire") or []]
        strong = sorted(((h, w) for h, w in heroes
                         if h.get("coef") is not None and h.get("sig")),
                        key=lambda p: -abs(p[0]["coef"]))[:3]
        for h, who in strong:
            # Знак `coef` — это вклад В ПОЛЬЗУ RADIANT (в draft_block он уже
            # умножен на сторону), а НЕ оценка героя. Без подписи «у кого он
            # взят» строка читается наоборот: «Bristleback — в пользу
            # Radiant» у героя, которого взял Dire, выглядит как «модель
            # считает Bristleback сильным за Radiant». На деле у него самый
            # низкий одиночный коэффициент из 127 (по выборке команда с ним
            # выигрывает 45.8% при 1210 матчах), и плюс Radiant тут ровно
            # потому, что взял его соперник.
            side = "Radiant" if h["coef"] > 0 else "Dire"
            own = ("по выборке он в плюс своей стороне" if side == who else
                   "по выборке он в минус своей стороне — отсюда плюс "
                   "сопернику")
            why.append({
                "text": f"{h['name']} у {who} — в пользу {side}",
                "weight": float(h["coef"]), "source": "модель",
                "counted": True, "sig": True,
                "note": f"{h['picks']} матчей в обучении, ошибка "
                        f"±{h['se']:.3f}; {own}"})
        weak = sum(1 for h, _ in heroes
                   if h.get("coef") is not None and not h.get("sig"))
        if weak:
            why.append({
                "text": f"остальные {weak} героев — в пределах шума",
                "weight": None, "source": "модель", "counted": True,
                "sig": False,
                "note": "коэффициент короче двух своих стандартных ошибок; "
                        "проверено 127 героев сразу, несколько «значимых» "
                        "появятся и случайно"})
        mu = mu or {}
        team = mu.get("team") or {}
        if team.get("wr") is not None:
            dv = (float(team["wr"]) - 0.5) * 100
            why.append({
                "text": (f"состав против состава: {dv:+.1f} п.п."
                         if team.get("sig") else
                         "состав против состава: неотличимо от ровного"),
                "weight": None, "source": mu.get("source") or "пары",
                "counted": False, "sig": bool(team.get("sig")),
                "note": f"{team.get('games', 0):,} матчей по "
                        f"{team.get('pairs', 0)} парам; в процент модели НЕ "
                        f"входит".replace(",", " ")})
        syn = mu.get("synergy") or {}
        for key, name in (("radiant", "Radiant"), ("dire", "Dire")):
            s = syn.get(key) or {}
            if s.get("wr") is None or not s.get("sig"):
                continue
            dv = (float(s["wr"]) - 0.5) * 100
            why.append({
                "text": f"синергия внутри {name}: {dv:+.1f} п.п.",
                "weight": None, "source": mu.get("source") or "пары",
                "counted": False, "sig": True,
                "note": f"{s.get('games', 0):,} матчей по {s.get('pairs', 0)} "
                        f"парам".replace(",", " ")})
        return why

    def stratz_live(self, match_id) -> dict | None:
        """Вероятность Stratz по этому же матчу, если он у них есть.

        None — обычное дело, а не сбой: живьём Stratz ведёт лиговые матчи и
        высокий рейтинг, рядовой паблик к ним может не попасть вовсе.
        Ошибка сети тоже не должна ронять опрос: наше число считается
        независимо и без Stratz.
        """
        if not match_id or not ST.token():
            self.sz_state = ("нет токена Stratz" if not ST.token()
                             else "матч не опознан")
            return None
        # Кэш держится ЗА match_id. Без этого переключение на другой матч
        # внутри окна TTL отдавало бы вероятность прошлого матча — и она
        # уходила бы в лайв-лог колонкой p_stratz, то есть в сравнение
        # «кто врал больше» подставлялось бы чужое число.
        mid = int(match_id)
        now = time.time()
        if self.sz_mid != mid:
            self.sz_mid, self.sz, self.sz_at = mid, None, 0.0
        elif (now - self.sz_at) < STRATZ_LIVE_TTL:
            return self.sz
        self.sz_at = now
        try:
            if self.sz_client is None:
                self.sz_client = ST.Client()
            got = ST.live_win_rate(int(match_id), self.sz_client)
        except ST.StratzError as e:
            self.sz_state = f"Stratz не ответил: {e}"
            return None
        except Exception as e:                              # noqa: BLE001
            self.sz_state = f"Stratz: {type(e).__name__}: {e}"
            return None
        if got is None:
            self.sz, self.sz_state = None, "Stratz этот матч не ведёт"
            return None
        self.sz = got
        self.sz_state = f"Stratz, {got['source']}"
        return got

    def matchups(self, rad: list[int], dire: list[int]) -> dict | None:
        """Сетка противостояний. Считается один раз на состав, в фоне.

        Источников два, и они несопоставимы по объёму: у Stratz на пару
        героев порядка 3300 матчей (ошибка 0.9 п.п.), у OpenDota — 81
        (ошибка 5.6 п.п.). Поэтому Stratz первый, OpenDota — запасной, и
        источник всегда подписан: на первом вопрос «кто кого контрит» имеет
        ответ, на втором почти нет.
        """
        key = (tuple(rad), tuple(dire))
        if key != self.mu_key:
            self.mu_key, self.mu = key, None
            self.mu_state = "загружаю…"
            need = rad + dire
            # Сначала ТОЛЬКО кэш: если он есть, сетка появится в этом же
            # снимке, без сети и без ожидания. Кэш проверяем независимо от
            # токена — токен нужен, чтобы ДОБРАТЬ, а не чтобы прочитать уже
            # добранное.
            tbl = ST.table(need, client=None)
            if len(tbl) == len(need):
                self.mu = ST.grid(rad, dire, tbl)
                self.mu_state = "Stratz, из кэша"
                return self.mu
            tbl = MU.table(need, client=None)
            if len(tbl) == len(need):
                self.mu = MU.grid(rad, dire, tbl)
                self.mu_state = "OpenDota, из кэша"
            threading.Thread(target=self._fetch_matchups, args=(key,),
                             daemon=True).start()
        return self.mu

    def _fetch_matchups(self, key: tuple) -> None:
        rad, dire = key
        need = list(rad) + list(dire)
        if ST.token():
            try:
                tbl = ST.table(need, client=ST.Client())
                if len(tbl) == len(need):
                    with self.lock:
                        if key == self.mu_key:
                            self.mu = ST.grid(list(rad), list(dire), tbl)
                            self.mu_state = "Stratz"
                    return
            except ST.StratzError as e:
                with self.lock:
                    if key == self.mu_key:
                        self.mu_state = f"Stratz не ответил ({e}); беру OpenDota"
            except Exception as e:                          # noqa: BLE001
                with self.lock:
                    if key == self.mu_key:
                        self.mu_state = f"Stratz: {type(e).__name__}; беру OpenDota"
        try:
            tbl = MU.table(need, client=OpenDotaClient(verbose=False))
        except Exception as e:                              # noqa: BLE001
            with self.lock:
                if key == self.mu_key:
                    self.mu_state = f"не добрано: {type(e).__name__}"
            return
        with self.lock:
            if key != self.mu_key:
                return                                      # состав уже сменился
            if len(tbl) == len(need):
                self.mu = MU.grid(list(rad), list(dire), tbl)
                self.mu_state = "OpenDota (Stratz недоступен)"
            else:
                self.mu_state = (f"добрано {len(tbl)} героев из {len(need)} — "
                                 f"сетка неполная")

    def map_block(self, payload: dict, minute: float = float("nan")) -> dict:
        st_b = M.standing(payload)
        b = [{"x": v["x"], "y": v["y"], "team": v["team"], "type": v["type"],
              "tier": v["tier"], "lane": v["lane"], "dead": False}
             for v in st_b.values()]
        b += [{"x": v["x"], "y": v["y"], "team": v["team"], "type": v["type"],
               "tier": v["tier"], "lane": v["lane"], "dead": True}
              for k, v in self.book.items() if k not in st_b]
        self.dead.update(payload, minute)
        heroes = []
        for t in (payload.get("teams") or []):
            for i, pl in enumerate(t.get("players") or []):
                if pl.get("x") is None or pl.get("y") is None:
                    continue
                hid = int(pl.get("heroid") or 0)
                slug, name = self.heroes.get(hid, ("", str(hid)))
                slot = pl.get("team_slot")
                team = int(pl.get("team", t.get("team_number", 0)))
                ds = self.dead.state(t.get("team_number", team),
                                     slot if slot is not None else i)
                since = ds.get("since")
                heroes.append({
                    "x": float(pl["x"]), "y": float(pl["y"]),
                    "team": team,
                    "id": hid, "slug": slug, "name": name,
                    "img": HERO_IMG.format(slug=slug) if slug else "",
                    "level": pl.get("level"), "nw": pl.get("net_worth"),
                    "k": pl.get("kill_count"), "d": pl.get("death_count"),
                    "a": pl.get("assists_count"),
                    # Мёртв — по замеренному признаку «координаты замерли
                    # с момента, когда вырос счётчик смертей» (dwp.deaths).
                    # `known` False означает «за этого героя мы смерть не
                    # видели»: подключились в середине, и он мог лежать
                    # ещё до нас. Это не «жив», это «не знаем».
                    "dead": bool(ds.get("dead")),
                    "dead_known": bool(ds.get("known")),
                    "dead_for": (None if since is None or minute != minute
                                 else max(0.0, (float(minute) - since) * 60.0)),
                })
        return {"scale": config.MAP_IMAGE_SCALE, "image": config.MAP_IMAGE_PATH.exists(),
                "buildings": b, "heroes": heroes, "lanes": self.lanes,
                "book": len(self.book), "book_full": config.N_BUILDINGS_TOTAL,
                "radiant": config.TEAM_RADIANT,
                "n_dead": self.dead.n_dead,
                "n_deaths_seen": self.dead.n_seen}

    # --- цикл ----------------------------------------------------------

    def poll_once(self) -> None:
        with self.lock:
            gen = self.gen
        if self.from_file is not None:
            raw = self.from_file.read_bytes()
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8-sig"
            payload = json.loads(raw.decode(enc))
        else:
            payload = live.realtime_stats(self.key, self.sid)
        snap = self.snapshot(payload)
        with self.lock:
            # Пока считали, могли переключить матч. Тогда этот снимок — про
            # прошлый матч, и класть его нельзя: он затёр бы «первый опрос
            # ещё не прошёл» чужими числами под новым sid.
            if gen != self.gen:
                return
            self.state = snap
        self.fails = 0

    def run(self) -> None:
        while not self.stop.is_set():
            if self.sid is None and self.from_file is None:
                self.wake.wait(1.0)
                self.wake.clear()
                continue
            try:
                self.poll_once()
            except live.LiveError as e:
                self.fails += 1
                with self.lock:
                    self.state = {**self.state, "ok": False, "error": str(e),
                                  "fails": self.fails}
            except Exception as e:                          # noqa: BLE001
                self.fails += 1
                with self.lock:
                    self.state = {**self.state, "ok": False,
                                  "error": f"{type(e).__name__}: {e} "
                                           f"(это баг dwp, приложите --dump)",
                                  "fails": self.fails}
            self.wake.wait(self.interval)
            self.wake.clear()

    def get(self) -> dict:
        with self.lock:
            return dict(self.state)


def record_guess(poller: "Poller", p_human: float) -> dict:
    """Записать догадку зрителя рядом с числом модели на ту же минуту.

    Зачем это в панели, а не только в консоли. Слепой тест в `dwp.blindtest`
    показывает положение текстом; здесь зритель видит то же, что видит
    комментатор, — карту, сборки, счёт, — и называет своё число ДО того, как
    ему покажут наше. Исход не знает никто, так что подсмотреть нечего.
    Считается это потом: `python -m dwp.blindtest --guesses`.
    """
    s = poller.get()
    if not s.get("ok"):
        return {"ok": False, "error": "матч не считается — нечего угадывать"}
    p_human = float(min(max(p_human, 0.01), 0.99))
    row = {
        "ts": f"{time.time():.0f}",
        "match_id": s.get("match_id") or "",
        "minute": ("" if s.get("minute") is None else f"{s['minute']:.1f}"),
        "p_model": f"{float(s['p']):.6f}",
        "p_human": f"{p_human:.4f}",
        "model": s.get("model") or "",
    }
    GUESS_PATH.parent.mkdir(parents=True, exist_ok=True)
    new = not GUESS_PATH.exists()
    with GUESS_PATH.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(row))
        if new:
            w.writeheader()
        w.writerow(row)
    return {"ok": True, "p_model": float(s["p"]), "p_human": p_human,
            "saved": str(GUESS_PATH)}


class Handler(BaseHTTPRequestHandler):
    server_version = "dwp"

    def __init__(self, poller: Poller, runner: "JOBS.Runner", *a, **kw):
        self.poller = poller
        self.runner = runner
        super().__init__(*a, **kw)

    def log_message(self, *a):
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj) -> None:
        self._send(200, json.dumps(obj, ensure_ascii=False,
                                   default=str).encode("utf-8"),
                   "application/json; charset=utf-8")

    def do_GET(self):                                       # noqa: N802
        raw = self.path.split("?", 1)
        path = raw[0]
        q = {}
        if len(raw) > 1:
            for part in raw[1].split("&"):
                k, _, v = part.partition("=")
                q[k] = v
        if path == "/api/state":
            self._json(self.poller.get())
        elif path == "/api/games":
            if q.get("force"):
                # Кнопка «обновить список» обязана действительно обновлять,
                # иначе она врёт: кэш держит ответ 20 с.
                with self.poller.lock:
                    self.poller.games_at = 0.0
            rows, err = self.poller.game_list()
            self._json({"games": rows, "error": err,
                        "watching": self.poller.sid,
                        # В режиме --from-file матч не выбирают: он один и
                        # уже считается. Без этого флага страница открывалась
                        # на списке игр, которого без ключа Steam нет вовсе,
                        # и панель приходилось искать.
                        "file_mode": self.poller.from_file is not None})
        elif path == "/api/watch":
            sid = q.get("sid") or None
            self.poller.watch(sid)
            self._json({"ok": True, "watching": sid})
        elif path == "/api/guess":
            # Догадка зрителя из слепого режима. Исход в этот момент
            # неизвестен НИКОМУ — в этом и смысл; посчитается потом,
            # `python -m dwp.blindtest --guesses`.
            try:
                p = float(q.get("p", ""))
            except ValueError:
                self._json({"ok": False, "error": "p не число"})
                return
            self._json(record_guess(self.poller, p))
        elif path == "/api/status":
            # Состояние машины для меню: что скачано, что обучено, чего нет.
            self._json({"data": JOBS.data_status(),
                        "tasks": self.runner.catalogue(),
                        "model": self.poller.model_name,
                        "n_models": len(live.members(self.poller.art)),
                        "features": len(self.poller.art["state_features"])})
        elif path == "/api/jobs":
            self._json({"current": self.runner.get(tail=400),
                        "history": self.runner.history()})
        elif path == "/api/job":
            try:
                jid = int(q.get("id", ""))
            except ValueError:
                jid = None
            got = self.runner.get(jid, tail=400)
            self._json(got or {"error": "такой задачи нет"})
        elif path == "/api/accuracy":
            # Считается на месте: чтение артефактов и логов занимает секунды,
            # а кэшировать значит однажды показать устаревшее и не заметить.
            try:
                self._json(ACC.everything(self.poller.log_dir))
            except Exception as e:                          # noqa: BLE001
                self._json({"error": f"{type(e).__name__}: {e}"})
        elif path == "/assets/map.jpg":
            p = config.MAP_IMAGE_PATH
            if p.exists():
                self._send(200, p.read_bytes(), "image/jpeg")
            else:
                self._send(404, b"map image not downloaded",
                           "text/plain; charset=utf-8")
        elif path in ("/", "/index.html"):
            self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")

    def do_POST(self):                                      # noqa: N802
        """Запуск и остановка задач.

        Именно POST, а не GET: GET браузер может дёрнуть сам —
        предзагрузкой ссылки, восстановлением вкладки, повтором из
        истории, — и обучение запустится без ведома человека. Тело
        разбирается как JSON, но команда из него НЕ берётся: приходит
        только ключ из фиксированного списка и числовые параметры
        (см. dwp/jobs.py).
        """
        path = self.path.split("?", 1)[0]
        try:
            n = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(n) or b"{}") if n else {}
        except (ValueError, json.JSONDecodeError):
            self._json({"ok": False, "error": "тело запроса не JSON"})
            return
        if not isinstance(body, dict):
            body = {}
        if path == "/api/jobs/start":
            try:
                job = self.runner.start(str(body.get("key") or ""),
                                        body.get("params") or {})
            except (ValueError, RuntimeError) as e:
                self._json({"ok": False, "error": str(e)})
                return
            self._json({"ok": True, "job": job.snapshot(tail=0)})
        elif path == "/api/jobs/stop":
            self._json({"ok": self.runner.stop()})
        else:
            self._send(404, b"not found", "text/plain; charset=utf-8")


PAGE = r"""<!doctype html>
<html lang="ru"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>dwp — лайв</title>
<style>
/* ПАЛИТРА. Пара Radiant/Dire выбрана не на вкус: прежняя (#3fb950 против
   #f85149) при дейтеранопии даёт OKLab dE = 2.2 при пороге 6 — то есть для
   каждого двадцатого зрителя это ОДИН И ТОТ ЖЕ цвет, а панель на нём
   держала всё. Нынешняя пара разведена ещё и по светлоте: dE = 22.0 при
   дейтеранопии и 37.6 при протанопии, оба хвоста далеко за порогом, а
   зелёный остаётся зелёным (тон 148) и красный красным (тон 28).
   Проверять при смене цветов: scripts/validate_palette.js из навыка
   dataviz либо тот же расчёт (Machado 2009 + OKLab). */
:root{--bg:#070a0f;--card:#0f141b;--raised:#151c26;--line:#1e2733;
      --fg:#e9eef5;--dim:#93a1b1;--faint:#5d6b7c;
      --rad:#5ddc7a;--dire:#d1332a;--warn:#e3b341;--acc:#6cb6ff;--r:12px}
*{box-sizing:border-box}
/* Фон задан и на html, и на body, и на обёртке. Так надёжнее: если body
   окажется короче окна или страница отрисуется частично, белого не будет
   нигде. color-scheme:dark заодно красит полосу прокрутки и поля ввода. */
html{background:#070a0f;color-scheme:dark;min-height:100%}
body{margin:0;min-height:100vh;background:var(--bg);color:var(--fg);
     font:13px/1.5 "Segoe UI",system-ui,sans-serif;
     -webkit-font-smoothing:antialiased}
/* Прозрачный фон — только для OBS (?bare=1), в обычном браузере он даст
   белое полотно, и это ожидаемо: там его подкладывает сам OBS. */
html:has(body.bare),body.bare{background:transparent}
body.bare header,body.bare .side,body.bare .nobare{display:none!important}
body.bare .wrap{padding:8px;max-width:760px}
/* В OBS раскладку не двигают мышью, и колонка там одна: карточки идут в
   полную ширину независимо от того, что человек настроил себе в браузере. */
body.bare #panel>.card{grid-column:span 12!important;zoom:1!important}
body.bare .rail,body.bare .zoomer,body.bare .hd,
body.bare .laybar{display:none!important}
/* Свободная раскладка в OBS не применяется вовсе (см. layMode), но если
   класс всё же остался от прежней вкладки — вот страховка на CSS. */
body.bare #panel.free{display:grid!important;height:auto!important}
body.bare #panel.free>.card{position:relative!important;left:auto!important;
  top:auto!important;width:auto!important;height:auto!important;
  padding:14px 16px!important}
body.bare #panel.free>.card>.cbody{position:static!important;padding:0!important}
header{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:10px;
       padding:9px 18px;border-bottom:1px solid var(--line);
       background:rgba(7,10,15,.92);backdrop-filter:blur(6px)}
header b{letter-spacing:.18em;font-size:11px;color:var(--dim);font-weight:700}
button{background:var(--raised);color:var(--fg);border:1px solid var(--line);
       border-radius:7px;padding:5px 11px;cursor:pointer;font:inherit;
       font-size:12px;transition:border-color .12s,background .12s}
button:hover{border-color:var(--acc);background:#1b2431}
button.on{border-color:var(--acc);color:var(--acc)}
input{background:#0a0f15;color:var(--fg);border:1px solid var(--line);
      border-radius:7px;padding:6px 10px;font:inherit}
input:focus{outline:none;border-color:var(--acc)}
/* РАСКЛАДКА ПАНЕЛИ — два режима, переключатель в полосе «раскладка».

     сетка      двенадцать колонок. Окно меняют местами с другим (тяга за
                левый край), растягивают по ширине (правый край) и
                увеличивают содержимое (правый нижний угол).
     свободно   у окна появляются координаты: шапка переносит его в любое
                место полотна, восемь маркеров по периметру задают ширину
                И высоту, Ctrl+колесо — масштаб содержимого.

   Раскладка обоих режимов лежит в localStorage и переживает перезагрузку. */
.wrap{max-width:1360px;margin:0 auto;padding:16px;display:grid;gap:14px;
      grid-template-columns:repeat(12,minmax(0,1fr));align-items:start}
.one{max-width:1360px;margin:0 auto;padding:16px}
.card{background:var(--card);border:1px solid var(--line);
      border-radius:var(--r);padding:14px 16px}
/* Масштаб через zoom, а НЕ через transform:scale. Разница не косметическая:
   scale рисует увеличенную карточку поверх соседей, не меняя её места в
   сетке, — увеличенные сборки залезли бы на карту. zoom меняет размер самой
   коробки, и сетка раздвигается. */
#panel>.card{position:relative;grid-column:span 12;zoom:var(--z,1)}
/* Полосы-ручки по краям. Ширина делится на масштаб, иначе у уменьшенной
   карточки ручка становится уже пальца. */
#panel>.card>.rail{position:absolute;top:0;bottom:0;z-index:3;
  width:calc(10px / var(--z,1));opacity:0;transition:opacity .12s}
#panel>.card:hover>.rail,body.laying #panel>.card>.rail{opacity:.9}
#panel>.card>.rail.mv{left:0;cursor:grab;border-radius:var(--r) 0 0 var(--r);
  background:linear-gradient(90deg,var(--acc),transparent)}
#panel>.card>.rail.wd{right:0;cursor:col-resize;
  border-radius:0 var(--r) var(--r) 0;
  background:linear-gradient(270deg,var(--acc),transparent)}
#panel>.card>.zoomer{position:absolute;right:1px;bottom:1px;z-index:3;
  width:calc(17px / var(--z,1));height:calc(17px / var(--z,1));
  cursor:nwse-resize;opacity:0;transition:opacity .12s;
  border-right:2px solid var(--acc);border-bottom:2px solid var(--acc);
  border-radius:0 0 var(--r) 0}
#panel>.card:hover>.zoomer,body.laying #panel>.card>.zoomer{opacity:.8}
#panel>.card.dragging{opacity:.4}
#panel>.card.dropzone{outline:2px dashed var(--acc);outline-offset:-3px}
/* Маркеры свободного режима не существуют в режиме сетки, а ручки сетки —
   в свободном. Иначе на правом крае окна их оказывалось бы две. */
#panel:not(.free)>.card>.hd{display:none}
#panel.free>.card>.rail,#panel.free>.card>.zoomer{display:none}

/* --- СВОБОДНЫЙ РЕЖИМ ------------------------------------------------------
   Полотно во всю ширину окна: «поставить куда угодно» и потолок в 1360 px
   плохо сочетаются. Высоту полотну проставляет JS — у абсолютных окон её
   взять неоткуда, а без неё страница не прокручивалась бы до нижнего. */
/* isolation:isolate — не украшение. Без своего контекста наложения окна
   с z-index конкурируют с липкой шапкой (у неё 5) и при прокрутке лезут на
   неё. С ним номера окон живут внутри полотна, а само полотно стоит под
   шапкой. */
#panel.free{display:block;position:relative;max-width:none;padding-bottom:0;
  z-index:0;isolation:isolate}
#panel.free>.card{position:absolute;grid-column:auto;padding:0;
  overflow:hidden;zoom:1}
/* Три уровня, и каждый нужен: коробка держит маркеры и не прокручивается,
   .cbody прокручивает содержимое, .czoom его масштабирует. Свести zoom и
   overflow в один элемент нельзя — zoom умножает и inset, и окно поехало бы
   вслед за масштабом. */
#panel.free>.card>.cbody{position:absolute;inset:0;overflow:auto;
  padding:14px 16px;overscroll-behavior:contain}
#panel.free>.card>.cbody>.czoom{zoom:var(--z,1)}
#panel.free>.card:hover{border-color:#2b3b4e}
#panel.free>.card.front{box-shadow:0 12px 36px rgba(0,0,0,.55)}
/* Маркеры лежат ВНУТРИ окна, а не выступают наружу: у коробки
   overflow:hidden, и выступающий край просто срезало бы. */
#panel.free>.card>.hd{position:absolute;z-index:4;background:transparent}
#panel.free>.card>.hd:hover{background:rgba(108,182,255,.30)}
#panel.free>.card>.hd.mv{left:14px;right:14px;top:8px;height:16px;cursor:grab;
  border-radius:3px;opacity:0;transition:opacity .12s;
  background:linear-gradient(180deg,var(--acc),transparent)}
#panel.free>.card:hover>.hd.mv,
body.laying #panel.free>.card>.hd.mv{opacity:.5}
#panel.free>.card>.hd.n{left:14px;right:14px;top:0;height:8px;cursor:ns-resize}
#panel.free>.card>.hd.s{left:14px;right:14px;bottom:0;height:8px;cursor:ns-resize}
#panel.free>.card>.hd.w{top:14px;bottom:14px;left:0;width:8px;cursor:ew-resize}
#panel.free>.card>.hd.e{top:14px;bottom:14px;right:0;width:8px;cursor:ew-resize}
#panel.free>.card>.hd.nw{left:0;top:0;cursor:nwse-resize}
#panel.free>.card>.hd.ne{right:0;top:0;cursor:nesw-resize}
#panel.free>.card>.hd.sw{left:0;bottom:0;cursor:nesw-resize}
#panel.free>.card>.hd.se{right:0;bottom:0;cursor:nwse-resize}
#panel.free>.card>.hd.nw,#panel.free>.card>.hd.ne,
#panel.free>.card>.hd.sw,#panel.free>.card>.hd.se{width:15px;height:15px}
/* Уголок видно на наведении — иначе про углы никто не догадается. */
#panel.free>.card:hover>.hd.se,body.laying #panel.free>.card>.hd.se{
  border-right:2px solid var(--acc);border-bottom:2px solid var(--acc);
  border-radius:0 0 var(--r) 0}
/* Направляющая прилипания: одна вертикаль и одна горизонталь, рисуются
   только пока тянут. */
.snapline{position:absolute;z-index:9;background:var(--acc);opacity:.55;
  pointer-events:none;display:none}
.laybar{max-width:1360px;margin:0 auto;padding:2px 16px 0;display:none;
  align-items:center;gap:10px;color:var(--faint);font-size:12px;flex-wrap:wrap}
body.laying .laybar{display:flex}
body.free .laybar{max-width:none}
.laybar .hint{color:var(--faint)}
@media (max-width:1100px){#panel:not(.free)>.card{grid-column:span 12!important}}
/* Узкий экран: колонка одна, и абсолютные координаты, снятые с монитора,
   сделали бы панель нечитаемой. Порог тот же, что в layMode(). */
@media (max-width:820px){
  #panel.free{display:grid;height:auto!important;
    grid-template-columns:repeat(12,minmax(0,1fr))}
  #panel.free>.card{position:relative!important;left:auto!important;
    top:auto!important;width:auto!important;height:auto!important;
    grid-column:span 12!important;padding:14px 16px;overflow:visible}
  #panel.free>.card>.cbody{position:static;padding:0}
  #panel.free>.card>.hd{display:none}
}
h2{margin:0 0 11px;font-size:10px;font-weight:700;letter-spacing:.13em;
   text-transform:uppercase;color:var(--faint)}
h2 .rt{float:right;text-transform:none;letter-spacing:0;font-weight:400;
   color:var(--faint)}
.tname{font-weight:600;font-size:18px;letter-spacing:-.01em}
.clock{font-variant-numeric:tabular-nums;font-size:19px;font-weight:600}
.gold{color:var(--warn);font-variant-numeric:tabular-nums;text-align:center;
      font-size:12px}
.sub{color:var(--dim);font-size:12px;display:flex;justify-content:space-between;
     gap:8px}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{padding:5px 6px;border-bottom:1px solid #141c26;white-space:nowrap;
      text-align:left}
th{color:var(--faint);font-weight:500;font-size:11px;letter-spacing:.04em}
tr:last-child td{border-bottom:none}
td.l{white-space:normal;color:var(--fg)}
td.v{text-align:right;color:var(--dim)}
tr.row{cursor:pointer}
tr.row:hover td{background:var(--raised)}
tr.on td{background:#13202a}
.mini{display:flex;gap:3px}
.mini img,.mini span{width:28px;height:16px;object-fit:cover;border-radius:3px;
  background:var(--raised);font-size:8px;display:inline-flex;align-items:center;
  justify-content:center;color:var(--dim)}
/* --- сборки --- */
.blds{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media (max-width:860px){.blds{grid-template-columns:1fr}}
.bteam{min-width:0}
.bhead{display:flex;align-items:baseline;gap:12px;flex-wrap:wrap;
       padding:0 0 7px;margin-bottom:4px;border-bottom:1px solid var(--line)}
.bhead .bt{font-size:15px;font-weight:700;flex:1;min-width:0;
           overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.bhead .bsum{font-size:14px;font-weight:600;color:var(--dim);
             font-variant-numeric:tabular-nums;text-align:right}
.bhead .bsum.hot{color:var(--warn)}
.bhead .bsum .u{display:block;font-size:9px;font-weight:500;color:var(--faint);
                letter-spacing:.04em;text-transform:uppercase}
.brow{display:flex;align-items:center;gap:7px;padding:5px 0 5px 7px;
      border-left:2px solid transparent;border-bottom:1px solid #141c26}
.brow:last-child{border-bottom:none}
.brow.core{border-left-color:var(--warn);background:#121a24}
.brow .hi{width:34px;height:20px;object-fit:cover;border-radius:3px;
  background:var(--raised);flex:0 0 auto}
.brow .lvl{flex:0 0 auto;width:19px;height:19px;border-radius:50%;
  background:var(--raised);border:1px solid var(--line);color:var(--dim);
  font-size:10px;font-weight:700;display:flex;align-items:center;
  justify-content:center;font-variant-numeric:tabular-nums}
/* Имя и полоса нетворса — одна колонка: полоса под именем читается как
   его продолжение, а не как отдельный столбец. */
.brow .nmw{flex:1;min-width:0}
.brow .nm{display:block;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:12px}
.brow .nwbar{display:block;height:3px;border-radius:2px;background:#141c26;
  margin-top:3px;overflow:hidden}
.brow .nwbar i{display:block;height:100%;border-radius:2px;opacity:.85}
.slots{display:flex;gap:2px;flex:0 0 auto}
.slots i{width:23px;height:17px;border-radius:3px;background:#0b1017;
  border:1px solid var(--line);background-size:cover;background-position:center;
  display:block}
.slots i.big{border-color:var(--warn)}
.slots i.cons{opacity:.55}
.slots i.empty{border-style:dashed;opacity:.5}
.bnum{width:52px;text-align:right;font-variant-numeric:tabular-nums;
      font-size:12px;color:var(--dim)}
.bnum.nwn{width:48px;font-weight:600;color:var(--fg)}
.bnum.gld{width:44px;color:var(--faint)}
.bnum.gld.hot{color:var(--warn);font-weight:600}
/* Табло: фраги крупно, процента модели рядом нет — он бы спорил с
   вердиктом, который как раз не меняется. */
.kbig{font-weight:700;font-size:38px;line-height:1;letter-spacing:-.02em;
      font-variant-numeric:tabular-nums;margin-top:2px}
/* --- вердикт ---
   Карточка намеренно крупная и стоит первой: это единственное место, где
   панель говорит одним словом, кто победит. Всё остальное на экране —
   величины, из которых это слово получилось. */
.vd{display:flex;align-items:baseline;gap:16px;flex-wrap:wrap}
.vd .kw{font-size:12px;font-weight:700;letter-spacing:.16em;
        text-transform:uppercase;color:var(--faint)}
.vd .who{font-size:40px;font-weight:700;line-height:1.05;letter-spacing:-.025em;
         overflow:hidden;text-overflow:ellipsis;max-width:100%}
.vd .wait{font-size:22px;font-weight:600;color:var(--dim)}
.vdnote{color:var(--faint);font-size:11px;margin-top:11px;line-height:1.5}
/* Ожидание перевеса: полоса до порога. Без числа — число здесь шевелилось
   бы рядом с вердиктом, который не шевелится. */
.vprog{height:4px;border-radius:2px;background:#141c26;margin-top:12px;
       overflow:hidden}
.vprog i{display:block;height:100%;border-radius:2px;background:var(--dim);
         transition:width .6s linear}
.tip{position:fixed;pointer-events:none;background:#0a0f16;color:var(--fg);
  border:1px solid var(--line);border-radius:7px;padding:6px 9px;font-size:12px;
  box-shadow:0 6px 22px rgba(0,0,0,.55);z-index:20;display:none;
  font-variant-numeric:tabular-nums}
/* Слепой режим: рамка вокруг ввода догадки. */
.chance{background:var(--raised);border:1px solid var(--line);
        border-radius:9px;padding:11px 13px}
/* Карта квадратная, и растянуть её на всю ширину экрана можно — но выше
   окна она быть не должна, иначе карточка перестаёт помещаться целиком. */
#mapbox{position:relative;width:100%;max-width:min(100%,calc(100vh - 170px));
        margin:0 auto;aspect-ratio:1/1;border:1px solid var(--line);
        border-radius:9px;overflow:hidden;background:#05080d}
#mapsvg,#mapimg{position:absolute;inset:0;width:100%;height:100%}
#mapimg{object-fit:fill}
#blds,#pins{position:absolute;inset:0;pointer-events:none}
#blds .b,#pins .h{pointer-events:auto}
#blds .b{position:absolute;border-radius:2px;
  box-shadow:0 0 0 1px rgba(0,0,0,.6)}
/* Кольцо цветом поверхности вокруг иконки: она читается и когда налезает
   на здание, и когда две иконки перекрываются в замесе. */
#pins .h{position:absolute;transform:translate(-50%,-50%);width:32px;height:32px;
        border-radius:50%;border:2px solid;overflow:hidden;background:#0b0e13;
        display:flex;align-items:center;justify-content:center;font-size:9px;
        box-shadow:0 0 0 2px #05080d;
        /* Переезд между опросами. Чуть короче интервала опроса (2 с), чтобы
           иконка успевала доехать и на мгновение замирала, а не дёргалась в
           середине пути новым значением. */
        transition:left 1.7s linear,top 1.7s linear;
        /* Слой по умолчанию — над мёртвыми, см. ниже. */
        z-index:2}
#pins .h img{width:100%;height:100%;object-fit:cover}
/* УБИТЫЙ ГЕРОЙ. Чёрно-белый и слоем НИЖЕ живых.
   Порядок слоёв тут важнее цвета: в замесе десять иконок налезают друг на
   друга, и если сверху оказывается труп, живого героя под ним не видно
   вовсе — а решают исход замеса именно живые. Серость же нужна затем,
   чтобы не считать мёртвых глазами как силу на точке.
   Признак «убит» — замеренный, а не выдуманный: координаты замирают в
   точке смерти и снова идут при возрождении (dwp/deaths.py, там числа).
   Цвет тут второй канал: то же самое написано словом в подсказке. */
#pins .h.dead{z-index:1;filter:grayscale(1) brightness(.6);opacity:.8;
        border-style:dashed;box-shadow:none}
.draft{display:grid;gap:6px}
.dr{display:flex;align-items:center;gap:8px}
.dr img{width:38px;height:23px;object-fit:cover;border-radius:3px;
  background:var(--raised)}
.dr .n{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.dr .g{width:104px;height:8px;background:#141c26;border-radius:4px;position:relative}
.dr .g i{position:absolute;top:0;height:100%;border-radius:4px}
.dr .q{width:54px;text-align:right;color:var(--dim);font-variant-numeric:tabular-nums}
details summary::-webkit-details-marker{color:var(--faint)}
details[open] summary{margin-bottom:2px}
/* --- лента событий --- */
.ev{display:flex;align-items:center;gap:7px;padding:5px 0;
  border-bottom:1px solid #141c26;font-size:12px}
.ev:last-child{border-bottom:none}
.ev .m{color:var(--faint);font-variant-numeric:tabular-nums;flex:0 0 40px}
.ev img,.ev .ph{width:26px;height:26px;border-radius:50%;object-fit:cover;
  border:2px solid;background:#0b0e13;flex:0 0 auto;display:inline-flex;
  align-items:center;justify-content:center;font-size:8px;color:var(--dim)}
.ev .dead{filter:grayscale(1);opacity:.55}
.ev .x{color:var(--faint);flex:0 0 auto}
.ev .tx{flex:1;min-width:0;color:var(--dim);overflow:hidden;
  text-overflow:ellipsis;white-space:nowrap}
.ev.guessy .tx{color:var(--warn)}
/* --- меню, задачи, точность --- */
.kpi{display:grid;gap:12px}
.kpi .n{font-size:22px;font-weight:700;font-variant-numeric:tabular-nums}
.kpi .t{color:var(--faint);font-size:11px;margin-top:3px;line-height:1.4}
.grid2{display:grid;grid-template-columns:repeat(auto-fit,minmax(310px,1fr));
  gap:14px}
.job{background:var(--card);border:1px solid var(--line);border-radius:var(--r);
  padding:14px 16px;display:flex;flex-direction:column;gap:8px}
.job h3{margin:0;font-size:14px;font-weight:600}
.job p{margin:0;color:var(--dim);font-size:12px;line-height:1.5}
.job .go{margin-top:auto;display:flex;gap:8px;align-items:center}
pre.log{background:#070b10;border:1px solid var(--line);border-radius:8px;
  padding:10px 12px;margin:0;max-height:340px;overflow:auto;font-size:11px;
  line-height:1.45;white-space:pre-wrap;word-break:break-word;color:var(--dim)}
.pill{display:inline-block;padding:1px 7px;border-radius:20px;font-size:11px;
  border:1px solid var(--line);color:var(--dim)}
.pill.on{border-color:var(--rad);color:var(--rad)}
.pill.off{border-color:#4b3a1c;color:var(--warn)}
.warn{color:var(--warn);font-size:12px;margin-top:8px;line-height:1.5}
/* Тёмно-красный хорош как заливка, но для мелкого текста его контраста к
   поверхности (3.7:1) мало. Для текста — светлая ступень того же тона. */
.err{color:#ff9a90}
.ok{color:var(--rad)}
</style></head><body>
<header>
  <b>DWP</b>
  <button class="nav" data-view="list">матчи</button>
  <button class="nav" data-view="lab">данные и обучение</button>
  <button class="nav" data-view="acc">точность</button>
  <button id="back" style="display:none">← к списку матчей</button>
  <button id="blind" title="скрыть число модели и назвать своё">слепой режим</button>
  <button id="lay" title="как двигать окна">раскладка</button>
  <span class="sub" id="hdr" style="margin-left:auto"></span>
</header>
<div class="laybar" id="laybar">
  <button id="laygrid" title="окна выстроены в двенадцать колонок">сетка</button>
  <button id="layfree" title="у окон свои координаты: ставьте куда угодно"
    >свободно</button>
  <span class="hint" id="layhint"></span>
  <button id="laytile" title="разложить окна заново, по сетке">плиткой</button>
  <button id="layreset">вернуть как было</button>
</div>
<div id="list" class="one"></div>
<div id="lab" class="one" style="display:none"></div>
<div id="acc" class="one" style="display:none"></div>
<!-- Порядок и размеры карточек задаёт JS из localStorage; здесь только
     разметка и порядок по умолчанию. Ручки (.rail/.zoomer) дописываются
     скриптом: писать их шесть раз руками значит шесть раз ошибиться. -->
<div id="panel" class="wrap" style="display:none">
  <!-- Класса nobare тут намеренно нет: в OBS-режиме вердикт как раз нужен
       больше всего остального. Прячется он только в слепом режиме, и
       прячется явно, внутри drawVerdict. -->
  <div class="card" id="vcard" style="display:none">
    <h2>вердикт<span class="rt" id="vtop"></span></h2>
    <div id="verdict"></div></div>
  <div class="card" id="mcard"><div id="main">загрузка…</div></div>
  <div class="card" id="bcard" style="display:none">
    <h2>сборки<span class="rt" id="btop"></span></h2>
    <div class="blds" id="builds"></div></div>
  <div class="card side" id="pcard"><h2>карта</h2>
    <div id="mapbox">
      <img id="mapimg" alt="карта" onerror="this.style.display='none';
           document.getElementById('mapsvg').style.display=''">
      <svg id="mapsvg" viewBox="0 0 100 100" preserveAspectRatio="none"></svg>
      <div id="blds"></div><div id="pins"></div></div>
    <div class="sub" id="mapnote" style="margin-top:8px;color:var(--warn)"></div></div>
  <div class="card side nobare" id="fcard"><h2>лента событий<span class="rt"
       id="frtop"></span></h2><div id="feed"></div></div>
  <div class="card side" id="dcard"><h2>драфт</h2>
    <div class="draft" id="draft"></div></div>
</div>
<div class="tip" id="tip"></div>
<script>
// Цвета сторон разведены и по тону, и по светлоте — иначе при дейтеранопии
// они сливаются (см. комментарий у палитры в CSS). Всё, что окрашено, ещё и
// подписано словом: цвет тут второй канал, а не единственный.
const RAD="#5ddc7a", DIRE="#d1332a", ACC="#6cb6ff", WARN="#e3b341";
const Q=new URLSearchParams(location.search);
if(Q.has("bare"))document.body.classList.add("bare");
const num=v=>v===null||v===undefined?"—":(Math.abs(v)>=100?
  Math.round(v).toLocaleString("ru-RU"):
  (Number.isInteger(+v)?String(+v):(+v).toFixed(2)));
const pct=v=>(v*100).toFixed(1)+"%";
const TIP=document.getElementById("tip");
function tipShow(html,x,y){TIP.innerHTML=html;TIP.style.display="block";
  const r=TIP.getBoundingClientRect();
  TIP.style.left=Math.min(x+14,innerWidth-r.width-8)+"px";
  TIP.style.top=Math.max(8,y-r.height-12)+"px";}
function tipHide(){TIP.style.display="none";}
const esc=s=>String(s??"").replace(/[&<>"]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;"}[c]));
const icon=o=>o.img
  ? `<img src="${esc(o.img)}" alt="${esc(o.name)}" title="${esc(o.name)}"
      onerror="this.replaceWith(Object.assign(document.createElement('span'),
      {textContent:${JSON.stringify((o.name||"?").slice(0,4))}}))">`
  : `<span>${esc((o.name||"?").slice(0,4))}</span>`;

let view="list", watching=null;
document.getElementById("back").onclick=()=>show("list");
const VIEWS=["list","lab","acc","panel"];
function show(v){
  view=v;
  VIEWS.forEach(k=>{const el=document.getElementById(k);
    if(el)el.style.display = k===v?"":"none";});
  document.getElementById("back").style.display = v==="panel"?"":"none";
  document.getElementById("blind").style.display = v==="panel"?"":"none";
  document.getElementById("lay").style.display = v==="panel"?"":"none";
  // Подсказка про раскладку — только над самой раскладкой.
  if(v!=="panel"){
    document.body.classList.remove("laying");
    document.getElementById("lay").classList.remove("on");
  }
  document.querySelectorAll("header .nav").forEach(b=>
    b.classList.toggle("on", b.dataset.view===v ||
      (v==="panel"&&b.dataset.view==="list")));
  if(v==="lab"){loadLab();}
  if(v==="acc"){loadAcc();}
  if(v==="list"){loadGames();}
  // Свободная раскладка меряет полотно, а у скрытого блока ширина нулевая.
  // Поэтому применяется она здесь, когда панель уже на экране, а не только
  // при загрузке страницы.
  if(v==="panel"){layApply();}
}
document.querySelectorAll("header .nav").forEach(b=>
  b.onclick=()=>show(b.dataset.view));

// --- РАСКЛАДКА ПАНЕЛИ -------------------------------------------------------
// Два режима, переключатель — в полосе «раскладка».
//
//   сетка      двенадцать колонок. Левый край окна тянут, чтобы поменять его
//              местами с другим, правый — чтобы задать ширину, правый нижний
//              угол — масштаб содержимого.
//   свободно   у окна появляются координаты: шапка переносит его в любое
//              место полотна, восемь маркеров по периметру задают ширину И
//              высоту, Ctrl+колесо — масштаб. Края прилипают к соседям и к
//              полотну, Alt на время тяги прилипание выключает.
//
// Свободный режим НЕ включается в OBS (?bare=1) и на узком экране: там
// колонка одна, и координаты, снятые с широкого монитора, сделали бы панель
// нечитаемой. Проверка стоит в layMode(), то есть в этих случаях и мышь
// ничего не делает, а не только CSS перекрывает. Сохранённое при этом не
// портится: вернулись на широкий экран — раскладка на месте.
//
// Ключ хранилища версионный НЕ для красоты: набор окон со временем меняется,
// и сохранённая раскладка не должна ни оживлять снятое окно, ни прятать
// новое. Поэтому сохранённое всегда чистится о текущий список CARDS, а не
// принимается на веру.
const CARDS=["vcard","mcard","bcard","pcard","fcard","dcard"];
const LAY_DEF={vcard:[12,1],mcard:[12,1],bcard:[8,1],
               pcard:[4,1],fcard:[6,1],dcard:[6,1]};
const LAY_KEY="dwp.layout.v2", LAY_KEY_V1="dwp.layout.v1";
const F_MINW=220, F_MINH=90, F_SNAP=7, F_GAP=14, F_NARROW=820;
const BARE=document.body.classList.contains("bare");
let LAY=layLoad(), layTop=1, layWas=null;

function layLoad(){
  const def={mode:"grid",order:CARDS.slice(),span:{},z:{},free:{},cw:0};
  for(const id of CARDS){def.span[id]=LAY_DEF[id][0];def.z[id]=LAY_DEF[id][1];}
  let saved=null;
  try{
    // v1 читается как v2 без свободной части: прежняя настроенная сетка
    // не должна пропасть от того, что появился второй режим.
    saved=JSON.parse(localStorage.getItem(LAY_KEY)
                     ||localStorage.getItem(LAY_KEY_V1)||"null");
  }catch(e){}
  if(!saved||typeof saved!=="object")return def;
  const order=(Array.isArray(saved.order)?saved.order:[])
    .filter((id,i,a)=>CARDS.includes(id)&&a.indexOf(id)===i);
  for(const id of CARDS)if(!order.includes(id))order.push(id);
  const span={},z={},free={};
  for(const id of CARDS){
    const s=Math.round(+(saved.span||{})[id]);
    span[id]=(s>=3&&s<=12)?s:LAY_DEF[id][0];
    const q=+(saved.z||{})[id];
    z[id]=(q>=0.6&&q<=2)?q:LAY_DEF[id][1];
    // Геометрия принимается только целиком и только числами. Окно с NaN в
    // ширине рисуется нулевым, и починить его мышью человек уже не сможет:
    // хватать нечего. Лучше выложить такое заново.
    const f=(saved.free||{})[id];
    if(f&&[f.x,f.y,f.w,f.h].every(v=>typeof v==="number"&&isFinite(v)))
      free[id]={x:Math.max(0,f.x), y:Math.max(0,f.y),
                w:Math.max(F_MINW,f.w), h:Math.max(F_MINH,f.h),
                zi:Math.max(1,+f.zi||1)};
  }
  const cw=+saved.cw;
  return {mode:saved.mode==="free"?"free":"grid", order, span, z, free,
          cw:(isFinite(cw)&&cw>0)?cw:0};
}
function laySave(){
  try{localStorage.setItem(LAY_KEY,JSON.stringify(LAY));}catch(e){}
}
function layMode(){
  return (!BARE&&LAY.mode==="free"&&innerWidth>F_NARROW)?"free":"grid";
}
// Полотно: где у него левый верхний угол на экране и какова его внутренняя
// ширина. Считается по факту, а не по константе: в OBS у обёртки другие
// отступы, а в свободном режиме снят и потолок в 1360 px.
function layCanvas(){
  const p=document.getElementById("panel"), cs=getComputedStyle(p);
  const pl=parseFloat(cs.paddingLeft)||0, pr=parseFloat(cs.paddingRight)||0;
  return {p, w:Math.max(F_MINW, p.clientWidth-pl-pr)};
}
// Ширина колонки вместе с зазором — для тяги за правый край в режиме сетки.
function layStep(){
  const c=layCanvas();
  const gap=parseFloat(getComputedStyle(c.p).columnGap)||F_GAP;
  return {u:(c.w+gap)/12, gap};
}

// --- применение -------------------------------------------------------------

function layApply(){
  const p=document.getElementById("panel"), free=layMode()==="free";
  layBar();
  layWas=free?"free":"grid";
  if(free){
    // Панель ещё скрыта (её показывает show()) — мерить нечего, а
    // померив ноль, мы бы сплющили все окна до минимума и сохранили это.
    if(!p.clientWidth)return;
    const miss=CARDS.filter(id=>!LAY.free[id]);
    if(miss.length){
      // Координаты снимаются с той сетки, которая сейчас на экране, — то
      // есть ДО смены режима. Иначе первое же включение «свободно»
      // разбросало бы окна, и человек решил бы, что что-то сломал.
      const now=layCanvas().w;
      const base=(miss.length===CARDS.length||!LAY.cw)?now:LAY.cw;
      const t=layTile(base);
      for(const id of miss)LAY.free[id]=t[id];
      LAY.cw=base;
    }
  }
  p.classList.toggle("free",free);
  document.body.classList.toggle("free",free);
  if(free)layApplyFree(p); else layApplyGrid(p);
}
function layApplyGrid(p){
  p.style.height="";
  for(const id of LAY.order){
    const el=document.getElementById(id);
    if(!el)continue;
    p.appendChild(el);                       // переносит в конец = задаёт порядок
    el.classList.remove("front");
    el.style.left=el.style.top=el.style.width=el.style.height="";
    el.style.zIndex="";
    el.style.gridColumn="span "+LAY.span[id];
    el.style.setProperty("--z",LAY.z[id]);
  }
  laySnapLine("v",null); laySnapLine("h",null);
}
function layApplyFree(p){
  const c=layCanvas();
  // Сменилась ширина окна или монитор — раскладку тянем пропорционально ПО
  // ГОРИЗОНТАЛИ. По вертикали не тянем: там прокрутка, а не дефицит места,
  // и растянутая по высоте карта перестала бы быть квадратной.
  if(LAY.cw>0&&Math.abs(c.w-LAY.cw)>1){
    const k=c.w/LAY.cw;
    for(const id of CARDS){
      const f=LAY.free[id];
      if(!f)continue;
      f.x=Math.round(f.x*k); f.w=Math.max(F_MINW,Math.round(f.w*k));
    }
  }
  LAY.cw=c.w;
  for(const id of LAY.order){
    const el=document.getElementById(id), f=LAY.free[id];
    if(!el||!f)continue;
    f.w=Math.min(f.w,c.w);
    f.x=Math.max(0,Math.min(f.x,c.w-f.w));
    f.y=Math.max(0,f.y);
    layPut(el,f);
    layTop=Math.max(layTop,f.zi||1);
  }
  layFit();
}
function layPut(el,f){
  el.style.gridColumn="";
  el.style.left=f.x+"px";  el.style.top=f.y+"px";
  el.style.width=f.w+"px"; el.style.height=f.h+"px";
  el.style.zIndex=f.zi||1;
  el.style.setProperty("--z",LAY.z[el.id]);
}
// Высота полотна. У абсолютных окон её взять неоткуда, а без неё страница
// не прокручивалась бы до нижнего окна. Считается по СОХРАНЁННЫМ числам, а
// не по замеру: замер скрытого окна дал бы ноль.
function layFit(){
  const p=document.getElementById("panel");
  if(layMode()!=="free"){p.style.height="";return;}
  let bottom=0;
  for(const id of LAY.order){
    const el=document.getElementById(id), f=LAY.free[id];
    if(!el||!f||el.style.display==="none")continue;
    bottom=Math.max(bottom,f.y+f.h);
  }
  p.style.height=(bottom+24)+"px";
}
// Плитка по той же сетке из двенадцати колонок: окна ложатся слева направо,
// перенос — когда следующее не влезает в остаток строки. Высота берётся
// замером, если окно на экране, иначе ставится по умолчанию: невидимое окно
// (вердикт до старта, сборки до первых предметов) не должно лечь полоской
// нулевой высоты, из-под которой его потом не вытащить.
function layTile(cw){
  const u=(cw+F_GAP)/12;
  let col=0, y=0, rowH=0, zi=1;
  const out={};
  for(const id of LAY.order){
    const n=LAY.span[id]||12;
    if(col&&col+n>12){y+=rowH+F_GAP; col=0; rowH=0;}
    const el=document.getElementById(id);
    const r=el?el.getBoundingClientRect():null;
    const h=Math.max(F_MINH,(r&&r.height>20)?Math.round(r.height):260);
    out[id]={x:Math.round(col*u), y:Math.round(y),
             w:Math.max(F_MINW,Math.round(n*u-F_GAP)), h, zi:zi++};
    col+=n; rowH=Math.max(rowH,h);
  }
  return out;
}

// --- прилипание -------------------------------------------------------------

function laySnapEdges(self,c){
  // Направляющие: края полотна, его середина и края соседних окон.
  const v=[0,Math.round(c.w/2),c.w], h=[0];
  for(const id of CARDS){
    const el=document.getElementById(id), f=LAY.free[id];
    if(!el||!f||el===self||el.style.display==="none")continue;
    v.push(f.x,f.x+f.w); h.push(f.y,f.y+f.h);
  }
  return {v,h};
}
function laySnapOne(v,cand){
  let best=null, d=F_SNAP;
  for(const c of cand){const q=Math.abs(v-c); if(q<d){d=q;best=c;}}
  return best;                                // null — не прилипло
}
// При переносе пробуем сперва ближний край, потом дальний, и выигрывает
// первый нашедший направляющую: если учитывать оба сразу, окно дёргается
// между ними, когда его ширина близка к расстоянию между направляющими.
function laySnapMove(a,size,cand){
  const s1=laySnapOne(a,cand);
  if(s1!==null)return {v:s1, line:s1};
  const s2=laySnapOne(a+size,cand);
  if(s2!==null)return {v:s2-size, line:s2};
  return {v:a, line:null};
}
let SNAPV=null, SNAPH=null;
function laySnapLine(axis,pos){
  const p=document.getElementById("panel");
  let el=axis==="v"?SNAPV:SNAPH;
  if(!el){
    el=document.createElement("i"); el.className="snapline";
    p.appendChild(el);
    if(axis==="v")SNAPV=el; else SNAPH=el;
  }
  if(pos===null){el.style.display="none";return;}
  el.style.display="block";
  if(axis==="v"){el.style.left=pos+"px"; el.style.top="0";
                 el.style.width="1px";   el.style.height="100%";}
  else          {el.style.top=pos+"px";  el.style.left="0";
                 el.style.height="1px";  el.style.width="100%";}
}

// --- мышь -------------------------------------------------------------------

let layDrag=null;
function layRaise(el){
  if(!LAY.free[el.id])return false;
  for(const id of CARDS){
    const o=document.getElementById(id);
    if(o)o.classList.toggle("front",o===el);
  }
  const ids=CARDS.filter(id=>LAY.free[id])
    .sort((a,b)=>(LAY.free[a].zi||1)-(LAY.free[b].zi||1));
  if(ids[ids.length-1]===el.id)return false;      // и так наверху
  // Номера пересчитываются с единицы, а не растут вверх при каждом щелчке:
  // иначе за сеанс они уползают в сотни, и в хранилище копится мусор,
  // который при следующем запуске нечем осадить.
  ids.splice(ids.indexOf(el.id),1); ids.push(el.id);
  ids.forEach((id,i)=>{
    LAY.free[id].zi=i+1;
    const o=document.getElementById(id);
    if(o)o.style.zIndex=i+1;
  });
  layTop=ids.length;
  return true;
}
function layGrab(e,el,kind,only){
  // Маркеры чужого режима спрятаны через CSS, но проверка нужна и здесь:
  // на узком экране класс .free снят, а обработчики висят те же.
  if(e.button!==0||layMode()!==only)return;
  e.preventDefault(); e.stopPropagation();
  const free=only==="free";
  if(free)layRaise(el);
  layDrag={el, kind, free, x:e.clientX, y:e.clientY,
           z0:LAY.z[el.id]||1, box:free?Object.assign({},LAY.free[el.id]):null,
           over:null};
  document.body.style.userSelect="none";
  if(kind==="move"&&!free){el.classList.add("dragging");el.style.pointerEvents="none";}
  addEventListener("pointermove",layMove);
  // pointercancel — не перестраховка: кнопку могут отпустить за пределами
  // окна, и тогда pointerup не придёт вовсе. В сетке это кончалось лишним
  // кадром, в свободном режиме окно осталось бы висеть на курсоре.
  addEventListener("pointerup",layDrop,{once:true});
  addEventListener("pointercancel",layDrop,{once:true});
}
function layMove(e){
  const d=layDrag;
  if(!d)return;
  if(d.free){layMoveFree(e,d);return;}
  if(d.kind==="move"){
    // Окно под курсором ищем через elementFromPoint, а перетаскиваемому на
    // время сняли pointer-events — иначе оно само оказывалось бы «под».
    const t=document.elementFromPoint(e.clientX,e.clientY);
    const over=t?t.closest("#panel>.card"):null;
    const next=(over&&over!==d.el)?over:null;
    if(next!==d.over){
      if(d.over)d.over.classList.remove("dropzone");
      d.over=next;
      if(next)next.classList.add("dropzone");
    }
    return;
  }
  if(d.kind==="width"){
    const {u,gap}=layStep(), r=d.el.getBoundingClientRect();
    // Ширина окна на n колонках = n*u - gap, отсюда и обратный счёт.
    let n=Math.round((e.clientX-r.left+gap)/u);
    n=Math.max(3,Math.min(12,n));
    if(n!==LAY.span[d.el.id]){
      LAY.span[d.el.id]=n;
      d.el.style.gridColumn="span "+n;
    }
    tipShow(`ширина ${LAY.span[d.el.id]} из 12`,e.clientX,e.clientY);
    return;
  }
  const shift=(e.clientX-d.x)+(e.clientY-d.y);
  let z=Math.round(d.z0*(1+shift/420)*20)/20;
  z=Math.max(0.6,Math.min(2,z));
  LAY.z[d.el.id]=z;
  d.el.style.setProperty("--z",z);
  tipShow(`масштаб ${Math.round(z*100)}%`,e.clientX,e.clientY);
}
function layMoveFree(e,d){
  const c=layCanvas(), b=d.box, k=d.kind, snap=!e.altKey;
  const dx=e.clientX-d.x, dy=e.clientY-d.y;
  let x=b.x, y=b.y, w=b.w, h=b.h, gv=null, gh=null;
  if(k==="move"){
    x=b.x+dx; y=b.y+dy;
    if(snap){
      const cand=laySnapEdges(d.el,c);
      const sx=laySnapMove(x,w,cand.v); x=sx.v; gv=sx.line;
      const sy=laySnapMove(y,h,cand.h); y=sy.v; gh=sy.line;
    }
    x=Math.max(0,Math.min(x,c.w-w));
    y=Math.max(0,y);
  }else{
    // Тяга за край двигает ИМЕННО ЭТОТ край, а противоположный стоит.
    // Поэтому считаем в координатах краёв, а ширину получаем разностью:
    // через «ширина плюс сдвиг» окно, ужатое за левый край до предела,
    // начинало бы уезжать вправо.
    let x2=b.x+b.w, y2=b.y+b.h;
    const cand=snap?laySnapEdges(d.el,c):{v:[],h:[]};
    if(k.indexOf("w")>=0){x=b.x+dx;
      const s=snap?laySnapOne(x,cand.v):null;   if(s!==null){x=s;gv=s;}}
    if(k.indexOf("e")>=0){x2=b.x+b.w+dx;
      const s=snap?laySnapOne(x2,cand.v):null;  if(s!==null){x2=s;gv=s;}}
    if(k.indexOf("n")>=0){y=b.y+dy;
      const s=snap?laySnapOne(y,cand.h):null;   if(s!==null){y=s;gh=s;}}
    if(k.indexOf("s")>=0){y2=b.y+b.h+dy;
      const s=snap?laySnapOne(y2,cand.h):null;  if(s!==null){y2=s;gh=s;}}
    x =Math.max(0,Math.min(x, x2-F_MINW));
    y =Math.max(0,Math.min(y, y2-F_MINH));
    x2=Math.min(c.w,Math.max(x2,x+F_MINW));
    y2=Math.max(y2,y+F_MINH);
    w=x2-x; h=y2-y;
  }
  const f=LAY.free[d.el.id];
  f.x=Math.round(x); f.y=Math.round(y);
  f.w=Math.round(w); f.h=Math.round(h);
  layPut(d.el,f);
  laySnapLine("v",gv); laySnapLine("h",gh);
  tipShow(k==="move"?`${f.x} · ${f.y}`:`${f.w} × ${f.h}`,e.clientX,e.clientY);
  layFit();
}
function layDrop(){
  const d=layDrag;
  layDrag=null;
  removeEventListener("pointermove",layMove);
  removeEventListener("pointerup",layDrop);
  removeEventListener("pointercancel",layDrop);
  document.body.style.userSelect="";
  tipHide();
  if(!d)return;
  d.el.classList.remove("dragging");
  d.el.style.pointerEvents="";
  if(d.free){
    laySnapLine("v",null); laySnapLine("h",null);
    layFit(); laySave(); return;
  }
  if(d.kind==="move"){
    if(d.over)d.over.classList.remove("dropzone");
    if(!d.over)return;
    // Меняем местами и ПОРЯДОК, и ширину. Только порядок — и сетка
    // перекладывается целиком: широкое окно уезжает в узкое место, всё
    // ниже съезжает, и «поменять местами» выглядит как «всё поехало».
    const i=LAY.order.indexOf(d.el.id), j=LAY.order.indexOf(d.over.id);
    if(i<0||j<0)return;
    LAY.order[i]=d.over.id; LAY.order[j]=d.el.id;
    const s=LAY.span[d.el.id];
    LAY.span[d.el.id]=LAY.span[d.over.id]; LAY.span[d.over.id]=s;
    layApply();
  }
  laySave();
}
// Ctrl+колесо — масштаб содержимого окна, в обоих режимах. preventDefault
// обязателен: без него браузер вместо окна увеличит всю страницу.
function layWheel(e){
  if(!e.ctrlKey)return;
  const el=e.target&&e.target.closest?e.target.closest("#panel>.card"):null;
  if(!el||CARDS.indexOf(el.id)<0)return;
  e.preventDefault();
  let z=(LAY.z[el.id]||1)*(e.deltaY<0?1.05:1/1.05);
  z=Math.max(0.6,Math.min(2,Math.round(z*20)/20));
  LAY.z[el.id]=z;
  el.style.setProperty("--z",z);
  tipShow(`масштаб ${Math.round(z*100)}%`,e.clientX,e.clientY);
  clearTimeout(layWheel.t);
  layWheel.t=setTimeout(()=>{tipHide();laySave();layFit();},600);
}

// --- полоса «раскладка» и запуск --------------------------------------------

function layBar(){
  const free=layMode()==="free", narrow=LAY.mode==="free"&&!free&&!BARE;
  document.getElementById("laygrid").classList.toggle("on",LAY.mode==="grid");
  document.getElementById("layfree").classList.toggle("on",LAY.mode==="free");
  document.getElementById("laytile").style.display=free?"":"none";
  document.getElementById("layhint").textContent = narrow
    ? "экран узкий — окна временно идут в столбик, раскладка сохранена"
    : free
      ? "шапка окна — перенести · края и углы — размер · Ctrl+колесо — масштаб · Alt — без прилипания"
      : "левый край окна — поменять местами · правый край — ширина · правый нижний угол — масштаб";
}
function laySetMode(m){
  if(LAY.mode===m)return;
  LAY.mode=m;
  layApply(); laySave();
}
// Содержимое окна заворачивается в две обёртки один раз, при старте.
// Идентификаторы внутри не трогаются, поэтому весь остальной код о них не
// знает и знать не должен. Зачем именно две — в комментарии к CSS.
function layWrap(el){
  const body=document.createElement("div"); body.className="cbody";
  const zoom=document.createElement("div"); zoom.className="czoom";
  while(el.firstChild)zoom.appendChild(el.firstChild);
  body.appendChild(zoom); el.appendChild(body);
}
function layInit(){
  for(const id of CARDS){
    const el=document.getElementById(id);
    if(!el)continue;
    layWrap(el);
    const add=(cls,title,kind,only)=>{
      const h=document.createElement("i");
      h.className=cls; h.title=title;
      h.addEventListener("pointerdown",e=>layGrab(e,el,kind,only));
      el.appendChild(h);
    };
    add("rail mv","перетащите на другое окно — поменяются местами","move","grid");
    add("rail wd","потяните — ширина окна","width","grid");
    add("zoomer","потяните — масштаб окна","zoom","grid");
    add("hd mv","перетащите — окно встанет куда угодно","move","free");
    for(const k of ["n","s","e","w","ne","nw","se","sw"])
      add("hd "+k,"потяните — ширина и высота окна",k,"free");
    // Щелчок по окну поднимает его над соседями. Именно pointerdown и без
    // stopPropagation: поднять надо и тогда, когда человек просто ткнул в
    // таблицу внутри, а не схватился за маркер.
    el.addEventListener("pointerdown",()=>{
      if(layMode()==="free"&&layRaise(el))laySave();
    });
  }
  document.getElementById("laygrid").onclick=()=>laySetMode("grid");
  document.getElementById("layfree").onclick=()=>laySetMode("free");
  document.getElementById("laytile").onclick=()=>{
    const w=layCanvas().w;
    LAY.free=layTile(w); LAY.cw=w; layApply(); laySave();
  };
  document.getElementById("layreset").onclick=()=>{
    const m=LAY.mode;
    try{localStorage.removeItem(LAY_KEY);localStorage.removeItem(LAY_KEY_V1);}
    catch(e){}
    LAY=layLoad(); LAY.mode=m; layApply(); laySave();
  };
  const b=document.getElementById("lay");
  b.onclick=()=>{
    const on=document.body.classList.toggle("laying");
    b.classList.toggle("on",on);
  };
  addEventListener("wheel",layWheel,{passive:false});
  // Сетке пересчёт при изменении размера окна не нужен — её раскладывает
  // сам браузер. Звать layApply вхолостую нельзя: он переносит карточки
  // через appendChild, то есть на каждый кадр протяжки окна дёргает DOM.
  let rafW=0;
  addEventListener("resize",()=>{
    if(rafW||(layMode()!=="free"&&layWas!=="free"))return;
    rafW=requestAnimationFrame(()=>{rafW=0;layApply();});
  });
  // Вердикт и сборки появляются посреди матча. Пока их нет, полотну нельзя
  // закладывать их высоту, а как только появились — надо. Наблюдатель
  // дешевле, чем звать layFit из каждой отрисовки и однажды забыть.
  if(window.ResizeObserver){
    let rafF=0;
    const ro=new ResizeObserver(()=>{
      if(rafF)return;
      rafF=requestAnimationFrame(()=>{rafF=0;layFit();});
    });
    for(const id of CARDS){
      const el=document.getElementById(id);
      if(el)ro.observe(el);
    }
  }
  layApply();
}
layInit();

async function startWatch(sid){
  if(!sid)return;
  await fetch("/api/watch?sid="+encodeURIComponent(sid));
  watching=sid; show("panel");
  document.getElementById("main").innerHTML='<div class="sub">подключаюсь…</div>';
  tick();
}

// Ручной ввод рисуется ВСЕГДА. Раньше ошибка списка подменяла собой всю
// страницу, и панель становилась недоступна даже при известном id матча —
// то есть отказ одного метода API убивал весь интерфейс.
function manualCard(){
  return `<div class="card"><h2>смотреть по id сервера</h2>
    <div class="dr"><input id="sid" placeholder="server_steam_id"
      style="flex:1;background:#0a0e13;color:var(--fg);border:1px solid var(--line);
      border-radius:6px;padding:6px 10px;font:inherit">
      <button id="go">смотреть</button>
      <button id="refresh">обновить список</button></div>
    <div class="sub" style="margin-top:8px"><span>id берётся из списка выше
      или из <code>python -m dwp.live --list</code>${watching?
      ` · сейчас смотрим <b>${esc(watching)}</b>`:""}</span></div></div>`;
}

function wireManual(){
  const go=()=>startWatch(document.getElementById("sid").value.trim());
  document.getElementById("go").onclick=go;
  document.getElementById("sid").addEventListener("keydown",e=>{
    if(e.key==="Enter")go();});
  document.getElementById("refresh").onclick=()=>loadGames(true);
  document.getElementById("refresh").textContent="обновить список";
  if(watching)document.getElementById("sid").value=watching;
}

async function loadGames(force){
  const el=document.getElementById("list");
  let d;
  try{ d=await (await fetch("/api/games"+(force?"?force=1":""),
                           {cache:"no-store"})).json(); }
  catch(e){
    el.innerHTML='<div class="card err">сервер dwp не отвечает</div>'+manualCard();
    wireManual(); return;
  }
  watching=d.watching;
  let top="";
  if(d.error)
    top=`<div class="card err">список матчей недоступен: ${esc(d.error)}
      <div class="sub" style="margin-top:6px"><span>Панель это не ломает —
      введите server_steam_id ниже.</span></div></div>`;
  else if(!d.games.length)
    top=`<div class="card warn">Сейчас в GetTopLiveGame пусто. Это штатно:
      метод показывает не все игры и между матчами отвечает пустым объектом.
      ${watching?"Идущий матч продолжает считаться.":""}</div>`;
  else
    top=`<div class="card"><h2>идущие матчи · клик чтобы смотреть</h2>
      <table><tr><th>команды</th><th>составы</th><th>лига</th>
      <th style="text-align:right">зрителей</th><th style="text-align:right">MMR</th></tr>
      ${d.games.map(g=>`<tr class="row ${g.sid===watching?"on":""}" data-sid="${esc(g.sid)}">
        <td class="l"><span style="color:${RAD}">${esc(g.radiant)}</span>
          <span style="color:var(--dim)"> vs </span>
          <span style="color:${DIRE}">${esc(g.dire)}</span></td>
        <td><span class="mini">${g.heroes.map(icon).join("")}</span></td>
        <td>${g.league||"—"}</td>
        <td style="text-align:right">${g.spectators||"—"}</td>
        <td style="text-align:right">${g.mmr||"—"}</td></tr>`).join("")}
      </table></div>`;
  el.innerHTML=top+manualCard();
  el.querySelectorAll("tr.row").forEach(tr=>
    tr.onclick=()=>startWatch(tr.dataset.sid));
  wireManual();
}

// Слепой режим: число модели закрыто, пока зритель не назовёт своё. Догадка
// уходит на сервер и ложится рядом с числом модели на ту же минуту; исход в
// этот момент неизвестен никому. Считается потом — dwp.blindtest --guesses.
let blind=Q.has("blind"), revealed=false, guessSent=null;
document.getElementById("blind").onclick=()=>{
  blind=!blind; revealed=false; guessSent=null;
  document.getElementById("blind").classList.toggle("on",blind);
  tick();
};
if(blind)document.getElementById("blind").classList.add("on");

// ВЕРДИКТ. Одна сторона вместо скачущего процента.
//
// Почему это не «второй прогнозный процент», который в проекте запрещён:
// никакой новой вероятности здесь не появляется. Показывается СТОРОНА той
// же самой калиброванной оценки, сглаженной по игровому времени, плюс
// доля, с которой такие вердикты сбывались на матчах, которых ни модель,
// ни подбор правила не видели. Число на экране остаётся одно.
//
// Правило и все доли — из data/verdict.json (dwp.verdict --tune). Нет
// файла — карточка не рисуется: сторона без замеренной доли попаданий
// была бы уверенностью, которой никто не мерил.
// Вердикт. Названная сторона больше не меняется до конца матча, поэтому и
// блок после фиксации СТАТИЧЕН: ни процент, ни доля попаданий здесь не
// шевелятся. Это не косметика — ровно это и просили («не должен ни разу
// колебаться, даже если противоположная сторона имеет 90%»).
function drawVerdict(s,hidden){
  const card=document.getElementById("vcard");
  const v=s.verdict;
  if(!v||hidden){card.style.display="none";return;}
  card.style.display="";
  const el=document.getElementById("verdict"), m=v.measured||{};

  if(!v.committed){
    const left=Math.max(0,(v.open_at||0)-(v.minute||0));
    document.getElementById("vtop").textContent="";
    // Полоса, а не процент: пока вердикта нет, зритель должен видеть, что
    // происходит, но числа рядом с будущим вердиктом быть не должно —
    // именно от скачущих чисел этот блок и заводился.
    const bar=(left>0||v.progress==null)?"":`
      <div class="vprog"><i style="width:${(v.progress*100).toFixed(0)}%"></i></div>`;
    el.innerHTML=`<div class="vd"><span class="kw">смотрю</span>
      <span class="wait">${left>0
        ? "вердикт с "+(+v.open_at).toFixed(0)+"-й минуты, осталось "
          +left.toFixed(1)+" мин"
        : "пока слишком близко — жду перевеса"}</span></div>${bar}`;
    return;
  }

  const col=v.side==="radiant"?RAD:DIRE;
  const h=v.hit;
  document.getElementById("vtop").textContent =
    `решено на ${(+v.commit_minute).toFixed(0)}-й минуте · больше не меняется`;
  el.innerHTML=`
    <div class="vd">
      <span class="kw">победит</span>
      <span class="who" style="color:${col}">${esc(v.name||v.side)}</span>
    </div>
    <div class="vdnote">${h
      ? `такие вердикты сбывались <b style="color:${
          h.hit>=0.7?RAD:WARN}">${pct(h.hit)}</b>
         <span style="color:var(--faint)">[${pct(h.lo)}, ${pct(h.hi)}] на
         ${h.n_matches} матчах холдаута</span>`
      : `<span style="color:var(--faint)">на такую позицию в холдауте меньше
         ${v.min_cell||40} матчей — доля не считается</span>`}</div>`;
}

// Табло. Процента модели здесь намеренно нет: он скачет, а на экране
// стоит вердикт, который не скачет, и два числа рядом спорили бы друг с
// другом. Здесь только факты матча — счёт, время, золото, здания.
function drawMain(s){
  const el=document.getElementById("main");
  const hidden=blind&&!revealed;
  const teamCell=(i,align)=>{
    const col=i?DIRE:RAD;
    return `<div style="text-align:${align};min-width:0">
      <div class="tname" style="font-size:20px;color:${col}">${esc(s.names[i])}</div>
      <div class="kbig">${s.kills[i]??"—"}</div></div>`;};
  el.innerHTML=`
   <div style="display:flex;justify-content:space-between;align-items:center;gap:14px">
     ${teamCell(0,"left")}
     <div style="text-align:center;flex:0 0 auto">
       <div class="clock">${esc(s.clock)}</div>
       <div class="gold">${s.gold_adv===null?"—":(s.gold_adv>0?"+":"")+num(s.gold_adv)+" золота"}</div>
     </div>
     ${teamCell(1,"right")}
   </div>
   <div class="sub" style="margin-top:6px">
     <span>вышек ${s.towers_lost[0]??"—"}:${s.towers_lost[1]??"—"}</span>
     <span>бараков ${s.rax_lost[0]??"—"}:${s.rax_lost[1]??"—"}</span></div>
   ${hidden?guessBox():""}`;
  if(hidden)wireGuess(s);
}

function guessBox(){
  if(guessSent!==null)
    return `<div class="chance" style="margin:10px 0 4px">
      <div>вы сказали <b>${(guessSent.p_human*100).toFixed(1)}%</b> за Radiant,
      модель — <b>${(guessSent.p_model*100).toFixed(1)}%</b>. Разница
      <b>${((guessSent.p_human-guessSent.p_model)*100>=0?"+":"")
        +((guessSent.p_human-guessSent.p_model)*100).toFixed(1)} п.п.</b></div>
      <div class="sub" style="margin-top:6px"><span>Ответ записан. Кто был прав,
      станет ясно после матча: <code>python -m dwp.blindtest --guesses</code></span></div>
      <div style="margin-top:8px"><button id="reveal">снять маску</button></div></div>`;
  return `<div class="chance" style="margin:10px 0 4px">
    <div style="margin-bottom:8px">Число модели закрыто. Назовите своё — по
    карте, счёту и сборкам.</div>
    <div style="display:flex;gap:8px;align-items:center">
      <input id="gs" type="number" min="1" max="99" step="1" placeholder="% за Radiant"
             style="width:150px">
      <button id="gsend">записать</button>
      <button id="reveal">просто показать</button></div></div>`;
}

function wireGuess(s){
  const rv=document.getElementById("reveal");
  if(rv)rv.onclick=()=>{revealed=true;tick();};
  const b=document.getElementById("gsend");
  if(!b)return;
  const go=async()=>{
    const v=parseFloat(document.getElementById("gs").value);
    if(!isFinite(v))return;
    try{
      const r=await(await fetch("/api/guess?p="+(v/100))).json();
      if(r.ok){guessSent=r;revealed=false;tick();}
    }catch(e){}
  };
  b.onclick=go;
  document.getElementById("gs").addEventListener("keydown",e=>{
    if(e.key==="Enter")go();});
}

// --- сборки ---------------------------------------------------------------
// Единственный блок, где видно, ВО ЧТО превратилось золото. Полоса
// нетворса общая на обе команды (масштаб — богатейший герой матча), иначе
// две колонки нормировались бы каждая на себя и отставание пропало бы.
function drawBuilds(s){
  const card=document.getElementById("bcard"), el=document.getElementById("builds");
  const b=s.builds;
  if(!b){card.style.display="none";return;}
  card.style.display="";
  const nw=t=>t.players.reduce((a,p)=>a+(p.nw||0),0);
  const totals=b.teams.map(nw);
  const mx=Math.max(1,...b.teams.flatMap(t=>t.players.map(p=>p.nw||0)));
  const lead=totals[0]>=totals[1]?0:1;
  const gap=Math.abs(totals[0]-totals[1]);
  document.getElementById("btop").textContent =
    gap>0?`+${Math.round(gap).toLocaleString("ru-RU")} нетворса у ${s.names[lead]}`:"";

  el.innerHTML=b.teams.map((t,i)=>{
    const col=i?DIRE:RAD;
    const rows=t.players.map(p=>{
      const w=Math.max(2,(p.nw||0)/mx*100);
      return `<div class="brow ${p.core?"core":""}">
        ${p.img?`<img class="hi" src="${esc(p.img)}" alt="">`
               :`<span class="hi"></span>`}
        <span class="lvl" title="уровень">${p.level??"?"}</span>
        <span class="nmw">
          <span class="nm">${esc(p.name)}</span>
          <span class="nwbar"><i style="width:${w}%;background:${col}"></i></span>
        </span>
        <span class="bnum nwn" title="нетворс">${num(p.nw)}</span>
        <span class="slots">${slotCells(p)}</span>
        <span class="bnum gld ${(p.gold||0)>2500?"hot":""}"
              title="свободное золото: ещё не превращено в предметы"
              >${num(p.gold)}</span>
      </div>`;}).join("");
    return `<div class="bteam">
      <div class="bhead">
        <span class="bt" style="color:${col}">${esc(s.names[i])}</span>
        <span class="bsum">${num(totals[i])}<span class="u">нетворс</span></span>
        <span class="bsum">${num(t.value)}<span class="u">в предметах</span></span>
        <span class="bsum ${(t.unspent||0)>6000?"hot":""}"
              >${num(t.unspent)}<span class="u">свободно</span></span>
      </div>
      ${rows}
    </div>`;}).join("");
}

function slotCells(p){
  let out="";
  for(let i=0;i<6;i++){
    const it=(p.items||[])[i];
    if(!it){out+=`<i class="empty"></i>`;continue;}
    const t=`${it.name}${it.cost?" · "+it.cost+" золота":""}`;
    out+=`<i class="${it.big?"big":""}${it.consumable?" cons":""}"
      title="${esc(t)}" style="background-image:url('${esc(it.img)}')"></i>`;
  }
  return out;
}

function drawMap(s){
  const m=s.map;
  // Единый пересчёт координат API в проценты картинки. Множитель подобран
  // наложением всех известных зданий на карту (config.MAP_IMAGE_SCALE).
  const S=m.scale;
  const px=v=>(0.5+v*S)*100, py=v=>(0.5-v*S)*100;
  const mapimg=document.getElementById("mapimg");
  mapimg.style.display=m.image?"":"none";
  if(m.image&&!mapimg.src)mapimg.src="/assets/map.jpg";
  // Своя рисовка остаётся запасным путём: нет картинки — карта не пропадает.
  document.getElementById("mapsvg").style.display=m.image?"none":"";
  if(!m.image)drawMapFallback(m,px,py);
  else document.getElementById("mapsvg").innerHTML="";
  drawPins(m,px,py);
  document.getElementById("mapnote").textContent = m.image
    ? (m.book<m.book_full?`справочник позиций ${m.book}/${m.book_full} — часть снесённых зданий ещё не отмечена`:"")
    : "карта не скачана, рисую схему — см. README, dwp.web скачивает её сама";
}

// Иконки героев НЕ пересоздаются каждый опрос, а переезжают. Разница не
// косметическая: при innerHTML браузер выбрасывает старый элемент вместе с
// его переходом, и герой скачками телепортируется. Сохраняя элемент, мы
// даём CSS доехать до новой точки за время между опросами — при опросе раз
// в 2 с движение выглядит непрерывным, хотя данных по-прежнему 30 в минуту.
const pinEls=new Map();
function drawPins(m,px,py){
  // Иконка 32px: если центр у самого края, половина уезжает за карту.
  // Поэтому центр прижимается к [3%, 97%] — позиция врёт максимум на
  // ширину иконки и только у самой кромки, зато герой всегда виден.
  const cl=v=>Math.max(3,Math.min(97,v));
  const box=document.getElementById("pins");
  const seen=new Set();
  for(const h of m.heroes){
    const key=h.team+":"+h.id;
    seen.add(key);
    let el=pinEls.get(key);
    if(!el){
      el=document.createElement("div");
      el.className="h";
      el.style.borderColor=h.team===m.radiant?RAD:DIRE;
      el.innerHTML=icon(h);
      box.appendChild(el);
      pinEls.set(key,el);
    }
    el.style.left=cl(px(h.x))+"%";
    el.style.top=cl(py(h.y))+"%";
    // Убитый герой уходит в чёрно-белое и ПОД живых. Второе не украшение:
    // в замесе десять иконок налезают друг на друга, и если сверху лежит
    // труп, то живого героя под ним просто не видно — а именно живые и
    // решают, чем замес кончится. Слой задаётся классом, а не инлайном,
    // чтобы правило было в одном месте (CSS .h.dead).
    el.classList.toggle("dead",!!h.dead);
    const kda=`${h.k??"?"}/${h.d??"?"}/${h.a??"?"}`;
    el.title=h.dead
      ? `${h.name} — УБИТ${h.dead_for!==null&&h.dead_for!==undefined
          ?` ${Math.round(h.dead_for)} с назад`:""}`
        +` · координаты стоят с точки смерти · ур.${h.level??"?"} · ${kda}`
      : `${h.name} · ур.${h.level??"?"} · ${kda} · ${num(h.nw)} нетворс`;
  }
  // Сменился матч — старые иконки убираем, иначе они останутся висеть.
  for(const [k,el] of pinEls){ if(!seen.has(k)){ el.remove(); pinEls.delete(k); } }
  let b="";
  for(const q of m.buildings){
    const col=q.team===m.radiant?RAD:DIRE;
    const t=`${q.dead?"снесено":"стоит"} · ${q.type===0?"вышка t"+q.tier:(q.type===1?"барак":"трон")}`;
    const sz=q.type===2?13:(q.type===1?9:8);
    b+=`<div class="b" title="${t}" style="left:${px(q.x)}%;top:${py(q.y)}%;
      width:${sz}px;height:${sz}px;background:${q.dead?"transparent":col};
      border:1.5px solid ${col};opacity:${q.dead?0.55:1};
      transform:translate(-50%,-50%) rotate(${q.type===2?45:0}deg)"></div>`;
  }
  document.getElementById("blds").innerHTML=b;
}

function drawMapFallback(m,px,py){
  // ГЕОМЕТРИЯ. Мид идёт по диагонали Radiant(низ-слева) -> Dire(верх-справа),
  // то есть в экранных процентах вдоль sy = 100 - sx. Река ПЕРПЕНДИКУЛЯРНА
  // миду, значит вдоль sy = sx: из левого верхнего угла в правый нижний.
  // Раньше река была нарисована вдоль мида — из-за этого карта не читалась.
  const RW=12;                                     // полуширина реки, в %
  const lanes=(m.lanes||[]).map(p=>p.map(q=>[px(q.x),py(q.y)]));
  let svg=`<defs>
      <linearGradient id="gr" x1="0" y1="1" x2="1" y2="0">
        <stop offset="0" stop-color="#16321c"/><stop offset="0.34" stop-color="#122616"/>
        <stop offset="0.5" stop-color="#0c1a18"/>
        <stop offset="0.66" stop-color="#241a1c"/><stop offset="1" stop-color="#33191b"/>
      </linearGradient>
      <linearGradient id="riv" x1="0" y1="0" x2="1" y2="1">
        <stop offset="0" stop-color="#12414f"/><stop offset="0.5" stop-color="#17566a"/>
        <stop offset="1" stop-color="#12414f"/>
      </linearGradient>
    </defs>
    <rect width="100" height="100" fill="url(#gr)"/>
    <polygon points="0,0 ${RW},0 100,${100-RW} 100,100 ${100-RW},100 0,${RW}"
             fill="url(#riv)" opacity="0.75"/>`;

  // Лес. Настоящих координат лагерей API не даёт, поэтому это ФАКТУРА:
  // детерминированная россыпь, из которой выброшено всё, что попало на
  // линию, в реку или на здание. Получаются ровно те четыре куска джунглей,
  // которые и есть между линиями.
  const near=(x,y,d)=>lanes.some(pl=>pl.some((p,i)=>{
    if(!i)return false;
    const [x1,y1]=pl[i-1],[x2,y2]=p, dx=x2-x1, dy=y2-y1;
    const L=dx*dx+dy*dy||1;
    let t=((x-x1)*dx+(y-y1)*dy)/L; t=t<0?0:t>1?1:t;
    return Math.hypot(x-(x1+t*dx), y-(y1+t*dy))<d;
  }));
  const bxy=m.buildings.map(b=>[px(b.x),py(b.y)]);
  // mulberry32: генератор фиксированный, чтобы лес не мельтешил между
  // опросами. Наивный LCG тут не годится — seed*1103515245 вылезает за 2^53,
  // младшие биты теряются, и последовательность вырождается.
  let seed=1337;
  const rnd=()=>{seed=seed+0x6D2B79F5|0;
    let t=Math.imul(seed^seed>>>15,1|seed);
    t=t+Math.imul(t^t>>>7,61|t)^t;
    return ((t^t>>>14)>>>0)/4294967296;};
  for(let i=0,put=0;i<1400&&put<230;i++){
    const x=rnd()*100, y=rnd()*100;
    if(Math.abs(y-x)<RW+2)continue;                       // река
    if(near(x,y,5.2))continue;                            // линия
    if(bxy.some(([bx,by])=>Math.hypot(x-bx,y-by)<5))continue;
    const dire=(x-y)>0;                                   // сторона Dire — выше диагонали
    const r=0.85+rnd()*0.75;
    svg+=`<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${r.toFixed(2)}"
          fill="${dire?"#243426":"#1e4526"}" opacity="${(0.5+rnd()*0.4).toFixed(2)}"/>`;
    put++;
  }

  for(const pl of lanes){
    const pts=pl.map(p=>`${p[0].toFixed(2)},${p[1].toFixed(2)}`).join(" ");
    svg+=`<polyline points="${pts}" fill="none" stroke="#5b5747" stroke-width="3.4"
          stroke-linejoin="round" stroke-linecap="round" opacity="0.55"/>
          <polyline points="${pts}" fill="none" stroke="#8d886f" stroke-width="1.5"
          stroke-linejoin="round" stroke-linecap="round" opacity="0.65"/>`;
  }
  // Здания и герои рисует drawPins поверх — одинаково для картинки и схемы.
  document.getElementById("mapsvg").innerHTML=svg;
}

// Состав за матч не меняется, а панель перерисовывается раз в 2 с. Пока
// блок собирался заново каждый тик, раскрытое «из чего это сложилось»
// схлопывалось через секунду: innerHTML пересоздаёт <details> и сбрасывает
// его open. Поэтому, во-первых, не перерисовываем без изменений,
// во-вторых, помним состояние раскрытия.
let draftKey="", draftOpen=false;

function drawDraft(s){
  const d=s.draft, el=document.getElementById("draft");
  if(!d||!d.ok){
    if(draftKey!=="none"){draftKey="none";
      el.innerHTML='<span class="warn">составы ещё не разобраны</span>';}
    return;
  }
  const key=JSON.stringify([d.radiant.map(h=>h.id), d.dire.map(h=>h.id),
                            d.logit, s.names]);
  if(key===draftKey)return;
  draftKey=key;

  // ГЛАВНОЕ И КРУПНО: у кого драфт сильнее. Всё остальное — под раскрытие.
  let verdict="";
  if(d.logit!==null){
    const p=1/(1+Math.exp(-d.logit)), pct=p*100;
    const lead=p>=0.5?0:1, col=p>=0.5?RAD:DIRE;
    const show=Math.max(pct,100-pct);
    const even=Math.abs(pct-50)<2;
    verdict=`<div style="font-size:15px;color:var(--dim)">драфт сильнее у</div>
      <div style="font-size:26px;font-weight:800;color:${even?"var(--dim)":col};
        line-height:1.15;margin:2px 0 4px">
        ${even?"поровну":esc(s.names[lead])}</div>
      <div style="font-size:15px"><b style="color:${col}">${show.toFixed(1)}%</b>
        <span style="color:var(--faint)"> до начала игры</span></div>`;
  }
  const second="";

  const all=[...d.radiant,...d.dire].filter(h=>h.coef!==null);
  const mx=Math.max(0.02,...all.map(h=>Math.abs(h.coef)));
  const row=(h,side)=>{
    const col=side==="radiant"?RAD:DIRE;
    if(h.coef===null)return `<div class="dr">${icon(h)}<span class="n">${esc(h.name)}</span>
      <span class="q">нет в справочнике</span></div>`;
    const w=Math.abs(h.coef)/mx*50, pos=h.coef>=0;
    // Незначимое рисуем блёклым: иначе полоска у героя, встреченного
    // 384 раза, выглядит так же уверенно, как у встреченного 2600 раз.
    const fill=h.sig?(pos?RAD:DIRE):"#39424e";
    const tip=`${h.picks} матчей в обучении, ошибка примерно ±${(h.se||0).toFixed(3)}`
      +(h.sig?"":" — перевес не отличим от нуля");
    return `<div class="dr" title="${tip}">${icon(h)}
      <span class="n" style="color:${col}">${esc(h.name)}</span>
      <span class="g"><i style="left:${pos?50:50-w}%;width:${w}%;
        background:${fill}"></i></span>
      <span class="q" style="${h.sig?"":"color:#4b5563"}">
        ${h.coef>=0?"+":""}${h.coef.toFixed(3)}</span>
      <span class="q" style="width:64px;color:#4b5563">${h.picks} м.</span></div>`;};
  const extra=[];
  if(d.elo)extra.push(`<div class="dr"><span style="width:38px"></span>
    <span class="n">${esc(d.elo.label||"рейтинг")} (${
      d.elo.value>=0?"+":""}${Math.round(d.elo.value)})</span>
    <span class="g"></span><span class="q">${d.elo.coef>=0?"+":""}${d.elo.coef.toFixed(3)}</span>
    <span class="q" style="width:64px"></span></div>`);
  if(d.intercept!==null)extra.push(`<div class="dr"><span style="width:38px"></span>
    <span class="n">свободный член</span><span class="g"></span>
    <span class="q">${d.intercept>=0?"+":""}${d.intercept.toFixed(3)}</span>
    <span class="q" style="width:64px"></span></div>`);
  const weak=all.filter(h=>!h.sig).length;

  el.innerHTML=verdict+second+drawWhy(d,s.names)+`
    <details style="margin-top:12px">
      <summary style="cursor:pointer;color:var(--dim);font-size:12px">
        из чего это сложилось (${weak} из ${all.length} героев — в пределах шума)
      </summary>
      <div class="sub" style="margin:8px 0 4px"><span>вправо — в пользу
        Radiant</span><span>матчей в обучении</span></div>`
    + d.radiant.map(h=>row(h,"radiant")).join("")
    + '<div style="height:6px"></div>'
    + d.dire.map(h=>row(h,"dire")).join("")
    + '<div style="height:6px"></div>' + extra.join("")
    + `<div class="warn" style="margin-top:8px">Блёклым помечено то, что не
       набрало двух ошибок: почти любая полоска короче 0.2 — шум.</div>
       </details>`;
  const det=el.querySelector("details");
  if(det){
    det.open=draftOpen;
    det.addEventListener("toggle",()=>{draftOpen=det.open;});
  }
}

// --- данные и обучение -----------------------------------------------------
// Команда НЕ приходит из браузера: страница называет ключ из фиксированного
// списка, а командную строку собирает сервер (dwp/jobs.py). Запуск идёт
// методом POST, потому что GET браузер может дёрнуть сам — предзагрузкой
// ссылки или восстановлением вкладки, — и обучение стартовало бы без ведома
// человека.
let labTimer=null, labJob=null, labTasks=[];

async function loadLab(){
  const el=document.getElementById("lab");
  let d,j;
  try{
    d=await (await fetch("/api/status",{cache:"no-store"})).json();
    j=await (await fetch("/api/jobs",{cache:"no-store"})).json();
  }catch(e){el.innerHTML='<div class="card err">сервер не отвечает</div>';return;}
  labTasks=d.tasks;
  const s=d.data;
  const pill=(ok,a,b)=>`<span class="pill ${ok?"on":"off"}">${esc(ok?a:b)}</span>`;
  const head=`<div class="card"><h2>что есть на этой машине</h2>
    <div class="kpi" style="grid-template-columns:repeat(auto-fit,minmax(150px,1fr))">
      <div><div class="n">${s.matches.toLocaleString("ru-RU")}</div>
        <div class="t">матчей скачано<br>из них ${s.holdout} в слепом холдауте</div></div>
      <div><div class="n">${s.ensemble.length||"—"}</div>
        <div class="t">моделей в ансамбле<br>${esc(d.model)}</div></div>
      <div><div class="n">${s.live_logs}</div>
        <div class="t">записанных матчей<br>исходы добраны у ${s.resolved}</div></div>
      <div><div class="n">${s.reveals}</div>
        <div class="t">взглядов в холдаут<br>чем больше, тем меньше слепоты</div></div>
    </div>
    <div class="sub" style="margin-top:12px;flex-wrap:wrap;gap:6px">
      ${pill(s.steam_key,"ключ Steam есть","ключа Steam нет")}
      ${pill(s.stratz_token,"токен Stratz есть","токена Stratz нет")}
      ${pill(s.horizon_table,"коридор посчитан","коридор не посчитан")}
      ${pill(s.comeback_table,"таблица камбэков есть","таблицы камбэков нет")}
      ${pill(s.learning_curve,"кривая обучения есть","кривой обучения нет")}
      <span style="flex:1"></span>
      <span style="color:var(--faint)">${esc(s.root)}</span></div></div>`;

  const cur=j.current;
  const busy=cur&&cur.status==="running";
  const cards=labTasks.map(t=>{
    const inputs=(t.params||[]).map(p=>
      `<label class="sub" style="display:inline-flex;gap:6px;align-items:center">
        <span style="color:var(--faint)">${esc(p.label)}</span>
        <input id="p-${esc(t.key)}-${esc(p.name)}" type="number" value="${p.default}"
          min="${p.min}" max="${p.max}" style="width:96px"></label>`).join("");
    return `<div class="job"><h3>${esc(t.title)}</h3><p>${esc(t.about)}</p>
      <div class="go">${inputs}
        <button data-run="${esc(t.key)}" ${busy?"disabled":""}
          style="${busy?"opacity:.45;cursor:not-allowed":""}">запустить</button>
        ${t.steps>1?`<span class="sub" style="color:var(--faint)">${t.steps} шагов</span>`:""}
      </div></div>`;}).join("");

  el.innerHTML=head+`<div class="card" style="margin-top:14px"><h2>задачи
    <span class="rt">${busy?"идёт «"+esc(cur.title)+"»":"свободно"}</span></h2>
    <div class="grid2">${cards}</div></div>`
    +jobCard(cur)
    +(j.history.length>1?histCard(j.history):"");
  el.querySelectorAll("[data-run]").forEach(b=>b.onclick=()=>runJob(b.dataset.run));
  const st=document.getElementById("jstop");
  if(st)st.onclick=async()=>{await fetch("/api/jobs/stop",{method:"POST"});loadLab();};
  if(busy&&!labTimer)labTimer=setInterval(()=>{if(view==="lab")loadLab();
    else{clearInterval(labTimer);labTimer=null;}},1500);
  if(!busy&&labTimer){clearInterval(labTimer);labTimer=null;}
}

function jobCard(c){
  if(!c)return `<div class="card" style="margin-top:14px"><h2>лог</h2>
    <div class="sub"><span style="color:var(--faint)">ещё ничего не
    запускали</span></div></div>`;
  const col=c.status==="running"?WARN:(c.code===0?RAD:DIRE);
  const state=c.status==="running"
    ? `идёт ${c.steps>1?`(шаг ${c.step+1} из ${c.steps})`:""}`
    : (c.code===0?"готово":`код возврата ${c.code}`);
  return `<div class="card" style="margin-top:14px">
    <h2>${esc(c.title)}<span class="rt" style="color:${col}">${esc(state)} ·
      ${Math.round(c.elapsed)} с</span></h2>
    <div class="sub" style="margin-bottom:8px"><span style="color:var(--faint)">
      <code>python -m ${esc(c.cmd)}</code></span>
      ${c.status==="running"?'<button id="jstop">прервать</button>':""}</div>
    <pre class="log" id="jlog">${esc((c.log||[]).join("\n"))}</pre>
    ${c.status!=="running"&&c.after?`<div class="warn">${esc(c.after)}</div>`:""}
    </div>`;
}

function histCard(h){
  return `<div class="card" style="margin-top:14px"><h2>что уже запускали</h2>
    <table><tr><th>задача</th><th>итог</th>
      <th style="text-align:right">секунд</th></tr>
    ${h.map(j=>`<tr><td class="l">${esc(j.title)}</td>
      <td style="color:${j.status==="running"?WARN:(j.code===0?RAD:DIRE)}">
        ${j.status==="running"?"идёт":(j.code===0?"готово":"код "+j.code)}</td>
      <td style="text-align:right">${Math.round(j.elapsed)}</td></tr>`).join("")}
    </table></div>`;
}

async function runJob(key){
  const t=labTasks.find(x=>x.key===key)||{};
  const params={};
  (t.params||[]).forEach(p=>{
    const el=document.getElementById(`p-${key}-${p.name}`);
    if(el)params[p.name]=parseInt(el.value,10);});
  try{
    const r=await(await fetch("/api/jobs/start",{method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({key,params})})).json();
    if(!r.ok)alert(r.error||"не запустилось");
  }catch(e){alert("сервер не ответил");}
  loadLab();
}

// --- точность --------------------------------------------------------------
async function loadAcc(){
  const el=document.getElementById("acc");
  el.innerHTML='<div class="card"><div class="sub">считаю…</div></div>';
  let d;
  try{ d=await (await fetch("/api/accuracy",{cache:"no-store"})).json(); }
  catch(e){el.innerHTML='<div class="card err">сервер не отвечает</div>';return;}
  if(d.error){el.innerHTML=`<div class="card err">${esc(d.error)}</div>`;return;}
  el.innerHTML=accHead(d)+accLive(d)+accModels(d)+accReveals(d);
}

// Две строки, ради которых раздел и открывают. Обе с одного вскрытия
// слепого холдаута: собственные тесты моделей между собой несравнимы —
// у разных сидов разные тестовые выборки.
function accHead(d){
  const h=d.headline;
  if(!h)return `<div class="card"><h2>точность</h2><div class="sub">
    <span style="color:var(--warn)">не посчитана. Раздел «данные и обучение»
    → «Все модели на холдауте».</span></div></div>`;
  const v=h.verdict||{};
  const row=(name,acc,extra,n)=>`<tr>
    <td class="l">${esc(name)}</td>
    <td class="v" style="font-size:20px;font-weight:700">${
      acc==null?"—":(acc*100).toFixed(1)+"%"}</td>
    <td class="l" style="color:var(--faint)">${extra}</td>
    <td class="v" style="color:var(--faint)">${n}</td></tr>`;
  const rows=[
    row("лайв-модель, каждую минуту матча", (h.live||{}).acc,
        `log loss ${(h.live||{}).log_loss?.toFixed(4)} · ECE ${
          (h.live||{}).ece?.toFixed(4)}`,
        `${h.n_matches} матчей`),
    h.draft?row("драфт-модель, до начала игры", h.draft.acc,
        `AUC ${h.draft.auc.toFixed(4)}`, `${h.draft.n_matches} матчей`):"",
    v.acc_commit!=null?row("ВЕРДИКТ, одна сторона на матч", v.acc_commit,
        `95% [${(v.acc_commit_lo*100).toFixed(1)}, ${
          (v.acc_commit_hi*100).toFixed(1)}] · называется на ${
          v.commit_minute_p50?.toFixed(0)}-й минуте`,
        `${v.n_commit} матчей`):"",
  ].join("");
  return `<div class="card"><h2>точность<span class="rt">слепой холдаут,
    ${d.holdout_permille/10}% матчей, которых обучение не видело</span></h2>
    <table>${rows}</table></div>`;
}

function accModels(d){
  const rows=d.models.filter(m=>!m.error).map(m=>`<tr>
    <td class="l">${esc(m.name)}${m.holdout?'':' <span class="pill off">видела холдаут</span>'}</td>
    <td class="v">${m.features}</td>
    <td class="v">${m.log_loss==null?"—":m.log_loss.toFixed(4)}</td>
    <td class="v">${m.brier==null?"—":m.brier.toFixed(4)}</td>
    <td class="v">${m.auc==null?"—":m.auc.toFixed(4)}</td>
    <td class="v">${m.baseline==null?"—":m.baseline.toFixed(4)}</td>
    <td class="v">${m.n_test_matches||"—"}</td>
    <td class="v">${m.draft_auc==null?"—":m.draft_auc.toFixed(3)}</td></tr>`).join("");
  return `<div class="card" style="margin-top:14px">
    <h2>что обучено<span class="rt">метрики собственного теста каждой
      модели — между собой НЕ сравнимы</span></h2>
    <table><tr><th>файл</th><th class="v">призн.</th><th class="v">log loss</th>
      <th class="v">brier</th><th class="v">auc</th><th class="v">база</th>
      <th class="v">тест, матчей</th><th class="v">драфт auc</th></tr>
      ${rows}</table></div>`;
}

function accLive(d){
  const L=d.live||{};
  if(L.note&&!(L.models||[]).length)
    return `<div class="card" style="margin-top:14px"><h2>лайв: как врало
      число на экране</h2><div class="sub"><span style="color:var(--warn)">
      ${esc(L.note)}</span></div></div>`;
  const rows=(L.models||[]).map(m=>`<tr>
    <td class="l">${esc(m.model)}</td>
    <td class="v">${m.n_matches}</td>
    <td class="v">${m.acc==null?"—":(m.acc*100).toFixed(1)+"%"}</td>
    <td class="v">${m.log_loss==null?"—":m.log_loss.toFixed(4)}</td>
    <td class="v" style="color:${m.ece>0.05?WARN:"var(--dim)"}">
      ${m.ece==null?"—":m.ece.toFixed(3)}</td>
    <td class="l" style="color:var(--faint)">${esc(m.note||"")}</td></tr>`).join("");
  return `<div class="card" style="margin-top:14px">
    <h2>на реально просмотренных матчах<span class="rt">${L.n_matches} матчей${
      L.pending?`, ещё ${L.pending} без исхода`:""}</span></h2>
    <table><tr><th>модель</th><th class="v">матчей</th><th class="v">точность</th>
      <th class="v">log loss</th><th class="v">ECE</th><th></th></tr>${rows}</table>
    </div>`;
}

function accReveals(d){
  const r=d.reveals||[];
  return `<div class="card" style="margin-top:14px">
    <h2>взгляды в слепой холдаут<span class="rt">${r.length} записей</span></h2>
    ${r.length?`<table><tr><th>когда</th><th>модель</th><th class="v">матчей</th>
      <th class="v">log loss</th><th class="v">ECE</th><th>примечание</th></tr>
      ${r.slice(-12).reverse().map(x=>`<tr><td class="l">${esc(x.when)}</td>
        <td class="l">${esc(x.model)}</td><td class="v">${esc(x.n_matches)}</td>
        <td class="v">${esc(x.log_loss)}</td><td class="v">${esc(x.ece)}</td>
        <td class="l" style="color:var(--faint)">${esc(x.note||"")}</td></tr>`).join("")}
      </table>`:`<div class="sub"><span>ни разу</span></div>`}
    <div class="warn">${d.holdout_permille/10}% матчей спрятаны от обучения
      навсегда — по хэшу match_id, а не по сиду, чтобы добор данных не
      переносил матчи туда-сюда. Каждая строка выше — это взгляд в слепую
      выборку: выбирая правку по этим числам, вы подгоняете модель под
      холдаут ровно так же, как раньше подгоняли под тест.</div></div>`;
}

// --- лента событий ---------------------------------------------------------
// Журнала в источнике нет: события выводятся из разницы счётчиков между
// опросами. Поэтому пара «кто кого» показывается ТОЛЬКО когда она следует
// из данных однозначно, а замес показывается замесом.
function drawFeed(s){
  const f=s.feed, el=document.getElementById("feed");
  const card=document.getElementById("fcard");
  if(!f){card.style.display="none";return;}
  card.style.display="";
  document.getElementById("frtop").textContent=`${f.n} событий`;
  if(!f.events.length){
    el.innerHTML=`<div class="sub"><span style="color:var(--faint)">Пока тихо
      </span></div>`;
    return;}
  const port=(h,dead)=>{
    const col=h.team===2?RAD:DIRE;
    return h.img
      ? `<img class="${dead?"dead":""}" style="border-color:${col}"
          src="${esc(h.img)}" alt="${esc(h.name)}" title="${esc(h.name)}">`
      : `<span class="ph ${dead?"dead":""}" style="border-color:${col}"
          title="${esc(h.name)}">${esc((h.name||"?").slice(0,3))}</span>`;};
  el.innerHTML=f.events.map(e=>{
    const t=e.minute==null?"":`${Math.floor(e.minute)}:`
      +String(Math.round((e.minute%1)*60)).padStart(2,"0");
    if(e.kind==="kill")
      return `<div class="ev"><span class="m">${t}</span>
        ${port(e.killers[0])}<span class="x">убил</span>${port(e.victims[0],1)}
        <span class="tx"></span></div>`;
    if(e.kind==="death")
      return `<div class="ev"><span class="m">${t}</span>
        ${e.victims.map(v=>port(v,1)).join("")}
        <span class="tx">${esc(e.note)}</span></div>`;
    if(e.kind==="fight")
      return `<div class="ev guessy"><span class="m">${t}</span>
        ${e.killers.map(v=>port(v)).join("")}<span class="x">×</span>
        ${e.victims.map(v=>port(v,1)).join("")}
        <span class="tx">замес, пары не восстановимы</span></div>`;
    return `<div class="ev"><span class="m">${t}</span>
      <span class="tx" style="color:${e.kind==="roshan"?WARN:"var(--dim)"}">
      ${esc(e.note)}</span></div>`;}).join("");
  // Подписи под лентой больше нет: оговорка про восстановимость пар стоит
  // там, где она к делу, — прямо в строке замеса.
}

// --- почему драфт сильнее --------------------------------------------------
function drawWhy(d,names){
  const w=d.why||[];
  if(!w.length)return "";
  const row=x=>{
    const col=x.weight==null?"var(--dim)":(x.weight>=0?RAD:DIRE);
    const val=x.weight==null?"":`<b style="color:${col}">
      ${x.weight>=0?"+":""}${x.weight.toFixed(3)}</b>`;
    return `<div class="sub" style="align-items:baseline;margin-top:5px"
       title="${esc(x.note||"")}">
      <span>${x.sig?"":"<span style='color:var(--faint)'>≈</span> "}
        ${esc(x.text)} ${val}</span>
      <span style="color:var(--faint);flex:0 0 auto">${esc(x.source)}${
        x.counted?"":" · не в проценте"}</span></div>`;};
  return `<div style="margin-top:10px;padding-top:8px;
      border-top:1px solid var(--line)">
    <div class="sub"><span style="color:var(--faint)">почему</span></div>
    ${w.map(row).join("")}</div>`;
}

async function tick(){
  if(view!=="panel")return;
  let s;
  try{ s=await (await fetch("/api/state",{cache:"no-store"})).json(); }
  catch(e){ document.getElementById("main").innerHTML=
    '<div class="err">сервер dwp не отвечает</div>'; return; }
  // Замечания живого пути (нет поля в ответе, признак ещё не набрался,
  // рейтинг известен не у всех) в панель не выводятся строками — их бывает
  // с десяток, и они забьют экран. Но и терять их нельзя: счётчик в шапке,
  // список — подсказкой на нём.
  const hdr=document.getElementById("hdr");
  const w=s.warns||[];
  hdr.innerHTML=(s.model?esc("модель "+s.model)+" · ":"")+esc(s.log||"")
    +(w.length?` · <span style="color:var(--warn)">замечаний ${w.length}
       </span>`:"");
  hdr.title=w.join("\n");
  if(!s.ok){
    document.getElementById("main").innerHTML=
      `<div style="color:${s.waiting?WARN:DIRE}">${esc(s.error||"нет данных")}</div>`;
    // Иначе на экране остался бы вердикт по прошлому матчу — молча и
    // выглядя свежим.
    document.getElementById("vcard").style.display="none";
    return;
  }
  if(lastMatch!==s.match_id){lastMatch=s.match_id;revealed=false;guessSent=null;}
  drawMain(s);
  // В слепом режиме прячется РОВНО вердикт. Раньше пряталась ещё и половина
  // панели — тогда на экране стоял разбор вкладов, по которому оценка
  // восстанавливалась за секунду. Его больше нет, а счёт, карта, сборки и
  // лента — это факты матча: именно по ним зритель и должен называть своё
  // число, прятать их значит просить угадать вслепую.
  drawVerdict(s,blind&&!revealed);
  drawBuilds(s); drawMap(s); drawDraft(s); drawFeed(s);
}
let lastMatch=null;
// Стартовать сразу в панели, если матч уже выбран (--server-steam-id).
(async()=>{
  let fileMode=false;
  try{
    const d=await (await fetch("/api/games",{cache:"no-store"})).json();
    watching=d.watching; fileMode=!!d.file_mode;
  }catch(e){}
  if(watching||fileMode){show("panel");tick();}else{show("list");}
  setInterval(()=>{
    if(view==="list")loadGames();
    else if(view==="panel")tick();
    // «данные» обновляются своим таймером, только пока идёт задача;
    // «точность» — по заходу в раздел: она читает артефакты и логи, и
    // дёргать это раз в две секунды незачем.
  },2000);
})();
</script></body></html>
"""


def ensure_map_image() -> bool:
    """Скачать подробную карту, если её ещё нет.

    В репозиторий не кладём: это ассет Valve, лежащий в odota/web. Качается
    один раз на машину пользователя. Не получилось — не беда, панель рисует
    свою схему, о чём и пишет под картой.
    """
    p = config.MAP_IMAGE_PATH
    if p.exists() and p.stat().st_size > 10000:
        return True
    try:
        import requests
        r = requests.get(config.MAP_IMAGE_URL, timeout=30)
        r.raise_for_status()
        if len(r.content) < 10000:
            raise ValueError(f"ответ {len(r.content)} байт — не похоже на карту")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(r.content)
        print(f"Карта скачана: {len(r.content)} байт -> {p}")
        return True
    except Exception as e:                                  # noqa: BLE001
        print(f"ВНИМАНИЕ: карту скачать не удалось ({type(e).__name__}: {e}).\n"
              f"  Панель нарисует свою схему. Можно положить файл вручную:\n"
              f"    {config.MAP_IMAGE_URL}\n  -> {p}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    ap = argparse.ArgumentParser(
        description="Веб-панель лайва: список матчей, выбор, аналитика.")
    ap.add_argument("--server-steam-id",
                    help="сразу смотреть этот матч, минуя список")
    ap.add_argument("--from-file", type=Path,
                    help="крутить один сохранённый ответ вместо сети")
    ap.add_argument("--model", type=Path, nargs="+", default=None,
                    help="модель или несколько: тогда вероятности "
                         "усредняются. По умолчанию — ансамбль "
                         "models\\ens_*.pkl, если он обучен, иначе "
                         "live_exact.pkl (замер: log loss 0.5233 против "
                         "0.5285, ECE 0.0082 против 0.0155-0.0315)")
    ap.add_argument("--interval", type=float, default=DEFAULT_POLL,
                    help=f"как часто опрашивать Steam, секунд (по умолчанию "
                         f"{DEFAULT_POLL:g}; источник обновляется раз в ~1.2 с)")
    ap.add_argument("--log-every", type=float, default=DEFAULT_LOG_EVERY,
                    help="как часто писать строку в лог, секунд")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1",
                    help="0.0.0.0 откроет панель всей локальной сети")
    ap.add_argument("--log-dir", type=Path, default=config.LIVE_LOG_DIR)
    ap.add_argument("--no-log", dest="log", action="store_false",
                    help="не записывать матч (по умолчанию записывается)")
    ap.set_defaults(log=True)
    args = ap.parse_args(argv)

    try:
        paths = live.resolve_models(args.model or live.default_models())
        art = live.load_models(paths)
    except live.LiveError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    ensure_map_image()

    key = None
    if args.from_file is None:
        try:
            key = live.get_key()
        except live.LiveError as e:
            # Не выходим: страница поднимется и объяснит, чего не хватает.
            print(f"ВНИМАНИЕ: {e}\n", file=sys.stderr)

    poller = Poller(art, live.log_model_name(art), key,
                    max(MIN_POLL, args.interval),
                    args.from_file, args.log_dir, args.log,
                    log_every=max(args.interval, args.log_every))
    if args.server_steam_id:
        poller.watch(args.server_steam_id)
    poller.start()
    runner = JOBS.Runner()

    srv = ThreadingHTTPServer((args.host, args.port),
                              partial(Handler, poller, runner))
    shown = "127.0.0.1" if args.host == "0.0.0.0" else args.host
    url = f"http://{shown}:{args.port}/"
    print(f"Панель: {url}")
    print("  меню: матчи, данные и обучение, точность")
    print("  окна панели двигаются мышью: левый край — поменять местами, "
          "правый — ширина,\n         правый нижний угол — масштаб; "
          "раскладка запоминается браузером")
    n_models = len(live.members(art))
    print(f"  модель {art.get('name')}"
          + (f" — усреднение по {n_models} моделям" if n_models > 1 else "")
          + f", опрос раз в {poller.interval:g} с "
          f"(источник обновляется раз в ~1.2 с), запись "
          + (f"раз в {poller.log_every:g} с в {args.log_dir}" if args.log
             else "выключена"))
    v = poller.vtable or {}
    m = v.get("measured") or {}
    r = v.get("rule") or {}
    print("  вердикт: "
          + (f"с {r.get('t_open', 7):g}-й минуты, сбывался в "
             f"{m['acc_commit'] * 100:.1f}% на {m['n_commit']} матчах холдаута"
             if m.get("acc_commit") is not None else
             "правило не собрано — python -m dwp.verdict --tune"))
    have_items = ("да" if poller.items.ok
                  else "НЕТ (справочник предметов не загрузился)")
    print(f"  сборки: {have_items}")
    print(f"  слепой режим — кнопка в шапке или {url}?blind=1; "
          f"догадки считает python -m dwp.blindtest --guesses")
    print(f"  для OBS: {url}?bare=1 (прозрачный фон). Ctrl+C для выхода.")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
        return 130
    finally:
        poller.stop.set()
        poller.wake.set()
        srv.server_close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
