"""Долгие задачи из браузера: скачать матчи, обучить, замерить.

ЗАЧЕМ. Панель — единственный интерфейс, которым пользуются во время
матча, а всё остальное живёт в консоли: сбор данных, обучение, замеры.
Держать это в двух разных местах — значит гарантировать, что половина
никогда не запустится.

ЧТО ЗДЕСЬ ВАЖНО НЕ СДЕЛАТЬ: дырку в машине. Веб-сервер слушает порт, и
принимать от него команды на исполнение нельзя ни в каком виде. Поэтому:

* команда НЕ приходит из браузера. Браузер называет ключ из
  фиксированного списка `SPECS`, а сама командная строка собирается здесь;
* параметры — только числа, и каждое проверяется границами;
* запускается `sys.executable -m dwp.<модуль>` списком аргументов, без
  оболочки. Строка в shell не попадает вообще, подставлять в неё нечего;
* одновременно идёт одна задача. Два обучения на одной машине просто
  отнимают друг у друга процессор, а два `collect` дерутся за один файл
  состояния.

Лог задачи держится в памяти кольцом: обучение печатает сотни строк, и
хранить их все ради странички незачем.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque

from . import config

MAX_LOG_LINES = 400


def _seed_name(seed: int) -> str:
    return f"ens_{seed:02d}.pkl"


# Что вообще разрешено запускать. Ключ приходит из браузера, всё
# остальное собирается здесь. `build` получает проверенные параметры и
# возвращает СПИСОК аргументов после `python -m`.
SPECS: dict[str, dict] = {
    "collect": {
        "title": "Скачать матчи",
        "about": "Возобновляемая загрузка про-матчей с OpenDota. Прерывать "
                 "можно: следующий запуск продолжит с того же места.",
        "params": {"n": {"min": 10, "max": 20000, "default": 500,
                         "label": "сколько матчей добрать"}},
        "build": lambda p: ["dwp.collect", "--n", str(p["n"])],
        "after": "После сбора обязательно `Проверить схему`, а потом "
                 "переобучить набор моделей.",
    },
    "check_schema": {
        "title": "Проверить схему данных",
        "about": "Эмпирическая проверка допущений парсера на реальных матчах: "
                 "что значит поле team в событиях о вышках и Рошане. Обязательна "
                 "перед обучением на новых данных.",
        "params": {},
        "build": lambda p: ["dwp.check_schema"],
    },
    "train_one": {
        "title": "Обучить одну модель",
        "about": "Одиночная модель с указанным сидом разбиения. Кладётся в "
                 "models/ens_NN.pkl и становится частью ансамбля.",
        "params": {"seed": {"min": 1, "max": 999, "default": 7,
                            "label": "сид разбиения"}},
        "build": lambda p: [
            "dwp.train", "--source", "real", "--live-features",
            "--extra-features", "--exact-features", "--gold-norm", "add",
            "--rating", "player", "--seed", str(p["seed"]),
            "--out", str(config.MODELS_DIR / _seed_name(p["seed"]))],
    },
    "train_ensemble": {
        "title": "Обучить ансамбль (5 сидов)",
        "about": "Пять моделей с разными сидами. Замерено на слепом холдауте "
                 "(1864 матча): ансамбль log loss 0.5180 против 0.5192-0.5273 "
                 "у одиночных, ECE 0.0121. Около десяти минут.",
        "params": {},
        "steps": lambda p: [
            ["dwp.train", "--source", "real", "--live-features",
             "--extra-features", "--exact-features", "--gold-norm", "add",
             "--rating", "player",
             "--seed", str(s), "--out", str(config.MODELS_DIR / _seed_name(s))]
            for s in (3, 5, 7, 11, 13)],
        "after": "Дальше — `Переобучить всё` или вручную: "
                 "`Все модели на холдауте`, затем `Правило вердикта`.",
    },
    # Одна кнопка на весь путь от свежих матчей до вердикта на экране.
    # Порядок здесь не косметический: verdict читает кэш вероятностей,
    # который пишет bench_models, а тот — модели, которые кладёт train.
    # Перепутать шаги значит настроить вердикт по прошлому ансамблю.
    "retrain_all": {
        "title": "Переобучить всё заново",
        "about": "Полный проход: пять моделей ансамбля, замер всех моделей на "
                 "слепом холдауте, подбор правила вердикта. Порядок важен — "
                 "вердикт настраивается на тот ансамбль, который получился. "
                 "Около пятнадцати минут, машина будет занята.",
        "params": {},
        "steps": lambda p: [
            ["dwp.train", "--source", "real", "--live-features",
             "--extra-features", "--exact-features", "--gold-norm", "add",
             "--rating", "player",
             "--seed", str(s), "--out", str(config.MODELS_DIR / _seed_name(s))]
            for s in (3, 5, 7, 11, 13)
        ] + [
            ["dwp.bench_models"],
            ["dwp.verdict", "--tune", "--frontier"],
        ],
        "after": "Панель подхватит новые модели после перезапуска "
                 "`python -m dwp.web`.",
    },
    "bench_ensemble": {
        "title": "Замерить ансамбль на холдауте",
        "about": "Ансамбль против средней одиночной модели на 15% матчей, "
                 "спрятанных от обучения. Каждый запуск дописывается в реестр "
                 "вскрытий: холдаут тем слепее, чем реже туда смотрят.",
        "params": {},
        "build": lambda p: ["dwp.bench_ensemble", "--models",
                            str(config.MODELS_DIR / "ens_*.pkl")],
    },
    "learning_curve": {
        "title": "Кривая обучения",
        "about": "Сколько точности стоит объём выгрузки: 5 долей пула x 3 сида, "
                 "15 обучений, около пятнадцати минут. Отвечает на вопрос, есть "
                 "ли смысл качать ещё матчи.",
        "params": {},
        "build": lambda p: ["dwp.learning_curve"],
    },
    "forecast_build": {
        "title": "Собрать коридор неопределённости",
        "about": "Распределение будущей оценки по истории: как часто лидер "
                 "меняется, как часто матч успевает кончиться. Это то, что "
                 "панель показывает в блоке «перспектива».",
        "params": {},
        "build": lambda p: ["dwp.forecast", "--build"],
    },
    "forecast_check": {
        "title": "Проверить экстраполяцию темпа",
        "about": "Бьёт ли сценарий «если темп сохранится» наивное «оценка не "
                 "изменится». Результат может оказаться отрицательным — так "
                 "тоже бывает, и это записывается.",
        "params": {},
        "build": lambda p: ["dwp.forecast", "--check"],
    },
    "livecheck": {
        "title": "Калибровка лайва",
        "about": "Добрать исходы записанных матчей и посчитать, насколько врало "
                 "число на экране. Единственный замер, который меряет реальный "
                 "боевой путь, а не выгрузку.",
        "params": {},
        "build": lambda p: ["dwp.livecheck"],
    },
    "selftest": {
        "title": "Прогнать все проверки",
        "about": "Гигиена: реплей лайв-пути против офлайн-пути, фикстуры, "
                 "геометрия карты, сборки, лента событий. Всё должно дать код 0.",
        "params": {},
        "steps": lambda p: [
            ["dwp.test_live_replay", "--n", "40", "--model",
             str(config.MODELS_DIR / "ens_*.pkl")],
            ["dwp.test_live_fixture"],
            ["dwp.test_collect_fake"],
            ["dwp.test_map_geometry"],
            ["dwp.test_builds"],
            ["dwp.test_killfeed"],
            ["dwp.test_forecast"],
            ["dwp.test_verdict"],
            ["dwp.test_web_snapshot"],
            ["dwp.livecheck", "--offline"]],
    },
    "bench_models": {
        "title": "Сравнить все модели",
        "about": "Все артефакты из models/ на слепом холдауте за один "
                 "проход: log loss, Brier, ECE, доля угаданных победителей "
                 "и парный бутстрап против боевого ансамбля. Пишет ОДНО "
                 "вскрытие холдаута в реестр и кэш вероятностей для "
                 "подбора вердикта.",
        "params": {},
        "build": lambda p: ["dwp.bench_models"],
        "after": "дальше: «Подобрать вердикт» — он читает этот кэш",
    },
    "verdict_tune": {
        "title": "Подобрать вердикт",
        "about": "Правило устойчивого вердикта: подбор на одной половине "
                 "слепого холдаута, замер и таблица попаданий — на другой. "
                 "Пишет data/verdict.json, который читает панель. Требует "
                 "кэша от «Сравнить все модели».",
        "params": {},
        "build": lambda p: ["dwp.verdict", "--tune", "--frontier"],
        "after": "перезапустите панель, чтобы она перечитала правило",
    },
}


class Job:
    def __init__(self, jid: int, key: str, steps: list[list[str]],
                 params: dict) -> None:
        self.id = jid
        self.key = key
        self.title = SPECS[key]["title"]
        self.steps = steps
        self.params = params
        self.status = "running"
        self.started = time.time()
        self.finished: float | None = None
        self.code: int | None = None
        self.step = 0
        self.lines: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.proc: subprocess.Popen | None = None
        self.lock = threading.Lock()

    def note(self, text: str) -> None:
        with self.lock:
            self.lines.append(text)

    def snapshot(self, tail: int = 200) -> dict:
        with self.lock:
            lines = list(self.lines)[-tail:]
        return {
            "id": self.id, "key": self.key, "title": self.title,
            "status": self.status, "code": self.code,
            "started": self.started, "finished": self.finished,
            "elapsed": (self.finished or time.time()) - self.started,
            "step": self.step, "steps": len(self.steps),
            "params": self.params, "log": lines,
            "cmd": " ".join(self.steps[min(self.step, len(self.steps) - 1)]),
            "after": SPECS[self.key].get("after", ""),
        }


class Runner:
    """Одна задача за раз. История последних задач — в памяти процесса."""

    def __init__(self, keep: int = 12) -> None:
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.current: Job | None = None
        self.keep = keep
        self._next = 1

    # --- запуск --------------------------------------------------------

    def validate(self, key: str, raw: dict) -> dict:
        spec = SPECS.get(key)
        if spec is None:
            raise ValueError(f"неизвестная задача {key!r}")
        out = {}
        for name, rule in (spec.get("params") or {}).items():
            v = raw.get(name, rule["default"])
            try:
                v = int(v)
            except (TypeError, ValueError):
                raise ValueError(
                    f"параметр {name} должен быть целым числом, получено {v!r}")
            if not (rule["min"] <= v <= rule["max"]):
                raise ValueError(
                    f"параметр {name} = {v} вне диапазона "
                    f"{rule['min']}..{rule['max']}")
            out[name] = v
        return out

    def start(self, key: str, raw: dict | None = None) -> Job:
        params = self.validate(key, raw or {})
        spec = SPECS[key]
        steps = (spec["steps"](params) if "steps" in spec
                 else [spec["build"](params)])
        with self.lock:
            if self.current is not None and self.current.status == "running":
                raise RuntimeError(
                    f"уже идёт задача «{self.current.title}». Одновременно "
                    f"выполняется одна: два обучения на одной машине только "
                    f"отнимают друг у друга процессор.")
            job = Job(self._next, key, steps, params)
            self._next += 1
            self.jobs.append(job)
            del self.jobs[: max(0, len(self.jobs) - self.keep)]
            self.current = job
        threading.Thread(target=self._run, args=(job,), daemon=True).start()
        return job

    def _run(self, job: Job) -> None:
        env = dict(os.environ)
        # Дочерний процесс печатает по-русски, а кодовая страница консоли
        # Windows этого не обещает. Без явного UTF-8 лог приходит кашей.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUNBUFFERED"] = "1"
        code = 0
        for i, step in enumerate(job.steps):
            job.step = i
            job.note(f"$ python -m {' '.join(step)}")
            try:
                proc = subprocess.Popen(
                    [sys.executable, "-m", *step],
                    cwd=str(config.ROOT), env=env,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            except OSError as e:
                job.note(f"НЕ ЗАПУСТИЛОСЬ: {e}")
                code = 1
                break
            job.proc = proc
            assert proc.stdout is not None
            for line in proc.stdout:
                job.note(line.rstrip("\n"))
            proc.wait()
            code = proc.returncode
            job.note(f"[код возврата {code}]")
            if code != 0:
                break
        job.code = code
        job.status = "done" if code == 0 else "failed"
        job.finished = time.time()
        job.proc = None
        with self.lock:
            if self.current is job:
                self.current = None

    def stop(self) -> bool:
        with self.lock:
            job = self.current
        if job is None or job.status != "running":
            return False
        proc = job.proc
        if proc is not None:
            proc.terminate()
        job.note("[прервано из панели]")
        return True

    # --- наружу --------------------------------------------------------

    def get(self, jid: int | None = None, tail: int = 200) -> dict | None:
        with self.lock:
            if jid is None:
                job = self.current or (self.jobs[-1] if self.jobs else None)
            else:
                job = next((j for j in self.jobs if j.id == jid), None)
        return job.snapshot(tail) if job is not None else None

    def history(self) -> list[dict]:
        with self.lock:
            jobs = list(self.jobs)
        return [j.snapshot(tail=0) for j in reversed(jobs)]

    def catalogue(self) -> list[dict]:
        return [{
            "key": k, "title": s["title"], "about": s["about"],
            "params": [{"name": n, **r} for n, r in (s.get("params") or {}).items()],
            "steps": len(s["steps"]({n: r["default"]
                                     for n, r in (s.get("params") or {}).items()}))
                     if "steps" in s else 1,
        } for k, s in SPECS.items()]


def data_status() -> dict:
    """Что сейчас есть на диске. Для страницы «состояние»."""
    from . import holdout as HO

    files = sorted(p for p in config.RAW_MATCHES_DIR.glob("*.json")
                   if p.stem.isdigit())
    hold = sum(1 for p in files if HO.is_holdout(p.stem))
    models = []
    for p in sorted(config.MODELS_DIR.glob("*.pkl")):
        models.append({"name": p.name, "size": p.stat().st_size,
                       "mtime": p.stat().st_mtime})
    logs = sorted(config.LIVE_LOG_DIR.glob("*.csv"))
    return {
        "matches": len(files),
        "holdout": hold,
        "trainable": len(files) - hold,
        "models": models,
        "ensemble": [p.name for p in sorted(config.MODELS_DIR.glob("ens_*.pkl"))],
        "live_logs": len(logs),
        "resolved": len(list(config.LIVE_RESOLVED_DIR.glob("*.json"))),
        "reveals": len(HO.registry_rows()),
        "horizon_table": (config.DATA_DIR / "horizon.json").exists(),
        "comeback_table": (config.DATA_DIR / "comeback.json").exists(),
        "learning_curve": (config.DATA_DIR / "learning_curve.json").exists(),
        "steam_key": bool(os.environ.get(config.STEAM_API_KEY_ENV)),
        "stratz_token": bool(os.environ.get(config.STRATZ_TOKEN_ENV)),
        "root": str(config.ROOT),
    }
