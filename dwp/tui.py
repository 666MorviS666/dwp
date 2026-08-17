"""Интерактивное консольное меню: стрелки, Enter, Backspace.

Запуск: `python -m dwp`

Ввод клавиш читается посимвольно, без Enter: на Windows через msvcrt,
на Unix через termios. Если ввод не с терминала (перенаправлен из файла
или конвейера), меню автоматически переключается на нумерованный режим —
иначе оно бы зависло, а именно так его и удобно тестировать.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from . import config
from . import live as L

UP, DOWN, ENTER, BACK, QUIT, OTHER = "UP", "DOWN", "ENTER", "BACK", "QUIT", "OTHER"


# --- Чтение клавиш ------------------------------------------------------

def _read_key_windows() -> str:
    import msvcrt
    ch = msvcrt.getch()
    if ch in (b"\x00", b"\xe0"):          # префикс специальной клавиши
        code = msvcrt.getch()
        return {b"H": UP, b"P": DOWN}.get(code, OTHER)
    if ch in (b"\r", b"\n"):
        return ENTER
    if ch == b"\x08":                     # Backspace
        return BACK
    if ch == b"\x1b":                     # Esc
        return BACK
    if ch in (b"q", b"Q", b"\x03"):       # q или Ctrl+C
        return QUIT
    if ch in (b"w", b"W"):
        return UP
    if ch in (b"s", b"S"):
        return DOWN
    return OTHER


def _read_key_unix() -> str:
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            nxt = sys.stdin.read(1)
            if nxt != "[":
                return BACK               # одиночный Esc
            return {"A": UP, "B": DOWN}.get(sys.stdin.read(1), OTHER)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    if ch in ("\r", "\n"):
        return ENTER
    if ch == "\x7f":
        return BACK
    if ch in ("q", "Q", "\x03"):
        return QUIT
    if ch in ("w", "W"):
        return UP
    if ch in ("s", "S"):
        return DOWN
    return OTHER


def read_key() -> str:
    return _read_key_windows() if os.name == "nt" else _read_key_unix()


def _tty() -> bool:
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def clear() -> None:
    if _tty():
        os.system("cls" if os.name == "nt" else "clear")


def _unicode_ok() -> bool:
    enc = getattr(sys.stdout, "encoding", None) or "ascii"
    try:
        "\u2192\u2502\u2500".encode(enc)
        return True
    except (UnicodeEncodeError, LookupError):
        return False


# --- Меню ---------------------------------------------------------------

def select(title: str, options: list[str], hint: str = "",
           get_key=None) -> int | None:
    """Возвращает индекс выбранного пункта или None, если пользователь вышел."""
    if not options:
        return None
    if not _tty() and get_key is None:
        # Неинтерактивный ввод: нумерованный режим. Без него меню
        # зависло бы на чтении клавиши из закрытого потока.
        print(f"\n{title}")
        for i, o in enumerate(options, 1):
            print(f"  {i}. {o}")
        raw = input("номер (пусто = назад): ").strip()
        if not raw.isdigit():
            return None
        n = int(raw)
        return n - 1 if 1 <= n <= len(options) else None

    get_key = get_key or read_key
    cur = 0
    arrow = "\u2192" if _unicode_ok() else ">"
    while True:
        clear()
        print(f"\n  {title}\n")
        for i, o in enumerate(options):
            mark = f"{arrow} " if i == cur else "  "
            print(f"  {mark}{o}")
        print(f"\n  {hint or 'стрелки — выбор, Enter — ок, Backspace — назад, q — выход'}")
        k = get_key()
        if k == UP:
            cur = (cur - 1) % len(options)
        elif k == DOWN:
            cur = (cur + 1) % len(options)
        elif k == ENTER:
            return cur
        elif k in (BACK, QUIT):
            return None


def prompt(text: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    try:
        val = input(f"  {text}{suffix}: ").strip()
    except EOFError:
        return default
    return val or default


def pause() -> None:
    if _tty():
        print("\n  Нажмите любую клавишу...")
        read_key()


def list_models() -> list[Path]:
    return sorted(config.MODELS_DIR.glob("*.pkl"))


def pick_model(title: str = "Выберите модель", get_key=None) -> Path | None:
    models = list_models()
    if not models:
        print("\n  Моделей нет. Сначала обучите: пункт «Обучить модель».")
        pause()
        return None
    labels = []
    for m in models:
        size = m.stat().st_size / 1e6
        labels.append(f"{m.name:<34} {size:5.1f} МБ")
    i = select(title, labels, get_key=get_key)
    return models[i] if i is not None else None


# --- Запуск модулей -----------------------------------------------------

def run_module(name: str, argv: list[str]) -> int:
    """Вызывает main() модуля напрямую, чтобы вывод шёл в ту же консоль."""
    import importlib
    mod = importlib.import_module(f"dwp.{name}")
    print(f"\n  > python -m dwp.{name} {' '.join(argv)}\n")
    try:
        return int(mod.main(argv) or 0)
    except SystemExit as e:                # argparse при неверных аргументах
        return int(e.code or 0)
    except KeyboardInterrupt:
        print("\n  Прервано.")
        return 130


# --- Экраны -------------------------------------------------------------

def screen_collect(get_key=None) -> None:
    n = prompt("Сколько матчей скачать", "3000")
    if not n.isdigit():
        print("  Нужно число.")
        pause()
        return
    print("\n  OpenDota ограничивает частоту запросов. На 429 скрипт ждёт и")
    print("  продолжает сам — прерывать не надо. Можно закрыть в любой момент,")
    print("  повторный запуск продолжит с того же места.\n")
    run_module("collect", ["--n", n])
    pause()


def screen_check(get_key=None) -> None:
    i = select("Какие данные проверить", ["Реальные (data/matches)",
                                          "Синтетика (data/synthetic)"],
               get_key=get_key)
    if i is None:
        return
    run_module("check_schema", ["--source", "real" if i == 0 else "synthetic"])
    pause()


# --extra-features входит в оба пресета: прирост измерен парным
# бутстрапом, интервал [-0.0074, -0.0018], ноль не накрыт. Выбор между
# «с ними» и «без них» пользователю предлагать не за чем.
TRAIN_PRESETS = [
    ("Для лайва — смотреть идущие матчи", ["--live-features", "--extra-features"]),
    ("Полная — разбирать прошедшие матчи", ["--extra-features"]),
]


def screen_train(get_key=None) -> None:
    i = select("Источник данных", ["Реальные матчи", "Синтетика"], get_key=get_key)
    if i is None:
        return
    source = "real" if i == 0 else "synthetic"
    j = select("Какую модель обучить", [p[0] for p in TRAIN_PRESETS], get_key=get_key)
    if j is None:
        return
    k = select("Калибратор драфт-модели",
               ["sigmoid (обычно лучше при малом объёме)", "isotonic", "none", "auto"],
               get_key=get_key)
    if k is None:
        return
    draft = ["sigmoid", "isotonic", "none", "auto"][k]
    m = select("Критерий выбора калибратора состояния",
               ["ece — честность числа на экране", "logloss"], get_key=get_key)
    if m is None:
        return
    argv = ["--source", source] + TRAIN_PRESETS[j][1] + [
        "--draft-calib", draft, "--calib-criterion", "ece" if m == 0 else "logloss"]
    name = prompt("Имя файла модели (пусто = по умолчанию)", "")
    if name:
        argv += ["--out", str(config.MODELS_DIR / name)]
    clear()
    run_module("train", argv)
    pause()


def screen_analyze(get_key=None) -> None:
    model = pick_model("Модель для разбора", get_key=get_key)
    if model is None:
        return
    i = select("Что разобрать", ["Случайные матчи из теста", "Конкретный match_id"],
               get_key=get_key)
    if i is None:
        return
    src = "real" if "real" in model.name else "synthetic"
    if i == 0:
        n = prompt("Сколько матчей", "3")
        argv = ["--source", src, "--model", str(model), "--random",
                n if n.isdigit() else "3"]
    else:
        mid = prompt("match_id", "")
        if not mid.isdigit():
            print("  Нужно число.")
            pause()
            return
        argv = ["--source", src, "--model", str(model), "--match-id", mid]
    clear()
    run_module("analyze", argv)
    pause()


def screen_compare(get_key=None) -> None:
    if len(list_models()) < 2:
        print("\n  Нужны хотя бы две модели. Обучите вторую с другим --out.")
        pause()
        return
    a = pick_model("Модель A (базовая)", get_key=get_key)
    if a is None:
        return
    b = pick_model("Модель B (с изменением)", get_key=get_key)
    if b is None:
        return
    if a == b:
        print("\n  Это одна и та же модель.")
        pause()
        return
    src = "real" if "real" in a.name else "synthetic"
    clear()
    run_module("compare", [str(a), str(b), "--source", src])
    pause()


def live_models() -> list[Path]:
    """Только модели, обученные с --live-features.

    Остальные опираются на признаки, которых в GetRealtimeStats нет
    (Рошан, Терзатель, опыт, выкупы), и в бою всегда получают там NaN.
    Раз выбор всё равно неправильный — не предлагать его вовсе.
    """
    import pickle
    out = []
    for m in list_models():
        try:
            with m.open("rb") as fh:
                art = pickle.load(fh)
        except Exception:
            continue
        if art.get("live_features"):
            out.append(m)
    return out


def screen_live(get_key=None) -> None:
    models = live_models()
    if not models:
        print("\n  Нет ни одной модели, обученной для лайва.")
        print("  Обучите: пункт «Обучить модель», режим «для лайва».")
        print("  Или командой:")
        print("    python -m dwp.train --source real --live-features "
              "--extra-features --draft-calib sigmoid --calib-criterion ece "
              "--seed 7 --out models\\live.pkl")
        pause()
        return

    print("\n  Запрашиваю список текущих игр...")
    try:
        games = L.top_live_games(L.get_key())
    except Exception as e:
        print(f"\n  Не удалось получить список: {type(e).__name__}: {e}")
        pause()
        return
    if not games:
        print("\n  Сейчас GetTopLiveGame не отдаёт ни одной игры. Это бывает "
              "между матчами.")
        pause()
        return

    rows = []
    for g in games:
        r, d = L.game_team_names(g)
        rows.append((str(g.get("server_steam_id", "?")), f"{r} vs {d}",
                     int(g.get("spectators") or 0),
                     int(g.get("game_time") or 0) // 60))
    rows.sort(key=lambda t: -t[2])
    labels = [f"{t[1][:38]:<39}{t[3]:>3} мин {t[2]:>7} зрителей" for t in rows]
    i = select("Какой матч смотреть", labels, get_key=get_key)
    if i is None:
        return
    sid = rows[i][0]

    if len(models) == 1:
        model = models[0]
    else:
        j = select("Модель", [m.name for m in models], get_key=get_key)
        if j is None:
            return
        model = models[j]

    clear()
    run_module("live", ["--watch", "--server-steam-id", sid,
                        "--model", str(model), "--interval", "15"])
    pause()


def screen_synthetic(get_key=None) -> None:
    n = prompt("Сколько синтетических матчей", "4000")
    clear()
    run_module("synthetic", ["--n", n if n.isdigit() else "4000", "--clean"])
    pause()


def screen_web(get_key=None) -> None:
    """Веб-панель. Она же главный интерфейс проекта, и до аудита её в меню
    не было вовсе — при том, что в README она описана первой."""
    print("\n  Панель откроется на http://127.0.0.1:8765/ — меню, список")
    print("  идущих матчей, карта, лента событий, перспектива, сборки,")
    print("  а также сбор данных, обучение и все замеры прямо из браузера.")
    print("  Ctrl+C закрывает сервер и возвращает сюда.\n")
    ens = sorted(config.MODELS_DIR.glob(L.ENSEMBLE_GLOB))
    opts = []
    if len(ens) >= 2:
        opts.append((f"Ансамбль из {len(ens)} моделей (умолчание, точнее)", None))
    opts.append(("Выбрать одну модель", "pick"))
    i = select("Чем считать", [o[0] for o in opts], get_key=get_key) if opts else None
    if i is None:
        return
    if opts[i][1] is None:
        clear()
        run_module("web", [])
        pause()
        return
    model = pick_model("Модель для панели", get_key=get_key)
    if model is None:
        return
    clear()
    run_module("web", ["--model", str(model)])
    pause()


def screen_ensemble(get_key=None) -> None:
    """Ансамбль по сидам и замеры вокруг него.

    Отдельным пунктом, потому что это единственное улучшение точности,
    которое замерено и не требует ни новых данных, ни новых признаков:
    на слепом холдауте из 1167 матчей ансамбль из пяти сидов дал log loss
    0.5233 против 0.5285 у средней одиночной модели и ECE 0.0082 против
    0.0155-0.0315. Для лайва важнее второе.
    """
    ens = sorted(config.MODELS_DIR.glob(L.ENSEMBLE_GLOB))
    print(f"\n  Сейчас в ансамбле моделей: {len(ens)}"
          + (f" ({', '.join(p.stem for p in ens)})" if ens else ""))
    i = select("Что сделать", [
        "Обучить набор из пяти сидов (около пяти минут)",
        "Замерить ансамбль на слепом холдауте",
        "Кривая обучения: нужны ли ещё матчи (около 15 минут)",
        "Собрать коридор неопределённости для блока «перспектива»",
        "Проверить экстраполяцию темпа",
    ], get_key=get_key)
    if i is None:
        return
    clear()
    if i == 0:
        for seed in (3, 5, 7, 11, 13):
            code = run_module("train", [
                "--source", "real", "--live-features", "--extra-features",
                "--exact-features", "--gold-norm", "add", "--seed", str(seed),
                "--out", str(config.MODELS_DIR / f"ens_{seed:02d}.pkl")])
            if code != 0:
                print(f"\n  Обучение с сидом {seed} завершилось кодом {code} — "
                      f"дальше не иду.")
                break
    elif i == 1:
        run_module("bench_ensemble",
                   ["--models", str(config.MODELS_DIR / L.ENSEMBLE_GLOB)])
    elif i == 2:
        run_module("learning_curve", [])
    elif i == 3:
        run_module("forecast", ["--build"])
    else:
        run_module("forecast", ["--check"])
    pause()


def screen_blind(get_key=None) -> None:
    """Слепой тест на холдауте: запечатать оценки, потом вскрыть."""
    model = pick_model("Модель для слепого теста", get_key=get_key)
    if model is None:
        return
    i = select("Что сделать", [
        "Запечатать оценки на слепом холдауте (исход не читается)",
        "Вскрыть последний конверт и посчитать",
        "Сыграть самому против модели",
        "Посчитать догадки из «слепого режима» панели",
        "Показать журнал вскрытий холдаута",
    ], get_key=get_key)
    if i is None:
        return
    clear()
    if i == 0:
        n = prompt("Сколько матчей запечатать", "200")
        run_module("blindtest", ["--seal", "--holdout", "--n",
                                 n if n.isdigit() else "200",
                                 "--model", str(model)])
    elif i == 1:
        run_module("blindtest", ["--reveal", "--holdout", "--worst", "10",
                                 "--model", str(model)])
    elif i == 2:
        n = prompt("Сколько позиций показать", "10")
        run_module("blindtest", ["--quiz", "--holdout", "--n",
                                 n if n.isdigit() else "10",
                                 "--model", str(model)])
    elif i == 3:
        run_module("blindtest", ["--guesses"])
    else:
        run_module("holdout", [])
    pause()


def screen_builds(get_key=None) -> None:
    """Сборки и шансы отыграться."""
    i = select("Что показать", [
        "Таблица шансов отыграться (уже посчитанная)",
        "Пересчитать таблицу по всей выгрузке (долго, ~25 минут)",
        "Разобрать сборки из сохранённого ответа лайва",
        "Есть ли урон в лайв-ответе",
    ], get_key=get_key)
    if i is None:
        return
    clear()
    if i == 0:
        run_module("builds", [])
    elif i == 1:
        run_module("builds", ["--table"])
    elif i == 2:
        run_module("builds", ["--from-file",
                              str(config.DATA_DIR / "live_dump_late.json")])
    else:
        run_module("damage", ["--check"])
    pause()


def screen_service(get_key=None) -> None:
    """Инструменты разработки. В основном меню им не место: во время
    матча они не нужны, а синтетика вообще не про предсказание — она
    нужна, чтобы проверить, что пайплайн исправен после правок кода."""
    items = [("Проверить схему данных", screen_check),
             ("Сгенерировать синтетику (проверка пайплайна)", screen_synthetic),
             ("Сборки, шансы отыграться, урон", screen_builds)]
    i = select("Служебное", [t[0] for t in items], get_key=get_key)
    if i is not None:
        items[i][1](get_key=get_key)


MENU = [
    ("Открыть веб-панель", screen_web),
    ("Смотреть матч в консоли", screen_live),
    ("Разобрать завершённый матч", screen_analyze),
    ("Ансамбль, кривая обучения, коридор", screen_ensemble),
    ("Слепой тест и холдаут", screen_blind),
    ("Скачать матчи с OpenDota", screen_collect),
    ("Обучить модель", screen_train),
    ("Сравнить две модели", screen_compare),
    ("Служебное", screen_service),
]


def _status() -> str:
    n_real = len(list(config.RAW_MATCHES_DIR.glob("*.json")))
    n_synth = len(list(config.SYNTH_MATCHES_DIR.glob("*.json")))
    n_models = len(list_models())
    n_ens = len(list(config.MODELS_DIR.glob(L.ENSEMBLE_GLOB)))
    key = "есть" if os.environ.get(config.STEAM_API_KEY_ENV) else "НЕТ"
    return (f"матчей: {n_real} реальных, {n_synth} синтетических | "
            f"моделей: {n_models}"
            + (f" (ансамбль из {n_ens})" if n_ens >= 2 else "")
            + f" | ключ Steam: {key}")


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    get_key = None
    if argv:                               # для тестов: последовательность клавиш
        keys = iter(argv)

        def get_key():
            return next(keys, QUIT)

    while True:
        i = select(f"dwp - вероятность победы в Dota 2\n  {_status()}",
                   [m[0] for m in MENU] + ["Выход"], get_key=get_key)
        if i is None or i == len(MENU):
            print("\n  Пока.")
            return 0
        try:
            MENU[i][1](get_key=get_key)
        except KeyboardInterrupt:
            print("\n  Прервано.")
        except Exception as e:             # меню не должно падать из-за экрана
            print(f"\n  ОШИБКА в пункте «{MENU[i][0]}»: {type(e).__name__}: {e}")
            pause()


if __name__ == "__main__":
    sys.exit(main())
