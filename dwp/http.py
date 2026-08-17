"""Общий GET с ретраями для обоих источников: OpenDota и Steam Web API.

Раньше полный клиент — сессия, backoff, разбор `Retry-After`, 429 и 5xx — был
только в `collect.py`, а `live._get` делал один запрос без единой попытки
повтора. Из-за этого `--watch` умирал от одного потерянного пакета, хотя
`config.HTTP_RETRIES` в проекте уже был; он просто не использовался в лайве.

Политика повторов здесь общая, а тексты ошибок — нет: у этого проекта каждая
ошибка обязана говорить, что делать дальше, а совет для «не отвечает OpenDota
посреди выгрузки» и для «пропал ответ посреди матча» разный. Поэтому вызывающий
передаёт конструктор своего исключения (`err`) и разбор неповторяемых кодов
(`fatal`).

Что считается временным (повторяем): обрыв соединения, таймаут, 429, 5xx.
Что временным не считается: TLS, а также всё, что вызывающий разобрал в `fatal`.
"""

from __future__ import annotations

import time
from typing import Callable

import requests

from . import config


def make_session(user_agent: str = "dwp/0.1 (dota win probability)") -> requests.Session:
    """Сессия с переиспользованием соединения.

    В лайве это не косметика: без сессии каждый опрос раз в 15 секунд заново
    поднимает TCP и TLS к api.steampowered.com.
    """
    s = requests.Session()
    s.headers["User-Agent"] = user_agent
    return s


def get_json(session: requests.Session, url: str, params: dict, *,
             err: Callable[[str], Exception],
             fatal: Callable[[int, requests.Response], Exception | None] | None = None,
             give_up_hint: str = "",
             timeout: float = config.HTTP_TIMEOUT,
             retries: int = config.HTTP_RETRIES,
             backoff: float = config.HTTP_BACKOFF,
             throttle: Callable[[], None] | None = None,
             verbose: bool = False) -> object:
    """GET с повторами. Возвращает разобранный JSON или бросает `err(...)`."""
    last: str = ""

    def pause(sec: float) -> None:
        # После ПОСЛЕДНЕЙ попытки ждать нечего: ретраев больше не будет, а
        # вызывающий всё это время сидит и ждёт исключения. На молчащем
        # Steam эта пауза добавляла к отказу лишние секунды на ровном месте.
        if attempt < retries - 1:
            time.sleep(sec)

    for attempt in range(retries):
        if throttle is not None:
            throttle()
        try:
            r = session.get(url, params=params, timeout=timeout)
        except requests.exceptions.SSLError as e:
            raise err(f"Ошибка TLS при обращении к {url}: {e}\n"
                      f"Что делать: проверьте системные сертификаты или прокси, "
                      f"перехватывающий HTTPS.") from e
        except requests.exceptions.ConnectionError as e:
            last = f"нет соединения ({type(e).__name__})"
            if verbose:
                print(f"  [сеть] {last}, попытка {attempt + 1}/{retries}")
            pause(backoff ** attempt)
            continue
        except requests.exceptions.Timeout as e:
            last = f"таймаут {timeout} с ({type(e).__name__})"
            pause(backoff ** attempt)
            continue

        if r.status_code == 200:
            try:
                return r.json()
            except ValueError as e:
                raise err(
                    f"{url} вернул 200, но тело не JSON "
                    f"(первые 200 байт: {r.text[:200]!r}).\n"
                    f"Что делать: обычно это страница-заглушка прокси или "
                    f"капча. Проверьте доступ к этому хосту из вашей сети."
                ) from e
        if r.status_code == 429:
            wait = _retry_after(r)
            last = f"429, ждём {wait:.0f} с"
            if verbose:
                print(f"  [лимит] {last}")
            pause(wait)
            continue
        if 500 <= r.status_code < 600:
            last = f"{r.status_code} от сервера"
            pause(backoff ** attempt)
            continue
        if fatal is not None:
            exc = fatal(r.status_code, r)
            if exc is not None:
                raise exc
        raise err(f"Неожиданный код {r.status_code} от {url}: {r.text[:200]!r}")

    raise err(f"Не удалось получить {url} за {retries} попыток. "
              f"Последняя ошибка: {last}"
              + (f"\nЧто делать: {give_up_hint}" if give_up_hint else ""))


def _retry_after(r: requests.Response) -> float:
    """Секунды из заголовка Retry-After. Мусор в заголовке не должен ронять."""
    raw = r.headers.get("Retry-After", "")
    try:
        return max(0.0, float(raw))
    except (TypeError, ValueError):
        return 30.0
