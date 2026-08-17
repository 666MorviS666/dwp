"""Загрузка про-матчей с OpenDota. Возобновляемая.

Возобновляемость обеспечивается двумя вещами: каждый матч лежит отдельным
файлом (есть файл — не качаем), и курсор пагинации хранится в
collect_state.json. Прервать можно в любой момент.

Все сетевые ошибки обрабатываются явно и с указанием, что делать дальше:
молчаливый except здесь означал бы датасет с дырами, о которых никто не
узнает, а по метрикам это не видно.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

from . import config
from . import http as H


class ApiError(RuntimeError):
    """Сетевая или протокольная ошибка с готовым советом пользователю."""


class OpenDotaClient:
    def __init__(self, rpm: int = config.OPENDOTA_RPM, verbose: bool = True):
        self.session = H.make_session()
        self.min_interval = 60.0 / max(rpm, 1)
        self._last = 0.0
        self.verbose = verbose
        self.api_key = os.environ.get(config.OPENDOTA_API_KEY_ENV)

    def _throttle(self) -> None:
        wait = self.min_interval - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        self._last = time.time()

    @staticmethod
    def _fatal(status: int, r: requests.Response) -> Exception | None:
        """Коды, которые повторять бессмысленно."""
        if status == 404:
            return ApiError(f"404 для {r.url}: ресурса нет.")
        if status == 403:
            deny = r.headers.get("x-deny-reason")
            return ApiError(
                f"403 для {r.url}"
                + (f" (x-deny-reason: {deny})" if deny else "")
                + "\nЧто делать: доступ запрещён на уровне сети или прокси. "
                  "Если вы в песочнице с белым списком доменов, добавьте "
                  "api.opendota.com в разрешённые.")
        return None

    def get(self, path: str, params: dict | None = None) -> object:
        # Политика повторов общая с лайвом и живёт в dwp.http; здесь остаётся
        # то, что специфично для OpenDota: троттлинг, api_key и свои советы.
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        return H.get_json(
            self.session, f"{config.OPENDOTA_BASE}{path}", params,
            err=ApiError, fatal=self._fatal, throttle=self._throttle,
            verbose=self.verbose,
            give_up_hint="проверьте сеть, затем запустите ту же команду снова — "
                         "скачанное сохранено, загрузка продолжится с места "
                         "остановки.")


def _load_state() -> dict:
    if config.COLLECT_STATE_PATH.exists():
        try:
            return json.loads(config.COLLECT_STATE_PATH.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            print(f"[collect] {config.COLLECT_STATE_PATH} повреждён, начинаю с нуля")
    return {"cursor": None, "seen": 0, "saved": 0, "unparsed": 0}


def _save_state(st: dict) -> None:
    config.COLLECT_STATE_PATH.write_text(json.dumps(st), encoding="utf-8")


def fetch_heroes(client: OpenDotaClient) -> int:
    data = client.get("/heroes")
    if not isinstance(data, list) or not data or "id" not in data[0]:
        raise ApiError(f"/heroes вернул неожиданную структуру: {str(data)[:300]!r}")
    config.HEROES_PATH.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"Героев сохранено: {len(data)} -> {config.HEROES_PATH}")
    return len(data)


def is_parsed(match: dict) -> bool:
    """Нужны только распарсенные матчи: у них непустой radiant_gold_adv.
    Без него поминутных признаков не построить."""
    return bool(match.get("radiant_gold_adv"))


def collect(target: int, out_dir: Path, restart: bool = False) -> int:
    client = OpenDotaClient()
    if not config.HEROES_PATH.exists():
        fetch_heroes(client)

    st = {"cursor": None, "seen": 0, "saved": 0, "unparsed": 0} if restart else _load_state()
    out_dir.mkdir(parents=True, exist_ok=True)
    have = {int(p.stem) for p in out_dir.glob("*.json") if p.stem.isdigit()}
    print(f"Уже в кэше: {len(have)} матчей. Цель: {target} распарсенных.")

    saved_now = 0
    while len(have) + saved_now < target:
        params = {}
        if st.get("cursor"):
            params["less_than_match_id"] = st["cursor"]
        page = client.get("/proMatches", params)
        if not isinstance(page, list):
            raise ApiError(f"/proMatches вернул не список: {str(page)[:300]!r}")
        if not page:
            print("OpenDota больше не отдаёт про-матчи — история кончилась.")
            break

        ids = [int(m["match_id"]) for m in page if "match_id" in m]
        if not ids:
            raise ApiError(
                f"/proMatches вернул {len(page)} записей без match_id. "
                f"Схема изменилась, ключи первой записи: {sorted(page[0])[:15]}")
        st["cursor"] = min(ids)

        for mid in ids:
            st["seen"] += 1
            if mid in have:
                continue
            try:
                m = client.get(f"/matches/{mid}")
            except ApiError as e:
                print(f"  [пропуск] матч {mid}: {e}")
                continue
            if not isinstance(m, dict) or "match_id" not in m:
                print(f"  [пропуск] матч {mid}: ответ не похож на матч "
                      f"({str(m)[:120]!r})")
                continue
            if not is_parsed(m):
                st["unparsed"] += 1
                continue
            (out_dir / f"{mid}.json").write_text(json.dumps(m), encoding="utf-8")
            have.add(mid)
            saved_now += 1
            st["saved"] += 1
            if saved_now % 20 == 0:
                _save_state(st)
                print(f"  сохранено {len(have)}/{target} "
                      f"(просмотрено {st['seen']}, без парса {st['unparsed']})")
            if len(have) >= target:
                break
        _save_state(st)

    _save_state(st)
    print(f"Готово. В кэше {len(have)} распарсенных матчей в {out_dir}.")
    print(f"Просмотрено id: {st['seen']}, отброшено нераспарсенных: {st['unparsed']}.")
    if len(have) < target:
        print("ВНИМАНИЕ: цель не достигнута. Запустите команду снова — "
              "загрузка продолжится с курсора.")
    return len(have)



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
    ap = argparse.ArgumentParser(description="Загрузка про-матчей с OpenDota.")
    ap.add_argument("--n", type=int, default=2000, help="целевое число распарсенных матчей")
    ap.add_argument("--out", type=Path, default=config.RAW_MATCHES_DIR)
    ap.add_argument("--heroes", action="store_true", help="только обновить справочник героев")
    ap.add_argument("--restart", action="store_true", help="сбросить курсор пагинации")
    args = ap.parse_args(argv)

    try:
        if args.heroes:
            fetch_heroes(OpenDotaClient())
            return 0
        collect(args.n, args.out, restart=args.restart)
        return 0
    except ApiError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nПрервано. Скачанное сохранено, повторный запуск продолжит с курсора.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
