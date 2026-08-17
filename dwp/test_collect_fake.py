"""Проверка collect.py без сети: подменяется только транспорт.

Подменяется OpenDotaClient.get, то есть вся логика пагинации, фильтра
нераспарсенных матчей, кэша и курсора выполняется настоящая. Сеть в этой
песочнице к api.opendota.com закрыта, поэтому иначе эту часть не проверить.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dwp import collect, config, synthetic  # noqa: E402
import numpy as np  # noqa: E402


def build_fake_api(n_pro: int = 60, unparsed_every: int = 3):
    """Отдаёт /heroes, /proMatches с пагинацией и /matches/{id}."""
    rng = np.random.default_rng(1)
    heroes = synthetic.make_heroes(30)
    hero_ids = np.array([h["id"] for h in heroes])
    hs = rng.normal(0, 0.2, len(heroes))
    ts = rng.normal(0, 0.4, 8)
    ids = sorted((9_000_000_000 + i * 11 for i in range(n_pro)), reverse=True)
    calls = {"n": 0}

    def fake_get(self, path, params=None):
        calls["n"] += 1
        if path == "/heroes":
            return heroes
        if path == "/proMatches":
            lt = (params or {}).get("less_than_match_id")
            pool = [i for i in ids if lt is None or i < lt]
            return [{"match_id": i, "radiant_win": True} for i in pool[:20]]
        if path.startswith("/matches/"):
            mid = int(path.rsplit("/", 1)[1])
            m = synthetic.generate_match(rng, mid, hero_ids, hs, ts, 1_700_000_000)
            if mid % unparsed_every == 0:      # имитация нераспарсенного матча
                m["radiant_gold_adv"] = None
                m["version"] = None
            return m
        raise AssertionError(f"неожиданный путь {path}")

    return fake_get, calls


def main() -> int:
    out = config.DATA_DIR / "_test_collect"
    if out.exists():
        shutil.rmtree(out)
    if config.COLLECT_STATE_PATH.exists():
        config.COLLECT_STATE_PATH.unlink()

    fake_get, calls = build_fake_api()
    real_get = collect.OpenDotaClient.get
    collect.OpenDotaClient.get = fake_get
    collect.OpenDotaClient._throttle = lambda self: None   # тест не должен спать
    try:
        n1 = collect.collect(target=12, out_dir=out, restart=True)
        assert n1 == 12, f"ожидалось 12 матчей, получено {n1}"
        st1 = json.loads(config.COLLECT_STATE_PATH.read_text())
        assert st1["unparsed"] > 0, "фильтр нераспарсенных матчей не сработал ни разу"
        calls_after_first = calls["n"]

        # Возобновление: повторный запуск с той же целью не должен качать заново.
        n2 = collect.collect(target=12, out_dir=out)
        assert n2 == 12
        assert calls["n"] == calls_after_first, (
            f"повторный запуск сделал {calls['n'] - calls_after_first} лишних запросов")

        # Дозагрузка: цель больше — качает только недостающее.
        n3 = collect.collect(target=20, out_dir=out)
        assert n3 == 20, f"ожидалось 20, получено {n3}"

        files = sorted(out.glob("*.json"))
        assert len(files) == 20
        for f in files:
            m = json.loads(f.read_text())
            assert collect.is_parsed(m), f"в кэш попал нераспарсенный матч {f.name}"
        st = json.loads(config.COLLECT_STATE_PATH.read_text())
        print("OK: скачано 12, возобновлено без лишних запросов, дозагружено до 20.")
        print(f"    просмотрено id: {st['seen']}, отброшено нераспарсенных: {st['unparsed']}")
        print(f"    курсор пагинации: {st['cursor']}")
        return 0
    finally:
        collect.OpenDotaClient.get = real_get
        shutil.rmtree(out, ignore_errors=True)
        if config.COLLECT_STATE_PATH.exists():
            config.COLLECT_STATE_PATH.unlink()


if __name__ == "__main__":
    sys.exit(main())
