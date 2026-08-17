"""Справочник предметов и восстановление инвентаря.

Зачем он понадобился. В ответе `GetRealtimeStats` у каждого игрока есть
`items` — девять чисел, id предметов, −1 на пустом месте. Это единственный
кусок лайв-ответа, который до сих пор не использовался вообще, а он отвечает
на вопрос, который по золоту не виден: **во что превращён нетворс**. Двадцать
тысяч на керри с шестью слотами и двадцать тысяч, из которых восемь лежат в
кармане, — разные позиции, а `gold_adv` у них одинаковый.

Что тут есть:

* справочник `id -> имя, цена, качество, картинка` (скачивается один раз);
* разбор лайв-инвентаря: сколько слотов занято, на сколько золота предметов,
  какие из них крупные;
* восстановление инвентаря ПО МИНУТАМ из `purchase_log` завершённого матча —
  без него сборку нельзя измерить на истории, а значит нельзя и оценить
  шансы отстающих (см. `dwp.builds`).

Источник справочника — `odota/dotaconstants`, тот же репозиторий, откуда
берётся подложка карты. В проект не коммитится, качается в `data/items.json`.

ПОЧЕМУ ЦЕНА, А НЕ ПРОСТО СЧЁТ СЛОТОВ. Шесть слотов ветками и тангами — это
не «шесть слотов». Поэтому везде, где считается сборка, консьюмаблы
(`qual == "consumable"`) не считаются вовсе, а рядом со счётом слотов идёт
стоимость предметов.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from dataclasses import dataclass

from . import config

ITEMS_URL = ("https://raw.githubusercontent.com/odota/dotaconstants/master/"
             "build/items.json")
ITEMS_PATH = config.DATA_DIR / "items.json"
IMG_BASE = "https://cdn.cloudflare.steamstatic.com"

# Сколько записей в `players[].items` живого ответа.
# ПРОВЕРЕНО НА ЖИВЫХ СЛЕПКАХ: ровно девять у всех десяти игроков в обоих
# слепках. Первые шесть — инвентарь, последние три — рюкзак; так это
# разложено у Valve во всех известных схемах, и на слепках согласуется:
# крупные предметы стоят в первых шести, а в хвосте лежат консьюмаблы
# (famango, ветки). Нейтральный предмет сюда НЕ попадает — ни одного
# предмета с `tier` в слепках в этом массиве нет (проверяется в
# test_builds).
N_INVENTORY_SLOTS = 6
N_BACKPACK_SLOTS = 3
N_ITEM_SLOTS = N_INVENTORY_SLOTS + N_BACKPACK_SLOTS

# Что считать «крупным предметом». Порог, а не список имён: список пришлось бы
# править каждым патчем. 1400 — это цена фазовых сапог и ниже дешёвых
# артефактов; всё, что дороже, покупается осознанно под сборку, а не по дороге.
BIG_ITEM_COST = 1400

# Предметы, которые ПРОПАДАЮТ из инвентаря при использовании: слот они не
# занимают, а в журнале покупок остаются. Список не выдуман — он получен
# замером: при восстановлении инвентаря по purchase_log на 4000 игроках
# набор крупных предметов совпадал с концом матча в 54.9% случаев, а лишним
# чаще всего оказывался ровно `aghanims_shard` (1281 раз из 4000). После
# исключения этого списка совпадение стало 77.8%, а медиана расхождения по
# стоимости упала с +900 до +450 золота.
#
# Проверка со стороны данных: у игрока есть флаг `aghanims_shard`, и он
# согласуется с фактом покупки шарда в 88.8% случаев — остальное это шарды,
# выпавшие с Рошана (в журнале покупок их нет вовсе).
CONSUMED_ON_USE = frozenset({
    "aghanims_shard", "aghanims_shard_roshan",
    "ultimate_scepter_2", "ultimate_scepter_roshan",
    "moon_shard", "tome_of_knowledge", "infused_raindrop",
})


@dataclass(frozen=True)
class Item:
    id: int
    name: str                 # внутреннее имя: black_king_bar
    dname: str                # читаемое: Black King Bar
    cost: int | None          # None — цены в справочнике нет (нейтралы, аегис)
    qual: str | None
    components: tuple[str, ...]
    img: str                  # полный URL картинки
    tier: int | None          # есть только у нейтральных предметов

    @property
    def consumable(self) -> bool:
        return bool(self.qual and "consumable" in self.qual)

    @property
    def neutral(self) -> bool:
        return self.tier is not None

    @property
    def big(self) -> bool:
        return (not self.consumable and not self.neutral
                and (self.cost or 0) >= BIG_ITEM_COST)


class ItemBook:
    """Справочник. Пустой (`ok is False`), если файла нет и скачать не вышло."""

    def __init__(self, by_name: dict[str, Item]):
        self.by_name = by_name
        self.by_id = {it.id: it for it in by_name.values()}

    @property
    def ok(self) -> bool:
        return bool(self.by_id)

    def get(self, item_id: int | None) -> Item | None:
        if item_id is None or int(item_id) < 0:
            return None
        return self.by_id.get(int(item_id))

    def cost_of(self, item_id: int | None) -> int | None:
        """Цена или None. НЕ ноль: «предмет неизвестен» и «предмет бесплатный»
        различаются, и молчаливый ноль здесь занизил бы стоимость сборки."""
        it = self.get(item_id)
        if it is None:
            return None
        return it.cost


_BOOK: ItemBook | None = None


def load(download: bool = True, verbose: bool = True) -> ItemBook:
    """Справочник из data/items.json, при необходимости скачивая его.

    Один раз на процесс. Не скачался — возвращается пустой справочник, и
    вызывающий обязан это заметить: блоки по сборкам просто не рисуются, а не
    показывают нули.
    """
    global _BOOK
    if _BOOK is not None:
        return _BOOK
    raw = None
    if ITEMS_PATH.exists():
        try:
            raw = json.loads(ITEMS_PATH.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            if verbose:
                print(f"ВНИМАНИЕ: {ITEMS_PATH} не читается ({e}), качаю заново",
                      file=sys.stderr)
            raw = None
    if raw is None and download:
        raw = _download(verbose)
    _BOOK = ItemBook(_parse(raw or {}))
    return _BOOK


def _download(verbose: bool) -> dict | None:
    try:
        import requests
        r = requests.get(ITEMS_URL, timeout=30)
        r.raise_for_status()
        data = r.json()
        if len(data) < 100:
            raise ValueError(f"в ответе {len(data)} предметов — не похоже на справочник")
        ITEMS_PATH.parent.mkdir(parents=True, exist_ok=True)
        ITEMS_PATH.write_text(json.dumps(data, ensure_ascii=False),
                              encoding="utf-8")
        if verbose:
            print(f"Справочник предметов скачан: {len(data)} шт -> {ITEMS_PATH}")
        return data
    except Exception as e:                                  # noqa: BLE001
        if verbose:
            print(f"ВНИМАНИЕ: справочник предметов не скачался "
                  f"({type(e).__name__}: {e}).\n"
                  f"  Блоки по сборкам будут отключены, остальное работает.\n"
                  f"  Файл можно положить руками: {ITEMS_URL} -> {ITEMS_PATH}",
                  file=sys.stderr)
        return None


def _parse(raw: dict) -> dict[str, Item]:
    out: dict[str, Item] = {}
    for name, v in raw.items():
        try:
            iid = int(v["id"])
        except (KeyError, TypeError, ValueError):
            continue
        cost = v.get("cost")
        img = v.get("img") or ""
        out[name] = Item(
            id=iid,
            name=name,
            dname=v.get("dname") or name,
            # cost=0 в справочнике стоит у нейтралов и событийных предметов,
            # то есть означает «цены нет», а не «бесплатно».
            cost=int(cost) if isinstance(cost, (int, float)) and cost else None,
            qual=v.get("qual"),
            components=tuple(v.get("components") or ()),
            img=(IMG_BASE + img.split("?")[0]) if img else "",
            tier=v.get("tier"),
        )
    return out


# --- Лайв: разбор players[].items ---------------------------------------

@dataclass
class Build:
    """Сборка одного игрока в текущий момент."""
    slots: int                 # занятых слотов инвентаря (из шести) — что видно в игре
    backpack: int              # занятых слотов рюкзака
    kit: int                   # предметов сборки: без консьюмаблов, с рюкзаком
    value: int                 # золота в предметах (без консьюмаблов)
    big: int                   # крупных предметов (дороже BIG_ITEM_COST)
    items: list[Item | None]   # что именно, по слотам инвентаря
    unknown: int               # id, которых нет в справочнике
    known: bool                # можно ли верить числам выше


def build_of(item_ids, book: ItemBook) -> Build:
    """Сборка по массиву `players[].items` живого ответа.

    Пустой слот — −1. Неизвестный id справочнику (свежий патч, а файл старый)
    считается ЗАНЯТЫМ слотом, но в стоимость не идёт и попадает в `unknown`:
    занизить стоимость честнее, чем выдумать её, а слот занят наверняка.

    ДВА РАЗНЫХ СЧЁТА, и путать их нельзя. `slots` — сколько слотов инвентаря
    занято; это то, что зритель видит в игре, и там пыль с вардами занимают
    слот наравне с бабочкой. `kit` — сколько из этого предметы сборки
    (консьюмаблы не в счёт, рюкзак в счёт). Сравнивать с историей можно
    только `kit`: восстановление по `purchase_log` видит именно его и не
    различает инвентарь и рюкзак.
    """
    if not isinstance(item_ids, list) or not book.ok:
        return Build(0, 0, 0, 0, 0, [], 0, False)
    inv = item_ids[:N_INVENTORY_SLOTS]
    bp = item_ids[N_INVENTORY_SLOTS:N_ITEM_SLOTS]
    got: list[Item | None] = []
    slots = kit = value = big = unknown = 0
    for raw in inv:
        if raw is None or int(raw) < 0:
            got.append(None)
            continue
        slots += 1
        it = book.get(raw)
        got.append(it)
        if it is None:
            unknown += 1
            continue
        if not it.consumable and not it.neutral:
            kit += 1
            value += it.cost or 0
            if it.big:
                big += 1
    used_bp = 0
    for raw in bp:
        if raw is None or int(raw) < 0:
            continue
        used_bp += 1
        it = book.get(raw)
        if it is not None and not it.consumable and not it.neutral:
            kit += 1
            value += it.cost or 0
            if it.big:
                big += 1
    return Build(slots, used_bp, kit, value, big, got, unknown, True)


# --- Офлайн: восстановление инвентаря из purchase_log --------------------

def inventory_at(purchase_log: list[dict], t_sec: float,
                 book: ItemBook) -> Counter:
    """Что у игрока на руках к секунде `t_sec`, по журналу покупок.

    Как это работает и почему вообще работает. `purchase_log` пишет и части,
    и собранный предмет: `ogre_axe`, `mithril_hammer`, потом
    `black_king_bar`. Значит сборка восстанавливается механически: покупая
    предмет с рецептом, вынимаем из инвентаря его составные части, если они
    там лежат. Пример из настоящего матча:

        929s  ogre_axe          1000
       2181s  mithril_hammer    1600
       2299s  black_king_bar    4050   <- ogre_axe и mithril_hammer уходят

    ЧЕГО ЭТО НЕ ВОССТАНАВЛИВАЕТ: продажи и подобранное (аегис, гем с трупа).
    Съедаемые предметы учтены отдельно (CONSUMED_ON_USE). Точность замерена
    на конце матча и записана в README: набор крупных предметов совпадает у
    77.8% игроков, ошибка почти всегда в плюс (продали, а мы этого не
    видим). `dwp.test_builds` проверяет, что она не выросла.
    """
    inv: Counter = Counter()
    for e in purchase_log or []:
        if not isinstance(e, dict):
            continue
        ts, key = e.get("time"), e.get("key")
        if ts is None or key is None or float(ts) > t_sec:
            continue
        it = book.by_name.get(str(key))
        if it is None:
            continue
        for c in it.components:
            if inv.get(c, 0) > 0:
                inv[c] -= 1
                if inv[c] <= 0:
                    del inv[c]
        if it.name in CONSUMED_ON_USE:
            continue
        inv[it.name] += 1
    return inv


def build_from_inventory(inv: Counter, book: ItemBook) -> tuple[int, int, int]:
    """(слотов, стоимость, крупных) по восстановленному инвентарю.

    Консьюмаблы не считаются: шесть слотов тангами и ветками — это не сборка,
    а на счёте слотов они дали бы «шесть» уже на второй минуте.
    """
    slots = value = big = 0
    for name, n in inv.items():
        it = book.by_name.get(name)
        if it is None or it.consumable or it.neutral:
            continue
        slots += n
        value += (it.cost or 0) * n
        if it.big:
            big += n
    return slots, value, big


def main(argv: list[str] | None = None) -> int:
    """`python -m dwp.items` — скачать справочник и показать, что в нём есть."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass
    book = load()
    if not book.ok:
        print("Справочник предметов недоступен.", file=sys.stderr)
        return 2
    big = [it for it in book.by_name.values() if it.big]
    cons = [it for it in book.by_name.values() if it.consumable]
    neut = [it for it in book.by_name.values() if it.neutral]
    print(f"Предметов в справочнике: {len(book.by_id)}")
    print(f"  крупных (дороже {BIG_ITEM_COST}): {len(big)}")
    print(f"  консьюмаблов: {len(cons)}   нейтральных: {len(neut)}")
    print(f"  без цены: {sum(1 for it in book.by_name.values() if it.cost is None)}"
          f"  (у них стоимость сборки не растёт — это нейтралы и событийные)")
    top = sorted(big, key=lambda it: -(it.cost or 0))[:8]
    print("  самые дорогие: " + ", ".join(f"{it.dname} {it.cost}" for it in top))
    print(f"\nФайл: {ITEMS_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
