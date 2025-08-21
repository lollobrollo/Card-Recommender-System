"""
EDH / Commander decklist harvesting for triplet-loss training.

Sources used (public, API-first to avoid scraping HTML / ToS issues):
  - Archidekt public API
  - Moxfield public API (undocumented but commonly used)

What this script does
---------------------
1) Fetches public Commander decks from Archidekt and Moxfield.
2) Normalizes them into a common structure with commander(s), colors, tags, and *oracle_id* lists.
3) Ensures variety (mono → five-color, different commanders and tags) and dedupes.
4) Optionally creates multiple "anchor" slices of different deck sizes (e.g., 10/25/50/99 cards) to support triplet sampling downstream.
5) Saves results to JSONL for easy incremental appends.
"""

from torch import load
import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Iterable, Set
import requests
import random
from tqdm import tqdm

class SimpleRateLimiter:
    def __init__(self, per_sec: float = 2.0):
        self.per_sec = per_sec
        self._last_t = 0.0

    def wait(self):
        if self.per_sec <= 0:
            return
        now = time.time()
        min_dt = 1.0 / self.per_sec
        dt = now - self._last_t
        if dt < min_dt:
            time.sleep(min_dt - dt)
        self._last_t = time.time()


def http_get_json(url: str, params: Optional[dict] = None, headers: Optional[dict] = None,
                  retries: int = 3, backoff: float = 1.5, timeout: float = 20.0):
    ex = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except Exception as e:
            ex = e
            time.sleep(backoff ** i)
    raise ex


@dataclass
class Deck:
    deck_id: str
    source: str  # 'archidekt' | 'moxfield'
    name: Optional[str]
    commanders: List[str]  # names
    commander_ids: List[str]  # oracle_ids
    color_identity: List[str]  # e.g., ["U","R"]
    tags: List[str]
    power: Optional[str]  # e.g., 'cEDH', 'casual', etc. (best-effort)
    budget: Optional[float]  # USD if available
    size: int  # number of mainboard cards (non-commander)
    mainboard_ids: List[str]  # oracle_ids

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


ARCHIDEKT_BASE = "https://archidekt.com/api"


def archidekt_iter_decks(limit: int, rate: SimpleRateLimiter) -> Iterable[dict]:
    """Yield Commander deck search results (metadata pages)."""
    page = 1
    fetched = 0
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    while fetched < limit:
        rate.wait()
        url = f"{ARCHIDEKT_BASE}/decks/"
        params = {
            "format": "Commander",
            "pageSize": 50,
            "page": page,
            "orderBy": "-createdAt"
        }
        data = http_get_json(url, params=params, headers=headers)
        results = data.get('results', [])
        if not results:
            break
        for item in results:
            yield item
            fetched += 1
            if fetched >= limit:
                break
        page += 1


def archidekt_fetch_deck(deck_id: int, name_to_id: Dict[str, str], rate: SimpleRateLimiter) -> Optional[Deck]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    url = f"{ARCHIDEKT_BASE}/decks/{deck_id}/"
    rate.wait()
    data = http_get_json(url, headers=headers)

    # Extract commanders and mainboard
    commanders: List[str] = []
    commander_ids: List[str] = []
    main_ids: List[str] = []
    color_identity: List[str] = []
    tags: List[str] = []
    power: Optional[str] = None
    budget: Optional[float] = None

    try:
        deck_name = data.get('name')
        # Tags/hubs
        if isinstance(data.get('tags'), list):
            tags = [str(t) for t in data['tags']]
        # Budget (best effort)
        if isinstance(data.get('prices'), dict):
            budget = data['prices'].get('usd')

        # Cards
        for c in data.get('cards', []):
            # Commander cards are marked in category
            cat = (c.get('category') or "").lower()
            is_commander = 'commander' in cat
            qty = int(c.get('quantity') or 0)
            scry_oracle = (
                c.get('card', {})
                 .get('oracleCard', {})
                 .get('scryfallOracleId')
            )
            name = c.get('card', {}).get('oracleCard', {}).get('name') or c.get('card', {}).get('name')
            if not name:
                continue
            oid = scry_oracle or name_to_id.get(name.lower())
            if not oid:
                continue
            if is_commander:
                commanders.append(name)
                commander_ids.append(oid)
            else:
                # mainboard: repeat by quantity
                for _ in range(qty or 1):
                    if oid in name_to_id.values(): # Only keep if I have a representation for this card
                        main_ids.append(oid)

        # Fallback color identity from commanders
        if data.get('colors'):
            color_identity = [c for c in data['colors'] if isinstance(c, str)]
        elif commander_ids:
            # derive from commanders' printed identity if provided (not always available via API)
            pass

        size = len(main_ids)
        if not commanders or size + len(commander_ids) != 100:  # Only keep legal EDH decks
            return None

        return Deck(
            deck_id=str(deck_id),
            source='archidekt',
            name=deck_name,
            commanders=commanders,
            commander_ids=commander_ids,
            color_identity=color_identity,
            tags=tags,
            power=power,
            budget=budget,
            size=size,
            mainboard_ids=main_ids,
        )
    except Exception:
        return None


MOXFIELD_BASE = "https://api.moxfield.com/v2"


def moxfield_search_commander(page: int, rate: SimpleRateLimiter) -> List[dict]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    # Sort by recency and popularity to diversify
    q = "format:commander -private"
    params = {"q": q, "pageNumber": page, "pageSize": 50, "sortType": "updated"}
    rate.wait()
    return http_get_json(f"{MOXFIELD_BASE}/decks/search", params=params, headers=headers).get('data', [])


def moxfield_fetch_deck(public_id: str, name_to_id: Dict[str, str], rate: SimpleRateLimiter) -> Optional[Deck]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    rate.wait()
    data = http_get_json(f"{MOXFIELD_BASE}/decks/all/{public_id}", headers=headers)

    try:
        meta = data.get('deck', {})
        board = data.get('boards', {})
        main = board.get('mainboard', {}) or {}
        commanders_board = board.get('commanders', {}) or {}

        # Commanders
        commanders: List[str] = []
        commander_ids: List[str] = []
        for entry in commanders_board.values():
            card_obj = entry.get('card', {})
            name = card_obj.get('name')
            oid = card_obj.get('scryfallOracleId') or name_to_id.get((name or '').lower())
            if name and oid:
                commanders.append(name)
                commander_ids.append(oid)

        # Mainboard
        main_ids: List[str] = []
        for entry in main.values():
            qty = int(entry.get('quantity') or 1)
            card_obj = entry.get('card', {})
            name = card_obj.get('name')
            oid = card_obj.get('scryfallOracleId') or name_to_id.get((name or '').lower())
            if oid:
                for _ in range(qty or 1):
                    if oid in name_to_id.values():
                        main_ids.append(oid)

        color_identity = meta.get('colorIdentity') or []
        tags = meta.get('hubTags') or []
        # Power/budget best-effort
        power = None
        budget = None
        if isinstance(meta.get('price'), dict):
            budget = meta['price'].get('market')

        size = len(main_ids)
        if not commanders or size + len(commander_ids) != 100:  # only legal EDH decks
            return None

        return Deck(
            deck_id=meta.get('publicId') or public_id,
            source='moxfield',
            name=meta.get('name'),
            commanders=commanders,
            commander_ids=commander_ids,
            color_identity=color_identity,
            tags=tags,
            power=power,
            budget=budget,
            size=size,
            mainboard_ids=main_ids,
        )
    except Exception:
        return None


def color_bucket(ci: List[str]) -> str:
    n = len(set(ci))
    return f"{n}-color"


def diversify(decks: List[Deck], per_bucket: int = 200) -> List[Deck]:
    """Keep up to per_bucket decks from each color-count bucket, favoring unique commanders."""
    buckets: Dict[str, List[Deck]] = {}
    for d in decks:
        buckets.setdefault(color_bucket(d.color_identity), []).append(d)

    out: List[Deck] = []
    for b, arr in buckets.items():
        seen_cmdr: Set[Tuple[str, ...]] = set()
        kept = 0
        for d in arr:
            key = tuple(sorted(d.commander_ids))
            if key in seen_cmdr:
                continue
            out.append(d)
            seen_cmdr.add(key)
            kept += 1
            if kept >= per_bucket:
                break
    return out


def make_anchor_slices(deck: Deck, sizes: List[int]) -> Dict[int, List[str]]:
    """ Create different-size anchors with randomized card selection """
    ids = deck.mainboard_ids.copy()
    random.shuffle(ids)  # randomize card order
    slices = {}
    for s in sorted(set(sizes)):
        s_eff = min(s, deck.size)
        slices[str(s_eff)] = ids[:s_eff]
    return slices


def main(   card_dict: str,
            out_jsonl: str,
            max_archidekt: int = 800,
            max_moxfield: int = 800,
            per_bucket: int = 200,
            anchor_sizes: Optional[List[int]] = None,
            rate_per_sec: float = 2.0) -> None:

    os.makedirs(os.path.dirname(out_jsonl) or '.', exist_ok=True)
    name_to_id = load(card_dict) # torch.load

    rate = SimpleRateLimiter(per_sec=rate_per_sec)

    # 1) Collect decks
    decks: List[Deck] = []

    # Archidekt
    count = 0
    for meta in tqdm(archidekt_iter_decks(limit=max_archidekt, rate=rate), desc="Fetching Archidekt deck summaries…", total=max_archidekt):
        did = meta.get('id')
        if did is None:
            continue
        d = archidekt_fetch_deck(int(did), name_to_id, rate)
        if d:
            decks.append(d)
        count += 1
        if count % 50 == 0:
            print(f"  Archidekt processed: {count}")

    # Moxfield
    print("Fetching Moxfield deck summaries…")
    fetched = 0
    page = 1
    
    with tqdm(total=max_moxfield) as pbar:
        while fetched < max_moxfield:
            results = moxfield_search_commander(page=page, rate=rate)
            if not results:
                break
            for row in results:
                pid = row.get('publicId')
                if not pid:
                    continue
                d = moxfield_fetch_deck(pid, name_to_id, rate)
                if d:
                    decks.append(d)
                    fetched += 1
                    pbar.update(1)
                if fetched >= max_moxfield:
                    break
            page += 1
            if page % 5 == 0:
                print(f"  Moxfield pages processed: {page-1}")
        if page % 5 == 0:
            pbar.set_description(f"Pages processed: {page-1}")

    # 2) Dedupe by (source,id)
    uniq: Dict[Tuple[str, str], Deck] = {}
    for d in decks:
        uniq[(d.source, d.deck_id)] = d
    decks = list(uniq.values())
    print(f"Collected {len(decks)} unique decks before diversification.")

    # 3) Diversify by color buckets and commander uniqueness
    decks = diversify(decks, per_bucket=per_bucket)
    print(f"Kept {len(decks)} decks after diversification.")

    # 4) Save to JSONL (with optional anchor slices)
    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for d in decks:
            row = asdict(d)
            if anchor_sizes:
                row['anchors'] = make_anchor_slices(d, anchor_sizes)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Wrote {len(decks)} decks → {out_jsonl}")



if __name__ == "__main__":
    this = os.path.dirname(__file__)

    card_dict = os.path.join(this, "data", "card_dict.pt")
    output=os.path.join(this, "data", "edh_decks.jsonl")
    max_archidekt = 800
    max_moxfield = 800
    per_bucket = 100
    anchor_sizes = [25, 50, 75, 90]
    rate = 8

    main(
        card_dict=card_dict,
        out_jsonl=output,
        max_archidekt=max_archidekt,
        max_moxfield=max_moxfield,
        per_bucket=per_bucket,
        anchor_sizes=anchor_sizes,
        rate_per_sec=rate,
    )
