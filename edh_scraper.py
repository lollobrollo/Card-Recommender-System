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
import re


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


def normalize_card_name(name: str) -> str: # To make mapping more consistent
    name = name.lower()
    if ' // ' in name: name = name.split(' // ')[0]
    name = re.sub(r"[^\w\s]", '', name)
    return name.strip()


def load_name_maps(card_dict_path: str):
    if not os.path.exists(card_dict_path):
        print(f"Card dictionary not found at '{card_dict_path}'"); return None, None
    
    loaded_dict = load(card_dict_path)
    name_to_id_map = {normalize_card_name(name):oid for name,oid in loaded_dict.items()}
    known_oracle_ids = set(name_to_id_map.values())
    return name_to_id_map, known_oracle_ids


@dataclass
class Deck:
    deck_id: str
    source: str  # 'archidekt' | 'moxfield'
    name: Optional[str]
    commanders: List[str]  # names
    commander_ids: List[str]  # oracle_ids
    color_identity: List[str]  # e.g., ["U","R"]
    tags: List[str]
    power: Optional[str]  # 'cEDH', 'casual', etc. (best-effort)
    budget: Optional[float]  # USD if available
    size: int  # number of mainboard cards (non-commander)
    mainboard_ids: List[str]  # oracle_ids

    def to_json(self):
        return json.dumps(asdict(self), ensure_ascii=False)


ARCHIDEKT_BASE = "https://archidekt.com/api"


def archidekt_iter_decks(limit: int, rate: SimpleRateLimiter) -> Iterable[dict]:
    """Yield Commander deck search results (metadata pages)."""
    page = 1; fetched = 0
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    while fetched < limit:
        rate.wait()
        url = f"{ARCHIDEKT_BASE}/decks/v3/"
        params = {
            "orderBy": "-createdAt",
            "formats": [3], # numerical id for Commander
            "pageSize": 50,
            "page" : page
        }
        try:
            data = http_get_json(url, params=params, headers=headers)
            results = data.get('results', [])
            if not results: break
            for item in results:
                yield item
                fetched += 1
                if fetched >= limit: break
            page += 1
        except Exception as e:
            print(f"Warning: Archidekt search failed on page {page}: {e}. Stopping Archidekt search.")
            break


def archidekt_fetch_deck(deck_id: int, name_to_id: Dict[str, str], known_ids: Set[str], rate: SimpleRateLimiter) -> Optional[Deck]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    url = f"{ARCHIDEKT_BASE}/decks/v3/{deck_id}/"
    rate.wait()

    try:
        data = http_get_json(url, headers=headers)
        commanders, commander_ids, main_ids = [], [], []
        for c in data.get('cards', []):
            is_commander = 'commander' in (c.get('category') or "").lower()
            qty = int(c.get('quantity') or 1)
            card_info = c.get('card', {}).get('oracleCard', {})
            name = card_info.get('name'); oid = card_info.get('scryfallOracleId')

            if not (name and oid and oid in known_ids): continue # Ignore cards for which I have no representation

            if is_commander:
                commanders.append(name); commander_ids.append(oid)
            else:
                main_ids.extend([oid] * qty)

        if not commanders or (len(main_ids) + len(commander_ids)) != 100: return None
        
        return Deck(deck_id=str(deck_id),
                    source='archidekt',
                    name=data.get('name'),
                    commanders=commanders,
                    commander_ids=commander_ids,
                    color_identity=list(data.get('colors', {}).keys()),
                    tags=[str(t.get('name')) for t in data.get('tags', []) if t.get('name')],
                    power=None, budget=data.get('prices', {}).get('usd'),
                    size=len(main_ids),
                    mainboard_ids=main_ids)
    except Exception:
        return None

""" 
# I can ask for permission of use for non-commercial project https://moxfield.com/help/faq#moxfield-api

MOXFIELD_BASE = "https://api.moxfield.com/v2"


def moxfield_search_commander(page: int, rate: SimpleRateLimiter) -> List[dict]:
    headers = { "User-Agent": "edh-dataset-bot/0.1 (contact: research)",
                "Referer": "https://www.moxfield.com/decks"}
    params = {"q": "format:commander -private", "pageNumber": page, "pageSize": 50, "sortType": "updated"}
    rate.wait()
    return http_get_json(f"{MOXFIELD_BASE}/decks/search", params=params, headers=headers).get('data', [])


def moxfield_fetch_deck(public_id: str, name_to_id: Dict[str, str], rate: SimpleRateLimiter) -> Optional[Deck]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    rate.wait()
    data = http_get_json(f"{MOXFIELD_BASE}/decks/all/{public_id}", headers=headers)

    try:
        data = http_get_json(f"{MOXFIELD_BASE}/decks/all/{public_id}", headers=headers)
        meta = data.get('deck', {}); board = data.get('boards', {})
        main = board.get('mainboard', {}) or {}; commanders_board = board.get('commanders', {}) or {}
        commanders, commander_ids, main_ids = [], [], []
        
        for entry in commanders_board.values():
            card_obj = entry.get('card', {}); name = card_obj.get('name')
            oid = card_obj.get('scryfallOracleId') or name_to_id.get(normalize_card_name(name or ''))
            if name and oid and oid in known_ids: commanders.append(name); commander_ids.append(oid)

        for entry in main.values():
            qty = int(entry.get('quantity') or 1); card_obj = entry.get('card', {}); name = card_obj.get('name')
            oid = card_obj.get('scryfallOracleId') or name_to_id.get(normalize_card_name(name or ''))
            if oid and oid in known_ids: main_ids.extend([oid] * qty)

        if not commanders or (len(main_ids) + len(commander_ids)) != 100: return None

        return Deck(deck_id=meta.get('publicId') or public_id,
                    source='moxfield',
                    name=meta.get('name'),
                    commanders=commanders,
                    commander_ids=commander_ids,
                    color_identity=meta.get('colorIdentity') or [],
                    tags=meta.get('hubTags') or [],
                    power=None, budget=meta.get('price', {}).get('market'),
                    size=len(main_ids),
                    mainboard_ids=main_ids)
    except Exception:
        return None

"""

def color_bucket(ci: List[str]) -> str:
    unique_colors = set(c for c in ci if c)
    if not unique_colors:
        return "C"
    sorted_colors = sorted(list(unique_colors))
    return "".join(sorted_colors)


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
            if key in seen_cmdr: continue
            out.append(d)
            seen_cmdr.add(key)
            kept += 1
            if kept >= per_bucket: break
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
            rate_per_sec: float = 2.0):

    os.makedirs(os.path.dirname(out_jsonl) or '.', exist_ok=True)
    name_to_id, known_oracle_ids = load_name_maps(card_dict_path)
    if not name_to_id: return

    rate = SimpleRateLimiter(per_sec=rate_per_sec)

    decks = []

    print("Fetching Archidekt decks…")
    for meta in tqdm(archidekt_iter_decks(limit=max_archidekt, rate=rate), total=max_archidekt):
        did = meta.get('id')
        if did:
            d = archidekt_fetch_deck(int(did), name_to_id, known_oracle_ids, rate)
            if d: decks.append(d)

    """ 
    # Not used for now
    print("Fetching Moxfield decks…")
    fetched = 0; page = 1
    with tqdm(total=max_moxfield) as pbar:
        while fetched < max_moxfield:
            try:
                results = moxfield_search_commander(page=page, rate=rate)
                if not results: break
                for row in results:
                    if fetched >= max_moxfield: break
                    pid = row.get('publicId')
                    if pid:
                        d = moxfield_fetch_deck(pid, name_to_id, rate)
                        if d:
                            decks.append(d)
                            fetched += 1
                            pbar.update(1)
                page += 1
            except Exception as e:
                print(f"Warning: Error on Moxfield page {page}: {e}. Skipping page."); page += 1
    """

    uniq = {(d.source, d.deck_id): d for d in decks}
    decks = list(uniq.values())
    print(f"\nCollected {len(decks)} unique decks before diversification.")
    decks = diversify(decks, per_bucket=per_bucket)
    print(f"Kept {len(decks)} decks after diversification.")

    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for d in tqdm(decks, desc="Saving decks to JSONL"):
            row = asdict(d)
            if anchor_sizes:
                row['anchors'] = make_anchor_slices(d, anchor_sizes)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Wrote {len(decks)} decks into {out_jsonl}")



if __name__ == "__main__":
    this = os.path.dirname(__file__) or "."
    
    card_dict_path = os.path.join(this, "data", "card_dict.pt")
    output_path = os.path.join(this, "data", "edh_decks.jsonl")
    
    main(
        card_dict=card_dict_path,
        out_jsonl=output_path,
        max_archidekt=50,
        #max_moxfield=100,
        per_bucket=500,
        anchor_sizes=[50, 75, 90],
        rate_per_sec=4.0
    )