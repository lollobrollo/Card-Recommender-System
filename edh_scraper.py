import torch
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
from collections import defaultdict
from models import TripletEDHDataset


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
    
    loaded_dict = torch.load(card_dict_path)
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

# Useful info for using the api:
# https://www.npmjs.com/package/archidekt (outdated)
# https://archidekt.com/forum/thread/10531812
# https://github.com/linkian209/pyrchidekt

ARCHIDEKT_BASE = "https://archidekt.com/api/decks"


def archidekt_iter_decks(limit: int, rate: SimpleRateLimiter) -> Iterable[dict]:
    """Yield Commander deck search results (metadata pages)."""
    page = 1; fetched = 0
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    url = f"{ARCHIDEKT_BASE}/v3/"
    while fetched < limit:
        rate.wait()
        params = {
            "size": 100, # Only completed decks
            "orderBy": "-createdAt",
            "deckFormat": 3,
            "pageSize": 50,
            "page" : page
        }
        try:
            data = http_get_json(url, params=params, headers=headers)
            results = data.get('results', [])
            if not results:
                print(f"\nCould not get results for page {page}")
                break
            for item in results:
                yield item
                fetched += 1
                if fetched >= limit: break
            page += 1
        except Exception as e:
            print(f"\nWarning: Archidekt search failed on page {page}: {e}. Stopping Archidekt search.")
            break


def archidekt_fetch_deck(deck_id:int, name_to_id: Dict[str, str], known_ids: Set[str], rate: SimpleRateLimiter) -> Optional[Deck]:
    headers = {"User-Agent": "edh-dataset-bot/0.1 (contact: research)"}
    rate.wait()
    try:
        url = f"{ARCHIDEKT_BASE}/{deck_id}/"
        deck = http_get_json(url, headers=headers)

        commanders, commander_ids, main_ids = [], [], []
        deck_color_identity = set()
        card_list = deck.get('cards')
        if not card_list:
            return None
        for c in card_list:
            is_commander = 'Commander' in c.get('categories', [])
            qty = int(c.get('quantity') or 1)
            card_info = c.get('card', {}).get('oracleCard', {})
            name = card_info.get('name')
            oid = card_info.get('uid')
            if not (name and oid and oid in known_ids): continue # Ignore cards for which I have no representation (LANDS ARE OMITTED)

            card_colors = card_info.get('colorIdentity', [])
            deck_color_identity.update(card_colors)

            if is_commander:
                commanders.append(name)
                commander_ids.append(oid)
            else:
                main_ids.extend([oid] * qty)

        if not commanders: return None

        tags_list = deck.get('tags')
        tags = [str(t.get('name')) for t in tags_list if t.get('name')] if tags_list else []
        
        return Deck(deck_id=str(deck_id),
                    source='archidekt',
                    name=deck.get('name'),
                    commanders=commanders,
                    commander_ids=commander_ids,
                    color_identity=sorted(list(deck_color_identity)),
                    tags=tags,
                    power=None, budget=deck.get('prices', {}).get('usd'),
                    size=len(main_ids),
                    mainboard_ids=main_ids)
    except Exception as e:
        # print(f"\nError while fetching for deck {deck_id} : {e}")
        return None

""" 
# I can ask for permission of use for a non-commercial project: https://moxfield.com/help/faq#moxfield-api

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


ALL_TAGS = [
    "+1+1 counters", "-1-1 counters", "adventures", "aggro", "alternate wincon", "angels", "aristocrats", "artifact", "assassins", "bears", "beasts", "big mana", "birds",
    "blink", "bounce", "budget", "burn", "cantips", "cats", "cascade", "casual", "cedh", "changeling", "chaos", "charge counter", "cheerios", "clerics", "clones", "clues",
    "coin flip", "combo", "commander matter", "connive", "control", "convoke", "counters", "counterspells", "crabs", "craft", "creatureless", "creatures", "crime", "curses",
    "cycling", "day/night", "deathtouch", "defender", "delirium", "delve", "demons", "descend", "desert", "devils", "devotion", "die roll", "dinosaur", "discard", "discover",
    "dogs", "donate", "dragons", "dragon's approach", "draw", "dredge", "druids", "dungeon", "dwarves", "eggs", "eldrazi", "elementals", "elves", "enchantments", "enchantress",
    "energy", "enrage", "equipment", "etb effect", "exalted", "exile", "experience counters", "exploit", "explore", "extra combats", "extra turns", "extra upkeep", "face-down",
    "faeries", "fight", "flash", "flashback", "flavor", "fling", "flying", "food", "forced combat", "foretell", "freerunning", "frogs", "fungi", "gates", "giants", "glass cannon",
    "gnomes", "goats", "goblins", "gods", "golems", "good stuff", "graveyard", "group hug", "group slug", "gyruda", "halflings", "hand size", "hare apparent", "haste", "hatebears",
    "hellbent", "heroic", "hidden commander", "hippos", "historic", "horrors", "horses", "humans", "hydras", "illusions", "improvise", "impulse draw", "indestructible", "infect",
    "insects", "jank", "jegantha", "kaheera", "keruga", "keywords", "kicker", "kithkin", "knights", "land animation", "land destruction", "landfall", "lands", "legends", "lhurgoyfs",
    "lifedrain", "lifegain", "limited", "lizards", "madness", "mercenaries", "merfolk", "mice", "midrange", "mill", "minotaurs", "modular", "monarch", "monkeys", "monks",
    "morph", "mounts", "multicolor matters", "mutants", "mutate", "myr", "myriad", "nightmares", "ninjas", "ninjutsu", "offspring", "oil counters", "old school", "oozes",
    "otter", "outlaws", "paradox", "party", "persistent petitioners", "phasing", "phoenixes", "phyrexians", "pillow fort", "pingers", "pirates", "planeswalkers", "plants", "plot",
    "pod", "politics", "polymorph", "populate", "power", "praetors", "primal surge", "prison", "proliferate", "prowess", "rabbits", "raccoons", "rad counters", "ramp", "rat colony",
    "rats", "reach", "reanimator", "rebels", "robots", "rock", "rogues", "rooms", "rotated standard", "rule zero", "saboteur", "sacrifice", "sagas", "salty", "samurai", "saprolings",
    "satyr", "scarecrows", "scry", "sea creatures", "self-damage", "self-discard", "self-mill", "shades", "shadowborn apostles", "shamans", "shapeshifters", "sharks", "shrines",
    "skeletons", "slime against humanity", "slivers", "snakes", "sneak attack", "snow", "soldiers", "spacecraft", "specters", "speed", "spell copy", "spellslinger", "sphinxes",
    "spiders", "spirits", "squirrels", "stax", "stickers", "stompy", "storm", "sunforger", "surveil", "suspend", "tap/untap", "tempest hawk", "templar knights", "tempo", "theft",
    "thopters", "time counters", "time lords", "tiny leaders", "tokens", "toolbox", "topdecks", "toughness", "transform", "trasure", "treefolk", "turbo fog", "turtles", "tyranids",
    "unblockable", "unicorns", "unmodified precon", "upgraded precon", "vampires", "vanilla", "vehicles", "voltron", "voting", "warriors", "weenies", "werewolves", "wheels",
    "wizards", "wolves", "wraiths", "wurms", "x spells", "zirda", "zombies", "zoo"
]



def extract_strategic_tags(deck_tags: List[str]) -> Set[str]:
    """
    Finds all relevant strategic tags from a deck's tag list based on a priority list
    """
    normalized_deck_tags = {tag.lower().strip() for tag in deck_tags}
    found_tags = set()
    for p_tag in ALL_TAGS:
        if p_tag in normalized_deck_tags:
            found_tags.add(p_tag)
    if not found_tags:
        return {'untagged'}
    return found_tags


def diversify(decks: List[Deck], per_bucket: int, n_duplicates: int) -> List[Deck]:
    """
    Diversification function with two levels of control:
    1. Keeps a maximum of 'per_bucket' decks for each color identity.
    2. Within that bucket, keeps up to 'n_duplicates' for each unique (commander, strategic_tag) combination.
    Most decks are untagged, so the limit for them is more lenient.
    """
    buckets: Dict[str, List[Deck]] = defaultdict(list)

    for d in decks:
        buckets[color_bucket(d.color_identity)].append(d)
    excluded_report = {k: 0 for k in buckets.keys()} # Used to save bucket results on file, for later checks 

    final_decks: List[Deck] = []
    
    for bucket_name, deck_list in sorted(buckets.items()):
        strategy_counts = defaultdict(int) # Key: (commander_tuple, tag_string), Value: count
        kept_deck_ids = set()
        bucket_output = []
        excluded_report[bucket_name] = max(0, len(bucket_output) - per_bucket)

        random.shuffle(deck_list)
        for deck in deck_list:
            if len(bucket_output) >= per_bucket:
                break
            commander_key = tuple(sorted(deck.commander_ids))
            strategic_tags = extract_strategic_tags(deck.tags)

            keep_deck = False
            for tag in strategic_tags:
                strategy_key = (commander_key, tag)
                if tag != "untagged" and strategy_counts[strategy_key] < n_duplicates:
                    keep_deck = True
                    strategy_counts[strategy_key] += 1
                if tag == "untagged" and strategy_counts["untagged"] < n_duplicates*4:
                    keep_deck = True
                    strategy_counts[strategy_key] += 1

            if keep_deck and deck.deck_id not in kept_deck_ids:
                bucket_output.append(deck)
                kept_deck_ids.add(deck.deck_id)

        final_decks.extend(bucket_output)
    excluded_report_sorted = dict(sorted(excluded_report.items(), key = lambda item: item[1], reverse=True))
    report_path = os.path.join(os.path.dirname(__file__), "data", "buckets_exclusion_report.pt")
    torch.save(excluded_report_sorted, report_path)

    return final_decks
    

def main(   card_dict: str = None,
            out_jsonl: str = None,
            max_archidekt: int = 800,
            max_moxfield: int = 800,
            per_color_bucket: int = 200,
            n_duplicates_per_strategy: int = 3,
            anchor_sizes: Optional[List[int]] = None,
            rate_per_sec: float = 2.0):

    dir = os.path.dirname(__file__) or "."
    if card_dict is None:
        card_dict = os.path.join(dir, "data", "card_dict.pt")
    if out_jsonl is None:
        out_jsonl = os.path.join(dir, "data", "edh_decks.jsonl")
    
    os.makedirs(dir, exist_ok=True)
    name_to_id, known_oracle_ids = load_name_maps(card_dict)
    if not name_to_id: return

    rate = SimpleRateLimiter(per_sec=rate_per_sec)

    decks = []
    print(f"Trying to fetch {max_archidekt} decks.")
    print("Fetching Archidekt decks…")
    for deck_meta in tqdm(archidekt_iter_decks(limit=max_archidekt, rate=rate), total=max_archidekt):
        did = deck_meta.get("id")
        if did:
            d = archidekt_fetch_deck(did, name_to_id, known_oracle_ids, rate)
            if d:
                decks.append(d)

    ### Not used in this version
    # print("Fetching Moxfield decks…")
    # fetched = 0; page = 1
    # with tqdm(total=max_moxfield) as pbar:
    #     while fetched < max_moxfield:
    #         try:
    #             results = moxfield_search_commander(page=page, rate=rate)
    #             if not results: break
    #             for row in results:
    #                 if fetched >= max_moxfield: break
    #                 pid = row.get('publicId')
    #                 if pid:
    #                     d = moxfield_fetch_deck(pid, name_to_id, rate)
    #                     if d:
    #                         decks.append(d)
    #                         fetched += 1
    #                         pbar.update(1)
    #             page += 1
    #         except Exception as e:
    #             print(f"Warning: Error on Moxfield page {page}: {e}. Skipping page."); page += 1

    uniq = {(d.source, d.deck_id): d for d in decks}
    decks = list(uniq.values())
    print(f"\nCollected {len(decks)} unique decks before diversification.")
    decks = diversify(decks, per_bucket=per_color_bucket, n_duplicates= n_duplicates_per_strategy)
    print(f"Kept {len(decks)} decks after diversification.")

    with open(out_jsonl, 'w', encoding='utf-8') as f:
        for d in tqdm(decks, desc="Saving decks to JSONL"):
            row = asdict(d)
            # if anchor_sizes:
            #     row['anchors'] = make_anchor_slices(d, anchor_sizes)
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f"Wrote {len(decks)} decks into {out_jsonl}")


def create_and_save_CPRdataset(decks_path: str, output_path: str, card_feature_map_path: str):
    """
    Creastes and saves a dataset for the training of the PipelineCPR model
    Extracts required informations from the decks scraped from the web (archidekt)
    """
    try:
        card_feature_map = torch.load(card_feature_map_path)
    except Exception as e:
        print(f"Couldn't load feature map: {e}")

    try:
        max_deck_size = 0; min_deck_size = 101
        decklists = []
        with open(decks_path, "r", encoding="utf-8") as decks:
            for line in decks:
                deck = json.loads(line)
                cards = deck.get("mainboard_ids", [])
                cards.extend(deck.get("commander_ids", []))
                decklists.append(cards)
                max_deck_size = max(max_deck_size, len(cards))
                min_deck_size = min(min_deck_size, len(cards))
        anchor_size_range = (min_deck_size, max_deck_size)
        
        dataset = TripletEDHDataset(decklists, card_feature_map, anchor_size_range)
        torch.save(dataset, output_path)
    except Exception as e:
        print(f"Couldn't save initialised dataset: {e}")
    print(f"successfully saved {len(decklists)} deckists into dataset to {output_path}")



if __name__ == "__main__":

    # main(
    #     max_archidekt=70000,
    #     #max_moxfield=100,
    #     per_color_bucket=2000,
    #     n_duplicates_per_strategy = 4,
    #     rate_per_sec=5.0
    # )

    this = os.path.dirname(__file__)
    decks_path = os.path.join(this, "data", "edh_decks.jsonl")
    dataset_path = os.path.join(this, "data", "cpr_dataset.pt")
    card_feature_map_path = os.path.join(this, "data", "card_repr_dict_v1.pt")
    create_and_save_CPRdataset(decks_path, dataset_path, card_feature_map_path)