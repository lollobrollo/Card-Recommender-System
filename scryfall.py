import requests
import shutil
import ijson
import json
import gzip
import decimal
import os
import tempfile
import pandas as pd
from datetime import  date
import re
from tqdm import tqdm
from urllib.parse import urlparse
import io
from PIL import Image
import itertools

def download_data(output_path):
    """
    Downloads the latest version of Scryfall's 'default_cards' bulk data.
    - Fetches metadata from the Scryfall bulk API
    - Extracts the latest download URL for the 'default_cards' dataset
    - Streams and decompresses the .json.gz file directly to disk
    - Saves the uncompressed JSON to the specified output path
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print("Fetching latest bulk metadata...")
    bulk_api_url = "https://api.scryfall.com/bulk-data"

    response = requests.get(bulk_api_url)
    response.raise_for_status()

    bulk_data = response.json()["data"]
    
    # Find the 'default_cards' entry
    default_entry = next((entry for entry in bulk_data if entry["type"] == "default_cards"), None)
    if not default_entry:
        raise ValueError("Couldn't find 'default_cards' in Scryfall bulk data.")

    download_url = default_entry["download_uri"]
    print(f"Downloading latest default-cards from:\n{download_url}")

    with requests.get(download_url, stream=True) as response:
        response.raise_for_status()
        # Decompress while streaming directly to a JSON file
        with gzip.GzipFile(fileobj=response.raw) as gzipped:
            with open(output_path, 'wb') as out_file:
                shutil.copyfileobj(gzipped, out_file)
                
    print(f"Download complete.")


def get_all_fields(input_file):
    """ Scans the input JSON file and returns a set of all unique field names used across cards """
    fields = set()
    with open(input_file, 'r', encoding='utf-8') as f:
        for obj in ijson.items(f, "item"):
            fields.update(obj.keys())
    return fields


def filter_json_fields(input_file, fields_to_keep, output_file=None, inplace=False):
    """
    Filters a JSON file by keeping only the specified fields in each object.
    Args:
        - input_file (str) : Path to the input JSON file.
        - fields_to_keep (set) : Fields to retain in each JSON object.
        - output_file (str) : Output path (ignored if inplace=True).
        - inplace (bool) : If True, overwrite the input file with filtered data.
    """

    if inplace:
        temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(os.path.dirname(input_file)))
        os.close(temp_fd)
        output_path = temp_path
    else:
        output_path = output_file

    with open(input_file, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        outfile.write("[\n")
        first = True

        for card in ijson.items(infile, 'item'):
            filtered = {k: v for k, v in card.items() if k in fields_to_keep}
            
            if not first:
                outfile.write(",\n")
            else:
                first = False

            json.dump(filtered, outfile, ensure_ascii=False,
                      default=lambda o: float(o) if isinstance(o, decimal.Decimal) else str(o))

        outfile.write("\n]")

    if inplace:
        shutil.move(output_path, input_file)


def check_image_urls(card):
    """
    Extracts image_uris to top level (if not top level already) and filters for img type.
    Args:
        card (dict): The card object to process.
    Returns:
        bool: True if there's at least one URL, False otherwise.
    """
    image_uris = None
    img_types_to_keep = ['large', 'png']
    
    # Case 1: Standard card with top-level image_uris
    if 'image_uris' in card and card.get('image_uris'):
        image_uris = card['image_uris']
    # Case 2: Multi-faced card, check the first face (as per Scryfall convention)
    elif 'card_faces' in card and card.get('card_faces') and 'image_uris' in card['card_faces'][0]:
        image_uris = card['card_faces'][0]['image_uris']
    else:
        return False
    
    card['image_uris'] = {
        image_type: image_uris.get(image_type)
        for image_type in img_types_to_keep
        if image_type in image_uris
    }
    
    return bool(card['image_uris'])


def filter_data_by_relevance(input_file, json_path='item'):
    """
    Filters the dataset to retain only 'relevant' cards, keeping only the most
    recent printing of each card and handling cards that may lack a release date.
    
    Relevance rules are applied, multi-faced cards are merged, and the output
    is written to a temporary file before replacing the original.
    """
    def is_relevant(card):
        if "legal" != card.get("legalities", {}).get("commander", "NA"):
            return False
        if "Basic Land" in card.get("type_line", ""):
            return False
        if "paper" not in card.get("games", ()):
            return False
        if card.get("layout") in {"token", "double_faced_token"}:
            return False
        if card.get("lang", "NA") != "en":
            return False
        if card.get("set_type") in {"art_series", "memorabilia", "token", "minigame"}:
            return False
        if card.get("oversized", False):
            return False
        return True

    latest_cards = {}

    # --- PASS 1: Find the latest version of each relevant card while filtering them ---
    with open(input_file, 'r', encoding='utf-8') as infile:
        for card in tqdm(ijson.items(infile, json_path), desc="Finding latest printings"):
            if not card.get("oracle_text") and isinstance(card.get("card_faces"), list):
                faces = card["card_faces"]
                card["oracle_text"] = "\n".join(f.get("oracle_text", "") for f in faces if f.get("oracle_text"))
                card["type_line"] = " // ".join(f.get("type_line", "") for f in faces if f.get("type_line"))
                card["name"] = " // ".join(f.get("name", "") for f in faces if f.get("name"))
                card["mana_cost"] = " // ".join(f.get("mana_cost", "") for f in faces if f.get("mana_cost"))
                card["keywords"] = list(set(itertools.chain.from_iterable(f.get("keywords", []) for f in faces)))

            oid = card.get('oracle_id')
            if not oid or not is_relevant(card) or not check_image_urls(card):
                continue

            # Check for release date and keep newer card (or one that has a date)
            current_release = card.get('released_at')
            stored = latest_cards.get(oid)
            if stored is None:
                latest_cards[oid] = card
            else:
                stored_release = stored.get('released_at')
                if current_release and not stored_release:
                    latest_cards[oid] = card
                elif current_release and stored_release:
                    if current_release > stored_release:  # Both are YYYY-MM-DD
                        latest_cards[oid] = card

    # --- PASS 2: Write the definitive list of cards to a new file ---
    temp_fd, temp_path = tempfile.mkstemp(suffix=".json", dir=os.path.dirname(input_file))
    with os.fdopen(temp_fd, 'w', encoding='utf-8') as outfile:
        outfile.write("[\n")
        first = True
        for card in latest_cards.values():
            if not card.get("color_identity"):
                card["color_identity"] = ["C"]

            if not first:
                outfile.write(",\n")
            json.dump(card, outfile, ensure_ascii=False,
                      default=lambda o: str(o) if isinstance(o, decimal.Decimal) else o)
            first = False
        outfile.write("\n]")

    shutil.move(temp_path, input_file)


def filter_data(raw_data, clean_data):
    """
    Function that calls all filters in one place and saves the cleaned data in path 'clean_data'.
    Returns the fields that have been kept after removing useless ones and the ones useful only in preprocessing.
    """
    os.makedirs(os.path.dirname(clean_data), exist_ok=True)

    print(f"Applying filters...")
    
    keep_fields = {
    "oracle_id", "oracle_text", "type_line", "name", "lang", "keywords", "mana_cost",
    "colors", "color_identity", "games", "layout", "card_faces", "set", "set_name",
    "security_stamp", "legalities", "image_uris"
    }
    filter_json_fields(raw_data, keep_fields, clean_data)
    filter_data_by_relevance(clean_data)
    
    count = count_cards(clean_data)
    message = f"Cards after filtering: {count}\n"
    temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", dir=os.path.dirname(clean_data))
    with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
        tmp_file.write(message)

    keep_fields ={
    "oracle_id", "name", "set", "set_name", "oracle_text", "type_line",
    "mana_cost", "colors", "color_identity", "keywords", "image_uris"
    }
    filter_json_fields(clean_data, keep_fields, inplace = True)

    print(f"Filtering completed.")
    return keep_fields


def count_cards(file_path):
    total = 0
    # monocolored = 0
    # multicolored = 0
    with open(file_path, 'r', encoding='utf-8') as f:
        # print('Reading data...')
        for card in ijson.items(f, 'item'):
            total += 1
            # colors = card.get("colors", [])
            # if isinstance(colors, list):
            #     if len(colors) == 1:
            #         monocolored += 1
            #     elif len(colors) > 1:
            #         multicolored += 1
    # print(f"Total cards: {total}")
    # print(f"Monocolored cards: {monocolored} ({monocolored / total:.2%})")
    # print(f"Multicolored cards: {multicolored} ({multicolored / total:.2%})\n")
    return total


def download_images(data_path, output_folder=None):
    """
    Downloads a unique image for every card in the pre-filtered dataset.
    It prioritizes the 'large' JPG format. If unavailable, it falls back
    to the 'png' format, resizes it to the 'large' dimensions (672x936),
    and saves it as a JPG to ensure dataset uniformity.
    Args:
        - data_path (str): path to the pre-filtered clean_data.json file.
        - output_folder (str): the folder where images will be saved.
    """
    if output_folder is None:
        output_folder = os.path.join(os.path.dirname(data_path), "images")
    os.makedirs(output_folder, exist_ok=True)
    print("Scanning clean JSON to collect image URLs...")
    # Stores tuples of (url, format_type) where format_type is 'large' or 'png'
    urls_to_process = {}
    with open(data_path, 'r', encoding='utf-8') as f:
        for card in ijson.items(f, 'item'):
            oracle_id = card.get('oracle_id')
            image_uris = card.get('image_uris')
            
            if not (oracle_id and image_uris):
                continue
            
            # Prioritize 'large' (JPG), fallback to 'png'
            if 'large' in image_uris:
                urls_to_process[oracle_id] = (image_uris['large'], 'large')
            elif 'png' in image_uris:
                urls_to_process[oracle_id] = (image_uris['png'], 'png')

    print(f"Found {len(urls_to_process)} unique card images.")
    
    errors = 0
    target_size = (672, 936) # The resolution of 'large' JPGs

    print("\nStarting image download and processing...")
    for oracle_id, (url, img_type) in tqdm(urls_to_process.items(), desc="Downloading Images"):
        output_path = os.path.join(output_folder, f"{oracle_id}.jpg")

        # Skip if the final JPG file already exists (can interrupt the process anytime)
        if os.path.isfile(output_path):
            continue

        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()

            if img_type == 'large':
                # If it's the large JPG, just write it to file directly
                with open(output_path, 'wb') as outfile:
                    shutil.copyfileobj(response.raw, outfile)
            
            elif img_type == 'png':
                # If it's a PNG, open, resize, convert to RGB, and save as JPG
                with Image.open(io.BytesIO(response.content)) as img:
                    resized_img = img.resize(target_size, Image.Resampling.LANCZOS)
                    rgb_img = resized_img.convert('RGB')
                    rgb_img.save(output_path, 'JPEG', quality=95)
        except (requests.exceptions.RequestException, IOError) as e:
            errors += 1
            tqdm.write(f"Error processing {url}: {e}")

    print(f"\nImage processing complete.")
    if errors > 0:
        print(f"Total errors during download/processing: {errors}")


def create_card_dict(img_path):
    """
    creates a card dictionary 
    """



if __name__ == "__main__":    
    # download = input("Download fresher data? (Y/N): ").strip().lower() == "y"
    base_dir = os.path.dirname(os.path.realpath(__file__))
    raw_data = os.path.join(base_dir, "scryfall_data", "raw_data.json")
    clean_data = os.path.join(base_dir, "scryfall_data", "clean_data.json")

    # download_data(raw_data)
    filter_data(raw_data, clean_data)
    download_images(clean_data)

    # Found 33504 unique card images to download/process.