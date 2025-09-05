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
import torch
from torchvision import transforms
import utils
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from models import load_img_encoder

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
    "security_stamp", "legalities", "image_uris", "power", "toughness", "rarity"
    }
    filter_json_fields(raw_data, keep_fields, clean_data)
    filter_data_by_relevance(clean_data)
    
    # count = count_cards(clean_data)
    # message = f"Cards after filtering: {count}\n"
    # temp_fd, temp_path = tempfile.mkstemp(suffix=".txt", dir=os.path.dirname(clean_data))
    # with os.fdopen(temp_fd, 'w', encoding='utf-8') as tmp_file:
    #     tmp_file.write(message)

    keep_fields ={
    "oracle_id", "name", "set", "set_name", "oracle_text", "type_line",
    "mana_cost", "colors", "color_identity", "keywords", "image_uris",
    "power", "toughness", "rarity"
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


def build_card_representations(batch_size=8, use_img=False):
    """
    Takes in card data and turns them into corresponding card representations,
    which are saved as a dictionary into a file for later use.
    Args:
        batch_size (int): Size of card batches processed by models
        use_img (bool): If True, add to the representation the card image embedding
    """
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    models_dir = os.path.join(base_dir, "models")

    cards_path = os.path.join(data_dir, "clean_data.json")
    images_dir = os.path.join(data_dir, "images")
    dict_path = os.path.join(data_dir, "card_dict.pt")
    type_and_keyw_path = os.path.join(data_dir, "type_and_keyw_dict.pt")
    output_path = ""
    if use_img:
        output_path = os.path.join(data_dir, "card_repr_dict_v2.pt")
    else:
        output_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    card_dict = torch.load(dict_path, weights_only=False)

    encoder_path = os.path.join(models_dir, "ImgEncoder.pt")
    if use_img:
        encoder = load_img_encoder(encoder_path, device)

    role_model_path = os.path.join(models_dir, "card-role-classifier-final") # one-hot encodes roles of cards
    role_tokenizer = AutoTokenizer.from_pretrained(role_model_path)
    role_model = AutoModelForSequenceClassification.from_pretrained(role_model_path).to(device)
    role_model.eval()

    foundation_model_path = os.path.join(models_dir, "magic-distilbert-base-v1") # specialized model for sentence embeddings
    semantic_text_model = SentenceTransformer(foundation_model_path, device=device)
    
    # Ordered lists used to create one-hot encodings of variables
    all_types, all_keywords = utils.get_all_card_types_and_keywords(cards_path)
    rarity_levels = ["common", "uncommon", "rare", "mythic"]
    color_id_levels = ["B", "C", "G", "R", "U", "W"]

    card_repr = {} # Used to save all card representations
    type_and_keyw = {} # Saves intermediate one-hot encodings, which are used during training by FeatureEncoder layer (reason: dimensionality reduction of sparse vectors)

    def safe_int(val): # Used to safely interpret power and toughness
        if val == '*':
            return 0
        try:
            return int(val)
        except (TypeError, ValueError):
            return -1

    def extract_cmc(mana_cost): # Converts a curly-brace mana cost string into its total integer cost
        if not mana_cost:
            return -1
        symbols = re.findall(r"\{([^}]+)\}", mana_cost)
        if not symbols:
            return 0
        total = 0
        for symbol in symbols:
            if symbol.isdigit():  # Generic mana like {4}
                total += int(symbol)
            else:  # Any colored/special mana counts as 1
                total += 1
        return total

    def process_batch(cards_batch):
        # Load and preprocess images as tensor batch
        imgs = []
        if use_img:
            for card in cards_batch:
                img_path = os.path.join(images_dir, f"{card['oracle_id']}.jpg")
                with Image.open(img_path) as img:
                    img_tensor = transforms.Compose([transforms.Resize((936, 672)), transforms.ToTensor()])(img)
                    imgs.append(img_tensor)
            imgs = torch.stack(imgs).to(device)
            # Encode images batch
            with torch.no_grad():
                img_encoded_batch = encoder.encode(imgs) # shape: (batch_size, 1024)

        # Encode semantic embeddings
        oracle_texts = [utils.preprocess_oracle_text(card.get("oracle_text", ""), card.get("name", "")) for card in cards_batch]
        text_emb_batch = semantic_text_model.encode(oracle_texts, batch_size=batch_size, convert_to_tensor=True)
        
        # Encode oracle texts
        semantic_emb_batch = semantic_text_model.encode(oracle_texts, convert_to_tensor=True, device=device)

        # Encode roles of cards
        role_inputs = role_tokenizer(oracle_texts, return_tensors='pt', padding=True, truncation=True).to(device)
        with torch.no_grad():
            role_logits_batch = role_model(**role_inputs).logits

        # Build vector representations
        for i, card in enumerate(cards_batch):
            types_present = utils.extract_card_types(card)
            card_types = [int(t in types_present) for t in all_types]

            keywords_present = utils.extract_card_keywords(card)
            keywords_encoded = [int(k in keywords_present) for k in all_keywords]

            stats = [safe_int(card.get("power")), safe_int(card.get("toughness"))]

            rarity_encoded = [int(card.get("rarity") == r) for r in rarity_levels]

            card_color_id = card.get("color_identity", [])
            if not card_color_id:
                card_color_id = ["C"]
            color_id_encoded = [int(c in card_color_id) for c in color_id_levels]
            
            cmc = extract_cmc(card.get("mana_cost"))

            # Concatenate all features as one tensor
            ## !!! CARD TYPES AND KEYWORDS ARE SAVED SEPARATELY, WILL BE COMPRESSED DURING TRAINING BY FeatureEncoder
            parts = [
                # torch.as_tensor(card_types, dtype=torch.float32, device="cpu"),
                # torch.as_tensor(keywords_encoded, dtype=torch.float32, device="cpu"),
                torch.as_tensor(stats, dtype=torch.float32, device="cpu"),
                torch.as_tensor(rarity_encoded, dtype=torch.float32, device="cpu"),
                torch.as_tensor(color_id_encoded, dtype=torch.float32, device="cpu"),
                torch.tensor([cmc], dtype=torch.float32),
                semantic_emb_batch[i].cpu(),
                role_logits_batch[i].cpu()
            ]
            if use_img and img_encoded_batch is not None:
                parts.append(img_encoded_batch[i].to("cpu"))

            card_vector = torch.cat(parts)
            oid = card["oracle_id"]
            card_repr[oid] = card_vector.cpu()

            type_and_keyw[oid] = {
                "types": torch.as_tensor(card_types, dtype=torch.float32, device="cpu"),
                "keywords": torch.as_tensor(keywords_encoded, dtype=torch.float32, device="cpu")
                }
    
    # Main loop: batch cards before processing
    batch = []
    with open(cards_path, 'r', encoding='utf-8') as f:
        for card in tqdm(ijson.items(f, 'item'), total=29444, desc="Creating representations"):
            batch.append(card)
            if len(batch) == batch_size:
                process_batch(batch)
                batch = []

    # Process remaining cards in batch if any
    if batch:
        process_batch(batch)

    try:
        torch.save(card_repr, output_path)
        torch.save(type_and_keyw, type_and_keyw_path)
        print(f"Successfully saved dictionaries to:\n{output_path}\n{type_and_keyw_path}\n")
    except Exception as e:
        print(f"\nError saving dictionary to {output_path}: {e}\n")    



if __name__ == "__main__":
    # download = input("Download fresher data? (Y/N): ").strip().lower() == "y"
    # base_dir = os.path.dirname(os.path.realpath(__file__))
    # raw_data = os.path.join(base_dir, "data", "raw_data.json")
    # clean_data = os.path.join(base_dir, "data", "clean_data.json")

    # download_data(raw_data)
    #filter_data(raw_data, clean_data)
    # download_images(clean_data)

    # # Found 33504 unique card images to download/process.
    # # cards after filtering: 29444

    # # Added stricter filters, need to delete some images
    # img_dir = os.path.join(os.path.dirname(__file__), "data", "images")
    # utils.synchronize_images_and_data(clean_data, img_dir)
    # Deleted 4060 stale images.

    # utils.generate_and_save_dict()

    # print(count_cards(clean_data))

    build_card_representations(batch_size=16, use_img=False)
    build_card_representations(batch_size=16, use_img=True)
    