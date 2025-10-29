"""
    Author : Lorenzo Bortolussi
    Year : 2024/2025
    This code is part of the implementation of the project developed for my Thesis in Artificial Intelligence and Data Analytics.
"""

from torch import save  # saves datasets and dictionaries to file
import os               # manages file paths
from tqdm import tqdm   # progress bars
import ijson            # iterate over long json files
import re               # string matching and filtering


def generate_and_save_card_dict(this):
    """
    Using data/clean_data.json, builds a case-insensitive map from card name to oracle_id
    Filters out cards without an associated image in data/images
    Saves the dictionary into data/card_dict.pt
    """
    data_dir = os.path.join(this, "data")
    json_path = os.path.join(data_dir, "clean_data.json")
    image_dir = os.path.join(data_dir, "images")
    output_path = os.path.join(data_dir, "card_dict.pt")
    
    # Get a set of all available image IDs
    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}")
        return

    # Create a set of oracle_ids from the image filenames
    try:
        available_image_ids = {os.path.splitext(f)[0] for f in os.listdir(image_dir)}
        print(f"Found {len(available_image_ids)} images in the '{os.path.basename(image_dir)}' directory.")
    except Exception as e:
        print(f"Error reading image directory {image_dir}: {e}")
        return

    # Iterate through the JSON and build the dictionary
    card_dict = {}
    print(f"Reading {os.path.basename(json_path)} to build dictionary...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for card in tqdm(ijson.items(f, 'item'), desc="Processing cards"):
                oracle_id = card.get('oracle_id')
                # Only add the card if its oracle_id has a corresponding image
                if oracle_id and oracle_id in available_image_ids:
                    name = card.get('name',"").lower()
                    if name:
                        card_dict[name] = oracle_id
    except FileNotFoundError:
        print(f"Error: Clean data file not found at {json_path}")
        return

    # Save the final dictionary
    if not card_dict:
        print("\nWarning: The final dictionary is empty. No matching images were found for cards in the JSON.")
        return
    print(f"\nGenerated dictionary with {len(card_dict)} cards.")
    try:
        save(card_dict, output_path)
        print(f"Successfully saved dictionary to: {output_path}")
    except Exception as e:
        print(f"\nError saving dictionary to {output_path}: {e}")


def synchronize_images_and_data(json_path, image_dir):
    """
    Synchronizes the image directory with the clean data file by deleting
    any images that do not have a corresponding entry in the clean JSON data.
    (needed when filtering applied becomes more strict, to avoid downloading images from scratch)
    Args:
        json_path (str): The path to the final clean_data.json file
        image_dir (str): The path to the directory containing downloaded images
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            json_oracle_ids = {card.get('oracle_id') for card in ijson.items(f, 'item') if card.get('oracle_id')}
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_path}. Aborting sync.")
        return
    print(f"Found {len(json_oracle_ids)} unique cards in the clean data file.")

    if not os.path.isdir(image_dir):
        print(f"Error: Image directory not found at {image_dir}. Nothing to sync.")
        return
        
    image_files = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)}
    image_oracle_ids = set(image_files.keys())
    print(f"Found {len(image_oracle_ids)} images in the '{os.path.basename(image_dir)}' directory.")

    ids_to_delete = image_oracle_ids - json_oracle_ids # Set difference operation
    if not ids_to_delete:
        print("\nImage directory is already in sync with the clean data. No files to delete.")
        return

    print(f"\nFound {len(ids_to_delete)} stale images to delete...")
    deleted_count = 0
    for oracle_id in tqdm(ids_to_delete, desc="Deleting stale images"):
        filename_to_delete = image_files[oracle_id]
        file_path_to_delete = os.path.join(image_dir, filename_to_delete)
        try:
            os.remove(file_path_to_delete)
            deleted_count += 1
        except OSError as e:
            tqdm.write(f"Error deleting file {file_path_to_delete}: {e}")
    print(f"\nSync complete. Deleted {deleted_count} stale images.")


def get_all_card_types_and_keywords(json_path):
    """
    Scans the dataset and extracts all unique types, supertypes, and subtypes from the 'type_line' of each card.
    Also extracts all keywords present in cards.
    Args:
        json_path (str): The path to the clean_data.json file.
    Returns two Python ordered lists containing:
        all unique type strings found in the dataset
        all unique keywords found in the dataset
    """
    print(f"Extracting all unique types an keywords from: {os.path.basename(json_path)}")
    all_types = set()
    all_keywords = set()
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            for card in ijson.items(f, 'item'):
                    all_types.update(extract_card_types(card))
                    all_keywords.update(extract_card_keywords(card))

    except FileNotFoundError:
        print(f"Error: Data file not found at {json_path}")
        return set()
    print(f"Extraction complete.\nFound {len(all_types)} unique types, supertypes, and subtypes.\nFound {len(all_keywords)} unique keywords.")

    return sorted(list(all_types)), sorted(list(all_keywords)) 


def extract_card_keywords(card):
    """
    Helper function that extract and returns all keywords of the card passed as argument
    """
    card_keywords = card.get("keywords")
    if not card_keywords:
        return set()
    return set(card_keywords)


def extract_card_types(card):
    """
    Helper function that extract and returns all types of the card passed as argument
    """
    card_types_set = set()
    type_line = card.get("type_line")
    if not type_line:
        return set()
    if "—" in type_line:
        parts = type_line.split('—')
        main_types_part = parts[0]
        sub_types_part = parts[1]
    else:
        main_types_part = type_line
        sub_types_part = "" # Card has no subtypes (e.g., "Instant")

    main_types = main_types_part.strip().split()
    card_types_set.update(main_types)

    if sub_types_part:
        sub_types = sub_types_part.strip().split()
        card_types_set.update(sub_types)

    return card_types_set


def preprocess_text(text, card_name="", remove_reminder=True, mana_as_words=True, mask_name=True):
    """
    Preprocess MRG-related text.
    Args:
        text (str): The raw oracle text
        card_name (srt): name of the card, if any. Used to mask its istances in the card.
        remove_reminder (bool): If True, strips reminder text in parentheses
        mana_as_words (bool): If True, replaces mana symbols like {G} with words ("green mana")
        mask_name (bool): If True, replace most istances of card self reference with a generic placeholder
    """
    if not text:
        return ""

    # Clean up repeated card name artifacts (which appear in EDHREC articles)
    pattern = r'\b(.+?)\s+\1\b'
    text = re.sub(pattern, r'\1', text)
    # Normalize common punctuation
    text = text.replace('“', '"').replace('”', '"')
    text = text.replace('‘', "'").replace('’', "'")
    text = text.replace('…', '...')

    # Remove Emojis and other non-ASCII symbols
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+",
        flags=re.UNICODE,
    )
    text = emoji_pattern.sub(r'', text)

    # Normalize newlines and remove artifacts like the replacement character
    text = text.strip().replace('�', '')
    text = re.sub(r"\s*\n\s*", ". ", text)

    # Remove reminder text
    if remove_reminder:
        text = re.sub(r"\([^)]*\)", "", text)
    # Convert mana cost symbols into words
    if mana_as_words:
        mana_map = {
            "{W}": "white mana", "{U}": "blue mana", "{B}": "black mana",
            "{R}": "red mana", "{G}": "green mana", "{C}": "colorless mana",
            "{T}": "tap", "{Q}": "untap"
        }
        mana_map.update({f"{{{i}}}": f"{i} generic mana" for i in range(0, 21)})
        for symbol, replacement in mana_map.items():
            text = text.replace(symbol, replacement)
    
    # Mask card self references
    if mask_name:
        self_ref_phrases = [
            r'this card', r'this creature', r'this artifact', r'this enchantment',
            r'this planeswalker', r'this land', r'this permanent'
        ]
        self_ref_regex = re.compile(r'\b(' + '|'.join(self_ref_phrases) + r')\b', flags=re.IGNORECASE)
        text = self_ref_regex.sub('[CARD]', text)

        if card_name and isinstance(card_name, str):
            name_to_mask = card_name.split(' // ')
            for name in name_to_mask:
                card_name_pattern = re.escape(name)
                card_name_regex = re.compile(rf'\b{card_name_pattern}\b', flags=re.IGNORECASE)
                text = card_name_regex.sub('[CARD]', text)
    
    # Final cleaning of spaces and punctuation
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    return text.strip()


def parse_mana_cost(mana_cost_str: str) -> dict:
    if not mana_cost_str:
        return {'cmc': 0, 'colors': [], 'generic': 0, 'x_cost': False}
    symbols = re.findall(r"\{([^}]+)\}", mana_cost_str)
    cmc = 0
    colors = []
    generic_mana = 0
    has_x_cost = False
    for symbol in symbols:
        if symbol.isdigit():
            cost = int(symbol)
            cmc += cost
            generic_mana += cost
        elif symbol.upper() == 'X':
            has_x_cost = True
        elif '/' in symbol: # Hybrid/phyrexian
            cmc += 1
            parts = symbol.upper().split('/')
            for part in parts:
                if part in "WUBRG":
                    colors.append(part)
        elif symbol.upper() in "WUBRG":
            cmc += 1
            colors.append(symbol.upper())
    return {
        'cmc': cmc,
        'colors': sorted(list(set(colors))),
        'generic': generic_mana,
        'x_cost': has_x_cost
    }



if __name__ == "__main__":
    this = os.path.dirname(__file__)
    data_dir = os.path.join(this, "data")
    # img_dir = os.path.join(data_dir, "images")
    # dataset_path = os.path.join(data_dir, "img_dataset.pt")
    generate_and_save_card_dict(this)
    # save_dataset_to_pt(img_dir, dataset_path)

    partial_data_path = os.path.join(data_dir, "card_repr_dict_v1.pt")
    type_and_keyw_path = os.path.join(data_dir, "type_and_keyw_dict.pt")
