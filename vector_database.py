# chromadb docs 
# https://cookbook.chromadb.dev/core/filters/
# https://docs.trychroma.com/docs/overview/introduction

import chromadb
import torch
from tqdm import tqdm
import models
from train import load_card_encoder
import json
import ijson
import os
import re
import edh_scraper
import utils
from more_itertools import chunked
from itertools import combinations

def build_and_save_chroma_db(card_to_embedding_path, cards_metadata_path, client, db_name):
    """
    Loads card features, generates embeddings using a trained model and saves them into a persistent ChromaDB database.
    """
    print("--- Building the Card Vector Database ---")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(card_to_embedding_path):
        print(f"Card to embedding mapping not found at '{card_to_embedding_path}'"); return
    card_emb_map = torch.load(card_to_embedding_path, weights_only = False)

    print(f"Initializing persistent ChromaDB.")
    card_collection = client.get_or_create_collection(name=db_name)
    
    card_metadata_map = build_metadata_map(cards_metadata_path, card_emb_map.keys())

    print("Preparing metadata for all cards...")
    all_ids = []
    all_embeddings = []
    all_metadatas = []

    with torch.no_grad():
        for oracle_id, embedding in card_emb_map.items():
            all_ids.append(oracle_id)
            all_embeddings.append(embedding.tolist())
            metadata = card_metadata_map.get(oracle_id, {})
            all_metadatas.append(metadata)
    
    chunk_size = 5000
    num_batches = len(all_ids) // chunk_size + bool(len(all_ids)%chunk_size)
    print(f"Adding {len(all_ids)} cards to the ChromaDB collection...")
    for batch_ids, batch_embs, batch_meta in tqdm(zip(
        chunked(all_ids, chunk_size),
        chunked(all_embeddings, chunk_size),
        chunked(all_metadatas, chunk_size)), total=num_batches
    ): # chunking since max batch size for collection.add() is 5461
        card_collection.add(
            ids=batch_ids,
            embeddings=batch_embs,
            metadatas=batch_meta
        )
    print(f"Vector database has been built and saved.")
    print(f"Total cards in collection: {card_collection.count()}")


def format_list_for_chroma(data: list) -> str:
    """
    Takes a list of strings and formats it into a single, delimited string safe for ChromaDB metadata
    Commas at the start and end make LIKE queries exact
    Example: ['Flying', 'Vigilance'] -> ",Flying,Vigilance,"
    Example: ['U', 'W'] -> ",U,W,"
    """
    if not data: return ","
    return f",{','.join(data)},"


def build_metadata_map(clean_data_path, all_oracle_ids):
    """
    Builds a dictionary mapping oracle_id to its clean, filter-ready metadata
    """
    print("Generating cards metadata map... ", end = "")
    card_metadata_map = {}
    with open(clean_data_path, 'r', encoding='utf-8') as f:
        mapped_ids = set(all_oracle_ids)

        for card in ijson.items(f, "item"):
            oracle_id = card.get("oracle_id")
            
            if oracle_id and oracle_id in mapped_ids:
                color_identity_list = card.get("color_identity")
                if not color_identity_list:
                    color_identity_list = ["C"]

                keywords_list = card.get("keywords", [])

                mana_cost_str = card.get("mana_cost", "")
                mana_details = utils.parse_mana_cost(mana_cost_str)
                mana_colors_list = mana_details.get('colors', [])

                color_identity_str = format_list_for_chroma(color_identity_list)
                keywords_str = format_list_for_chroma(keywords_list)
                mana_colors_str = format_list_for_chroma(mana_colors_list)

                card_metadata_map[oracle_id] = {
                    "name": card.get("name", "Unknown"),
                    "color_identity": color_identity_str, # e.g., ",U,W,"
                    "cmc": mana_details['cmc'],
                    "mana_colors": mana_colors_str, # e.g., ",U,W,"
                    "x_cost": mana_details['x_cost'],
                    "type_line": card.get("type_line", ""),
                    "keywords": keywords_str, # e.g., ",Flying,Vigilance,"
                    "oracle_text": card.get("oracle_text", "")
                }
    print("done")
    return card_metadata_map



# - - - - - - - - - - - - - Card Search - - - - - - - - - - - - -


def create_card_representation(oid, partial_map, cat_map, feature_encoder):
    """
    creates and returns a complete card representation for a single card
    """
    card_types, card_keyw = cat_map[oid].values()
    card_types = card_types.unsqueeze(0).cpu()
    card_keyw = card_keyw.unsqueeze(0).cpu()
    cat_embeddings = feature_encoder(card_types, card_keyw).squeeze(0)
    partial_tensor = partial_map[oid].cpu()
    final_tensor = torch.cat((partial_tensor, cat_embeddings)).cpu()
    return final_tensor


def process_deck_for_search(deck_id, partial_repr_dict, cat_repr_dict_path, embedder_checkpoint_path, num_types, num_keyw, client=None, db_name=None):
    """
    Given a deck id, returns its embedding and its colors
    """
    partial_repr_dict = torch.load(partial_repr_dict, weights_only = False)
    type_keyw_dict = torch.load(cat_repr_dict_path, weights_only = False)

    all_known_ids = set(partial_repr_dict.keys())
    deck = edh_scraper.archidekt_fetch_deck(deck_id, known_ids=all_known_ids)
    if not deck:
        print(f"Could not fetch or process deck with ID: {deck_id}")
        return None, None

    decklist_ids = deck.commander_ids + deck.mainboard_ids
    assert type(decklist_ids) == list

    model = load_card_encoder(embedder_checkpoint_path, num_types, num_keyw, torch.device("cpu"))

    repr_decklist = []
    for oid in decklist_ids:
        card_repr = create_card_representation(oid, partial_repr_dict, type_keyw_dict, model.feature_encoder)
        repr_decklist.append(card_repr)

    if not repr_decklist:
        print("Deck contains no known cards.")
        return None, None

    # Model expects a batch -> add a batch dimension with unsqueeze(0)
    deck_tensor = torch.stack(repr_decklist).unsqueeze(0)
    with torch.no_grad():
        deck_emb_tensor = model.deck_embedding(deck_tensor).squeeze(0)
    deck_emb = deck_emb_tensor.cpu().tolist()

    if client is None:
        print("Please provide a client.")
        return None, None
    if db_name is None:
        print("Please provide database name.")
        return None, None
    
    card_collection = client.get_collection(name=db_name)
    if not card_collection:
        print("Client is not available.")
        return None, None

    results = card_collection.get(
        ids=decklist_ids,
        include=["metadatas"]
    )

    deck_colors = set()
    if results and results.get("metadatas"):
        for metadata in results['metadatas']:
            color_str = metadata.get("color_identity", ",")
            card_colors = [color for color in color_str.strip(',').split(',') if color]
            deck_colors.update(card_colors)
    # print(f"deck colors: {deck_colors}")

    return deck_emb, list(sorted(deck_colors))


def recommend_cards(deck_embedding:list, n:int, colors=None, client=None, db_name=None):
    """
    Takes a deck embedding and returns the top N recommended cards, optionally filtered by color.
    """
    if client is None:
        print("Please provide a client.")
        return []
    if db_name is None:
        print("Please provide database name.")
        return []

    card_collection = client.get_collection(name=db_name)
    if not card_collection:
        print("Client is not available.")
        return None, None

    filter_metadata = {}
    if colors:
        filter_metadata = build_color_subset_filter(colors)
    
    try:
        results = card_collection.query(
            query_embeddings=[deck_embedding],
            n_results=n,
            where=filter_metadata if filter_metadata else None
        )

        card_names = []
        if results and results['ids'][0]:
            for metadata in results['metadatas'][0]:
                card_names.append(metadata.get("name", "Unknown Name"))
        
        return card_names

    except Exception as e:
        print(f"An error occurred during ChromaDB query: {e}")
        return []


def build_color_subset_filter(colors):
    """
    Builds a ChromaDB '$or' filter to find cards whose color identity is an exact subset of the provided colors (including colorless)
    """
    if isinstance(colors, str):
        color_list = [colors.upper()]
    else:
        color_list = sorted(list({c.upper() for c in colors}))

    allowed_subsets = []
    for r in range(len(color_list) + 1):
        for subset in combinations(color_list, r):
            if not subset:
                # if processing empty subset, add "C" for colorless
                allowed_subsets.append(['C'])
            elif 'C' not in subset: # Check wether 'C' was inside original color_list and is getting coupled with other colors
                allowed_subsets.append(sorted(list(subset)))
    # print(f"allowed colors: {allowed_subsets}")

    # Format each subset into the delimited string format used in the DB
    # Example: ['U', 'W'] -> ",U,W,"
    formatted_subsets = []
    for subset in allowed_subsets:
        formatted_string = f",{','.join(subset)},"
        formatted_subsets.append(formatted_string)
    
    or_conditions = [{"color_identity": {"$eq": s}} for s in formatted_subsets]
    return {"$or": or_conditions}



if __name__ == "__main__":
    this = os.path.dirname(__file__)
    db_path = os.path.join(this, "card_db")
    card_emb_path = os.path.join(this, "data", "emb_dict_v1_all_20_3.pt")
    cards_metadata = os.path.join(this, "data", "clean_data.json")
    db_name = "mtg_cards_v1_all_20_3"

    client = chromadb.PersistentClient(path=db_path)
    build_and_save_chroma_db(card_emb_path, cards_metadata, client, db_name)


    card_emb_path = os.path.join(this, "data", "emb_dict_v1_all_200_3.pt")
    db_name = "mtg_cards_v1_all_200_3"
    build_and_save_chroma_db(card_emb_path, cards_metadata, client, db_name)

    # rielle_id = 11032857
    # repr_dict_path = os.path.join(this, "data", "card_repr_dict_v1.pt")
    # deck_emb, deck_colors = process_deck_for_search(rielle_id, repr_dict_path, embedder_checkpoint_path, client)
    # results_1 = recommend_cards(deck_embedding=deck_emb, n=10, client=client)
    # results_2 = recommend_cards(deck_embedding=deck_emb, n=10, colors=deck_colors, client=client)
    # print(results_1)
    # print(results_2)