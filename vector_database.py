import chromadb
import torch
from tqdm import tqdm
import models
import json
import ijson
import os
import re
import edh_scraper
import utils
from more_itertools import chunked
from itertools import combinations

def build_and_save_chroma_db(card_feature_map_path, embedder_checkpoint_path, cards_metadata_path, client):
    """
    Loads card features, generates embeddings using a trained model and saves them into a persistent ChromaDB database
    """
    print("--- Building the Card Vector Database ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(card_feature_map_path):
        print(f"Card feature map not found at '{card_feature_map_path}'"); return
    card_feature_map = torch.load(card_feature_map_path)

    card_encoder = load_card_encoder(embedder_checkpoint_path, device)
    
    print(f"Initializing persistent ChromaDB.")
    card_collection = client.get_or_create_collection(name="mtg_cards_v1")
    
    card_metadata_map = build_metadata_map(cards_metadata_path, card_feature_map)

    print("Generating embeddings for all cards...")
    all_ids = []
    all_embeddings = []
    all_metadatas = []

    with torch.no_grad():
        for oracle_id, feature_tensor in tqdm(card_feature_map.items(), desc="Encoding cards"):
            # Model expects a batch -> add a batch dimension with unsqueeze(0)
            feature_tensor = feature_tensor.unsqueeze(0).to(device)
            embedding_tensor = card_encoder.card_embedding(feature_tensor).squeeze(0)
            embedding = embedding_tensor.cpu().tolist()
            all_ids.append(oracle_id)
            all_embeddings.append(embedding)
            metadata = card_metadata_map.get(oracle_id, {})
            all_metadatas.append(metadata)

    print(f"Adding {len(all_ids)} cards to the ChromaDB collection...")
    for batch_ids, batch_embs, batch_meta in zip(
        chunked(all_ids, 5000),
        chunked(all_embeddings, 5000),
        chunked(all_metadatas, 5000)
    ): # chunking since max batch size for collection.add() is 5461
        card_collection.add(
            ids=batch_ids,
            embeddings=batch_embs,
            metadatas=batch_meta
        )
    print(f"Vector database has been built and saved.")
    print(f"Total cards in collection: {card_collection.count()}")


def load_card_encoder(path, device):
    checkpoint = torch.load(path, weights_only=False)
    model = models.PipelineCPR()
    model.load_state_dict(checkpoint['model_state_dict']) 
    model.to(device)
    model.eval()
    return model


def format_list_for_chroma(data: list) -> str:
    """
    Takes a list of strings and formats it into a single, delimited string safe for ChromaDB metadata
    Commas at the start and end make LIKE queries exact
    Example: ['Flying', 'Vigilance'] -> ",Flying,Vigilance,"
    Example: ['U', 'W'] -> ",U,W,"
    """
    if not data: return ","
    return f",{','.join(data)},"


def build_metadata_map(clean_data_path, card_feature_map):
    """
    Builds a dictionary mapping oracle_id to its clean, filter-ready metadata
    """
    print("Generating cards metadata map... ", end = "")
    card_metadata_map = {}
    with open(clean_data_path, 'r', encoding='utf-8') as f:
        mapped_ids = set(card_feature_map.keys())

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


def process_deck_for_search(deck_id:int, repr_dict_path:str, embedder_checkpoint_path:str, client=None):
    """
    Given a deck id, returns its embedding and its colors
    """
    repr_dict = torch.load(repr_dict_path, weights_only = False)

    deck = edh_scraper.archidekt_fetch_deck(deck_id, known_ids=set(repr_dict.keys()))
    if not deck:
        print(f"Could not fetch or process deck with ID: {deck_id}")
        return None, None

    decklist_ids = deck.commander_ids + deck.mainboard_ids
    assert type(decklist_ids) == list

    repr_decklist = [repr_dict[o_id] for o_id in decklist_ids]

    model = models.PipelineCPR()
    checkpoint = torch.load(embedder_checkpoint_path, weights_only = False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.cpu().eval()
    # Model expects a batch -> add a batch dimension with unsqueeze(0)
    deck_tensor = torch.stack(repr_decklist).unsqueeze(0)
    with torch.no_grad():
        deck_emb_tensor = model.deck_embedding(deck_tensor).squeeze(0)
    deck_emb = deck_emb_tensor.cpu().tolist()

    if client == None:
        print("Client is not available.")
        return None, None
    card_collection = client.get_collection(name="mtg_cards_v1")
    
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

    return deck_emb, list(sorted(deck_colors))


def recommend_cards(deck_embedding:list, n:int, colors=None, client=None):
    """
    Takes a deck embedding and returns the top N recommended cards, optionally filtered by color.
    """
    if client == None:
        print("Client is not available.")
        return []

    card_collection = client.get_collection(name="mtg_cards_v1")

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
            if not subset: # empty subset
                allowed_subsets.append(['C'])
            else:
                allowed_subsets.append(list(subset))
    
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
    feature_map = os.path.join(this, "data", "card_repr_dict_v1.pt")
    embedder_checkpoint_path = os.path.join(this, "models", "cpr_checkpoint_3d.pt")
    cards_metadata = os.path.join(this, "data", "clean_data.json")

    client = chromadb.PersistentClient(path=db_path)
    # build_and_save_chroma_db(feature_map, embedder_checkpoint_path, cards_metadata, client)

    rielle_id = 11032857
    repr_dict_path = os.path.join(this, "data", "card_repr_dict_v1.pt")
    deck_emb, deck_colors = process_deck_for_search(rielle_id, repr_dict_path, embedder_checkpoint_path, client)
    results_1 = recommend_cards(deck_embedding=deck_emb, n=10, client=client)
    results_2 = recommend_cards(deck_embedding=deck_emb, n=10, colors=deck_colors, client=client)
    print(results_1)
    print(results_2)