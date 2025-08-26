import chromadb
import torch
from tqdm import tqdm
import models
import ijson
import os
import re


def build_and_save_chroma_db(card_feature_map_path, card_encoder_checkpoint, cards_metadata_path, db_path):
    """
    Loads card features, generates embeddings using a trained model and saves them into a persistent ChromaDB database
    """
    print("--- Building the Card Vector Database ---")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.exists(card_feature_map_path):
        print(f"Card feature map not found at '{card_feature_map_path}'"); return
    card_feature_map = torch.load(card_feature_map_path)

    card_encoder = load_card_encoder(card_encoder_checkpoint, device)
    
    print(f"Initializing persistent ChromaDB client at: {db_path}")
    client = chromadb.PersistentClient(path=db_path)
    
    card_collection = client.get_or_create_collection(name="mtg_cards_v1")
    
    print("Generating embeddings for all cards...")
    all_ids = []
    all_embeddings = []
    all_metadatas = [] 

    card_metadata_map = {}
    with open(cards_metadata, 'r', encoding='utf-8') as f:
        mapped_ids = set(card_feature_map.keys())
        for card in ijson.items(f, "item"):
            oracle_id = card.get("oracle_id")    
            if oracle_id and oracle_id in mapped_ids:
                color_identity = card.get("color_identity")
                if not color_identity:
                    color_identity = ["C"]

                mana_cost_str = card.get("mana_cost", "")
                mana_details = parse_mana_cost(mana_cost_str)

                card_metadata_map[oracle_id] = {
                "name": card.get("name", ""),
                "color_identity": card.get("color_identity", []),
                 "cmc": mana_details['cmc'],
                "generic_cost": mana_details['generic'],
                "x_cost": mana_details['x_cost'],
                "type_line": card.get("type_line", ""),
                "keywords": card.get("keywords", []),
                "oracle_text": card.get("oracle_text", "")
                }

    with torch.no_grad():
        for oracle_id, feature_tensor in tqdm(card_feature_map.items(), desc="Encoding cards"):
            # Model expects a batch -> add a batch dimension with unsqueeze(0)
            feature_tensor = feature_tensor.unsqueeze(0).to(device)
            embedding = card_encoder.card_embedding(feature_tensor).squeeze(0).cpu().tolist()
            all_ids.append(oracle_id)
            all_embeddings.append(embedding)
            metadata = card_metadata_map.get(oracle_id, {})
            all_metadatas.append(metadata)

    print(f"Adding {len(all_ids)} cards to the ChromaDB collection...")
    card_collection.add(
        ids=all_ids,
        embeddings=all_embeddings,
        metadatas=all_metadatas
    )

    print(f"Vector database has been built and saved to the '{db_path}' directory.")
    print(f"Total cards in collection: {card_collection.count()}")


def load_card_encoder(path, device):
    checkpoint = torch.load(path, weights_only=False)
    model = models.PipelineCPR()
    model.load_state_dict(checkpoint['model_state_dict']) 
    model.to(device)
    model.eval()
    return model


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


def recommend_cards(deck_embedding, n=10, colors=None):
    """
    Takes a deck embedding and returns the top N recommended cards, optionally filtered by color.
    """
    filter_metadata = {}
    if colors:
        # Example for a multi-color filter: find cards that are blue OR green
        # filter_metadata = {"color_identity": {"$in": ["U", "G"]}}
        filter_metadata = {"color_identity": colors}

    results = card_collection.query(
        query_embeddings=[deck_embedding],
        n_results=n,
        where=filter_metadata if filter_metadata else None
    )
    return results


if __name__ == "__main__":
    this = os.path.dirname(__file__)
    feature_map = os.path.join(this, "data", "card_repr_dict_v1.pt")
    card_encoder_checkpoint = os.path.join(this, "models", "cpr_checkpoint_3d.pt")
    cards_metadata = os.path.join(this, "data", "clean_data.json")
    db_path = os.path.join(this, "card_db")


    build_and_save_chroma_db(feature_map, card_encoder_checkpoint, cards_metadata, db_path)
