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
from sentence_transformers import SentenceTransformers

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


SEMANTIC_DIM = 768 # The output dim of 'magic-distilbert-base-v1'
ROLE_DIM = 16 # The number of labels in ROLE_LABELS in text_embeddings.py

class CardEmbedder:
    """
    An inference class to handle the creation of full card representations and embeddings.
    It loads all necessary models and data maps once for batched processing.
    """
    def __init__(self, device, cpr_checkpoint_path, partial_map_path, cat_map_path, llm_path, num_types, num_keywords):
        self.device = device
        self.cpr_model = load_card_encoder(cpr_checkpoint_path, num_types, num_keywords, self.device)
        self.partial_map = torch.load(partial_map_path)
        self.cat_map = torch.load(cat_map_path)
        self.all_known_ids = set(self.partial_map.keys())
        self.llm = SentenceTransformer(llm_path, device=self.device)

    def assemble_full_representations(self, oids: list):
        """
        Takes a list of oracle_ids and efficiently assembles their full
        925-dimensional representation tensors in a batch.
        """
        batch_partials = [self.partial_map[oid] for oid in oids]
        batch_types = [self.cat_map[oid]["types"] for oid in oids]
        batch_keywords = [self.cat_map[oid]["keywords"] for oid in oids]

        partials_tensor = torch.stack(batch_partials).to(self.device)
        types_tensor = torch.stack(batch_types).to(self.device)
        keywords_tensor = torch.stack(batch_keywords).to(self.device)

        with torch.no_grad():
            cat_embeddings = self.cpr_model.feature_encoder(types_tensor, keywords_tensor)
        
        full_repr_batch = torch.cat([partials_tensor, cat_embeddings], dim=1)
        return full_repr_batch

    def get_card_embeddings(self, oids: list):
        """
        Gets the final 512-dim embeddings for a list of cards.
        """
        if not oids:
            return torch.tensor([])
        
        full_reprs = self.assemble_full_representations(oids)
        with torch.no_grad():
            embeddings = self.cpr_model.card_embedding(full_reprs)
        return embeddings

    def get_deck_embedding(self, oids: list):
        """
        Gets the final 512-dim embedding for a deck (list of cards).
        """
        if not oids:
            return torch.tensor([])

        full_reprs = self.assemble_full_representations(oids).unsqueeze(0) # Shape: [1, num_cards, 925]
        with torch.no_grad():
            embedding = self.cpr_model.deck_embedding(full_reprs).squeeze(0)
        return embedding

    def get_prompt_dummy_embedding(self, prompt: str, deck_oids: list):
        """
        Creates a 512-dim embedding representing the prompt's intent within the deck's context.
        """
        with torch.no_grad():
            deck_reprs = self.assemble_full_representations(deck_oids)
            dummy_repr = deck_reprs.mean(dim=0)
            
            prompt_semantic_emb = torch.tensor(self.llm.encode(prompt), device=self.device)
            
            # The semantic embedding is at the end of the partial representation, the role embedding is right before it.
            # full_repr = [stats, rarity, color, cmc, semantic, roles, types, keywords]
            partial_dummy = dummy_repr[:-(self.cpr_model.feature_encoder.output_dim)]
            
            # Replace the semantic part of the partial dummy representation
            numerical_and_roles = torch.cat([
                partial_dummy[:-SEMANTIC_DIM], 
                prompt_semantic_emb
            ])

            categorical_dummy = dummy_repr[-(self.cpr_model.feature_encoder.output_dim):]

            dummy_full_repr = torch.cat([
                numerical_and_roles,
                categorical_dummy
            ]).unsqueeze(0)

            prompt_intent_embedding = self.cpr_model.card_embedding(dummy_full_repr).squeeze(0)

        return prompt_intent_embedding


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
    formatted_subsets = []
    for subset in allowed_subsets:
        formatted_string = f",{','.join(subset)},"
        formatted_subsets.append(formatted_string)
    
    or_conditions = [{"color_identity": {"$eq": s}} for s in formatted_subsets]
    return {"$or": or_conditions}


# def create_card_representation(oid, partial_map, cat_map, feature_encoder):
#     """
#     creates and returns a complete card representation for a single card
#     """
#     card_types, card_keyw = cat_map[oid].values()
#     card_types = card_types.unsqueeze(0).cpu()
#     card_keyw = card_keyw.unsqueeze(0).cpu()
#     cat_embeddings = feature_encoder(card_types, card_keyw).squeeze(0)
#     partial_tensor = partial_map[oid].cpu()
#     final_tensor = torch.cat((partial_tensor, cat_embeddings)).cpu()
#     return final_tensor


def process_deck_for_search(deck_id: int, embedder: CardEmbedder, client=None, db_name=None):
    """
    Given a deck id, returns its embedding and its colors using the CardEmbedder class.
    """
    deck = edh_scraper.archidekt_fetch_deck(deck_id, known_ids=embedder.all_known_ids)
    if not deck:
        print(f"Could not fetch or process deck with ID: {deck_id}")
        return None, None, None
    
    decklist_ids = [oid for oid in (deck.commander_ids + deck.mainboard_ids) if oid in embedder.all_known_ids]
    if not decklist_ids:
        print("Deck contains no known cards.")
        return None, None, None

    deck_emb_tensor = embedder.get_deck_embedding(decklist_ids)
    deck_emb_list = deck_emb_tensor.cpu().tolist()
    
    deck_colors = set()
    if client and db_name:
        card_collection = client.get_collection(name=db_name)
        results = card_collection.get(ids=decklist_ids, include=["metadatas"])
        if results and results.get("metadatas"):
            for metadata in results['metadatas']:
                color_str = metadata.get("color_identity", ",")
                card_colors = [color for color in color_str.strip(',').split(',') if color]
                deck_colors.update(card_colors)

    return decklist_ids, deck_emb_list, list(sorted(deck_colors))


def query_db(querry_embedding, n:int, colors=None, client=None, db_name=None):
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
            query_embeddings=[querry_embedding],
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


def recommend_cards_with_prompt(
    deck_id: int,
    n: int,
    embedder: CardEmbedder,
    client: chromadb.Client,
    db_name: str,
    prompt: str  = None,
    alpha: float = 0.7
):
    """
    Main orchestrator for prompt-based recommendations using the CardEmbedder.
    """
    deck_oids, deck_emb_list, deck_colors = process_deck_for_search(deck_id, embedder, client, db_name)
    if deck_emb_list is None:
        print("Failed to process deck, no recommendation is possible.")
        return []

    if prompt and prompt.strip():
        print(f"--- Generating guided recommendations ---")
        print(f"Prompt: '{prompt}' with influence alpha={alpha}")

        prompt_intent_emb = embedder.get_prompt_hypothetical_embedding(prompt, deck_oids)
        deck_emb = torch.tensor(deck_emb_list, device=embedder.device)

        final_query_tensor = F.normalize(
            (1 - alpha) * deck_emb + alpha * prompt_intent_emb,
            p=2, dim=0
        )
        final_query_emb = final_query_tensor.cpu().tolist()

    else:
        print(f"--- Generating synergy recommendations ---")
        final_query_emb = deck_emb_list

    print("Querying the database...")
    return query_db(
        deck_embedding=final_query_emb,
        n=n,
        colors=deck_colors,
        client=client,
        db_name=db_name
    )



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