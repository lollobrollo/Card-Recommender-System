# Contextual Preference Ranking for Card Recommendations in Magic: The Gathering

[![Python Version](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow)](https://huggingface.co/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete source code for the Bachelor's Thesis "Contextual Preference Ranking for Card Recommendations in Magic: The Gathering." The project develops a novel recommender system for the Commander (EDH) format, leveraging deep learning to understand and predict synergistic card relationships.

---

## About the Project

This project aims to create a recommender system for MTG cards in EDH applying a **Contextual Preference Ranking (CPR)** framework. The system learns the synergistic relationships between cards in the context of a partially built deck, allowing it to make relevant suggestions.

The repository contains a complete, end-to-end implementation, covering:
1.  Large-scale data collection via web scraping and public APIs.
2.  A three-stage training pipeline for an expert MTG language model.
3.  Training of a two-tower Siamese network to learn card-deck synergies.
4.  Deployment of the final models in a searchable vector database with a web interface for evaluation.

## Key Features

- **Robust Data Pipeline:** Features restartable scrapers and a structured, multi-stage data processing workflow.
- **Synergy-Aware Recommendations:** Suggests cards that fit the specific strategy of a given deck, not just popular staples.
- **Multi-Model Backend:** Supports multiple CPR models trained with varying datasets, loss functions, and durations for comparative analysis.
- **Interactive Evaluation UI:** A Gradio-based web demo for real-time interaction and human feedback collection.


## Usage: The Full Pipeline

The project is structured as a sequential pipeline. The scripts should be run in order to progress from raw data to a functional demo. The `complete_pipeline.ipynb` notebook provides a comprehensive, step-by-step guide for running this entire workflow.

**Warning:** Many of these steps, especially data retrieval and model training, are computationally expensive and/or long-running.

1.  **Stage 1: Create the Text Corpus (`article_scraper.py`)**
    - Scrapes thousands of MTG articles to create a corpus for language model training. This process is restartable.

2.  **Stage 2: Train Language Models (`text_embeddings.py`)**
    - Executes the three-stage pipeline to produce the domain-adapted foundation model and the final card role classifier. This is a highly GPU-intensive process.

3.  **Stage 3: Process Card Data (`preprocess_cards.py`)**
    - Downloads the latest card data from Scryfall, filters it, downloads images, and creates the multi-modal card representations.

4.  **Stage 4: Scrape Decklists & Create Datasets (`edh_scraper.py`)**
    - Scrapes tens of thousands of decklists from Archidekt and processes them into the final training datasets (`.pt` files).

5.  **Stage 5: Train CPR Models (`train.py`)**
    - Trains the eight different CPR model variants and generates the final card embedding dictionaries for each.

6.  **Stage 6: Build the Vector Database (`vector_database.py`)**
    - Populates the ChromaDB vector database with the embeddings and metadata for all cards, creating a separate collection for each of the eight models.

7.  **Stage 7: Launch the Demo (`app.py`)**
    - Starts the Gradio web server, providing an interactive UI to query the models and collect feedback.

## Project Structure

The repository is organized as follows:
```
├── data/                               # Stores raw and processed data files (card data, decks, etc.)
├── models/                             # Stores trained model checkpoints
├── misc/                               # Miscellaneous files like feedback logs and plots
├── article_scraper.py                  # Script to scrape the text corpus for Stage 2
├── text_embeddings.py                  # 3-stage training pipeline for language models
├── preprocess_cards.py                 # Script for downloading and processing card data
├── edh_scraper.py                      # Script for scraping decklists and creating training datasets
├── train.py                            # Main training script for the CPR models
├── vector_database.py                  # Core logic for the retriever and database interaction
├── app.py                              # The Gradio web application for the interactive demo
├── search_results.py                   # Script for local testing and feedback visualization
├── models.py                           # Defines the PyTorch model architectures
├── utils.py                            # Utility functions used across the project
├── CPR_for_card_recommendations.pdf    # The PDF file containing the thesis
└── complete_pipeline.ipynb             # Jupyter Notebook demonstrating the entire workflow

```

## Technical Methodology

The core of this project is the **Contextual Preference Ranking (CPR)** framework, which learns to rank cards based on their synergy with a given deck (context). This is achieved with a **two-tower Siamese network** trained via contrastive learning (Triplet Margin Loss and InfoNCE Loss).

The card representations are multi-modal, combining numerical stats, categorical features, and two powerful text embeddings derived from a **three-stage domain adaptation pipeline** for a `distilbert-base-uncased` model.

The final system uses a **vector offset method** for prompt-guided search and a **multi-alpha pooling** strategy combined with **Maximal Marginal Relevance (MMR)** to ensure diverse and relevant recommendations.

## License

This project is licensed under the [**Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License (CC BY-NC-SA 4.0)**](https://creativecommons.org/licenses/by-nc-sa/4.0/).

## Acknowledgments

- This work is inspired by the foundational research on Contextual Preference Ranking by Timo Bertram et al.
- Card data is sourced from the invaluable [Scryfall API](https://scryfall.com/docs/api).
- Deck data is sourced from [Archidekt](https://archidekt.com/).
- Article text is sourced from [EDHREC](https://edhrec.com/).
- This project was made possible by open-source libraries including PyTorch, Hugging Face Transformers, Gradio, and ChromaDB.