import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from graph.brain import Brain, EdgeSource
from embedding_index import EmbeddingIndex
from ingestion.ingestor import Ingestor
from observer.observer import Observer

# The union of ALL articles across ALL dimension 2 tests
CORPUS = [
    {"id": "dna",               "title": "DNA"},
    {"id": "thermodynamics",    "title": "Thermodynamics"},
    {"id": "natural_selection", "title": "Natural selection"},
    {"id": "ann",               "title": "Artificial neural network"},
    {"id": "game_theory",       "title": "Game theory"},
    {"id": "genetics",          "title": "Genetics"},
    {"id": "epigenetics",       "title": "Epigenetics"},
    {"id": "mutation",          "title": "Mutation"},
]

def fetch_wikipedia(title: str) -> str:
    import requests
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query", "titles": title,
        "prop": "extracts", "format": "json",
    }
    resp = requests.get(api, params=params, timeout=20,
                        headers={"User-Agent": "AutoScientist-Benchmark/2.0"})
    pages = resp.json().get("query", {}).get("pages", {})
    for page in pages.values():
        return page.get("extract", "")[:8000]
    return ""

def main():
    print("============================================================")
    print("Preparing Shared Brain for Dimension 2 Tests")
    print("============================================================")
    
    shared_dir = os.path.join(ROOT, "benchmark", "dim2", "shared")
    os.makedirs(shared_dir, exist_ok=True)
    
    brain_path = os.path.join(shared_dir, "brain.json")
    index_path = os.path.join(shared_dir, "embedding_index")
    
    brain = Brain()
    emb_index = EmbeddingIndex(dimension=384)
    observer = Observer(brain)
    ingestor = Ingestor(brain, research_agenda=observer, embedding_index=emb_index)
    
    for article in CORPUS:
        print(f"  Ingesting: {article['title']}...")
        text = fetch_wikipedia(article["title"])
        if text:
            ingestor.ingest(text, source=EdgeSource.READING)
            time.sleep(1)
            
    print(f"\nSaving shared brain with {len(brain.all_nodes())} nodes...")
    brain.save(brain_path)
    emb_index.save(index_path)
    print(f"Done. Tests can now be accelerated!")

if __name__ == "__main__":
    main()
