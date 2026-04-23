"""
EmbeddingIndex — FAISS-backed vector index for fast similarity search.

Replaces O(n²) brute-force cosine scans in Ingestor and Consolidator
with SIMD-optimized inner-product search on normalized embeddings.

Usage:
    index = EmbeddingIndex(dimension=384)
    index.add("node-uuid", embedding_vector)
    results = index.query(query_vector, threshold=0.8, top_k=5)
    pairs = index.all_pairwise_above(threshold=0.88)
"""

import os
import json
import numpy as np

try:
    import faiss
except ImportError:
    raise ImportError(
        "faiss-cpu is required for EmbeddingIndex. "
        "Install with: pip install faiss-cpu"
    )


class EmbeddingIndex:
    """
    Maintains a FAISS inner-product index over node embeddings.

    Embeddings MUST be L2-normalized before adding (cosine similarity
    on normalized vectors = inner product). The shared `embed()` function
    in embedding.py already normalizes, so this is satisfied by default.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._id_to_int: dict[str, int] = {}   # node_id -> faiss int id
        self._int_to_id: dict[int, str] = {}   # faiss int id -> node_id
        self._embeddings: dict[str, np.ndarray] = {}  # node_id -> vector
        self._next_int = 0
        self._index = faiss.IndexFlatIP(dimension)
        self._dirty = False  # True when embeddings updated but index not rebuilt

    @property
    def size(self) -> int:
        return self._index.ntotal

    def has(self, node_id: str) -> bool:
        return node_id in self._id_to_int

    def add(self, node_id: str, embedding: np.ndarray):
        """Add a node embedding to the index. Replaces if already exists."""
        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

        if node_id in self._id_to_int:
            # FAISS IndexFlatIP doesn't support in-place update —
            # store the new embedding and defer rebuild to flush()
            self._embeddings[node_id] = embedding[0]
            self._dirty = True
            return

        int_id = self._next_int
        self._next_int += 1

        self._id_to_int[node_id] = int_id
        self._int_to_id[int_id] = node_id
        self._embeddings[node_id] = embedding[0]
        self._index.add(embedding)

    def remove(self, node_id: str):
        """Remove a node from the index. Defers rebuild to flush()."""
        if node_id not in self._embeddings:
            return
        del self._embeddings[node_id]
        self._dirty = True

    def prune_node_ids(self, node_ids: list[str] | set[str]) -> int:
        """Remove multiple node IDs and flush immediately for consistency."""
        removed = 0
        for node_id in node_ids or []:
            if node_id in self._embeddings:
                del self._embeddings[node_id]
                removed += 1
        if removed:
            self._dirty = True
            self.flush()
        return removed

    def flush(self):
        """Rebuild FAISS index if any deferred updates/removals exist."""
        if self._dirty:
            self._rebuild()
            self._dirty = False

    def _rebuild(self):
        """Rebuild the FAISS index from scratch using stored embeddings."""
        self._index = faiss.IndexFlatIP(self.dimension)
        self._id_to_int.clear()
        self._int_to_id.clear()
        self._next_int = 0

        if not self._embeddings:
            return

        node_ids = list(self._embeddings.keys())
        vectors = np.array(
            [self._embeddings[nid] for nid in node_ids],
            dtype=np.float32
        )

        for i, nid in enumerate(node_ids):
            self._id_to_int[nid] = i
            self._int_to_id[i] = nid

        self._next_int = len(node_ids)
        self._index.add(vectors)

    def get_embedding(self, node_id: str) -> np.ndarray | None:
        """Get the stored embedding for a node."""
        return self._embeddings.get(node_id)

    def query(self, embedding: np.ndarray, threshold: float = 0.5,
              top_k: int = 10) -> list[tuple[str, float]]:
        """
        Find the top-k most similar nodes above the threshold.

        Returns list of (node_id, similarity_score) sorted by score descending.
        """
        if self._index.ntotal == 0:
            return []

        embedding = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
        k = min(top_k, self._index.ntotal)

        scores, indices = self._index.search(embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            if score < threshold:
                continue
            node_id = self._int_to_id.get(int(idx))
            if node_id:
                results.append((node_id, float(score)))

        return results

    def all_pairwise_above(self, threshold: float) -> list[tuple[str, str, float]]:
        """
        Find all pairs of nodes with similarity above threshold.

        Uses batch matrix multiply for efficiency:
          similarities = embeddings @ embeddings.T

        Returns list of (node_id_a, node_id_b, similarity) with a < b
        to avoid duplicates.
        """
        if self._index.ntotal < 2:
            return []

        node_ids = list(self._embeddings.keys())
        matrix = np.array(
            [self._embeddings[nid] for nid in node_ids],
            dtype=np.float32
        )

        # batch cosine similarity via matrix multiply (vectors are normalized)
        sim_matrix = matrix @ matrix.T

        pairs = []
        n = len(node_ids)
        for i in range(n):
            for j in range(i + 1, n):
                if sim_matrix[i, j] >= threshold:
                    pairs.append((
                        node_ids[i], node_ids[j],
                        float(sim_matrix[i, j])
                    ))

        return pairs

    # ── Persistence ──────────────────────────────────────────────────────────

    def save(self, path: str = "data/embedding_index"):
        """Save the index to disk (FAISS index + JSON id map)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".",
                    exist_ok=True)

        faiss.write_index(self._index, path + ".faiss")

        meta = {
            "dimension": self.dimension,
            "id_to_int": self._id_to_int,
            "int_to_id": {str(k): v for k, v in self._int_to_id.items()},
            "next_int": self._next_int,
            "embeddings": {
                nid: emb.tolist()
                for nid, emb in self._embeddings.items()
            }
        }
        with open(path + ".json", "w") as f:
            json.dump(meta, f)

        print(f"EmbeddingIndex saved — {self.size} vectors")

    @classmethod
    def load(cls, path: str = "data/embedding_index") -> "EmbeddingIndex":
        """Load an index from disk."""
        with open(path + ".json", "r") as f:
            meta = json.load(f)

        idx = cls(dimension=meta["dimension"])
        idx._id_to_int = meta["id_to_int"]
        idx._int_to_id = {int(k): v for k, v in meta["int_to_id"].items()}
        idx._next_int = meta["next_int"]
        idx._embeddings = {
            nid: np.array(emb, dtype=np.float32)
            for nid, emb in meta["embeddings"].items()
        }
        idx._index = faiss.read_index(path + ".faiss")

        print(f"EmbeddingIndex loaded — {idx.size} vectors")
        return idx

    @classmethod
    def build_from_brain(cls, brain, embed_fn,
                         dimension: int = 384) -> "EmbeddingIndex":
        """
        Build an index from all existing nodes in a Brain.

        Args:
            brain: Brain instance
            embed_fn: function that takes a string and returns a numpy array
            dimension: embedding dimension (384 for all-MiniLM-L6-v2)
        """
        idx = cls(dimension=dimension)
        nodes = brain.all_nodes()
        if not nodes:
            return idx

        for nid, data in nodes:
            stmt = data.get("statement", "")
            if stmt:
                emb = embed_fn(stmt)
                idx.add(nid, emb)

        print(f"EmbeddingIndex built from brain — {idx.size} vectors")
        return idx
