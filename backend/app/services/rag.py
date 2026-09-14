"""
CropGuard AI — RAG (Retrieval-Augmented Generation) Service
============================================================
Provides a ChromaDB-backed vector retrieval layer over the CropGuard
agricultural knowledge base.

How it works:
1. On startup, all KNOWLEDGE_BASE entries are embedded using Google's
   text-embedding model and stored in a persistent ChromaDB collection.
2. At query time, the disease name + crop + observed symptoms are embedded
   and the top-K most semantically similar knowledge entries are retrieved.
3. Retrieved context is returned as a structured dict and injected into
   the Gemini/Groq prompt to ground the AI diagnosis in verified
   agricultural knowledge.

This is a pure enhancement layer — if ChromaDB is unavailable for any
reason (import error, first-run seeding failure) the system falls back
transparently to the original KNOWLEDGE_BASE exact-key lookup so
accuracy is NEVER degraded.
"""

from __future__ import annotations

import json
import os
import hashlib
import threading
from typing import Optional

# ── Lazy ChromaDB import (keeps startup fast, gracefully degrades) ─────────
_chroma_available = False
try:
    import chromadb
    from chromadb.config import Settings as ChromaSettings
    _chroma_available = True
except ImportError:
    pass

from app.services.knowledge_base import KNOWLEDGE_BASE

# ── Constants ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "cropguard_knowledge"
CHROMA_PERSIST_DIR = os.path.join(os.path.dirname(__file__), ".chroma_db")
TOP_K = 3          # number of similar entries to retrieve
_SEED_VERSION = "v1"   # bump this to force re-seed when KB changes

# ── Module-level singletons ───────────────────────────────────────────────
_client: Optional[object] = None
_collection: Optional[object] = None
_lock = threading.Lock()
_seeded = False


# ─────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────

def _kb_fingerprint() -> str:
    """Hash the disease keys in KNOWLEDGE_BASE so we detect when KB changes."""
    keys = sorted(KNOWLEDGE_BASE.keys())
    return hashlib.md5(json.dumps(keys).encode()).hexdigest()[:8]


def _build_document(disease: str, data: dict) -> str:
    """
    Convert a KNOWLEDGE_BASE entry into a plain-text document for embedding.
    Uses English content only for the vector representation.
    """
    en = data.get("en", {})
    parts = [f"Disease: {disease}"]
    if en.get("symptoms"):
        parts.append("Symptoms: " + "; ".join(en["symptoms"]))
    if en.get("causes"):
        parts.append("Causes: " + "; ".join(en["causes"]))
    if en.get("treatments"):
        parts.append("Treatments: " + "; ".join(en["treatments"]))
    if en.get("prevention"):
        parts.append("Prevention: " + "; ".join(en["prevention"]))
    return "\n".join(parts)


def _get_collection():
    """Lazy-initialise ChromaDB client + collection (thread-safe)."""
    global _client, _collection, _seeded

    if not _chroma_available:
        return None

    with _lock:
        if _collection is not None:
            return _collection

        try:
            os.makedirs(CHROMA_PERSIST_DIR, exist_ok=True)
            _client = chromadb.PersistentClient(
                path=CHROMA_PERSIST_DIR,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            _collection = _client.get_or_create_collection(
                name=COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
            )
            _seed_collection(_collection)
            return _collection
        except Exception as exc:
            print(f"[RAG] ChromaDB init failed (will use fallback): {exc}")
            return None


def _seed_collection(collection) -> None:
    """
    Populate ChromaDB with KNOWLEDGE_BASE embeddings if not already seeded
    or if the knowledge base has changed since last seed.
    """
    global _seeded

    fingerprint = _kb_fingerprint()
    seed_marker = f"{_SEED_VERSION}:{fingerprint}"

    # Check if already seeded with the current KB version
    try:
        existing = collection.get(ids=["__seed_marker__"])
        if existing["documents"] and existing["documents"][0] == seed_marker:
            print("[RAG] ChromaDB already seeded — skipping.")
            _seeded = True
            return
    except Exception:
        pass

    print(f"[RAG] Seeding ChromaDB with {len(KNOWLEDGE_BASE)} knowledge entries…")

    ids, documents, metadatas = [], [], []

    for disease, data in KNOWLEDGE_BASE.items():
        if disease == "Healthy":
            continue  # Healthy is not a disease entry to retrieve against
        doc = _build_document(disease, data)
        en = data.get("en", {})
        ids.append(disease)
        documents.append(doc)
        metadatas.append({
            "disease": disease,
            "symptoms": "; ".join(en.get("symptoms", [])),
            "causes": "; ".join(en.get("causes", [])),
        })

    # Batch upsert (ChromaDB default embedding function — no extra API key needed)
    BATCH = 50
    for i in range(0, len(ids), BATCH):
        collection.upsert(
            ids=ids[i : i + BATCH],
            documents=documents[i : i + BATCH],
            metadatas=metadatas[i : i + BATCH],
        )

    # Write seed marker
    collection.upsert(ids=["__seed_marker__"], documents=[seed_marker], metadatas=[{}])
    _seeded = True
    print(f"[RAG] ChromaDB seeded successfully — {len(ids)} entries indexed.")


# ─────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────

def retrieve_knowledge(disease_name: str, crop: str, symptoms: list[str] | None = None) -> dict:
    """
    Retrieve the most semantically similar agricultural knowledge entries
    for a given disease + crop + observed symptoms.

    Returns a dict with keys:
        retrieved      – list of matched disease names
        context        – concatenated context string to inject into prompts
        source         – "chromadb" | "fallback"
        top_match      – the single closest disease name (or None)

    Falls back to exact-key KNOWLEDGE_BASE lookup if ChromaDB is unavailable.
    """
    query_parts = [f"Crop: {crop}", f"Disease: {disease_name}"]
    if symptoms:
        query_parts.append("Symptoms: " + "; ".join(symptoms))
    query_text = "\n".join(query_parts)

    # ── Try ChromaDB path ────────────────────────────────────────────────
    collection = _get_collection()
    if collection is not None:
        try:
            results = collection.query(
                query_texts=[query_text],
                n_results=min(TOP_K, collection.count() - 1),  # -1 for seed marker
                where={"disease": {"$ne": ""}},               # exclude seed marker
            )
            if results and results["ids"] and results["ids"][0]:
                matched_ids = results["ids"][0]
                matched_docs = results["documents"][0]
                context = "\n\n---\n\n".join(matched_docs)
                return {
                    "retrieved": matched_ids,
                    "context": context,
                    "source": "chromadb",
                    "top_match": matched_ids[0] if matched_ids else None,
                }
        except Exception as exc:
            print(f"[RAG] ChromaDB query error (using fallback): {exc}")

    # ── Fallback: exact key lookup ────────────────────────────────────────
    return _fallback_lookup(disease_name)


def _fallback_lookup(disease_name: str) -> dict:
    """Original exact-key lookup — zero accuracy loss over previous behaviour."""
    cleaned = disease_name.strip().lower()
    matched_key = None

    # Exact match first
    for key in KNOWLEDGE_BASE:
        if key.lower() == cleaned:
            matched_key = key
            break

    # Partial match
    if matched_key is None:
        for key in KNOWLEDGE_BASE:
            if key != "Healthy" and (key.lower() in cleaned or cleaned in key.lower()):
                matched_key = key
                break

    entry = KNOWLEDGE_BASE.get(matched_key or "", KNOWLEDGE_BASE.get("Healthy", {}))
    en = entry.get("en", {})
    doc = _build_document(matched_key or disease_name, entry) if matched_key else ""

    return {
        "retrieved": [matched_key] if matched_key else [],
        "context": doc,
        "source": "fallback",
        "top_match": matched_key,
    }


def warm_up() -> None:
    """
    Pre-warm the ChromaDB collection at startup (non-blocking).
    Call this from the FastAPI lifespan handler.
    """
    if not _chroma_available:
        print("[RAG] chromadb not installed — vector retrieval disabled (using fallback).")
        return
    import threading
    t = threading.Thread(target=_get_collection, daemon=True, name="rag-warmup")
    t.start()
