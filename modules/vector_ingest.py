"""
Weaviate vector store ingestion.

Takes the raw extracted JSON (from extract.py) and:
  1. Splits it into per-page text inputs + metadata
  2. Generates embeddings via Cohere embed-v4.0
  3. Batch-inserts pages into a Weaviate "service_manuals" collection

Each page becomes one document in Weaviate with its own embedding vector,
enabling hybrid (BM25 + vector) search at query time.

Pipeline position:
    extract -> vector_ingest (this) -> llm_parser -> graph_store
"""

import warnings
import os
import weaviate
from weaviate.classes.config import Configure, Property, DataType
import json
import cohere
from dotenv import load_dotenv

load_dotenv()

# Suppress harmless SSL socket warnings from the Cohere client
warnings.filterwarnings("ignore", category=ResourceWarning)

# Module-level Cohere client (reused across calls, avoids repeated SSL handshakes)
co = cohere.ClientV2()


# ── Shared collection config (imported by search.py) ──

COLLECTION_NAME = "service_manuals"
COLLECTION_PROPERTIES = [
    Property(name="page_number", data_type=DataType.INT),
    Property(name="has_tables", data_type=DataType.BOOL),
    Property(name="raw_text", data_type=DataType.TEXT),
    Property(name="tables", data_type=DataType.TEXT),
    Property(name="model_id", data_type=DataType.TEXT),
]


# ── Weaviate helpers ──

def get_weaviate_client():
    """Connect to Weaviate Cloud when env vars are present, else local Docker.

    Expected env vars for cloud:
      - WEAVIATE_URL (or WEAVIATE_REST_ENDPOINT)
      - WEAVIATE_API_KEY
    """
    weaviate_url = os.getenv("WEAVIATE_URL") or os.getenv("WEAVIATE_REST_ENDPOINT")
    weaviate_api_key = os.getenv("WEAVIATE_API_KEY")

    if weaviate_url:
        if not weaviate_api_key:
            raise ValueError(
                "WEAVIATE_URL is set but WEAVIATE_API_KEY is missing. "
                "Set both for Weaviate Cloud access.",
            )
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=weaviate_url,
            auth_credentials=weaviate_api_key,
        )

    # Local dev fallback
    return weaviate.connect_to_local()


def create_or_get_collection(client, collection_name: str = COLLECTION_NAME):
    """
    Ensure the Weaviate collection exists, creating it if needed.

    Uses the shared COLLECTION_PROPERTIES constant so the schema
    is defined in one place (also imported by search.py).
    """
    if not client.collections.exists(collection_name):
        print(f"  [+] Creating collection '{collection_name}'")
        client.collections.create(
            name=collection_name,
            description="A collection of service manuals",
            inverted_index_config=Configure.inverted_index(
                bm25_b=0.75,
                bm25_k1=1.2,
            ),
            properties=COLLECTION_PROPERTIES,
        )
    return client.collections.get(collection_name)


# ── Data preparation ──

def extract_texts_from_json(page_obj):
    """
    Convert extracted manual JSON into Cohere-compatible text inputs
    and a parallel list of Weaviate metadata dicts.

    Args:
        page_obj: JSON string or dict from extract.py.

    Returns:
        (text_inputs, metadata) — both are lists aligned by index.
        text_inputs: list of dicts for Cohere embed API.
        metadata:    list of dicts for Weaviate properties.
    """
    text_inputs = []
    metadata = []

    # Accept both string and dict input
    manual_json = json.loads(page_obj) if isinstance(
        page_obj, str) else page_obj

    for page in manual_json["pages"]:
        # Format required by Cohere embed-v4.0
        manual_content = [{"type": "text", "text": page["raw_text"]}]

        # Derive model_id from the source file name (e.g. "Model-CPB050JC-S-0-EV")
        metadata.append({
            "model_id": manual_json["source"]["path"].split("/")[-1].split(".")[0],
            "page_number": page["page_number"],
            "has_tables": page["has_tables"],
            "raw_text": page["raw_text"],
            # Weaviate expects TEXT, not list
            "tables": json.dumps(page["tables"]),
        })

        text_inputs.append({"content": manual_content})

    return text_inputs, metadata


# ── Embedding ──

def embed_text(text_inputs):
    """
    Generate float embeddings for a batch of text inputs via Cohere embed-v4.0.

    Uses input_type="search_document" (for ingestion).
    At query time, use input_type="search_query" instead.
    """
    response = co.embed(
        inputs=text_inputs,
        model="embed-v4.0",
        input_type="search_document",
        embedding_types=["float"],
    )
    return response.embeddings.float_


# ── Duplicate check ──

def is_already_ingested(client, collection_name: str, model_id: str) -> bool:
    """Check if any page for this model_id already exists in Weaviate."""
    from weaviate.classes.query import Filter
    collection = client.collections.get(collection_name)
    result = collection.query.fetch_objects(
        filters=Filter.by_property("model_id").equal(model_id),
        limit=1,
    )
    return len(result.objects) > 0


# ── Main ingestion function ──

def ingest_to_weaviate(page_obj) -> bool:
    """
    Full pipeline: extract texts -> embed -> batch insert into Weaviate.

    Skips if the model is already in the collection (idempotent).

    Args:
        page_obj: JSON string or dict from extract.py.

    Returns:
        True on success (or already exists), False on failure.
    """
    # Prepare text inputs for Cohere and metadata for Weaviate
    text_inputs, metadata = extract_texts_from_json(page_obj)
    if not metadata:
        print("  [!] No pages found — skipping")
        return False

    model_id = metadata[0]["model_id"]
    num_pages = len(metadata)

    client = get_weaviate_client()
    try:
        create_or_get_collection(client)

        # Skip if already ingested (avoids duplicate embeddings / API cost)
        if is_already_ingested(client, COLLECTION_NAME, model_id):
            print(f"  [=] Already in Weaviate — skipping")
            return True

        # Generate embeddings via Cohere
        print(f"  [~] Embedding {num_pages} pages via Cohere...")
        vectors = embed_text(text_inputs)
        if not vectors:
            print(f"  [x] Failed to generate embeddings")
            return False

        # Batch insert into Weaviate (one object per page)
        print(f"  [~] Inserting into Weaviate...")
        collection = client.collections.get(COLLECTION_NAME)
        with collection.batch.dynamic() as batch:
            for vector, meta in zip(vectors, metadata):
                batch.add_object(
                    properties=meta,
                    vector=vector,
                )
        print(f"  [ok] Done — {num_pages} pages ingested")

    except Exception as e:
        print(f"  [x] Error: {e}")
        return False
    finally:
        client.close()

    return True
