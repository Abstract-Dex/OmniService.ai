"""
Knowledge base search — queries both Weaviate (vector) and Neo4j (graph)
and returns a single combined JSON payload.

Flow:
  1. Hybrid search (BM25 + vector) on Weaviate, filtered by model_id
  2. Rerank results via Cohere rerank-v4.0-pro to get top 3
  3. Fetch the model's structured subgraph from Neo4j
  4. Combine both into one JSON-safe dict

Usage:
    from modules.search import search_knowledge_base
    result = search_knowledge_base("CPB050JC-S-0-EV", "head pressure dropping in cold weather")
"""

import warnings
import json
import weaviate
import cohere
from weaviate.classes.config import Configure
from weaviate.classes.query import MetadataQuery, Filter
from neo4j import Query
from modules.graph_store import get_driver
from modules.vector_ingest import COLLECTION_NAME, COLLECTION_PROPERTIES
from dotenv import load_dotenv

load_dotenv()

# Suppress harmless SSL socket warnings from the Cohere client
warnings.filterwarnings("ignore", category=ResourceWarning)

# Cohere client (reads COHERE_API_KEY from env)
co = cohere.ClientV2()


# ── Weaviate helpers ──

def _get_weaviate_client():
    """Connect to the local Weaviate instance."""
    return weaviate.connect_to_local()


def _get_or_create_collection(client, collection_name: str = COLLECTION_NAME):
    """Ensure the collection exists and return a handle to it."""
    if not client.collections.exists(collection_name):
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


# ── Embedding ──

def _embed_query(query: str) -> list[float]:
    """Embed a search query using Cohere embed-v4.0 (search_query mode)."""
    response = co.embed(
        inputs=[{"content": [{"type": "text", "text": query}]}],
        model="embed-v4.0",
        input_type="search_query",
        embedding_types=["float"],
    )
    return response.embeddings.float_[0]


# ── Vector search (Weaviate) ──

def _hybrid_search(model_id: str, query: str, limit: int = 8) -> list:
    """
    Run a hybrid (BM25 + vector) search on Weaviate, filtered by model_id.

    Returns a list of Weaviate result objects.
    """
    query_vector = _embed_query(query)

    client = _get_weaviate_client()
    try:
        collection = _get_or_create_collection(client)
        response = collection.query.hybrid(
            query=query,
            vector=query_vector,
            limit=limit,
            alpha=0.5,  # balanced keyword + vector
            filters=Filter.by_property("model_id").equal(model_id),
            return_metadata=MetadataQuery(score=True, explain_score=True),
        )
        return response.objects
    finally:
        client.close()


def _rerank(query: str, objects, top_n: int = 3) -> dict:
    """
    Rerank hybrid search results via Cohere rerank-v4.0-pro.

    Returns a dict with a "results" list containing the top N entries,
    each with: rank, rerank_score, hybrid_score, raw_text, tables.
    """
    if not objects:
        return {"results": []}

    docs = [obj.properties["raw_text"] for obj in objects]

    rerank_response = co.rerank(
        model="rerank-v4.0-pro",
        query=query,
        documents=docs,
        top_n=top_n,
    )

    results = []
    for rank, result in enumerate(rerank_response.results, 1):
        original_obj = objects[result.index]
        props = original_obj.properties
        hybrid_score = original_obj.metadata.score if original_obj.metadata else None

        results.append({
            "rank": rank,
            "rerank_score": float(result.relevance_score),
            "hybrid_score": float(hybrid_score) if hybrid_score is not None else None,
            "raw_text": props.get("raw_text", ""),
            "tables": props.get("tables"),
        })

    return {"results": results}


# ── Graph search (Neo4j) ──

def _fetch_model_graph(model_id: str, max_hops: int = 3) -> dict:
    """
    Fetch the full subgraph for a model from Neo4j.

    Returns a JSON-safe dict with:
      - model_id
      - nodes: list of {id, labels, properties}
    """
    # Build the Cypher query with the hop depth baked in
    # (max_hops is an int we control, not user input — safe to interpolate)
    cypher = Query(
        "MATCH p=(m:Model {model_id: $model_id})-[*0.."
        + str(max_hops)
        + "]-(n) RETURN p"
    )

    driver = get_driver()
    try:
        with driver.session() as session:
            result = session.run(cypher, model_id=model_id)

            node_map = {}
            for record in result:
                path = record["p"]
                for node in path.nodes:
                    nid = node.element_id
                    if nid not in node_map:
                        node_map[nid] = {
                            "id": nid,
                            "labels": sorted(list(node.labels)),
                            "properties": dict(node.items()),
                        }

            return {
                "model_id": model_id,
                "nodes": list(node_map.values()),
            }
    finally:
        driver.close()


# ── Combined search ──

def search_knowledge_base(
    model_id: str,
    problem_description: str,
    vector_limit: int = 8,
    rerank_top_n: int = 3,
) -> dict:
    """
    Query both Weaviate and Neo4j for a given model and problem description.

    Steps:
      1. Hybrid search on Weaviate (filtered by model_id)
      2. Rerank to get the top N most relevant pages
      3. Fetch the model's structured graph from Neo4j
      4. Combine into a single JSON-safe payload

    Args:
        model_id:            The model to search for (e.g. "CPB050JC-S-0-EV").
        problem_description: The technician's problem / question in plain text.
        vector_limit:        How many results to fetch from Weaviate before reranking.
        rerank_top_n:        How many results to keep after reranking.

    Returns:
        A dict with the structure:
        {
            "model_id": "...",
            "query": "...",
            "vector_results": { "results": [...] },
            "graph_context": { "model_id": "...", "nodes": [...] }
        }
    """
    # Vector DB: hybrid search + rerank
    objects = _hybrid_search(model_id, problem_description, limit=vector_limit)
    vector_results = _rerank(problem_description, objects, top_n=rerank_top_n)

    # Graph DB: structured model data
    graph_context = _fetch_model_graph(model_id)

    # Combine into one payload
    payload = {
        "model_id": model_id,
        "query": problem_description,
        "vector_results": vector_results,
        "graph_context": graph_context,
    }

    return payload


# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(description="Search the knowledge base")
#     parser.add_argument("--model", "-m", type=str, required=True,
#                         help="Model ID to search for")
#     parser.add_argument("--query", "-q", type=str, required=True,
#                         help="Problem description / search query")
#     args = parser.parse_args()

#     result = search_knowledge_base(args.model, args.query)
#     print(json.dumps(result, ensure_ascii=False, indent=2))
