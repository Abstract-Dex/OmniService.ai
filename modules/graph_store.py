"""
Neo4j graph store — ingests parsed UnifiedModel data into Neo4j.

Creates the following node types and relationships:

    (Model) -[:HAS_CLASSIFICATION]-> (Classification)
    (Model) -[:HAS_SPECIFICATION]->  (Specification)
    (Model) -[:HAS_COMPONENT]->      (Component)          # root of sBOM tree
             (Component) -[:HAS_CHILD]-> (Component)       # recursive children
    (Model) -[:HAS_RULE]->           (ConflictRule)
    (Model) -[:HAS_TROUBLESHOOTING]->(Troubleshooting)
    (Component) -[:SHARED_PART]->    (Component)           # cross-model links

Pipeline position:
    extract -> vector_ingest -> llm_parser -> graph_store (this)
"""

import os
from neo4j import GraphDatabase
from dotenv import load_dotenv
from modules.scheme import UnifiedModel

load_dotenv()

# Neo4j connection settings (override via .env if needed)
NEO4J_URI = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "omniservice")


def get_driver():
    """Create and return a Neo4j driver instance."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# ── Pre-check ──

def is_model_in_graph(model_id: str) -> bool:
    """
    Check if a Model node with this model_id already exists in Neo4j.

    Useful for skipping expensive LLM parsing when the graph already
    has the data. Opens and closes its own driver connection.
    """
    driver = get_driver()
    try:
        with driver.session() as session:
            record = session.run(
                "MATCH (m:Model {model_id: $mid}) RETURN count(m) > 0 AS exists",
                mid=model_id
            ).single()
            return record["exists"] if record else False
    finally:
        driver.close()


# ── Ingestion ──

def ingest_model(driver, model: dict) -> bool:
    """
    Ingest a single parsed model (as a plain dict) into Neo4j.

    Creates:
      - Model node
      - Classification node + relationship
      - Specification node + relationship
      - Component tree (recursive sBOM)
      - ConflictRule nodes + relationships
      - Troubleshooting nodes + relationships
    """
    model_id = model["model_id"]

    try:
        with driver.session() as session:

            # Create the root Model node
            session.run(
                "MERGE (m:Model {model_id: $model_id})", model_id=model_id)

            # Classification — system type, location, temp class, mount
            c = model["classification"]
            session.run("""
                MATCH (m:Model {model_id: $model_id})
                MERGE (c:Classification {model_id: $model_id})
                SET c.system_type = $system_type, c.location_type = $location_type,
                    c.temperature_class = $temperature_class, c.mount_type = $mount_type
                MERGE (m)-[:HAS_CLASSIFICATION]->(c)
            """, model_id=model_id, **c)

            # Specification — refrigerant, voltage, phase, frequency, HP
            s = model["specification"]
            session.run("""
                MATCH (m:Model {model_id: $model_id})
                MERGE (sp:Specification {model_id: $model_id})
                SET sp.refrigerant = $refrigerant, sp.voltage = $voltage,
                    sp.phase = $phase, sp.frequency = $frequency, sp.horsepower = $horsepower
                MERGE (m)-[:HAS_SPECIFICATION]->(sp)
            """, model_id=model_id, **s)

            # sBOM component tree — recursively creates all Component nodes
            _create_component_tree(
                session, model_id, None, model["sbom"]["level_0"], is_root=True)

            # Conflict rules — safety / compatibility constraints
            for rule in model.get("conflict_rules", []):
                session.run("""
                    MATCH (m:Model {model_id: $model_id})
                    MERGE (r:ConflictRule {model_id: $model_id, rule_id: $rule_id})
                    SET r.name = $name, r.trigger = $trigger,
                        r.validation = $validation, r.failure_consequence = $failure_consequence
                    MERGE (m)-[:HAS_RULE]->(r)
                """, model_id=model_id, **rule)

            # Troubleshooting — symptom / cause / fix entries
            for ts in model.get("troubleshooting", []):
                session.run("""
                    MATCH (m:Model {model_id: $model_id})
                    MERGE (t:Troubleshooting {model_id: $model_id, symptom: $symptom})
                    SET t.probable_cause = $probable_cause, t.corrective_action = $corrective_action
                    MERGE (m)-[:HAS_TROUBLESHOOTING]->(t)
                """, model_id=model_id, **ts)

        print(f"  [ok] Ingested {model_id}")
        return True

    except Exception as e:
        print(f"  [x] Error ingesting {model_id}: {e}")
        return False


def _create_component_tree(session, model_id, parent_id, component, is_root=False):
    """
    Recursively create Component nodes and parent-child edges.

    Each component stores: level, description, part_number, specification,
    failure_symptom, service_logic.

    If is_root=True, links the component to the Model node via HAS_COMPONENT.
    Otherwise, links it to its parent Component via HAS_CHILD.
    """
    comp_id = component["id"]

    # Create or update the Component node
    session.run("""
        MERGE (c:Component {model_id: $model_id, component_id: $comp_id})
        SET c.level = $level, c.description = $description,
            c.part_number = $part_number, c.specification = $spec,
            c.failure_symptom = $failure_symptom, c.service_logic = $service_logic
    """,
                model_id=model_id, comp_id=comp_id,
                level=component["level"], description=component["description"],
                part_number=component.get("part_number"), spec=component.get("specification"),
                failure_symptom=component.get("failure_symptom"), service_logic=component.get("service_logic")
                )

    # Root component links directly to the Model node
    if is_root:
        session.run("""
            MATCH (m:Model {model_id: $model_id})
            MATCH (c:Component {model_id: $model_id, component_id: $comp_id})
            MERGE (m)-[:HAS_COMPONENT]->(c)
        """, model_id=model_id, comp_id=comp_id)

    # Non-root components link to their parent
    if parent_id is not None:
        session.run("""
            MATCH (p:Component {model_id: $model_id, component_id: $parent_id})
            MATCH (c:Component {model_id: $model_id, component_id: $comp_id})
            MERGE (p)-[:HAS_CHILD]->(c)
        """, model_id=model_id, parent_id=parent_id, comp_id=comp_id)

    # Recurse into children
    for child in component.get("children", []):
        _create_component_tree(session, model_id, comp_id, child)


def create_shared_part_links(driver):
    """
    Link Components that share the same part_number across different models.

    This enables cross-model queries like:
    "Which other models use Norlake #154111?"
    """
    with driver.session() as session:
        result = session.run("""
            MATCH (c1:Component), (c2:Component)
            WHERE c1.part_number = c2.part_number
              AND c1.part_number IS NOT NULL
              AND c1.model_id < c2.model_id
            MERGE (c1)-[:SHARED_PART]->(c2)
            RETURN count(*) as links_created
        """)
        record = result.single()
        count = record["links_created"] if record else 0
        print(f"  [ok] Created {count} SHARED_PART links")


# ── Deletion (for cleanup / re-ingestion) ──

def delete_model(driver, model_id: str):
    """Delete a single model and ALL its related nodes (cascade)."""
    with driver.session() as session:
        session.run(
            "MATCH (n {model_id: $model_id}) DETACH DELETE n", model_id=model_id)
        print(f"  [ok] Deleted {model_id}")


def delete_all(driver):
    """Delete every node and relationship in the database."""
    with driver.session() as session:
        result = session.run(
            "MATCH (n) DETACH DELETE n RETURN count(n) as deleted")
        record = result.single()
        deleted = record["deleted"] if record else 0
        print(f"  [ok] Deleted {deleted} nodes")


# ── Main entry point ──

def ingest_sbom(parsed_manual: UnifiedModel, skip_existing: bool = False) -> bool:
    """
    Ingest a single UnifiedModel (from llm_parser) into Neo4j.

    Steps:
      1. Check connectivity
      2. Skip if model already exists (unless skip_existing=False)
      3. Convert Pydantic model to dict and call ingest_model
      4. Create cross-model SHARED_PART links

    Args:
        parsed_manual: A validated UnifiedModel from llm_parser.parse_manual().
        skip_existing: If True, skip models that are already in Neo4j.

    Returns:
        True on success, False on failure.
    """
    driver = get_driver()

    try:
        driver.verify_connectivity()

        model_id = parsed_manual.model_id
        print(f"[*] {model_id}")

        # Check if model is already in the graph
        if skip_existing:
            with driver.session() as session:
                record = session.run(
                    "MATCH (m:Model {model_id: $mid}) RETURN count(m) > 0 as e", mid=model_id
                ).single()
                exists = record["e"] if record else False
            if exists:
                print(f"  [=] Already exists — skipping")
                return True

        # Convert Pydantic model to plain dict (ingest_model uses dict access)
        success = ingest_model(driver, parsed_manual.model_dump())

        # Link shared parts across all models currently in the graph
        print("Creating SHARED_PART links...")
        create_shared_part_links(driver)

        return success

    finally:
        driver.close()
