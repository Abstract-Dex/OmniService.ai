"""
Pydantic schemas for structured manual data.

These models define the shape of the LLM-parsed output.
The LLM (Claude Haiku) reads raw extracted text and returns JSON
that is validated against these schemas before being stored in Neo4j.

Hierarchy (sBOM):
    UnifiedModel
    ├── Classification   — cooler/freezer, indoor/outdoor, temp class, mount
    ├── Specification    — refrigerant, voltage, phase, frequency, HP
    ├── SBOM             — recursive component tree (levels 0-3)
    │   └── SBOMComponent (self-referencing via `children`)
    ├── ConflictRules[]  — safety / compatibility rules between parts
    └── Troubleshooting[]— symptom -> cause -> fix mappings
"""

from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Optional


# ── Enum types for classification fields ──

class SystemType(str, Enum):
    """Whether the unit is a cooler or freezer."""
    Cooler = "Cooler"
    Freezer = "Freezer"


class LocationType(str, Enum):
    """Indoor vs outdoor installation."""
    Indoor = "Indoor"
    Outdoor = "Outdoor"


class TemperatureClass(str, Enum):
    """Operating temperature range category."""
    Low = "Low"
    Medium = "Medium"
    High = "High"


class MountType(str, Enum):
    """Physical mounting style of the unit."""
    Floor = "Floor"
    Wall = "Wall"
    Ceiling = "Ceiling"


# ── Core data models ──

class Classification(BaseModel):
    """High-level categorisation of a refrigeration model."""
    system_type: SystemType
    location_type: LocationType
    temperature_class: TemperatureClass
    mount_type: MountType


class Specification(BaseModel):
    """Electrical and refrigerant specs for the unit."""
    refrigerant: str
    voltage: str
    phase: str
    frequency: str
    horsepower: str


class SBOMComponent(BaseModel):
    """
    A single node in the structured Bill of Materials tree.

    Levels:
        0 — Complete system assembly (root)
        1 — Major assemblies (Compressor, Motor, Controller)
        2 — Sub-components (Capacitor, Relay, Probe)
        3 — Individual parts (rarely used)

    `children` makes this model self-referencing so we can
    represent the full sBOM hierarchy recursively.
    """
    level: int = Field(ge=0, le=3)
    id: str
    description: str
    part_number: Optional[str] = None
    specification: Optional[str] = None
    failure_symptom: Optional[str] = None
    service_logic: Optional[str] = None
    children: List["SBOMComponent"] = []


class SBOM(BaseModel):
    """Wrapper around the root (level-0) component of the sBOM tree."""
    level_0: SBOMComponent


class ConflictRules(BaseModel):
    """
    A safety / compatibility rule for a model.
    E.g. "Do not use R449A components in an R290 system."
    """
    rule_id: str
    name: str
    trigger: str          # condition that activates the rule
    validation: str       # how to verify compliance
    failure_consequence: str  # what happens if violated


class Troubleshooting(BaseModel):
    """Symptom -> probable cause -> corrective action mapping."""
    symptom: str
    probable_cause: str
    corrective_action: str


class UnifiedModel(BaseModel):
    """
    Top-level schema for a single refrigeration model manual.

    This is what the LLM returns after parsing the raw extracted text.
    It is ingested into Neo4j (graph_store) as nodes and relationships.
    """
    model_id: str
    classification: Classification
    specification: Specification
    sbom: SBOM
    conflict_rules: List[ConflictRules]
    troubleshooting: List[Troubleshooting]
