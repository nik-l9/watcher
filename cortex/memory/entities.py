"""Knowledge graph entity and relationship vocabulary.

Node labels and relationship types are closed enums rather than free strings.
Cortex is a knowledge graph, not a document dump: if the agent could invent
labels at will, traversal queries would silently stop matching and recall would
degrade in ways no test would catch.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class NodeLabel(enum.StrEnum):
    PERSON = "Person"
    PRODUCT = "Product"
    FEATURE = "Feature"
    RELEASE = "Release"
    PR = "PR"
    DEPLOY = "Deploy"
    METRIC = "Metric"
    METRIC_POINT = "MetricPoint"
    EXPERIMENT = "Experiment"
    CAMPAIGN = "Campaign"
    CUSTOMER = "Customer"
    SUPPORT_TICKET = "SupportTicket"
    MEETING = "Meeting"
    DECISION = "Decision"


class RelType(enum.StrEnum):
    # Delivery chain: Campaign -PROMOTES-> Feature -DELIVERED_BY-> PR -SHIPPED_IN-> Release
    PROMOTES = "PROMOTES"
    DELIVERED_BY = "DELIVERED_BY"
    SHIPPED_IN = "SHIPPED_IN"
    DEPLOYED_AS = "DEPLOYED_AS"

    # Causal / observational links the investigator walks.
    AFFECTS = "AFFECTS"
    MEASURED_BY = "MEASURED_BY"
    POINT_OF = "POINT_OF"

    # Commercial chain: Customer -MENTIONS-> Feature, Customer -RAISED-> Ticket
    MENTIONS = "MENTIONS"
    RAISED = "RAISED"
    BELONGS_TO = "BELONGS_TO"

    # Human context.
    AUTHORED = "AUTHORED"
    ATTENDED = "ATTENDED"
    DECIDED = "DECIDED"
    ABOUT = "ABOUT"


@dataclass(slots=True)
class Node:
    """A graph node.

    `key` is the stable natural identifier from the source system (a PR number, a
    HubSpot company id). Upserts are keyed on (label, key) so nightly syncs are
    idempotent.
    """

    label: NodeLabel
    key: str
    props: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class Edge:
    rel: RelType
    from_label: NodeLabel
    from_key: str
    to_label: NodeLabel
    to_key: str
    props: dict[str, Any] = field(default_factory=dict)
