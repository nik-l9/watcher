"""The single resolver from tenant identity to physical storage names.

This module exists so that no other code anywhere constructs a graph name or a
Qdrant collection name. Isolation in Cortex is a property of *which graph you
open*, not of a filter you remembered to add — which only holds if name
construction lives in exactly one place.
"""

from __future__ import annotations

import re
import uuid

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9-]{1,61}[a-z0-9]$")

GRAPH_PREFIX = "cortex_g_"
COLLECTION_PREFIX = "cortex_c_"


class InvalidTenantSlug(ValueError):
    pass


def validate_slug(slug: str) -> str:
    """Reject anything that could be used to escape into another tenant's namespace.

    FalkorDB graph names are addressed as Redis keys and Qdrant collections appear
    in URL paths, so slugs are restricted to a conservative DNS-like alphabet.
    """
    if not isinstance(slug, str) or not _SLUG_RE.match(slug):
        raise InvalidTenantSlug(
            "tenant slug must be 3-63 chars of lowercase letters, digits and hyphens, "
            "and may not start or end with a hyphen"
        )
    return slug


def graph_name_for_new_tenant(slug: str, tenant_id: uuid.UUID) -> str:
    """Assign the physical graph name at tenant creation time.

    The uuid suffix guarantees uniqueness even if a slug is ever recycled, so a
    new tenant can never inherit a deleted tenant's graph.
    """
    validate_slug(slug)
    return f"{GRAPH_PREFIX}{slug}_{tenant_id.hex}"


def collection_name(graph_name: str, kind: str) -> str:
    """Qdrant collection for a given tenant graph and content kind.

    Per-tenant collections mean vector isolation is structural too, rather than
    a payload filter on a shared collection.
    """
    if not graph_name.startswith(GRAPH_PREFIX):
        raise InvalidTenantSlug(f"not a Cortex graph name: {graph_name!r}")
    if not re.fullmatch(r"[a-z0-9_]{1,32}", kind):
        raise InvalidTenantSlug(f"invalid collection kind: {kind!r}")
    return f"{COLLECTION_PREFIX}{graph_name.removeprefix(GRAPH_PREFIX)}_{kind}"
