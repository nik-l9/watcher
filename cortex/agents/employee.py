"""Employee contracts.

The doc's rule is that every employee has a role, goal, tone, tools, memory,
KPIs, escalation policy and output format, and is "never free-form". This module
loads that contract from YAML and validates it, so a specialist is configuration
rather than code — Phase 4's PMM and Sales analysts are new files, not new
modules.

Validation matters more than it looks: an employee naming a tool that does not
exist, or omitting a budget, would fail deep inside a running investigation. It
fails at load instead.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

EMPLOYEE_DIR = Path(__file__).parent / "employees"


class Budget(BaseModel):
    """Bounds on one investigation.

    Every limit is required. A missing budget would mean an unbounded loop, and
    the failure would arrive as an invoice rather than an error.
    """

    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(ge=1, le=200)
    max_tool_calls: int = Field(ge=1, le=500)
    max_tokens: int = Field(ge=1000)
    max_seconds: int = Field(ge=10, le=3600)
    #: A loop calling tools without forming or testing a hypothesis is spinning.
    #: Distinct from max_steps: it stops the spin before the budget is exhausted.
    max_steps_without_new_evidence: int = Field(default=4, ge=1, le=50)


class EscalationRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    condition: str = Field(min_length=1)
    action: str = Field(min_length=1)


class Employee(BaseModel):
    """One specialist's contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: str = Field(min_length=1, pattern=r"^[a-z][a-z0-9_]*$")
    title: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    tools: list[str] = Field(min_length=1)
    budget: Budget
    kpis: list[str] = Field(default_factory=list)
    escalation: list[EscalationRule] = Field(default_factory=list)
    system_prompt: str = Field(min_length=100)

    def validate_against(self, available_tools: set[str]) -> None:
        """Confirm every named tool exists.

        Called with the registry's tool names at startup so a typo surfaces then,
        rather than as a mid-investigation resolution failure.
        """
        missing = sorted(set(self.tools) - available_tools)
        if missing:
            raise ValueError(
                f"employee {self.role!r} names tools that are not registered: "
                f"{', '.join(missing)}; available: {', '.join(sorted(available_tools))}"
            )


class EmployeeNotFound(Exception):
    pass


def load_employee(role: str, *, directory: Path | None = None) -> Employee:
    """Load and validate one employee contract."""
    if not role.replace("_", "").isalnum():
        # The role becomes a filename, so it is validated rather than trusted.
        raise EmployeeNotFound(f"invalid employee role: {role!r}")

    path = (directory or EMPLOYEE_DIR) / f"{role}.yaml"
    if not path.is_file():
        raise EmployeeNotFound(f"no employee contract at {path}")

    raw: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise EmployeeNotFound(f"{path} does not contain a mapping")

    employee = Employee.model_validate(raw)
    if employee.role != role:
        # A file whose declared role disagrees with its name would make the two
        # ways of referring to an employee resolve differently.
        raise EmployeeNotFound(f"{path} declares role {employee.role!r} but is named {role!r}")
    return employee


@lru_cache
def gtm_data_analyst() -> Employee:
    """The V1 employee. Cached — the contract is immutable at runtime."""
    return load_employee("gtm_data_analyst")
