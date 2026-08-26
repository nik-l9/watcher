"""Detecting a loop that is busy but not progressing.

A loop can be spending tokens and making calls while getting nowhere, and the shapes that
indicate it are recognisable: the same action repeated, the same error repeated, the same
observation coming back, and a run producing nothing new at all.

**Why Cortex needs it.** Our only stall signal was `barren_streak`: consecutive steps that
called tools and produced no evidence. That has a hole big enough to drive an investigation
through. An analyst that calls the *same capability with the same parameters* four times in a
row writes a new evidence row each time, so the streak resets on every one of them and the
loop runs to its step limit doing nothing. The budget eventually stops it, which means the
symptom is a slow expensive investigation rather than an error — the worst way to fail,
because it looks like work.

**What we watch.** The signals are shaped by evidence gathering rather than by editing code,
which is what makes them specific to this loop -- a monologue matters when an agent talks
instead of acting, and neither that nor an alternating edit pattern says anything here:

  - **The same call repeated.** `(capability, parameters)` identical to a recent call. The
    pattern the barren streak missed, and the most common way a loop stalls.
  - **The same observation returned.** Identical `payload_hash` seen repeatedly, even from
    different calls. Two different queries returning byte-identical results means the analyst
    is circling, and the hash is already computed and indexed for the grounding gate.
  - **Repeated failure of the same call.** A capability failing the same way three times will
    fail the fourth; the credential is wrong or the parameters are.
  - **No new evidence at all.** The original `barren_streak`, kept because it catches the
    distinct case of a loop calling tools that all error.

Deliberately *not* borrowed: monologue and alternating-pattern detection. A GTM analyst
thinking for two turns without a tool call is reasoning, not stalling, and our loop already
treats a turn with no tool request as a decision to conclude.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class Thresholds:
    """How many repetitions before a pattern counts as spinning.

    Configurable, as theirs are, because the right number depends on the question: a
    scenario that legitimately re-reads one metric across four periods is not spinning, and a
    threshold of two would stop it.
    """

    #: Identical (capability, params) calls before the loop is spinning. Three, not four:
    #: the second identical call is already suspicious and the third has no defence.
    same_call: int = 3
    #: Identical observation payloads before the loop is circling.
    same_observation: int = 4
    #: Repeated identical failures of one capability.
    same_failure: int = 3


@dataclass(frozen=True, slots=True)
class Spin:
    """A detected non-progression, with the sentence to record."""

    pattern: str
    detail: str

    @property
    def reason(self) -> str:
        return f"{self.pattern}: {self.detail}"


@dataclass(slots=True)
class SpinDetector:
    """Watches one investigation for patterns that mean it is not progressing.

    Fed by the loop as it goes rather than reconstructed from the audit trail afterwards,
    because the point is to stop early — a detector that can only explain the waste after the
    budget is spent has not saved anything.
    """

    thresholds: Thresholds = field(default_factory=Thresholds)
    _calls: dict[str, int] = field(default_factory=dict)
    _observations: dict[str, int] = field(default_factory=dict)
    _failures: dict[str, int] = field(default_factory=dict)
    #: Consecutive steps that gathered nothing. The original signal, kept.
    barren_streak: int = 0

    def record_call(self, capability: str, params: dict[str, Any]) -> None:
        self._calls[_signature(capability, params)] = (
            self._calls.get(_signature(capability, params), 0) + 1
        )

    def record_observation(self, payload_hash: str) -> None:
        if payload_hash:
            self._observations[payload_hash] = self._observations.get(payload_hash, 0) + 1

    def record_failure(self, capability: str, error: str) -> None:
        # Keyed on the error's *type and shape* rather than its full text: an upstream that
        # embeds a request id in every message would otherwise look like four different
        # failures. The first line is enough to tell them apart.
        key = f"{capability}|{error.splitlines()[0][:120] if error else ''}"
        self._failures[key] = self._failures.get(key, 0) + 1

    def record_step(self, *, gathered_evidence: bool) -> None:
        self.barren_streak = 0 if gathered_evidence else self.barren_streak + 1

    def spinning(self, *, max_barren_steps: int) -> Spin | None:
        """The first pattern that has crossed its threshold, or None.

        Checked in order of how specific the diagnosis is, so the recorded reason is the most
        useful one available rather than whichever fired first.
        """
        for signature, count in self._calls.items():
            if count >= self.thresholds.same_call:
                capability, params = signature.split("|", 1)
                return Spin(
                    "repeated the same call",
                    f"{capability} called {count} times with identical parameters "
                    f"({params[:160]}); the answer will not change",
                )

        for key, count in self._failures.items():
            if count >= self.thresholds.same_failure:
                capability, _, error = key.partition("|")
                return Spin(
                    "repeated the same failure",
                    f"{capability} failed {count} times with the same error ({error[:120]}); "
                    f"the credential or the parameters are wrong, not the timing",
                )

        for _, count in self._observations.items():
            if count >= self.thresholds.same_observation:
                return Spin(
                    "kept receiving the same observation",
                    f"{count} calls returned byte-identical results; different questions "
                    f"reaching the same data means the investigation is circling",
                )

        if self.barren_streak >= max_barren_steps:
            return Spin(
                "gathered no new evidence",
                f"{self.barren_streak} consecutive steps called tools and recorded nothing",
            )
        return None


def _signature(capability: str, params: dict[str, Any]) -> str:
    """A stable key for one call.

    Sorted keys, because the same call with its parameters in a different order is the same
    call — and a dict's insertion order is whatever the model happened to emit.
    """
    try:
        rendered = json.dumps(params, sort_keys=True, default=str)
    except (TypeError, ValueError):
        rendered = repr(sorted(params.items()))
    return f"{capability}|{rendered}"
