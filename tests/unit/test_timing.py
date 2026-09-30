"""Phase timing.

The numbers here decide what gets optimised, so the failure that matters is a
breakdown that looks plausible and attributes time to the wrong phase.
"""

from __future__ import annotations

import time

from cortex.agents.llm import Usage
from cortex.agents.timing import DRAFT, LOOP_MODEL, LOOP_TOOLS, VERIFY, Phase, Timings


class TestAttribution:
    def test_time_lands_in_the_named_phase(self) -> None:
        timings = Timings()
        with timings.measure(LOOP_MODEL):
            time.sleep(0.02)
        assert timings.phases[LOOP_MODEL].calls == 1
        assert timings.phases[LOOP_MODEL].seconds >= 0.02
        assert LOOP_TOOLS not in timings.phases

    def test_repeated_phases_accumulate(self) -> None:
        timings = Timings()
        for _ in range(3):
            with timings.measure(LOOP_MODEL):
                pass
        assert timings.phases[LOOP_MODEL].calls == 3

    def test_a_failing_phase_still_records_its_cost(self) -> None:
        """A timeout that burned two minutes is exactly the measurement wanted. Skipping
        it on the error path would make a slow failure look free."""
        timings = Timings()
        try:
            with timings.measure(DRAFT):
                time.sleep(0.01)
                raise RuntimeError("provider timed out")
        except RuntimeError:
            pass
        assert timings.phases[DRAFT].seconds >= 0.01

    def test_usage_is_attributed_separately_from_timing(self) -> None:
        """Token counts are only known after a call returns."""
        timings = Timings()
        with timings.measure(DRAFT):
            pass
        timings.record_usage(DRAFT, Usage(input_tokens=100, output_tokens=20))
        assert timings.phases[DRAFT].usage.input_tokens == 100


class TestCacheVisibility:
    def test_no_caching_is_stated_plainly(self) -> None:
        """The line exists so an unimplemented optimisation cannot be mistaken for a
        working one."""
        timings = Timings()
        with timings.measure(LOOP_MODEL):
            pass
        timings.record_usage(LOOP_MODEL, Usage(input_tokens=1000, output_tokens=50))
        assert "prompt caching is not enabled" in timings.render()

    def test_a_cache_hit_rate_is_reported_when_present(self) -> None:
        timings = Timings()
        with timings.measure(LOOP_MODEL):
            pass
        timings.record_usage(
            LOOP_MODEL, Usage(input_tokens=200, output_tokens=50, cache_read_input_tokens=800)
        )
        rendered = timings.render()
        assert "cache hit rate: 80%" in rendered

    def test_hit_rate_is_zero_without_dividing_by_zero(self) -> None:
        assert Usage().cache_hit_rate == 0.0


class TestRendering:
    def test_shares_are_measured_against_the_whole_run(self) -> None:
        """The denominator has to cover every phase.

        A wall figure that stopped when the loop returned left gate and verify outside
        it, and the shares summed past 100% — which is how a breakdown misleads while
        looking precise.
        """
        timings = Timings()
        for phase in (LOOP_MODEL, DRAFT, VERIFY):
            with timings.measure(phase):
                time.sleep(0.01)
        rendered = timings.render()
        assert "unmeasured" not in rendered
        assert timings.measured_seconds <= (time.monotonic() - timings.started) + 0.001

    def test_an_unmeasured_gap_is_named(self) -> None:
        """A large gap means the instrumentation is missing a phase, which is worth
        knowing before drawing a conclusion from the rest."""
        timings = Timings()
        with timings.measure(LOOP_MODEL):
            pass
        assert "unmeasured" in timings.render(wall_seconds=60.0)

    def test_totals_sum_across_phases(self) -> None:
        timings = Timings()
        timings.add(LOOP_MODEL, 1.0, Usage(input_tokens=10, output_tokens=1))
        timings.add(DRAFT, 2.0, Usage(input_tokens=20, output_tokens=2))
        assert timings.total_usage.input_tokens == 30
        assert timings.measured_seconds == 3.0


class TestAFastPhaseIsNotAnAbsentOne:
    """One decimal place put four different things in the same cell.

    The grounding gate resolves citations against a dict in about two milliseconds and
    printed `0.0`. The survey against canned fixture payloads printed `0.0`. Recall with no
    memory configured printed `0.0`. And the sufficiency gate, correctly skipped because the
    report asserted no cause, printed `0.0`.

    A reader sees four zeros in a table of phases and concludes those steps did not run. Two
    of them are the system working exactly as designed, and one of those two is the gate
    every grounding claim in this project rests on.
    """

    def _rendered(self, **phases: float) -> str:
        timings = Timings()
        for name, seconds in phases.items():
            phase = timings.phases.setdefault(name, Phase())
            phase.calls += 1
            phase.seconds += seconds
        return timings.render(97.0)

    def test_a_two_millisecond_phase_is_visible(self) -> None:
        assert "0.002" in self._rendered(gate=0.0021)

    def test_a_phase_that_did_nothing_still_reads_as_zero(self) -> None:
        # A bare zero now means what it says, because the fast phases no longer look like it.
        rendered = self._rendered(sufficiency=0.0)
        assert "0.000" not in rendered

    def test_seconds_keep_one_decimal(self) -> None:
        # The long phases are the ones a reader is budgeting against; three places there
        # would be noise.
        assert "37.0" in self._rendered(draft=37.0)

    def test_fast_and_absent_no_longer_render_alike(self) -> None:
        fast = self._rendered(gate=0.002)
        absent = self._rendered(gate=0.0)
        assert fast != absent
