"""Phase timing.

The numbers here decide what gets optimised, so the failure that matters is a
breakdown that looks plausible and attributes time to the wrong phase.
"""

from __future__ import annotations

import time

from cortex.agents.llm import Usage
from cortex.agents.timing import DRAFT, LOOP_MODEL, LOOP_TOOLS, VERIFY, Timings


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
