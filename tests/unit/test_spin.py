"""Detecting a loop that is busy without progressing.

The first test is the one that matters: it is the hole the previous detector had, and it was
found by indexing OpenHands' SDK as a graph and asking what their `StuckDetector` watches for
that ours did not.

Our only signal was `barren_streak` — consecutive steps that called tools and recorded no
evidence. An analyst repeating one identical call writes a *new evidence row every time*, so
the streak reset on each one and the loop ran to its step limit. The budget eventually stopped
it, which made the symptom a slow expensive investigation rather than an error. That is the
worst way to fail: it looks like work.
"""

from __future__ import annotations

from cortex.agents.spin import SpinDetector, Thresholds

MAX_BARREN = 4


class TestTheHoleTheOldDetectorHad:
    def test_the_same_call_repeated_is_caught_even_though_each_one_gathers_evidence(
        self,
    ) -> None:
        detector = SpinDetector()
        for _ in range(3):
            detector.record_call("ga4__get_sessions", {"start_date": "2026-07-01"})
            # Each call succeeds and writes a distinct evidence row, so the old barren
            # counter reset here every single time.
            detector.record_observation(f"hash-{_}")
            detector.record_step(gathered_evidence=True)

        assert detector.barren_streak == 0, "the old signal sees nothing wrong"
        spin = detector.spinning(max_barren_steps=MAX_BARREN)
        assert spin is not None
        assert "repeated the same call" in spin.reason
        assert "ga4__get_sessions" in spin.reason

    def test_the_same_capability_with_different_parameters_is_progress(self) -> None:
        """Re-reading one metric across four periods is exactly what a comparison needs. A
        detector that stopped it would break the product to protect the budget."""
        detector = SpinDetector()
        for day in range(1, 6):
            detector.record_call("ga4__get_sessions", {"start_date": f"2026-07-0{day}"})
            detector.record_observation(f"hash-{day}")
            detector.record_step(gathered_evidence=True)

        assert detector.spinning(max_barren_steps=MAX_BARREN) is None

    def test_parameter_order_does_not_disguise_a_repeat(self) -> None:
        """A dict's insertion order is whatever the model happened to emit, so the same call
        must not look different for having its arguments the other way round."""
        detector = SpinDetector()
        for _ in range(3):
            params = {"end_date": "2026-07-07", "start_date": "2026-07-01"}
            reversed_params = {"start_date": "2026-07-01", "end_date": "2026-07-07"}
            detector.record_call("ga4__get_funnel", params if _ % 2 else reversed_params)
            detector.record_step(gathered_evidence=True)

        assert detector.spinning(max_barren_steps=MAX_BARREN) is not None


class TestRepeatedFailure:
    def test_the_same_failure_three_times_stops_the_loop(self) -> None:
        """A capability failing the same way three times will fail the fourth: the credential
        or the parameters are wrong, not the timing."""
        detector = SpinDetector()
        for _ in range(3):
            detector.record_failure("slack__search_messages", "AuthRejected: invalid_auth")
            detector.record_step(gathered_evidence=False)

        spin = detector.spinning(max_barren_steps=MAX_BARREN)
        assert spin is not None
        assert "repeated the same failure" in spin.reason
        assert "slack__search_messages" in spin.reason

    def test_different_failures_are_not_a_repeat(self) -> None:
        """Three different upstreams failing once each is a bad day, not a stuck loop — and
        the analyst can still route around each one."""
        detector = SpinDetector()
        detector.record_failure("slack__search_messages", "RateLimited: slow down")
        detector.record_failure("github__commits", "UpstreamError: timeout")
        detector.record_failure("hubspot__contacts", "CredentialMissing: not connected")
        detector.record_step(gathered_evidence=False)

        assert detector.spinning(max_barren_steps=MAX_BARREN) is None

    def test_a_varying_request_id_does_not_disguise_one_failure(self) -> None:
        """An upstream that embeds a request id in every message would otherwise look like
        four distinct failures."""
        detector = SpinDetector()
        for n in range(3):
            detector.record_failure(
                "posthog__event_trend",
                f"UpstreamError: 504 from POST /query/\nrequest_id=abc{n}",
            )
        assert detector.spinning(max_barren_steps=MAX_BARREN) is not None


class TestCircling:
    def test_identical_observations_from_different_calls_are_caught(self) -> None:
        """Two different questions reaching byte-identical data means the analyst is
        circling. The hash is already computed and indexed for the grounding gate."""
        detector = SpinDetector()
        for capability in (
            "ga4__get_sessions",
            "ga4__run_report",
            "ga4__top_pages",
            "ga4__get_funnel",
        ):
            detector.record_call(capability, {"start_date": "2026-07-01"})
            detector.record_observation("identical-payload-hash")
            detector.record_step(gathered_evidence=True)

        spin = detector.spinning(max_barren_steps=MAX_BARREN)
        assert spin is not None
        assert "same observation" in spin.reason

    def test_an_absent_hash_is_not_counted(self) -> None:
        """A missing hash is unknown, not identical. Counting empties together would make
        four unhashable observations look like one repeated four times."""
        detector = SpinDetector()
        for _ in range(5):
            detector.record_observation("")
        assert detector.spinning(max_barren_steps=MAX_BARREN) is None


class TestTheOriginalSignalStillWorks:
    def test_consecutive_barren_steps_stall(self) -> None:
        """Kept, because it catches the distinct case of a loop whose calls all error."""
        detector = SpinDetector()
        for n in range(MAX_BARREN):
            detector.record_failure("slack__search_messages", f"error {n}")
            detector.record_step(gathered_evidence=False)

        spin = detector.spinning(max_barren_steps=MAX_BARREN)
        assert spin is not None
        assert "no new evidence" in spin.reason

    def test_gathering_evidence_resets_the_streak(self) -> None:
        detector = SpinDetector()
        detector.record_step(gathered_evidence=False)
        detector.record_step(gathered_evidence=False)
        detector.record_step(gathered_evidence=True)
        assert detector.barren_streak == 0
        assert detector.spinning(max_barren_steps=MAX_BARREN) is None


class TestTheDiagnosisIsSpecific:
    def test_the_most_specific_pattern_is_reported(self) -> None:
        """ "Stopped early" and "stopped early because it asked the same question three times"
        call for different fixes, so the recorded reason is the most useful one available
        rather than whichever fired first."""
        detector = SpinDetector()
        for _ in range(4):
            detector.record_call("ga4__get_sessions", {"start_date": "2026-07-01"})
            detector.record_step(gathered_evidence=False)

        spin = detector.spinning(max_barren_steps=MAX_BARREN)
        assert spin is not None
        # Both patterns have fired; the actionable one wins.
        assert "repeated the same call" in spin.reason

    def test_thresholds_are_configurable(self) -> None:
        """The right number depends on the question: a scenario legitimately re-reading one
        metric across four periods is not spinning."""
        detector = SpinDetector(thresholds=Thresholds(same_call=10))
        for _ in range(5):
            detector.record_call("ga4__get_sessions", {"start_date": "2026-07-01"})
            detector.record_step(gathered_evidence=True)
        assert detector.spinning(max_barren_steps=MAX_BARREN) is None

    def test_a_healthy_investigation_is_never_flagged(self) -> None:
        detector = SpinDetector()
        for n in range(6):
            detector.record_call(f"tool_{n}__capability", {"n": n})
            detector.record_observation(f"hash-{n}")
            detector.record_step(gathered_evidence=True)
        assert detector.spinning(max_barren_steps=MAX_BARREN) is None
