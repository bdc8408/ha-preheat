"""Feedback listener tests (fork): Ben's Yes/No notification responses
adjust the learner.

The notification automations fire `preheat_feedback` with
{type: 'home'|'bedtime'|'away', response: 'yes'|'no'}. A 'no' means the
system's call was wrong and must not be reinforced; a 'yes' confirms
it. Malformed events are ignored and never break the update cycle.
"""
import sys
import os
import types
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import unittest

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_components.preheat.schedule_learner import ScheduleLearner
from custom_components.preheat.units import fahrenheit_to_celsius
from custom_components.preheat.const import (
    CONF_HEATING_PROFILE,
    PROFILE_RADIATOR_NEW,
)


# ---------------------------------------------------------------------------
# Synthetic household helpers (mirrors test_phase4_drive.py)
# ---------------------------------------------------------------------------

def workdays(start: date, count: int) -> list[date]:
    days = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def routine_intervals(return_min: int) -> list[tuple[int, int]]:
    return [(0, 7 * 60 + 30), (return_min, 24 * 60)]


def routine_days(start: date, count: int) -> list[tuple[date, list[tuple[int, int]]]]:
    days = []
    for i, day in enumerate(workdays(start, count)):
        days.append((day, routine_intervals(16 * 60 + 45 + (i % 4) * 5)))
    return days


def motion_events_for_days(days: list[date]) -> list[datetime]:
    """Motion: last ~22:15, first ~06:05, plus midday activity."""
    events = []
    for i, day in enumerate(days):
        bed = 22 * 60 + 10 + (i % 3) * 5
        wake = 6 * 60 + (i % 3) * 5
        for minutes in (bed - 30, bed - 5, bed, 12 * 60 + 15, wake, wake + 10):
            events.append(
                datetime.combine(day, datetime.min.time())
                + timedelta(minutes=minutes)
            )
    return sorted(events)


def _seed_bedtimes(learner: ScheduleLearner, count: int = 7) -> None:
    learner.ingest_motion_history(
        motion_events_for_days(workdays(date(2026, 11, 2), count))
    )


# ---------------------------------------------------------------------------
# Learner-level: bedtime feedback
# ---------------------------------------------------------------------------

def test_bedtime_no_discards_observation():
    """A 'no' to the bedtime call discards that date's bedtime."""
    learner = ScheduleLearner()
    _seed_bedtimes(learner, 7)
    bedtime, _, bed_n = learner.learned_bedtime("workday")
    assert bed_n == 7
    assert 22 * 60 + 5 <= bedtime <= 22 * 60 + 25

    # Ben taps 'no' on the Nov 10 bedtime notification (22:30).
    changed = learner.apply_bedtime_feedback(
        datetime(2026, 11, 10, 22, 30), confirmed=False
    )
    assert changed is True
    assert "2026-11-10" in learner._rejected_bedtimes
    # One observation removed; the median of the rest still ~22:15.
    bedtime, _, bed_n = learner.learned_bedtime("workday")
    assert bed_n == 6
    assert 22 * 60 + 5 <= bedtime <= 22 * 60 + 25
    # Repeating the same 'no' is a no-op (idempotent).
    assert learner.apply_bedtime_feedback(
        datetime(2026, 11, 10, 22, 35), confirmed=False
    ) is False


def test_bedtime_no_before_finalisation_prevents_derivation():
    """A 'no' received before the day finalises blocks the bedtime.

    The rejection is recorded first; when the day is later finalised
    the bedtime is skipped, but the wake observation is still kept.
    """
    learner = ScheduleLearner()
    base = datetime.combine(date(2026, 11, 2), datetime.min.time())
    # Evening motion (Nov 2) + next-morning motion (Nov 3), fed as raw
    # events so nothing is finalised yet.
    for minutes in (22 * 60, 22 * 60 + 15, 30 * 60 + 5, 30 * 60 + 15):
        learner.record_motion(base + timedelta(minutes=minutes))

    changed = learner.apply_bedtime_feedback(
        datetime(2026, 11, 2, 22, 30), confirmed=False
    )
    assert changed is True
    assert "2026-11-02" in learner._rejected_bedtimes

    learner._finalise_sleep_for_day(date(2026, 11, 2))
    learner._finalise_sleep_for_day(date(2026, 11, 3))
    # Bedtime: rejected, so the prior stands with zero confidence.
    bedtime, bed_conf, bed_n = learner.learned_bedtime("workday")
    assert bed_n == 0
    assert bedtime == 22 * 60  # prior
    assert bed_conf == 0.0
    # Wake: unaffected by a bedtime rejection.
    wake, _, wake_n = learner.learned_wake("workday")
    assert wake_n == 1
    assert 6 * 60 <= wake <= 6 * 60 + 15


def test_bedtime_yes_keeps_observation():
    """A 'yes' confirms the bedtime call: nothing changes."""
    learner = ScheduleLearner()
    _seed_bedtimes(learner, 7)
    before = learner.learned_bedtime("workday")
    assert learner.apply_bedtime_feedback(
        datetime(2026, 11, 10, 22, 30), confirmed=True
    ) is False
    assert learner.learned_bedtime("workday") == before
    assert learner._rejected_bedtimes == set()


def test_bedtime_feedback_after_midnight_targets_last_night():
    """A response sent after midnight still refers to last night."""
    learner = ScheduleLearner()
    base = datetime.combine(date(2026, 11, 2), datetime.min.time())
    learner.ingest_motion_history([base + timedelta(minutes=22 * 60)])
    assert learner.learned_bedtime("workday")[2] == 1

    # Ben taps 'no' at 01:00 on Nov 3: the latest derived bedtime is
    # Nov 2's, so that is the one discarded.
    changed = learner.apply_bedtime_feedback(
        datetime(2026, 11, 3, 1, 0), confirmed=False
    )
    assert changed is True
    assert "2026-11-02" in learner._rejected_bedtimes
    assert learner.learned_bedtime("workday")[2] == 0


def test_rejected_bedtimes_persist():
    """A rejected bedtime survives a store round trip."""
    learner = ScheduleLearner()
    _seed_bedtimes(learner, 7)
    learner.apply_bedtime_feedback(datetime(2026, 11, 10, 22, 30), confirmed=False)
    restored = ScheduleLearner.from_dict(learner.to_dict())
    assert "2026-11-10" in restored._rejected_bedtimes
    assert restored.learned_bedtime("workday")[2] == 6
    # And a later finalisation still will not re-derive it.
    restored._finalise_sleep_for_day(date(2026, 11, 10))
    assert restored.learned_bedtime("workday")[2] == 6


# ---------------------------------------------------------------------------
# Learner-level: home/away feedback
# ---------------------------------------------------------------------------

def test_home_no_downweights_wrong_call():
    """A 'no' to a home call adds one away observation for the slot."""
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    before = learner.predict_occupancy(datetime(2026, 11, 16, 18, 0))
    assert before > 0.9  # home on every observed day at 18:00

    changed = learner.apply_occupancy_feedback(
        datetime(2026, 11, 16, 18, 0), confirmed=False
    )
    assert changed is True
    after = learner.predict_occupancy(datetime(2026, 11, 16, 18, 0))
    assert after < before  # down-weighted, still home
    assert after > 0.5


def test_home_no_when_system_thinks_away_adds_home():
    """A 'no' to an away call adds a home observation for the slot."""
    learner = ScheduleLearner()  # prior only: away 07:00-15:00
    assert learner.predict_occupancy(datetime(2026, 11, 2, 10, 0)) == 0.0
    changed = learner.apply_occupancy_feedback(
        datetime(2026, 11, 2, 10, 0), confirmed=False
    )
    assert changed is True
    # One home observation against the 2-day prior: (0 + 1) / (2 + 1).
    assert learner.predict_occupancy(datetime(2026, 11, 2, 10, 0)) == pytest.approx(
        1 / 3, abs=0.01
    )


def test_home_yes_keeps_it():
    """A 'yes' to a home/away call changes nothing."""
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    before = learner.predict_occupancy(datetime(2026, 11, 16, 18, 0))
    assert learner.apply_occupancy_feedback(
        datetime(2026, 11, 16, 18, 0), confirmed=True
    ) is False
    assert learner.predict_occupancy(datetime(2026, 11, 16, 18, 0)) == before


def test_feedback_with_mock_datetime_is_ignored():
    """A mocked/non-datetime input is ignored, never raises."""
    learner = ScheduleLearner()
    _seed_bedtimes(learner, 3)
    mock_now = MagicMock()
    assert learner.apply_bedtime_feedback(mock_now, confirmed=False) is False
    assert learner.apply_occupancy_feedback(mock_now, confirmed=False) is False
    assert learner.learned_bedtime("workday")[2] == 3


# ---------------------------------------------------------------------------
# Coordinator-level: the preheat_feedback event listener
# ---------------------------------------------------------------------------

T0 = datetime(2026, 11, 9, 4, 0, tzinfo=timezone.utc)


class FakeClock:
    """Controllable replacement for homeassistant.util.dt inside the coordinator."""

    def __init__(self, start):
        self.t = start

    def utcnow(self):
        return self.t

    def now(self):
        return self.t

    def as_local(self, dt):
        return dt

    def advance(self, minutes):
        self.t += timedelta(minutes=minutes)


class TestFeedbackListenerCoordinator(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = FakeClock(T0)
        patcher = patch("custom_components.preheat.coordinator.dt_util", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

        from custom_components.preheat.coordinator import PreheatingCoordinator

        hass = MagicMock()
        entry = MagicMock()
        entry.entry_id = "zone"
        entry.title = "Office"
        entry.options = {CONF_HEATING_PROFILE: PROFILE_RADIATOR_NEW}
        entry.data = {}
        with patch.object(PreheatingCoordinator, "_setup_listeners"):
            self.coord = PreheatingCoordinator(hass, entry)
        self.coord._store = MagicMock()
        self.coord._store.async_save = AsyncMock()
        self.coord._store.async_load = AsyncMock(return_value={})
        self.coord._outdoor_is_real = True
        self.coord.deadtime_analyzer.analyze = MagicMock(return_value=None)
        self.coord.history_buffer.get_average_valve = MagicMock(return_value=80.0)
        self.coord.async_refresh = AsyncMock()
        self.hass = hass

    def _fire(self, data):
        """Fire a fake preheat_feedback event at the coordinator."""
        event = types.SimpleNamespace(data=data)
        self.coord._handle_feedback(event)

    def _seed_learned_routine(self):
        for day, intervals in routine_days(date(2026, 11, 2), 10):
            self.coord.schedule_learner.record_day(day, intervals)
        self.coord.schedule_learner.ingest_motion_history(
            motion_events_for_days(workdays(date(2026, 11, 2), 7))
        )

    def test_feedback_bedtime_no_discards_via_event(self):
        self._seed_learned_routine()
        assert self.coord.schedule_learner.learned_bedtime("workday")[2] == 7
        # Ben taps 'no' on the Nov 10 bedtime notification.
        self.clock.t = datetime(2026, 11, 10, 22, 30, tzinfo=timezone.utc)
        self._fire({"type": "bedtime", "response": "no"})
        assert self.coord.schedule_learner.learned_bedtime("workday")[2] == 6
        assert "2026-11-10" in self.coord.schedule_learner._rejected_bedtimes
        # The feedback was logged and a save was scheduled.
        assert len(self.coord._feedback_log) == 1
        assert self.coord._feedback_log[0]["type"] == "bedtime"
        assert self.coord._feedback_log[0]["response"] == "no"
        assert self.coord._feedback_log[0]["changed"] is True
        assert self.hass.async_create_task.called

    def test_feedback_bedtime_yes_keeps_it(self):
        self._seed_learned_routine()
        self.clock.t = datetime(2026, 11, 10, 22, 30, tzinfo=timezone.utc)
        self._fire({"type": "bedtime", "response": "yes"})
        assert self.coord.schedule_learner.learned_bedtime("workday")[2] == 7
        assert self.coord._feedback_log[0]["changed"] is False

    def test_feedback_home_no_corrects_occupancy(self):
        self._seed_learned_routine()
        self.clock.t = datetime(2026, 11, 16, 18, 0, tzinfo=timezone.utc)
        before = self.coord.schedule_learner.predict_occupancy(
            datetime(2026, 11, 16, 18, 0)
        )
        assert before > 0.9
        self._fire({"type": "home", "response": "no"})
        after = self.coord.schedule_learner.predict_occupancy(
            datetime(2026, 11, 16, 18, 0)
        )
        assert after < before
        assert self.coord._feedback_log[0]["type"] == "home"

    def test_feedback_away_no_marks_rejected(self):
        self.coord.away_mode = True
        self._fire({"type": "away", "response": "no"})
        assert self.coord._away_rejected is True
        assert self.coord._feedback_log[0]["type"] == "away"
        assert self.coord._feedback_log[0]["changed"] is True

    def test_feedback_away_yes_keeps_it(self):
        self.coord.away_mode = True
        self._fire({"type": "away", "response": "yes"})
        assert self.coord._away_rejected is False
        assert self.coord._feedback_log[0]["changed"] is False

    def test_feedback_malformed_events_ignored(self):
        self._seed_learned_routine()
        before_bed = self.coord.schedule_learner.learned_bedtime("workday")
        before_prob = self.coord.schedule_learner.predict_occupancy(
            datetime(2026, 11, 16, 18, 0)
        )
        malformed = [
            types.SimpleNamespace(data=None),  # no data at all
            types.SimpleNamespace(data={}),  # missing type/response
            types.SimpleNamespace(data={"type": "bedtime"}),  # missing response
            types.SimpleNamespace(data={"type": "unknown", "response": "no"}),
            types.SimpleNamespace(data={"type": "bedtime", "response": "maybe"}),
            types.SimpleNamespace(data={"type": 42, "response": "no"}),
            types.SimpleNamespace(data="not-a-dict"),
        ]
        for event in malformed:
            self.coord._handle_feedback(event)  # must not raise
        assert self.coord.schedule_learner.learned_bedtime("workday") == before_bed
        assert (
            self.coord.schedule_learner.predict_occupancy(datetime(2026, 11, 16, 18, 0))
            == before_prob
        )
        assert self.coord._feedback_log == []
        assert self.coord._away_rejected is False
        # No save was scheduled for any malformed event.
        assert not self.hass.async_create_task.called

    def test_feedback_response_is_case_insensitive(self):
        self._seed_learned_routine()
        self.clock.t = datetime(2026, 11, 10, 22, 30, tzinfo=timezone.utc)
        self._fire({"type": "Bedtime", "response": "NO"})
        assert self.coord.schedule_learner.learned_bedtime("workday")[2] == 6

    def test_away_mode_suppresses_ticks(self):
        """While away mode is on, no occupancy ticks reach the learner."""
        learner = self.coord.schedule_learner
        day = datetime(2026, 11, 16, 18, 0)
        self.coord._record_occupancy_tick(day, True)
        assert 36 in learner._current_slots  # 18:00 slot
        # Away on: the 19:00 tick is dropped, not recorded.
        self.coord.away_mode = True
        self.coord._record_occupancy_tick(datetime(2026, 11, 16, 19, 0), False)
        assert 38 not in learner._current_slots
        assert 36 in learner._current_slots
        # Away off: ticks flow again.
        self.coord.away_mode = False
        self.coord._record_occupancy_tick(datetime(2026, 11, 16, 19, 0), False)
        assert 38 in learner._current_slots

    def test_away_rejection_resets_on_toggle(self):
        self.coord.away_mode = True
        self._fire({"type": "away", "response": "no"})
        assert self.coord._away_rejected is True
        # Toggling away off (trip over) clears the rejection flag.
        import asyncio

        asyncio.get_event_loop().run_until_complete(self.coord.set_away_mode(False))
        assert self.coord._away_rejected is False

    def test_feedback_log_persists_in_storage(self):
        self._seed_learned_routine()
        self.clock.t = datetime(2026, 11, 10, 22, 30, tzinfo=timezone.utc)
        self._fire({"type": "bedtime", "response": "no"})
        self.coord.away_mode = True
        self._fire({"type": "away", "response": "no"})
        data = self.coord._get_data_for_storage()
        assert "feedback_log" in data
        assert len(data["feedback_log"]) == 2
        assert data["feedback_log"][0]["type"] == "bedtime"
        assert data["away_rejected"] is True
        assert "rejected_bedtimes" in data["schedule_learning"]
        assert "2026-11-10" in data["schedule_learning"]["rejected_bedtimes"]
