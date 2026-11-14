"""Phase 4 (drive update) tests: drive-mode learned signals, holistic
bedtime, and away mode (fork, Ben Nov 10).

Drive mode: the learned schedule DRIVES the blueprint (learned-occupancy
/ sleeping / wake-window binary sensors), with the fixed schedule as the
prior so the signals are safe from day one. Away mode overrides the
learned schedule while active.
"""
import sys
import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import unittest

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_components.preheat.schedule_learner import ScheduleLearner
from custom_components.preheat.quiet_detector import QuietDetector
from custom_components.preheat.thermal_learner import ThermalLearner
from custom_components.preheat.units import fahrenheit_to_celsius
from custom_components.preheat.const import (
    CONF_DRIVE_MODE,
    CONF_HEATING_PROFILE,
    PROFILE_RADIATOR_NEW,
)


# ---------------------------------------------------------------------------
# Synthetic household helpers (mirrors test_phase4_learning.py)
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


# ---------------------------------------------------------------------------
# Drive mode: ScheduleLearner as a drive signal
# ---------------------------------------------------------------------------

def test_drive_prior_is_fixed_schedule():
    """Day one, no data: the drive signal IS the fixed schedule."""
    learner = ScheduleLearner()
    # Leave window (workday 10:00) -> away; evening -> home.
    assert learner.predict_occupancy(datetime(2026, 11, 2, 10, 0)) == 0.0
    assert learner.predict_occupancy(datetime(2026, 11, 2, 19, 0)) == 1.0
    # Fixed sleep window 22:00-05:30, fixed wake window 05:30-07:00.
    assert learner.is_sleeping(datetime(2026, 11, 2, 23, 0)) is True
    assert learner.is_sleeping(datetime(2026, 11, 2, 10, 0)) is False
    assert learner.in_wake_window(datetime(2026, 11, 2, 6, 0)) is True
    assert learner.in_wake_window(datetime(2026, 11, 2, 10, 0)) is False


def test_drive_learned_schedule_shifts():
    """After 10 routine days the learned routine drives, not the prior."""
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    learner.ingest_motion_history(motion_events_for_days(workdays(date(2026, 11, 2), 7)))
    # 16:00: the fixed schedule says home (return 15:00), the learned
    # routine says away (return ~16:45) -> drive signal is away.
    assert learner.predict_occupancy(datetime(2026, 11, 16, 16, 0)) < 0.5
    # 18:00: home on every observed day.
    assert learner.predict_occupancy(datetime(2026, 11, 16, 18, 0)) > 0.9
    # Learned sleep ~22:15-06:05, wake window 06:05-07:35.
    assert learner.is_sleeping(datetime(2026, 11, 16, 23, 0)) is True
    assert learner.is_sleeping(datetime(2026, 11, 16, 21, 0)) is False
    assert learner.in_wake_window(datetime(2026, 11, 16, 6, 30)) is True
    assert learner.in_wake_window(datetime(2026, 11, 16, 8, 0)) is False


# ---------------------------------------------------------------------------
# Holistic bedtime: media + hallway activity, not motion alone
# ---------------------------------------------------------------------------

def test_media_playing_blocks_bedtime():
    """TV on until 23:30: the bedtime call is 23:30, not the 22:00 motion."""
    day = date(2026, 11, 2)  # Monday
    base = datetime.combine(day, datetime.min.time())
    learner = ScheduleLearner()
    learner.ingest_motion_history([base + timedelta(minutes=22 * 60)])
    learner.ingest_media_history(
        [
            (base + timedelta(minutes=22 * 60), True),   # TV on at 22:00
            (base + timedelta(minutes=23 * 60 + 30), False),  # off at 23:30
        ]
    )
    bedtime, _, bed_n = learner.learned_bedtime("workday")
    assert bed_n == 1
    assert bedtime == 23 * 60 + 30  # 23:30, not 22:00

    # Control: the same motion with no media -> bedtime is 22:00.
    control = ScheduleLearner()
    control.ingest_motion_history([base + timedelta(minutes=22 * 60)])
    bedtime, _, bed_n = control.learned_bedtime("workday")
    assert bed_n == 1
    assert bedtime == 22 * 60


def test_no_media_motion_only_confirms_bedtime():
    """No media data at all: hallway motion alone still yields a bedtime."""
    learner = ScheduleLearner()
    learner.ingest_motion_history(motion_events_for_days(workdays(date(2026, 11, 2), 7)))
    bedtime, bed_conf, bed_n = learner.learned_bedtime("workday")
    wake, wake_conf, wake_n = learner.learned_wake("workday")
    assert bed_n == 7 and wake_n == 7
    assert 22 * 60 + 5 <= bedtime <= 22 * 60 + 25
    assert 6 * 60 <= wake <= 6 * 60 + 15
    assert bed_conf > 0.5 and wake_conf > 0.5


def test_water_trip_does_not_reset_bedtime():
    """A 01:00-01:03 water trip merges into the night; bedtime stays 22:15."""
    day = date(2026, 11, 2)  # Monday
    base = datetime.combine(day, datetime.min.time())
    events = [
        base + timedelta(minutes=22 * 60 + 15),  # bedtime activity
        base + timedelta(minutes=25 * 60),       # 01:00 water trip
        base + timedelta(minutes=25 * 60 + 3),   # back to bed
        base + timedelta(minutes=30 * 60 + 5),   # 06:05 wake
        base + timedelta(minutes=30 * 60 + 15),  # 06:15
    ]
    learner = ScheduleLearner()
    learner.ingest_motion_history(sorted(events))
    bedtime, _, bed_n = learner.learned_bedtime("workday")
    wake, _, wake_n = learner.learned_wake("workday")
    assert bed_n >= 1 and wake_n >= 1
    # Without the merge the bedtime would be 01:03 (the trip), not 22:15.
    assert 22 * 60 + 10 <= bedtime <= 22 * 60 + 20
    assert 6 * 60 <= wake <= 6 * 60 + 10


def test_quiet_detector_persistence_round_trip():
    """Motion + media history survives a store round trip."""
    learner = ScheduleLearner()
    learner.ingest_motion_history(motion_events_for_days(workdays(date(2026, 11, 2), 3)))
    day = date(2026, 11, 2)
    base = datetime.combine(day, datetime.min.time())
    learner.ingest_media_history(
        [(base + timedelta(minutes=21 * 60), True),
         (base + timedelta(minutes=22 * 60), False)]
    )
    restored = ScheduleLearner.from_dict(learner.to_dict())
    assert restored.learned_bedtime("workday") == learner.learned_bedtime("workday")
    assert restored.learned_wake("workday") == learner.learned_wake("workday")
    # The quiet detector itself round-trips (media state is preserved).
    assert restored.quiet_detector.media_playing_at(
        base + timedelta(minutes=21 * 60 + 30)
    ) is True


# ---------------------------------------------------------------------------
# Live wake-up: sustained morning activity ends the sleep window early
# (fork, Nov 14 2026 — Ben was up and moving 07:54-08:09, learned wake
# 09:04, but learned_sleeping stayed on until the clock said so).
# ---------------------------------------------------------------------------

def _nov14_detector() -> tuple[QuietDetector, datetime]:
    """Detector fed the Nov 13 night + Nov 14 morning (Ben's actual day).

    Last night motion 21:55; morning motion 07:54:07, 07:54:32,
    07:55:06, 07:56:03, 07:57:22, 08:05:54, 08:06:25, 08:09:19.
    Returns (detector, morning base = Nov 14 00:00).
    """
    detector = QuietDetector()
    night = datetime(2026, 11, 13, 21, 55)
    detector.record_activity(night)
    morning = datetime(2026, 11, 14, 7, 54, 7)
    for minutes, seconds in [(0, 0), (0, 25), (0, 59), (1, 56), (3, 15),
                             (11, 47), (12, 18), (15, 12)]:
        detector.record_activity(morning + timedelta(minutes=minutes, seconds=seconds))
    return detector, datetime(2026, 11, 14)


def test_awake_since_none_when_quiet():
    """A quiet house (no morning noise at all) is never awake."""
    detector = QuietDetector()
    detector.record_activity(datetime(2026, 11, 13, 21, 55))
    assert detector.awake_since(datetime(2026, 11, 14, 8, 0)) is None
    assert detector.awake_since(datetime(2026, 11, 14, 9, 30)) is None


def test_awake_since_ignores_water_trip():
    """A 01:00-01:02 water trip never sustains long enough to wake."""
    detector = QuietDetector()
    detector.record_activity(datetime(2026, 11, 13, 21, 55))
    detector.record_activity(datetime(2026, 11, 14, 1, 0))
    detector.record_activity(datetime(2026, 11, 14, 1, 2))
    assert detector.awake_since(datetime(2026, 11, 14, 1, 10)) is None
    assert detector.awake_since(datetime(2026, 11, 14, 1, 40)) is None


def test_awake_since_after_sustained_morning_activity():
    """Nov 14 replay: awake at ~08:06, still awake at 08:20, quiet again by 08:40."""
    detector, _ = _nov14_detector()
    # 08:00: only ~3 min of activity so far — not sustained yet.
    assert detector.awake_since(datetime(2026, 11, 14, 8, 0)) is None
    # 08:06: 11+ min of repeated motion — awake since the first move.
    assert detector.awake_since(datetime(2026, 11, 14, 8, 6)) == datetime(
        2026, 11, 14, 7, 54, 7
    )
    # 08:20: 11 min after the last motion, still inside the 30-min latch.
    assert detector.awake_since(datetime(2026, 11, 14, 8, 20)) == datetime(
        2026, 11, 14, 7, 54, 7
    )
    # 08:40: 31 min after the last motion — the house is quiet again.
    assert detector.awake_since(datetime(2026, 11, 14, 8, 40)) is None


def test_awake_since_single_blip_never_counts():
    """One motion event, then nothing: no stretch, never awake."""
    detector = QuietDetector()
    detector.record_activity(datetime(2026, 11, 13, 21, 55))
    detector.record_activity(datetime(2026, 11, 14, 7, 54))
    assert detector.awake_since(datetime(2026, 11, 14, 8, 0)) is None
    assert detector.awake_since(datetime(2026, 11, 14, 8, 30)) is None


def test_awake_since_media_sustained():
    """Media playing for 10+ min counts as awake; a 5-min burst does not."""
    detector = QuietDetector()
    detector.record_activity(datetime(2026, 11, 13, 22, 0))
    detector.record_media(datetime(2026, 11, 14, 7, 0), True)
    assert detector.awake_since(datetime(2026, 11, 14, 7, 30)) == datetime(
        2026, 11, 14, 7, 0
    )

    control = QuietDetector()
    control.record_activity(datetime(2026, 11, 13, 22, 0))
    control.record_media(datetime(2026, 11, 14, 7, 0), True)
    control.record_media(datetime(2026, 11, 14, 7, 5), False)
    assert control.awake_since(datetime(2026, 11, 14, 7, 15)) is None


# ---------------------------------------------------------------------------
# Coordinator wiring: drive + away mode
# ---------------------------------------------------------------------------

T0 = datetime(2026, 11, 9, 4, 0, tzinfo=timezone.utc)
COMFORT_C = fahrenheit_to_celsius(71.0)  # 21.67°C
AWAY_C = fahrenheit_to_celsius(61.0)     # 16.11°C (71 - 10°F, the floor)


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


class TestPhase4DriveCoordinator(unittest.IsolatedAsyncioTestCase):
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
        # Snapshot calls must not run a real update cycle in these tests.
        self.coord.async_refresh = AsyncMock()
        self.hass = hass

    def _snapshot(self, now, outdoor=0.0, target=COMFORT_C):
        return self.coord._phase4_snapshot(
            {"now": now, "outdoor_temp": outdoor, "target_setpoint": target},
            {"duration_source": "physics"},
        )

    def _seed_learned_routine(self):
        """10 routine days + 7 nights of motion: learned return ~16:45."""
        for day, intervals in routine_days(date(2026, 11, 2), 10):
            self.coord.schedule_learner.record_day(day, intervals)
        self.coord.schedule_learner.ingest_motion_history(
            motion_events_for_days(workdays(date(2026, 11, 2), 7))
        )

    def _seed_heating_rate(self, duration_min: float = 78.0):
        """5 warm-up episodes: ``duration_min`` for a 2 K rise at 0°C
        outdoor, 10 min deadtime. The default 78 min gives ~1.76 K/hr,
        whose warm-up duration sits inside the 50% physics guard in
        the test harness (whose physics model is the untuned default),
        so the fitted rate drives the return pre-heat start.
        """
        for _ in range(5):
            self.coord.thermal_learner.record_episode(duration_min, 2.0, 0.0, 10.0)
        assert self.coord.thermal_learner.is_confident

    # -- drive mode ------------------------------------------------------

    def test_snapshot_exposes_drive_signals(self):
        self._seed_learned_routine()
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["learned_drive_mode"] is True
        # 19:00: home on every observed day.
        assert snap["learned_occupancy_prob"] > 0.9
        # 19:00 is not sleeping, not in the wake window.
        assert snap["learned_sleeping"] is False
        assert snap["learned_wake_window"] is False
        assert snap["learned_wake_time"] is not None
        assert snap["learned_bedtime_time"] is not None

    def test_snapshot_sleeping_and_wake_window(self):
        self._seed_learned_routine()
        snap = self._snapshot(datetime(2026, 11, 16, 23, 0))
        assert snap["learned_sleeping"] is True
        assert snap["learned_wake_window"] is False
        snap = self._snapshot(datetime(2026, 11, 16, 6, 30))
        assert snap["learned_sleeping"] is False
        assert snap["learned_wake_window"] is True

    def test_live_wake_up_ends_sleep_early(self):
        """Nov 14 replay: learned wake ~09:00, but sustained 07:54-08:09
        motion wakes the house at ~08:06; without it, sleeping holds."""
        # 7 workdays whose first morning motion is ~09:00.
        events = []
        for day in workdays(date(2026, 11, 2), 7):
            base = datetime.combine(day, datetime.min.time())
            events.append(base + timedelta(minutes=22 * 60 + 15))  # bedtime
            events.append(base + timedelta(minutes=9 * 60))        # 09:00
            events.append(base + timedelta(minutes=9 * 60 + 10))   # 09:10
        self.coord.schedule_learner.ingest_motion_history(sorted(events))
        wake, _, wake_n = self.coord.schedule_learner.learned_wake("workday")
        assert wake_n == 7 and 8 * 60 + 55 <= wake <= 9 * 60 + 5
        # Monday Nov 16, 08:00: the clock says sleeping, no live motion yet.
        snap = self._snapshot(datetime(2026, 11, 16, 8, 0))
        assert snap["learned_sleeping"] is True
        assert snap["learned_wake_window"] is False
        # Sustained morning motion (the Nov 14 pattern, replayed Nov 16).
        morning = datetime(2026, 11, 16, 7, 54, 7)
        for minutes, seconds in [(0, 0), (0, 25), (0, 59), (1, 56), (3, 15),
                                 (11, 47), (12, 18)]:
            self.coord.schedule_learner.record_motion(
                morning + timedelta(minutes=minutes, seconds=seconds)
            )
        snap = self._snapshot(datetime(2026, 11, 16, 8, 6))
        assert snap["learned_sleeping"] is False
        # Control: the same morning without the motion still sleeps.
        control = ScheduleLearner()
        control.ingest_motion_history(sorted(events))
        assert control.is_sleeping(datetime(2026, 11, 16, 8, 6)) is True
        assert control.is_awake(datetime(2026, 11, 16, 8, 6)) is False

    def test_drive_mode_off_reports_suggest_only(self):
        self.coord.entry.options[CONF_DRIVE_MODE] = False
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["learned_drive_mode"] is False
        # The learned values are still computed (report), just flagged off.
        assert snap["learned_schedule_suggestion"] is not None

    # -- away mode -------------------------------------------------------

    def test_away_mode_holds_deep_setback(self):
        self._seed_learned_routine()
        self.coord.away_mode = True
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["away_mode"] is True
        assert snap["away_setback_active"] is True
        assert snap["away_preheat"] is False
        assert snap["away_setpoint"] is not None
        assert abs(snap["away_setpoint"] - AWAY_C) < 0.05  # 61°F
        # The learned drive is forced off while away (away wins).
        assert snap["learned_occupancy_prob"] == 0.0
        assert snap["learned_sleeping"] is False
        assert snap["learned_wake_window"] is False
        assert snap["learned_next_arrival"] is None

    def test_away_return_preheat_starts_at_right_time(self):
        self._seed_heating_rate()
        self.coord.away_mode = True
        # Return Mon 17:00. Warm-up: 5.56 K at the fitted ~1.76 K/hr +
        # 10 min deadtime = ~199 min (thermal learner, inside the 50%
        # physics guard), + 10 min buffer -> start ~13:31.
        ret = datetime(2026, 11, 16, 17, 0)
        self.coord.away_return_dt = ret
        delta_c = COMFORT_C - AWAY_C
        thermal_duration = self.coord.thermal_learner.predict_duration_min(delta_c, 0.0)
        assert thermal_duration is not None
        expected_start = ret - timedelta(minutes=thermal_duration + 10.0)
        snap = self._snapshot(datetime(2026, 11, 16, 10, 0))
        assert snap["away_return_preheat_start"] is not None
        start = snap["away_return_preheat_start"]
        # The start is driven by the fitted thermal rate, not the physics
        # fallback (which would put it ~35 min later).
        assert abs((start - expected_start).total_seconds()) < 60
        assert start.hour == 13 and 28 <= start.minute <= 34  # ~13:31
        # Well before the start: still in deep setback.
        assert snap["away_setback_active"] is True
        assert snap["away_preheat"] is False
        # Inside the return window (14:00): warming back to Comfort.
        snap = self._snapshot(datetime(2026, 11, 16, 14, 0))
        assert snap["away_setback_active"] is False
        assert snap["away_preheat"] is True
        # After the return time: the window has closed.
        snap = self._snapshot(datetime(2026, 11, 16, 17, 30))
        assert snap["away_preheat"] is False

    def test_away_no_return_stays_low_energy_until_off(self):
        self.coord.away_mode = True
        self.coord.away_return_dt = None
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["away_setback_active"] is True
        assert snap["away_preheat"] is False
        assert snap["away_return_preheat_start"] is None
        # Even days later, still low-energy.
        snap = self._snapshot(datetime(2026, 11, 20, 19, 0))
        assert snap["away_setback_active"] is True
        # Switching off resumes the normal learned drive.
        self.coord.away_mode = False
        snap = self._snapshot(datetime(2026, 11, 20, 19, 0))
        assert snap["away_setback_active"] is False
        assert snap["away_preheat"] is False
        assert snap["learned_occupancy_prob"] == 1.0

    def test_away_overrides_learned_occupancy(self):
        self._seed_learned_routine()
        # 19:00 with the learned routine: home.
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["learned_occupancy_prob"] > 0.9
        # Away on: the same moment reads empty.
        self.coord.away_mode = True
        snap = self._snapshot(datetime(2026, 11, 16, 19, 0))
        assert snap["learned_occupancy_prob"] == 0.0

    async def test_set_away_mode_setters(self):
        # Production datetimes are timezone-aware (T0 is UTC here).
        # A stale (past) return is cleared when away is switched on.
        self.coord.away_return_dt = datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc)
        await self.coord.set_away_mode(True)
        assert self.coord.away_mode is True
        assert self.coord.away_return_dt is None
        # A future return is kept.
        future = datetime(2026, 11, 20, 17, 0, tzinfo=timezone.utc)
        await self.coord.set_away_return(future)
        assert self.coord.away_return_dt == future
        # A past return is ignored.
        await self.coord.set_away_return(datetime(2026, 11, 8, 12, 0, tzinfo=timezone.utc))
        assert self.coord.away_return_dt is None
        # Switching off clears the return and resumes normal drive.
        await self.coord.set_away_return(future)
        await self.coord.set_away_mode(False)
        assert self.coord.away_mode is False
        assert self.coord.away_return_dt is None

    def test_away_state_persists_in_storage(self):
        self.coord.away_mode = True
        self.coord.away_return_dt = datetime(2026, 11, 20, 17, 0)
        data = self.coord._get_data_for_storage()
        assert data["away_mode"] is True
        assert data["away_return"] == "2026-11-20T17:00:00"
