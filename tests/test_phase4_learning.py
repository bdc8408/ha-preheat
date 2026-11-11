"""Phase 4 tests: schedule learning + explicit thermal model (fork).

Validation strategy (per the plan): primarily synthetic history, since the
fork was only installed tonight and real recorder history is thin.
Synthetic household: workdays away 07:30-~16:45, home otherwise; sleep
~22:15, wake ~06:05. The fixed schedule predicts a 15:00 return; the
learned schedule must beat it on held-out days.
"""
import sys
import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import unittest

import pytest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_components.preheat.schedule_learner import (
    ScheduleLearner,
    backtest_return_prediction,
    day_type_for,
)
from custom_components.preheat.thermal_learner import ThermalLearner
from custom_components.preheat.units import celsius_delta_to_fahrenheit


# ---------------------------------------------------------------------------
# Synthetic household helpers
# ---------------------------------------------------------------------------

def workdays(start: date, count: int) -> list[date]:
    """First ``count`` weekdays (Mon-Fri) starting at ``start``."""
    days = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def routine_intervals(return_min: int) -> list[tuple[int, int]]:
    """Occupied intervals for a workday: home until 07:30, back at return."""
    return [(0, 7 * 60 + 30), (return_min, 24 * 60)]


def routine_days(start: date, count: int) -> list[tuple[date, list[tuple[int, int]]]]:
    """Workdays with returns jittered 16:45-17:00 (actual routine)."""
    days = []
    for i, day in enumerate(workdays(start, count)):
        days.append((day, routine_intervals(16 * 60 + 45 + (i % 4) * 5)))
    return days


def motion_events_for_days(days: list[date]) -> list[datetime]:
    """Motion events: last ~22:15, first ~06:05, plus midday activity."""
    events = []
    for i, day in enumerate(days):
        bed = 22 * 60 + 10 + (i % 3) * 5  # 22:10-22:20
        wake = 6 * 60 + (i % 3) * 5  # 06:00-06:10
        for minutes in (bed - 30, bed - 5, bed, 12 * 60 + 15, wake, wake + 10):
            events.append(
                datetime.combine(day, datetime.min.time())
                + timedelta(minutes=minutes)
            )
    return sorted(events)


# ---------------------------------------------------------------------------
# ScheduleLearner
# ---------------------------------------------------------------------------

def test_prior_only_predicts_fixed_schedule():
    learner = ScheduleLearner()
    monday = datetime(2026, 11, 2, 10, 0)  # workday, Leave window
    evening = datetime(2026, 11, 2, 19, 0)
    assert learner.predict_occupancy(monday) == 0.0
    assert learner.predict_occupancy(evening) == 1.0
    assert learner.confidence == 0.0
    bedtime, bed_conf, bed_n = learner.learned_bedtime("workday")
    wake, wake_conf, wake_n = learner.learned_wake("workday")
    assert (bedtime, bed_conf, bed_n) == (22 * 60, 0.0, 0)
    assert (wake, wake_conf, wake_n) == (5 * 60 + 30, 0.0, 0)
    suggestion = learner.suggested_schedule("workday")
    assert suggestion["delta_minutes"] == {
        "bedtime_min": 0,
        "wake_min": 0,
        "leave_min": 0,
        "return_min": 0,
    }


def test_occupancy_profile_learns_from_days():
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    assert learner.days_of_data == 10
    # 10:00 on a workday: away on every observed day.
    assert learner.predict_occupancy(datetime(2026, 11, 16, 10, 0)) < 0.3
    # 18:00 on a workday: home on every observed day.
    assert learner.predict_occupancy(datetime(2026, 11, 16, 18, 0)) > 0.9
    leave, ret = learner.learned_leave_return("workday")
    assert leave == 7 * 60 + 30
    assert 16 * 60 + 30 <= ret <= 17 * 60


def test_backtest_learned_beats_fixed_schedule():
    days = routine_days(date(2026, 11, 2), 14)
    result = backtest_return_prediction(days, fixed_return_min=15 * 60)
    assert result["status"] == "ok"
    assert result["test_days"] >= 4
    # Actual returns are ~16:45-17:00; the fixed 15:00 prediction is ~2h off.
    assert result["fixed_mae_min"] > 60
    assert result["learned_mae_min"] < 45
    assert result["learned_mae_min"] < result["fixed_mae_min"]


def test_sleep_wake_from_motion():
    learner = ScheduleLearner()
    days = workdays(date(2026, 11, 2), 7)
    learner.ingest_motion_history(motion_events_for_days(days))
    bedtime, bed_conf, bed_n = learner.learned_bedtime("workday")
    wake, wake_conf, wake_n = learner.learned_wake("workday")
    assert bed_n == 7 and wake_n == 7
    assert 22 * 60 + 5 <= bedtime <= 22 * 60 + 25  # ~22:15
    assert 6 * 60 <= wake <= 6 * 60 + 15  # ~06:05
    assert bed_conf > 0.5 and wake_conf > 0.5


def test_no_motion_keeps_prior_with_zero_confidence():
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 5):
        learner.record_day(day, intervals)
    bedtime, bed_conf, bed_n = learner.learned_bedtime("workday")
    assert (bedtime, bed_conf, bed_n) == (22 * 60, 0.0, 0)


def test_schedule_persistence_round_trip():
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 5):
        learner.record_day(day, intervals)
    learner.ingest_motion_history(
        motion_events_for_days(workdays(date(2026, 11, 2), 3))
    )
    restored = ScheduleLearner.from_dict(learner.to_dict())
    assert restored.days_of_data == learner.days_of_data
    for slot in (0, 20, 33, 40):
        assert restored.occupancy_probability("workday", slot) == pytest.approx(
            learner.occupancy_probability("workday", slot)
        )
    assert restored.learned_bedtime("workday") == learner.learned_bedtime("workday")
    assert restored.learned_wake("workday") == learner.learned_wake("workday")


def test_ingest_state_history_and_no_double_count():
    states = []
    for day in workdays(date(2026, 11, 2), 3):
        base = datetime.combine(day, datetime.min.time())
        states.extend(
            [
                (base + timedelta(minutes=0), "on"),
                (base + timedelta(minutes=7 * 60 + 30), "off"),
                (base + timedelta(minutes=16 * 60 + 45), "on"),
                (base + timedelta(minutes=23 * 60 + 59), "off"),
            ]
        )
    learner = ScheduleLearner()
    learner.ingest_state_history(states)
    assert learner.days_of_data == 3
    assert learner.predict_occupancy(datetime(2026, 11, 4, 10, 0)) == 0.0
    assert learner.predict_occupancy(datetime(2026, 11, 4, 19, 0)) == 1.0
    # Re-ingesting the same history must not double-count.
    learner.ingest_state_history(states)
    assert learner.days_of_data == 3


def test_record_tick_finalizes_on_rollover():
    learner = ScheduleLearner()
    day1 = date(2026, 11, 2)  # Monday
    tick = datetime.combine(day1, datetime.min.time())
    for _ in range(7 * 60):
        learner.record_tick(tick, True)
        tick += timedelta(minutes=1)
    learner.record_tick(datetime.combine(date(2026, 11, 3), datetime.min.time()), True)
    assert learner.days_of_data == 1
    assert learner.predict_occupancy(datetime(2026, 11, 2, 3, 0)) == 1.0


def test_predict_next_transition():
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    # Monday 16:00, away: next change is the ~16:45 arrival.
    when, kind, conf = learner.predict_next_transition(datetime(2026, 11, 16, 16, 0))
    assert kind == "arrival"
    assert when is not None and when.hour == 16 and when.minute in (30, 45)
    assert conf > 0
    # Monday 20:00, home: next change is the 07:30 departure (the first
    # away slot; the house is occupied until 07:30).
    when, kind, _ = learner.predict_next_transition(datetime(2026, 11, 16, 20, 0))
    assert kind == "departure"
    assert when is not None and when.hour == 7 and when.minute == 30


def test_mock_datetime_is_ignored():
    learner = ScheduleLearner()
    learner.record_tick(MagicMock(), True)  # must not raise (test harness mocks dt)
    learner.record_motion(MagicMock())
    assert learner.predict_occupancy(MagicMock()) == 0.0
    assert learner.predict_next_transition(MagicMock()) == (None, None, 0.0)
    assert learner.days_of_data == 0


def test_suggested_schedule_reports_delta_vs_fixed():
    learner = ScheduleLearner()
    for day, intervals in routine_days(date(2026, 11, 2), 10):
        learner.record_day(day, intervals)
    suggestion = learner.suggested_schedule("workday")
    # Learned return ~16:45 vs fixed 15:00 -> ~+105 min.
    assert 90 <= suggestion["delta_minutes"]["return_min"] <= 120
    assert suggestion["learned"]["leave_min"] == 7 * 60 + 30
    assert suggestion["confidence"] > 0


# ---------------------------------------------------------------------------
# ThermalLearner
# ---------------------------------------------------------------------------

def _synthetic_episodes(learner, count=20):
    """House: 2.0 °F/hr at 32°F outdoor, +0.02 °F/hr per °F warmer outside."""
    for i in range(count):
        outdoor_c = [-10.0, -5.0, 0.0, 5.0, 10.0][i % 5]
        outdoor_f = outdoor_c * 9.0 / 5.0 + 32.0
        rate_f = 2.0 + 0.02 * (outdoor_f - 32.0)
        rate_k = rate_f / 1.8
        noise = 2.0 if i % 2 == 0 else -2.0  # deterministic ±2 min
        duration = 10.0 + (2.0 / rate_k) * 60.0 + noise
        assert learner.record_episode(duration, 2.0, outdoor_c, deadtime_min=10.0)


def test_heating_rate_converges_to_truth():
    learner = ThermalLearner()
    _synthetic_episodes(learner, 20)
    assert learner.episode_count == 20
    assert learner.is_confident
    rate_32f = learner.reference_heating_rate_f_per_h()
    assert rate_32f is not None and abs(rate_32f - 2.0) < 0.15
    slope = learner.rate_slope_f_per_f()
    assert slope is not None and abs(slope - 0.02) < 0.01


def test_fahrenheit_inputs_do_not_skew_rate():
    # A 5°F rise over 2.5 h of effective heating = 2.0 °F/hr.
    # Correctly converted to °C at ingestion: 5°F -> 2.778 K.
    correct = ThermalLearner()
    for _ in range(5):
        assert correct.record_episode(150.0, 5.0 * 5.0 / 9.0, 0.0, deadtime_min=0.0)
    assert abs(correct.reference_heating_rate_f_per_h() - 2.0) < 0.05

    # The Phase 1 bug class: the same 5°F rise read raw as 5°C would
    # learn 3.6 °F/hr — ~1.8x too fast. The converted path must not.
    buggy = ThermalLearner()
    for _ in range(5):
        assert buggy.record_episode(150.0, 5.0, 0.0, deadtime_min=0.0)
    assert abs(buggy.reference_heating_rate_f_per_h() - 3.6) < 0.05
    assert correct.reference_heating_rate_f_per_h() < 2.5


def test_cooling_rate_from_tau():
    learner = ThermalLearner()
    assert learner.cooling_rate_f_per_h() is None  # no tau yet
    learner.sync(10.0, 5.0, 6.0, 0.8)
    assert abs(learner.cooling_rate_f_per_h() - 10.0 / 6.0) < 0.01
    learner.sync(10.0, 5.0, None, 0.0)
    assert learner.cooling_rate_f_per_h() is None


def test_heat_loss_min_per_f():
    learner = ThermalLearner()
    learner.sync(10.0, 5.0, 6.0, 0.8)  # loss factor 5 min/K
    assert abs(learner.heat_loss_min_per_f() - 5.0 * 5.0 / 9.0) < 0.01


def test_predict_duration_uses_deadtime_and_rate():
    learner = ThermalLearner()
    for _ in range(3):
        learner.record_episode(118.0, 2.0, 0.0, deadtime_min=10.0)
    assert learner.predict_duration_min(2.0, 0.0) is None  # not confident yet
    assert learner.heating_rate_f_per_h(0.0) is not None  # rate reportable at 3
    for _ in range(2):
        learner.record_episode(118.0, 2.0, 0.0, deadtime_min=10.0)
    duration = learner.predict_duration_min(2.0, 0.0)
    assert duration is not None and abs(duration - 118.0) < 5.0
    assert learner.predict_duration_min(0.0, 0.0) == 0.0


def test_rejects_implausible_episodes():
    learner = ThermalLearner()
    assert learner.record_episode(60.0, 0.1, 0.0, 0.0) is False  # rise too small
    assert learner.record_episode(12.0, 2.0, 0.0, 10.0) is False  # no effective heat-up
    assert learner.record_episode(16.0, 5.0, 0.0, 10.0) is False  # 50 K/hr, absurd
    assert learner.record_episode(60.0, MagicMock(), 0.0, 0.0) is False  # mocked rise
    assert learner.episode_count == 0


def test_thermal_persistence_round_trip():
    learner = ThermalLearner()
    _synthetic_episodes(learner, 6)
    learner.sync(12.0, 4.5, 6.5, 0.7)
    restored = ThermalLearner.from_dict(learner.to_dict())
    assert restored.episode_count == learner.episode_count
    assert restored.reference_heating_rate_f_per_h() == pytest.approx(
        learner.reference_heating_rate_f_per_h()
    )
    assert restored.cooling_tau_hours == 6.5
    assert restored.deadtime_min == 12.0
    assert restored.heat_loss_min_per_f() == pytest.approx(
        learner.heat_loss_min_per_f()
    )


def test_celsius_delta_conversion():
    assert celsius_delta_to_fahrenheit(1.0) == pytest.approx(1.8)
    assert celsius_delta_to_fahrenheit(5.0) == pytest.approx(9.0)


# ---------------------------------------------------------------------------
# Coordinator wiring
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


class TestPhase4CoordinatorWiring(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.clock = FakeClock(T0)
        patcher = patch("custom_components.preheat.coordinator.dt_util", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)

        from custom_components.preheat.coordinator import PreheatingCoordinator
        from custom_components.preheat.const import (
            CONF_HEATING_PROFILE,
            PROFILE_RADIATOR_NEW,
        )

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
        self.hass = hass

    def test_learners_exist(self):
        assert isinstance(self.coord.schedule_learner, ScheduleLearner)
        assert isinstance(self.coord.thermal_learner, ThermalLearner)

    def test_storage_round_trip_includes_phase4(self):
        for day, intervals in routine_days(date(2026, 11, 2), 5):
            self.coord.schedule_learner.record_day(day, intervals)
        self.coord.thermal_learner.record_episode(118.0, 2.0, 0.0, 10.0)
        data = self.coord._get_data_for_storage()
        assert "schedule_learning" in data
        assert "thermal_learning" in data
        restored_schedule = ScheduleLearner.from_dict(data["schedule_learning"])
        restored_thermal = ThermalLearner.from_dict(data["thermal_learning"])
        assert restored_schedule.days_of_data == 5
        assert restored_thermal.episode_count == 1

    def test_phase4_snapshot_with_mocked_clock(self):
        # Regression: the coordinator test harness mocks dt_util.now();
        # the snapshot must fall back to defaults, not raise.
        snapshot = self.coord._phase4_snapshot(
            {"now": MagicMock(), "outdoor_temp": 5.0},
            {"duration_source": "physics"},
        )
        assert snapshot["learned_occupancy_prob"] == 0.0
        assert snapshot["learned_next_arrival"] is None
        assert snapshot["thermal_duration_source"] == "physics"

    def test_phase4_snapshot_with_real_clock(self):
        for day, intervals in routine_days(date(2026, 11, 2), 10):
            self.coord.schedule_learner.record_day(day, intervals)
        snapshot = self.coord._phase4_snapshot(
            {"now": datetime(2026, 11, 16, 16, 0), "outdoor_temp": 0.0},
            {"duration_source": "physics"},
        )
        assert snapshot["learned_next_arrival"] is not None
        assert snapshot["learned_next_arrival"].hour == 16
        assert snapshot["learned_schedule_suggestion"]["delta_minutes"]["return_min"] > 60

    async def test_stop_preheat_feeds_thermal_learner(self):
        # Drive one preheat run through the coordinator: 19.0 -> 20.9°C
        # target reached at +40 min, held, stopped at +116 min.
        async def cycle(temp, should_start):
            dec = {"should_start": should_start, "frost_override": False}
            await self.coord._execute_control_actions(
                {"operative_temp": temp, "target_setpoint": 21.0, "outdoor_temp": 5.0},
                dec,
            )

        await cycle(19.0, True)
        self.clock.advance(20)
        await cycle(20.0, True)
        self.clock.advance(20)
        await cycle(20.9, True)  # target reached at +40 min
        self.clock.advance(76)
        await cycle(21.0, True)
        await cycle(21.0, False)  # stop
        assert self.coord.thermal_learner.episode_count == 1

    async def test_physics_simulation_prefers_thermal_duration(self):
        # House: 2 K rise takes 118 min (10 deadtime + 108 heating).
        for _ in range(5):
            self.coord.thermal_learner.record_episode(118.0, 2.0, 0.0, 10.0)
        ctx = {
            "target_setpoint": 21.0,
            "operative_temp": 19.0,
            "outdoor_temp": 0.0,
            "forecasts": None,
        }
        pred = await self.coord._run_physics_simulation(ctx)
        # Physics (Radiator Modern: 15 + 20*2 + 5*21 = 160 min) vs thermal
        # 118 min: 26% apart, inside the 50% guard -> thermal wins.
        assert pred["duration_source"] == "thermal_learner"
        assert abs(pred["predicted_duration"] - 118.0) < 5.0

    async def test_physics_simulation_falls_back_without_episodes(self):
        ctx = {
            "target_setpoint": 21.0,
            "operative_temp": 19.0,
            "outdoor_temp": 0.0,
            "forecasts": None,
        }
        pred = await self.coord._run_physics_simulation(ctx)
        assert pred["duration_source"] == "physics"
        # Bare-default physics (no profile loaded in this harness):
        # 0 deadtime + 10*2 mass + 5*21 loss = 125 min, under the 2.5h cap.
        assert abs(pred["predicted_duration"] - 125.0) < 0.01
