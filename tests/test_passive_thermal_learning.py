"""Passive thermal learning (fork v2.14.2).

Part A — the preheat decision uses the Phase 4 learned schedule's next
arrival when the house collector has no usable prediction (confident
path at >= 0.7, fallback path below it), reusing the existing
arbitration branches. The prior alone (confidence 0, no finalized
day) must never drive preheat.

Part B — a warm-up observer watches the climate entity each cycle and
feeds blueprint-driven heating sessions to the thermal model with the
same calls _stop_preheat makes. Data-only: no control behaviour
changes until the learner is confident and the gates pass.

Part C — CoolingAnalyzer: the segment-split rate cap (0.5 K/min) sat
below one 1 degF quantization step per minute (0.56 K/min), so the
Ecobee's dithering 70<->71 degF readings split every segment and no
cooling segment ever survived. Regression test with dithered data.
"""
import sys
import os
from datetime import date, datetime, timedelta, timezone
from unittest.mock import MagicMock, patch, AsyncMock
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from custom_components.preheat.const import (
    CONF_DRIVE_MODE,
    CONF_HEATING_PROFILE,
    CONF_OCCUPANCY,
    PROFILE_RADIATOR_NEW,
    KEY_PROVIDER_SELECTED,
    PROVIDER_NONE,
)
from custom_components.preheat.cooling_analyzer import CoolingAnalyzer
from custom_components.preheat.providers import ProviderDecision
from custom_components.preheat.units import fahrenheit_to_celsius

COMFORT_C = fahrenheit_to_celsius(71.0)  # 21.67 C
SLEEP_C = fahrenheit_to_celsius(68.0)  # 20.0 C


# ---------------------------------------------------------------------------
# Shared helpers (mirrors test_phase4_drive.py)
# ---------------------------------------------------------------------------

def workdays(start: date, count: int) -> list[date]:
    days = []
    day = start
    while len(days) < count:
        if day.weekday() < 5:
            days.append(day)
        day += timedelta(days=1)
    return days


def routine_days(start: date, count: int) -> list[tuple[date, list[tuple[int, int]]]]:
    """10-14 routine days: home until 07:30, return ~16:45 (+0-15 min)."""
    days = []
    for i, day in enumerate(workdays(start, count)):
        ret = 16 * 60 + 45 + (i % 4) * 5
        days.append((day, [(0, 7 * 60 + 30), (ret, 24 * 60)]))
    return days


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


class _CoordinatorHarness(unittest.IsolatedAsyncioTestCase):
    """Full-cycle coordinator harness (mirrors test_phase4_drive.py /
    test_coordinator.py's TestCoordinatorArbitration)."""

    CLOCK_START = datetime(2026, 11, 16, 15, 10, tzinfo=timezone.utc)  # Mon

    def setUp(self):
        self.clock = FakeClock(self.CLOCK_START)
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
        self.coord._startup_time = datetime.now(timezone.utc) - timedelta(hours=2)
        self.hass = hass

        # Providers: nothing valid except where a test says otherwise.
        self.coord.schedule_provider = MagicMock()
        self.coord.schedule_provider.get_next_start.return_value = None
        self.coord.schedule_provider.get_decision.return_value = ProviderDecision(
            should_stop=False, session_end=None, is_valid=False, is_shadow=False,
            invalid_reason="unavailable",
        )
        self.coord.learned_provider = MagicMock()
        self.coord.learned_provider.get_decision.return_value = ProviderDecision(
            should_stop=False, session_end=None, is_valid=False, is_shadow=True,
            invalid_reason="blocked_by_gates",
        )
        # Cooling analyzer + physics: mocked, as in TestCoordinatorArbitration.
        self.coord.cooling_analyzer = MagicMock()
        self.coord.cooling_analyzer.learned_tau = 3.0
        self.coord.cooling_analyzer.confidence = 0.5
        self.coord.physics = MagicMock()
        self.coord.physics.deadtime = 15.0
        self.coord.physics.mass_factor = 20.0
        self.coord.physics.loss_factor = 5.0
        self.coord.physics.sample_count = 10
        self.coord.physics.calculate_duration.return_value = 30.0
        self.coord.weather_service = MagicMock()
        self.coord.weather_service.get_forecasts = AsyncMock(return_value=[])
        self.coord.optimal_stop_manager = MagicMock()
        self.coord.optimal_stop_manager.is_active = False
        self.coord.optimal_stop_manager.stop_time = None
        self.coord.optimal_stop_manager._reason = "mock"
        self.coord.optimal_stop_manager._savings_total = 0.0
        self.coord.optimal_stop_manager._savings_remaining = 0.0
        self.coord.session_resolver = MagicMock()
        self.coord.planner.get_schedule_summary = MagicMock(return_value={})
        # Cycle inputs.
        self.coord._get_operative_temperature = AsyncMock(return_value=20.0)
        self.coord._get_target_setpoint = AsyncMock(return_value=21.0)
        self.coord._get_outdoor_temp_current = AsyncMock(return_value=0.0)
        self.coord._get_valve_position = MagicMock(return_value=0.0)
        # House collector: starved by default (the live Nov 18 state).
        self.collector = MagicMock()
        self.collector.get_next_arrival.return_value = (None, 0.0, "none")
        self.coord.house_collector = self.collector

    def _seed_routine(self, days=10):
        for day, intervals in routine_days(date(2026, 11, 2), days):
            self.coord.schedule_learner.record_day(day, intervals)

    async def _run_cycle(self):
        return await self.coord._async_update_data()


# ---------------------------------------------------------------------------
# Part A — learned-schedule arrival feeds the preheat decision
# ---------------------------------------------------------------------------

class TestLearnedArrivalFallback(_CoordinatorHarness):
    async def test_learned_arrival_selects_house_fallback(self):
        """10 routine days, Mon 15:10: arrival ~16:40 at conf 0.6 ->
        house fallback branch; planned start = arrival - lead."""
        self._seed_routine(10)
        expected, kind, conf = self.coord.schedule_learner.predict_next_transition(self.clock.t)
        assert kind == "arrival" and conf < 0.7 and conf > 0.0
        data = await self._run_cycle()
        assert data.next_arrival == expected
        trace = data.decision_trace
        assert trace[KEY_PROVIDER_SELECTED] == "house"
        assert trace["start_source"] == "house_fallback"
        assert trace["provider_candidates"]["house"]["source"] == "learned_fallback"
        assert trace["provider_candidates"]["house"]["confidence"] == conf
        # Planned start is arrival - (30 min physics + 10 min buffer).
        assert data.next_start_time is not None
        assert data.next_start_time == expected - timedelta(minutes=40)
        assert data.preheat_active is False

    async def test_learned_arrival_starts_preheat_inside_lead(self):
        """Same routine, Mon 16:20 (25 min before the ~16:50 arrival,
        inside the 40 min lead): preheat starts."""
        self._seed_routine(10)
        self.clock.t = datetime(2026, 11, 16, 16, 20, tzinfo=timezone.utc)
        expected, kind, conf = self.coord.schedule_learner.predict_next_transition(self.clock.t)
        assert kind == "arrival"
        data = await self._run_cycle()
        assert data.next_arrival == expected
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == "house"
        assert data.preheat_active is True

    async def test_collector_confident_wins_over_learner(self):
        """A confident collector prediction is used, not the learner's."""
        self._seed_routine(10)
        collector_event = datetime(2026, 11, 16, 17, 30, tzinfo=timezone.utc)
        self.collector.get_next_arrival.return_value = (collector_event, 0.9, "confident")
        data = await self._run_cycle()
        assert data.next_arrival == collector_event
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == "house"
        assert data.decision_trace["start_source"] == "house"
        assert data.decision_trace["provider_candidates"]["house"]["source"] == "confident"

    async def test_occupied_house_ignores_learned_arrival(self):
        """Occupied: there is no arrival to preheat for."""
        self._seed_routine(10)
        self.coord.entry.options[CONF_OCCUPANCY] = "binary_sensor.house_occupied"
        self.hass.states.is_state = MagicMock(return_value=True)
        data = await self._run_cycle()
        assert data.next_arrival is None
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == PROVIDER_NONE

    async def test_departure_prediction_is_not_an_event(self):
        """19:00, home: the next transition is a departure, not an event."""
        self._seed_routine(10)
        self.clock.t = datetime(2026, 11, 16, 19, 0, tzinfo=timezone.utc)
        nxt, kind, _ = self.coord.schedule_learner.predict_next_transition(self.clock.t)
        assert kind == "departure"
        data = await self._run_cycle()
        assert data.next_arrival is None
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == PROVIDER_NONE

    async def test_prior_only_never_drives_preheat(self):
        """Regression: with no finalized day the prior's 15:00 arrival
        (confidence 0) must not hijack next_event from the zone event."""
        self.clock.t = datetime(2026, 11, 16, 12, 0, tzinfo=timezone.utc)
        zone_event = self.clock.t + timedelta(minutes=30)
        self.coord.planner.get_next_scheduled_event = MagicMock(return_value=zone_event)
        nxt, kind, conf = self.coord.schedule_learner.predict_next_transition(self.clock.t)
        assert kind == "arrival" and conf == 0.0  # prior only
        data = await self._run_cycle()
        assert data.next_arrival == zone_event
        assert data.decision_trace["provider_candidates"]["house"]["source"] == "none"

    async def test_high_confidence_learner_takes_confident_path(self):
        """14 routine days: arrival conf 0.81 -> confident branch."""
        self._seed_routine(14)
        expected, kind, conf = self.coord.schedule_learner.predict_next_transition(self.clock.t)
        assert kind == "arrival" and conf >= 0.7
        data = await self._run_cycle()
        assert data.next_arrival == expected
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == "house"
        assert data.decision_trace["start_source"] == "house"
        assert data.decision_trace["provider_candidates"]["house"]["source"] == "learned"

    async def test_away_mode_ignores_learned_arrival(self):
        """Away mode: the return is driven by the away signal, not the routine."""
        self._seed_routine(10)
        self.coord.away_mode = True
        data = await self._run_cycle()
        assert data.next_arrival is None
        assert data.decision_trace[KEY_PROVIDER_SELECTED] == PROVIDER_NONE


# ---------------------------------------------------------------------------
# Part B — passive warm-up observer
# ---------------------------------------------------------------------------

class TestPassiveWarmupObserver(_CoordinatorHarness):
    def _ctx(self, now, temp_c, target_c, action, outdoor_c=0.0):
        return {
            "now": now,
            "operative_temp": temp_c,
            "target_setpoint": target_c,
            "outdoor_temp": outdoor_c,
            "hvac_action": action,
            "hvac_mode": "heat",
            "is_window_open": False,
            "preheat_active": False,
        }

    async def _drive(self, steps):
        """steps: (minutes from T0, temp_c, target_c, hvac_action)."""
        t0 = datetime(2026, 11, 16, 5, 30, tzinfo=timezone.utc)
        for minutes, temp, target, action in steps:
            now = t0 + timedelta(minutes=minutes)
            await self.coord._observe_passive_warmup(self._ctx(now, temp, target, action))

    def setUp(self):
        super().setUp()
        self.coord._async_save_data = AsyncMock()

    async def test_morning_warmup_records_episode(self):
        """Sleep 20.0 C -> Home 21.0 C over ~65 min: one episode."""
        await self._drive([
            (0, 19.0, 21.0, "heating"),
            (10, 19.3, 21.0, "heating"),
            (20, 19.6, 21.0, "heating"),
            (30, 19.9, 21.0, "heating"),
            (40, 20.2, 21.0, "heating"),
            (50, 20.5, 21.0, "heating"),
            (60, 20.8, 21.0, "heating"),
            (65, 20.9, 21.0, "heating"),  # within 0.2 K of target
        ])
        assert self.coord.thermal_learner.episode_count == 1
        assert self.coord._passive_episodes == 1
        assert self.coord._async_save_data.await_count >= 1
        assert self.coord._passive_warmup is None

    async def test_small_rise_is_not_an_episode(self):
        """0.2 K rise: below MIN_DELTA_K, no episode."""
        await self._drive([
            (0, 20.8, 21.0, "heating"),
            (30, 21.0, 21.0, "heating"),
        ])
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord._passive_episodes == 0

    async def test_short_session_is_not_an_episode(self):
        """0.9 K in 4 min: effective heat-up (4 - 15 deadtime) too short."""
        await self._drive([
            (0, 20.0, 21.0, "heating"),
            (4, 20.9, 21.0, "heating"),
        ])
        assert self.coord.thermal_learner.episode_count == 0

    async def test_window_open_discards_session(self):
        """Window opens mid warm-up: the session is discarded, not learned."""
        await self._drive([(0, 19.0, 21.0, "heating"), (30, 19.9, 21.0, "heating")])
        assert self.coord._passive_warmup is not None
        self.coord._window_open_detected = True
        await self._drive([(40, 20.2, 21.0, "heating"), (65, 20.9, 21.0, "heating")])
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord._passive_warmup is None

    async def test_integration_run_is_not_double_fed(self):
        """A preheat run starting mid-session abandons the passive session."""
        await self._drive([(0, 19.0, 21.0, "heating"), (30, 19.9, 21.0, "heating")])
        self.coord._preheat_active = True
        await self._drive([(40, 20.2, 21.0, "heating")])
        assert self.coord._passive_warmup is None
        self.coord._preheat_active = False
        await self._drive([(65, 20.9, 21.0, "heating")])
        assert self.coord.thermal_learner.episode_count == 0

    async def test_setpoint_change_splits_session(self):
        """Target 21.0 -> 22.0 mid warm-up: first episode closes, a
        second session opens and records its own episode."""
        await self._drive([
            (0, 19.0, 21.0, "heating"),
            (20, 19.8, 21.0, "heating"),
            (40, 20.5, 21.0, "heating"),
            (41, 20.5, 22.0, "heating"),  # setpoint jumps > 0.5 K
            (60, 21.2, 22.0, "heating"),
            (100, 21.8, 22.0, "heating"),  # within 0.2 K of 22.0
        ])
        assert self.coord.thermal_learner.episode_count == 2
        assert self.coord._passive_episodes == 2

    async def test_away_recovery_counts(self):
        """Deep-setback recovery (16 C -> 21 C) is real thermal data."""
        self.coord.away_mode = True
        await self._drive([
            (0, 16.0, 21.0, "heating"),
            (60, 17.7, 21.0, "heating"),
            (120, 19.4, 21.0, "heating"),
            (165, 20.9, 21.0, "heating"),
        ])
        assert self.coord.thermal_learner.episode_count == 1

    async def test_brief_idle_does_not_end_session(self):
        """2 min idle (fan run-on) mid warm-up: the session survives."""
        await self._drive([
            (0, 19.0, 21.0, "heating"),
            (30, 19.9, 21.0, "heating"),
            (32, 20.0, 21.0, "idle"),
            (34, 20.0, 21.0, "heating"),
            (65, 20.9, 21.0, "heating"),
        ])
        assert self.coord.thermal_learner.episode_count == 1

    async def test_long_idle_ends_session(self):
        """Idle >= 5 min ends the session against the temperature reached."""
        await self._drive([
            (0, 19.0, 21.0, "heating"),
            (30, 19.9, 21.0, "heating"),
            (32, 20.0, 21.0, "idle"),
            (37, 20.0, 21.0, "idle"),  # 5 min non-heating
        ])
        assert self.coord.thermal_learner.episode_count == 1
        assert self.coord._passive_warmup is None

    async def test_five_episodes_reach_confidence(self):
        """Five morning warm-ups: the learner becomes confident."""
        for day in range(5):
            base = day * 24 * 60
            await self._drive([
                (base + 0, 19.0, 21.0, "heating"),
                (base + 30, 19.9, 21.0, "heating"),
                (base + 65, 20.9, 21.0, "heating"),
            ])
        assert self.coord.thermal_learner.episode_count == 5
        assert self.coord.thermal_learner.is_confident is True


# ---------------------------------------------------------------------------
# Part C — CoolingAnalyzer rate cap vs 1 degF quantization
# ---------------------------------------------------------------------------

class TestCoolingAnalyzerDither(unittest.TestCase):
    def test_dithered_overnight_cooling_yields_a_fit(self):
        """Real Nov 17-18 shape: 71 -> 66 degF over ~250 min, readings
        quantized to whole degF and dithering at the rounding boundary.
        Before the v2.14.2 cap fix (0.5 K/min) every dither split the
        segment and analyze() found nothing; now the long segment
        survives and fits a tau."""
        analyzer = CoolingAnalyzer()
        start = datetime(2026, 11, 17, 22, 25, tzinfo=timezone.utc)
        # True temp falls smoothly; the reported temp is whole degF.
        for minute in range(251):
            true_f = 71.0 - 5.0 * minute / 250.0
            reported_f = round(true_f)  # 1 degF quantization + dither
            t_in_c = (reported_f - 32) * 5.0 / 9.0
            analyzer.add_data_point(
                dt=start + timedelta(minutes=minute),
                t_in=t_in_c,
                t_out=4.0,
                is_heating=False,
                window_open=False,
                valid_cooling=True,
            )
        segments = analyzer._extract_segments()
        assert len(segments) >= 1
        longest = max(segments, key=lambda s: len(s))
        assert len(longest) >= 60
        result = analyzer.analyze()
        assert "tau" in result, result
        assert 5.0 < result["tau"] < 48.0
        assert result["confidence"] > 0.0

    def test_flat_signal_still_fits_nothing(self):
        """A constant-temperature stretch is not a cooling segment fit."""
        analyzer = CoolingAnalyzer()
        start = datetime(2026, 11, 17, 22, 25, tzinfo=timezone.utc)
        for minute in range(120):
            analyzer.add_data_point(
                dt=start + timedelta(minutes=minute),
                t_in=19.44,
                t_out=4.0,
                is_heating=False,
                window_open=False,
                valid_cooling=True,
            )
        result = analyzer.analyze()
        assert result.get("status") == "no_valid_fits"
