"""Thermal history backfill (fork v2.14.3).

A fresh thermal model sits at zero episodes for days because the passive
observer only learns from warm-ups that complete after the install.
The one-time backfill reconstructs warm-ups from the last 10 days of
recorder history with the SAME session rules as the live observer and
feeds them through the same calls. Data-only: no gate is relaxed, so
the model reports whatever episode count the history supports.

Covered here:
- pure reconstruction (incl. the Nov 17 15:32 63->71 degF return shape
  and the Nov 18 mode flaps that must not inflate the count);
- the coordinator backfill end to end on realistic Nov 10-18 history
  (15 routine warm-ups + 2 Nov 18 morning sessions, 0 from flaps; the
  Nov 10 morning session is skipped — no outdoor reading covers it);
- the once-only flag, the already-learned skip, the zero-history and
  no-climate cases, and the load-time scheduling hook;
- the real-physics pins (Fix A): with no valve source the feed path
  passes valve_position=None so the physics model learns too
  (sample_count > 0); a real valve sensor reading 0% still keeps the
  guard (sample_count stays 0) while the learner accumulates episodes;
- the no-outdoor skip (Fix D): a session with no outdoor coverage is
  skipped, never fed with the fabricated 10.0 C placeholder.

Phase 8 (v2.14.4): the v2.14.3 backfill ran live but fed only 1 of
164 sessions — the Radiator profile's 15-min deadtime (never
measured, only a profile prior) was subtracted from 5-13 min cycles
and inflated longer legs past the rate cap, and sessions were
shredded by 1°F setpoint steps. Covered here:
- Fix 1: an unmeasured deadtime is passed as 0.0, so the effective
  duration is the full duration; a measured deadtime is still
  subtracted (and the learner keeps the synced profile prior when
  fed 0.0, so prediction is untouched).
- Fix 2: a setpoint step UP mid-warm-up retargets the session
  instead of closing it (a step DOWN still closes it); the backfill
  grace is 15 min, so a furnace cycle mid-warm-up does not split it.
- Fix 3: a versioned re-run — a v2.14.3 store (flag set, no version)
  re-runs once after the reset, and the same version never re-runs.
- Fix 4: the backfill summary logs at WARNING with rejection counts.
"""
import sys
import os
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, AsyncMock
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from test_passive_thermal_learning import _CoordinatorHarness
from custom_components.preheat.const import (
    CONF_CLIMATE,
    CONF_HEATING_PROFILE,
    CONF_WEATHER_ENTITY,
    TARGET_REACHED_TOLERANCE,
)
from custom_components.preheat.coordinator import (
    BACKFILL_VERSION,
    PASSIVE_NON_HEATING_GRACE_MIN,
    PASSIVE_SETPOINT_CHANGE_K,
)
from custom_components.preheat.history_backfill import (
    ClimateSample,
    attach_outdoor,
    reconstruct_sessions,
)
from custom_components.preheat.physics import ThermalPhysics
from custom_components.preheat.const import HEATING_PROFILES, PROFILE_RADIATOR_NEW
from custom_components.preheat.units import fahrenheit_to_celsius

CLIMATE = "climate.living_room_thermostat"
WEATHER = "weather.forecast_home"


class FakeState:
    """Minimal recorder State: .state, .last_changed, .attributes."""

    def __init__(self, state, last_changed, attributes=None):
        self.state = state
        self.last_changed = last_changed
        self.attributes = attributes or {}


def _sample(minutes, temp_f, target_f, heating, start=None):
    base = start or datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
    return ClimateSample(
        ts=(base + timedelta(minutes=minutes)).timestamp(),
        temp_c=fahrenheit_to_celsius(temp_f),
        target_c=fahrenheit_to_celsius(target_f),
        heating=heating,
    )


def _reconstruct(samples):
    return reconstruct_sessions(
        samples,
        non_heating_grace_min=PASSIVE_NON_HEATING_GRACE_MIN,
        setpoint_change_k=PASSIVE_SETPOINT_CHANGE_K,
        target_tolerance_k=TARGET_REACHED_TOLERANCE,
    )


# ---------------------------------------------------------------------------
# Pure reconstruction
# ---------------------------------------------------------------------------

class TestReconstructSessions(unittest.TestCase):
    def test_nov17_afternoon_return_shape(self):
        """Nov 17 15:32: setpoint 63 -> 71 degF, temp follows to 71."""
        sessions = _reconstruct([
            _sample(0, 63, 63, False),
            _sample(32, 64, 71, True),
            _sample(60, 66, 71, True),
            _sample(90, 69, 71, True),
            _sample(110, 71, 71, True),
        ])
        assert len(sessions) == 1
        s = sessions[0]
        assert abs((s.end_temp_c - s.start_temp_c)
                   - (fahrenheit_to_celsius(71) - fahrenheit_to_celsius(64))) < 0.01
        assert abs((s.end_ts - s.start_ts) / 60.0 - 78.0) < 0.01
        assert s.target_c == fahrenheit_to_celsius(71)

    def test_mode_flaps_produce_no_qualifying_session(self):
        """Nov 18 flaps: 62 <-> 70 step-ups within minutes. The state
        machine closes each flap quickly; the rises are too small for
        an episode (the feed-time guard rejects them)."""
        sessions = _reconstruct([
            _sample(0, 65, 70, True),
            _sample(8, 65.5, 62, True),
            _sample(20, 65.5, 70, True),
            _sample(25, 65.5, 62, False),
        ])
        assert len(sessions) == 2
        for s in sessions:
            assert (s.end_temp_c - s.start_temp_c) < 0.3  # MIN_DELTA_K
            assert (s.end_ts - s.start_ts) / 60.0 <= 8.0

    def test_long_idle_closes_session(self):
        """Burner off >= 5 min ends the session at the temp reached."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
            _sample(32, 68.5, 71, False),
            _sample(37, 68.5, 71, False),
        ])
        assert len(sessions) == 1
        assert abs((sessions[0].end_ts - sessions[0].start_ts) / 60.0 - 37.0) < 0.01

    def test_brief_idle_does_not_end_session(self):
        """2 min idle mid warm-up: the session survives to the target."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
            _sample(32, 68.5, 71, False),
            _sample(34, 68.5, 71, True),
            _sample(65, 71, 71, True),
        ])
        assert len(sessions) == 1

    def test_break_closes_session(self):
        """An unavailable gap closes the session at the last good sample
        instead of merging two warm-ups across it."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
            None,
            _sample(120, 66, 71, True),
            _sample(180, 71, 71, True),
        ])
        assert len(sessions) == 2
        assert abs((sessions[0].end_ts - sessions[0].start_ts) / 60.0 - 30.0) < 0.01

    def test_unclosed_session_is_discarded(self):
        """A warm-up still running at the end of the window is not fed."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
        ])
        assert sessions == []

    def test_setpoint_step_up_merges_session(self):
        """Phase 8 Fix 2: target 71 -> 72 mid warm-up retargets the
        session instead of closing it — one session, no double-counted
        rise. A 1°F step is 0.56 K, above the change threshold, so the
        old rule shredded every routine Sleep -> Wake -> Home warm-up."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
            _sample(31, 68, 72, True),
            _sample(70, 71.5, 72, True),
            _sample(75, 72, 72, True),
        ])
        assert len(sessions) == 1
        assert sessions[0].start_temp_c == fahrenheit_to_celsius(66)
        assert sessions[0].end_temp_c == fahrenheit_to_celsius(72)
        assert sessions[0].target_c == fahrenheit_to_celsius(72)
        assert abs((sessions[0].end_ts - sessions[0].start_ts) / 60.0 - 75.0) < 0.01

    def test_setpoint_step_down_splits_session(self):
        """Phase 8 Fix 2: a step DOWN still ends the session — only a
        step UP merges. Target 72 -> 71 mid warm-up: first session
        closes, a second opens from the current temperature."""
        sessions = _reconstruct([
            _sample(0, 66, 72, True),
            _sample(30, 68, 72, True),
            _sample(31, 68, 71, True),
            _sample(70, 71, 71, True),
        ])
        assert len(sessions) == 2
        assert sessions[0].end_temp_c == sessions[1].start_temp_c

    def test_step_up_at_target_closes_session(self):
        """Phase 8 Fix 2: a step UP when the room is already at the
        new target does not merge — the warm-up is over, so the
        session closes and nothing new opens."""
        sessions = _reconstruct([
            _sample(0, 66, 71, True),
            _sample(30, 68, 71, True),
            _sample(31, 71.9, 72, True),
            _sample(60, 71.9, 72, True),
        ])
        assert len(sessions) == 1
        assert abs((sessions[0].end_ts - sessions[0].start_ts) / 60.0 - 31.0) < 0.01
        assert sessions[0].end_temp_c == fahrenheit_to_celsius(71.9)

    def test_long_idle_survives_with_backfill_grace(self):
        """Phase 8 Fix 2: with the backfill's 15-min grace, a 10-min
        furnace cycle mid-warm-up does not split the session (the
        live observer's 5-min grace would close it at minute 37)."""
        sessions = reconstruct_sessions(
            [
                _sample(0, 66, 71, True),
                _sample(30, 68, 71, True),
                _sample(32, 68.5, 71, False),
                _sample(42, 68.5, 71, True),
                _sample(65, 71, 71, True),
            ],
            non_heating_grace_min=15.0,
            setpoint_change_k=PASSIVE_SETPOINT_CHANGE_K,
            target_tolerance_k=TARGET_REACHED_TOLERANCE,
        )
        assert len(sessions) == 1
        assert abs((sessions[0].end_ts - sessions[0].start_ts) / 60.0 - 65.0) < 0.01

    def test_two_sample_session_has_outdoor_coverage(self):
        """Phase 8: a session of just an opener + closer still counts
        its outdoor readings — the opener's reading is accumulated
        when the session opens, so it is not wrongly skipped as having
        no outdoor coverage."""
        base = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
        sessions = reconstruct_sessions(
            [
                ClimateSample(base.timestamp(), 18.0, 21.0, True, outdoor_c=5.0),
                ClimateSample(
                    (base + timedelta(minutes=20)).timestamp(),
                    21.0, 21.0, True, outdoor_c=6.0,
                ),
            ],
            non_heating_grace_min=PASSIVE_NON_HEATING_GRACE_MIN,
            setpoint_change_k=PASSIVE_SETPOINT_CHANGE_K,
            target_tolerance_k=TARGET_REACHED_TOLERANCE,
        )
        assert len(sessions) == 1
        assert sessions[0].outdoor_known is True
        assert sessions[0].outdoor_avg_c == 5.0

    def test_attach_outdoor_carry_forward(self):
        """Sparse outdoor samples carry forward to the climate samples."""
        base = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
        samples = attach_outdoor(
            [
                ClimateSample((base).timestamp(), 18.0, 21.0, True),
                ClimateSample((base + timedelta(minutes=30)).timestamp(), 19.0, 21.0, True),
                ClimateSample((base + timedelta(minutes=90)).timestamp(), 20.0, 21.0, True),
            ],
            [
                ((base - timedelta(hours=1)).timestamp(), 7.0),
                ((base + timedelta(minutes=60)).timestamp(), 5.0),
            ],
        )
        assert samples[0].outdoor_c == 7.0
        assert samples[1].outdoor_c == 7.0
        assert samples[2].outdoor_c == 5.0

    def test_attach_outdoor_none_before_first_sample(self):
        """Climate samples before any outdoor reading stay unknown."""
        base = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
        samples = attach_outdoor(
            [ClimateSample(base.timestamp(), 18.0, 21.0, True)],
            [((base + timedelta(minutes=60)).timestamp(), 5.0)],
        )
        assert samples[0].outdoor_c is None


# ---------------------------------------------------------------------------
# Coordinator backfill end to end
# ---------------------------------------------------------------------------

def _climate_state(day, hh, mm, setpoint_f, current_f, action):
    return FakeState(
        "heat",
        datetime(2026, 11, day, hh, mm, tzinfo=timezone.utc),
        {
            "temperature": float(setpoint_f),
            "current_temperature": float(current_f),
            "hvac_action": action,
            "min_temp": 45,
            "max_temp": 92,
        },
    )


def _weather_state(day, hh, mm, temp_f):
    return FakeState(
        "cloudy",
        datetime(2026, 11, day, hh, mm, tzinfo=timezone.utc),
        {"temperature": float(temp_f), "temperature_unit": "°F"},
    )


def _nov_10_to_18_history():
    """Realistic Nov 10-18 history: 8 days x morning + afternoon
    warm-ups, Nov 18 morning sessions, Nov 18 midday mode flaps.
    The Nov 10 morning session has no outdoor coverage (first weather
    reading is Nov 10 12:00), so Fix D skips it: 17 episodes fed."""
    climate = []
    for day in range(10, 18):  # Nov 10-17
        climate += [
            _climate_state(day, 5, 0, 68, 68, "idle"),
            _climate_state(day, 5, 30, 71, 68, "heating"),
            _climate_state(day, 6, 0, 71, 69.5, "heating"),
            _climate_state(day, 6, 40, 71, 71, "heating"),
            _climate_state(day, 9, 0, 63, 71, "idle"),
            _climate_state(day, 12, 0, 63, 68, "idle"),
            _climate_state(day, 15, 30, 71, 65, "heating"),
            _climate_state(day, 16, 0, 71, 68, "heating"),
            _climate_state(day, 16, 30, 71, 70, "heating"),
            _climate_state(day, 16, 50, 71, 71, "heating"),
            _climate_state(day, 22, 0, 68, 71, "idle"),
        ]
    # Nov 18 morning (live observations: 05:08 67->69, 06:30 69->70,
    # 06:38 69->71, then warm to 71).
    climate += [
        _climate_state(18, 5, 8, 69, 67, "heating"),
        _climate_state(18, 5, 40, 69, 68, "heating"),
        _climate_state(18, 6, 20, 69, 69, "heating"),
        _climate_state(18, 6, 30, 70, 69, "heating"),
        _climate_state(18, 6, 38, 71, 69, "heating"),
        _climate_state(18, 7, 10, 71, 70, "heating"),
        _climate_state(18, 7, 30, 71, 71, "heating"),
        # Nov 18 midday mode flaps: 62 <-> 70 within minutes.
        _climate_state(18, 11, 0, 70, 65, "heating"),
        _climate_state(18, 11, 8, 62, 65.5, "heating"),
        _climate_state(18, 11, 20, 70, 65.5, "heating"),
        _climate_state(18, 11, 25, 62, 65.5, "idle"),
    ]
    weather = [
        _weather_state(10, 12, 0, 45),
        _weather_state(14, 12, 0, 40),
        _weather_state(18, 12, 0, 47),
    ]
    return {CLIMATE: climate, WEATHER: weather}


class TestThermalBackfill(_CoordinatorHarness):
    def setUp(self):
        super().setUp()
        self.coord._async_save_data = AsyncMock()
        self.coord.physics.sample_count = 0
        self.coord.entry.options[CONF_CLIMATE] = CLIMATE
        self.coord.entry.options[CONF_WEATHER_ENTITY] = WEATHER
        self.coord._backfill_cutoff = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)
        self.coord._fetch_significant_states = AsyncMock(
            return_value=_nov_10_to_18_history()
        )

    async def test_backfill_feeds_17_episodes_and_confident(self):
        """15 routine warm-ups + 2 Nov 18 morning sessions; the flaps
        add nothing, and the Nov 10 morning session is skipped (no
        outdoor coverage). The learner becomes confident (>= 5 episodes)."""
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 17
        assert self.coord.thermal_learner.is_confident is True
        rate = self.coord.thermal_learner.reference_heating_rate_f_per_h()
        assert rate is not None
        assert 0.5 < rate < 10.0
        assert self.coord._backfill_done is True
        assert self.coord._async_save_data.await_count >= 1

    async def test_backfill_queries_last_10_days_to_cutoff(self):
        """History window is (cutoff - 10 days) .. cutoff, both entities."""
        await self.coord._backfill_thermal_history()
        args = self.coord._fetch_significant_states.await_args.args
        assert args[0] == datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)
        assert args[1] == datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)
        assert args[2] == [CLIMATE, WEATHER]

    async def test_backfill_runs_once_only(self):
        """A second call does nothing: no second history fetch."""
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 1
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 1
        assert self.coord.thermal_learner.episode_count == 17

    async def test_backfill_skipped_when_model_has_episodes(self):
        """The live observer got there first: no fetch, flag set."""
        self.coord.thermal_learner.record_episode(70.0, 1.7, 5.0, 15.0)
        assert self.coord.thermal_learner.episode_count == 1
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 0
        assert self.coord.thermal_learner.episode_count == 1
        assert self.coord._backfill_done is True

    async def test_backfill_empty_history_is_not_a_crash(self):
        """No recorder history: 0 episodes, flag set, no retry loop."""
        self.coord._fetch_significant_states = AsyncMock(return_value={})
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord._backfill_done is True

    async def test_backfill_no_climate_entity(self):
        """No climate entity configured: nothing to read, flag set."""
        del self.coord.entry.options[CONF_CLIMATE]
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 0
        assert self.coord._backfill_done is True

    async def test_backfill_fetch_failure_is_not_a_crash(self):
        """Recorder error: logged, flag set, model untouched."""
        self.coord._fetch_significant_states = AsyncMock(
            side_effect=RuntimeError("recorder down")
        )
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord._backfill_done is True

    async def test_valveless_house_physics_learns_from_backfill(self):
        """Fix A: with no valve source at all the feed path passes
        valve_position=None, so the physics model learns too — its
        sample_count rises alongside the learner's episodes. (Before
        the fix the fallback fed 0.0 and the valve guard rejected
        every warm-up, so sample_count stayed 0.)"""
        self.coord.physics = ThermalPhysics(
            profile_data=HEATING_PROFILES[PROFILE_RADIATOR_NEW],
            learning_rate=0.1,
        )
        # Restore the real valve average (harness mocks it to 80.0):
        # the live buffer holds no points for historical windows.
        del self.coord.history_buffer.get_average_valve
        # No valve source anywhere (harness mocks this to 0.0).
        self.coord._get_valve_position = MagicMock(return_value=None)
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 17
        assert self.coord.physics.sample_count > 0

    async def test_valve_sensor_at_zero_percent_still_rejects(self):
        """Fix A guard: a real valve sensor reading 0% is NOT the
        valveless case — the 0.0 keeps the physics valve guard, so
        sample_count stays 0 while the learner still accumulates
        episodes from the same warm-ups."""
        self.coord.physics = ThermalPhysics(
            profile_data=HEATING_PROFILES[PROFILE_RADIATOR_NEW],
            learning_rate=0.1,
        )
        del self.coord.history_buffer.get_average_valve
        # A real valve source exists and reads 0% (harness default).
        self.coord._get_valve_position = MagicMock(return_value=0.0)
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 17
        assert self.coord.physics.sample_count == 0

    async def test_backfill_skips_session_with_no_outdoor_coverage(self):
        """Fix D: a session with no outdoor coverage is skipped, not
        fed with the fabricated 10.0 C placeholder. Morning warm-up
        before any outdoor reading + afternoon warm-up after one:
        only the afternoon episode is fed."""
        history = {
            CLIMATE: [
                # Morning warm-up, before the first outdoor reading.
                _climate_state(10, 5, 0, 68, 68, "idle"),
                _climate_state(10, 5, 30, 71, 68, "heating"),
                _climate_state(10, 6, 0, 71, 69.5, "heating"),
                _climate_state(10, 6, 40, 71, 71, "heating"),
                # Afternoon warm-up, after the first outdoor reading.
                _climate_state(10, 15, 30, 71, 65, "heating"),
                _climate_state(10, 16, 0, 71, 68, "heating"),
                _climate_state(10, 16, 50, 71, 71, "heating"),
                _climate_state(10, 22, 0, 68, 71, "idle"),
            ],
            WEATHER: [_weather_state(10, 12, 0, 45)],
        }
        self.coord._fetch_significant_states = AsyncMock(return_value=history)
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 1

    # -- Phase 8 Fix 1: never subtract an unmeasured deadtime --------

    def _use_real_physics(self):
        """Swap the harness's mocked physics for a real one (Radiator
        profile, deadtime 15 min, never measured) so the measured
        flag is meaningful."""
        self.coord.physics = ThermalPhysics(
            profile_data=HEATING_PROFILES[PROFILE_RADIATOR_NEW],
            learning_rate=0.1,
        )

    async def _feed_return_shape(self):
        """The Nov 17 15:32 return as one session: 20.7 min, 2.22 K
        rise (67 -> 71 degF). True rate 6.4 K/hr; with the 15-min
        Radiator deadtime subtracted it inflates to 23.4 K/hr and
        the rate cap rejects it — the exact v2.14.3 live failure."""
        base = datetime(2026, 11, 17, 15, 32, tzinfo=timezone.utc)
        return await self.coord._feed_warmup_session(
            start_ts=base.timestamp(),
            start_temp_c=fahrenheit_to_celsius(67),
            target_c=fahrenheit_to_celsius(71),
            end_temp_c=fahrenheit_to_celsius(71),
            end_ts=(base + timedelta(minutes=20.7)).timestamp(),
            outdoor_avg_c=5.0,
            outdoor_valid=True,
            analyze_deadtime=False,
            source="Backfilled",
        )

    async def test_unmeasured_deadtime_records_full_duration(self):
        """Phase 8 Fix 1: the deadtime was never measured (profile
        prior only), so it is passed as 0.0 — the effective duration
        is the full 20.7 min and the 6.4 K/hr episode is recorded."""
        self._use_real_physics()
        assert self.coord.physics.deadtime_measured is False
        assert await self._feed_return_shape() is True
        assert self.coord.thermal_learner.episode_count == 1

    async def test_measured_deadtime_still_subtracted(self):
        """Phase 8 Fix 1: once the analyzer measures a deadtime, it
        IS subtracted — the same session is then rejected by the rate
        cap (23.4 K/hr), and the learner says why."""
        self._use_real_physics()
        self.coord.physics.update_deadtime(15.0)
        assert self.coord.physics.deadtime_measured is True
        assert await self._feed_return_shape() is False
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord.thermal_learner.last_reject_reason == "rate"

    async def test_measured_deadtime_rejects_short_session(self):
        """Phase 8 Fix 1: a 12-min session with a measured 15-min
        deadtime has no effective heat-up — rejected as deadtime,
        not silently dropped."""
        self._use_real_physics()
        self.coord.physics.update_deadtime(15.0)
        base = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)
        recorded = await self.coord._feed_warmup_session(
            start_ts=base.timestamp(),
            start_temp_c=fahrenheit_to_celsius(68),
            target_c=fahrenheit_to_celsius(71),
            end_temp_c=fahrenheit_to_celsius(71),
            end_ts=(base + timedelta(minutes=12)).timestamp(),
            outdoor_avg_c=5.0,
            outdoor_valid=True,
            analyze_deadtime=False,
            source="Backfilled",
        )
        assert recorded is False
        assert self.coord.thermal_learner.episode_count == 0
        assert self.coord.thermal_learner.last_reject_reason == "deadtime"

    # -- Phase 8 Fix 3: versioned re-run ------------------------------

    async def test_versioned_rerun_resets_and_refeeds(self):
        """Phase 8 Fix 3: a v2.14.3 store (flag set, no version, 1
        stray episode) re-runs once: the learner is reset first, so
        the window is re-fed from a clean slate instead of
        double-counting the survivor."""
        self.coord._backfill_done = True
        self.coord.thermal_learner.record_episode(70.0, 1.7, 5.0, 15.0)
        assert self.coord.thermal_learner.episode_count == 1
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 1
        assert self.coord.thermal_learner.episode_count == 17
        assert self.coord._backfill_version == 1
        assert self.coord._backfill_done is True
        self.coord.physics.reset.assert_called_once()

    async def test_same_version_does_not_rerun(self):
        """Phase 8 Fix 3: the same version never re-runs, even with a
        learned model in place."""
        self.coord._backfill_done = True
        self.coord._backfill_version = 1
        await self.coord._backfill_thermal_history()
        assert self.coord._fetch_significant_states.await_count == 0
        assert self.coord.thermal_learner.episode_count == 0

    # -- Phase 8 Fix 2 (end to end) + Fix 4 ---------------------------

    async def test_backfill_survives_ten_minute_idle_gap(self):
        """Phase 8 Fix 2 end to end: the backfill's 15-min grace keeps
        a warm-up with a 10-min furnace cycle in one session — one
        episode, not two fragments (the live 5-min grace would split
        it at the 16:10 idle sample)."""
        history = {
            CLIMATE: [
                _climate_state(10, 15, 30, 71, 65, "heating"),
                _climate_state(10, 16, 0, 71, 68, "heating"),
                _climate_state(10, 16, 5, 71, 68.5, "idle"),
                _climate_state(10, 16, 10, 71, 68.5, "idle"),
                _climate_state(10, 16, 15, 71, 68.5, "heating"),
                _climate_state(10, 16, 50, 71, 71, "heating"),
                _climate_state(10, 22, 0, 68, 71, "idle"),
            ],
            WEATHER: [_weather_state(10, 4, 0, 45)],
        }
        self.coord._fetch_significant_states = AsyncMock(return_value=history)
        await self.coord._backfill_thermal_history()
        assert self.coord.thermal_learner.episode_count == 1

    async def test_backfill_summary_is_warning_with_rejection_counts(self):
        """Phase 8 Fix 4: the backfill summary logs at WARNING (the
        HA UI log only surfaces WARNING+) with a rejection breakdown
        — one session each rejected by deadtime, rate cap, and rise."""
        history = {
            CLIMATE: [
                # Good session: 70 min, 1.67 K rise -> recorded.
                _climate_state(10, 5, 0, 68, 68, "idle"),
                _climate_state(10, 5, 30, 71, 68, "heating"),
                _climate_state(10, 6, 0, 71, 69.5, "heating"),
                _climate_state(10, 6, 40, 71, 71, "heating"),
                # 12-min session, 1.67 K rise -> deadtime (15 - 12 < 0).
                _climate_state(10, 8, 0, 71, 68, "heating"),
                _climate_state(10, 8, 12, 71, 71, "heating"),
                # 21-min session, 2.22 K rise -> rate (22.2 K/hr).
                _climate_state(10, 10, 0, 71, 67, "heating"),
                _climate_state(10, 10, 21, 71, 71, "heating"),
                # 30-min session, 0.28 K rise -> rise too small.
                _climate_state(10, 12, 0, 71, 70.5, "heating"),
                _climate_state(10, 12, 30, 71, 71, "heating"),
                _climate_state(10, 22, 0, 68, 71, "idle"),
            ],
            WEATHER: [_weather_state(10, 4, 0, 45)],
        }
        self.coord._fetch_significant_states = AsyncMock(return_value=history)
        with self.assertLogs(
            "custom_components.preheat.coordinator", level="WARNING"
        ) as cm:
            await self.coord._backfill_thermal_history()
        summary = [r for r in cm.output if "Thermal backfill:" in r]
        assert len(summary) == 1
        assert "1 episode(s) from 4 warm-up session(s)" in summary[0]
        assert "rejected: 1 deadtime, 1 rate, 1 other" in summary[0]
        assert self.coord.thermal_learner.episode_count == 1

    async def test_fetch_seam_uses_recorder_api(self):
        """The history seam calls get_significant_states through the
        recorder instance's executor job, as the arrival bootstrap does."""
        # Save whatever another test module installed here at import time
        # (test_retroactive_learning.py) and restore it afterwards.
        prev_recorder = sys.modules.get("homeassistant.components.recorder")

        def _restore_recorder():
            if prev_recorder is None:
                sys.modules.pop("homeassistant.components.recorder", None)
            else:
                sys.modules["homeassistant.components.recorder"] = prev_recorder

        sys.modules["homeassistant.components.recorder"] = MagicMock()
        self.addCleanup(_restore_recorder)
        mock_recorder = sys.modules["homeassistant.components.recorder"]
        mock_instance = MagicMock()
        mock_recorder.get_instance.return_value = mock_instance
        mock_recorder.history = MagicMock()
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        future.set_result(_nov_10_to_18_history())
        mock_instance.async_add_executor_job = MagicMock(return_value=future)
        start = datetime(2026, 11, 8, 20, 0, tzinfo=timezone.utc)
        end = datetime(2026, 11, 18, 20, 0, tzinfo=timezone.utc)
        # Call the real seam (not the AsyncMock from setUp).
        data = await self.coord.__class__._fetch_significant_states(
            self.coord, start, end, [CLIMATE, WEATHER]
        )
        assert set(data.keys()) == {CLIMATE, WEATHER}
        mock_instance.async_add_executor_job.assert_called_once()
        call_args = mock_instance.async_add_executor_job.call_args.args
        assert call_args[0] is mock_recorder.history.get_significant_states
        assert call_args[2] == start
        assert call_args[3] == end
        assert call_args[4] == [CLIMATE, WEATHER]


class TestBackfillScheduling(_CoordinatorHarness):
    def setUp(self):
        super().setUp()
        self.coord._store.async_load = AsyncMock(return_value={})

    async def test_fresh_install_schedules_backfill(self):
        """Empty store: the backfill is scheduled (cutoff captured).
        The arrival bootstrap is scheduled alongside it — both at 30 s,
        so call_later fires twice on a fresh install."""
        await self.coord.async_load_data()
        assert self.coord._backfill_cutoff is not None
        assert self.coord._backfill_done is False
        assert self.hass.loop.call_later.call_count == 2
        for call in self.hass.loop.call_later.call_args_list:
            assert call.args[0] == 30

    async def test_v2143_store_schedules_rerun(self):
        """Phase 8 Fix 3: a v2.14.3 store (flag set, no version)
        schedules the backfill once — the recorded version (0) is
        older than BACKFILL_VERSION, so the fixed backfill re-runs
        after the upgrade instead of being blocked by the flag."""
        self.coord._store.async_load = AsyncMock(
            return_value={"backfill_done": True}
        )
        await self.coord.async_load_data()
        assert self.coord._backfill_done is True
        assert self.coord._backfill_version == 0
        assert self.coord._backfill_cutoff is not None
        assert self.hass.loop.call_later.call_count == 2

    async def test_current_version_does_not_schedule(self):
        """Phase 8 Fix 3: a store already at the current backfill
        version does not schedule it (cutoff stays None). Only the
        arrival bootstrap is scheduled."""
        self.coord._store.async_load = AsyncMock(
            return_value={
                "backfill_done": True,
                "backfill_version": BACKFILL_VERSION,
            }
        )
        await self.coord.async_load_data()
        assert self.coord._backfill_done is True
        assert self.coord._backfill_version == BACKFILL_VERSION
        assert self.coord._backfill_cutoff is None
        assert self.hass.loop.call_later.call_count == 1
