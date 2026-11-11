"""
Fahrenheit regression tests (fork v2.13.0-fork.1).

Upstream v2.12.2 assumed every temperature it read was degrees Celsius.
A Fahrenheit thermostat — the Ecobee over HomeKit reports °F numbers with
no ``temperature_unit`` attribute — had its 71°F setpoint read as 71°C and
displayed as 159.8°F, and the learned heating rate was skewed ~1.8x.

These tests fail on upstream and pass on the fork. All internal values
are °C; the fix converts at the ingestion boundary only.
"""
import sys
import os
import unittest
import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

# Add parent dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- MOCK Home Assistant (mirrors tests/test_coordinator.py) ---
import types

ha = types.ModuleType("homeassistant")
ha.__path__ = []
sys.modules["homeassistant"] = ha

sys.modules["homeassistant.core"] = MagicMock()

# Config-entry flow shims (mirrors tests/test_config_flow.py): the flow
# classes must subclass REAL classes. Subclassing a MagicMock breaks the
# async step methods, so config_flow would import but its steps would
# return mocks instead of coroutines.
mock_config_entries = MagicMock()


class _MockConfigFlow:
    """Shim for ConfigFlow."""

    def __init__(self, *args, **kwargs):
        self.context = {}

    def __init_subclass__(cls, **kwargs):
        pass

    async def async_set_unique_id(self, *args, **kwargs):
        pass

    def _abort_if_unique_id_configured(self, *args, **kwargs):
        pass


class _MockOptionsFlow:
    """Shim for OptionsFlow."""

    def __init__(self, config_entry):
        self.config_entry = config_entry


mock_config_entries.ConfigFlow = _MockConfigFlow
mock_config_entries.OptionsFlow = _MockOptionsFlow
sys.modules["homeassistant.config_entries"] = mock_config_entries
sys.modules["homeassistant.data_entry_flow"] = MagicMock()
sys.modules["homeassistant.const"] = MagicMock()
sys.modules["homeassistant.util"] = MagicMock()
mock_dt = MagicMock()
mock_dt.UTC = timezone.utc
mock_dt.utcnow.side_effect = lambda: datetime.now(timezone.utc)
sys.modules["homeassistant.util.dt"] = mock_dt
sys.modules["homeassistant.helpers"] = MagicMock()
sys.modules["homeassistant.helpers.event"] = MagicMock()
sys.modules["homeassistant.helpers.storage"] = MagicMock()
sys.modules["homeassistant.helpers.issue_registry"] = MagicMock()
sys.modules["homeassistant.exceptions"] = MagicMock()


class MockDataUpdateCoordinator:
    def __init__(self, hass, logger, name, update_interval, **kwargs):
        self.hass = hass
        self.logger = logger
        self.name = name
        self.update_interval = update_interval
        self.data = None

    async def _async_update_data(self):
        pass

    def async_add_listener(self, *args):
        pass

    async def async_refresh(self):
        pass

    async def async_request_refresh(self):
        pass

    def __class_getitem__(cls, item):
        return cls


class MockUpdateFailed(Exception):
    pass


mock_duc_mod = MagicMock()
mock_duc_mod.DataUpdateCoordinator = MockDataUpdateCoordinator
mock_duc_mod.UpdateFailed = MockUpdateFailed
sys.modules["homeassistant.helpers.update_coordinator"] = mock_duc_mod

from custom_components.preheat.coordinator import PreheatingCoordinator
from custom_components.preheat.const import (
    CONF_CLIMATE,
    CONF_TEMPERATURE,
    CONF_OUTDOOR_TEMP,
    CONF_WEATHER_ENTITY,
    CONF_COMFORT_FALLBACK,
)
from custom_components.preheat import units


def _climate_state(temperature, current_temperature, min_temp, max_temp,
                   temperature_unit=None):
    """Fake climate entity state, Ecobee-over-HomeKit shaped by default."""
    attrs = {
        "temperature": temperature,
        "current_temperature": current_temperature,
        "min_temp": min_temp,
        "max_temp": max_temp,
    }
    if temperature_unit is not None:
        attrs["temperature_unit"] = temperature_unit
    return SimpleNamespace(state="heat", attributes=attrs)


def _sensor_state(value, unit=None):
    attrs = {}
    if unit is not None:
        attrs["unit_of_measurement"] = unit
    return SimpleNamespace(state=str(value), attributes=attrs)


def _weather_state(temperature, temperature_unit=None):
    attrs = {"temperature": temperature}
    if temperature_unit is not None:
        attrs["temperature_unit"] = temperature_unit
    return SimpleNamespace(state="sunny", attributes=attrs)


class TestUnitHelpers(unittest.TestCase):
    """Pure unit-detection / conversion helpers (no coordinator)."""

    def test_fahrenheit_to_celsius(self):
        self.assertAlmostEqual(units.fahrenheit_to_celsius(71), 21.67, places=2)
        self.assertAlmostEqual(units.fahrenheit_to_celsius(32), 0.0, places=2)
        self.assertAlmostEqual(units.fahrenheit_to_celsius(212), 100.0, places=2)

    def test_to_celsius_passthrough(self):
        self.assertEqual(units.to_celsius(21.5, "°C"), 21.5)
        self.assertEqual(units.to_celsius(21.5, None), 21.5)

    def test_detect_climate_unit_prefers_attribute(self):
        # Explicit attribute wins even if min_temp looks Celsius.
        self.assertEqual(
            units.detect_climate_unit(
                {"temperature_unit": "°F", "min_temp": 7, "max_temp": 35}
            ),
            "°F",
        )
        self.assertEqual(
            units.detect_climate_unit(
                {"temperature_unit": "°C", "min_temp": 45, "max_temp": 92}
            ),
            "°C",
        )

    def test_detect_climate_unit_min_temp_heuristic(self):
        # Ecobee-like: no attribute, min 45 -> °F.
        self.assertEqual(
            units.detect_climate_unit({"min_temp": 45, "max_temp": 92}), "°F"
        )
        # Metric thermostat: min 7 -> °C.
        self.assertEqual(
            units.detect_climate_unit({"min_temp": 7, "max_temp": 35}), "°C"
        )

    def test_detect_climate_unit_defaults_celsius(self):
        self.assertEqual(units.detect_climate_unit({}), "°C")
        self.assertEqual(units.detect_climate_unit(None), "°C")

    def test_detect_sensor_unit_prefers_attribute(self):
        self.assertEqual(
            units.detect_sensor_unit({"unit_of_measurement": "°F"}, 71), "°F"
        )
        self.assertEqual(
            units.detect_sensor_unit({"unit_of_measurement": "°C"}, 21.5), "°C"
        )

    def test_detect_sensor_unit_range_heuristic(self):
        # No attribute: a 71 indoor reading is °F, a 21.5 one is °C.
        self.assertEqual(units.detect_sensor_unit({}, 71), "°F")
        self.assertEqual(units.detect_sensor_unit({}, 21.5), "°C")

    def test_detect_outdoor_unit_no_range_heuristic(self):
        # Outdoor readings overlap in °F/°C, so a bare value must NOT
        # be guessed from its magnitude: 30 with no other info is °C.
        self.assertEqual(units.detect_outdoor_unit({}, 30), "°C")
        self.assertEqual(
            units.detect_outdoor_unit({"temperature_unit": "°F"}, 30), "°F"
        )
        self.assertEqual(
            units.detect_outdoor_unit({"unit_of_measurement": "°F"}, 30), "°F"
        )
        self.assertEqual(units.detect_outdoor_unit({}, ha_unit="°F"), "°F")

    def test_hass_temperature_unit_mock_safe(self):
        # A MagicMock hass (as in tests) must normalise to None, not crash.
        self.assertIsNone(units.hass_temperature_unit(MagicMock()))
        hass = MagicMock()
        hass.config.units.temperature_unit = "°F"
        self.assertEqual(units.hass_temperature_unit(hass), "°F")
        self.assertTrue(units.hass_uses_fahrenheit(hass))
        hass.config.units.temperature_unit = "°C"
        self.assertFalse(units.hass_uses_fahrenheit(hass))


class TestCoordinatorFahrenheit(unittest.IsolatedAsyncioTestCase):
    """Coordinator ingestion points with Fahrenheit sources."""

    def _make_coordinator(self, states, data=None, options=None):
        hass = MagicMock()
        hass.states.get.side_effect = lambda entity_id: states.get(entity_id)
        entry = MagicMock()
        entry.entry_id = "test"
        entry.title = "Test"
        entry.data = data or {}
        entry.options = options or {}
        with patch(
            "custom_components.preheat.coordinator.PreheatingCoordinator._setup_listeners"
        ), patch(
            "custom_components.preheat.coordinator.PreheatingCoordinator.async_load_data"
        ):
            coord = PreheatingCoordinator(hass, entry)
        return coord, hass

    async def test_target_setpoint_ecobee_no_unit_attr(self):
        """The reported bug: 71°F with no temperature_unit must read 21.7°C."""
        states = {
            "climate.living_room_thermostat": _climate_state(
                temperature=71, current_temperature=71, min_temp=45, max_temp=92
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.living_room_thermostat"}
        )
        result = await coord._get_target_setpoint()
        self.assertAlmostEqual(result, 21.7, places=1)
        self.assertLess(result, 30.0)  # Never the raw 71 (== 159.8°F display).

    async def test_target_setpoint_with_unit_attr_f(self):
        states = {
            "climate.test": _climate_state(
                temperature=71, current_temperature=70, min_temp=45, max_temp=92,
                temperature_unit="°F",
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        result = await coord._get_target_setpoint()
        self.assertAlmostEqual(result, 21.7, places=1)

    async def test_target_setpoint_celsius_control(self):
        """Metric thermostat behaviour is unchanged."""
        states = {
            "climate.test": _climate_state(
                temperature=21.5, current_temperature=21.0, min_temp=7, max_temp=35,
                temperature_unit="°C",
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        result = await coord._get_target_setpoint()
        self.assertAlmostEqual(result, 21.5, places=1)

    async def test_target_setpoint_below_comfort_min_falls_back(self):
        """60°F is 15.6°C — below the 19°C comfort min — so it must fall
        back to 21.0°C. Upstream saw raw 60 > 19 and returned 60."""
        states = {
            "climate.test": _climate_state(
                temperature=60, current_temperature=60, min_temp=45, max_temp=92
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        result = await coord._get_target_setpoint()
        self.assertAlmostEqual(result, 21.0, places=1)

    async def test_operative_temp_dedicated_f_sensor(self):
        states = {
            "sensor.room_temp": _sensor_state(71, "°F"),
            "climate.test": _climate_state(
                temperature=71, current_temperature=71, min_temp=45, max_temp=92
            ),
        }
        coord, _ = self._make_coordinator(
            states,
            data={
                CONF_CLIMATE: "climate.test",
                CONF_TEMPERATURE: "sensor.room_temp",
            },
        )
        result = await coord._get_operative_temperature()
        self.assertAlmostEqual(result, 21.7, places=1)

    async def test_operative_temp_climate_current_f(self):
        states = {
            "climate.test": _climate_state(
                temperature=71, current_temperature=71, min_temp=45, max_temp=92
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        result = await coord._get_operative_temperature()
        self.assertAlmostEqual(result, 21.7, places=1)

    async def test_operative_temp_celsius_control(self):
        states = {
            "climate.test": _climate_state(
                temperature=21.5, current_temperature=21.0, min_temp=7,
                max_temp=35, temperature_unit="°C",
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        result = await coord._get_operative_temperature()
        self.assertAlmostEqual(result, 21.0, places=1)

    async def test_operative_temp_prefers_dedicated_sensor(self):
        """Dedicated sensor wins over the climate entity's current temp."""
        states = {
            "sensor.room_temp": _sensor_state(70, "°F"),
            "climate.test": _climate_state(
                temperature=71, current_temperature=71, min_temp=45, max_temp=92
            ),
        }
        coord, _ = self._make_coordinator(
            states,
            data={
                CONF_CLIMATE: "climate.test",
                CONF_TEMPERATURE: "sensor.room_temp",
            },
        )
        result = await coord._get_operative_temperature()
        self.assertAlmostEqual(result, 21.1, places=1)

    async def test_gradient_not_skewed_by_fahrenheit(self):
        """A 0.7°F dip is a ~0.39°C dip, not 0.7°C. The window-open
        threshold is -0.4K in 5 min, so the fixed code must NOT flag it —
        upstream's unit-blind read saw -0.7K and falsely tripped window
        detection. (A 1°F dip is -0.56K, which legitimately exceeds the
        threshold even when converted, so it does not discriminate.)"""
        states = {
            "climate.test": _climate_state(
                temperature=71, current_temperature=72, min_temp=45, max_temp=92
            )
        }
        coord, hass = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        now = datetime(2026, 11, 10, 12, 0, 0, tzinfo=timezone.utc)
        op1 = await coord._get_operative_temperature()
        coord._track_temperature_gradient(op1, now)
        # Room cools 0.7°F over 5 minutes.
        states["climate.test"] = _climate_state(
            temperature=71, current_temperature=71.3, min_temp=45, max_temp=92
        )
        op2 = await coord._get_operative_temperature()
        self.assertAlmostEqual(op1 - op2, 0.39, places=1)
        coord._track_temperature_gradient(op2, now + timedelta(minutes=5))
        self.assertFalse(coord.window_open_detected)

    async def test_stored_setpoint_migration(self):
        """A 71 persisted by the unit-blind upstream loads as 21.7°C."""
        states = {
            "climate.test": _climate_state(
                temperature=71, current_temperature=71, min_temp=45, max_temp=92
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        coord._last_comfort_setpoint = 71.0
        coord._migrate_stored_setpoint()
        self.assertAlmostEqual(coord._last_comfort_setpoint, 21.7, places=1)
        # And the target setpoint then uses the migrated value, not 71.
        result = await coord._get_target_setpoint()
        self.assertAlmostEqual(result, 21.7, places=1)

    async def test_stored_setpoint_migration_celsius_untouched(self):
        """A stored 21.5 with a metric climate entity is left alone."""
        states = {
            "climate.test": _climate_state(
                temperature=21.5, current_temperature=21.0, min_temp=7,
                max_temp=35, temperature_unit="°C",
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        coord._last_comfort_setpoint = 21.5
        coord._migrate_stored_setpoint()
        self.assertAlmostEqual(coord._last_comfort_setpoint, 21.5, places=1)

    async def test_stored_setpoint_migration_needs_f_evidence(self):
        """A stored 71 with a metric climate entity is NOT converted:
        migration requires positive evidence the climate is Fahrenheit."""
        states = {
            "climate.test": _climate_state(
                temperature=21.5, current_temperature=21.0, min_temp=7,
                max_temp=35, temperature_unit="°C",
            )
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_CLIMATE: "climate.test"}
        )
        coord._last_comfort_setpoint = 71.0
        coord._migrate_stored_setpoint()
        self.assertAlmostEqual(coord._last_comfort_setpoint, 71.0, places=1)

    async def test_outdoor_temp_weather_fahrenheit(self):
        states = {
            "weather.home": _weather_state(32, "°F"),
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_WEATHER_ENTITY: "weather.home"}
        )
        result = await coord._get_outdoor_temp_current()
        self.assertAlmostEqual(result, 0.0, places=1)

    async def test_outdoor_temp_dedicated_sensor_f(self):
        states = {
            "sensor.outdoor": _sensor_state(41, "°F"),
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_OUTDOOR_TEMP: "sensor.outdoor"}
        )
        result = await coord._get_outdoor_temp_current()
        self.assertAlmostEqual(result, 5.0, places=1)

    async def test_outdoor_temp_celsius_control(self):
        states = {
            "weather.home": _weather_state(5, "°C"),
        }
        coord, _ = self._make_coordinator(
            states, data={CONF_WEATHER_ENTITY: "weather.home"}
        )
        result = await coord._get_outdoor_temp_current()
        self.assertAlmostEqual(result, 5.0, places=1)


class TestConfigFlowFahrenheit(unittest.IsolatedAsyncioTestCase):
    """Comfort Fallback selector in the options flow on Imperial systems."""

    async def asyncSetUp(self):
        # Inspect the selector mock through the config_flow module itself:
        # `from homeassistant.helpers import selector` can resolve to a
        # different MagicMock child depending on which test file's parent
        # mock is installed, so it is not reliably the object config_flow
        # bound at import time. The module attribute always is.
        from custom_components.preheat import config_flow as cf_module

        self.selector_mod = cf_module.selector
        self.hass = MagicMock()
        self.hass.config.units.temperature_unit = "°F"

    def _make_flow(self, options=None, data=None):
        from custom_components.preheat.config_flow import PreheatingOptionsFlow

        entry = MagicMock()
        entry.unique_id = "test-entry"
        entry.data = data or {}
        entry.options = options or {}
        flow = PreheatingOptionsFlow(entry)
        flow.hass = self.hass
        flow.async_create_entry = MagicMock(
            return_value={"type": "create_entry"}
        )
        flow.async_show_form = MagicMock(
            side_effect=lambda **kwargs: {"type": "form", **kwargs}
        )
        flow.add_suggested_values_to_schema = MagicMock(
            side_effect=lambda schema, suggestions: schema
        )
        return flow

    async def test_fallback_selector_shown_in_fahrenheit(self):
        """On an Imperial system the fallback selector is 59–77°F."""
        self.selector_mod.NumberSelectorConfig.reset_mock()
        flow = self._make_flow()
        result = await flow.async_step_init()
        self.assertEqual(result["type"], "form")
        f_calls = [
            call
            for call in self.selector_mod.NumberSelectorConfig.call_args_list
            if call.kwargs.get("unit_of_measurement") == "°F"
        ]
        self.assertTrue(f_calls, "no °F fallback selector was built")
        kwargs = f_calls[0].kwargs
        self.assertEqual(kwargs["min"], 59.0)
        self.assertEqual(kwargs["max"], 77.0)
        # Suggested value: stored 21.0°C is shown as 70°F.
        suggestions = flow.add_suggested_values_to_schema.call_args.args[1]
        self.assertEqual(suggestions[CONF_COMFORT_FALLBACK], 70)

    async def test_fallback_submit_converts_back_to_celsius(self):
        """Submitting 70°F stores 21.1°C."""
        flow = self._make_flow()
        result = await flow.async_step_init({CONF_COMFORT_FALLBACK: 70})
        self.assertEqual(result["type"], "create_entry")
        stored = flow.async_create_entry.call_args.kwargs["data"]
        self.assertAlmostEqual(stored[CONF_COMFORT_FALLBACK], 21.1, places=1)

    async def test_fallback_metric_system_unchanged(self):
        """On a metric system the selector stays °C and nothing converts."""
        self.hass.config.units.temperature_unit = "°C"
        self.selector_mod.NumberSelectorConfig.reset_mock()
        flow = self._make_flow()
        result = await flow.async_step_init()
        self.assertEqual(result["type"], "form")
        f_calls = [
            call
            for call in self.selector_mod.NumberSelectorConfig.call_args_list
            if call.kwargs.get("unit_of_measurement") == "°F"
        ]
        self.assertFalse(f_calls, "unexpected °F selector on metric system")
        suggestions = flow.add_suggested_values_to_schema.call_args.args[1]
        self.assertEqual(suggestions[CONF_COMFORT_FALLBACK], 21.0)


if __name__ == "__main__":
    unittest.main()
