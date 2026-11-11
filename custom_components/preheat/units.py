"""Temperature unit detection and conversion.

Fork addition (v2.13.0-fork.1): upstream assumed every temperature it read
was degrees Celsius. Fahrenheit thermostats — notably the Ecobee over
HomeKit, which reports °F numbers with no ``temperature_unit`` attribute —
were read as °C (a 71°F setpoint became 71°C and was displayed as 159.8°F).

All internal math stays in °C. These helpers convert at the ingestion
boundary only, so the planner, physics and learning modules are untouched.
"""

from __future__ import annotations

CELSIUS = "°C"
FAHRENHEIT = "°F"

# A climate entity whose min_temp is above this is Fahrenheit: °C
# thermostats bottom out near 7°C, °F ones near 45°F. Same heuristic the
# household's Climate Control blueprint already uses (verified in service).
FAHRENHEIT_MIN_TEMP_THRESHOLD = 30.0

# Indoor air-temperature readings above this are Fahrenheit: a room at
# 40°C (104°F) is a sauna; a room at 40°F (4.4°C) is freezing. Real indoor
# readings sit in 15–30°C / 59–86°F, so 40 splits them cleanly.
FAHRENHEIT_INDOOR_THRESHOLD = 40.0

# Plausible Fahrenheit comfort setpoints persisted by the unit-blind
# upstream (a home thermostat's setpoint). Used to recognise a stored
# value that needs migration, and to reject implausible ones.
FAHRENHEIT_SETPOINT_MIN = 50.0
FAHRENHEIT_SETPOINT_MAX = 95.0


def normalize_unit(unit) -> str | None:
    """Normalize a unit value (°C/°F, string or enum) or return None."""
    if unit is None:
        return None
    if isinstance(unit, str):
        text = unit.strip()
        if text in ("°C", "C", "celsius", "Celsius"):
            return CELSIUS
        if text in ("°F", "F", "fahrenheit", "Fahrenheit"):
            return FAHRENHEIT
        return None
    value = getattr(unit, "value", None)
    if isinstance(value, str):
        return normalize_unit(value)
    return None


def fahrenheit_to_celsius(value: float) -> float:
    """Convert a Fahrenheit temperature to Celsius."""
    return (value - 32.0) * 5.0 / 9.0


def celsius_to_fahrenheit(value: float) -> float:
    """Convert a Celsius temperature to Fahrenheit."""
    return value * 9.0 / 5.0 + 32.0


def to_celsius(value: float, unit: str | None) -> float:
    """Convert ``value`` to Celsius if ``unit`` is Fahrenheit, else pass through."""
    if normalize_unit(unit) == FAHRENHEIT:
        return fahrenheit_to_celsius(value)
    return value


def detect_climate_unit(attrs: dict | None, ha_unit: str | None = None) -> str:
    """Detect a climate entity's temperature unit from its attributes.

    1. Prefer the ``temperature_unit`` attribute.
    2. Fall back to the min_temp heuristic: min_temp > 30 means °F.
    3. Fall back to the Home Assistant unit system, if given.
    4. Default to °C (upstream behaviour) when nothing is known.
    """
    attrs = attrs or {}
    unit = normalize_unit(attrs.get("temperature_unit"))
    if unit:
        return unit
    min_temp = attrs.get("min_temp")
    if min_temp is not None:
        try:
            if float(min_temp) > FAHRENHEIT_MIN_TEMP_THRESHOLD:
                return FAHRENHEIT
        except (TypeError, ValueError):
            pass
    if normalize_unit(ha_unit) == FAHRENHEIT:
        return FAHRENHEIT
    return CELSIUS


def detect_sensor_unit(
    attrs: dict | None, value: float | None = None, ha_unit: str | None = None
) -> str:
    """Detect a temperature sensor's unit from its attributes.

    1. Prefer the ``unit_of_measurement`` attribute.
    2. Fall back to the indoor range heuristic: a reading above 40 is °F
       (a room at 40°C is a sauna; at 40°F it is freezing).
    3. Fall back to the Home Assistant unit system, if given.
    4. Default to °C (upstream behaviour) when nothing is known.
    """
    attrs = attrs or {}
    unit = normalize_unit(attrs.get("unit_of_measurement"))
    if unit:
        return unit
    if value is not None:
        try:
            if float(value) > FAHRENHEIT_INDOOR_THRESHOLD:
                return FAHRENHEIT
        except (TypeError, ValueError):
            pass
    if normalize_unit(ha_unit) == FAHRENHEIT:
        return FAHRENHEIT
    return CELSIUS


def detect_outdoor_unit(attrs: dict | None, ha_unit: str | None = None) -> str:
    """Detect an outdoor temperature source's unit from its attributes.

    Outdoor readings overlap in °F and °C (0°F and 0°C are both ordinary),
    so no range heuristic is safe here. Prefer the explicit unit attribute
    (``unit_of_measurement`` on sensors, ``temperature_unit`` on weather
    entities), then the Home Assistant unit system, then °C.
    """
    attrs = attrs or {}
    unit = normalize_unit(
        attrs.get("unit_of_measurement") or attrs.get("temperature_unit")
    )
    if unit:
        return unit
    if normalize_unit(ha_unit) == FAHRENHEIT:
        return FAHRENHEIT
    return CELSIUS


def hass_temperature_unit(hass) -> str | None:
    """Home Assistant's configured temperature unit, or None if unknown.

    Defensive by design: in tests ``hass`` is a mock, whose attribute
    chain returns mocks — those normalise to None instead of raising.
    """
    try:
        return normalize_unit(hass.config.units.temperature_unit)
    except Exception:  # noqa: BLE001 - any shape of hass/config is possible in tests
        return None


def hass_uses_fahrenheit(hass) -> bool:
    """True when Home Assistant's unit system is Fahrenheit."""
    return hass_temperature_unit(hass) == FAHRENHEIT
