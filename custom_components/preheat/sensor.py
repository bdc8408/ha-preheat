"""Sensor platform for Preheat."""
from __future__ import annotations

from typing import Any
from datetime import datetime, timedelta

from homeassistant.components.sensor import (
    SensorDeviceClass,
    SensorEntity,
    SensorStateClass,
)
from homeassistant.helpers.entity import Entity
from homeassistant.util import dt as dt_util
from .house_collector import HouseArrivalCollector, get_percentile
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from homeassistant.const import UnitOfTemperature
from .const import DOMAIN, VERSION, CONF_CLIMATE, CONF_TEMPERATURE, ATTR_DECISION_TRACE
from .coordinator import PreheatingCoordinator

async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up sensors."""
    if entry.unique_id == "preheat_system":
        if house := hass.data[DOMAIN].get("house"):
            global_sensors = [
                PreheatHouseNextArrivalSensor(house),
                PreheatHouseArrivalConfidenceSensor(house),
                PreheatHouseArrivalWindowSensor(house),
            ]
            async_add_entities(global_sensors)
        return

    coordinator: PreheatingCoordinator = entry.runtime_data
    
    sensors = [
        PreheatStatusSensor(coordinator, entry),
        NextEventSensor(coordinator, entry),
        ThermalModelSensor(coordinator, entry),
        PhysicsSensor(coordinator, entry, "mass_factor", "mass_factor", "min/K"),
        PhysicsSensor(coordinator, entry, "loss_factor", "loss_factor", "min/K"),
        PreheatConfidenceSensor(coordinator, entry),
        PreheatOptimalStopTimeSensor(coordinator, entry),
        # v3.0 Spec Entities (Non-Breaking Additions)
        PreheatNextStartSensor(coordinator, entry),
        PreheatDurationSensor(coordinator, entry),
        PreheatTargetTempSensor(coordinator, entry),
        PreheatNextArrivalSensor(coordinator, entry),
        PreheatNextSessionEndSensor(coordinator, entry),
        # Fork Phase 4: learned schedule + explicit thermal model (DRIVE mode)
        # + away mode.
        PreheatLearnedOccupancySensor(coordinator, entry),
        PreheatLearnedWakeSensor(coordinator, entry),
        PreheatLearnedBedtimeSensor(coordinator, entry),
        PreheatLearnedHeatingRateSensor(coordinator, entry),
        PreheatLearnedCoolingRateSensor(coordinator, entry),
        PreheatLearnedDeadtimeSensor(coordinator, entry),
        PreheatLearnedHeatLossSensor(coordinator, entry),
        PreheatAwaySetpointSensor(coordinator, entry),
        PreheatAwayReturnStartSensor(coordinator, entry),
    ]

    async_add_entities(sensors)

class PreheatBaseSensor(CoordinatorEntity[PreheatingCoordinator], SensorEntity):
    """Base sensor."""
    _attr_has_entity_name = True


    def __init__(self, coordinator: PreheatingCoordinator, entry: ConfigEntry) -> None:
        super().__init__(coordinator)
        self._entry = entry
        self._attr_device_info = {
            "identifiers": {(DOMAIN, entry.entry_id)},
            "name": entry.title,
            "manufacturer": "Ecronika",
            "model": "Intelligent Preheating v2",
            "sw_version": VERSION,
        }

class PreheatStatusSensor(PreheatBaseSensor):
    """Main Status Sensor."""
    _attr_translation_key = "status"
    _attr_device_class = SensorDeviceClass.ENUM
    _attr_options = ["idle", "preheating"]

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_status"

    @property
    def native_value(self) -> str:
        if self.coordinator.data.preheat_active:
            return "preheating"
        return "idle"
    
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        physics = self.coordinator.physics
        return {
            "target_temp": data.target_setpoint,
            "current_temp": data.operative_temp,
            "predicted_duration": round(data.predicted_duration, 1),
            "lead_time": round(data.lead_time, 1),
            "confidence": physics.get_confidence(),
            "avg_error": round(physics.avg_error, 2),
            "sample_count": physics.sample_count,
            "window_open": data.window_open,
            "learned_setpoint": data.last_comfort_setpoint,
            "deadtime_min": round(data.deadtime, 1),
            "health_score": physics.health_score,
            ATTR_DECISION_TRACE: data.decision_trace,
            "pattern_data": data.detected_modes,
            "next_start_time": data.next_start_time.isoformat() if data.next_start_time else None,
            "next_arrival": data.next_arrival.isoformat() if data.next_arrival else None,
            "next_departure": data.next_departure.isoformat() if data.next_departure else None,
            "optimal_stop_time": data.optimal_stop_time.isoformat() if data.optimal_stop_time else None,
            # --- NEU (2.11.3): Verknüpfungen für das UI (Smart Setpoint Card) ---
            "climate_entity": self.coordinator._get_conf(CONF_CLIMATE),
            "operative_sensor": self.coordinator._get_conf(CONF_TEMPERATURE),
            "integration_version": VERSION,
        }

class NextEventSensor(PreheatBaseSensor):
    """Next Planned Event."""
    _attr_translation_key = "next_event"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_entity_registry_enabled_default = False # Deprecated in v2.9
    _attr_has_entity_name = True

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_next_event"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.next_arrival
    
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        attrs = {}
        if data.next_start_time:
            attrs["planned_start"] = data.next_start_time.isoformat()
        if data.schedule_summary:
            attrs["learned_schedule"] = data.schedule_summary

        # v2.6 Patterns
        attrs["pattern_type"] = data.pattern_type
        attrs["pattern_confidence"] = round(data.pattern_confidence, 2)
        attrs["pattern_stability"] = round(data.pattern_stability, 2)
        attrs["fallback_used"] = data.fallback_used
        
        if data.detected_modes:
            attrs["detected_modes"] = data.detected_modes
            
        return attrs

class ThermalModelSensor(PreheatBaseSensor):
    """Combined Model Status."""
    _attr_translation_key = "model_status"
    _attr_entity_registry_enabled_default = False
    _attr_icon = "mdi:chart-line"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_model_status"
    
    @property
    def native_value(self) -> str:
        if self.coordinator.data.learning_active:
            return "learning"
        return "ready"

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        val = self.coordinator.data.valve_signal
        return {
            "valve_coverage": "ok" if val is not None else "none",
            "valve_position": f"{val}%" if val is not None else "n/a"
        }

class PhysicsSensor(PreheatBaseSensor):
    """Raw Physics Values."""
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_entity_registry_enabled_default = False
    _attr_icon = "mdi:function"

    def __init__(self, coordinator, entry, key, translation_key, unit):
        super().__init__(coordinator, entry)
        self._key = key
        self._attr_translation_key = translation_key
        self._attr_native_unit_of_measurement = unit

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_{self._key}"

    @property
    def native_value(self) -> float:
        return round(getattr(self.coordinator.data, self._key, 0.0), 2)

class PreheatConfidenceSensor(PreheatBaseSensor):
    """Confidence in the learned model."""
    _attr_translation_key = "confidence"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:shield-check"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_confidence"

    @property
    def native_value(self) -> int:
        return self.coordinator.physics.get_confidence()

class PreheatOptimalStopTimeSensor(PreheatBaseSensor):
    """Optimal Stop trigger time."""
    _attr_translation_key = "optimal_stop_time"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:clock-end"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_optimal_stop_time"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.optimal_stop_time

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "savings_total_min": round(data.savings_total, 1),
            "savings_remaining_min": round(data.savings_remaining, 1),
            "reason": data.stop_reason,
            "coast_tau_hours": round(data.coast_tau, 1),
            "tau_confidence": round(data.tau_confidence * 100, 1),
            "is_active": data.optimal_stop_active
        }

# --- v3.0 Spec Implementations ---

class PreheatNextStartSensor(PreheatBaseSensor):
    """Calculated start time for preheating."""
    _attr_has_entity_name = True
    _attr_translation_key = "next_start" # Will map to 'Next Preheat Start'
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:clock-start"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_next_start_time"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.next_start_time

class PreheatDurationSensor(PreheatBaseSensor):
    """Predicted duration in minutes."""
    _attr_has_entity_name = True
    _attr_translation_key = "predicted_duration"
    _attr_native_unit_of_measurement = "min"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:timer-sand"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_predicted_duration"

    @property
    def native_value(self) -> float:
        return round(self.coordinator.data.predicted_duration, 1)

class PreheatTargetTempSensor(PreheatBaseSensor):
    """Target temperature the system aimed for."""
    _attr_has_entity_name = True
    _attr_translation_key = "target_temperature"
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    
    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_target_temperature"

    @property
    def native_value(self) -> float:
        return self.coordinator.data.target_setpoint

class PreheatNextArrivalSensor(PreheatBaseSensor):
    """Next expected arrival time (Alias)."""
    _attr_has_entity_name = True
    _attr_translation_key = "next_arrival_time"
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    # No icon needed, device class provides it

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_next_arrival_time"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.next_arrival

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return learned arrival patterns."""
        attrs = {}
        data = self.coordinator.data
        if data.schedule_summary:
            attrs["learned_arrivals"] = data.schedule_summary
        return attrs

class PreheatNextSessionEndSensor(PreheatBaseSensor):
    """Next scheduled session end time (e.g. for Optimal Stop)."""
    _attr_has_entity_name = True
    _attr_translation_key = "next_session_end"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_next_session_end"

    @property
    def native_value(self) -> datetime | None:
        # We need to expose next_departure from data
        return self.coordinator.data.next_departure

        """Return learned departure patterns."""
        attrs = {}
        data = self.coordinator.data
        if data.departure_summary:
            attrs["learned_departures"] = data.departure_summary
        return attrs


# --- Fork Phase 4: learned schedule + explicit thermal model (DRIVE mode) ---


def _minutes_to_hhmm(minutes: int | None) -> str | None:
    """Minutes since midnight as "HH:MM" (values >1440 wrap past midnight)."""
    if minutes is None:
        return None
    minutes = minutes % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


class PreheatLearnedOccupancySensor(PreheatBaseSensor):
    """Predicted probability the house is occupied right now (0-100%).

    From the learned schedule (occupancy history + fixed schedule as
    prior). DRIVE mode (Ben, Nov 10): the companion binary sensor is
    the blueprint's occupancy input; this is the percentage view.
    """
    _attr_translation_key = "learned_occupancy"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:home-account"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_occupancy"

    @property
    def native_value(self) -> float:
        return round(self.coordinator.data.learned_occupancy_prob * 100, 1)

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        attrs: dict[str, Any] = {
            "confidence": round(data.learned_schedule_confidence * 100, 1),
            "days_of_data": data.learned_schedule_suggestion.get("days_of_data")
            if data.learned_schedule_suggestion
            else 0,
        }
        if data.learned_next_arrival:
            attrs["next_predicted_arrival"] = data.learned_next_arrival.isoformat()
        if data.learned_next_departure:
            attrs["next_predicted_departure"] = data.learned_next_departure.isoformat()
        if data.learned_schedule_suggestion:
            suggestion = data.learned_schedule_suggestion
            attrs["suggested_wake"] = _minutes_to_hhmm(
                suggestion["learned"].get("wake_min")
            )
            attrs["suggested_bedtime"] = _minutes_to_hhmm(
                suggestion["learned"].get("bedtime_min")
            )
            attrs["suggested_leave"] = _minutes_to_hhmm(
                suggestion["learned"].get("leave_min")
            )
            attrs["suggested_return"] = _minutes_to_hhmm(
                suggestion["learned"].get("return_min")
            )
            attrs["suggestion_delta_minutes"] = suggestion.get("delta_minutes")
        return attrs


class PreheatLearnedWakeSensor(PreheatBaseSensor):
    """Next predicted wake time from learned sleep/wake patterns."""
    _attr_translation_key = "learned_wake_time"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:alarm"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_wake_time"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.learned_wake_time

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        suggestion = self.coordinator.data.learned_schedule_suggestion or {}
        learned = suggestion.get("learned", {})
        fixed = suggestion.get("fixed", {})
        return {
            "median_wake": _minutes_to_hhmm(learned.get("wake_min")),
            "fixed_wake": _minutes_to_hhmm(fixed.get("wake_min")),
            "wake_observations": suggestion.get("wake_observations", 0),
            "confidence": round(
                self.coordinator.data.learned_schedule_confidence * 100, 1
            ),
        }


class PreheatLearnedBedtimeSensor(PreheatBaseSensor):
    """Next predicted bedtime from learned sleep/wake patterns."""
    _attr_translation_key = "learned_bedtime"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:bed"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_bedtime"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.learned_bedtime_time

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        suggestion = self.coordinator.data.learned_schedule_suggestion or {}
        learned = suggestion.get("learned", {})
        fixed = suggestion.get("fixed", {})
        return {
            "median_bedtime": _minutes_to_hhmm(learned.get("bedtime_min")),
            "fixed_bedtime": _minutes_to_hhmm(fixed.get("bedtime_min")),
            "bedtime_observations": suggestion.get("bedtime_observations", 0),
            "confidence": round(
                self.coordinator.data.learned_schedule_confidence * 100, 1
            ),
        }


class PreheatLearnedHeatingRateSensor(PreheatBaseSensor):
    """Learned heating rate in °F/hr at the current outdoor temperature.

    Fitted from warm-up episodes (internal math in °C/K, converted at
    exposure). Unknown until enough episodes exist.
    """
    _attr_translation_key = "learned_heating_rate"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "°F/h"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:thermometer-chevron-up"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_heating_rate"

    @property
    def native_value(self) -> float | None:
        value = self.coordinator.data.heating_rate_f_per_h
        return round(value, 2) if value is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        learner = self.coordinator.thermal_learner
        return {
            "rate_at_32f": (
                round(rate, 2)
                if (rate := learner.reference_heating_rate_f_per_h()) is not None
                else None
            ),
            "slope_f_per_f": (
                round(slope, 3)
                if (slope := learner.rate_slope_f_per_f()) is not None
                else None
            ),
            "episodes": learner.episode_count,
            "confident": learner.is_confident,
            "duration_source": self.coordinator.data.thermal_duration_source,
        }


class PreheatLearnedCoolingRateSensor(PreheatBaseSensor):
    """Learned cooling rate in °F/hr at a 10°F indoor-outdoor delta.

    Derived from the cooling time constant tau.
    """
    _attr_translation_key = "learned_cooling_rate"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "°F/h"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:thermometer-chevron-down"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_cooling_rate"

    @property
    def native_value(self) -> float | None:
        value = self.coordinator.data.cooling_rate_f_per_h
        return round(value, 2) if value is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        return {
            "cooling_tau_hours": round(self.coordinator.data.coast_tau, 2),
            "tau_confidence": round(self.coordinator.data.tau_confidence * 100, 1),
        }


class PreheatLearnedDeadtimeSensor(PreheatBaseSensor):
    """Learned dead time (lag before heat reaches the room), in minutes."""
    _attr_translation_key = "learned_deadtime"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "min"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:timer-outline"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_deadtime"

    @property
    def native_value(self) -> float:
        return round(self.coordinator.data.deadtime, 1)


class PreheatLearnedHeatLossSensor(PreheatBaseSensor):
    """Heat-loss dependence: extra preheat minutes per °F of outdoor delta."""
    _attr_translation_key = "learned_heat_loss"
    _attr_entity_registry_enabled_default = False
    _attr_native_unit_of_measurement = "min/°F"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:home-thermometer-outline"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_learned_heat_loss"

    @property
    def native_value(self) -> float:
        return round(self.coordinator.data.heat_loss_min_per_f, 2)


class PreheatAwaySetpointSensor(PreheatBaseSensor):
    """Deep-setback target while away mode is on (native °C, shown °F).

    Comfort minus the away offset (default 10°F), floored at 61°F.
    Unknown while away mode is off. The blueprint reads this to hold
    the away setpoint instead of the routine.
    """
    _attr_translation_key = "away_setpoint"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TEMPERATURE
    _attr_native_unit_of_measurement = UnitOfTemperature.CELSIUS
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:thermometer-low"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_away_setpoint"

    @property
    def native_value(self) -> float | None:
        value = self.coordinator.data.away_setpoint
        return round(value, 2) if value is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        return {
            "away_mode": data.away_mode,
            "away_setback_active": data.away_setback_active,
        }


class PreheatAwayReturnStartSensor(PreheatBaseSensor):
    """When the away return warm-up starts (timestamp).

    Return time minus the predicted warm-up duration (learned thermal
    model when confident, physics fallback) and the buffer. Unknown
    while away mode is off or no return time is set. The blueprint's
    away return pre-heat binary sensor fires from this.
    """
    _attr_translation_key = "away_return_start"
    _attr_entity_registry_enabled_default = False
    _attr_device_class = SensorDeviceClass.TIMESTAMP
    _attr_icon = "mdi:timer-sand"

    @property
    def unique_id(self) -> str:
        return f"{self._entry.entry_id}_away_return_start"

    @property
    def native_value(self) -> datetime | None:
        return self.coordinator.data.away_return_preheat_start

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        data = self.coordinator.data
        attrs: dict[str, Any] = {"away_mode": data.away_mode}
        if data.away_return_dt is not None:
            attrs["away_return"] = data.away_return_dt.isoformat()
        return attrs


class PreheatHouseBaseSensor(Entity):
    """Base sensor for Preheat House global sensors."""
    _attr_has_entity_name = True

    def __init__(self, house: HouseArrivalCollector) -> None:
        """Initialize the sensor."""
        self.house = house
        self._attr_device_info = {
            "identifiers": {(DOMAIN, "house")},
            "name": "Preheat House",
            "manufacturer": "Ecronika",
            "model": "House Arrival Hub",
            "sw_version": VERSION,
        }

    @property
    def should_poll(self) -> bool:
        """No polling needed."""
        return False

    async def async_added_to_hass(self) -> None:
        """Register callbacks."""
        self.async_on_remove(self.house.async_add_listener(self.async_write_ha_state))


class PreheatHouseNextArrivalSensor(PreheatHouseBaseSensor, SensorEntity):
    """House global next arrival sensor."""
    _attr_translation_key = "house_next_arrival"
    _attr_device_class = SensorDeviceClass.TIMESTAMP

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return "house_next_arrival"

    @property
    def native_value(self) -> datetime | None:
        """Return the next predicted house arrival."""
        val, _, _ = self.house.get_next_arrival(dt_util.now())
        return val


class PreheatHouseArrivalConfidenceSensor(PreheatHouseBaseSensor, SensorEntity):
    """House global arrival confidence sensor."""
    _attr_translation_key = "house_arrival_confidence"
    _attr_native_unit_of_measurement = "%"
    _attr_state_class = SensorStateClass.MEASUREMENT
    _attr_icon = "mdi:shield-check"

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return "house_arrival_confidence"

    @property
    def native_value(self) -> int:
        """Return prediction confidence."""
        _, conf, _ = self.house.get_next_arrival(dt_util.now())
        return int(round(conf * 100))


class PreheatHouseArrivalWindowSensor(PreheatHouseBaseSensor, SensorEntity):
    """House global arrival window sensor."""
    _attr_translation_key = "house_arrival_window"
    _attr_icon = "mdi:clock-time-four-outline"

    @property
    def unique_id(self) -> str:
        """Return unique ID."""
        return "house_arrival_window"

    @property
    def native_value(self) -> str | None:
        """Return expected arrival window."""
        now = dt_util.now()
        now_local = dt_util.as_local(now)
        today_date = now_local.date()
        
        for day_offset in range(8):
            check_date = today_date + timedelta(days=day_offset)
            is_weekend = (check_date.weekday() >= 5)
            am_list, pm_list = self.house.get_pooled_arrivals(is_weekend)
            
            if len(am_list) >= 2:
                p25 = get_percentile(am_list, 0.25)
                p75 = get_percentile(am_list, 0.75)
                spread = p75 - p25
                conf = max(0.0, 1.0 - (spread / 300.0))
                if conf >= 0.7:
                    event_dt = datetime.combine(check_date, datetime.min.time(), tzinfo=now_local.tzinfo) + timedelta(minutes=p25)
                    if event_dt > now_local:
                        h_start = p25 // 60
                        m_start = p25 % 60
                        h_end = p75 // 60
                        m_end = p75 % 60
                        return f"{h_start:02d}:{m_start:02d}-{h_end:02d}:{m_end:02d}"
            
            if len(pm_list) >= 2:
                p25 = get_percentile(pm_list, 0.25)
                p75 = get_percentile(pm_list, 0.75)
                spread = p75 - p25
                conf = max(0.0, 1.0 - (spread / 300.0))
                
                q = 0.25
                if self.house.comfort_bias == "comfort":
                    q = 0.15
                elif self.house.comfort_bias == "economy":
                    q = 0.50
                    
                target_time = get_percentile(pm_list, q)
                
                if conf >= 0.7:
                    event_dt = datetime.combine(check_date, datetime.min.time(), tzinfo=now_local.tzinfo) + timedelta(minutes=target_time)
                    if event_dt > now_local:
                        h_start = target_time // 60
                        m_start = target_time % 60
                        h_end = p75 // 60
                        m_end = p75 % 60
                        return f"{h_start:02d}:{m_start:02d}-{h_end:02d}:{m_end:02d}"
                else:
                    if self.house.evening_window_str:
                        return self.house.evening_window_str
        return None