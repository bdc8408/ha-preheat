"""Switch platform for Preheat integration."""
from __future__ import annotations

from functools import cached_property
from typing import Any, TYPE_CHECKING

from homeassistant.components.switch import SwitchEntity
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.update_coordinator import CoordinatorEntity

from .const import DOMAIN, VERSION, ATTR_DECISION_TRACE, CONF_PREHEAT_HOLD

if TYPE_CHECKING:
    from homeassistant.core import HomeAssistant
    from homeassistant.helpers.entity_platform import AddEntitiesCallback
    from . import PreheatConfigEntry
    from .coordinator import PreheatingCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: PreheatConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up Preheat switch."""
    if entry.unique_id == "preheat_system":
        return
    coordinator = entry.runtime_data
    async_add_entities([
        PreheatingSwitch(coordinator, entry),
        PreheatHoldSwitch(coordinator, entry),
        PreheatEnabledSwitch(coordinator, entry),
        # Fork Phase 4: away mode (Ben, Nov 10).
        PreheatAwayModeSwitch(coordinator, entry),
    ])


class PreheatingSwitch(CoordinatorEntity["PreheatingCoordinator"], SwitchEntity):
    """Switch to control preheat mode."""

    _attr_icon = "mdi:radiator"
    _attr_has_entity_name = True
    _attr_translation_key = "preheat"
    _attr_entity_registry_enabled_default = False

    def __init__(self, coordinator: PreheatingCoordinator, entry: PreheatConfigEntry) -> None:
        """Initialize the switch."""
        super().__init__(coordinator)
        self.entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_preheat"

    @cached_property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.coordinator.device_name,
            manufacturer="Ecronika",
            model="Intelligent Preheating",
            sw_version=VERSION,
        )

    @property
    def is_on(self) -> bool:
        """Return true if preheat is on."""
        return self.coordinator.data.preheat_active
        
    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return decision trace."""
        attrs = {}
        if self.coordinator.data.decision_trace:
             attrs[ATTR_DECISION_TRACE] = self.coordinator.data.decision_trace
        return attrs

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on preheat."""
        await self.coordinator.force_preheat_on()

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off preheat manually."""
        await self.coordinator.stop_preheat_manual()

class PreheatHoldSwitch(CoordinatorEntity["PreheatingCoordinator"], SwitchEntity):
    """Switch to manually HOLD (Block) preheat logic."""

    _attr_icon = "mdi:hand-back-left"
    _attr_has_entity_name = True
    _attr_translation_key = "preheat_hold"

    def __init__(self, coordinator: PreheatingCoordinator, entry: PreheatConfigEntry) -> None:
        """Initialize the switch."""
        super().__init__(coordinator)
        self.entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_{CONF_PREHEAT_HOLD}"

    @cached_property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.coordinator.device_name,
            manufacturer="Ecronika",
            model="Intelligent Preheating",
            sw_version=VERSION,
        )

    @property
    def is_on(self) -> bool:
        """Return true if hold is active."""
        return self.coordinator.hold_active

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on hold."""
        await self.coordinator.set_hold(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off hold."""
        await self.coordinator.set_hold(False)

class PreheatEnabledSwitch(CoordinatorEntity["PreheatingCoordinator"], SwitchEntity):
    """Master switch to enable/disable the preheat logic."""

    _attr_icon = "mdi:power"
    _attr_has_entity_name = True
    _attr_translation_key = "enabled"

    def __init__(self, coordinator: PreheatingCoordinator, entry: PreheatConfigEntry) -> None:
        """Initialize the switch."""
        super().__init__(coordinator)
        self.entry_id = entry.entry_id
        # Use existing 'enabled' convention if possible? No, we use custom unique_id
        self._attr_unique_id = f"{entry.entry_id}_enabled"

    @cached_property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.coordinator.device_name,
            manufacturer="Ecronika",
            model="Intelligent Preheating",
            sw_version=VERSION,
        )

    @property
    def is_on(self) -> bool:
        """Return true if enabled."""
        return self.coordinator.enable_active

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Enable."""
        await self.coordinator.set_enabled(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Disable."""
        await self.coordinator.set_enabled(False)


class PreheatAwayModeSwitch(CoordinatorEntity["PreheatingCoordinator"], SwitchEntity):
    """Away-mode switch: hold a deep low-energy setback while on a trip.

    While on, the learned schedule is overridden (away wins) and the
    blueprint holds the away setpoint instead of the routine. If a
    return date/time is set, the house warms back to Comfort by the
    return time. Turning off resumes the normal learned drive and
    clears the return time. Persisted across restarts (unlike the
    manual hold): a restart must not silently cancel a trip and heat
    an empty house.
    """

    _attr_icon = "mdi:home-export-outline"
    _attr_has_entity_name = True
    _attr_translation_key = "away_mode"

    def __init__(self, coordinator: PreheatingCoordinator, entry: PreheatConfigEntry) -> None:
        """Initialize the switch."""
        super().__init__(coordinator)
        self.entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_away_mode"

    @cached_property
    def device_info(self) -> DeviceInfo:
        """Return device information."""
        return DeviceInfo(
            identifiers={(DOMAIN, self.entry_id)},
            name=self.coordinator.device_name,
            manufacturer="Ecronika",
            model="Intelligent Preheating",
            sw_version=VERSION,
        )

    @property
    def is_on(self) -> bool:
        """Return true if away mode is active."""
        return self.coordinator.away_mode

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return away-mode state."""
        data = self.coordinator.data
        attrs: dict[str, Any] = {
            "away_setback_active": data.away_setback_active,
            "away_preheat": data.away_preheat,
        }
        if data.away_setpoint is not None:
            attrs["away_setpoint_f"] = round(data.away_setpoint * 9.0 / 5.0 + 32.0, 1)
        if data.away_return_dt is not None:
            attrs["away_return"] = data.away_return_dt.isoformat()
        if data.away_return_preheat_start is not None:
            attrs["return_preheat_start"] = data.away_return_preheat_start.isoformat()
        return attrs

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on away mode."""
        await self.coordinator.set_away_mode(True)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off away mode."""
        await self.coordinator.set_away_mode(False)