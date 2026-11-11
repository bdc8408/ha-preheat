"""DateTime platform for Preheat (fork Phase 4: away-mode return time)."""
from __future__ import annotations

from datetime import datetime
from functools import cached_property
from typing import Any

from homeassistant.components.datetime import DateTimeEntity
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import HomeAssistant
from homeassistant.helpers.entity import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.update_coordinator import CoordinatorEntity
from homeassistant.util import dt as dt_util

from .const import DOMAIN, VERSION
from .coordinator import PreheatingCoordinator


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up the away-return datetime entity."""
    if entry.unique_id == "preheat_system":
        return
    coordinator: PreheatingCoordinator = entry.runtime_data
    async_add_entities([PreheatAwayReturnDateTime(coordinator, entry)])


class PreheatAwayReturnDateTime(
    CoordinatorEntity["PreheatingCoordinator"], DateTimeEntity
):
    """Return date/time for away mode.

    Ben sets this in the HA UI when leaving for a trip (optional).
    The fork computes a return pre-heat start so the house reaches
    Comfort by this time. Clearing it (or leaving it unset) means the
    house stays in the deep setback until away mode is switched off.
    A past value is ignored by the coordinator. Cleared automatically
    when away mode is switched off.
    """

    _attr_icon = "mdi:calendar-clock"
    _attr_has_entity_name = True
    _attr_translation_key = "away_return"

    def __init__(self, coordinator: PreheatingCoordinator, entry: ConfigEntry) -> None:
        """Initialize the datetime entity."""
        super().__init__(coordinator)
        self.entry_id = entry.entry_id
        self._attr_unique_id = f"{entry.entry_id}_away_return"

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
    def native_value(self) -> datetime | None:
        """Return the current away return time (local)."""
        value = self.coordinator.away_return_dt
        return dt_util.as_local(value) if value is not None else None

    @property
    def extra_state_attributes(self) -> dict[str, Any]:
        """Return away-mode state."""
        data = self.coordinator.data
        attrs: dict[str, Any] = {"away_mode": data.away_mode}
        if data.away_return_preheat_start is not None:
            attrs["return_preheat_start"] = data.away_return_preheat_start.isoformat()
        return attrs

    async def async_set_value(self, value: datetime) -> None:
        """Set the away return time from the HA UI."""
        await self.coordinator.set_away_return(value)
