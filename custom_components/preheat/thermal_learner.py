"""Explicit house thermal model (fork Phase 4).

Upstream learns the preheat duration implicitly through
``ThermalPhysics`` (mass/loss factors fitted from preheat runs). This
module makes the house's thermal behaviour explicit and first-class:

- heating rate (°F/hr) as a function of outdoor temperature, fitted
  directly from warm-up episodes (rate = rise / effective heating time);
- cooling rate (°F/hr at a reference indoor-outdoor delta), derived from
  the ``CoolingAnalyzer`` time constant tau;
- dead time / lag (minutes), synced from the physics model;
- heat-loss dependence on outdoor temperature, as the fitted slope of
  heating rate vs outdoor temperature and as min-per-°F from the physics
  loss factor.

All internal math stays in °C/K (the Phase 1 unit rule). Fahrenheit
appears only at the exposure boundary: rates in °F/hr are K/hr × 9/5, and
the heat-loss figure in min/°F is min/K × 5/9. Episodes arrive in °C —
the coordinator already converts at ingestion — so a Fahrenheit
thermostat can no longer skew the learned rate ~1.8×.

The coordinator feeds one episode per completed preheat run (the same
data ``ThermalPhysics.update_model`` learns from) and uses
: meth:`ThermalLearner.predict_duration_min` for the preheat start time
once the learner is confident (>= ``MIN_EPISODES_CONFIDENT`` episodes),
falling back to the physics model otherwise.
"""

from __future__ import annotations

import logging

from .units import celsius_delta_to_fahrenheit

_LOGGER = logging.getLogger(__name__)

# Episodes needed before the fitted rate is trusted for start times.
MIN_EPISODES_CONFIDENT = 5
# Episodes needed before a rate is reported at all.
MIN_EPISODES_FOR_RATE = 3

# Plausible house heating rates (K/hr). Outside this band an episode is
# noise (bad sensor, window open, furnace off) and is rejected.
MIN_RATE_K_PER_H = 0.2
MAX_RATE_K_PER_H = 15.0

# Minimum achieved rise for an episode to be usable (same guard as
# ThermalPhysics.update_model).
MIN_DELTA_K = 0.3

# Reference points for the exposed figures.
REFERENCE_OUTDOOR_C = 0.0  # 32°F
REFERENCE_COOLING_DELTA_F = 10.0  # °F indoor-outdoor delta

# Sanity bounds for the synced cooling time constant (hours).
MIN_TAU_HOURS = 0.5
MAX_TAU_HOURS = 48.0


def _coerce_float(value, default: float | None = None) -> float | None:
    """Best-effort float conversion; None when the value is not numeric.

    The coordinator's test harness replaces the physics model with a
    MagicMock, whose attributes are mocks, not numbers — the learner must
    reject those episodes instead of raising.
    """
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


class ThermalLearner:
    """Learn the house's heating/cooling behaviour from warm-up episodes.

    The heating-rate fit is an incremental ordinary least squares of rate
    (K/hr) against outdoor temperature (°C), kept as sufficient statistics
    so it can be persisted and updated one episode at a time.
    """

    def __init__(self) -> None:
        # OLS sufficient statistics: rate = a + b * outdoor_c.
        self._n = 0
        self._sum_x = 0.0  # outdoor °C
        self._sum_y = 0.0  # rate K/hr
        self._sum_xx = 0.0
        self._sum_xy = 0.0
        # Recent episodes for diagnostics: (outdoor_c, rate_k_per_h).
        self._recent_episodes: list[tuple[float, float]] = []
        # Synced from the physics model / cooling analyzer (coordinator).
        self.deadtime_min = 0.0
        self.loss_factor_min_per_k = 0.0
        self.cooling_tau_hours: float | None = None
        self.cooling_confidence = 0.0

    # -- learning ---------------------------------------------------------

    def record_episode(
        self,
        duration_min: float,
        delta_t_in_c: float,
        outdoor_c: float,
        deadtime_min: float = 0.0,
    ) -> bool:
        """Learn from one warm-up episode. Returns True if it was used.

        ``duration_min`` is the full heat-up time (start until the target
        was reached); the dead time is subtracted because the house does
        not warm during it. All temperatures in °C/K.
        """
        duration_min = _coerce_float(duration_min)
        delta_t_in_c = _coerce_float(delta_t_in_c)
        outdoor_c = _coerce_float(outdoor_c)
        deadtime_min = _coerce_float(deadtime_min, 0.0)
        if duration_min is None or delta_t_in_c is None or outdoor_c is None:
            return False
        if delta_t_in_c < MIN_DELTA_K:
            _LOGGER.debug(
                "ThermalLearner: skipping episode, rise %.2f K too small", delta_t_in_c
            )
            return False
        effective_min = duration_min - deadtime_min
        if effective_min <= 5.0:
            _LOGGER.debug(
                "ThermalLearner: skipping episode, effective heat-up %.1f min too short",
                effective_min,
            )
            return False
        # The episode's deadtime is the physics deadtime in effect for the
        # run; retain the latest so duration prediction works even before
        # the coordinator's next sync.
        self.deadtime_min = deadtime_min
        rate = delta_t_in_c / (effective_min / 60.0)  # K/hr
        if not (MIN_RATE_K_PER_H <= rate <= MAX_RATE_K_PER_H):
            _LOGGER.debug(
                "ThermalLearner: rejecting implausible rate %.2f K/hr", rate
            )
            return False
        self._n += 1
        self._sum_x += outdoor_c
        self._sum_y += rate
        self._sum_xx += outdoor_c * outdoor_c
        self._sum_xy += outdoor_c * rate
        self._recent_episodes.append((outdoor_c, rate))
        if len(self._recent_episodes) > 50:
            self._recent_episodes = self._recent_episodes[-50:]
        return True

    def sync(
        self,
        deadtime_min: float,
        loss_factor_min_per_k: float,
        cooling_tau_hours: float | None,
        cooling_confidence: float,
    ) -> None:
        """Pull the parameters this module does not fit itself."""
        deadtime = _coerce_float(deadtime_min, 0.0)
        loss = _coerce_float(loss_factor_min_per_k, 0.0)
        tau = _coerce_float(cooling_tau_hours)
        conf = _coerce_float(cooling_confidence, 0.0)
        self.deadtime_min = max(0.0, deadtime if deadtime is not None else 0.0)
        self.loss_factor_min_per_k = max(0.0, loss if loss is not None else 0.0)
        if tau is not None and MIN_TAU_HOURS <= tau <= MAX_TAU_HOURS:
            self.cooling_tau_hours = tau
        else:
            self.cooling_tau_hours = None
        self.cooling_confidence = max(0.0, min(1.0, conf if conf is not None else 0.0))

    # -- fitted heating rate ----------------------------------------------

    @property
    def episode_count(self) -> int:
        return self._n

    @property
    def is_confident(self) -> bool:
        """True once enough episodes exist to trust the fitted rate."""
        return self._n >= MIN_EPISODES_CONFIDENT

    def _fit(self) -> tuple[float, float] | None:
        """(intercept, slope) of rate [K/hr] vs outdoor [°C], or None."""
        if self._n < MIN_EPISODES_FOR_RATE:
            return None
        denom = self._n * self._sum_xx - self._sum_x * self._sum_x
        if abs(denom) < 1e-9:
            # All episodes at (nearly) one outdoor temperature: no slope
            # information, use the mean rate.
            return self._sum_y / self._n, 0.0
        slope = (self._n * self._sum_xy - self._sum_x * self._sum_y) / denom
        intercept = (self._sum_y - slope * self._sum_x) / self._n
        return intercept, slope

    def heating_rate_c_per_h(self, outdoor_c: float) -> float | None:
        """Predicted heating rate in K/hr at an outdoor temperature."""
        fit = self._fit()
        if fit is None:
            return None
        intercept, slope = fit
        rate = intercept + slope * outdoor_c
        return max(MIN_RATE_K_PER_H, min(MAX_RATE_K_PER_H, rate))

    def heating_rate_f_per_h(self, outdoor_c: float) -> float | None:
        """Predicted heating rate in °F/hr at an outdoor temperature."""
        rate_c = self.heating_rate_c_per_h(outdoor_c)
        if rate_c is None:
            return None
        return celsius_delta_to_fahrenheit(rate_c)

    def reference_heating_rate_f_per_h(self) -> float | None:
        """Heating rate in °F/hr at the 32°F (0°C) reference outdoor temp."""
        return self.heating_rate_f_per_h(REFERENCE_OUTDOOR_C)

    def rate_slope_f_per_f(self) -> float | None:
        """How much the heating rate (°F/hr) changes per °F of outdoor temp.

        A °F/hr-per-°F slope is numerically identical to the K/hr-per-K
        slope (both numerator and denominator scale by 9/5).
        """
        fit = self._fit()
        if fit is None:
            return None
        return fit[1]

    # -- cooling / heat loss ------------------------------------------------

    def cooling_rate_f_per_h(
        self, delta_t_f: float = REFERENCE_COOLING_DELTA_F
    ) -> float | None:
        """Cooling rate in °F/hr at an indoor-outdoor delta.

        From the cooling time constant: dT/dt = delta / tau. Returns None
        until the cooling analyzer has produced a plausible tau.
        """
        if self.cooling_tau_hours is None or self.cooling_tau_hours <= 0:
            return None
        return delta_t_f / self.cooling_tau_hours

    def heat_loss_min_per_f(self) -> float:
        """Extra preheat minutes per °F of outdoor delta (from loss factor).

        The physics loss factor is min per K of outdoor delta; 1°F of
        delta is 5/9 K, so min/°F = min/K × 5/9.
        """
        return self.loss_factor_min_per_k * 5.0 / 9.0

    # -- preheat start times -------------------------------------------------

    def predict_duration_min(self, delta_t_in_c: float, outdoor_c: float) -> float | None:
        """Minutes to raise the house by ``delta_t_in_c`` at ``outdoor_c``.

        deadtime + rise / rate. None until the learner is confident.
        """
        if delta_t_in_c <= 0:
            return 0.0
        if not self.is_confident:
            return None
        rate = self.heating_rate_c_per_h(outdoor_c)
        if rate is None or rate <= 0:
            return None
        return self.deadtime_min + (delta_t_in_c / rate) * 60.0

    # -- persistence ----------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise learned state for the coordinator store."""
        return {
            "n": self._n,
            "sum_x": self._sum_x,
            "sum_y": self._sum_y,
            "sum_xx": self._sum_xx,
            "sum_xy": self._sum_xy,
            "recent_episodes": [[x, y] for x, y in self._recent_episodes[-50:]],
            "deadtime_min": self.deadtime_min,
            "loss_factor_min_per_k": self.loss_factor_min_per_k,
            "cooling_tau_hours": self.cooling_tau_hours,
            "cooling_confidence": self.cooling_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict | None) -> "ThermalLearner":
        """Restore learned state; tolerates missing/partial data."""
        learner = cls()
        if not data:
            return learner
        try:
            learner._n = int(data.get("n", 0))
            learner._sum_x = float(data.get("sum_x", 0.0))
            learner._sum_y = float(data.get("sum_y", 0.0))
            learner._sum_xx = float(data.get("sum_xx", 0.0))
            learner._sum_xy = float(data.get("sum_xy", 0.0))
            episodes = data.get("recent_episodes")
            if isinstance(episodes, list):
                learner._recent_episodes = [
                    (float(x), float(y)) for x, y in episodes[-50:]
                ]
            learner.deadtime_min = max(0.0, float(data.get("deadtime_min", 0.0)))
            learner.loss_factor_min_per_k = max(
                0.0, float(data.get("loss_factor_min_per_k", 0.0))
            )
            tau = data.get("cooling_tau_hours")
            learner.cooling_tau_hours = (
                float(tau)
                if tau is not None and MIN_TAU_HOURS <= float(tau) <= MAX_TAU_HOURS
                else None
            )
            learner.cooling_confidence = max(
                0.0, min(1.0, float(data.get("cooling_confidence", 0.0)))
            )
        except (TypeError, ValueError, AttributeError) as err:
            _LOGGER.warning("ThermalLearner: ignoring corrupt stored state: %s", err)
        return learner
