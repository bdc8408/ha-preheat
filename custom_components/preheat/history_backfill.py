"""Reconstruct warm-up sessions from recorder history (fork v2.14.3).

Pure functions (no Home Assistant imports) so the session logic is
unit-testable without a coordinator. The state machine mirrors the live
passive observer (``PreheatingCoordinator._observe_passive_warmup``)
except where noted below:

- a session opens when the burner is heating and the room is below the
  target minus the reached tolerance;
- it closes when the setpoint moves DOWN by more than the setpoint-change
  threshold, when the target is reached, or after the non-heating grace
  period with the burner off;
- a session still open at the end of the samples is discarded — it
  belongs to the live observer, which may still be watching it.

Thresholds are parameters (single source of truth stays in the
coordinator / const), so the backfill and the live observer can never
drift apart silently: the coordinator passes its own constants.

Deliberately NOT mirrored from the live path:

- No window-open check. The backfill has no window sensor history to
  consult (and none is configured). The rate band in
  ``ThermalLearner.record_episode`` (0.2–15 K/hr) is the guard that
  rejects a window-open warm-up: with a window open the house heats
  too slowly to qualify.
- No deadtime re-analysis. The live close analyses the in-memory
  history buffer, which only holds post-startup samples; analysing it
  for a historical session would be meaningless. Backfilled sessions
  use the physics model's current deadtime.
- No cooling-analyzer feeding. This module is warm-up episodes only.

Fork Phase 8: a setpoint step UP while the room is still heating
toward the new target RETARGETS the open session instead of closing
it — a 1°F step is 0.56 K, above the change threshold, so the old
rule shredded every routine Sleep -> Wake -> Home warm-up into
fragments that each failed the learner's guards. Only a step DOWN
(or the target actually being reached) ends a session. The live
observer still closes on any setpoint move (it cannot tell a step-up
from a mode change in real time); the backfill, which sees the whole
window, can. The backfill also uses a longer non-heating grace
(15 min, passed by the coordinator) because the furnace cycles off
mid-warm-up.

All temperatures are °C/K (the Phase 1 unit rule); unit conversion
happens in the coordinator at ingestion.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ClimateSample:
    """One climate-entity state, converted to °C at ingestion."""

    ts: float  # unix timestamp (UTC)
    temp_c: float  # current_temperature, °C
    target_c: float  # setpoint (``temperature`` attribute), °C
    heating: bool  # hvac_action == "heating"
    outdoor_c: float | None = None  # outdoor temp at this sample, °C


@dataclass(frozen=True)
class WarmupSession:
    """One completed warm-up, ready to feed to the thermal model."""

    start_ts: float
    end_ts: float
    start_temp_c: float
    end_temp_c: float
    target_c: float
    outdoor_avg_c: float  # mean outdoor over the session (°C)
    outdoor_known: bool  # False when no real outdoor sample covered it


def attach_outdoor(
    samples: list[ClimateSample | None],
    outdoor: list[tuple[float, float]],
) -> list[ClimateSample | None]:
    """Fill each climate sample's outdoor temp from sparse outdoor samples.

    Carry-forward: a sample takes the most recent outdoor sample at or
    before its timestamp (the live system caches the outdoor reading
    the same way). Samples before the first outdoor reading keep
    ``outdoor_c=None``. ``None`` entries (breaks) pass through.
    """
    filled: list[ClimateSample | None] = []
    outdoor_sorted = sorted(outdoor)
    idx = 0
    current: float | None = None
    for sample in samples:
        if sample is None:
            filled.append(None)
            continue
        while idx < len(outdoor_sorted) and outdoor_sorted[idx][0] <= sample.ts:
            current = outdoor_sorted[idx][1]
            idx += 1
        sample.outdoor_c = current
        filled.append(sample)
    return filled


def reconstruct_sessions(
    samples: list[ClimateSample | None],
    *,
    non_heating_grace_min: float,
    setpoint_change_k: float,
    target_tolerance_k: float,
) -> list[WarmupSession]:
    """Reconstruct completed warm-up sessions from time-ordered samples.

    ``samples`` must be time-ordered; a ``None`` entry marks a break
    (unavailable / unparseable state) and closes any open session at
    the last good sample — a gap must not merge two warm-ups into one.
    Sessions still open at the end are discarded.
    """
    sessions: list[WarmupSession] = []
    open_session: dict | None = None
    last_ts: float | None = None
    last_temp: float | None = None

    def finish(session: dict, end_ts: float, end_temp: float) -> WarmupSession:
        n = session["outdoor_n"]
        return WarmupSession(
            start_ts=session["start_ts"],
            end_ts=end_ts,
            start_temp_c=session["start_temp_c"],
            end_temp_c=end_temp,
            target_c=session["target_c"],
            outdoor_avg_c=(session["outdoor_sum_c"] / n) if n > 0 else 10.0,
            outdoor_known=n > 0,
        )

    def open_new(sample: ClimateSample) -> dict | None:
        if sample.heating and sample.temp_c < sample.target_c - target_tolerance_k:
            # Fork Phase 8: the opening sample's outdoor reading
            # counts toward the session. Without this, a session of
            # just an opener + closer accumulated nothing (only
            # continuation samples were counted) and was wrongly
            # skipped as having no outdoor coverage.
            return {
                "start_ts": sample.ts,
                "start_temp_c": sample.temp_c,
                "target_c": sample.target_c,
                "outdoor_sum_c": sample.outdoor_c if sample.outdoor_c is not None else 0.0,
                "outdoor_n": 1 if sample.outdoor_c is not None else 0,
                "non_heating_since": None,
            }
        return None

    for sample in samples:
        if sample is None:
            # Break: close any open session at the last good sample.
            if open_session is not None and last_ts is not None and last_temp is not None:
                sessions.append(finish(open_session, last_ts, last_temp))
                open_session = None
            continue
        if open_session is None:
            open_session = open_new(sample)
        else:
            # A setpoint move ends the session against the temperature
            # achieved so far; this sample may open a fresh one.
            # Fork Phase 8: a step UP while still heating toward the
            # new target retargets the session instead of closing it
            # (see module docstring). Only a step DOWN closes it.
            delta_target = sample.target_c - open_session["target_c"]
            closed = False
            if (
                delta_target > setpoint_change_k
                and sample.temp_c < sample.target_c - target_tolerance_k
            ):
                open_session["target_c"] = sample.target_c
            elif abs(delta_target) > setpoint_change_k:
                sessions.append(finish(open_session, sample.ts, sample.temp_c))
                open_session = open_new(sample)
                closed = True
            elif sample.target_c <= sample.temp_c + target_tolerance_k:
                sessions.append(finish(open_session, sample.ts, sample.temp_c))
                open_session = None
                closed = True
            if not closed and open_session is not None:
                if sample.outdoor_c is not None:
                    open_session["outdoor_sum_c"] += sample.outdoor_c
                    open_session["outdoor_n"] += 1
                if sample.heating:
                    open_session["non_heating_since"] = None
                else:
                    if open_session["non_heating_since"] is None:
                        open_session["non_heating_since"] = sample.ts
                    elif (
                        sample.ts - open_session["non_heating_since"]
                    ) / 60.0 >= non_heating_grace_min:
                        sessions.append(finish(open_session, sample.ts, sample.temp_c))
                        open_session = None
        last_ts = sample.ts
        last_temp = sample.temp_c

    # A session still open at the end of the window is discarded: the
    # warm-up may still be running, and the live observer owns it.
    return sessions
