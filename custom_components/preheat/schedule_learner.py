"""Schedule learning (fork Phase 4) — DRIVE mode (Ben, Nov 10).

Learns the household's actual routine from occupancy history and exposes
it as a learned schedule that DRIVES the heating automation: the fork's
learned-occupancy binary sensor replaces the fixed schedule's home
window, and learned-sleeping / learned-wake-window sensors replace the
fixed Sleep / Wake schedule helpers as the blueprint's sleep and
optimal-stop inputs. The blueprint stays the sole thermostat controller;
the fork only provides the timing/occupancy signals.

Bedtime/wake are derived from a HOLISTIC quiet signal (see
``quiet_detector``), not hallway motion alone: a bedtime only counts
once the house has gone quiet — no media playing anywhere AND no
hallway activity (motion or lights) for a sustained window. A single
hallway-motion event is too weak on its own (a 01:00 water trip is not
a bedtime; a last motion at 21:00 says nothing if the TV then played
until 23:30).

Sources, in priority order:
  1. The occupancy sensor (``binary_sensor.house_occupied``) — live ticks
     from the coordinator and recorder backfill. This is the primary source.
  2. The quiet detector (hallway motion + hallway lights + any media
     player) — bedtime / wake-time inference. Without quiet data those
     stay at their prior values with confidence 0.
  3. The fixed schedule as a *prior*: until real days accumulate, the
     learner predicts the routine Ben already set (Sleep 22:00-05:30,
     Wake 05:30-07:00, Leave 07:00-15:00 away, Return 15:00-22:00).
     Priors count as ``PRIOR_WEIGHT_DAYS`` of evidence and are outweighed
     by real observations as days accumulate.

Model: per day-type (workday / weekend) and 30-minute slot, the posterior
probability the house is occupied. All times are local minutes since
midnight; datetimes are used as given (the coordinator passes local time).
"""

from __future__ import annotations

import logging
import statistics
from datetime import date, datetime, time, timedelta

from .quiet_detector import QuietDetector, DEFAULT_QUIET_MINUTES

_LOGGER = logging.getLogger(__name__)

SLOT_MINUTES = 30
SLOTS_PER_DAY = 24 * 60 // SLOT_MINUTES

# The fixed schedule counts as this many days of evidence. Real days
# outweigh it once ~3+ days of a slot are observed.
PRIOR_WEIGHT_DAYS = 2.0

# Days of sleep/wake observations needed for full confidence.
FULL_CONFIDENCE_DAYS = 10

# A slot counts as "observed" on a day once this many minutes of ticks
# landed in it (a full slot is 30 min; a partial day still counts).
MIN_OBSERVED_MINUTES_PER_SLOT = 15

# Wake window: the learned Wake (optimal-stop) window runs this long
# after the learned wake time (mirrors the fixed 05:30-07:00 = 90 min).
WAKE_WINDOW_MINUTES = 90

# Bedtime: last motion in this evening window. Wake: first motion in this
# morning window. (Local times.) These bound the quiet-period search in
# _finalise_sleep_for_day; the quiet detector does the actual inference.
BEDTIME_WINDOW = (19 * 60, 26 * 60)  # 19:00 -> 02:00 next day
WAKE_WINDOW = (4 * 60, 10 * 60)  # 04:00 -> 10:00

# Fixed-schedule prior (the household's blueprint schedule, Nov 10 2026):
# Sleep 22:00-05:30, Wake 05:30-07:00, Leave 07:00-15:00 (away),
# Return 15:00-22:00. Occupied everywhere except the Leave window.
DEFAULT_PRIOR_BEDTIME_MIN = 22 * 60
DEFAULT_PRIOR_WAKE_MIN = 5 * 60 + 30
DEFAULT_PRIOR_AWAY_START_MIN = 7 * 60
DEFAULT_PRIOR_AWAY_END_MIN = 15 * 60

DAY_TYPE_WORKDAY = "workday"
DAY_TYPE_WEEKEND = "weekend"


def day_type_for(d: date) -> str:
    """Workday (Mon-Fri) or weekend (Sat-Sun) bucket for a date."""
    return DAY_TYPE_WEEKEND if d.weekday() >= 5 else DAY_TYPE_WORKDAY


def _minutes_since_midnight(dt: datetime) -> int | None:
    """Local minutes since midnight, or None for non-datetime input.

    The coordinator's test harness mocks ``dt_util.now()`` with a
    MagicMock; the learner must degrade silently instead of raising.
    """
    try:
        hour = dt.hour
        minute = dt.minute
    except AttributeError:
        return None
    if not isinstance(hour, int) or not isinstance(minute, int):
        return None
    return hour * 60 + minute


def _day_type_of(dt: datetime) -> str | None:
    """Day-type for a datetime, or None for non-datetime input."""
    try:
        day = dt.date()
    except AttributeError:
        return None
    if not isinstance(day, date):
        return None
    return day_type_for(day)


def _slot_index(minutes: int) -> int:
    return max(0, min(SLOTS_PER_DAY - 1, minutes // SLOT_MINUTES))


def _median(values: list[int | float]) -> float | None:
    if not values:
        return None
    return float(statistics.median(values))


class ScheduleLearner:
    """Learn occupancy and sleep/wake patterns from history.

    Pure-Python: no Home Assistant imports, so it is unit-testable in
    isolation. The coordinator feeds it ticks, motion/media/light events
    and recorder backfill. Its outputs drive the blueprint (via the
    learned-occupancy / learned-sleeping / learned-wake-window binary
    sensors); the fork itself never writes the thermostat.
    """

    def __init__(
        self,
        prior_bedtime_min: int = DEFAULT_PRIOR_BEDTIME_MIN,
        prior_wake_min: int = DEFAULT_PRIOR_WAKE_MIN,
        prior_away_start_min: int = DEFAULT_PRIOR_AWAY_START_MIN,
        prior_away_end_min: int = DEFAULT_PRIOR_AWAY_END_MIN,
        prior_weight_days: float = PRIOR_WEIGHT_DAYS,
        quiet_minutes: int = DEFAULT_QUIET_MINUTES,
    ) -> None:
        self.prior_bedtime_min = prior_bedtime_min
        self.prior_wake_min = prior_wake_min
        self.prior_away_start_min = prior_away_start_min
        self.prior_away_end_min = prior_away_end_min
        self.prior_weight_days = prior_weight_days

        # Holistic quiet detector: hallway motion + lights + media.
        # Bedtime/wake come from its quiet periods, never motion alone.
        self.quiet_detector = QuietDetector(quiet_minutes=quiet_minutes)

        # Per day-type, per slot: [days observed occupied, days observed].
        self._slot_occupied: dict[str, list[int]] = {
            DAY_TYPE_WORKDAY: [0] * SLOTS_PER_DAY,
            DAY_TYPE_WEEKEND: [0] * SLOTS_PER_DAY,
        }
        self._slot_days: dict[str, list[int]] = {
            DAY_TYPE_WORKDAY: [0] * SLOTS_PER_DAY,
            DAY_TYPE_WEEKEND: [0] * SLOTS_PER_DAY,
        }
        # Per day-type: list of (bedtime_min, wake_min); either may be None.
        self._sleep_observations: dict[str, list[tuple[int | None, int | None]]] = {
            DAY_TYPE_WORKDAY: [],
            DAY_TYPE_WEEKEND: [],
        }
        # Dates already finalised (avoid double counting on replays).
        self._finalised_dates: set[str] = set()
        # (bedtime, wake) per date. Keyed by date so re-deriving a day
        # (e.g. motion backfill first, media backfill later) replaces its
        # observation instead of appending a duplicate.
        self._sleep_by_date: dict[str, tuple[int | None, int | None]] = {}
        # Dates whose bedtime call Ben rejected via a notification
        # response (see apply_bedtime_feedback). A rejected date's
        # bedtime is never (re-)derived; its wake is unaffected.
        self._rejected_bedtimes: set[str] = set()

        # In-progress day accumulation (not persisted; a partial day is
        # rebuilt from ticks and dropped on restart).
        self._current_date: date | None = None
        # slot -> [occupied minutes, observed minutes]
        self._current_slots: dict[int, list[float]] = {}
        # Motion events for the current + recent days, capped.
        self._motion_events: list[datetime] = []

    # -- priors ---------------------------------------------------------

    def _prior_occupied(self, slot: int) -> float:
        """Fixed-schedule prior P(occupied) for a slot (1.0 home, 0.0 away)."""
        start_min = slot * SLOT_MINUTES
        if self.prior_away_start_min <= start_min < self.prior_away_end_min:
            return 0.0
        return 1.0

    # -- live feeding ---------------------------------------------------

    def record_tick(self, dt: datetime, occupied: bool) -> None:
        """Record one coordinator-cycle observation (called ~every minute)."""
        try:
            day = dt.date()
        except AttributeError:
            return
        if not isinstance(day, date):
            return  # Mocked/non-datetime input (test harness); ignore.
        minutes = _minutes_since_midnight(dt)
        if minutes is None:
            return
        if self._current_date is not None and day != self._current_date:
            self.finalize_day(self._current_date)
        self._current_date = day
        slot = _slot_index(minutes)
        bucket = self._current_slots.setdefault(slot, [0.0, 0.0])
        bucket[1] += 1.0
        if occupied:
            bucket[0] += 1.0

    def record_motion(self, dt: datetime) -> None:
        """Record a hallway-motion event.

        Motion is one activity input to the holistic quiet detector;
        bedtime/wake are derived from quiet periods, never from motion
        alone.
        """
        if not isinstance(dt, datetime):
            return  # Mocked/non-datetime input (test harness); ignore.
        self._motion_events.append(dt)
        self.quiet_detector.record_activity(dt)
        # Keep a bounded window (enough for ~2 months of events).
        if len(self._motion_events) > 4000:
            self._motion_events = self._motion_events[-4000:]

    def record_light(self, dt: datetime) -> None:
        """Record a hallway-light-on event (activity, like motion)."""
        if not isinstance(dt, datetime):
            return
        self.quiet_detector.record_activity(dt)

    def record_media(self, dt: datetime, playing: bool) -> None:
        """Record a media state change (any media_player, playing or not)."""
        if not isinstance(dt, datetime):
            return
        self.quiet_detector.record_media(dt, playing)

    def record_day(self, day: date, occupied_intervals: list[tuple[int, int]]) -> None:
        """Record one full day of occupancy from (start_min, end_min) intervals.

        Used for recorder backfill. Intervals are minutes since local
        midnight; end is exclusive and may exceed 1440 (past midnight).
        """
        key = day.isoformat()
        if key in self._finalised_dates:
            return
        dtype = day_type_for(day)
        occupied_slots: set[int] = set()
        for start, end in occupied_intervals:
            start = max(0, start)
            end = min(24 * 60, end)
            for minute in range(start, end, SLOT_MINUTES):
                occupied_slots.add(_slot_index(minute))
            # A trailing partial slot still counts if it covers >= half.
            if end > start and (end - start) % SLOT_MINUTES >= SLOT_MINUTES // 2:
                occupied_slots.add(_slot_index(end - 1))
        for slot in range(SLOTS_PER_DAY):
            self._slot_days[dtype][slot] += 1
            if slot in occupied_slots:
                self._slot_occupied[dtype][slot] += 1
        self._finalised_dates.add(key)

    def ingest_state_history(self, states: list[tuple[datetime, str]]) -> None:
        """Build per-day occupancy intervals from (datetime, state) pairs.

        States are chronological recorder states for the occupancy sensor.
        An "on" interval runs until the next "off"; a trailing "on" with no
        "off" is capped at the last timestamp in the list.
        """
        if not states:
            return
        intervals_by_day: dict[date, list[tuple[int, int]]] = {}
        on_since: datetime | None = None
        for dt, state in states:
            if state == "on" and on_since is None:
                on_since = dt
            elif state != "on" and on_since is not None:
                self._add_interval(intervals_by_day, on_since, dt)
                on_since = None
        if on_since is not None:
            self._add_interval(intervals_by_day, on_since, states[-1][0])
        for day, intervals in intervals_by_day.items():
            self.record_day(day, intervals)

    @staticmethod
    def _add_interval(
        intervals_by_day: dict[date, list[tuple[int, int]]],
        start: datetime,
        end: datetime,
    ) -> None:
        """Split an occupancy interval across local-midnight boundaries."""
        cursor = start
        while cursor.date() < end.date():
            midnight = datetime.combine(
                cursor.date() + timedelta(days=1), time.min, tzinfo=cursor.tzinfo
            )
            start_min = _minutes_since_midnight(cursor)
            if start_min is None:
                return
            intervals_by_day.setdefault(cursor.date(), []).append(
                (start_min, 24 * 60)
            )
            cursor = midnight
        start_min = _minutes_since_midnight(cursor)
        end_min = _minutes_since_midnight(end)
        if start_min is None or end_min is None:
            return
        intervals_by_day.setdefault(cursor.date(), []).append((start_min, end_min))

    def ingest_motion_history(self, motion_times: list[datetime]) -> None:
        """Feed past motion timestamps (recorder backfill for sleep/wake)."""
        for dt in motion_times:
            self.record_motion(dt)
        # Re-derive sleep observations for days covered by the new events.
        days = {dt.date() for dt in motion_times}
        for day in sorted(days):
            self._finalise_sleep_for_day(day)

    def ingest_media_history(self, states: list[tuple[datetime, bool]]) -> None:
        """Feed past media state changes (recorder backfill).

        ``states`` is a chronological list of (datetime, playing) pairs
        for any media_player. Re-derives sleep for the covered days.
        """
        for dt, playing in states:
            self.record_media(dt, playing)
        days = {dt.date() for dt, _ in states}
        for day in sorted(days):
            self._finalise_sleep_for_day(day)
            # A media event late in the evening belongs to the next
            # morning's wake window too (the night's quiet period ends
            # the following morning); re-derive the next day as well.
            self._finalise_sleep_for_day(day + timedelta(days=1))

    def ingest_activity_history(self, activity_times: list[datetime]) -> None:
        """Feed past hallway-activity timestamps (e.g. light-on events)."""
        for dt in activity_times:
            self.record_light(dt)
        days = {dt.date() for dt in activity_times}
        for day in sorted(days):
            self._finalise_sleep_for_day(day)

    # -- notification feedback (Ben's Yes/No responses) -----------------

    def _bedtime_feedback_target_date(self, now: datetime) -> str | None:
        """The night the feedback refers to, from the event time.

        A response sent in the evening (>= 19:00) is about tonight's
        bedtime call; one sent before 19:00 (after midnight, in the
        morning, or during the day) is about last night's. The date is
        derived from the event time, not from the latest observation:
        repeating the same 'no' must be a no-op, not discard a
        different night's bedtime.
        """
        try:
            day = now.date()
            hour = now.hour
        except AttributeError:
            return None
        if not isinstance(day, date) or not isinstance(hour, int):
            return None
        if hour >= 19:
            return day.isoformat()
        return (day - timedelta(days=1)).isoformat()

    def apply_bedtime_feedback(self, now: datetime, confirmed: bool) -> bool:
        """Apply a Yes/No response to the latest bedtime call.

        ``confirmed=True`` ('yes'): the derived observation stands —
        no change. ``confirmed=False`` ('no'): the call was wrong, so
        the bedtime observation for that date is discarded and the
        date is marked rejected, so a later finalisation will not
        re-derive it. The wake observation for the date is unaffected
        (a wrong bedtime says nothing about the wake). Returns True
        when the learner state changed.
        """
        if confirmed:
            return False
        date_key = self._bedtime_feedback_target_date(now)
        if date_key is None:
            return False
        changed = False
        if date_key not in self._rejected_bedtimes:
            self._rejected_bedtimes.add(date_key)
            changed = True
        obs = self._sleep_by_date.get(date_key)
        if obs is not None and obs[0] is not None:
            self._sleep_by_date[date_key] = (None, obs[1])
            self._rebuild_sleep_observations(day_type_for(date.fromisoformat(date_key)))
            changed = True
        return changed

    def apply_occupancy_feedback(self, now: datetime, confirmed: bool) -> bool:
        """Apply a Yes/No response to the latest home/away call.

        ``confirmed=True`` ('yes'): the call was right — no change.
        ``confirmed=False`` ('no'): the system's occupancy call was
        wrong, so the true state is the opposite of what the learner
        currently predicts for this slot. One correction observation
        (a day's worth of evidence) is added for the current
        slot/day-type, down-weighting the wrong call. Returns True
        when the learner state changed.
        """
        if confirmed:
            return False
        dtype = _day_type_of(now)
        minutes = _minutes_since_midnight(now)
        if dtype is None or minutes is None:
            return False
        slot = _slot_index(minutes)
        thinks_home = self.predict_occupancy(now) >= 0.5
        actually_home = not thinks_home
        self._slot_days[dtype][slot] += 1
        if actually_home:
            self._slot_occupied[dtype][slot] += 1
        return True

    # -- day finalisation ------------------------------------------------

    def finalize_day(self, day: date) -> None:
        """Roll the in-progress day's ticks into the slot counts."""
        if self._current_date is not None and day != self._current_date:
            # The caller finalised a stale day; nothing to do.
            return
        key = day.isoformat()
        if key in self._finalised_dates:
            self._current_slots = {}
            self._current_date = None
            return
        dtype = day_type_for(day)
        for slot, (occ_min, obs_min) in self._current_slots.items():
            if obs_min >= MIN_OBSERVED_MINUTES_PER_SLOT:
                self._slot_days[dtype][slot] += 1
                if occ_min >= obs_min / 2.0:
                    self._slot_occupied[dtype][slot] += 1
        self._finalised_dates.add(key)
        self._current_slots = {}
        self._current_date = None
        # Sleep for the PREVIOUS day: its bedtime window runs to 02:00
        # this morning, so it is only complete now. (Today's bedtime
        # window is still open; it finalises at the next rollover.)
        self._finalise_sleep_for_day(day - timedelta(days=1))

    def _finalise_sleep_for_day(self, day: date) -> None:
        """Derive one day's (bedtime, wake) from the holistic quiet signal.

        Bedtime = start of the last quiet period beginning in the evening
        window (19:00 -> 02:00 next day). Wake = end of the first quiet
        period ending in the morning window (04:00 -> 10:00). A bedtime
        is only credible once the house has gone quiet — no media playing
        anywhere AND no hallway activity for a sustained window — so a
        01:00 water trip or a TV left on does not move it. Idempotent:
        re-deriving a day replaces its observation (motion backfill,
        media backfill and light backfill all feed the same day).
        """
        dtype = day_type_for(day)
        # Wide window: the night before (for the wake) through the next
        # morning (for the bedtime quiet period's end).
        window_start = datetime.combine(day - timedelta(days=1), time(19, 0))
        window_end = datetime.combine(day + timedelta(days=1), time(10, 0))
        if not self.quiet_detector.has_data(window_start, window_end):
            return
        periods = self.quiet_detector.quiet_periods(window_start, window_end)
        if not periods:
            return

        midnight = datetime.combine(day, time.min)

        def _to_minutes(dt: datetime) -> int:
            return int((dt.replace(tzinfo=None) - midnight).total_seconds() // 60)

        # Bedtime: last quiet period starting in [19:00, 02:00 next day).
        evening_start = datetime.combine(day, time(19, 0))
        evening_end = datetime.combine(day + timedelta(days=1), time(2, 0))
        bedtime: int | None = None
        for period_start, _period_end in periods:
            if evening_start <= period_start < evening_end:
                bedtime = _to_minutes(period_start)
        # Wake: first quiet period ending in [04:00, 10:00).
        morning_start = datetime.combine(day, time(4, 0))
        morning_end = datetime.combine(day, time(10, 0))
        wake: int | None = None
        for _period_start, period_end in periods:
            if period_end is not None and morning_start <= period_end < morning_end:
                wake = _to_minutes(period_end)
                break
        if day.isoformat() in self._rejected_bedtimes:
            # Ben rejected this bedtime call via a notification
            # response: keep the wake (a separate observation) but
            # never (re-)derive the bedtime for this date.
            bedtime = None
        if bedtime is None and wake is None:
            return
        self._sleep_by_date[day.isoformat()] = (bedtime, wake)
        self._rebuild_sleep_observations(dtype)

    def _rebuild_sleep_observations(self, dtype: str) -> None:
        """Rebuild one day-type's observation list from the per-day map."""
        self._sleep_observations[dtype] = [
            obs
            for date_key, obs in sorted(self._sleep_by_date.items())
            if day_type_for(date.fromisoformat(date_key)) == dtype
        ][-60:]

    def is_sleeping(self, dt: datetime) -> bool:
        """True when ``dt`` falls inside the learned sleep window.

        Uses the learned bedtime/wake medians for the day-type; falls
        back to the fixed-schedule prior (22:00-05:30) until real sleep
        data exists. This is the drive signal for the blueprint's sleep
        input.
        """
        dtype = _day_type_of(dt)
        minutes = _minutes_since_midnight(dt)
        if dtype is None or minutes is None:
            return False
        bedtime, _, _ = self.learned_bedtime(dtype)
        wake, _, _ = self.learned_wake(dtype)
        if bedtime is None or wake is None:
            return False
        bed = bedtime % (24 * 60)
        wak = wake % (24 * 60)
        if bed > wak:  # window crosses midnight (the normal case)
            return minutes >= bed or minutes < wak
        return bed <= minutes < wak

    def is_awake(self, dt: datetime) -> bool:
        """True when the house is demonstrably awake right now.

        Sustained hallway activity / media after a quiet night (see
        ``QuietDetector.awake_since``). A brief night burst does not
        count, so a water trip cannot wake the house. The coordinator
        uses this to end the learned sleep window early: the clock
        alone kept the house "asleep" until the learned wake time even
        when Ben was up and moving (Nov 14 2026).
        """
        return self.quiet_detector.awake_since(dt) is not None

    def in_wake_window(self, dt: datetime, window_minutes: int = WAKE_WINDOW_MINUTES) -> bool:
        """True when ``dt`` is inside the learned wake window.

        The wake window runs ``window_minutes`` after the learned wake
        time (mirrors the fixed 05:30-07:00 = 90 min). Drive signal for
        the blueprint's optimal-stop (Wake) input.
        """
        dtype = _day_type_of(dt)
        minutes = _minutes_since_midnight(dt)
        if dtype is None or minutes is None:
            return False
        wake, _, _ = self.learned_wake(dtype)
        if wake is None:
            return False
        wak = wake % (24 * 60)
        return (minutes - wak) % (24 * 60) < window_minutes

    # -- queries ----------------------------------------------------------

    @property
    def days_of_data(self) -> int:
        """Number of distinct days finalised (both day-types)."""
        return len(self._finalised_dates)

    @property
    def confidence(self) -> float:
        """Overall schedule confidence 0..1 from days of data."""
        return min(1.0, self.days_of_data / 14.0)

    def occupancy_probability(self, dtype: str, slot: int) -> float:
        """Posterior P(occupied) for a day-type and slot (prior + data)."""
        prior = self._prior_occupied(slot)
        observed = self._slot_days[dtype][slot]
        occupied = self._slot_occupied[dtype][slot]
        weight = self.prior_weight_days
        return (weight * prior + occupied) / (weight + observed)

    def predict_occupancy(self, dt: datetime) -> float:
        """Predicted P(occupied) at a datetime (0.0 for non-datetime input)."""
        dtype = _day_type_of(dt)
        minutes = _minutes_since_midnight(dt)
        if dtype is None or minutes is None:
            return 0.0
        return self.occupancy_probability(dtype, _slot_index(minutes))

    def predict_next_transition(
        self, now: datetime
    ) -> tuple[datetime | None, str | None, float]:
        """Next predicted occupancy change within 24h.

        Returns (datetime, "arrival" | "departure", confidence) or
        (None, None, 0.0) when the profile predicts no change.
        """
        dtype = _day_type_of(now)
        if dtype is None:
            return None, None, 0.0
        current_p = self.predict_occupancy(now)
        currently_home = current_p >= 0.5
        for step in range(1, SLOTS_PER_DAY + 1):
            candidate = now + timedelta(minutes=SLOT_MINUTES * step)
            p = self.predict_occupancy(candidate)
            home = p >= 0.5
            if home != currently_home:
                kind = "arrival" if home else "departure"
                candidate_dtype = _day_type_of(candidate)
                candidate_minutes = _minutes_since_midnight(candidate)
                if candidate_dtype is None or candidate_minutes is None:
                    return None, None, 0.0
                slot = _slot_index(candidate_minutes)
                slot_p = self.occupancy_probability(candidate_dtype, slot)
                confidence = self.confidence * (0.5 + abs(slot_p - 0.5))
                return candidate, kind, round(min(1.0, confidence), 2)
        return None, None, 0.0

    def learned_bedtime(self, dtype: str) -> tuple[int | None, float, int]:
        """(bedtime minutes, confidence, observations) for a day-type.

        Falls back to the prior when no motion data exists; confidence 0.
        """
        values = [b for b, _ in self._sleep_observations[dtype] if b is not None]
        if not values:
            return self.prior_bedtime_min, 0.0, 0
        median = _median(values)
        return (
            int(round(median)) if median is not None else None,
            round(min(1.0, len(values) / FULL_CONFIDENCE_DAYS), 2),
            len(values),
        )

    def learned_wake(self, dtype: str) -> tuple[int | None, float, int]:
        """(wake minutes, confidence, observations) for a day-type."""
        values = [w for _, w in self._sleep_observations[dtype] if w is not None]
        if not values:
            return self.prior_wake_min, 0.0, 0
        median = _median(values)
        return (
            int(round(median)) if median is not None else None,
            round(min(1.0, len(values) / FULL_CONFIDENCE_DAYS), 2),
            len(values),
        )

    def learned_leave_return(self, dtype: str) -> tuple[int | None, int | None]:
        """(leave minutes, return minutes) from the occupancy profile.

        Leave = first slot after 06:00 predicted away; return = first slot
        after 12:00 predicted home. None when the profile never leaves /
        never returns (e.g. prior-only weekend data is home all day).
        """
        leave: int | None = None
        ret: int | None = None
        for slot in range(SLOTS_PER_DAY):
            start_min = slot * SLOT_MINUTES
            if start_min < 6 * 60:
                continue
            if self.occupancy_probability(dtype, slot) < 0.5:
                if leave is None:
                    leave = start_min
            elif start_min >= 12 * 60 and leave is not None and ret is None:
                ret = start_min
        return leave, ret

    def suggested_schedule(self, dtype: str = DAY_TYPE_WORKDAY) -> dict:
        """Compare the learned routine against the fixed schedule.

        The learned routine now DRIVES the blueprint (via the learned
        binary sensors); this comparison is the report of how far the
        learned routine has moved from the fixed schedule. Deltas are
        in minutes (learned minus fixed); positive means later than fixed.
        """
        bedtime, bed_conf, bed_n = self.learned_bedtime(dtype)
        wake, wake_conf, wake_n = self.learned_wake(dtype)
        leave, ret = self.learned_leave_return(dtype)
        fixed = {
            "bedtime_min": self.prior_bedtime_min,
            "wake_min": self.prior_wake_min,
            "leave_min": self.prior_away_start_min,
            "return_min": self.prior_away_end_min,
        }
        learned = {
            "bedtime_min": bedtime,
            "wake_min": wake,
            "leave_min": leave,
            "return_min": ret,
        }
        deltas = {
            key: (learned[key] - fixed[key]) if learned[key] is not None else None
            for key in fixed
        }
        return {
            "day_type": dtype,
            "fixed": fixed,
            "learned": learned,
            "delta_minutes": deltas,
            "confidence": round(self.confidence, 2),
            "bedtime_observations": bed_n,
            "wake_observations": wake_n,
            "days_of_data": self.days_of_data,
        }

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise learned state for the coordinator store."""
        return {
            "slot_occupied": self._slot_occupied,
            "slot_days": self._slot_days,
            "sleep_observations": {
                dtype: [[b, w] for b, w in obs]
                for dtype, obs in self._sleep_observations.items()
            },
            "finalised_dates": sorted(self._finalised_dates)[-90:],
            # Per-day (bedtime, wake) map; the source of truth for the
            # observation lists above (rebuilt from it on restore).
            "sleep_by_date": {
                date_key: [b, w]
                for date_key, (b, w) in sorted(self._sleep_by_date.items())[-90:]
            },
            # Holistic quiet detector event history (activity + media).
            "quiet": self.quiet_detector.to_dict(),
            # Bedtime calls Ben rejected via notification feedback.
            "rejected_bedtimes": sorted(self._rejected_bedtimes)[-90:],
        }

    @classmethod
    def from_dict(cls, data: dict | None, **kwargs) -> "ScheduleLearner":
        """Restore learned state; tolerates missing/partial data."""
        learner = cls(**kwargs)
        if not data:
            return learner
        try:
            for dtype in (DAY_TYPE_WORKDAY, DAY_TYPE_WEEKEND):
                occ = data.get("slot_occupied", {}).get(dtype)
                days = data.get("slot_days", {}).get(dtype)
                if isinstance(occ, list) and len(occ) == SLOTS_PER_DAY:
                    learner._slot_occupied[dtype] = [int(v) for v in occ]
                if isinstance(days, list) and len(days) == SLOTS_PER_DAY:
                    learner._slot_days[dtype] = [int(v) for v in days]
                sleep = data.get("sleep_observations", {}).get(dtype)
                if isinstance(sleep, list):
                    learner._sleep_observations[dtype] = [
                        (b, w) for b, w in sleep[:60]
                    ]
            dates = data.get("finalised_dates")
            if isinstance(dates, list):
                learner._finalised_dates = set(dates[-90:])
            # Per-day sleep map (preferred); rebuild the observation
            # lists from it so re-derivation stays idempotent.
            sleep_by_date = data.get("sleep_by_date")
            if isinstance(sleep_by_date, dict):
                for date_key, obs in sleep_by_date.items():
                    if isinstance(obs, list) and len(obs) == 2:
                        learner._sleep_by_date[date_key] = (obs[0], obs[1])
                for dtype in (DAY_TYPE_WORKDAY, DAY_TYPE_WEEKEND):
                    learner._rebuild_sleep_observations(dtype)
            # Back-compat: old stores kept only the observation lists and
            # the finalised-date set; those load above and _sleep_by_date
            # simply stays empty until new days are finalised.
            rejected = data.get("rejected_bedtimes")
            if isinstance(rejected, list):
                learner._rejected_bedtimes = {
                    item for item in rejected if isinstance(item, str)
                }
            quiet = data.get("quiet")
            if isinstance(quiet, dict):
                learner.quiet_detector = QuietDetector.from_dict(quiet)
        except (TypeError, ValueError, AttributeError) as err:
            _LOGGER.warning("ScheduleLearner: ignoring corrupt stored state: %s", err)
        return learner


def backtest_return_prediction(
    days: list[tuple[date, list[tuple[int, int]]]],
    fixed_return_min: int = DEFAULT_PRIOR_AWAY_END_MIN,
    train_fraction: float = 0.7,
) -> dict:
    """Hold-out test: learned return time vs the fixed schedule's return.

    ``days`` is chronological (date, occupied intervals). Trains a fresh
    learner on the first ``train_fraction`` of days and evaluates the rest:
    predicted return = first slot after 12:00 with P(occupied) >= 0.5;
    actual return = first occupied slot after 12:00 that day. Returns MAE
    in minutes for the learned and the fixed predictions, plus per-day
    errors. Days with no return (home all day) are skipped.
    """
    if len(days) < 4:
        return {"status": "insufficient_data", "days": len(days)}
    split = max(2, int(len(days) * train_fraction))
    train, test = days[:split], days[split:]
    learner = ScheduleLearner()
    for day, intervals in train:
        learner.record_day(day, intervals)

    learned_errors: list[float] = []
    fixed_errors: list[float] = []
    per_day: list[dict] = []
    for day, intervals in test:
        actual = None
        for start, _end in sorted(intervals):
            if start >= 12 * 60:
                actual = start
                break
        if actual is None:
            continue
        dtype = day_type_for(day)
        predicted: int | None = None
        for slot in range(SLOTS_PER_DAY):
            start_min = slot * SLOT_MINUTES
            if start_min < 12 * 60:
                continue
            if learner.occupancy_probability(dtype, slot) >= 0.5:
                predicted = start_min
                break
        if predicted is None:
            continue
        learned_errors.append(abs(predicted - actual))
        fixed_errors.append(abs(fixed_return_min - actual))
        per_day.append(
            {"date": day.isoformat(), "actual": actual, "predicted": predicted}
        )

    if not learned_errors:
        return {"status": "no_test_returns", "days": len(days)}
    return {
        "status": "ok",
        "days": len(days),
        "train_days": len(train),
        "test_days": len(per_day),
        "learned_mae_min": round(sum(learned_errors) / len(learned_errors), 1),
        "fixed_mae_min": round(sum(fixed_errors) / len(fixed_errors), 1),
        "per_day": per_day,
    }
