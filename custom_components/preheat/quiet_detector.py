"""Holistic "house is quiet / asleep" detection (fork Phase 4, drive update).

A bedtime is only credible when the house has gone quiet: no media
playing anywhere AND no hallway activity (motion or lights) for a
sustained window. A single hallway-motion event is a weak bedtime signal
on its own — someone walking to the kitchen at 01:00 is not going to
bed, and a last motion at 21:00 says nothing if the TV then played
until 23:30.

Model: the house is "noisy" at activity instants (hallway motion on,
hallway light on) and during media-playing intervals. Quiet periods are
the gaps between noise, kept only when they last at least QUIET_MINUTES.
Adjacent quiet periods separated by a brief activity-only burst (<
MERGE_GAP_MINUTES, no media playing, burst ends before 04:00) are
merged, so a 01:00 water trip does not reset the bedtime — but a burst
inside the morning wake window is never merged, so it cannot swallow
the wake event.

Pure Python: no Home Assistant imports, unit-testable in isolation.
Datetimes are normalised to naive local time on ingestion (the learner
passes local datetimes; tests pass naive ones).
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

_LOGGER = logging.getLogger(__name__)

# Sustained quiet window: a bedtime only counts once the house has been
# quiet (no media, no hallway activity) for this long.
DEFAULT_QUIET_MINUTES = 30

# A brief activity-only burst shorter than this does not break a quiet
# stretch (e.g. a 01:00 water trip). Media playing during the burst
# prevents the merge — the house was genuinely awake.
MERGE_GAP_MINUTES = 15

# Sustained activity needed to count as awake (live wake-up, Nov 14
# 2026): a water trip is ~2 min of motion; a morning routine is
# repeated motion over 10+ minutes. See awake_since().
AWAKE_SUSTAIN_MINUTES = 10

# Lookback for the live wake-up check: the night's quiet period
# started ~12 h ago; a full day covers it with margin.
AWAKE_LOOKBACK_HOURS = 24

# Activity / media events are capped (enough for ~2 months of events).
MAX_EVENTS = 4000


def _naive(dt: datetime) -> datetime | None:
    """Local naive datetime, or None for non-datetime input."""
    if not isinstance(dt, datetime):
        return None
    return dt.replace(tzinfo=None)


class QuietDetector:
    """Track hallway activity + media state; report quiet periods.

    The coordinator feeds it activity events (hallway motion on,
    hallway light on) and media state changes (any media_player). The
    schedule learner asks it for the quiet periods of a night and
    derives bedtime/wake from those — never from motion alone.
    """

    def __init__(
        self,
        quiet_minutes: int = DEFAULT_QUIET_MINUTES,
        merge_gap_minutes: int = MERGE_GAP_MINUTES,
    ) -> None:
        self.quiet_minutes = quiet_minutes
        self.merge_gap_minutes = merge_gap_minutes
        # Activity instants (motion on / light on), chronological.
        self._activity: list[datetime] = []
        # Media state changes as (datetime, playing), chronological.
        self._media: list[tuple[datetime, bool]] = []

    # -- feeding ---------------------------------------------------------

    def record_activity(self, dt: datetime) -> None:
        """One hallway-activity instant (motion on, or light on)."""
        naive = _naive(dt)
        if naive is None:
            return
        self._activity.append(naive)
        if len(self._activity) > MAX_EVENTS:
            self._activity = self._activity[-MAX_EVENTS:]

    def record_media(self, dt: datetime, playing: bool) -> None:
        """One media state change (any media_player playing / not)."""
        naive = _naive(dt)
        if naive is None:
            return
        self._media.append((naive, bool(playing)))
        if len(self._media) > MAX_EVENTS:
            self._media = self._media[-MAX_EVENTS:]

    # -- queries ----------------------------------------------------------

    def _noise_intervals(
        self, start_n: datetime, end_n: datetime
    ) -> list[tuple[datetime, datetime, bool]]:
        """Merged noise intervals in [start_n, end_n].

        Activity instants are points; media-on spans are intervals (a
        span still playing at ``end_n`` is clipped to it). Overlapping
        intervals merge (a motion point inside a media span, overlapping
        media spans, ...). Shared by quiet_periods() and awake_since().
        """
        noise: list[tuple[datetime, datetime, bool]] = []
        for event in self._activity:
            if start_n <= event <= end_n:
                noise.append((event, event, False))
        media_on: datetime | None = None
        for event_dt, playing in self._media:
            if playing and media_on is None:
                media_on = event_dt
            elif not playing and media_on is not None:
                interval_start, interval_end = media_on, event_dt
                media_on = None
                if interval_end > start_n and interval_start < end_n:
                    noise.append(
                        (max(interval_start, start_n), min(interval_end, end_n), True)
                    )
        if media_on is not None and media_on < end_n:
            noise.append((max(media_on, start_n), end_n, True))

        if not noise:
            return []
        noise.sort(key=lambda item: item[0])
        merged: list[tuple[datetime, datetime, bool]] = []
        for item_start, item_end, is_media in noise:
            if merged and item_start <= merged[-1][1]:
                prev_start, prev_end, prev_media = merged[-1]
                merged[-1] = (
                    prev_start,
                    max(prev_end, item_end),
                    prev_media or is_media,
                )
            else:
                merged.append((item_start, item_end, is_media))
        return merged

    def media_playing_at(self, dt: datetime) -> bool:
        """Whether any media was playing at ``dt`` (False if unknown)."""
        naive = _naive(dt)
        if naive is None:
            return False
        playing = False
        for event_dt, event_playing in self._media:
            if event_dt > naive:
                break
            playing = event_playing
        return playing

    def is_quiet(self, now: datetime, quiet_minutes: int | None = None) -> bool:
        """True when nothing is playing and no activity in the last window."""
        naive = _naive(now)
        if naive is None:
            return False
        window = quiet_minutes if quiet_minutes is not None else self.quiet_minutes
        if self.media_playing_at(naive):
            return False
        cutoff = naive - timedelta(minutes=window)
        for event in reversed(self._activity):
            if event <= cutoff:
                break
            if event > cutoff:
                return False
        return True

    def quiet_since(self, now: datetime) -> datetime | None:
        """When the current quiet period started, or None if not quiet."""
        naive = _naive(now)
        if naive is None or self.media_playing_at(naive):
            return None
        last_noise: datetime | None = None
        for event in reversed(self._activity):
            if event <= naive:
                last_noise = event
                break
        # A media interval that ended recently also ends the quiet stretch.
        for event_dt, playing in reversed(self._media):
            if event_dt > naive:
                continue
            if not playing and (last_noise is None or event_dt > last_noise):
                last_noise = event_dt
            break
        return last_noise

    def awake_since(
        self, now: datetime, sustain_minutes: int = AWAKE_SUSTAIN_MINUTES
    ) -> datetime | None:
        """Start of the current sustained-awake stretch, or None.

        Live wake-up (fork, Nov 14 2026): the learned sleep window is a
        clock (learned bedtime -> learned wake), so on Nov 14 the house
        stayed "asleep" until the learned 09:04 wake even though Ben was
        up and moving from 07:54. The coordinator clears the sleeping
        signal when this returns a time.

        A stretch starts at the first noise (hallway activity or media
        starting) after a quiet period of at least ``quiet_minutes``.
        It counts as awake only once the noise has lasted
        ``sustain_minutes`` — a 2-minute water trip never reaches it,
        and a burst that ends before the threshold never counts, even
        after 10 quiet minutes have passed since. Once awake, the
        stretch stays awake until the house has gone quiet for
        ``quiet_minutes`` again, so sitting still after the morning
        routine does not flip the house back to sleep after a few
        quiet minutes. Media playing counts as awake, as it does for
        bedtime; a TV left playing all night therefore keeps the house
        awake (a warmer night, not a cold one).

        Returns the naive-local start of the stretch, or None when the
        house is currently quiet, has no preceding quiet period, or the
        current noise has not sustained long enough.
        """
        naive = _naive(now)
        if naive is None:
            return None
        window_start = naive - timedelta(hours=AWAKE_LOOKBACK_HOURS)
        merged = self._noise_intervals(window_start, naive)
        if not merged:
            return None
        # Last completed quiet period (gap >= quiet_minutes) before now;
        # its end is the first noise of the current stretch.
        cursor = window_start
        last_quiet_end: datetime | None = None
        for noise_start, noise_end, _ in merged:
            if noise_start > cursor:
                if (noise_start - cursor).total_seconds() / 60.0 >= self.quiet_minutes:
                    last_quiet_end = noise_start
            cursor = max(cursor, noise_end)
        # Trailing gap: the house has gone quiet again -> stretch over.
        if (naive - cursor).total_seconds() / 60.0 >= self.quiet_minutes:
            return None
        if last_quiet_end is None:
            return None
        # Latest noise at or before now; a media span still playing at
        # ``now`` was clipped to the window end, so it reads as now.
        last_noise = max(noise_end for _, noise_end, _ in merged)
        if (last_noise - last_quiet_end).total_seconds() / 60.0 >= sustain_minutes:
            return last_quiet_end
        return None

    def has_data(self, start: datetime, end: datetime) -> bool:
        """Any activity or media event in [start, end] (i.e. signal, not silence)."""
        start_n, end_n = _naive(start), _naive(end)
        if start_n is None or end_n is None:
            return False
        for event in self._activity:
            if start_n <= event <= end_n:
                return True
        for event_dt, _ in self._media:
            if start_n <= event_dt <= end_n:
                return True
        return False

    def quiet_periods(
        self, start: datetime, end: datetime
    ) -> list[tuple[datetime, datetime | None]]:
        """Quiet periods in [start, end], each >= quiet_minutes long.

        Returns merged periods as (start, end) with end None for a period
        still running at ``end``. A period is a gap between noise
        (activity instants + media-playing intervals) that lasts at least
        ``quiet_minutes``. Adjacent periods separated by a brief
        activity-only burst (< merge_gap_minutes, no media, burst ends
        before 04:00) are merged into one.
        """
        start_n, end_n = _naive(start), _naive(end)
        if start_n is None or end_n is None or end_n <= start_n:
            return []

        merged = self._noise_intervals(start_n, end_n)
        if not merged:
            # No noise at all in the window: one long quiet period.
            if (end_n - start_n).total_seconds() / 60.0 >= self.quiet_minutes:
                return [(start_n, None)]
            return []

        # Gaps between merged noise intervals are quiet candidates.
        periods: list[tuple[datetime, datetime | None]] = []
        cursor = start_n
        for noise_start, noise_end, _ in merged:
            if noise_start > cursor:
                if (noise_start - cursor).total_seconds() / 60.0 >= self.quiet_minutes:
                    periods.append((cursor, noise_start))
            cursor = max(cursor, noise_end)
        if cursor < end_n:
            if (end_n - cursor).total_seconds() / 60.0 >= self.quiet_minutes:
                periods.append((cursor, None))

        return self._merge_brief_bursts(periods)

    def _merge_brief_bursts(
        self, periods: list[tuple[datetime, datetime | None]]
    ) -> list[tuple[datetime, datetime | None]]:
        """Merge quiet periods split by a brief activity-only burst.

        Two adjacent periods are merged when the noise between them is
        shorter than merge_gap_minutes, contains no media playing, and
        ends before 04:00 (a burst inside the morning wake window is a
        wake event, never a water trip).
        """
        if len(periods) < 2:
            return periods
        merged: list[tuple[datetime, datetime | None]] = [periods[0]]
        for period_start, period_end in periods[1:]:
            prev_start, prev_end = merged[-1]
            if prev_end is None:
                # Previous period is still running; nothing after it.
                merged.append((period_start, period_end))
                continue
            gap_min = (period_start - prev_end).total_seconds() / 60.0
            burst_has_media = self._media_played_between(prev_end, period_start)
            burst_ends_before_4am = period_start.hour < 4
            if (
                gap_min < self.merge_gap_minutes
                and not burst_has_media
                and burst_ends_before_4am
            ):
                merged[-1] = (prev_start, period_end)
            else:
                merged.append((period_start, period_end))
        return merged

    def _media_played_between(self, start: datetime, end: datetime) -> bool:
        """Whether any media was playing at any point in [start, end]."""
        playing = False
        for event_dt, event_playing in self._media:
            if event_dt > end:
                break
            if event_dt <= start:
                playing = event_playing
                continue
            if playing:
                return True
            playing = event_playing
        return playing

    # -- persistence ------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise event history for the coordinator store."""
        return {
            "activity": [dt.isoformat() for dt in self._activity[-MAX_EVENTS:]],
            "media": [
                [dt.isoformat(), playing] for dt, playing in self._media[-MAX_EVENTS:]
            ],
            "quiet_minutes": self.quiet_minutes,
            "merge_gap_minutes": self.merge_gap_minutes,
        }

    @classmethod
    def from_dict(cls, data: dict | None, **kwargs) -> "QuietDetector":
        """Restore event history; tolerates missing/partial data."""
        detector = cls(**kwargs)
        if not data:
            return detector
        try:
            activity = data.get("activity")
            if isinstance(activity, list):
                detector._activity = [
                    datetime.fromisoformat(item)
                    for item in activity[-MAX_EVENTS:]
                    if isinstance(item, str)
                ]
            media = data.get("media")
            if isinstance(media, list):
                restored: list[tuple[datetime, bool]] = []
                for item in media[-MAX_EVENTS:]:
                    if isinstance(item, list) and len(item) == 2:
                        restored.append((datetime.fromisoformat(item[0]), bool(item[1])))
                detector._media = restored
            if isinstance(data.get("quiet_minutes"), (int, float)):
                detector.quiet_minutes = int(data["quiet_minutes"])
            if isinstance(data.get("merge_gap_minutes"), (int, float)):
                detector.merge_gap_minutes = int(data["merge_gap_minutes"])
        except (TypeError, ValueError, AttributeError) as err:
            _LOGGER.warning("QuietDetector: ignoring corrupt stored state: %s", err)
        return detector
