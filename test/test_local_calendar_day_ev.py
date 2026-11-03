"""Legacy EV overnight events keep their local calendar dates across DST."""

import unittest
import datetime as dt

import pandas as pd

from examples.five_home_block.five_home_block.home_factory import (
    _event_timestamp,
    _next_local_calendar_day,
)


class LocalCalendarDayTests(unittest.TestCase):
    def test_fall_back_advances_to_next_local_midnight(self):
        start = pd.Timestamp("2025-11-02 00:00", tz="America/Indiana/Indianapolis")
        next_day = _next_local_calendar_day(start)
        self.assertEqual(next_day.isoformat(), "2025-11-03T00:00:00-05:00")
        self.assertEqual(next_day - start, pd.Timedelta(hours=25))

    def test_spring_forward_advances_to_next_local_midnight(self):
        start = pd.Timestamp("2026-03-08 00:00", tz="America/Indiana/Indianapolis")
        next_day = _next_local_calendar_day(start)
        self.assertEqual(next_day.isoformat(), "2026-03-09T00:00:00-04:00")
        self.assertEqual(next_day - start, pd.Timedelta(hours=23))

    def test_repeated_hour_uses_first_occurrence(self):
        start = pd.Timestamp("2025-11-01 01:30", tz="America/Indiana/Indianapolis")
        next_day = _next_local_calendar_day(start)
        self.assertEqual(next_day.isoformat(), "2025-11-02T01:30:00-04:00")

    def test_trip_event_uses_first_repeated_hour_and_rejects_missing_hour(self):
        fall = pd.Timestamp("2025-11-02 00:00", tz="America/Indiana/Indianapolis")
        self.assertEqual(
            _event_timestamp(fall, dt.time(1, 30)).isoformat(),
            "2025-11-02T01:30:00-04:00",
        )
        spring = pd.Timestamp("2026-03-08 00:00", tz="America/Indiana/Indianapolis")
        with self.assertRaisesRegex(ValueError, "2026-03-08 02:30"):
            _event_timestamp(spring, dt.time(2, 30))


if __name__ == "__main__":
    unittest.main()
