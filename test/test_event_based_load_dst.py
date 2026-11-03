"""Event loads keep aware event and sentinel times through the fall DST hour."""

import datetime as dt
import unittest

import pandas as pd

from ochre.Equipment.EventBasedLoad import EventBasedLoad


class EventBasedLoadDSTTests(unittest.TestCase):
    def test_event_rounding_and_exhausted_event_sentinel_remain_aware(self):
        start = pd.Timestamp("2025-11-02 00:00", tz="America/Indiana/Indianapolis")
        event_start = start + pd.Timedelta(minutes=90)
        events = pd.DataFrame({
            "start_time": [event_start],
            "end_time": [event_start + pd.Timedelta(minutes=1)],
            "power": [1.0],
        })
        load = EventBasedLoad(
            name="Test Load", start_time=start, duration=dt.timedelta(hours=3),
            time_res=dt.timedelta(minutes=1), event_schedule=events,
            save_results=False, verbosity=0,
        )
        for _ in range(180):
            load.update()
        self.assertEqual(load.current_time.isoformat(), "2025-11-02T02:00:00-05:00")
        self.assertIsNotNone(load.event_start.tzinfo)
        self.assertGreater(load.event_start, load.current_time)
        self.assertEqual(load.event_start, load.event_end)


if __name__ == "__main__":
    unittest.main()
