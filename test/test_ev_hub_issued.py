"""An EV decision can use only an already issued Home Hub snapshot."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from examples.single_home_hems.single_home_hems.ev_hub_issued import IssuedEVHubForecasts


class IssuedEVHubForecastTests(unittest.TestCase):
    def test_selects_latest_available_issue_and_requires_complete_horizon(self):
        decision = pd.Timestamp("2026-09-22 03:00")
        rows = []
        for issued, mean in ((decision - pd.Timedelta(minutes=15), 1.0),
                             (decision + pd.Timedelta(minutes=15), 2.0)):
            for target in pd.date_range(decision, periods=2, freq="15min"):
                rows.append({
                    "issued_at": issued,
                    "inputs_ready_at": issued,
                    "interval_start": target,
                    "source": "test model",
                    "eligible_load_mean_kw": mean,
                    "eligible_mean_kw": mean,
                    "eligible_min_kw": mean,
                    "hub_mean_kw": mean,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "issued.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            provider = IssuedEVHubForecasts(path)
            history = pd.DataFrame(index=[decision - pd.Timedelta(minutes=1)])
            issued = provider.issue(decision, history, pd.Timedelta(minutes=30), 9.6)
            self.assertEqual(issued.issued_at, decision - pd.Timedelta(minutes=15))
            self.assertEqual(issued.intervals.loc[decision, "hub_mean_kw"], 1.0)
            # The interval ending exactly at the decision boundary is complete.
            provider.issue(decision, pd.DataFrame(index=[decision]), pd.Timedelta(minutes=30), 9.6)
            with self.assertRaisesRegex(ValueError, "uncompleted"):
                provider.issue(decision, pd.DataFrame(index=[decision + pd.Timedelta(minutes=1)]),
                               pd.Timedelta(minutes=30), 9.6)
            with self.assertRaisesRegex(ValueError, "complete 15-minute horizon"):
                provider.issue(decision, history, pd.Timedelta(minutes=45), 9.6)


if __name__ == "__main__":
    unittest.main()
