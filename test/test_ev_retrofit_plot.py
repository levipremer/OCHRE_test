"""Read-only retrofit figures preserve time, actual modes and frozen inputs."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.plot_ev_trajectories import CASES, actual_mode, build_figure, load_cases, main


class EVRetrofitPlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.run = self.root/"run"
        self.index = pd.date_range("2018-02-03", periods=4, freq="1min")
        for case, *_ in CASES:
            folder = self.run/"home_01"/case/"data"
            folder.mkdir(parents=True)
            pd.DataFrame({"EV_01A SOC (-)": [.4, .5, .5, .4],
                          "EV_01A End SOC (-)": [.5, .5, .4, .3],
                          "EV_01A Electric Power (kW)": [2, 0, -1, 0],
                          "EV_01A Parked": [True, True, True, False]},
                         index=self.index).to_csv(folder/"raw_results.csv")
            pd.DataFrame({"electricity_price_usd_per_kwh": [.1, .1, .3, .3]},
                         index=self.index).to_csv(folder/"controls.csv")

    def test_realized_modes_and_soc_endpoint(self):
        cases = load_cases(self.run, "home_01", "01A")
        data = cases["advanced"]
        self.assertEqual(data["mode"].tolist(), [1, 0, -1, -2])
        self.assertEqual(data["soc"].index[-1], self.index[-1]+pd.Timedelta(minutes=1))
        self.assertEqual(data["soc"].tolist(), [.4, .5, .5, .4, .3])
        fig = build_figure(cases, price_alpha=.2)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes), 3)  # Two panels plus price twin axis.
        self.assertEqual(len(fig.axes[0].lines), 3)
        self.assertEqual(len(fig.axes[1].lines), 3)
        self.assertEqual(fig.axes[2].lines[0].get_alpha(), .2)

    def test_two_case_run_without_twin(self):
        twin = self.run/"home_01/non_v2h_twin"
        twin.rename(self.root/"unused_twin_fixture")
        cases = load_cases(self.run, "home_01", "01A")
        self.assertEqual(set(cases), {"all_baseline", "advanced"})
        fig = build_figure(cases)
        self.addCleanup(plt.close, fig)
        self.assertEqual(len(fig.axes[0].lines), 2)

    def test_price_or_timestamp_mismatch_rejected(self):
        file = self.run/"home_01/all_baseline/data/controls.csv"
        pd.DataFrame({"electricity_price_usd_per_kwh": [1, 1, 1, 1]}, index=self.index).to_csv(file)
        with self.assertRaisesRegex(ValueError, "different tariffs"):
            load_cases(self.run, "home_01", "01A")
        pd.DataFrame({"electricity_price_usd_per_kwh": [.1, .1, .3, .3]},
                     index=self.index+pd.Timedelta(minutes=1)).to_csv(file)
        with self.assertRaisesRegex(ValueError, "timestamps do not match"):
            load_cases(self.run, "home_01", "01A")

    def test_false_string_is_away_not_truthy(self):
        raw = pd.DataFrame({"EV_01A Electric Power (kW)": [0, 0], "EV_01A Parked": ["False", "True"]})
        self.assertEqual(actual_mode(raw, "EV_01A").tolist(), [-2, 0])
        raw.loc[0, "EV_01A Parked"] = np.nan
        with self.assertRaisesRegex(ValueError, "availability"):
            actual_mode(raw, "EV_01A")

    def test_cli_preserves_inputs_and_refuses_overwrite(self):
        originals = {p: p.read_bytes() for p in self.run.rglob("*.csv")}
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.run)]), 0)
        target = self.root/"retrofit_plots/run/home_01_EV_01A_clean.png"
        self.assertGreater(target.stat().st_size, 1000)
        self.assertTrue(target.with_suffix(".svg").exists())
        for path, content in originals.items():
            self.assertEqual(path.read_bytes(), content)
        with self.assertRaises(SystemExit):
            main([str(self.run)])
        with self.assertRaises(SystemExit):
            main([str(self.run), "--output-directory", str(self.run/"figures")])

    def test_crop_and_opacity_validation(self):
        cases = load_cases(self.run, "home_01", "01A")
        with self.assertRaisesRegex(ValueError, "within"):
            build_figure(cases, start="2018-01-01")
        with self.assertRaisesRegex(ValueError, "price_alpha"):
            build_figure(cases, price_alpha=2)


if __name__ == "__main__":
    unittest.main()
