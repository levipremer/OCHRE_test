"""The EV-only runner aligns source plant replay and requires issued forecasts."""

from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from examples.single_home_hems.run_ev_home_hub import (
    HORIZON, _peak_objective_metadata, _validated_args, build_argument_parser, map_replay_to_simulation,
    validate_hub_issuance,
)
from examples.single_home_hems.run_whole_home_joint import (
    build_argument_parser as whole_home_argument_parser,
)
from examples.single_home_hems.single_home_hems.ev_hub_issued import IssuedEVHubForecasts
from examples.single_home_hems.single_home_hems.home_hub_replay import PreparedHomeHubReplay
from examples.single_home_hems.single_home_hems.rates import (
    WINTER_TOU_DEMAND_CPP, WinterTOURate,
)


class EVHomeHubRunnerTests(unittest.TestCase):
    def test_sparse_solver_backend_is_explicitly_selectable(self):
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).solver_backend, "cvxpy")
        args = parser.parse_args(["--solver-backend", "sparse_native_v1"])
        self.assertEqual(args.solver_backend, "sparse_native_v1")

    def test_twenty_hour_time_cap_is_opt_in_and_explicit_cap_wins(self):
        self.assertEqual(_validated_args(["--validate-only"]).time_limit_seconds, 600)
        self.assertEqual(_validated_args([
            "--validate-only", "--horizon-hours", "20",
        ]).time_limit_seconds, 180)
        self.assertEqual(_validated_args([
            "--validate-only", "--horizon-hours", "20",
            "--time-limit-seconds", "240",
        ]).time_limit_seconds, 240)

    def test_outside_forecast_defaults_to_approved_resistance_inclusive_rf(self):
        parser = build_argument_parser()
        args = parser.parse_args([])
        self.assertEqual(args.outside_forecast_mode, "rf_multires_weather_persistence")
        self.assertEqual(args.outside_rf_artifact.name, "outside_hub_er_multires_rf.joblib")
        self.assertEqual(args.outside_rf_artifact.parent.name, "models")
        explicit = parser.parse_args(["--outside-forecast-mode", "legacy_20d"])
        self.assertEqual(explicit.outside_forecast_mode, "legacy_20d")

    def test_predictive_cli_defaults_use_limited_persistence(self):
        for parser in (build_argument_parser(), whole_home_argument_parser()):
            args = parser.parse_args([])
            self.assertEqual(args.ev_scenario_method, "tod_dow_state_hier_limit_persistence")
            self.assertEqual(args.ev_scenario_count, 7)
            explicit = parser.parse_args(["--ev-scenario-method", "tod_dow_state_hier"])
            self.assertEqual(explicit.ev_scenario_method, "tod_dow_state_hier")

    def test_grid2_demand_cli_keeps_daily_default_and_accepts_legacy(self):
        parser = build_argument_parser()
        self.assertEqual(parser.parse_args([]).grid2_demand_formulation,
                         "weekday_daily_dt_over_n_v1")
        args = parser.parse_args(["--grid2-demand-formulation", "legacy_horizon_peak"])
        self.assertEqual(args.grid2_demand_formulation, "legacy_horizon_peak")

    def test_grid2_metadata_distinguishes_legacy_objective_from_current_billing(self):
        schedule = WinterTOURate(WINTER_TOU_DEMAND_CPP, cpp_event_date=date(2025, 12, 5))
        controller = SimpleNamespace(
            rate_structure=WINTER_TOU_DEMAND_CPP, uses_daily_grid2_demand=False,
            peak_limit_kw=None, peak_weight_usd_per_kw=0.71, horizon=HORIZON,
        )
        legacy = _peak_objective_metadata(controller, schedule)
        self.assertIn("0.71*maximum", legacy["peak_formula"])
        self.assertIn("no normalization or historical peak floor", legacy["peak_formula"])
        demand = legacy["grid2_demand_objective"]
        self.assertEqual(demand["on_peak_usd_per_kw"], 0.71)
        self.assertEqual(demand["anytime_usd_per_kw"], 0)
        self.assertEqual(demand["objective_scale"], 1)
        self.assertIsNone(demand["dt_over_n"])
        self.assertEqual(demand["already_observed_peaks"], "not included")
        self.assertIn("current Scenario 2 daily on-peak and anytime tariff", demand["billing_tariff"])

        controller.uses_daily_grid2_demand = True
        daily = _peak_objective_metadata(controller, schedule)
        self.assertIn("includes observed earlier-today peaks", daily["peak_formula"])
        demand = daily["grid2_demand_objective"]
        self.assertAlmostEqual(demand["on_peak_usd_per_kw_day"], 2.34 / 21)
        self.assertAlmostEqual(demand["anytime_usd_per_kw_day"], 4.47 / 21)
        self.assertEqual(demand["normalization_steps"], 32)
        self.assertAlmostEqual(demand["dt_over_n"], .25 / 32)
        controller.horizon = pd.Timedelta(hours=168)
        full = _peak_objective_metadata(controller, schedule)["grid2_demand_objective"]
        self.assertEqual(full["normalization_steps"], 672)
        self.assertAlmostEqual(full["dt_over_n"], .25 / 672)

    def test_source_split_maps_to_matching_simulation_clock(self):
        source_start = pd.Timestamp("2026-09-22 03:00")
        source_times = pd.date_range(source_start - pd.Timedelta(minutes=2), periods=3, freq="min")
        split = pd.DataFrame({
            "hub_other_power_kw": [1.0, 1.2, 1.4],
            "outside_hub_power_kw": [2.0, 2.2, 2.4],
        }, index=source_times)
        prepared = PreparedHomeHubReplay(split, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        sim_start = pd.Timestamp("2018-01-01 03:00")
        mapped = map_replay_to_simulation(prepared, sim_start, source_start)
        self.assertEqual(mapped.index[0], sim_start - pd.Timedelta(minutes=2))
        self.assertEqual(mapped.index[-1], sim_start)
        pd.testing.assert_frame_equal(mapped.reset_index(drop=True), split.reset_index(drop=True))

    def test_run_requires_explicit_hub_forecast_file(self):
        with self.assertRaisesRegex(ValueError, "hub-issued-forecast-csv is required"):
            _validated_args(["--duration-hours", "0.25"])
        args = _validated_args(["--validate-only", "--duration-hours", "0.25"])
        self.assertIsNone(args.hub_issued_forecast_csv)

    def test_preflight_requires_each_complete_already_issued_horizon(self):
        start = pd.Timestamp("2018-01-01 03:00")
        rows = []
        for issue in (start, start + pd.Timedelta(minutes=15)):
            for interval in pd.date_range(issue, periods=32, freq="15min"):
                rows.append({
                    "issued_at": issue,
                    "inputs_ready_at": issue,
                    "interval_start": interval,
                    "source": "test issued model",
                    "eligible_load_mean_kw": 2.0,
                    "eligible_mean_kw": 2.0,
                    "eligible_min_kw": 1.0,
                    "hub_mean_kw": 2.5,
                })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hub_issued.csv"
            pd.DataFrame(rows).to_csv(path, index=False)
            provider = IssuedEVHubForecasts(path)
            self.assertEqual(validate_hub_issuance(
                provider, start, pd.Timedelta(minutes=30), 9.6,
            ), 2)
            self.assertEqual(provider.history, [])
            with self.assertRaisesRegex(ValueError, "complete 15-minute horizon"):
                validate_hub_issuance(
                    provider, start, pd.Timedelta(minutes=45), 9.6,
                )
            self.assertEqual(HORIZON, pd.Timedelta(hours=8))


if __name__ == "__main__":
    unittest.main()
