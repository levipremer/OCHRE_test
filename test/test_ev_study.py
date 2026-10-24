"""Study staging, paired replay, resume, economics, and explicit pending methods."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd
import numpy as np

from test.test_ev_v2h import ROOT, schedule, vehicle, forecast
from examples.single_home_hems.single_home_hems import ev_study as study
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.unified import baseline_economics


class EVStudyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.spec = study.read_json(ROOT/"examples/single_home_hems/config/ev_study_pilot.json")
        self.spec["schedule_cases"] = self.spec["schedule_cases"][:1]
        self.spec["initial_socs"] = [.2, .8]
        self.spec["configurations"] = self.spec["configurations"][:2]
        self.spec_file = self.directory/"study.json"

    def stage(self):
        study.write_json(self.spec_file, self.spec)
        def make_mobility(config, home):
            mobility = schedule([(10, 20, 1)])
            trips = mobility.trips.copy()
            for col in ("departure", "arrival"):
                trips[col] += pd.Timestamp(config.start_time)-mobility.start
            return {"01A": replace(mobility, trips=trips,
                start=pd.Timestamp(config.start_time)-pd.Timedelta(days=home.evs[0].hems.history_days),
                end=pd.Timestamp(config.start_time)+config.duration+pd.Timedelta(hours=home.evs[0].hems.forecast_margin_hours),
                metadata=dict(history_days=home.evs[0].hems.history_days))}
        with patch.object(study, "build_ev_schedules", side_effect=make_mobility) as build:
            result = study.stage_study(self.spec_file, ROOT, output_directory=self.directory/"outputs")
        return result, build.call_count

    def test_optional_twin_staged_in_frozen_control(self):
        self.spec["run_non_v2h_twin"] = False
        manifest, _ = self.stage()
        paths = list((self.directory/"outputs/cases").glob("*/control.json"))
        paths += list((self.directory/"outputs/jobs").glob("*/control.json"))
        self.assertTrue(paths)
        for path in paths:
            self.assertIs(study.read_json(path)["run_non_v2h_twin"], False)
        with patch.object(study, "run_unified", side_effect=self.fake_run):
            study.run_study(manifest)
        metrics = pd.read_csv(manifest.parent/"study_metrics.csv")
        self.assertEqual(set(metrics.case), {"all_baseline", "advanced"})
        paired = pd.read_csv(manifest.parent/"paired_summary.csv")
        self.assertEqual(set(paired.case), {"all_baseline", "advanced"})
        for bad in (0, "false", None):
            self.spec["run_non_v2h_twin"] = bad
            with self.assertRaisesRegex(ValueError, "run_non_v2h_twin must be boolean"):
                study.validate_study(self.spec)

    def fake_run(self, config, selected, *, mobility_by_home, baseline_cache=None, progress=None):
        self.assertEqual(selected, ["home_01"])
        self.assertEqual(mobility_by_home["home_01"]["01A"].trips.distance_km.sum(), 1)
        if self.spec.get("reuse_all_baseline"):
            self.assertIsNotNone(baseline_cache)
            self.assertEqual(Path(baseline_cache["directory"]).name, "baseline_cache")
        path = config.neighborhood.output_directory/"data"
        path.mkdir(parents=True)
        rows = pd.DataFrame({"case": ["all_baseline", "non_v2h_twin", "advanced"],
            "home_id": "home_01", "device": "whole_home", "cost_usd": [10, 9, 8],
            "reference_degradation_cost_usd": [2, 2.5, 3], "final_soc": [.4, .5, .6],
            "mean_solve_seconds": [0, .1, .2], "max_solve_seconds": [0, .2, .4],
            "initial_stored_energy_kwh": [10, 10, 11], "final_stored_energy_kwh": [12, 13, 11],
            "initial_energy_value_usd_per_kwh": .2, "final_energy_value_usd_per_kwh": .2})
        if not config.run_non_v2h_twin:
            rows = rows.loc[rows.case.ne("non_v2h_twin")]
        baseline_economics(rows).to_csv(path/"performance.csv", index=False)

    def test_stage_freezes_schedules_across_socs_without_execution(self):
        with patch.object(study, "run_unified") as run:
            path, count = self.stage()
        self.assertEqual(count, 1)
        run.assert_not_called()
        manifest = study.read_json(path)
        self.assertEqual((len(manifest["cases"]), len(manifest["jobs"])), (2, 4))
        snapshots = [study.read_json(path.parent/"cases"/key/"mobility.json") for key in manifest["cases"]]
        self.assertEqual(snapshots[0], snapshots[1])
        _, repeated = self.stage()
        self.assertEqual(repeated, 0)

    def test_append_configuration_preserves_existing_jobs(self):
        path, _ = self.stage()
        before = study.read_json(path)
        extra = deepcopy(self.spec["configurations"][1])
        extra.update(id="health_new_weight")
        extra["ev_controls"]["health_soc_weight"] = 100
        self.spec["configurations"].append(extra)
        path, count = self.stage()
        after = study.read_json(path)
        self.assertEqual(count, 0)
        self.assertEqual(len(after["jobs"]), 6)
        for key, value in before["jobs"].items():
            self.assertEqual(value, after["jobs"][key])
        self.spec["configurations"][0]["ev_controls"]["health_soc_weight"] = 3
        with self.assertRaisesRegex(ValueError, "new ID"):
            self.stage()

    def test_pending_methods_block_before_any_simulation(self):
        self.spec["configurations"].append({"id": "smoothed", "ev_controls": {
            "enable_degradation": True, "degradation_method": "rainflow_smoothed"}})
        path, _ = self.stage()
        with patch.object(study, "run_unified") as run, self.assertRaisesRegex(NotImplementedError, "no runs started"):
            study.run_study(path)
        run.assert_not_called()

    def test_resume_and_output_integrity(self):
        path, _ = self.stage()
        with patch.object(study, "run_unified", side_effect=self.fake_run) as run:
            study.run_study(path)
            self.assertEqual(run.call_count, 4)
            study.run_study(path)
            self.assertEqual(run.call_count, 4)
        data = pd.read_csv(path.parent/"study_metrics.csv")
        self.assertEqual(len(data), 12)
        self.assertEqual(set(data.study_initial_soc), {.2, .8})
        self.assertTrue(data.loc[data.case.eq("all_baseline"), "net_savings_usd"].eq(0).all())
        summary = pd.read_csv(path.parent/"paired_summary.csv")
        self.assertEqual(len(summary), 6)
        self.assertTrue(summary.loc[summary.case.eq("advanced"), "net_savings_usd_mean"].eq(1).all())
        self.assertTrue(summary.loc[summary.case.eq("non_v2h_twin"), "net_savings_usd_mean"].eq(.5).all())
        self.assertTrue(summary.net_savings_usd_std.eq(0).all())
        self.assertTrue(np.allclose(summary.loc[summary.case.eq("advanced"), "energy_adjusted_net_savings_usd_mean"], .6))
        self.assertTrue(np.allclose(summary.loc[summary.case.eq("non_v2h_twin"), "energy_adjusted_net_savings_usd_mean"], .7))
        self.assertTrue((path.parent/"figures/advanced_formulation_savings.png").exists())
        wide = pd.read_csv(path.parent/"paired_metrics.csv")
        self.assertEqual(len(wide), 2)
        self.assertTrue(wide.paired_case_count.eq(2).all())
        self.assertTrue(wide.advanced_whole_home_net_savings_usd_mean.eq(1).all())
        self.assertTrue(wide.non_v2h_twin_whole_home_net_savings_usd_mean.eq(.5).all())
        job_id = next(iter(study.read_json(path)["jobs"]))
        status = study.read_json(path.parent/"jobs"/job_id/"status.json")
        metrics = path.parent/status["run_directory"]/"data/performance.csv"
        metrics.write_text("corrupted", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "output changed"):
            study.run_study(path)

    def test_failure_is_preserved_and_retry_uses_new_attempt(self):
        path, _ = self.stage()
        with patch.object(study, "run_unified", side_effect=RuntimeError("fixture failure")):
            with self.assertRaisesRegex(RuntimeError, "fixture failure"):
                study.run_study(path)
        failed = pd.read_csv(path.parent/"study_runs.csv")
        self.assertEqual((failed.state == "failed").sum(), 1)
        with patch.object(study, "run_unified", side_effect=self.fake_run):
            study.run_study(path)
        result = pd.read_csv(path.parent/"study_runs.csv")
        self.assertEqual(result.attempt.max(), 2)

    def test_forecast_metrics_propagate_without_pooling_control_cases(self):
        self.spec["initial_socs"] = [.4]
        path, _ = self.stage()

        def run_with_scores(config, selected, *, mobility_by_home):
            self.fake_run(config, selected, mobility_by_home=mobility_by_home)
            output = config.neighborhood.output_directory/"data/performance.csv"
            whole = pd.read_csv(output)
            ev = whole.assign(device="EV_01A", forecast_issuance_count=[np.nan, 2, 2],
                forecast_brier_home=[np.nan, .1, .3], forecast_no_scenarios_count=[np.nan, 0, 1])
            pd.concat([whole, ev], ignore_index=True).to_csv(output, index=False)

        with patch.object(study, "run_unified", side_effect=run_with_scores):
            study.run_study(path)
        summary = pd.read_csv(path.parent/"paired_summary.csv")
        ev = summary.loc[summary.device.eq("EV_01A")]
        self.assertTrue(ev.loc[ev.case.eq("advanced"), "forecast_brier_home_mean"].eq(.3).all())
        self.assertTrue(ev.loc[ev.case.eq("non_v2h_twin"), "forecast_brier_home_mean"].eq(.1).all())
        self.assertTrue(ev.loc[ev.case.eq("all_baseline"), "forecast_brier_home_mean"].isna().all())
        for case in ("advanced", "non_v2h_twin"):
            self.assertTrue((path.parent/"figures"/f"{case}_EV_01A_formulation_forecast_accuracy.png").exists())

    def test_frozen_inputs_unknown_ids_and_changed_environment(self):
        path, _ = self.stage()
        with self.assertRaisesRegex(ValueError, "selection"):
            study.run_study(path, configuration_ids=["typo"])
        with patch.object(study, "run_unified", side_effect=self.fake_run):
            study.run_study(path)
        with patch.object(study, "environment", return_value={"different": True}), self.assertRaisesRegex(ValueError, "environment changed"):
            study.run_study(path)
        manifest = study.read_json(path)
        case_id = next(iter(manifest["cases"]))
        study.write_json(path.parent/"cases"/case_id/"mobility.json", {})
        with self.assertRaisesRegex(ValueError, "Frozen case changed"):
            study.run_study(path)

    def test_smoke_is_bounded_and_does_not_change_pilot(self):
        study.write_json(self.spec_file, self.spec)
        with patch.object(study, "build_ev_schedules", return_value={"01A": schedule()}) as build:
            path = study.stage_study(self.spec_file, ROOT, output_directory=self.directory/"smoke", smoke=True)
        runtime = build.call_args.args[0]
        self.assertEqual(runtime.start_time, pd.Timestamp(self.spec["schedule_cases"][0]["start_time"])-pd.Timedelta(minutes=15))
        self.assertEqual(runtime.duration, pd.Timedelta(minutes=60))
        manifest = study.read_json(path)
        self.assertTrue(manifest["smoke"])
        self.assertEqual((len(manifest["cases"]), len(manifest["jobs"])), (1, 2))
        case_id = next(iter(manifest["cases"]))
        home = study.read_json(path.parent/"cases"/case_id/"home.json")
        self.assertEqual(home["simulation"]["duration_hours"], .5)
        self.assertEqual(home["simulation"]["warmup_hours"], .25)
        self.assertEqual(home["simulation"]["cooldown_hours"], .25)
        self.assertEqual(study.read_json(self.spec_file)["duration_hours"], 120)

    def test_pilot_is_forty_eight_paired_jobs_with_eight_objectives(self):
        spec = study.read_json(ROOT/"examples/single_home_hems/config/ev_study_pilot.json")
        study.validate_study(spec)
        self.assertEqual(len(spec["configurations"]), 8)
        self.assertEqual(len(spec["schedule_cases"]), 6)
        self.assertEqual(spec["warmup_hours"], 24)
        self.assertEqual(spec["cooldown_hours"], 24)
        self.assertTrue(all(c["ev_controls"]["degradation_method"] in study.METHODS for c in spec["configurations"]))
        self.assertEqual(spec["initial_socs"], [.4])
        self.assertEqual(spec["common_ev_controls"]["comfort_weight"], 1000)
        self.assertEqual(spec["common_ev_controls"]["persistence_weight"], 0)
        self.assertEqual([c["ev_controls"]["health_soc_weight"] for c in spec["configurations"]], [0, 0]+[1000]*6)
        self.assertEqual([c["ev_controls"]["formulation"] for c in spec["configurations"]], ["convex", "milp"]+["convex"]*3+["milp"]*3)
        self.assertTrue(spec["common_ev_controls"]["enforce_terminal_soc"])
        self.assertNotIn("terminal_soc_weight", spec["common_ev_controls"])
        if spec["common_ev_controls"]["solver"] == "SCIPY":
            self.assertEqual(spec["common_ev_controls"]["solver_options"], {})

    def test_identical_mobility_replayed_across_months(self):
        second = dict(self.spec["schedule_cases"][0], id="feb_pair", start_time="2018-02-03T00:00:00")
        self.spec["schedule_cases"].append(second)
        path, count = self.stage()
        self.assertEqual(count, 1)
        manifest = study.read_json(path)
        traces = [study.restore_schedule(study.read_json(path.parent/"cases"/key/"mobility.json")["01A"])
                  for key in manifest["cases"]]
        for trace in traces[1:]:
            pd.testing.assert_series_equal(trace.trips.departure-trace.start, traces[0].trips.departure-traces[0].start)
            pd.testing.assert_series_equal(trace.trips.arrival-trace.start, traces[0].trips.arrival-traces[0].start)
            pd.testing.assert_series_equal(trace.trips.distance_km, traces[0].trips.distance_km)
        _, repeated = self.stage()
        self.assertEqual(repeated, 0)

    def test_replay_keeps_history_before_simulation_anchor(self):
        original = schedule([(10, 20, 1)])
        original = replace(original, start=original.start-pd.Timedelta(days=30),
                           metadata=dict(history_days=30))
        target = pd.Timestamp("2018-02-01")
        shifted = study.restore_schedule(study.shift_schedule(study.frozen_schedule(original), target))
        self.assertEqual(shifted.start, target-pd.Timedelta(days=30))
        self.assertEqual(shifted.trips.departure.iloc[0], target+pd.Timedelta(minutes=10))
        self.assertEqual(shifted.metadata["simulation_start"], str(target))
        again = study.restore_schedule(study.shift_schedule(study.frozen_schedule(shifted), target))
        pd.testing.assert_frame_equal(shifted.trips, again.trips)

    def test_factorial_scenario_config_and_staging(self):
        source = study.read_json(ROOT/"examples/single_home_hems/config/ev_study_scenarios.json")
        axes = source["configuration_axes"]
        expected_count = int(np.prod([len(entries) for entries in axes.values()]))
        spec = study.expand_configurations(source)
        study.validate_study(spec)
        self.assertEqual(len(spec["configurations"]), expected_count)
        self.assertEqual(len({c["id"] for c in spec["configurations"]}), expected_count)
        self.assertEqual({c["factors"]["scenario_count"] for c in spec["configurations"]},
                         {a["value"] for a in axes["scenario_count"]})
        self.assertEqual({c["factors"]["scenario_selection"] for c in spec["configurations"]},
                         {a["value"] for a in axes["scenario_selection"]})
        self.spec = spec
        self.spec["schedule_cases"] = spec["schedule_cases"][:1]
        self.spec["configurations"] = spec["configurations"][-2:]
        with patch.object(study, "run_unified") as runner:
            path, count = self.stage()
        runner.assert_not_called()
        self.assertEqual(count, 1)
        manifest = study.read_json(path)
        self.assertTrue(manifest["staged_only"])
        self.assertTrue(manifest["reuse_all_baseline"])
        rows = pd.read_csv(path.parent/"paired_metrics.csv")
        self.assertEqual(len(rows), 2)
        self.assertTrue(rows.paired_case_count.eq(0).all())
        for key in manifest["jobs"]:
            control = study.read_json(path.parent/"jobs"/key/"control.json")["homes"]["home_01"]["evs"]["01A"]
            self.assertEqual(control["forecast"]["mode"], "scenario")
            self.assertEqual(control["forecast"]["lookback_days"], 28)
            self.assertEqual(control["forecast"]["prefix_control_intervals"], 4)
        ids = [j["configuration_id"] for j in manifest["jobs"].values()]
        with patch.object(study, "run_unified", side_effect=self.fake_run):
            study.run_study(path, configuration_ids=ids[:1])
            partial = pd.read_csv(path.parent/"paired_metrics.csv")
            self.assertTrue(partial.paired_case_count.eq(0).all())
            study.run_study(path)
        summary = pd.read_csv(path.parent/"paired_metrics.csv")
        self.assertEqual(len(summary), 2)
        self.assertTrue(summary.paired_case_count.eq(1).all())
        self.assertTrue(summary.advanced_whole_home_net_savings_usd_mean.eq(1).all())
        self.assertTrue(summary.non_v2h_twin_whole_home_net_savings_usd_mean.eq(.5).all())
        self.assertEqual(set(summary.scenario_selection),
                         {c["factors"]["scenario_selection"] for c in self.spec["configurations"]})

    def test_scenario_stage_rejects_missing_history(self):
        self.spec["common_ev_controls"]["forecast"] = dict(mode="scenario", lookback_days=28, prefix_control_intervals=4)
        study.write_json(self.spec_file, self.spec)
        with patch.object(study, "build_ev_schedules", return_value={"01A": schedule()}), self.assertRaisesRegex(ValueError, "history/tail"):
            study.stage_study(self.spec_file, ROOT, output_directory=self.directory/"bad_history")

    def test_horizon_bank_grid_stages_paired_inputs_without_running(self):
        self.spec = study.read_json(ROOT/"examples/single_home_hems/config/ev_study_horizons.json")
        self.spec["schedule_cases"] = self.spec["schedule_cases"][:1]
        with patch.object(study, "run_unified") as run:
            path, count = self.stage()
        run.assert_not_called()
        self.assertEqual(count, 1)
        manifest = study.read_json(path)
        self.assertEqual(len(manifest["jobs"]), 9)
        pairs = set()
        for key, job in manifest["jobs"].items():
            doc = study.read_json(path.parent/"jobs"/key/"control.json")
            c = doc["homes"]["home_01"]["evs"]["01A"]
            pairs.add((c["horizon_hours"], c["forecast"]["lookback_days"]))
            self.assertEqual(c["forecast_hours"], c["horizon_hours"])
            self.assertEqual(c["forecast"]["scenario_count"], 5)
            self.assertEqual(c["forecast"]["scenario_method"], "tod_dow_state_hier")
            self.assertEqual(c["forecast"]["prefix_control_intervals"], 4)
            self.assertEqual((c["solver"], c["formulation"], c["enable_degradation"]), ("GUROBI", "milp", False))
            trace = study.restore_schedule(study.read_json(path.parent/"cases"/job["case_id"]/"mobility.json")["01A"])
            start = pd.Timestamp(self.spec["schedule_cases"][0]["start_time"])-pd.Timedelta(hours=24)
            self.assertLessEqual(trace.start, start-pd.Timedelta(days=28))
            self.assertGreaterEqual(trace.end, start+pd.Timedelta(hours=168+48))
        self.assertEqual(pairs, {(h, b) for h in (6, 12, 24) for b in (14, 21, 28)})
        rows = pd.read_csv(path.parent/"paired_metrics.csv")
        self.assertEqual(len(rows), 9)
        self.assertTrue(rows.paired_case_count.eq(0).all())

    def test_smoke_overrides_factor_horizons(self):
        source = ROOT/"examples/single_home_hems/config/ev_study_horizons.json"
        trace = replace(schedule(), start=pd.Timestamp("2017-01-01"), end=pd.Timestamp("2020-01-01"))
        with patch.object(study, "build_ev_schedules", return_value={"01A": trace}):
            path = study.stage_study(source, ROOT, output_directory=self.directory/"smoke", smoke=True)
        for key in study.read_json(path)["jobs"]:
            c = study.read_json(path.parent/"jobs"/key/"control.json")["homes"]["home_01"]["evs"]["01A"]
            self.assertEqual((c["horizon_hours"], c["forecast_hours"]), (.5, .5))
            self.assertEqual(c["forecast"]["prefix_control_intervals"], 2)
            self.assertEqual(study.read_json(path)["jobs"][key]["factors"]["horizon_hours"], .5)

    def test_progress_reports_resume(self):
        path, _ = self.stage()
        with patch("builtins.print") as output, patch.object(study, "run_unified", side_effect=self.fake_run) as run:
            study.run_study(path, progress=True)
            self.assertTrue(all(call.kwargs.get("progress") for call in run.call_args_list))
            study.run_study(path, progress=True)
        messages = "\n".join(call.args[0] for call in output.call_args_list)
        self.assertIn("job 4/4", messages)
        self.assertIn("already complete; skipped", messages)
        self.assertIn("reports saved", messages)
        self.assertTrue(all(call.kwargs["flush"] for call in output.call_args_list))

    def test_progress_throttles_steps_but_not_phase_or_final(self):
        with patch.object(study.time, "perf_counter", return_value=0), patch("builtins.print") as output:
            reporter = study.StudyProgress(2)
            reporter.begin(1, "example")
            reporter(phase="simulating", completed=0, total=10, timestamp="start")
            reporter(phase="simulating", completed=1, total=10, timestamp="step1")
            self.assertEqual(output.call_count, 1)
            reporter(phase="simulating", completed=10, total=10, timestamp="end")
            reporter(phase="scoring")
            self.assertEqual(output.call_count, 3)
            self.assertIn("100.0%", output.call_args_list[1].args[0])

    def test_horizon_heatmaps_preserve_grid_and_control_cases(self):
        from matplotlib.axes import Axes
        from examples.single_home_hems.single_home_hems import ev_study_plots as plots
        rows = pd.DataFrame([dict(objective_variant="health", scenario_count=5,
            scenario_selection="tod_dow_state_hier", horizon_hours=h, lookback_days=b,
            study_case=s, case=c, device="whole_home", net_savings_usd=h+b+d)
            for h in (6, 12, 24) for b in (14, 21, 28)
            for s, d in (("jan", 0), ("feb", 2)) for c in ("advanced", "non_v2h_twin")])
        with patch.object(plots, "_horizon_scores") as draw:
            plots.plot_study(self.directory, {}, [], rows)
        self.assertEqual(draw.call_count, 2)
        for call in draw.call_args_list:
            self.assertEqual(call.args[0].case.nunique(), 1)
            self.assertEqual(len(call.args[0]), 18)
        selected = rows.loc[rows.case.eq("advanced")].copy()
        selected["max_solve_seconds"] = selected.net_savings_usd
        selected["failed_solves"] = selected.net_savings_usd
        selected["missing"] = np.nan
        imshow = Axes.imshow
        with patch.object(Axes, "imshow", autospec=True, side_effect=imshow) as meshes:
            plots._horizon_scores(selected, ["net_savings_usd", "max_solve_seconds", "failed_solves", "missing"],
                                  "Fixture", self.directory/"horizons.png")
        base = np.array([[h+b for h in (6, 12, 24)] for b in (14, 21, 28)])
        np.testing.assert_array_equal(meshes.call_args_list[0].args[1], base+1)
        np.testing.assert_array_equal(meshes.call_args_list[1].args[1], base+2)
        np.testing.assert_array_equal(meshes.call_args_list[2].args[1], 2*base+2)
        self.assertTrue(np.ma.getmaskarray(meshes.call_args_list[3].args[1]).all())
        self.assertGreater((self.directory/"horizons.png").stat().st_size, 1000)

    def test_parametric_window_padding_is_explicit_in_resolved_config(self):
        from examples.single_home_hems.single_home_hems.control_config import load_unified
        path, _ = self.stage()
        manifest = study.read_json(path)
        job_id, job = next(iter(manifest["jobs"].items()))
        config = load_unified(path.parent/"cases"/job["case_id"]/"home.json",
                              path.parent/"jobs"/job_id/"control.json", ROOT)
        self.assertEqual(config.neighborhood.duration, pd.Timedelta(hours=120))
        self.assertEqual(config.simulation_window().duration, pd.Timedelta(hours=168))
        self.assertEqual((config.warmup_hours, config.cooldown_hours), (24, 24))

    def test_cross_formulation_trajectory_plots_are_separate_by_case(self):
        from examples.single_home_hems.single_home_hems.ev_study_plots import plot_study
        manifest = {"cases": {"paired": {}}, "jobs": {}}
        statuses, metrics = [], []
        index = pd.date_range("2018-01-03", periods=3, freq="15min")
        for variant in ("convex", "milp"):
            manifest["jobs"][variant] = dict(case_id="paired", configuration_id=variant,
                                            home_id="home_01", vehicle_id="01A")
            statuses.append(dict(job_id=variant, state="complete", run_directory=variant))
            for case in ("all_baseline", "non_v2h_twin", "advanced"):
                folder = self.directory/variant/"home_01"/case/"data"
                folder.mkdir(parents=True)
                pd.DataFrame({"EV_01A Electric Power (kW)": [1, 0, -1], "EV_01A SOC (-)": [.4, .5, .5],
                    "EV_01A End SOC (-)": [.5, .5, .4], "EV_01A Parked": [1, 0, 1]}, index=index).to_csv(folder/"raw_results.csv")
                pd.DataFrame({"electricity_price_usd_per_kwh": [.1, .2, .3]}, index=index).to_csv(folder/"controls.csv")
                pd.DataFrame({"total_electric_power_kw": [3, 4, 2]}, index=index).to_csv(folder/"standardized.csv")
                metrics.append(dict(configuration_id=variant, study_case="paired", case=case,
                    device="EV_01A", capacity_kwh=60, forecast_issuance_count=3,
                    forecast_brier_home=.1 if case == "advanced" else .3,
                    forecast_arrival_energy_mae_kwh=.2, forecast_no_scenarios_fraction=0))
        plot_study(self.directory, manifest, statuses, pd.DataFrame(metrics))
        for case in ("advanced", "non_v2h_twin"):
            path = self.directory/"figures"/f"paired_{case}_formulations.png"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)
            self.assertTrue((self.directory/"figures"/f"{case}_EV_01A_formulation_forecast_accuracy.png").exists())

    def test_factorial_plots_average_only_situations(self):
        from matplotlib.axes import Axes
        from examples.single_home_hems.single_home_hems import ev_study_plots as plots
        rows = pd.DataFrame([
            dict(objective_variant=objective, scenario_count=count, scenario_selection=selector,
                 study_case=situation, case=case, device="whole_home",
                 net_savings_usd=offset+count+delta)
            for objective in ("health", "throughput")
            for case in ("advanced", "non_v2h_twin")
            for selector, offset in (("tod_only", 0), ("knn_state_6h", 100))
            for count in (3, 5)
            for situation, delta in (("jan", 0), ("feb", 2))])
        with patch.object(plots, "_factorial_scores") as draw:
            plots.plot_study(self.directory, {}, [], rows)
        self.assertEqual(draw.call_count, 4)
        for call in draw.call_args_list:
            data = call.args[0]
            self.assertEqual(data.objective_variant.nunique(), 1)
            self.assertEqual(data.case.nunique(), 1)
            self.assertEqual(data.scenario_selection.nunique(), 2)
            self.assertEqual(data.scenario_count.nunique(), 2)
            self.assertEqual(data.study_case.nunique(), 2)
        selected = rows.loc[rows.objective_variant.eq("health") & rows.case.eq("advanced")]
        errorbar = Axes.errorbar
        with patch.object(Axes, "errorbar", autospec=True, side_effect=errorbar) as lines:
            plots._factorial_scores(selected, ["net_savings_usd"], "Grouping test",
                                    self.directory/"factorial.png")
        self.assertEqual(lines.call_count, 2)
        for call in lines.call_args_list:
            offset = 100 if call.kwargs["label"] == "knn_state_6h" else 0
            np.testing.assert_array_equal(call.args[1], [3, 5])
            np.testing.assert_array_equal(call.args[2], [offset+4, offset+6])
            np.testing.assert_allclose(call.kwargs["yerr"], np.sqrt(2))
        self.assertGreater((self.directory/"factorial.png").stat().st_size, 1000)

    def test_none_and_multicut_alias_preserve_objectives(self):
        for method, expected in (("none", False), ("rainflow_multicut", True)):
            ev = vehicle(formulation="milp", solver="SCIPY", enable_degradation=True, degradation_method=method)
            controller = EVMPCController(ev)
            _, plan = controller.build_problem(schedule().start, .5, forecast(ev, schedule()))
            self.assertEqual(plan["rainflow_theta"] is not None, expected)

    def test_economics_nonbattery_missing_aging_and_negative_savings(self):
        rows = pd.DataFrame([dict(home_id="h", device="hpwh", case="all_baseline", cost_usd=2),
                             dict(home_id="h", device="hpwh", case="advanced", cost_usd=3)])
        result = baseline_economics(rows)
        self.assertEqual(result.net_savings_usd.tolist(), [0, -1])
        self.assertTrue(result.degradation_cost_savings_usd.eq(0).all())
        rows["device"] = "whole_home"
        with self.assertRaisesRegex(ValueError, "Reference degradation"):
            baseline_economics(rows)


if __name__ == "__main__":
    unittest.main()
