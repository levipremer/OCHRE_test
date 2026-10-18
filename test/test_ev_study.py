"""Study staging, paired replay, resume, economics, and explicit pending methods."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

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
        mobility = schedule([(10, 20, 1)])
        mobility = replace(mobility, end=mobility.end+pd.Timedelta(days=4))
        with patch.object(study, "build_ev_schedules", return_value={"01A": mobility}) as build:
            result = study.stage_study(self.spec_file, ROOT, output_directory=self.directory/"outputs")
        return result, build.call_count

    def fake_run(self, config, selected, *, mobility_by_home):
        self.assertEqual(selected, ["home_01"])
        self.assertEqual(mobility_by_home["home_01"]["01A"].trips.distance_km.sum(), 1)
        path = config.neighborhood.output_directory/"data"
        path.mkdir(parents=True)
        rows = pd.DataFrame({"case": ["all_baseline", "non_v2h_twin", "advanced"],
            "home_id": "home_01", "device": "whole_home", "cost_usd": [10, 9, 8],
            "reference_degradation_cost_usd": [2, 2.5, 3], "final_soc": [.4, .5, .6],
            "mean_solve_seconds": [0, .1, .2], "max_solve_seconds": [0, .2, .4]})
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
        self.assertTrue((path.parent/"figures/advanced_formulation_savings.png").exists())
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

    def test_staged_pilot_is_thirty_paired_jobs_with_five_objectives(self):
        spec = study.read_json(ROOT/"examples/single_home_hems/config/ev_study_pilot.json")
        study.validate_study(spec)
        self.assertEqual(len(spec["configurations"]), 5)
        self.assertEqual(len(spec["schedule_cases"]), 6)
        self.assertEqual(spec["warmup_hours"], 24)
        self.assertEqual(spec["cooldown_hours"], 24)
        self.assertTrue(all(c["ev_controls"]["degradation_method"] in study.METHODS for c in spec["configurations"]))
        self.assertEqual(spec["initial_socs"], [.4])
        self.assertEqual(spec["common_ev_controls"]["comfort_weight"], 1000)
        self.assertEqual(spec["common_ev_controls"]["persistence_weight"], 0)
        self.assertEqual([c["ev_controls"]["health_soc_weight"] for c in spec["configurations"]], [0, 0, 1000, 1000, 1000])
        self.assertEqual([c["ev_controls"]["formulation"] for c in spec["configurations"]], ["convex"]+["milp"]*4)

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
                    device="EV_01A", capacity_kwh=60))
        plot_study(self.directory, manifest, statuses, pd.DataFrame(metrics))
        for case in ("advanced", "non_v2h_twin"):
            path = self.directory/"figures"/f"paired_{case}_formulations.png"
            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 1000)

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
