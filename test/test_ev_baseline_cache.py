"""Baseline reuse preserves trajectories, rescoring and input isolation."""
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule
from examples.single_home_hems.single_home_hems import baseline_cache as cache_module
from examples.single_home_hems.single_home_hems.baseline_cache import baseline_key, baseline_result, _cache_path, FRAMES
from examples.single_home_hems.single_home_hems.control_config import UnifiedConfiguration, resolve_control
from examples.single_home_hems.single_home_hems.simulation import SingleHomeHEMSSimulation
from examples.single_home_hems.single_home_hems.unified import run_home


class EVBaselineCacheTests(unittest.TestCase):
    def test_transient_publish_lock_is_retried(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            staging, entry = Path(temporary)/"staging", Path(temporary)/"entry"
            staging.mkdir()
            rename = Path.rename
            attempts = []

            def publish(source, target):
                attempts.append(source)
                if len(attempts) < 3:
                    raise PermissionError("fixture OneDrive sharing lock")
                return rename(source, target)

            with patch.object(Path, "rename", autospec=True, side_effect=publish), \
                 patch.object(cache_module.time, "sleep") as sleep:
                self.assertIsNone(cache_module._publish_cache(staging, entry))
            self.assertEqual(len(attempts), 3)
            self.assertEqual(sleep.call_count, 2)
            self.assertTrue(entry.exists())

    def test_persistent_publish_lock_keeps_valid_result_and_records_uncached(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            data = directory/"data"
            data.mkdir()
            frames = tuple(pd.DataFrame({"value": [i]}) for i in range(len(FRAMES)))
            sim = SimpleNamespace(controller=SimpleNamespace(device_controllers=[]), unified_context={},
                data_directory=data, raw_directory=data/"raw", _run_stepwise=lambda **kwargs: frames)
            events = []
            with patch.object(cache_module, "baseline_key", return_value="a"*64), \
                 patch.object(Path, "rename", side_effect=PermissionError("fixture access denied")) as rename, \
                 patch.object(cache_module.time, "sleep") as sleep, \
                 self.assertWarnsRegex(RuntimeWarning, "continuing|without caching"):
                result = baseline_result(sim, dict(directory=directory/"cache", scope="test"),
                                         progress=lambda **event: events.append(event))
            self.assertEqual(rename.call_count, 6)
            self.assertEqual(sleep.call_count, 5)
            self.assertIs(result, frames)
            record = json.loads((data/"baseline_reuse.json").read_text())
            self.assertFalse(record["reused"])
            self.assertFalse(record["published"])
            self.assertIn("access denied", record["publication_error"])
            self.assertTrue(events)
            self.assertFalse((directory/"cache"/("a"*64)).exists())

    @unittest.skipUnless(os.name == "nt", "Windows extended-path regression")
    def test_cache_publish_and_reuse_above_windows_path_limit(self):
        with tempfile.TemporaryDirectory(dir=_cache_path(ROOT)) as temporary:
            directory = Path(temporary)
            # Parent remains short enough to create normally; publishing the
            # full key pushes artifact paths beyond the legacy 260-char limit.
            parent = directory/("cache_" + "x"*max(0, 188-len(str(directory))))
            cache = dict(directory=parent, scope="long-path-test")
            frames = tuple(pd.DataFrame({"value": [i]}) for i in range(len(FRAMES)))
            sims = []
            for label in ("first", "second"):
                data = directory/label
                raw = data/"raw"
                raw.mkdir(parents=True)
                sims.append(SimpleNamespace(controller=SimpleNamespace(device_controllers=[]),
                    unified_context={}, data_directory=data, raw_directory=raw))

            def simulate(**kwargs):
                (sims[0].raw_directory/"home_01_Clothes Dryer_events.csv").write_text("event\n1\n")
                return frames

            key = "a"*64
            with patch("examples.single_home_hems.single_home_hems.baseline_cache.baseline_key", return_value=key), \
                 patch.object(sims[0], "_run_stepwise", create=True, side_effect=simulate) as first_run, \
                 patch.object(sims[1], "_run_stepwise", create=True) as second_run:
                baseline_result(sims[0], cache)
                result = baseline_result(sims[1], cache)
            self.assertEqual(first_run.call_count, 1)
            second_run.assert_not_called()
            for expected, actual in zip(frames, result):
                pd.testing.assert_frame_equal(expected, actual)
            artifact = parent/key/"artifacts/raw/home_01_Clothes Dryer_events.csv"
            self.assertGreater(len(str(artifact)), 260)
            self.assertEqual(_cache_path(artifact).read_text(), "event\n1\n")
            self.assertEqual((sims[1].raw_directory/artifact.name).read_text(), "event\n1\n")
            self.assertTrue(json.loads((sims[1].data_directory/"baseline_reuse.json").read_text())["reused"])

    def test_real_three_case_runs_reuse_only_baseline_and_rescore(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary, redirect_stdout(io.StringIO()):
            directory = Path(temporary)
            cache = dict(directory=directory/"cache", scope="test_inputs_and_code")
            mobility = schedule([(7, 12, .5)])
            results, sims, events = [], [], []
            original = SingleHomeHEMSSimulation._run_stepwise

            def execute(sim, **kwargs):
                sims.append(sim)
                return original(sim, **kwargs)

            with patch.object(SingleHomeHEMSSimulation, "_run_stepwise", autospec=True, side_effect=execute) as run:
                for method in ("none", "linear_throughput"):
                    ev = vehicle(formulation="milp", solver="SCIPY", horizon_hours=1,
                                 enable_degradation=method != "none", degradation_method=method)
                    home = replace(CONFIG.homes[0], evs=(ev,))
                    doc = {"defaults": {"ev": dict(controller="mpc", formulation="milp", solver="SCIPY", horizon_hours=1)}}
                    controls = {"evs": {ev.vehicle_id: resolve_control(doc, home.home_id, "ev", ev.vehicle_id)},
                        **{d: resolve_control(doc, home.home_id, d) for d in ("hpwh", "hvac")}}
                    cfg = replace(CONFIG, homes=(home,), duration=pd.Timedelta(minutes=16), output_directory=directory/method)
                    unified = UnifiedConfiguration(cfg, {home.home_id: controls}, {home.home_id: {
                        "backed_loads": ["heat_pump", "water_heater"], "water_heater_hp_only": True}}, {}, doc,
                        warmup_hours=0, cooldown_hours=0)
                    results.append(run_home(unified, home, {ev.vehicle_id: mobility}, baseline_cache=cache,
                                            progress=lambda **event: events.append(event)))
                self.assertEqual(run.call_count, 5)  # 1 baseline + 2 independent twins + 2 advanced.
            self.assertTrue(any(e["phase"] == "cached baseline reused" for e in events))
            finished = [e for e in events if e["phase"] == "simulating" and e["completed"] == e["total"]]
            self.assertEqual(len(finished), 5)
            self.assertEqual({e["case"] for e in finished}, {"all_baseline", "non_v2h_twin", "advanced"})
            pd.testing.assert_frame_equal(results[0].cases["all_baseline"]["raw"], results[1].cases["all_baseline"]["raw"])
            rows = [r.metrics.loc[r.metrics.case.eq("all_baseline") & r.metrics.device.str.startswith("EV_")].iloc[0] for r in results]
            self.assertTrue(pd.isna(rows[0].estimated_degradation_cost_usd))
            self.assertGreater(rows[1].estimated_degradation_cost_usd, 0)
            self.assertEqual(rows[0].reference_degradation_cost_usd, rows[1].reference_degradation_cost_usd)
            reused = directory/"linear_throughput"/home.home_id/"all_baseline/data/baseline_reuse.json"
            self.assertTrue(json.loads(reused.read_text())["reused"])
            self.assertEqual(len(list((directory/"cache").glob("*/manifest.json"))), 1)
            # A different physical state/comfort policy cannot share this baseline.
            sim = sims[0]
            old_home, old_config = sim.home, sim.config
            key = baseline_key(sim, cache["scope"])
            for change in (dict(initial_soh=.8), dict(comfort_soc_min=.6)):
                changed = replace(old_home.evs[0], hems=replace(old_home.evs[0].hems, **change))
                sim.home = replace(old_home, evs=(changed,))
                self.assertNotEqual(key, baseline_key(sim, cache["scope"]))
            sim.home = old_home
            self.assertNotEqual(key, baseline_key(sim, "different_source_version"))
            # Restore the cache-key runtime interval (run_home changes config for scoring).
            sim.config = replace(old_config, start_time=CONFIG.start_time, duration=pd.Timedelta(minutes=16))
            entry = next((directory/"cache").glob("*/manifest.json")).parent
            (entry/"raw.parquet").write_bytes(b"corrupted fixture")
            with self.assertRaisesRegex(ValueError, "Baseline cache changed"):
                baseline_result(sim, cache)


if __name__ == "__main__":
    unittest.main()
