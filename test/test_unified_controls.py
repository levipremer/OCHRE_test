"""Independent cycling, bounded solves, standalone templates and reporting."""
from dataclasses import replace
import itertools
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import cvxpy as cp
import numpy as np
import pandas as pd

from test.test_ev_v2h import ROOT, CONFIG, vehicle, schedule, forecast
import test.test_water_heater_scenario_optimizer as scenario_fixture
from examples.single_home_hems.single_home_hems.control_config import load_unified, resolve_control
from examples.single_home_hems.single_home_hems.controllers.ev import EVMPCController
from examples.single_home_hems.single_home_hems.controllers.solver import solve_ocp, usable_solution
from examples.single_home_hems.single_home_hems.ev_metrics import ev_operating_state, ev_metrics
from examples.single_home_hems.single_home_hems.unified import run_unified

CONFIG_DIR = ROOT/"examples/single_home_hems/config"


class IndependentCyclingTests(unittest.TestCase):
    def test_all_eight_ev_combinations_in_both_formulations(self):
        for kind, caps, dwell, persistence in itertools.product(("convex", "milp"), (False, True), (False, True), (0, 2)):
            with self.subTest(kind=kind, caps=caps, dwell=dwell, persistence=persistence):
                ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                    enforce_cycle_limit=caps, enforce_dwell_constraints=dwell,
                    persistence_weight=persistence, max_charge_starts_24h=1, max_discharge_starts_24h=1,
                    dwell_on_minutes=15, dwell_off_minutes=15)
                c, mobility = EVMPCController(ev), schedule()
                p, plan = c.build_problem(mobility.start, .5, forecast(ev, mobility))
                self.assertEqual(p.is_mixed_integer(), kind == "milp")
                solve_ocp(p, ev.hems.solver)
                self.assertTrue(usable_solution(p))
                if persistence == 0:
                    self.assertEqual(plan["persistence"].value, 0)
                else:
                    # EV stages scale by dt/N; convex persistence is chemical.
                    if kind == "convex":
                        plan["chemical"].value = np.ones(4)*3.6
                    else:
                        variables = {v.name(): v for v in p.variables()}
                        variables["b_c"].value = np.ones(4)
                        variables["b_d"].value = np.zeros(4)
                    # Idle -> charging changes two indicators, not 4 kW.
                    expected = persistence*.25/4*(3.6 if kind == "convex" else 2)
                    self.assertAlmostEqual(plan["persistence"].value, expected)

    def test_caps_are_milp_ocp_only_never_projected(self):
        for kind in ("milp", "convex"):
            ev = vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                enforce_cycle_limit=True, max_charge_starts_24h=0, max_discharge_starts_24h=0)
            c, m = EVMPCController(ev), schedule()
            p, plan = c.build_problem(m.start, .5, forecast(ev, m))
            solve_ocp(p, ev.hems.solver)
            self.assertTrue(usable_solution(p))
            if kind == "milp":
                self.assertLess(max(plan["discharge"].value), 1e-7)
            else:
                self.assertGreater(max(plan["discharge"].value), .1)
            self.assertEqual(c._project(m.start, -3, .5, True, 3)[:2], (-1, -3))

    def test_projector_retains_safety_but_not_dwell(self):
        for kind in ("milp", "convex"):
            c = EVMPCController(vehicle(formulation=kind, solver="SCIPY" if kind == "milp" else "CLARABEL",
                enforce_dwell_constraints=True, persistence_weight=1, dwell_on_minutes=30))
            now = schedule().start
            c.previous_mode, c.previous_power = 1, 4
            c.last_change[1] = now
            self.assertEqual(c._project(now+pd.Timedelta(minutes=1), 0, .5, True, 3)[0], 0)
            self.assertEqual(c._project(now+pd.Timedelta(minutes=1), 4, 1, True, 3), (0, 0, True))
            self.assertEqual(c._project(now+pd.Timedelta(minutes=1), 4, .5, False, 3), (0, 0, True))

    def test_all_eight_hpwh_combinations_and_projector(self):
        for kind, caps, dwell, persistence in itertools.product(("convex", "milp"), (False, True), (False, True), (0, 2)):
            with self.subTest(kind=kind, caps=caps, dwell=dwell, persistence=persistence):
                fixture = scenario_fixture.WaterHeaterScenarioOptimizerTests
                c = fixture.controller(kind, enforce_cycle_limit=caps, enforce_dwell_constraints=dwell,
                    persistence_weight=persistence, dwell_time_on="15min", dwell_time_off="15min")
                builder = c.optimization_problem_convex if kind == "convex" else c.optimization_problem_milp
                opts = {} if kind == "convex" else dict(prior_on=0, recent_cycles=0)
                p, heat, *_ = builder(np.full(2, 40), np.array([.2, .3]), np.zeros((1, 2)),
                    model=fixture.model(), previous_heat_kw=0, **opts)
                self.assertEqual(p.is_mixed_integer(), kind == "milp")
                solve_ocp(p, "SCIPY" if kind == "milp" else "CLARABEL")
                self.assertTrue(usable_solution(p))
                heat.value = np.array([2., 2.])
                self.assertAlmostEqual(c._objective_component_expressions["persistence_objective_cost"].value, persistence*.25)
                c._actuator_timestamp = schedule().start
                c._last_observed_on = 0
                c._last_observed_transition_time = c._actuator_timestamp
                c._last_setpoint = c.maximum_setpoint_c
                c.max_cycles = 0
                command = c._setpoint_control()["Water Heating"]
                if caps or dwell:
                    self.assertEqual(command["HP Duty Cycle"], 0)
                else:
                    self.assertNotIn("HP Duty Cycle", command)

    def test_hpwh_electricity_weight_scales_only_its_term(self):
        for kind in ("convex", "milp"):
            fixture = scenario_fixture.WaterHeaterScenarioOptimizerTests
            c = fixture.controller(kind, electricity_weight=3)
            opts = {} if kind == "convex" else dict(prior_on=0, recent_cycles=0)
            builder = c.optimization_problem_convex if kind == "convex" else c.optimization_problem_milp
            p, heat, *_ = builder(np.full(2, 40), [.2], np.zeros((1, 1)),
                model=fixture.model(), previous_heat_kw=0, **opts)
            heat.value = np.array([2.])
            self.assertAlmostEqual(c._objective_component_expressions["electricity_objective_cost"].value, .3)


class SolverPolicyTests(unittest.TestCase):
    def test_backend_limits_quiet_and_options_not_mutated(self):
        for backend in ("SCIPY", "CLARABEL", "GUROBI"):
            p = cp.Problem(cp.Minimize(cp.Constant(0)))
            opts = {"verbose": True}
            if backend == "SCIPY":
                opts["scipy_options"] = {"time_limit": 900, "disp": True}
            with patch.object(p, "solve") as solve:
                solve_ocp(p, backend, opts)
            sent = solve.call_args.kwargs
            self.assertFalse(sent["verbose"])
            self.assertTrue(opts["verbose"])
            if backend == "SCIPY":
                self.assertEqual(sent["scipy_options"], {"time_limit": 300, "disp": False})
                self.assertEqual(opts["scipy_options"]["time_limit"], 900)
            elif backend == "CLARABEL":
                self.assertEqual(sent["time_limit"], 300)
            else:
                self.assertEqual((sent["TimeLimit"], sent["OutputFlag"], sent["LogToConsole"]), (300, 0, 0))

    def test_user_limit_requires_feasible_finite_integral_incumbent(self):
        v = cp.Variable(boolean=True)
        p = cp.Problem(cp.Minimize(v), [v >= 0])
        p._status = cp.USER_LIMIT
        p.hems_diagnostic = {}
        for value, accepted in ((1, True), (.5, False), (-1, False), (np.nan, False)):
            v.save_value(np.array(value))
            self.assertEqual(usable_solution(p), accepted)
            self.assertEqual(p.hems_diagnostic["fallback"], not accepted)


class StandaloneConfigurationTests(unittest.TestCase):
    def test_examples_and_documented_templates_need_no_legacy_config(self):
        with patch("examples.five_home_block.five_home_block.config.load_configuration", side_effect=AssertionError("legacy lookup")):
            for suffix in (".json", ".template.json"):
                c = load_unified(CONFIG_DIR/f"unified_home{suffix}", CONFIG_DIR/f"unified_control{suffix}", ROOT)
                e = c.neighborhood.homes[0].evs[0]
                physical = json.loads((CONFIG_DIR/f"unified_home{suffix}").read_text())["homes"]["home_01"]["evs"]["01A"]
                self.assertEqual((e.capacity_kwh, e.charger_power_kw, e.initial_soc_fraction, e.hems.seed),
                    (physical["capacity_kwh"], physical["charger_power_kw"], physical["initial_soc_fraction"], physical["mobility"]["seed"]))
                original = json.loads((CONFIG_DIR/f"unified_control{suffix}").read_text())
                expected = resolve_control(original, "home_01", "hpwh")["parameters"]["persistence_weight"]
                self.assertEqual(c.controls["home_01"]["hpwh"]["parameters"]["persistence_weight"], expected)
                self.assertEqual(c.controls["home_01"]["evs"]["01A"]["persistence_weight"], 0)

    def test_missing_physics_reports_full_field_path(self):
        home = json.loads((CONFIG_DIR/"unified_home.json").read_text())
        control = (CONFIG_DIR/"unified_control.json").read_text()
        del home["homes"]["home_01"]["evs"]["01A"]["capacity_kwh"]
        with patch.object(Path, "read_text", side_effect=[json.dumps(home), control]):
            with self.assertRaisesRegex(ValueError, r"homes.home_01.evs.01A.*capacity_kwh.*no legacy"):
                load_unified("h.json", "c.json", ROOT)

    def test_per_ev_weights_do_not_modify_hpwh_or_other_ev(self):
        doc = {"defaults": {"ev": {"persistence_weight": 2}, "hpwh": {"parameters": {"persistence_weight": 7}}},
               "homes": {"h": {"evs": {"a": {"persistence_weight": 9}}}}}
        self.assertEqual(resolve_control(doc, "h", "ev", "a")["persistence_weight"], 9)
        self.assertEqual(resolve_control(doc, "h", "ev", "b")["persistence_weight"], 2)
        self.assertEqual(resolve_control(doc, "h", "hpwh")["parameters"]["persistence_weight"], 7)

    def test_run_refuses_existing_output(self):
        c = load_unified(CONFIG_DIR/"unified_home.json", CONFIG_DIR/"unified_control.json", ROOT)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp:
            (Path(temp)/"keep.txt").write_text("existing results")
            c.neighborhood = replace(c.neighborhood, output_directory=Path(temp))
            with patch("examples.single_home_hems.single_home_hems.unified.make_controllers"), self.assertRaises(FileExistsError):
                run_unified(c)
            self.assertEqual((Path(temp)/"keep.txt").read_text(), "existing results")

    def test_actual_states_and_cycle_counts_include_initial_start(self):
        ev = vehicle()
        name = f"EV_{ev.vehicle_id}"
        raw = pd.DataFrame({f"{name} Electric Power (kW)": [4, 4, 0, -2, 0],
            f"{name} Parked": [1, 1, 1, 1, 0], f"{name} SOC (-)": [.5]*5,
            f"{name} Unmet Load (kWh)": [0]*5, f"{name} Safety Override": [0]*5})
        self.assertEqual(ev_operating_state(raw, name).tolist(), ["charging", "charging", "idle", "discharging", "away"])
        row = ev_metrics(raw, replace(CONFIG.homes[0], evs=(ev,)),
                         pd.Series(.2, index=raw.index), 1/60).iloc[0]
        self.assertEqual((row.charge_starts, row.discharge_starts, row.cycle_count), (1, 1, 2))
