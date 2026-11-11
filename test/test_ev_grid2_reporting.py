"""Grid 2 reports monetary bills separately from normalized demand penalties."""

import pandas as pd
import pytest

from examples.single_home_hems.run_whole_home_joint import _metrics
from examples.single_home_hems.single_home_hems.ev_home_hub_trace import objective_component_frames


def test_actual_bill_includes_independent_weekday_maxima_without_dt_over_n():
    index = pd.date_range("2025-12-05", periods=3 * 24 * 60, freq="min")
    power = pd.Series(1.0, index=index)
    power.loc["2025-12-05 01:00":"2025-12-05 01:29"] = 10.0
    power.loc["2025-12-05 17:00":"2025-12-05 17:29"] = 4.0
    power.loc["2025-12-06":"2025-12-07 23:59"] = 100.0
    frame = pd.DataFrame({"total_electric_power_kw": power,
                          "ev_power_kw": 0.0, "ev_soc_fraction": 0.5})
    prices = pd.Series(0.10, index=index)
    result = _metrics(frame, prices, 2.34 / 21,
                      anytime_demand_rate_usd_per_kw_day=4.47 / 21)
    assert result["on_peak_demand_charge_usd"] == pytest.approx(4 * 2.34 / 21)
    assert result["anytime_demand_charge_usd"] == pytest.approx(10 * 4.47 / 21)
    assert result["demand_charge_usd"] == pytest.approx((4 * 2.34 + 10 * 4.47) / 21)
    assert result["total_bill_usd"] == pytest.approx(
        result["electricity_cost_usd"] + result["demand_charge_usd"])


def test_demand_breakdown_is_diagnostic_and_not_double_counted_in_objective():
    issue = "2025-12-01 17:00:00"
    plans = [{"timestamp": issue, "objective": 5.3,
              "objective_terms": {"electricity": 5.0, "modeled_home_peak": 0.3},
              "grid2_demand_formulation": "weekday_daily_dt_over_n_v1",
              "daily_demand_scale": 0.25 / 32,
              "daily_demand_objective_terms": {"on_peak": 0.1, "anytime": 0.2},
              "scenarios": []}]
    aggregate, _ = objective_component_frames(plans, [{"timestamp": issue, "successful": True}])
    row = aggregate.iloc[0]
    assert row.daily_demand_dt_over_n == pytest.approx(0.25 / 32)
    assert row.daily_demand_on_peak_objective_cost == pytest.approx(0.1)
    assert row.daily_demand_anytime_objective_cost == pytest.approx(0.2)
    assert row.component_sum == pytest.approx(5.3)
    assert row.objective_balance_error == pytest.approx(0.0)


def test_reports_complete_scenario_costs_and_selected_components():
    issue = "2025-12-01 17:00:00"
    plan = {"timestamp": issue, "objective": 8.,
            "objective_terms": {"electricity": 5., "modeled_home_peak": 3.},
            "grid2_demand_formulation": "weekday_daily_dt_over_n_v1",
            "daily_demand_aggregation": "worst_complete_scenario_v1",
            "daily_demand_worst_scenario_index": 1,
            "daily_demand_cost_scenarios": [1., 3., 2.],
            "daily_demand_objective_terms": {"on_peak": 1., "anytime": 2.},
            "scenarios": [{}, {}, {}]}
    aggregate, paths = objective_component_frames([plan], [{"timestamp": issue, "successful": True}])
    assert aggregate.iloc[0].daily_demand_worst_scenario_index == 1
    assert aggregate.iloc[0].objective_balance_error == pytest.approx(0)
    assert paths.modeled_home_peak_path_cost.tolist() == [1., 3., 2.]
    assert paths.aggregation_note.str.contains("complete per-scenario demand costs", regex=False).all()
