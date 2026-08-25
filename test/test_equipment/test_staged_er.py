import datetime as dt
import unittest
from types import SimpleNamespace

from ochre.Equipment.HVAC import ASHPHeater
from ochre.utils import OCHREException


def make_stage_controller(stage_capacities=(10000, 10000, 5000)):
    """Create only the ASHP state needed to test staged ER control."""
    heater = ASHPHeater.__new__(ASHPHeater)
    heater.use_ideal_capacity = False
    heater.use_staged_er_control = True
    heater.er_capacity_list = [0]
    total = 0
    for capacity in stage_capacities:
        total += capacity
        heater.er_capacity_list.append(total)
    heater.er_stage_idx = 0
    heater.er_stage_timer = dt.timedelta()
    heater.er_stage_delay = dt.timedelta(minutes=10)
    heater.er_lockout_temp = 4.44
    heater.er_lockout_time = dt.timedelta()
    heater.er_hard_lockout_time = dt.timedelta()
    heater.er_soft_lockout_time = dt.timedelta()
    heater.er_setpoint_offset = 1.5
    heater.er_deadband_temp = 1.5
    heater.temp_setpoint = 20
    heater.prev_setpoint = 20
    heater.temp_indoor_prev = 18
    heater.time_res = dt.timedelta(minutes=1)
    heater.zone = SimpleNamespace(temperature=18)
    heater.current_schedule = {'Ambient Dry Bulb (C)': 0}
    heater.er_capacity_rated = heater.er_capacity_list[-1]
    heater.er_defrost_capacity_list = heater.er_capacity_list.copy()
    heater.er_ext_capacity = None
    heater.er_ext_capacity_frac = 1
    heater.defrost_model = 'Discrete'
    heater.defrost_remaining = dt.timedelta()
    heater.defrost = False
    heater.defrost_er_strategy = 'Aggressive'
    return heater


class ERStageSizingTestCase(unittest.TestCase):
    def configure(
        self,
        capacity,
        number=None,
        stages=None,
        ideal=False,
        staged_control=None,
    ):
        heater = ASHPHeater.__new__(ASHPHeater)
        heater.use_ideal_capacity = ideal
        heater.use_staged_er_control = not ideal if staged_control is None else staged_control
        return heater.configure_er_stages(capacity, number, stages)

    def test_default_stage_combinations_and_rounding(self):
        expected = {
            5000: [0, 5000],
            10000: [0, 5000, 10000],
            15000: [0, 10000, 15000],
            20000: [0, 10000, 20000],
            26300: [0, 10000, 20000, 25000],
            31300: [0, 10000, 20000, 30000],
            38300: [0, 10000, 20000, 30000, 40000],
        }
        for capacity, stages in expected.items():
            with self.subTest(capacity=capacity):
                self.assertEqual(self.configure(capacity), stages)

    def test_stage_capacity_override_sets_total_and_validates_count(self):
        stages = self.configure(26000, number=3, stages=[10000, 5000, 5000])
        self.assertEqual(stages, [0, 10000, 15000, 20000])

        with self.assertRaisesRegex(OCHREException, 'must match'):
            self.configure(26000, number=2, stages=[10000, 5000, 5000])

    def test_stage_count_override_uses_valid_five_and_ten_kw_elements(self):
        self.assertEqual(
            self.configure(25000, number=4),
            [0, 10000, 15000, 20000, 25000],
        )
        with self.assertRaisesRegex(OCHREException, 'requires between'):
            self.configure(25000, number=2)

    def test_ideal_capacity_preserves_unrounded_total_without_stage_inputs(self):
        self.assertEqual(self.configure(26300, ideal=True), [0, 26300])
        self.assertEqual(
            self.configure(26300, number=3, stages=[10000, 10000, 5000], ideal=True),
            [0, 10000, 20000, 25000],
        )

    def test_variable_speed_discrete_ideal_capacity_uses_automatic_stages(self):
        self.assertEqual(
            self.configure(26300, ideal=True, staged_control=True),
            [0, 10000, 20000, 25000],
        )


class ERStageThermostatTestCase(unittest.TestCase):
    def test_stages_escalate_after_each_unsuccessful_recovery_interval(self):
        heater = make_stage_controller()

        self.assertEqual(heater.run_er_thermostat_control(), 'On')
        self.assertEqual(heater.er_stage_idx, 1)
        self.assertEqual(heater.update_er_capacity(0), 10000)

        for _ in range(9):
            heater.run_er_thermostat_control()
        self.assertEqual(heater.er_stage_idx, 1)

        heater.run_er_thermostat_control()
        self.assertEqual(heater.er_stage_idx, 2)
        self.assertEqual(heater.update_er_capacity(0), 20000)

        for _ in range(10):
            heater.run_er_thermostat_control()
        self.assertEqual(heater.er_stage_idx, 3)
        self.assertEqual(heater.update_er_capacity(0), 25000)

    def test_recovery_deadband_holds_stages_and_resets_timer(self):
        heater = make_stage_controller()
        heater.er_stage_idx = 2
        heater.er_stage_timer = dt.timedelta(minutes=7)
        heater.zone.temperature = 19

        self.assertEqual(heater.run_er_thermostat_control(), 'On')
        self.assertEqual(heater.er_stage_idx, 2)
        self.assertEqual(heater.er_stage_timer, dt.timedelta())

    def test_upper_threshold_releases_all_stages(self):
        heater = make_stage_controller()
        heater.er_stage_idx = 3
        heater.zone.temperature = 20

        self.assertEqual(heater.run_er_thermostat_control(), 'Off')
        self.assertEqual(heater.er_stage_idx, 0)
        self.assertEqual(heater.update_er_capacity(0), 0)

    def test_ideal_heat_pump_modulates_around_active_er_stage(self):
        heater = make_stage_controller()
        heater.use_ideal_capacity = True
        heater.capacity_ideal = 18000
        heater.capacity_max = 16000
        heater.ext_capacity_frac = 1
        heater.er_stage_idx = 1
        heater.set_ideal_speed_idx = lambda capacity: setattr(
            heater, 'last_speed_capacity', capacity
        )

        self.assertEqual(heater.calculate_staged_ideal_hp_capacity(), 8000)
        self.assertEqual(heater.last_speed_capacity, 8000)

        heater.capacity_ideal = 30000
        self.assertEqual(heater.calculate_staged_ideal_hp_capacity(), 16000)

        heater.capacity_ideal = 6000
        self.assertEqual(heater.calculate_staged_ideal_hp_capacity(), 0)

    def test_defrost_strategy_selects_full_bank_or_first_stage(self):
        heater = make_stage_controller()
        heater.defrost = True

        self.assertEqual(heater.get_defrost_backup_capacity(), 25000)
        self.assertEqual(heater.get_defrost_er_commanded_stage(), 3)

        heater.defrost_er_strategy = 'Conservative'
        self.assertEqual(heater.get_defrost_backup_capacity(), 10000)
        self.assertEqual(heater.get_defrost_er_commanded_stage(), 1)

    def test_conservative_defrost_has_first_stage_for_coarse_ideal_er(self):
        heater = make_stage_controller()
        heater.use_staged_er_control = False
        heater.er_capacity_rated = 26300
        heater.er_capacity_list = [0, 26300]
        heater.er_defrost_capacity_list = [0, 10000, 20000, 25000]
        heater.defrost_er_strategy = 'Conservative'

        self.assertEqual(heater.get_defrost_backup_capacity(), 10000)
        heater.defrost_er_strategy = 'Aggressive'
        self.assertEqual(heater.get_defrost_backup_capacity(), 26300)

    def test_active_defrost_pauses_normal_stage_timer(self):
        heater = make_stage_controller()
        heater.er_stage_idx = 1
        heater.er_stage_timer = dt.timedelta(minutes=6)
        heater.defrost_remaining = dt.timedelta(minutes=4)

        self.assertEqual(heater.run_er_thermostat_control(), 'On')
        self.assertEqual(heater.er_stage_idx, 1)
        self.assertEqual(heater.er_stage_timer, dt.timedelta(minutes=6))


if __name__ == '__main__':
    unittest.main()
