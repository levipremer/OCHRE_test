import datetime as dt
import unittest

from ochre.Equipment.HVAC import HeatPumpHeater


def make_discrete_state(control_type, time_res_minutes=1):
    """Create only the state needed to exercise the timestep-independent controller."""
    heater = HeatPumpHeater.__new__(HeatPumpHeater)
    heater.defrost_control_type = control_type
    heater.defrost_timer_interval = dt.timedelta(minutes=90)
    heater.defrost_duration = dt.timedelta(minutes=7)
    heater.defrost_temp_threshold = 4.4445
    heater.defrost_initial_accumulation = 0
    heater.time_res = dt.timedelta(minutes=time_res_minutes)
    heater.current_schedule = {'Ambient Dry Bulb (C)': 0}
    heater.defrost_timer = dt.timedelta()
    heater.defrost_accumulation = dt.timedelta()
    heater.defrost_remaining = dt.timedelta()
    heater.defrost_active_fraction = 0
    heater.defrost_cycle_count = 0
    heater.defrost_hp_capacity = 0
    heater.defrost_hp_power = 0
    heater._reset_defrost_state()
    return heater


class DiscreteDefrostStateTestCase(unittest.TestCase):
    def test_timer_runs_90_minutes_then_defrosts_for_7(self):
        heater = make_discrete_state('Timer')

        normal_fractions = [heater._advance_discrete_defrost(True, 0.1) for _ in range(90)]
        active_fractions = [heater._advance_discrete_defrost(True, 0.1) for _ in range(7)]

        self.assertEqual(normal_fractions, [0] * 90)
        self.assertEqual(active_fractions, [1] * 7)
        self.assertEqual(heater.defrost_cycle_count, 1)
        self.assertEqual(heater.defrost_remaining, dt.timedelta())

    def test_demand_accumulator_preserves_expected_fraction(self):
        heater = make_discrete_state('Demand')
        legacy_fraction = 0.1

        # f/(1-f) accumulation requires 63 normal minutes to earn a
        # seven-minute event when f=0.1.
        normal_fractions = [
            heater._advance_discrete_defrost(True, legacy_fraction) for _ in range(63)
        ]
        active_fractions = [
            heater._advance_discrete_defrost(True, legacy_fraction) for _ in range(7)
        ]

        self.assertEqual(normal_fractions, [0] * 63)
        self.assertEqual(active_fractions, [1] * 7)
        self.assertEqual(heater.defrost_cycle_count, 1)

    def test_timer_pauses_when_compressor_is_off_and_resets_when_warm(self):
        heater = make_discrete_state('Timer')
        for _ in range(30):
            heater._advance_discrete_defrost(True, 0.1)

        heater._advance_discrete_defrost(False, 0.1)
        self.assertEqual(heater.defrost_timer, dt.timedelta(minutes=30))

        heater.current_schedule['Ambient Dry Bulb (C)'] = 5
        heater._advance_discrete_defrost(False, 0.1)
        self.assertEqual(heater.defrost_timer, dt.timedelta())

    def test_partial_timestep_is_time_weighted(self):
        heater = make_discrete_state('Timer', time_res_minutes=5)
        heater.defrost_timer = dt.timedelta(minutes=88)

        active_fraction = heater._advance_discrete_defrost(True, 0.1)

        self.assertAlmostEqual(active_fraction, 3 / 5)
        self.assertEqual(heater.defrost_remaining, dt.timedelta(minutes=4))


if __name__ == '__main__':
    unittest.main()
