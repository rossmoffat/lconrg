"""Set of tests for lconrg module."""
from lconrg import __version__
from lconrg.lconrg import (
    calculate_srmc,
    carbon_costs_profile,
    energy_production_profile,
    fixed_costs_profile,
    fuel_costs_profile,
    present_value_factor,
)


def test_version():
    """Tests the version number is accurate."""
    assert __version__ == "0.1.0"


def test_present_value():
    """Should return a dict of input values."""
    base_year = 2020
    discount_rate = 0.2
    no_of_years = 5

    pv_factors = present_value_factor(base_year, discount_rate, no_of_years)
    expected = {
        2020: 1,
        2021: 0.8333333333333334,
        2022: 0.6944444444444444,
        2023: 0.5787037037037038,
        2024: 0.4822530864197531,
    }
    assert pv_factors == expected


def test_fixed_costs_profile():
    """Should return a dict of fixed costs."""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    fc = 10000
    result = fixed_costs_profile(load_factors, fc)
    expected = {2021: 10000, 2022: 10000}
    assert result == expected


def test_fuel_costs_profile():
    """Should return a dict of gas costs."""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    gas_prices = {2020: 25, 2021: 25, 2022: 25}
    ng_flow_hhv = 100
    result = fuel_costs_profile(gas_prices, load_factors, ng_flow_hhv)
    expected = {2020: 0, 2021: 13.14, 2022: 10.95}
    assert result == expected


def test_carbon_costs_profile():
    """Should return a dict of carbon costs."""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    carbon_prices = {2020: 50, 2021: 50, 2022: 50}
    ng_flow_kgh = 12371
    carbon_capture_rate = 0.95
    carbon_fraction = 0.72284
    co2_transport_storage_cost = 15
    result = carbon_costs_profile(
        carbon_prices,
        load_factors,
        ng_flow_kgh,
        carbon_capture_rate,
        carbon_fraction,
        co2_transport_storage_cost,
    )
    expected = {2020: 0, 2021: 2.886639370451373, 2022: 2.4055328087094776}
    assert result == expected


def test_energy_production_profile():
    """Should return a dict of energy production by year."""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    energy_output = 100
    result = energy_production_profile(load_factors, energy_output)
    expected = {2020: 0, 2021: 525600, 2022: 438000}
    assert result == expected


def test_calculate_srmc():
    """Should return int of short run marginal cost."""
    load_factors = {2020: 0.0, 2021: 0.4, 2022: 0.5}
    energy_output = 100
    gas_prices = {2020: 25, 2021: 25, 2022: 25}
    fuel_flow_hhv = 178
    base_year = 2020
    discount_rate = 0.133
    vc = 1
    carbon_prices = {2020: 70, 2021: 50, 2022: 50}
    fuel_flow_kgh = 12371
    carbon_capture_rate = 0.95
    carbon_fraction = 0.72284
    co2_transport_storage_cost = 15
    result = calculate_srmc(
        gas_prices,
        load_factors,
        fuel_flow_hhv,
        carbon_prices,
        fuel_flow_kgh,
        carbon_capture_rate,
        carbon_fraction,
        co2_transport_storage_cost,
        vc,
        energy_output,
        discount_rate,
        base_year,
    )
    expected = 50.00208403814948
    assert result == expected
