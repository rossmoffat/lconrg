from lconrg import __version__
from lconrg.lconrg import fixed_costs_profile, fuel_costs_profile, present_value_factor


def test_version():
    assert __version__ == "0.1.0"


def test_present_value():
    """Should return a dict of input values"""
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


def test_fixed_costs():
    """Should return a dict of fixed costs"""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    fc = 10000
    result = fixed_costs_profile(load_factors, fc)
    expected = {2021: 10000, 2022: 10000}
    assert result == expected


def test_gas_cost():
    """Should return a dict of gas costs"""
    load_factors = {2020: 0, 2021: 0.6, 2022: 0.5}
    gas_prices = {2020: 25, 2021: 25, 2022: 25}
    ng_flow_hhv = 100
    result = fuel_costs_profile(gas_prices, load_factors, ng_flow_hhv)
    expected = {2020: 0, 2021: 13.14, 2022: 10.95}
    assert result == expected
