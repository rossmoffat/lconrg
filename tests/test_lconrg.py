"""Set of tests for lconrg module."""
import datetime

import numpy as np
import pytest

from lconrg import __version__
from lconrg.lconrg import (
    Plant,
    present_value_factor,
)


def test_version():
    """Tests the version number is accurate."""
    assert __version__ == "0.2.0"


@pytest.fixture
def example_Plant_data():
    """Dummy data for Plant class."""
    return {
        "fuel": "gas",
        "hhv_eff": 0.55,
        "availability": 0.91,
        "cod_date": datetime.date(2022, 1, 1),
        "lifetime": 5,
        "net_capacity_mw": 700,
        "capital_cost": {
            datetime.date(2020, 1, 1): 100000,
            datetime.date(2021, 1, 1): 100000,
        },
        "cost_base_date": datetime.date(2022, 1, 1),
        "discount_rate": 0.1,
        "fuel_carbon_intensity": 0.185,
        "carbon_capture_rate": 0.95,
    }


@pytest.mark.parametrize(
    "opex",
    [
        40,
        {
            datetime.date(2022, 1, 1): 40,
            datetime.date(2023, 1, 1): 40,
            datetime.date(2024, 1, 1): 40,
            datetime.date(2025, 1, 1): 40,
            datetime.date(2026, 1, 1): 40,
        },
    ],
)
def test_Plant(example_Plant_data, opex):
    """Tests the plant class constructor."""
    test_plant_object = Plant(
        **example_Plant_data, fixed_opex_kgbp=opex, variable_opex_gbp_hr=opex
    )
    result = test_plant_object.net_capacity_mw
    expected = 700
    assert result == expected


def test_present_value():
    """Should return a dict of input values."""
    base_date = datetime.date(2020, 1, 1)
    discount_rate = 0.2
    no_of_years = 5

    pv_factors = present_value_factor(base_date, discount_rate, no_of_years)
    expected = (
        [
            np.arange(
                base_date,
                np.datetime64(base_date, "Y") + np.timedelta64(no_of_years, "Y"),
                dtype="datetime64[Y]",
            )
        ],
        [
            1,
            0.8333333333333334,
            0.6944444444444444,
            0.5787037037037038,
            0.4822530864197531,
        ],
    )
    assert (pv_factors[0] == expected[0]).all() | (pv_factors[1] == expected[1]).all()


@pytest.mark.parametrize(
    "opex",
    [
        40,
        {
            datetime.date(2022, 1, 1): 40,
            datetime.date(2023, 1, 1): 40,
            datetime.date(2024, 1, 1): 40,
            datetime.date(2025, 1, 1): 40,
            datetime.date(2026, 1, 1): 40,
        },
    ],
)
def test_fuel_costs_profile(example_Plant_data, opex):
    """Should return a Tuple of dates and gas costs."""
    test_plant_object = Plant(
        **example_Plant_data, fixed_opex_kgbp=opex, variable_opex_gbp_hr=opex
    )
    load_factors = 0.5
    gas_prices = 20
    result = test_plant_object.fuel_costs_profile(gas_prices, load_factors)
    expected = (
        np.arange(
            datetime.date(2022, 1, 1),
            np.datetime64(datetime.date(2022, 1, 1), "Y") + np.timedelta64(5, "Y"),
            dtype="datetime64[Y]",
        ),
        np.full(5, 111363.63636363635),
    )
    assert (result[0] == expected[0]).all() | (result[1] == expected[1]).all()
