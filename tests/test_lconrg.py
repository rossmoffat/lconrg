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
    assert __version__ == "2023.10.0"


@pytest.fixture
def float_Plant_data():
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


@pytest.fixture
def dict_Plant_data():
    """Dummy data for Plant class."""
    return {
        "fuel": "gas",
        "hhv_eff": 0.55,
        "availability": {
            datetime.date(2022, 1, 1): 0.91,
            datetime.date(2023, 1, 1): 0.91,
            datetime.date(2024, 1, 1): 0.91,
            datetime.date(2025, 1, 1): 0.91,
            datetime.date(2026, 1, 1): 0.91,
        },
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
    "example_Plant_data",
    [
        pytest.param(
            "float_Plant_data",
            id="Plant floats",
        ),
        pytest.param(
            "dict_Plant_data",
            id="Plant dicts",
        ),
    ],
)
@pytest.mark.parametrize(
    "opex",
    [
        pytest.param(40.0, id="opex single float"),
        pytest.param(
            {
                datetime.date(2022, 1, 1): 40.0,
                datetime.date(2023, 1, 1): 40.0,
                datetime.date(2024, 1, 1): 40.0,
                datetime.date(2025, 1, 1): 40.0,
                datetime.date(2026, 1, 1): 40.0,
            },
            id="opex dict",
        ),
    ],
)
def test_Plant(example_Plant_data, opex, request):
    """Tests the plant class constructor."""
    example_Plant_data = request.getfixturevalue(example_Plant_data)
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
    "example_Plant_data",
    [
        pytest.param(
            "float_Plant_data",
            id="Plant floats",
        ),
        pytest.param(
            "dict_Plant_data",
            id="Plant dicts",
        ),
    ],
)
@pytest.mark.parametrize(
    "opex",
    [
        40.0,
        {
            datetime.date(2022, 1, 1): 40.0,
            datetime.date(2023, 1, 1): 40.0,
            datetime.date(2024, 1, 1): 40.0,
            datetime.date(2025, 1, 1): 40.0,
            datetime.date(2026, 1, 1): 40.0,
        },
    ],
)
def test_fuel_costs_profile(example_Plant_data, opex, request):
    """Should return a Tuple of dates and gas costs."""
    example_Plant_data = request.getfixturevalue(example_Plant_data)
    test_plant_object = Plant(
        **example_Plant_data, fixed_opex_kgbp=opex, variable_opex_gbp_hr=opex
    )
    load_factors = 0.5
    gas_prices = 20.0
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


"""Test __str__ for Plant class."""


def test_str():
    """Tests that the __str__ method returns the expected output."""
    plant = Plant(
        fuel="gas",
        hhv_eff=0.55,
        availability=0.91,
        cod_date=datetime.date(2022, 1, 1),
        lifetime=5,
        net_capacity_mw=700,
        capital_cost={
            datetime.date(2020, 1, 1): 100000,
            datetime.date(2021, 1, 1): 100000,
        },
        fixed_opex_kgbp=5000.0,
        variable_opex_gbp_hr=20.0,
        cost_base_date=datetime.date(2022, 1, 1),
        discount_rate=0.1,
        fuel_carbon_intensity=0.185,
        carbon_capture_rate=0.95,
    )
    result = str(plant)
    expected = (
        "Plant(Fuel: gas\n"
        + "      HHV Efficiency: 55.00%\n"
        + "      Average Availability Factor: 91.00%\n"
        + "      COD Date: 01-Jan-2022\n"
        + "      Expected Lifetime: 5 years\n"
        + "      Net Capacity: 700 MW\n"
        + "      Capital Cost (£/kW): "
        + "285.7142857142857\n"
        + "      Capital Cost (£k): 200000\n"
        + "        2020: 100000\n"
        + "        2021: 100000\n"
        + "      Fixed Operating Costs (£k):\n"
        + "        2022: 5000.0\n"
        + "        2023: 5000.0\n"
        + "        2024: 5000.0\n"
        + "        2025: 5000.0\n"
        + "        2026: 5000.0\n"
        + "      Variable Opex (£ per hour): 20.0\n"
        + "      Cost Base Date: 01-Jan-2022\n"
        + "      Discount Rate: 10.00%\n"
        + "      Fuel Carbon Intensity: 0.185te/MWh\n"
        + "      Carbon Capture Rate: 95.0%"
    )
    assert result == expected


# print repr for plant class
# test exceptions for hhv_eff and availability
# test attribute errors
# test type errors
# test check dates series
# test energy production profile function
# test tuple provision for fuel costs profile
# test tuple provision of load factors in fuel cost profile
# test carbon cost profile
# test variable cost profile
# test fixed cost profile
# test build cashflows
# test build pv cashflows
# test calculate lcoe
# test calculate annual lcoe


def test_hhv_efficiency_out_of_range():
    """Tests that an error is raised if hhv_efficiency is out of range."""
    with pytest.raises(ValueError):
        Plant(
            fuel="gas",
            hhv_eff=-0.1,
            availability=0.91,
            cod_date=datetime.date(2022, 1, 1),
            lifetime=5,
            net_capacity_mw=700,
            capital_cost={
                datetime.date(2020, 1, 1): 100000,
                datetime.date(2021, 1, 1): 100000,
            },
            fixed_opex_kgbp=10.0,
            variable_opex_gbp_hr=0.1,
            cost_base_date=datetime.date(2022, 1, 1),
            discount_rate=0.1,
            fuel_carbon_intensity=0.185,
            carbon_capture_rate=0.95,
        )


def test_availability_factor_out_of_range():
    """Tests that an error is raised if availability factor is out of range."""
    with pytest.raises(ValueError):
        Plant(
            fuel="gas",
            hhv_eff=0.55,
            availability=1.1,
            cod_date=datetime.date(2022, 1, 1),
            lifetime=5,
            net_capacity_mw=700,
            capital_cost={
                datetime.date(2020, 1, 1): 100000,
                datetime.date(2021, 1, 1): 100000,
            },
            fixed_opex_kgbp=10.0,
            variable_opex_gbp_hr=0.1,
            cost_base_date=datetime.date(2022, 1, 1),
            discount_rate=0.1,
            fuel_carbon_intensity=0.185,
            carbon_capture_rate=0.95,
        )


def test_build_profile():
    """Tests that the build_profile method returns the expected output."""
    availability = {
        datetime.date(2022, 1, 1): 0.91,
        datetime.date(2023, 1, 1): 0.91,
        datetime.date(2024, 1, 1): 0.91,
        datetime.date(2025, 1, 1): 0.91,
        datetime.date(2026, 1, 1): 0.91,
    }
    plant = Plant(
        fuel="gas",
        hhv_eff=0.55,
        availability=availability,
        cod_date=datetime.date(2022, 1, 1),
        lifetime=5,
        net_capacity_mw=700,
        capital_cost={
            datetime.date(2020, 1, 1): 100000,
            datetime.date(2021, 1, 1): 100000,
        },
        fixed_opex_kgbp=10.0,
        variable_opex_gbp_hr=0.1,
        cost_base_date=datetime.date(2022, 1, 1),
        discount_rate=0.1,
        fuel_carbon_intensity=0.185,
        carbon_capture_rate=0.95,
    )
    result = plant.build_profile(availability, datetime.date(2022, 1, 1), 5)
    expected = np.array(
        [
            np.array(["2022", "2023", "2024", "2025", "2026"], dtype="datetime64[Y]"),
            np.array([0.91, 0.91, 0.91, 0.91, 0.91]),
        ]
    )
    assert (result == expected).all()


def test_fuel_costs_profile_tuple():
    """Tests that the method works when provided with a tuple."""
    plant = Plant(
        fuel="gas",
        hhv_eff=0.55,
        availability=0.91,
        cod_date=datetime.date(2022, 1, 1),
        lifetime=5,
        net_capacity_mw=700,
        capital_cost={
            datetime.date(2020, 1, 1): 100000,
            datetime.date(2021, 1, 1): 100000,
        },
        fixed_opex_kgbp=10.0,
        variable_opex_gbp_hr=0.1,
        cost_base_date=datetime.date(2022, 1, 1),
        discount_rate=0.1,
        fuel_carbon_intensity=0.185,
        carbon_capture_rate=0.95,
    )
    gas_prices = (datetime.date(2022, 1, 1), 20.0)
    load_factors = (datetime.date(2022, 1, 1), 0.5)
    result = plant.fuel_costs_profile(gas_prices, load_factors)
    expected = (
        np.arange(
            datetime.date(2022, 1, 1),
            np.datetime64(datetime.date(2022, 1, 1), "Y") + np.timedelta64(5, "Y"),
            dtype="datetime64[Y]",
        ),
        np.full(5, 111363.63636363635),
    )
    assert (result[0] == expected[0]).all() | (result[1] == expected[1]).all()
