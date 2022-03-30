"""Levelised Cost of eNeRGy."""
from collections import Counter, namedtuple
from datetime import date
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd


class Plant:
    """A class to hold the information about the plant being modelled."""

    def __init__(
        self,
        fuel: str,
        hhv_eff: float,
        cod_date: date,
        lifetime: int,
        net_capacity_mw: float,
        capital_cost: dict[date, int],
        fixed_opex_kgbp: Union[float, dict[date, float]],
        variable_opex_gbp_hr: Union[float, dict[date, float]],
        cost_base_date: date,
        discount_rate: float,
        fuel_carbon_intensity: Optional[float] = None,
        carbon_capture_rate: Optional[float] = None,
    ) -> None:
        """Object class for energy production plant.

        Args:
            fuel (str): The fuel used by the plant.
            hhv_eff (float): The HHV efficiency, a factor between 0 and 1.
            cod_date (date): The Commercial Operations Date.
            lifetime (int): Lifetime in full years.
            net_capacity_mw (float): Net power output of the plant in HHV MW.
            capital_cost (dict[date, int]): A dictionary containing the capital
                cost profile, where key is a date and value is the Capital cost
                in kGBP.
            fixed_opex_kgbp (Union[float, dict[date, float]]): The fixed opex
                profile, provided as either an annual figure to be repeated
                through the plant lifetime, or a profile of costs in a dictionary
                where key is the date and value is the fixed cost.  Both in KGBP.
            variable_opex_gbp_hr (Union[float, dict[date, float]]): The variable opex
                profile, provided as either an annual figure to be repeated
                through the plant lifetime, or a profile of costs in a dictionary
                where key is the date and value is the fixed cost.  Both in GBP per hr.
            cost_base_date (date): The base date for all costs.
            discount_rate (float): The target discount rate for the investment.
            fuel_carbon_intensity (Optional[float], optional): The carbon intensity of
                the fuel in te/MWh.
            carbon_capture_rate (Optional[float], optional): The carbon capture rate as
                a factor between 0 and 1.
        """
        # TODO: refactor below to ensure that changing the value subsequently is caught.
        if not 0 <= hhv_eff <= 1:
            raise ValueError("hhv_eff is out of range!")

        self.fuel = fuel
        self.hhv_eff = hhv_eff
        self.cod = cod_date
        self.lifetime = lifetime
        self.net_capacity_mw = net_capacity_mw
        self.capital_cost = capital_cost
        self.fixed_opex_kgbp = self.build_profile(
            fixed_opex_kgbp, self.cod, self.lifetime
        )
        self.variable_opex_gbp_hr = variable_opex_gbp_hr
        self.cost_base = cost_base_date
        self.discount_rate = discount_rate
        self.fuel_carbon_intensity = fuel_carbon_intensity
        self.carbon_capture_rate = carbon_capture_rate
        self.date_range = np.arange(
            self.cod,
            np.datetime64(self.cod, "Y") + np.timedelta64(self.lifetime, "Y"),
            dtype="datetime64[Y]",
        )

    def build_profile(
        self, num: Union[float, dict[date, float]], start_date: date, years: int
    ) -> Tuple:
        """Checks input and builds or returns profile of prices."""
        date_range = np.arange(
            start_date,
            np.datetime64(start_date, "Y") + np.timedelta64(years, "Y"),
            dtype="datetime64[Y]",
        )
        if type(num) is dict:
            if ([x.year for x in num.keys()] != np.int32(date_range + 1970)).all():
                raise AttributeError("Input doesn't match plant lifetime!")
            else:
                return (date_range, np.fromiter(num.values(), dtype=float))

        return (date_range, np.full(years, num))

    def check_dates(self, data: tuple) -> bool:
        """Checks a tuple of numpy arrays for date alignment.

        Args:
            data (tuple): The tuple to be checked.  Expected to be in the format
                ([dates], [data])

        Returns:
            bool: Boolean which is True if the dates match
        """
        if np.all(data[0] != self.date_range):
            raise AttributeError("Input doesn't match plant lifetime!")

    def energy_production_profile(
        self, load_factors: Union[float, Tuple], hours_in_year: int = 8760
    ) -> Tuple:
        """Function to calculate the energy production per year in GWh HHV.

        Args:
            load_factors (Union[float, Tuple]): Factor representing % of running in
                the year.  Can be either a single figure which is applied to each
                year or a profile in the form of a Tuple of two numpy arrays, the
                first containing the date, the second the load factors.
            hours_in_year (int, optional): Number of hours in a year. Defaults to 8760.

        Returns:
            Tuple: Two numpy arrays, first showing dates and the second showing
                generation in GWh.
        """
        if type(load_factors) is tuple:
            self.check_dates(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                (self.net_capacity_mw * load_factors * hours_in_year / 1000),
            ),
        )

    def fuel_costs_profile(
        self,
        fuel_prices: Union[float, Tuple],
        load_factors: Union[float, Tuple],
        hours_in_year: int = 8760,
    ) -> Tuple:
        """Calculates an annual profile for Natural Gas feed costs.

        Args:
            gas_prices: A Dict of floats with key of year and value of
                        prices in £/MWh (HHV basis).
            load_factors: Dict of floats with key of year and value of
                          percentage 'load_factors' in the given year.
            ng_flow_hhv: Represents the NG Feed Flow Rate (HHV terms MWth).
            hours_in_year: optional The number of hours in a year, default 8760.

        Returns:
             Dict of gas cost in each given year.

        """
        if type(fuel_prices) is tuple:
            self.check_dates(fuel_prices)
            fuel_prices = fuel_prices[1]

        if type(load_factors) is tuple:
            self.check_dates(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                fuel_prices
                * hours_in_year
                * load_factors
                * self.net_capacity_mw
                / self.hhv_eff
                / 1000,
            ),
        )

    def carbon_cost_profile(
        self,
        carbon_prices: Union[float, Tuple],
        load_factors: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> Tuple:
        """_summary_.

        Args:
            carbon_prices (Union[float, Tuple]): _description_
            load_factors (Union[float, Tuple]): _description_
            co2_transport_storage_cost (float): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: _description_
        """
        if type(carbon_prices) is tuple:
            self.check_dates(carbon_prices)
            carbon_prices = carbon_prices[1]

        if type(load_factors) is tuple:
            self.check_dates(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                carbon_prices
                * hours_in_year
                * load_factors
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * (1 - self.carbon_capture_rate)
                / 1000,
            ),
            np.full(
                self.lifetime,
                co2_transport_storage_cost
                * hours_in_year
                * load_factors
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * self.carbon_capture_rate
                / 1000,
            ),
        )

    def variable_cost_profile(
        self, load_factors: Union[float, Tuple], hours_in_year: int = 8760
    ) -> Tuple:
        """_summary_.

        Args:
            load_factors (Union[float, Tuple]): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: _description_
        """
        if type(load_factors) is tuple:
            self.check_dates(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                self.variable_opex_gbp_hr * load_factors * hours_in_year / 1000,
            ),
        )

    def fixed_cost_profile(
        self,
    ) -> Tuple:
        """_summary_.

        Args:
            load_factors (Union[float, Tuple]): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: _description_
        """
        if type(self.fixed_opex_kgbp) is tuple:
            self.check_dates(self.fixed_opex_kgbp)
            fixed_costs = self.fixed_opex_kgbp[1]
        else:
            fixed_costs = self.fixed_opex_kgbp

        return (
            self.date_range,
            np.full(self.lifetime, fixed_costs),
        )

    def build_cashflows(
        self,
        load_factors: Union[float, Tuple],
        fuel_prices: Union[float, Tuple],
        carbon_prices: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> dict:
        """_summary_.

        Args:
            load_factors (Union[float, Tuple]): _description_
            fuel_prices (Union[float, Tuple]): _description_
            carbon_prices (Union[float, Tuple]): _description_
            co2_transport_storage_cost (float): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            dict: _description_
        """
        production = ("production_GWth", self.energy_production_profile(load_factors))
        capital = (
            "capital_kgbp",
            self.build_profile(
                self.capital_cost, next(iter(self.capital_cost)), len(self.capital_cost)
            ),
        )
        fixed = ("fixed_opex_kgbp", self.fixed_cost_profile())
        var = ("variable_opex_kgbp", self.variable_cost_profile(load_factors))
        fuel = ("fuel_kgbp", self.fuel_costs_profile(fuel_prices, load_factors))
        carbon_costs = self.carbon_cost_profile(
            carbon_prices, load_factors, co2_transport_storage_cost
        )
        co2emit = ("carbon_emissions_kgbp", (carbon_costs[0], carbon_costs[1]))
        co2_store = ("carbon_storage_kgbp", (carbon_costs[0], carbon_costs[2]))

        source = [production, capital, fixed, var, fuel, co2emit, co2_store]

        return pd.DataFrame(
            {name: (pd.Series(data[1], index=data[0])) for name, data in source},
        )

    def calculate_lcoe(
        self,
        load_factors: Union[float, Tuple],
        fuel_prices: Union[float, Tuple],
        carbon_prices: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> float:
        """_summary_.

        Args:
            load_factors (Union[float, Tuple]): _description_
            fuel_prices (Union[float, Tuple]): _description_
            carbon_prices (Union[float, Tuple]): _description_
            co2_transport_storage_cost (float): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            float: _description_
        """
        pvs = present_value_factor(self.cost_base, self.discount_rate)
        pvs = pd.DataFrame(pvs[1], index=pvs[0], columns=["discount_rate"])
        cf = self.build_cashflows(
            load_factors, fuel_prices, carbon_prices, co2_transport_storage_cost
        )
        cf.fillna(0, inplace=True)
        pv_cf = cf.multiply(pvs[pvs.index.isin(cf.index)]["discount_rate"], axis=0)
        srmc = pv_cf[
            [
                "variable_opex_kgbp",
                "fuel_kgbp",
                "carbon_emissions_kgbp",
                "carbon_storage_kgbp",
            ]
        ]
        lrmc = pv_cf[["capital_kgbp", "fixed_opex_kgbp"]]
        srmc = sum(srmc.stack().values) / sum(pv_cf.production_GWth.values)
        lrmc = sum(lrmc.stack().values) / sum(pv_cf.production_GWth.values)
        lcoe = srmc + lrmc
        full = cf.drop("production_GWth", axis=1).sum() / cf.production_GWth.sum()

        Lcoe = namedtuple(
            "LCONRG",
            ["lcoe", "srmc", "lrmc"] + list(full.index),
        )
        return Lcoe(
            lcoe,
            srmc,
            lrmc,
            full.capital_kgbp,
            full.fixed_opex_kgbp,
            full.variable_opex_kgbp,
            full.fuel_kgbp,
            full.carbon_emissions_kgbp,
            full.carbon_storage_kgbp,
        )


def present_value_factor(
    base_date: date,
    discount_rate: float,
    no_of_years: int = 50,
) -> Tuple:
    """_summary_.

    Args:
        base_date (date): _description_
        discount_rate (float): _description_
        no_of_years (int, optional): _description_. Defaults to 50.

    Returns:
        tuple: _description_
    """
    date_range = np.arange(
        base_date,
        np.datetime64(base_date, "Y") + np.timedelta64(no_of_years, "Y"),
        dtype="datetime64[Y]",
    )

    data = 1 / (
        (1 + discount_rate) ** np.int32(date_range - np.datetime64(base_date, "Y"))
    )

    return (date_range, data)


def fuel_costs(
    gas_prices: dict,
    load_factors: dict,
    fuel_flow_hhv: int,
    hours_in_year: int = 8760,
) -> dict:
    """Calculates an annual profile for Natural Gas feed costs.

    Args:
        gas_prices: A Dict of floats with key of year and value of
                    prices in £/MWh (HHV basis).
        load_factors: Dict of floats with key of year and value of
                      percentage 'load_factors' in the given year.
        ng_flow_hhv: Represents the NG Feed Flow Rate (HHV terms MWth).
        hours_in_year: optional The number of hours in a year, default 8760.

    Returns:
         Dict of gas cost in each given year.

    """
    return {
        year: fuel_flow_hhv * gas_prices[year] * hours_in_year * lf / 1000000
        for year, lf in load_factors.items()
    }


def carbon_costs(
    carbon_prices: dict,
    load_factors: dict,
    fuel_flow_kgh: int,
    carbon_capture_rate: float,
    carbon_fraction: float,
    co2_transport_storage_cost: float,
    carbon_to_co2: float = 3.6667,
    hours_in_year: int = 8760,
) -> dict:
    """Calculates an annual profile of carbon costs.

    Total resulting cost is for the market cost of emissions and the fee
    paid to a CO2 transport and storage operator.

    Args:
        carbon_prices: Dict of floats with key of year and value of
                       total price in GBP per tonne of CO2.
        load_factors: Dict of floats with key of year and value of
                      the percentage of running in the given year.
        fuel_flow_kgh: Fuel Feed Flow Rate (kg per hour)
        carbon_capture_rate: float
            A float representing the carbon capture rate.
        carbon_fraction : float, optional
            A float representing the fraction of fuel which is carbon.
        co2_transport_storage_cost : float, optional
            The fee in GBP per tonne of transport and storage of any CO2 captured.
            This price is used along with `carbon_prices` to calculate a volume-
            weighted average price of CO2 in GBP per tonne.
        carbon_to_co2 : float, optional
            A float representing the conversion factor from Carbon to Carbon Dioxide.
            The default is 3.6667.
        hours_in_year : int, optional
            The number of hours in a year, default 8760.

    Returns:
        Dict of carbon cost per year for years where load factor is greater than zero.
    """
    return {
        year: (
            fuel_flow_kgh
            * carbon_fraction
            * carbon_to_co2
            * lf
            * hours_in_year
            * (
                (1 - carbon_capture_rate) * carbon_prices[year]
                + (carbon_capture_rate * co2_transport_storage_cost)
            )
        )
        / 1000
        / 1000000
        for year, lf in load_factors.items()
    }


def energy_production(
    load_factors: dict, energy_output: int, hours_in_year: int = 8760
) -> dict:
    """Calculates the volume of MWh energy production for each year of operation.

    Args:
        load_factors: Dict of floats with key of year and value of
                      the percentage of running in the given year.
        energy_output: Energy Output in MWh per hour.
        hours_in_year: The number of hours in a year, default 8760.

    Returns:
        A Dict of MWth HHV production volumes of Hydrogen for years where
        load factor is greater than zero.
    """
    return {
        year: (energy_output * lf * hours_in_year) for year, lf in load_factors.items()
    }


def variable_costs_profile(
    load_factors: dict, variable_opex_gbp_hr: int, hours_in_year: int = 8760
) -> dict:
    """Calculates the variable cost profile for each year of operation.

    Args:
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    hours_in_year : int, optional
        The number of hours in a year, default 8760.

    Returns:
        A Dict of variable costs for years where load factor is greater than zero.
    """
    return {
        year: (variable_opex_gbp_hr * lf * hours_in_year) / 1000000
        for year, lf in load_factors.items()
    }


def calculate_srmc(
    gas_prices: dict,
    load_factors: dict,
    fuel_flow_hhv: int,
    carbon_prices: dict,
    fuel_flow_kgh: int,
    carbon_capture_rate: float,
    carbon_fraction: float,
    co2_transport_storage_cost: float,
    variable_opex_gbp_hr: int,
    energy_output: int,
    discount_rate: float,
    base_year: int,
) -> float:
    """Calculates the Short Run Marginal Cost given inputs.

    This function takes the time-dependent variables and calculates an SRMC.

    Args:
        gas_prices: Profile of gas prices by year in £/MWh HHV.
        load_factors: Dict of floats with index of year and value of
                      percentage 'load_factors' in the given year.
        fuel_flow_hhv:
        carbon_capture_rate:
        carbon_fraction:
        co2_transport_storage_cost:
        variable_opex_gbp_hr
        energy_output:
        discount_rate:
        base_year:

    Returns:
        A float representing the Short Run Marginal Cost.

    """
    pvs = present_value_factor(base_year, discount_rate)
    fuel_cost = fuel_costs(gas_prices, load_factors, fuel_flow_hhv)
    carbon_cost = carbon_costs(
        carbon_prices,
        load_factors,
        fuel_flow_kgh,
        carbon_capture_rate,
        carbon_fraction,
        co2_transport_storage_cost,
    )
    variable_cost = variable_costs_profile(load_factors, variable_opex_gbp_hr)
    production_profile = energy_production(load_factors, energy_output)
    cost = {
        year: (fuel_cost[year] + carbon_cost[year] + variable_cost[year]) * pvs[year]
        for year in fuel_cost
    }
    production = {
        year: production_profile[year] * pvs[year] for year in production_profile
    }
    return sum(cost.values()) / sum(production.values()) * 1000000


def calculate_lrmc(
    capital_cost: dict,
    fixed_opex_mgbp_yr: dict,
    energy_output: int,
    load_factors: dict,
    discount_rate: float,
    base_year: int,
) -> float:
    """Calculates the Short Run Marginal Cost given inputs.

    This function takes the time-dependent variables and calculates an SRMC.

    Args:
        capital_cost:
        fixed_opex_mgbp_yr:
        energy_output:
        load_factors: Dict of floats with index of year and value of
                      percentage 'load_factors' in the given year.
        discount_rate:
        base_year:

    Returns:
        A float representing the Long Run Marginal Cost (excluding SRMC).

    """
    pvs = present_value_factor(base_year, discount_rate)
    production_profile = energy_production(load_factors, energy_output)
    capex = {year: capital_cost[year] * pvs[year] for year in capital_cost}
    opex = {year: fixed_opex_mgbp_yr[year] * pvs[year] for year in fixed_opex_mgbp_yr}
    cost = dict(Counter(capex) + Counter(opex))
    production = {
        year: production_profile[year] * pvs[year] for year in production_profile
    }
    return sum(cost.values()) / sum(production.values()) * 1000000


def build_cashflows(
    capital_cost: dict,
    fixed_opex_mgbp_yr: dict,
    energy_output: int,
    load_factors: dict,
    discount_rate: float,
    base_year: int,
    gas_prices: dict,
    fuel_flow_hhv: int,
    carbon_prices: dict,
    fuel_flow_kgh: int,
    carbon_capture_rate: float,
    carbon_fraction: float,
    co2_transport_storage_cost: float,
    variable_opex_gbp_hr: int,
) -> dict:
    """_summary_.

    Args:
        capital_cost (dict): _description_
        fixed_opex_mgbp_yr (dict): _description_
        energy_output (int): _description_
        load_factors (dict): _description_
        discount_rate (float): _description_
        base_year (int): _description_
        gas_prices (dict): _description_
        fuel_flow_hhv (int): _description_
        carbon_prices (dict): _description_
        fuel_flow_kgh (int): _description_
        carbon_capture_rate (float): _description_
        carbon_fraction (float): _description_
        co2_transport_storage_cost (float): _description_
        variable_opex_gbp_hr (int): _description_

    Returns:
        dict: _description_
    """
    pvs = present_value_factor(base_year, discount_rate)
    production_profile = energy_production(load_factors, energy_output)
    fuel_cost = fuel_costs(gas_prices, load_factors, fuel_flow_hhv)
    carbon_cost = carbon_costs(
        carbon_prices,
        load_factors,
        fuel_flow_kgh,
        carbon_capture_rate,
        carbon_fraction,
        co2_transport_storage_cost,
    )
    variable_cost = variable_costs_profile(load_factors, variable_opex_gbp_hr)
    pv_capex = sum(capital_cost[year] * pvs[year] for year in capital_cost)
    pv_production = sum(
        production_profile[year] * pvs[year] for year in production_profile
    )
    capital_annuity = {
        year: ((pv_capex / pv_production) * production_profile[year])
        for year in production_profile
    }
    return (
        dict(
            {
                "production_MWth": production_profile,
                "capital_mgpb": capital_cost,
                "fixed_opex_mgbp": fixed_opex_mgbp_yr,
                "fuel_mgbp": fuel_cost,
                "carbon_mgbp": carbon_cost,
                "variable_opex_mgbp": variable_cost,
                "capital_annuity_mgbp": capital_annuity,
            }
        ),
        pvs,
    )
