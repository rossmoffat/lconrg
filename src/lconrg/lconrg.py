"""Levelised Cost of eNeRGy."""
from collections import namedtuple
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
        availability: float,
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
            availability (float): The annual availability, a factor between 0 and 1.
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

        Raises:
            ValueError: Where the HHV Efficiency or Availability Factor is
                negative or greater than 1
        """
        # TODO: refactor below to ensure that changing the value subsequently is caught.
        if not 0 <= hhv_eff <= 1:
            raise ValueError("hhv_eff is out of range!")

        if not 0 <= availability <= 1:
            raise ValueError("Availability factor is out of range!")

        self.fuel = fuel
        self.hhv_eff = hhv_eff
        self.availability = availability
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

    def __str__(self) -> str:
        """String representation for Plant class.

        Returns:
            str: Summary of Plant class
        """
        capex = str()
        opex = str()

        for key, values in self.capital_cost.items():
            capex += f"        {key:%Y}: {values}\n"

        for i, data in enumerate(self.fixed_opex_kgbp[0]):
            opex += f"        {data}: {self.fixed_opex_kgbp[1][i]}\n"

        return (
            f"Plant(Fuel: {self.fuel}\n"
            + f"      HHV Efficiency: {self.hhv_eff: .2%}\n"
            + f"      Availability Factor: {self.availability: .2%}\n"
            + f"      COD Date: {self.cod:%d-%b-%Y}\n"
            + f"      Expected Lifetime: {self.lifetime} years\n"
            + f"      Net Capacity: {self.net_capacity_mw} MW\n"
            + f"      Capital Cost (£k): {sum(self.capital_cost.values())}\n"
            + f"{capex}"
            + "      Fixed Operating Costs (£k):\n"
            + f"{opex}"
            + f"      Variable Opex (£ per hour): {self.variable_opex_gbp_hr}\n"
            + f"      Cost Base Date: {self.cost_base: %d-%b-%Y}\n"
            + f"      Discount Rate: {self.discount_rate: .2%}\n"
            + f"      Fuel Carbon Intensity: {self.fuel_carbon_intensity}te/MWh\n"
            + f"      Carbon Capture Rate: {self.carbon_capture_rate: .1%}"
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

    def check_dates_tuple(self, data: tuple) -> bool:
        """Checks a tuple of numpy arrays for date alignment.

        Args:
            data (tuple): The tuple to be checked.  Expected to be in the format
                ([dates], [data])

        Raises:
            AttributeError: The dates in the Tuple don't match the
                expected lifetime of the Plant.
        """
        if np.all(data[0] != self.date_range):
            raise AttributeError("Input doesn't match plant lifetime!")

    def check_dates_series(self, data: pd.Series) -> bool:
        """Checks a pandas series for date alignment.

        Args:
            data (pd.Series): The pandas series to be checked.  Expected to have
                index of dates.

        Raises:
            AttributeError: The dates in the series don't match the
                expected liftime of the Plant.
        """
        if np.all(data.index != self.date_range):
            raise AttributeError("Input doesn't match plant lifetime!")

    def energy_production_profile(
        self, load_factors: Union[float, Tuple], hours_in_year: Optional[int] = 8760
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
            self.check_dates_tuple(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                (
                    self.net_capacity_mw
                    * load_factors
                    * self.availability
                    * hours_in_year
                    / 1000
                ),
            ),
        )

    def fuel_costs_profile(
        self,
        fuel_prices: Union[float, Tuple],
        load_factors: Union[float, Tuple],
        hours_in_year: Optional[int] = 8760,
    ) -> Tuple:
        """Calculates an annual fuel costs profile in kGBP.

        Args:
            fuel_prices (Union[float, Tuple]): Factor representing cost of fuel in
                GBP/HHV MWh.  Can be either a single figure which is applied to each
                year or a profile in the form of a Tuple of two numpy arrays, the
                first containing the date, the second the fuel prices.
            load_factors (Union[float, Tuple]): Factor representing % of running in
                the year.  Can be either a single figure which is applied to each
                year or a profile in the form of a Tuple of two numpy arrays, the
                first containing the date, the second the load factors.
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: Two numpy arrays, first showing dates and the second showing
                fuel costs in kGBP.
        """
        if type(fuel_prices) is tuple:
            self.check_dates_tuple(fuel_prices)
            fuel_prices = fuel_prices[1]

        if type(load_factors) is tuple:
            self.check_dates_tuple(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                fuel_prices
                * hours_in_year
                * load_factors
                * self.availability
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
        hours_in_year: Optional[int] = 8760,
    ) -> Tuple:
        """Calculates annual carbon cost profiles for emission and storage in kGBP.

        Args:
            carbon_prices (Union[float, Tuple]): Factor representing cost to emit
                carbon in GBP/te.  Can be either a single figure which is applied
                to each year or a profile in the form of a Tuple of two numpy
                arrays, the first containing the date, the second the fuel prices.
            load_factors (Union[float, Tuple]): Factor representing % of running in
                the year.  Can be either a single figure which is applied to each
                year or a profile in the form of a Tuple of two numpy arrays, the
                first containing the date, the second the load factors.
            co2_transport_storage_cost (float): Cost to transport and store carbon in
                GBP/te.
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: Three numpy arrays, first showing dates, second showing cost of
                emissions in kGBP, third showing cost of storage in kGBP.
        """
        if type(carbon_prices) is tuple:
            self.check_dates_tuple(carbon_prices)
            carbon_prices = carbon_prices[1]

        if type(load_factors) is tuple:
            self.check_dates_tuple(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                carbon_prices
                * hours_in_year
                * load_factors
                * self.availability
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * (1 - self.carbon_capture_rate)
                * self.net_capacity_mw
                / 1000,
            ),
            np.full(
                self.lifetime,
                co2_transport_storage_cost
                * hours_in_year
                * load_factors
                * self.availability
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * self.carbon_capture_rate
                * self.net_capacity_mw
                / 1000,
            ),
        )

    def variable_cost_profile(
        self, load_factors: Union[float, Tuple], hours_in_year: Optional[int] = 8760
    ) -> Tuple:
        """_summary_.

        Args:
            load_factors (Union[float, Tuple]): _description_
            hours_in_year (int, optional): _description_. Defaults to 8760.

        Returns:
            Tuple: _description_
        """
        if type(load_factors) is tuple:
            self.check_dates_tuple(load_factors)
            load_factors = load_factors[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                self.variable_opex_gbp_hr
                * load_factors
                * self.availability
                * hours_in_year
                / 1000,
            ),
        )

    def fixed_cost_profile(
        self,
    ) -> Tuple:
        """_summary_.

        Args:

        Returns:
            Tuple: _description_
        """
        if type(self.fixed_opex_kgbp) is tuple:
            self.check_dates_tuple(self.fixed_opex_kgbp)
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
        hours_in_year: Optional[int] = 8760,
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

    def build_pv_cashflows(
        self,
        load_factors: Union[float, Tuple],
        fuel_prices: Union[float, Tuple],
        carbon_prices: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: Optional[int] = 8760,
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
        return pv_cf

    def calculate_lcoe(
        self,
        load_factors: Union[float, Tuple],
        fuel_prices: Union[float, Tuple],
        carbon_prices: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: Optional[int] = 8760,
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
        pv_cf = self.build_pv_cashflows(
            load_factors, fuel_prices, carbon_prices, co2_transport_storage_cost
        )
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
        full = pv_cf.drop("production_GWth", axis=1).sum() / pv_cf.production_GWth.sum()

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

    def calculate_annual_lcoe(
        self,
        load_factors: Union[float, Tuple],
        fuel_prices: Union[float, Tuple],
        carbon_prices: Union[float, Tuple],
        co2_transport_storage_cost: float,
        hours_in_year: Optional[int] = 8760,
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
        pv_cf = self.build_pv_cashflows(
            load_factors, fuel_prices, carbon_prices, co2_transport_storage_cost
        )
        pv_cf["capital_annuity_kgbp"] = (
            sum(pv_cf.capital_kgbp) / sum(pv_cf.production_GWth)
        ) * pv_cf.production_GWth
        pv_cf.drop("capital_kgbp", axis=1, inplace=True)
        pv_profile = (
            pv_cf.drop("production_GWth", axis=1)
            .divide(pv_cf.production_GWth, axis=0)
            .sum(axis=1)
        )
        return pv_profile


def present_value_factor(
    base_date: date,
    discount_rate: float,
    no_of_years: Optional[int] = 50,
) -> Tuple:
    """_summary_.

    Args:
        base_date (date): _description_
        discount_rate (float): _description_
        no_of_years (int, optional): _description_. Defaults to 50.

    Returns:
        Tuple: _description_
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
