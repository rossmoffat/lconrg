"""Levelised Cost of eNeRGy."""

from collections import namedtuple
from datetime import date
from typing import Any, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd

NDArrayDate = npt.NDArray[np.datetime64]
NDArrayFloat = npt.NDArray[np.float64]
NDArrayInt = npt.NDArray[np.int32]


class Plant:
    """A class to hold the information about the plant being modelled."""

    def __init__(
        self,
        fuel: str,
        hhv_eff: float,
        availability: Union[float, dict[date, float]],
        cod_date: date,
        lifetime: int,
        net_capacity_mw: Union[float, int],
        capital_cost: dict[date, Union[float, int]],
        fixed_opex_kgbp: Union[float, int, dict[date, Union[float, int]]],
        variable_opex_gbp_hr: Union[float, int, dict[date, Union[float, int]]],
        cost_base_date: date,
        discount_rate: float,
        fuel_carbon_intensity: float,
        carbon_capture_rate: float,
    ) -> None:
        """
        Object class for energy production plant.

        Parameters
        ----------
        fuel: str
            The fuel used by the plant.
        hhv_eff: float
            The HHV efficiency, a factor between 0 and 1.
        availability: float or dict of date, float
            The annual availability profile, a factor between 0 and 1.
            Provided either as a single figure to be repeated through
            the plant lifetime, or a profile of costs in a dictionary
            where the key is a date and the value is the availability
            factor.
        cod_date: date
            The Commercial Operations Date.
        lifetime: int
            Lifetime in full years.
        net_capacity_mw: float or int
            Net power output of the plant in HHV MW.
        capital_cost: dict of date, float or int
            A dictionary containing the capital cost profile, where key
            is a date and value is the Capital cost in kGBP.
        fixed_opex_kgbp: float or int or dict of date, float or int
            The fixed opex profile, provided as either an annual figure to be
            repeated through the plant lifetime, or a profile of costs in a
            dictionary where key is the date and value is the fixed cost. Both
            in kGBP.
        variable_opex_gbp_hr: float or int or dict of date, float or int
            The variable opex profile, provided as either an annual figure to
            be repeated through the plant lifetime, or a profile of costs in a
            dictionary where key is the date and value is the fixed cost.
            Both in GBP per hr.
        cost_base_date: date
            The base date for all costs.
        discount_rate: float
            The target discount rate for the investment.
        fuel_carbon_intensity: float
            The carbon intensity of the fuel in te/MWh.
        carbon_capture_rate: float
            The carbon capture rate as a factor between 0 and 1.

        Raises
        ------
        ValueError
            Where the HHV Efficiency or Availability Factor is negative or
            greater than 1
        """
        # TODO: refactor below to ensure that changing the value subsequently is caught.
        if not 0 <= hhv_eff <= 1:
            raise ValueError("hhv_eff is out of range!")

        if type(availability) is float:
            if not 0 <= availability <= 1:
                raise ValueError("Availability factor is out of range!")
        elif type(availability) is dict:
            if not all(0 < x < 1 for x in availability.values()):
                raise ValueError("Availability factor is out of range!")

        self.fuel = fuel
        self.hhv_eff = hhv_eff
        self.cod = cod_date
        self.lifetime = lifetime
        self.availability = self.build_profile(availability, self.cod, self.lifetime)
        self.net_capacity_mw = float(net_capacity_mw)
        self.capital_cost = capital_cost
        self.fixed_opex_kgbp = self.build_profile(
            fixed_opex_kgbp, self.cod, self.lifetime
        )
        self.variable_opex_gbp_hr = self.build_profile(
            variable_opex_gbp_hr, self.cod, self.lifetime
        )
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
        """
        Print string for plant class.

        Returns
        -------
        str
            Summary of Plant class
        """
        capex = str()

        for key, values in self.capital_cost.items():
            capex += f"        {key:%Y}: {values}\n"

        opex_list = [
            f"        {year}" + f": {self.fixed_opex_kgbp[1][i]}\n"
            for i, year in enumerate(self.fixed_opex_kgbp[0])
        ]
        opex = "".join(opex_list)

        return (
            f"Plant(Fuel: {self.fuel}\n"
            + f"      HHV Efficiency: {self.hhv_eff:.2%}\n"
            + "      Average Availability Factor: "
            + f"{np.mean(self.availability[1]):.2%}\n"
            + f"      COD Date: {self.cod:%d-%b-%Y}\n"
            + f"      Expected Lifetime: {self.lifetime} years\n"
            + f"      Net Capacity: {int(self.net_capacity_mw)} MW\n"
            + "      Capital Cost (£/kW): "
            + f"{(sum(self.capital_cost.values()) / self.net_capacity_mw)}\n"
            + f"      Capital Cost (£k): {sum(self.capital_cost.values())}\n"
            + f"{capex}"
            + "      Fixed Operating Costs (£k):\n"
            + f"{opex}"
            + "      Variable Opex (£ per hour): "
            + f"{np.mean(self.variable_opex_gbp_hr[1])}\n"
            + f"      Cost Base Date: {self.cost_base:%d-%b-%Y}\n"
            + f"      Discount Rate: {self.discount_rate:.2%}\n"
            + f"      Fuel Carbon Intensity: {self.fuel_carbon_intensity}te/MWh\n"
            + f"      Carbon Capture Rate: {self.carbon_capture_rate:.1%}"
        )

    def build_profile(
        self,
        num: Union[float, int, dict[date, Union[float, int]]],
        start_date: date,
        years: int,
    ) -> Tuple[NDArrayDate, NDArrayFloat]:
        """
        Check input and return profile of prices.

        Parameters
        ----------
        num: float or int or dict of date, float or int
            The values to be used for the profile. Can either be a dict
            with keys of date and value, or can be a single float which
            is used to build a flat profile.
        start_date: date
            The first date for the profile.
        years: int
            The number of years to include in the profile.

        Returns
        -------
        Tuple of NDArrayDate, NDArrayFloat
            Two numpy ndarrays, the first containing year in
            numpy.datetime64 and the second values.

        Raises
        ------
        AttributeError
            The dates in the dict don't match the expected lifetime
            of the Plant.
        TypeError
            The input type is neither a float, nor a Tuple.
        """
        date_range = np.arange(
            start_date,
            np.datetime64(start_date, "Y") + np.timedelta64(years, "Y"),
            dtype="datetime64[Y]",
        )
        if type(num) is dict:
            if np.all([x.year for x in num.keys()] != np.int32(date_range + 1970)):
                raise AttributeError("Input doesn't match plant lifetime!")
            else:
                return (date_range, np.fromiter(num.values(), dtype=float))

        elif type(num) is float:
            return (date_range, np.full(years, num))

        elif type(num) is int:
            return (date_range, np.full(years, float(num)))

        else:
            raise TypeError(f"Unexpected type: {type(num)}")

    def check_dates_tuple(
        self, data: Union[tuple[date, float], tuple[NDArrayDate, NDArrayFloat]]
    ) -> None:
        """
        Check a tuple of numpy arrays for date alignment.

        Parameters
        ----------
        data: tuple of date, float
            The tuple to be checked.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            The dates in the Tuple don't match the expected
            lifetime of the Plant.
        """
        if np.all(data[0] != self.date_range):
            raise AttributeError("Input doesn't match plant lifetime!")
        else:
            return None

    def check_dates_series(self, data: pd.Series) -> None:
        """
        Check a pandas series for date alignment.

        Parameters
        ----------
        data : pd.Series
            The pandas series to be checked. Expected to have index of dates.

        Returns
        -------
        None

        Raises
        ------
        AttributeError
            If the dates in the series don't match the expected lifetime of
            the Plant.

        """
        if np.all(data.index != self.date_range):
            raise AttributeError("Input doesn't match plant lifetime!")
        else:
            return None

    def energy_production_profile(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        hours_in_year: int = 8760,
    ) -> tuple[NDArrayDate, NDArrayFloat]:
        """
        Calculate the energy production per year in GWh HHV.

        Parameters
        ----------
        load_factors : float or tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year. Can be either a
            single figure which is applied to each year or a profile in
            the form of a Tuple of two numpy arrays, the first containing
            the date, the second the load factors.
        hours_in_year : int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        tuple
            Two numpy arrays, first showing dates and the second showing
            generation in GWh.

        Raises
        ------
        TypeError
            If the load_factors is neither a float nor a tuple.
        """
        if isinstance(load_factors, tuple):
            self.check_dates_tuple(load_factors)
            load_factor: Any = load_factors[1]
        elif isinstance(load_factors, float):
            load_factor = load_factors
        elif isinstance(load_factors, pd.Series):
            load_factor = self.pd_series_to_daterange_tuple(load_factors)
            load_factor = load_factor[1]
        else:
            raise TypeError()

        self.check_dates_tuple(self.availability)
        availability = self.availability[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                (
                    self.net_capacity_mw
                    * load_factor
                    * availability
                    * hours_in_year
                    / 1000
                ),
            ),
        )

    def fuel_costs_profile(
        self,
        fuel_prices: Union[float, int, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        hours_in_year: int = 8760,
    ) -> tuple[NDArrayDate, NDArrayFloat]:
        """
        Calculate an annual fuel costs profile in kGBP.

        Parameters
        ----------
        fuel_prices : Union[float, Tuple]
            Factor representing cost of fuel in GBP/HHV MWh. Can be either a single
            figure which is applied to each year or a profile in the form of a Tuple
            of two numpy arrays, the first containing the date, the second the fuel
            prices.
        load_factors : Union[float, Tuple]
            Factor representing % of running in the year. Can be either a single
            figure which is applied to each year or a profile in the form of a Tuple
            of two numpy arrays, the first containing the date, the second the load
            factors.
        hours_in_year : int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        Tuple
            Two numpy arrays, first showing dates and the second showing fuel costs
            in kGBP.

        Raises
        ------
        TypeError
            If the fuel_prices or load_factors are neither a float nor a tuple.
        """
        if isinstance(fuel_prices, tuple):
            self.check_dates_tuple(fuel_prices)
            fuel_price: Any = fuel_prices[1]
        elif isinstance(fuel_prices, float):
            fuel_price = fuel_prices
        elif isinstance(fuel_prices, int):
            fuel_price = float(fuel_prices)
        elif isinstance(fuel_prices, pd.Series):
            fuel_price = self.pd_series_to_daterange_tuple(fuel_prices)
            fuel_price = fuel_price[1]
        else:
            raise TypeError()

        if isinstance(load_factors, tuple):
            self.check_dates_tuple(load_factors)
            load_factor: Any = load_factors[1]
        elif isinstance(load_factors, float):
            load_factor = load_factors
        elif isinstance(load_factors, pd.Series):
            load_factor = self.pd_series_to_daterange_tuple(load_factors)
            load_factor = load_factor[1]
        else:
            raise TypeError()

        self.check_dates_tuple(self.availability)
        availability = self.availability[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                fuel_price
                * hours_in_year
                * load_factor
                * availability
                * self.net_capacity_mw
                / self.hhv_eff
                / 1000,
            ),
        )

    def carbon_cost_profile(
        self,
        carbon_prices: Union[float, int, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        co2_transport_storage_cost: Union[float, int],
        hours_in_year: int = 8760,
    ) -> tuple[NDArrayDate, NDArrayFloat, NDArrayFloat]:
        """
        Calculate annual carbon cost profiles for emission and storage in kGBP.

        Parameters
        ----------
        carbon_prices : float or int or tuple of NDArrayDate, NDArrayFloat
            Factor representing cost to emit carbon in GBP/te. Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the carbon prices.
        load_factors : float or tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year. Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the load factors.
        co2_transport_storage_cost : float
            Cost to transport and store carbon in GBP/te.
        hours_in_year : int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        Tuple
            Three numpy arrays, first showing dates, second showing cost of
            emissions in kGBP, third showing cost of storage in kGBP.

        Raises
        ------
        TypeError
            If the carbon_prices or load_factors are neither a float nor a tuple.
        """
        if isinstance(carbon_prices, tuple):
            self.check_dates_tuple(carbon_prices)
            carbon_price: Any = carbon_prices[1]
        elif isinstance(carbon_prices, float):
            carbon_price = carbon_prices
        elif isinstance(carbon_prices, int):
            carbon_price = float(carbon_prices)
        elif isinstance(carbon_prices, pd.Series):
            carbon_price = self.pd_series_to_daterange_tuple(carbon_prices)
            carbon_price = carbon_price[1]
        else:
            raise TypeError()

        if isinstance(load_factors, tuple):
            self.check_dates_tuple(load_factors)
            load_factor: Any = load_factors[1]
        elif isinstance(load_factors, float):
            load_factor = load_factors
        elif isinstance(load_factors, pd.Series):
            load_factor = self.pd_series_to_daterange_tuple(load_factors)
            load_factor = load_factor[1]
        else:
            raise TypeError()

        self.check_dates_tuple(self.availability)
        availability = self.availability[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                carbon_price
                * hours_in_year
                * load_factor
                * availability
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * (1 - self.carbon_capture_rate)
                * self.net_capacity_mw
                / 1000,
            ),
            np.full(
                self.lifetime,
                co2_transport_storage_cost
                * hours_in_year
                * load_factor
                * availability
                * (self.fuel_carbon_intensity / self.hhv_eff)
                * self.carbon_capture_rate
                * self.net_capacity_mw
                / 1000,
            ),
        )

    def variable_cost_profile(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        hours_in_year: int = 8760,
    ) -> tuple[NDArrayDate, NDArrayFloat]:
        """
        Calculate annual variable cost in kGBP.

        Parameters
        ----------
        load_factors: float or tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the
            date, the second the load factors.
        hours_in_year: int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        Tuple
            Two numpy arrays, first showing dates and the second showing
            variable costs in kGBP.

        Raises
        ------
        TypeError
            If the load_factors is neither a float nor a tuple.
        """
        if isinstance(load_factors, tuple):
            self.check_dates_tuple(load_factors)
            load_factor: Any = load_factors[1]
        elif isinstance(load_factors, float):
            load_factor = load_factors
        elif isinstance(load_factors, pd.Series):
            load_factor = self.pd_series_to_daterange_tuple(load_factors)
            load_factor = load_factor[1]
        else:
            raise TypeError()

        self.check_dates_tuple(self.availability)
        availability = self.availability[1]

        self.check_dates_tuple(self.variable_opex_gbp_hr)
        variable_cost = self.variable_opex_gbp_hr[1]

        return (
            self.date_range,
            np.full(
                self.lifetime,
                variable_cost * load_factor * availability * hours_in_year / 1000,
            ),
        )

    def fixed_cost_profile(
        self,
    ) -> tuple[NDArrayDate, NDArrayFloat]:
        """
        Calculate annual fixed cost in kGBP.

        Returns
        -------
        Tuple of NDArrayDate, NDArrayFloat

        """
        self.check_dates_tuple(self.fixed_opex_kgbp)
        fixed_cost = self.fixed_opex_kgbp[1]

        return (
            self.date_range,
            np.full(self.lifetime, fixed_cost),
        )

    def build_cashflows(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        fuel_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        carbon_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> pd.DataFrame:
        """
        Build a profile of annual cashflows for the Plant class.

        Parameters
        ----------
        load_factors: float or pandas.Series with datetime index
            Factor representing % of running in the year.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the
            date, the second the load factors.
        fuel_prices: float or pandas.Series with datetime index
            Factor representing cost of fuel in GBP/HHV MWh.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the fuel prices.
        carbon_prices: float or pandas.Series with datetime index
            Factor representing cost to emit carbon in GBP/te.  Can be either
            a single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the carbon prices.
        co2_transport_storage_cost: float
            Cost to transport and store carbon in GBP/te.
        hours_in_year: int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        pd.DataFrame
            A Pandas Dataframe indexed by year including cashflows
            for Capital, Fixed, Variable, Fuel and Carbon costs.
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
        capacity = (
            "capacity_mw",
            self.build_profile(self.net_capacity_mw, self.date_range[0], self.lifetime),
        )

        source = [capacity, production, capital, fixed, var, fuel, co2emit, co2_store]

        return pd.DataFrame(
            {name: (pd.Series(data[1], index=data[0])) for name, data in source},
        )

    def build_pv_cashflows(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        fuel_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        carbon_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> pd.DataFrame:
        """
        Build a profile of present value cashflows for the Plant class.

        Parameters
        ----------
        load_factors: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the
            date, the second the load factors.
        fuel_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost of fuel in GBP/HHV MWh.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the fuel prices.
        carbon_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost to emit carbon in GBP/te.  Can be either
            a single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the carbon prices.
        co2_transport_storage_cost: float
            Cost to transport and store carbon in GBP/te.
        hours_in_year: int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        pd.DataFrame
            A Pandas Dataframe indexed by year including cashflows
            for Capital, Fixed, Variable, Fuel and Carbon costs.
        """
        pvs = present_value_factor(self.cost_base, self.discount_rate)
        pvs_df = pd.DataFrame(pvs[1], index=pvs[0], columns=["discount_rate"])
        cf = self.build_cashflows(
            load_factors, fuel_prices, carbon_prices, co2_transport_storage_cost
        )
        cf.fillna(0, inplace=True)
        pv_cf = cf.multiply(
            pvs_df[pvs_df.index.isin(cf.index)]["discount_rate"], axis=0  # type: ignore
        )
        return pv_cf

    def calculate_lcoe(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        fuel_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        carbon_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> Tuple:
        """
        Calculate the Levelised Cost of Energy.

        Parameters
        ----------
        load_factors: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the
            date, the second the load factors.
        fuel_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost of fuel in GBP/HHV MWh.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the fuel prices.
        carbon_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost to emit carbon in GBP/te.  Can be either
            a single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the carbon prices.
        co2_transport_storage_cost: float
            Cost to transport and store carbon in GBP/te.
        hours_in_year: int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        Tuple
            Calculated Levelised Cost of Energy.
        """
        pv_cf = self.build_pv_cashflows(
            load_factors,
            fuel_prices,
            carbon_prices,
            co2_transport_storage_cost,
            hours_in_year,
        )
        srmc_df = pv_cf[
            [
                "variable_opex_kgbp",
                "fuel_kgbp",
                "carbon_emissions_kgbp",
                "carbon_storage_kgbp",
            ]
        ]
        lrmc_df = pv_cf[["capital_kgbp", "fixed_opex_kgbp"]]
        srmc = sum(srmc_df.stack().values) / sum(  # type: ignore
            pv_cf.production_GWth.values
        )
        srmc_cap = (  # type: ignore
            sum(srmc_df.stack().values)  # type: ignore
            / sum(pv_cf.capacity_mw.values)  # type: ignore
            / 1000
        )
        lrmc = sum(lrmc_df.stack().values) / sum(  # type: ignore
            pv_cf.production_GWth.values
        )
        lrmc_cap = (  # type: ignore
            sum(lrmc_df.stack().values)  # type: ignore
            / sum(pv_cf.capacity_mw.values)  # type: ignore
            / 1000
        )
        lcoe = srmc + lrmc
        lcoe_cap = srmc_cap + lrmc_cap
        full = (
            pv_cf.drop(["production_GWth", "capacity_mw"], axis=1).sum()
            / pv_cf.production_GWth.sum()
        )
        full_cap = (
            pv_cf.drop(["production_GWth", "capacity_mw"], axis=1).sum()
            / pv_cf.capacity_mw.sum()
            / 1000
        )

        LCONRG = namedtuple(
            "LCONRG",
            [
                "lcoe_gbp_mwh",
                "srmc_gbp_mwh",
                "lrmc_gbp_mwh",
                "capital_gbp_mwh",
                "fixed_opex_gbp_mwh",
                "variable_opex_gbp_mwh",
                "fuel_gbp_mwh",
                "carbon_emissions_gbp_mwh",
                "carbon_storage_gbp_mwh",
                "lcoe_gbp_kw",
                "srmc_gbp_kw",
                "lrmc_gbp_kw",
                "capital_gbp_kw",
                "fixed_opex_gbp_kw",
                "variable_opex_gbp_kw",
                "fuel_gbp_kw",
                "carbon_emissions_gbp_kw",
                "carbon_storage_gbp_kw",
            ],
        )
        return LCONRG(
            lcoe,
            srmc,
            lrmc,
            full.capital_kgbp,
            full.fixed_opex_kgbp,
            full.variable_opex_kgbp,
            full.fuel_kgbp,
            full.carbon_emissions_kgbp,
            full.carbon_storage_kgbp,
            lcoe_cap,
            srmc_cap,
            lrmc_cap,
            full_cap.capital_kgbp,
            full_cap.fixed_opex_kgbp,
            full_cap.variable_opex_kgbp,
            full_cap.fuel_kgbp,
            full_cap.carbon_emissions_kgbp,
            full_cap.carbon_storage_kgbp,
        )

    def calculate_annual_lcoe(
        self,
        load_factors: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        fuel_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        carbon_prices: Union[float, tuple[NDArrayDate, NDArrayFloat], pd.Series],
        co2_transport_storage_cost: float,
        hours_in_year: int = 8760,
    ) -> pd.Series:
        """
        Calculate an annual profile of Levelised Cost of Energy.

        Parameters
        ----------
        load_factors: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing % of running in the year.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the
            date, the second the load factors.
        fuel_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost of fuel in GBP/HHV MWh.  Can be either a
            single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the fuel prices.
        carbon_prices: float or Tuple of NDArrayDate, NDArrayFloat
            Factor representing cost to emit carbon in GBP/te.  Can be either
            a single figure which is applied to each year or a profile in the
            form of a Tuple of two numpy arrays, the first containing the date,
            the second the carbon prices.
        co2_transport_storage_cost: float
            Cost to transport and store carbon in GBP/te.
        hours_in_year: int
            Number of hours in a year. Defaults to 8760.

        Returns
        -------
        pd.DataFrame
            A profile of annual Levelised Cost of Energy.
        """
        pv_cf = self.build_pv_cashflows(
            load_factors, fuel_prices, carbon_prices, co2_transport_storage_cost
        )
        pv_cf["capital_annuity_kgbp"] = (
            sum(pv_cf.capital_kgbp) / sum(pv_cf.production_GWth)
        ) * pv_cf.production_GWth
        pv_cf.drop("capital_kgbp", axis=1, inplace=True)
        pv_profile = (
            pv_cf.drop(["production_GWth", "capacity_mw"], axis=1)
            .divide(pv_cf.production_GWth, axis=0)  # type: ignore
            .sum(axis=1)
        )
        return pv_profile

    def calculate_kw_cost(
        self,
    ) -> float:
        """Calculate the cost per kW of the plant."""
        discount_rate = (
            "discount_rate",
            present_value_factor(self.cost_base, self.discount_rate),
        )
        capital = (
            "capital_kgbp",
            self.build_profile(
                self.capital_cost, next(iter(self.capital_cost)), len(self.capital_cost)
            ),
        )
        fixed = ("fixed_opex_kgbp", self.fixed_cost_profile())
        capacity = (
            "capacity",
            self.build_profile(self.net_capacity_mw, self.cod, self.lifetime),
        )
        source = [capital, fixed, capacity, discount_rate]

        cf = pd.DataFrame(
            {name: (pd.Series(data[1], index=data[0])) for name, data in source},
        )
        cf.fillna(0, inplace=True)
        pv_cf = cf.multiply(cf["discount_rate"], axis=0)  # type: ignore
        return sum(pv_cf[["capital_kgbp", "fixed_opex_kgbp"]].sum()) / sum(
            pv_cf.capacity
        )

    def pd_series_to_daterange_tuple(
        self, series: pd.Series
    ) -> tuple[NDArrayDate, NDArrayFloat]:
        """
        Convert a pandas series to a tuple of numpy arrays.

        Parameters
        ----------
        series: pd.Series
            The pandas series to be converted.

        Returns
        -------
        Tuple
            A tuple of numpy arrays, the first containing the date and the
            second the values.
        """
        series = series[series.index.isin(self.date_range)]
        return (
            (series.index.to_numpy().astype("datetime64[Y]")),
            np.array(series.values),
        )


def present_value_factor(
    base_date: date,
    discount_rate: float,
    no_of_years: int = 50,
) -> tuple[NDArrayDate, NDArrayFloat]:
    """
    Create an annual discount factor profile.

    Parameters
    ----------
    base_date: date
        The base date for the calculation of Present Value factors.
    discount_rate: float
        The annual discount rate to use.
    no_of_years: int
        The number of years to be included. Defaults to 50.

    Returns
    -------
    Tuple
        A pair of numpy arrays, the first representing the date and the second
        the corresponding discount factor.
    """
    date_range: NDArrayDate = np.arange(
        base_date,
        np.datetime64(base_date, "Y") + np.timedelta64(no_of_years, "Y"),
        dtype="datetime64[Y]",
    )

    data: NDArrayFloat = np.full(
        no_of_years,
        1
        / ((1 + discount_rate) ** np.int32(date_range - np.datetime64(base_date, "Y"))),
    )

    return (date_range, data)
