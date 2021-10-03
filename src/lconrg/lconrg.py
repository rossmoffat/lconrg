import pandas as pd


def present_value_factor(
    base_year: int, discount_rate: float, no_of_years: int
) -> dict:
    """Return dict of discount rates.

    Calculates annual discount rates and returns a dict.

    Args:
        base_year: The base year for the calculation.
        discount_rate: The percentage discount rate.
        no_of_years: The number of years from the base year to be calculated.

    Returns:
        Dict of Present Value discount factors from base year to end year.
    """
    return {
        year: 1 / ((1 + discount_rate) ** (year - base_year))
        for year in range(base_year, base_year + no_of_years)
    }


def fixed_costs_profile(load_factors: dict, fixed_opex_mgbp_yr: int) -> dict:
    """Return dict of annual fixed costs
    Calculates a fixed cost profile by applying an annual fixed costs
    to any year where the load factor is greater than zero.

    Args:
        load_factors: Dict of floats with index of year and value of
                      percentage 'load_factors' in the given year.
        fixed_opex_mgbp_yr: An integer representing the
                            Fixed Opex costs in mGBP per annum.

    Returns:
        Dict of fixed opex costs in each given year where the load factor is
        greater than zero.
    """
    return {year: (fixed_opex_mgbp_yr) for year, lf in load_factors.items() if lf > 0}


def fuel_costs_profile(
    gas_prices: dict,
    load_factors: dict,
    ng_flow_hhv: int,
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
        year: ng_flow_hhv * gas_prices[year] * hours_in_year * lf / 1000000
        for year, lf in load_factors.items()
    }


def carbon_costs_profile(
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


def energy_production_profile(
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
    Parameters
    --------------
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    hours_in_year : int, optional
        The number of hours in a year, default 8760.
    Returns
    --------------
    out : dict
        A Dict of variable costs for years where load factor is greater than zero.
    """
    return {
        year: (variable_opex_gbp_hr * lf * hours_in_year) / 1000000
        for year, lf in load_factors.items()
    }


def build_cashflow(
    capital_cost: dict,
    product_flow_kg: int,
    fixed_opex_mgbp_yr: int,
    ng_flow_hhv: int,
    ng_flow_kgh: int,
    carbon_capture_rate: float,
    variable_opex_gbp_hr: int,
    load_factors: dict,
    gas_prices: dict,
    carbon_prices: dict,
    co2_transport_storage_cost: float,
) -> pd.DataFrame:
    """
    Builds a cashflow profile for capital, fixed, fuel, carbon,
    variable costs and output
    Parameters
    --------------
    params : dict
        A dict containing the key data for the asset being calculated.
    capital_cost : dict
        A Dict containing keys of years and values of capital costs
    product_flow_kg : integer
        An integer representing the Product Flow Rate in kg/h.
    fixed_opex_mgbp_yr : integer
        An integer representing the Fixed Opex costs in mGBP per annum.
    ng_flow_hhv : integer
       An integer representing the NG Feed Flow Rate (HHV terms MWth).
    ng_flow_kgh : integer
        An integer representing the NG Feed Flow Rate (kg per hour)
    carbon_capture_rate: float
        A float representing the carbon capture rate.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    gas_prices : series
        A Pandas series of floats with index of year and value of
        prices in p/therm (HHV basis).
    carbon_prices : series
        A Pandas Series of floats with index of year and value of
        total price in GBP per tonne of CO2.
    co2_transport_storage_cost : float
        The fee in GBP per tonne of transport and storage of any CO2 captured.
        This price is used along with `carbon_prices` to calculate a volume-
        weighted average price of CO2 in GBP per tonne.

    Returns
    -------------
    out : dataframe
        A Pandas dataframe with index as year and values of
        cashflow elements and output profile.

    """
    return pd.DataFrame(
        [
            energy_production_profile(load_factors, product_flow_kg),
            capital_cost,
            fixed_costs_profile(load_factors, fixed_opex_mgbp_yr),
            fuel_costs_profile(gas_prices, load_factors, ng_flow_hhv),
            carbon_costs_profile(
                carbon_prices,
                load_factors,
                ng_flow_kgh,
                carbon_capture_rate,
                co2_transport_storage_cost=co2_transport_storage_cost,
            ),
            variable_costs_profile(load_factors, variable_opex_gbp_hr),
        ],
        index=[
            "h2_produced_therms_HHV",
            "capital_cost_mgbp",
            "fixed_cost_mgbp",
            "fuel_cost_mgbp",
            "carbon_cost_mgbp",
            "variable_cost_mgbp",
        ],
    ).T.sort_index()


def calculate_lcoh(
    capital_cost,
    product_flow_kg,
    fixed_opex_mgbp_yr,
    ng_flow_hhv,
    ng_flow_kgh,
    carbon_capture_rate,
    variable_opex_gbp_hr,
    load_factors,
    gas_prices,
    carbon_prices,
    base_year,
    discount_rate,
    co2_transport_storage_cost=19.0,
    **kwargs
):
    """
    Returns the present value Levelised Cost of Electricity in GBP per MWh.
    Parameters
    --------------
    capital_cost : dict
        A Dict containing keys of years and values of capital costs
    product_flow_kg : integer
        An integer representing the Product Flow Rate in kg/h.
    fixed_opex_mgbp_yr : integer
        An integer representing the Fixed Opex costs in mGBP per annum.
    ng_flow_hhv : integer
       An integer representing the NG Feed Flow Rate (HHV terms MWth).
    ng_flow_kgh : integer
        An integer representing the NG Feed Flow Rate (kg per hour)
    carbon_capture_rate: float
        A float representing the carbon capture rate.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    gas_prices : series
        A Pandas series of floats with index of year and value of
        prices in p/therm (HHV basis).
    carbon_prices : series
        A Pandas Series of floats with index of year and value of
        total price in GBP per tonne of CO2.
    base_year : int
        The base year for the calculation.
    discount_rate : float
        The percentage discount rate.
    co2_transport_storage_cost : float, optional
        The fee in GBP per tonne of transport and storage of any CO2 captured.
        This price is used along with `carbon_prices` to calculate a volume-
        weighted average price of CO2 in GBP per tonne.

    Returns
    -------------
    out : float
        The Present Value Levelised Cost of Electricity in GBP per therm

    """
    cf = build_cashflow(
        capital_cost,
        product_flow_kg,
        fixed_opex_mgbp_yr,
        ng_flow_hhv,
        ng_flow_kgh,
        carbon_capture_rate,
        variable_opex_gbp_hr,
        load_factors,
        gas_prices,
        carbon_prices,
        co2_transport_storage_cost=19.0,
    )
    pvs = pd.DataFrame(
        present_value_factor(
            base_year, discount_rate, (cf.index.max() - base_year + 1)
        ),
        index=["present_value_factor"],
    ).T
    pvcf = cf.multiply(pvs[pvs.index.isin(cf.index)].values, axis=0)
    pvcf = pvcf.sum(axis=0)
    lcoh = (
        pvcf[pvcf.index != "h2_produced_therms_HHV"].sum()
        / pvcf[pvcf.index == "h2_produced_therms_HHV"]
    )
    return lcoh * 1000000


def calculate_annual_lcoh(
    capital_cost,
    product_flow_kg,
    fixed_opex_mgbp_yr,
    ng_flow_hhv,
    ng_flow_kgh,
    carbon_capture_rate,
    variable_opex_gbp_hr,
    load_factors,
    gas_prices,
    carbon_prices,
    base_year,
    discount_rate,
    co2_transport_storage_cost=19.0,
    **kwargs
):
    """
    Returns the present value Levelised Cost of Electricity in GBP per MWh.
    Parameters
    --------------
    capital_cost : dict
        A Dict containing keys of years and values of capital costs
    product_flow_kg : integer
        An integer representing the Product Flow Rate in kg/h.
    fixed_opex_mgbp_yr : integer
        An integer representing the Fixed Opex costs in mGBP per annum.
    ng_flow_hhv : integer
       An integer representing the NG Feed Flow Rate (HHV terms MWth).
    ng_flow_kgh : integer
        An integer representing the NG Feed Flow Rate (kg per hour)
    carbon_capture_rate: float
        A float representing the carbon capture rate.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    gas_prices : series
        A Pandas series of floats with index of year and value of
        prices in p/therm (HHV basis).
    carbon_prices : series
        A Pandas Series of floats with index of year and value of
        total price in GBP per tonne of CO2.
    base_year : int
        The base year for the calculation.
    discount_rate : float
        The percentage discount rate.
    co2_transport_storage_cost : float, optional
        The fee in GBP per tonne of transport and storage of any CO2 captured.
        This price is used along with `carbon_prices` to calculate a volume-
        weighted average price of CO2 in GBP per tonne.

    Returns
    -------------
    out : float
        The Present Value Levelised Cost of Electricity in GBP per MWh

    """
    cf = build_cashflow(
        capital_cost,
        product_flow_kg,
        fixed_opex_mgbp_yr,
        ng_flow_hhv,
        ng_flow_kgh,
        carbon_capture_rate,
        variable_opex_gbp_hr,
        load_factors,
        gas_prices,
        carbon_prices,
        co2_transport_storage_cost,
    )
    pvs = pd.DataFrame(
        present_value_factor(
            base_year, discount_rate, (cf.index.max() - base_year + 1)
        ),
        index=["present_value_factor"],
    ).T
    pvcf = cf.multiply(pvs[pvs.index.isin(cf.index)].values, axis=0)
    pvcf = pvcf.sum(axis=0)
    capital_annuity = pvcf.capital_cost_mgbp / pvcf.h2_produced_therms_HHV
    lcoh = cf.drop("capital_cost_mgbp", axis=1)
    lcoh["capital_cost_annuity_mgbp"] = capital_annuity * lcoh.h2_produced_therms_HHV
    lcoh["h2_cost_p_th_hhv"] = (
        lcoh.drop("h2_produced_therms_HHV", axis=1).sum(axis=1)
        / lcoh.h2_produced_therms_HHV
    ) * 100000000
    return lcoh


def calculate_annual_lcoh_profile(
    capital_cost,
    product_flow_kg,
    fixed_opex_mgbp_yr,
    ng_flow_hhv,
    ng_flow_kgh,
    carbon_capture_rate,
    variable_opex_gbp_hr,
    load_factors,
    gas_prices,
    carbon_prices,
    base_year,
    discount_rate,
    operation_year,
    co2_transport_storage_cost=19.0,
    **kwargs
):
    """
    Returns the annual present value Levelised Cost of Hydrogen in p/therm for
    Load Factors between 5% and 100%.
    Parameters
    --------------
    capital_cost : dict
        A Dict containing keys of years and values of capital costs
    product_flow_kg : integer
        An integer representing the Product Flow Rate in kg/h.
    fixed_opex_mgbp_yr : integer
        An integer representing the Fixed Opex costs in mGBP per annum.
    ng_flow_hhv : integer
       An integer representing the NG Feed Flow Rate (HHV terms MWth).
    ng_flow_kgh : integer
        An integer representing the NG Feed Flow Rate (kg per hour)
    carbon_capture_rate: float
        A float representing the carbon capture rate.
    variable_opex_gbp_hr : integer
        An integer representing the Variable operating costs in GBP per hour.
    load_factors : series
        A Pandas series of floats with index of year and value of
        the percentage of running in the given year.
    gas_prices : series
        A Pandas series of floats with index of year and value of
        prices in p/therm (HHV basis).
    carbon_prices : series
        A Pandas Series of floats with index of year and value of
        total price in GBP per tonne of CO2.
    base_year : int
        The base year for the calculation.
    discount_rate : float
        The percentage discount rate.
    co2_transport_storage_cost : float, optional
        The fee in GBP per tonne of transport and storage of any CO2 captured.
        This price is used along with `carbon_prices` to calculate a volume-
        weighted average price of CO2 in GBP per tonne.

    Returns
    -------------
    out : array
        The Annual Present Value Levelised Cost of Hydrogen in pence per therm (HHV).

    """
    lcoh_array = []

    for x in range(5, 105, 5):
        load_factors = pd.DataFrame(
            [x / 100] * 25,
            index=range(operation_year, operation_year + 25),
            columns=["load_factor"],
        )
        lcoh = calculate_annual_lcoh(
            capital_cost,
            product_flow_kg,
            fixed_opex_mgbp_yr,
            ng_flow_hhv,
            ng_flow_kgh,
            carbon_capture_rate,
            variable_opex_gbp_hr,
            load_factors,
            gas_prices,
            carbon_prices,
            base_year,
            discount_rate,
            co2_transport_storage_cost=19.0,
        )
        lcoh_array.append(lcoh.h2_cost_p_th_hhv)

    return pd.DataFrame(lcoh_array, index=range(5, 105, 5)).T
