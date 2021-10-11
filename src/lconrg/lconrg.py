"""Levelised Cost of eNeRGy."""


def present_value_factor(
    base_year: int, discount_rate: float, no_of_years: int = 50
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
    """Return dict of annual fixed costs.

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
    fuel_cost = fuel_costs_profile(gas_prices, load_factors, fuel_flow_hhv)
    carbon_cost = carbon_costs_profile(
        carbon_prices,
        load_factors,
        fuel_flow_kgh,
        carbon_capture_rate,
        carbon_fraction,
        co2_transport_storage_cost,
    )
    variable_cost = variable_costs_profile(load_factors, variable_opex_gbp_hr)
    production_profile = energy_production_profile(load_factors, energy_output)
    cost_profile = {
        year: fuel_cost[year] + carbon_cost[year] + variable_cost[year]
        for year in fuel_cost
    }
    cost = {year: cost_profile[year] * pvs[year] for year in cost_profile}
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
) -> int:
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
    production_profile = energy_production_profile(load_factors, energy_output)
    cost = {
        year: (capital_cost[year] + fixed_opex_mgbp_yr[year]) * pvs[year]
        for year in pvs
    }
    production = {
        year: production_profile[year] * pvs[year] for year in production_profile
    }
    return sum(cost.values()) / sum(production.values()) * 1000000
