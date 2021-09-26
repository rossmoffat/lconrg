def present_value(base_year: int, discount_rate: float, no_of_years: int) -> dict:
    """
    Calculates a list of discount rates by year
    Parameters
    --------------
    base_year : int
        The base year for the calculation.
    discount_rate : float
        The percentage discount rate.
    no_of_years : int
        The number of years from the base year to be calculated.
    Returns
    -------------
    out : dict
        Dict of Present Value discount factors from base year to end year.
    """

    return {
        i: 1 / ((1 + discount_rate) ** (i - base_year))
        for i in range(base_year, base_year + no_of_years)
    }
