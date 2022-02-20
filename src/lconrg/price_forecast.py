"""Price Forecast Class."""
from typing import Optional


class SimplePriceForecast:
    """A class to hold price forecast information."""

    def __init__(
        self,
        commodity: str,
        base_year: int,
        price_forecast: Optional[dict] = None,
        flat_price: Optional[float] = None,
        start_yr: Optional[int] = None,
        end_yr: Optional[int] = None,
    ) -> None:
        """Creates new instance of Price Forecast object.

        Args:
            commodity (str): _description_
            base_year (int): _description_
            price_forecast (Optional[dict]): _description_
            flat_price (Optional[float]): _description_
            start_yr (Optional[int]): _description_
            end_yr (Optional[int]): _description_
        """
        self.commodity = commodity
        self.base_yr = base_year

        if price_forecast is not None:
            self.price_forecast = price_forecast
        else:
            if None in [flat_price, start_yr, end_yr]:
                raise AttributeError(
                    "You need to provide either a flat price, "
                    "start year and end year or a price forecast dict."
                )
            self.price_forecast = self.build_price_forecast(
                flat_price, start_yr, end_yr
            )

    def build_price_forecast(
        self, flat_price: float, start_yr: int, end_yr: int
    ) -> dict:
        """_summary_.

        Args:
            flat_price (float): _description_
            start_yr (int): _description_
            end_yr (int): _description_

        Returns:
            dict: _description_
        """
        return {year: flat_price for year in range(start_yr, end_yr + 1)}
