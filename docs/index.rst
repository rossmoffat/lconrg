The LCONRG Project
==============================

.. toctree::
    :hidden:
    :maxdepth: 1

    license
    reference

A Project which calculates the Levelised Cost Of eNeRGy
for a given set of input parameters.  The Levelised Cost is
the cost of energy over the lifetime of the plant, and is
calculated by dividing the total cost of the plant by the
total energy produced.  Cash flows are discounted using
a specified discount rate and reflects the time value of
money.

.. _installation:

Installation
------------

To install the LCONRG project,
first install it using pip:

.. code-block:: console

   $ pip install lconrg

Usage
-----

LCONRG usage is based on the ``Plant`` class.  The ``Plant`` class
holds all the information about the plant, and includes methods to
calculate the LCOE.

To create a new ``Plant`` object, use the ``Plant`` class.  The
parameters can be added directly, but readabilty is improved by
creating a dictionary of parameters and passing that to the
``Plant`` class.

  .. code-block:: python

     import datetime
     import lconrg.lconrg as lconrg

     data = {fuel="gas",
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
             carbon_capture_rate=0.95
             }

     plant = lconrg.Plant(**data)


To calculate an LCOE, use the ``calculate_lcoe`` method.  This
method takes additional parameters including ``load_factors``,
``fuel_prices``, ``carbon_prices`` and ``co2_transport_storage_cost``.

.. autofunction:: lconrg.lconrg.Plant.calculate_lcoe
