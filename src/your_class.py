# ============================================================
# Solow Growth Model Pipeline
#
# 1. Load macro data from DuckDB database
# 2. Clean + validate inputs
# 3. Calibrate model parameters (s, n, B, K0)
# 4. Compute steady state equilibrium
# 5. Analyze convergence (half-life)
# 6. Provide tables and plots for interpretation
# ============================================================

import duckdb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class SolowAnalyzer:
    """Toolkit for Solow open-economy growth analysis from a local database.

    The class connects to the DuckDB database and exposes methods to inspect 
    country-level macro indicators, calibrate Solow model parameters, compute open-economy
    steady states, and visualise convergence behaviour for a panel of 14 countries over
    2000-2015.

    The class is designed as a reusable toolkit: country codes, indicators,
    time windows, and counterfactual parameters are arguments rather than
    hard-coded values, so the same methods can support questions beyond the
    original analysis.

    Country codes and indicator keys are **case-insensitive**: ``"usa"``,
    ``"USA"``, and ``"Usa"`` are all equivalent, and the same applies to
    indicator names like ``"GDP"`` vs ``"gdp"``. Internally, country codes
    are normalised to uppercase and indicators to lowercase.

    Most of the *output* methods (plots, tables) accept several shapes for
    the country argument; ``None`` (use all available), a single string,
    or a list, and report bad input or unstable calibrations by printing
    a message and returning ``None`` rather than raising. The low-level
    *validation*, *data*, and *model* methods still raise exceptions so
    they remain composable.

    Parameters
    ----------
    db_path : str or pathlib.Path, optional
        Path to the DuckDB database file. If None, defaults to
        ``../data/fred_WB.db`` relative to this source file.
    alpha : float, default 0.33
        Capital share of output in the Cobb–Douglas production function.
        Must be strictly between 0 and 1.
    delta : float, default 0.05
        Annual depreciation rate of capital. Must be non-negative.
    g : float, default 0.02
        Long-run rate of technological progress. No range check.
    world_interest_rate : float, default 0.02
        Exogenous world real interest rate r used in the open-economy
        steady-state equations. Must be non-negative. Lowered from typical
        textbook values to keep ``s * r < n`` for every country in the sample.
    ky_ratio : float, default 3.0
        Capital-to-output ratio used to approximate the initial capital
        stock K0. Must be positive. Empirically, it tends to be [2.5, 4.0].

    Raises
    ------
    ValueError
        If ``alpha`` is not strictly between 0 and 1, if ``delta`` or
        ``world_interest_rate`` is negative, or if ``ky_ratio`` is not
        positive.
    FileNotFoundError
        If no DuckDB database can be opened at ``db_path``.

    Attributes
    ----------
    con : duckdb.DuckDBPyConnection
        Read-only connection to the database.
    """

    # The five indicators populated by data/build_data.py. Used by _validate_indicator to reject typos before they reach the database.
    _VALID_INDICATORS = {"gdp","population","population_growth","savings_rate","labor_force",}

    # note that the value selected for the different variable come directly from the Lecture of Macroeconomics III on the Solow model in open economy as they seems to match emprical data. We are thus going to use those as well.
    def __init__(self, db_path=None, alpha=0.33, delta=0.05, g=0.02, world_interest_rate=0.02, ky_ratio=3.0):
        """Initialise the analyser and open the database connection.

        Validates all structural parameters before storing them, then
        opens a read-only connection to the DuckDB database. The list of
        available country codes is loaded from the ``countries`` table at
        instantiation and cached in ``self._available_countries`` so that
        subsequent validation calls do not require additional database
        queries. All parameter validation happens at construction time so
        misconfigured instances are caught immediately rather than partway
        through an analysis pipeline.

        See the class-level docstring for the meaning of each parameter.

        Raises
        ------
        ValueError
            If ``alpha`` is not strictly between 0 and 1, if ``delta`` or
            ``world_interest_rate`` is negative, or if ``ky_ratio`` is not
            positive.
        FileNotFoundError
            If no DuckDB database can be opened at ``db_path``. The error
            message includes the exact path that was tried and instructions
            for rebuilding the database.
        """

        # check value make economic sens
        if not (0 < alpha < 1):
            raise ValueError(f"alpha must be strictly between 0 and 1 (got {alpha})")
        if ky_ratio <= 0:
            raise ValueError(f"ky_ratio must be positive (got {ky_ratio})")
        if delta < 0:
            raise ValueError(f"delta must be non-negative (got {delta})")
        if world_interest_rate < 0:
            raise ValueError(f"world_interest_rate must be non-negative (got {world_interest_rate})")

        self.alpha = alpha
        self.delta = delta
        self.g = g
        self.world_interest_rate = world_interest_rate
        self.ky_ratio = ky_ratio

        # define connection directly to the data folder
        if db_path is None:
            db_path = Path(__file__).resolve().parents[1] / "data" / "fred_WB.db"

        # make sure the connection works and handle error message ( note that we read_only as we are using database as source and we are not changing value in it)
        try:
            self.con = duckdb.connect(str(db_path), read_only=True)
        except Exception:
            raise FileNotFoundError(f"Database not found at '{db_path}'. "
                                    f"Rebuild it by running 'uv run python build_data.py' "
                                    f"from inside the data/ folder.")

        # simply keep the countires 
        self._available_countries = (self.con.execute("SELECT country_code FROM countries").fetchdf()["country_code"].tolist())




    # -------------------------------
    # VALIDATION
    # -------------------------------

    def _validate_country(self, country):
        """Normalise and check that a single country code exists in the database.

        Accepts a country code in any case (e.g. ``"usa"``, ``"USA"``,
        ``"Usa"``) and returns the uppercase version so callers can use
        it directly in downstream queries. Invalid codes are rejected
        with a clear message rather than producing a cryptic database
        error or a silent empty result.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive). Must be one of the codes
            present in the ``countries`` table of the database. Use
            ``get_available_countries()`` to see the full list of valid codes.

        Returns
        -------
        str
            The country code normalised to uppercase (e.g. ``"usa"`` becomes
            ``"USA"``), ready to be used directly in database queries or
            passed to other methods.

        Raises
        ------
        TypeError
            If ``country`` is not a string. For example, if an integer or
            list is passed by mistake.
        ValueError
            If the normalised code is not present in the ``countries`` table.
            The error message includes the full list of available codes so
            the user can identify and correct the typo immediately.
        """

        if not isinstance(country, str):                                            # check if country exist
            raise TypeError(
                f"country must be a string (got {type(country).__name__})"
            )
        country = country.upper()                                                   # convert everything into uppercase to avoid smaller issue
        if country not in self._available_countries:                                # if not available country sends an error 
            raise ValueError(f"Unknown country '{country}'. "
                             f"Available: {sorted(self._available_countries)}")
        return country

    def _validate_countries(self, countries):
        """Normalise and check that ``countries`` is a list of valid country codes.

        Extends ``_validate_country`` to handle multiple countries at once.
        Each element is upper-cased and validated individually against the
        ``countries`` table. The full normalised list is returned so callers
        can use it directly without further processing.

        Parameters
        ----------
        countries : list of str
            ISO-3 country codes (case-insensitive). Each element must be a
            string present in the ``countries`` table. Use
            ``get_available_countries()`` to see all valid codes.

        Returns
        -------
        list of str
            Country codes normalised to uppercase, in the same order as the
            input (e.g. ``["usa", "Ken"]`` becomes ``["USA", "KEN"]``).

        Raises
        ------
        TypeError
            If ``countries`` is not a list, the error message explicitly
            tells the user to wrap a single code in a list like ``["USA"]``.
            Also raised if any element of the list is not a string.
        ValueError
            If any element after normalisation is not present in the
            ``countries`` table. The error message includes the full list
            of available codes. Validation stops at the first invalid code.
        """
        # it does the same but for many countries (when we want to compare)
        if not isinstance(countries, list):
            raise TypeError(f"countries must be a list, got {type(countries).__name__}. "
                            f"For a single country, wrap it in a list like ['USA'].")
        return [self._validate_country(c) for c in countries]

    def _validate_indicator(self, indicator_key):
        """Normalise and check that an indicator key matches one of the stored series.

        Accepts an indicator name in any case (e.g. ``"GDP"``, ``"gdp"``,
        ``"Gdp"``) and returns the lowercase version so callers can use
        it directly in database queries without further processing.
        This method is called automatically by every method that accepts
        an ``indicator_key`` argument, so invalid or misspelled indicator
        names are caught early with a clear message rather than producing
        a silent empty result from the database.

        The five valid indicator keys are: ``"gdp"``, ``"population"``,
        ``"population_growth"``, ``"savings_rate"``, ``"labor_force"``.
        These correspond exactly to the series stored by the
        data-collection pipeline in ``data/build_data.py``.

        Parameters
        ----------
        indicator_key : str
            Indicator name to validate (case-insensitive). Must be one of
            the five series stored in the database.

        Returns
        -------
        str
            The indicator key normalised to lowercase, ready to be used directly in database
            queries.

        Raises
        ------
        TypeError
            If ``indicator_key`` is not a string. 
        ValueError
            If the normalised key is not one of the five valid options.
            The error message lists every valid indicator so the user
            can identify and correct the typo immediately.
        """
        # check the indicator
        if not isinstance(indicator_key, str):
            raise TypeError(
                f"indicator_key must be a string (got {type(indicator_key).__name__})"
            )
        indicator_key = indicator_key.lower()                                                   # make sure the are all in lower case
        if indicator_key not in self._VALID_INDICATORS:
            raise ValueError(
                f"Unknown indicator '{indicator_key}'. "
                f"Valid options: {sorted(self._VALID_INDICATORS)}"
            )
        return indicator_key

    # -------------------------------
    # DATA
    # -------------------------------

    # get country list
    def get_available_countries(self):
        """Return the list of country codes loaded in the database.

        This is the recommended first call after instantiating the class.
        All methods that accept a ``country`` or ``countries`` argument
        expect codes from this list. Passing any other value will be
        rejected with the full list of valid codes shown so it can be
        corrected.

        Returns
        -------
        list of str
            ISO-3 country codes present in the ``countries`` table, as
            returned by the database.
        """
        return self._available_countries

    def get_indicator(self, country, indicator_key, start_year=None, end_year=None):
        """Query a single indicator time series for a country from the database.

        The primary data access method; all output methods build on this
        internally. Can also be called directly to inspect raw data before
        running any model calculations.

        Country codes and indicator keys are case-insensitive.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive). Use
            ``get_available_countries()`` to see all valid codes.
        indicator_key : str
            One of ``"gdp"``, ``"population"``, ``"population_growth"``,
            ``"savings_rate"``, ``"labor_force"`` (case-insensitive).
        start_year : int, optional
            First year to include (inclusive). If None, no lower bound.
        end_year : int, optional
            Last year to include (inclusive). If None, no upper bound.

        Returns
        -------
        pandas.DataFrame
            Two columns ``year`` and ``value``, sorted by year.

        Raises
        ------
        TypeError
            If ``country`` or ``indicator_key`` is not a string.
        ValueError
            If the country code is not in the database, if
            ``indicator_key`` is not one of the five valid options, or
            if both ``start_year`` and ``end_year`` are given and
            ``end_year < start_year``.
        """

        # use already define function to validate both indicator and country
        country = self._validate_country(country)
        indicator_key = self._validate_indicator(indicator_key)

        # 
        query = """
        SELECT year, value
        FROM observations
        WHERE country_code = ? AND indicator_key = ?
        """
        params = [country, indicator_key]

        if start_year is not None and end_year is not None:                                      # handle error when issue with the year
            if end_year < start_year:
                raise ValueError(f"end_year ({end_year}) must be >= start_year ({start_year})")
            query += " AND year >= ? AND year <= ?"
            params += [start_year, end_year]
        elif start_year is not None:                                                             # user select one specific start and end year
            query += " AND year >= ?"
            params += [start_year]
        elif end_year is not None:
            query += " AND year <= ?"
            params += [end_year]

        query += " ORDER BY year"                                                                # order by year 

        return self.con.execute(query, params).fetchdf()

    # -------------------------------
    # HELPERS
    # -------------------------------

    def _get_average(self, country, indicator):
        """Return the time-average of an indicator series.

        Computes the arithmetic mean across all years available in the
        database for a given country and indicator. Used internally by
        ``build_country_parameters`` to compute the average savings rate
        and average population growth rate, which are the key inputs to
        the Solow steady-state calibration.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive).
        indicator : str
            Indicator key (e.g. ``"savings_rate"``, case-insensitive).

        Returns
        -------
        float
            Arithmetic mean of all available values for that country and
            indicator over the full sample period.

        Raises
        ------
        TypeError
            If ``country`` or ``indicator`` is not a string (propagated
            from ``get_indicator``).
        ValueError
            If the country code is not in the database, or if
            ``indicator`` is not one of the five valid options.
        """
        return self.get_indicator(country, indicator)["value"].mean()

    def _get_initial_value(self, country, indicator):
        """Return the earliest available value of an indicator.

        Sorts the indicator time series by year and returns the value
        from the first observation. Used internally by
        ``build_country_parameters`` to set the initial conditions for
        the Solow model calibration — specifically the initial GDP and
        initial labor force, from which the capital stock (via
        ``ky_ratio * gdp``) and total factor productivity ``B`` are
        derived.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive).
        indicator : str
            Indicator key (case-insensitive).

        Returns
        -------
        float
            Value from the earliest year available in the database for
            that country and indicator.
        """
        df = self.get_indicator(country, indicator)
        return df.sort_values("year").iloc[0]["value"]

    def _get_gdp_per_capita(self, country, start_year=None, end_year=None):
        """Compute GDP per capita time series for a country.

        Retrieves the ``gdp`` and ``population`` series separately from
        the database, merges them on ``year``, and computes the ratio.
        Years with no matching population observation are dropped from
        the merge. Used internally by ``plot_gdp_comparison`` and
        ``plot_convergence`` to produce the per-capita figures, and by
        ``plot_scenario`` to retrieve the most recent observed GDP per
        capita as the starting point for the simulation.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive).
        start_year : int, optional
            First year to include. If None, no lower bound is applied.
        end_year : int, optional
            Last year to include. If None, no upper bound is applied.

        Returns
        -------
        pandas.DataFrame
            Four columns: ``year``, ``gdp``, ``population``, and
            ``gdp_per_capita``, sorted by year.
        """
        # fetch data and rename colomn otherwise both have same name value.
        gdp = self.get_indicator(country, "gdp", start_year, end_year).rename(columns={"value": "gdp"})
        pop = self.get_indicator(country, "population", start_year, end_year).rename(columns={"value": "population"})
        # merge colomn and compute
        df = pd.merge(gdp, pop, on="year")
        df["gdp_per_capita"] = df["gdp"] / df["population"]
        return df

    # -------------------------------
    # MODEL
    # -------------------------------

    def _calculate_half_life(self, n):
        """Compute the convergence half-life toward the Solow steady state.

        Returns the number of years required to close half of the gap
        between the current GDP per capita and the steady-state value,
        based on the linearised convergence speed around the steady state.
        The convergence rate is given by:

            lambda = (1 - alpha) * (n + g + delta)

        A higher lambda means faster convergence and a shorter half-life.

        Parameters
        ----------
        n : float
            Population growth rate (decimal, e.g. 0.02 for 2%).

        Returns
        -------
        float or None
            Half-life in years, or ``None`` if ``lambda`` is non-positive,
            which would imply the model does not converge. In practice this
            only occurs when ``n + g + delta <= 0``, which should not arise
            with reasonable parameter values. A short message is also
            printed in the non-convergent case.
        """
        
        lambda_val = (1 - self.alpha) * (n + self.g + self.delta)             # apply formula
        if lambda_val <= 0: 
            print("No convergence the (nominator should be positive)")                                                  # no convergence (does not work mathematically)
            return None
        return np.log(2) / lambda_val

    def _compute_steady_state(self, s, n, B):
        """Solve the open-economy Solow steady state.

        Computes capital, output, wage, foreign assets and consumption
        per worker in the long run, given savings rate, population
        growth, and total factor productivity.

        Parameters
        ----------
        s : float
            Savings rate (decimal).
        n : float
            Population growth rate (decimal).
        B : float
            Calibrated total factor productivity.

        Returns
        -------
        dict
            Keys ``k_star``, ``y_star``, ``c_star`` — steady-state
            capital, output, and consumption per worker, each rounded
            to 3 decimals.

        Raises
        ------
        ValueError
            If ``s * r >= n`` (the model is dynamically unstable: foreign
            assets explode and no finite steady state exists).
        """
        r = self.world_interest_rate
        # special case define in the lecture: In open econonmy if the return of intereste rate and savings rate outpace the population growht there is no finite steady state
        if s * r >= n:
            raise ValueError(
                f"Steady state undefined: s*r = {s * r:.4f} >= n = {n:.4f}. "
                f"The open-economy Solow model has no finite steady state "
                f"when foreign-asset accumulation outpaces population growth. "
                f"Try a lower savings rate, a higher population growth rate, "
                f"or a lower world_interest_rate.")

        # compute the value of the model using formula
        k_star = ((self.alpha * B) / r) ** (1 / (1 - self.alpha))
        y_star = B * k_star**self.alpha
        w_star = (1 - self.alpha) * y_star
        v_star = (s * w_star) / (n - s * r)
        f_star = v_star - k_star
        yn_star = y_star + r * f_star
        c_star = (1 - s) * yn_star

        return {"k_star": round(k_star, 3),"y_star": round(y_star, 3),"c_star": round(c_star, 3),}

    def build_country_parameters(self, country):
        """Compute the full Solow parameter bundle for a country.

        The core computational method that all output methods build on.
        Executes the full calibration pipeline in four steps:

        1. Retrieves the average savings rate and population growth from
           the database, converting both from percent to decimal by
           dividing by 100.
        2. Retrieves initial GDP and labor force to approximate the
           initial capital stock (``K0 = ky_ratio * GDP``) and calibrate
           total factor productivity ``B`` from the Cobb-Douglas
           production function.
        3. Computes the open-economy steady-state values using
           ``_compute_steady_state``.
        4. Computes the convergence half-life using
           ``_calculate_half_life``.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive). Use
            ``get_available_countries()`` to see all valid codes.

        Returns
        -------
        dict
            Calibration bundle with the following keys (all numeric
            values are rounded for display, so the dict is intended for
            inspection and reporting rather than further high-precision
            computation):

            - ``country`` : str - normalised uppercase country code.
            - ``s`` : float - average savings rate as a decimal
              (e.g. ``0.17`` for 17%), rounded to 3 decimals.
            - ``n`` : float - average population growth rate as a
              decimal, rounded to 3 decimals.
            - ``B`` : float - calibrated total factor productivity,
              rounded to 3 decimals.
            - ``half_life`` : float or None - years to close half the
              gap to steady state, rounded to 2 decimals, or ``None``
              if the model does not converge.
            - ``k_star`` : float - steady-state capital per worker,
              rounded to 3 decimals.
            - ``y_star`` : float - steady-state GDP per capita,
              rounded to 3 decimals.
            - ``c_star`` : float - steady-state consumption per capita,
              rounded to 3 decimals.

        Raises
        ------
        TypeError
            If ``country`` is not a string.
        ValueError
            If the country is not in the database, or if the calibrated
            parameters yield an unstable model (``s * r >= n``).
        """
        country = self._validate_country(country)
        # get real data from database
        s = self._get_average(country, "savings_rate")/100
        n = self._get_average(country, "population_growth")/100
        gdp = self._get_initial_value(country, "gdp")
        labor = self._get_initial_value(country, "labor_force")
        # compute parameter
        K0 = self.ky_ratio * gdp
        B = gdp / ((K0 ** self.alpha) * (labor ** (1 - self.alpha)))
        # compute the steady state and half-life
        ss = self._compute_steady_state(s, n, B)
        half_life = self._calculate_half_life(n)

        return {"country": country,"s": round(s, 3),"n": round(n, 3),"B": round(B, 3),"half_life": round(half_life, 2) if half_life is not None else None,**ss}


    # -------------------------------
    # OUTPUTS / Method
    # -------------------------------

    def summary_table(self, country):
        """Print and return tidy summary tables for one or more countries.

        Each table has six rows: savings rate, population growth,
        productivity, steady-state GDP per capita, steady-state
        consumption, and convergence half-life. A separate table is
        printed for every requested country, and the dataframes are
        also returned so they can be embedded in notebooks or further
        processed.

        Parameters
        ----------
        country : str, list of str, or None
            What to summarise.

            - ``None`` - all countries in the database.
            - ``str`` - a single ISO-3 code (case-insensitive).
            - ``list of str`` - several codes.
        Returns
        -------
        pandas.DataFrame, list of pandas.DataFrame, or None

            - A single ``DataFrame`` if exactly one country was processed
              successfully.
            - A list of ``DataFrame`` objects if several were processed.
            - ``None`` if the input could not be interpreted, or if every
              requested country was either malformed or dynamically unstable.
        """

        # if no country is given, use all available
        if country is None:
            countries = self._available_countries
        elif isinstance(country, list):
            countries = country
        else:
            try:
                countries = [self._validate_country(country)]
            except (TypeError, ValueError) as e:
                print(f"Bad input:\n  {country}: {e}")
                return None
            
        # future outcome    
        bad_input = []
        unstable = [] 
        results = []
        # loop through any country
        for c in countries:
            try:
                params = self.build_country_parameters(c)
            except (TypeError, KeyError) as e:
                bad_input.append((c, str(e)))
                continue
            except ValueError as e:
                unstable.append((c, str(e)))
                continue  

            # return
            df = pd.DataFrame([("Savings rate", params["s"]), ("Population growth", params["n"]), ("Productivity", params["B"]), ("GDP per capita*", params["y_star"]), ("Consumption*", params["c_star"]), ("Half-life", params["half_life"])], columns=["Parameter", "Value"])
            print(f"\n--- {c} ---")
            print(df.to_string(index=False))
            print()
            results.append(df)
            
        # report problems (wrong input and stability condition)
        if bad_input:
            print("Bad input:")
            for c, reason in bad_input:
                print(f"  {c}: {reason}")
            return None
        
        if unstable:
            print("Unstable country:")
            for c, reason in unstable:
                print(f"  {c}: {reason}")
            return None
        
        return results if len(results) != 1 else results[0]
        
        

    def plot_gdp_comparison(self, countries=None, start_year=None, end_year=None, show=True):
        """Plot GDP per capita over time for one or more countries.

        Retrieves GDP and population from the database for each country,
        computes GDP per capita, and plots all series on a single figure.
        Useful for visualising income levels and growth trajectories
        across development groups, and for identifying major economic events.

        Parameters
        ----------
        countries : str, list of str, or None, optional
            What to plot.

            - ``None`` (default) - all countries in the database.
            - ``str`` - a single ISO-3 code (case-insensitive), wrapped
              into a one-element list internally.
            - ``list of str`` - several codes (case-insensitive).

            Use ``get_available_countries()`` to see all valid codes.
        start_year : int, optional
            First year to include. Must be in ``[2000, 2100]``. If None,
            no lower bound is applied.
        end_year : int, optional
            Last year to include. Must be in ``[2000, 2100]`` and
            ``>= start_year`` if both are given. If None, no upper
            bound is applied.
        show : bool, default True
            If True, call ``plt.show()`` after building the figure.
            Set to False to suppress display and only return the figure
            object — useful for saving to disk or embedding in reports.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object, returned so the caller can save or
            further customise it (e.g. ``fig.savefig("gdp.png")``).
            ``None`` is returned only when input validation fails
            before plotting (bad country argument, or an out-of-range
            or inconsistent year argument).

        Notes
        -----
        This method does not raise on bad input. Validation errors
        (invalid country argument, year outside ``[2000, 2100]``, or
        ``end_year < start_year``) are reported via print statements
        and ``None`` is returned. Per-country failures (missing data,
        invalid code in a list) are reported in a summary printed
        after the figure is shown, and the figure is still returned
        with whatever countries succeeded.
        """
        # plot the GDP per capita over time
        if countries is None:
            countries = self._available_countries
        elif isinstance(countries, list):
            countries = countries
        else:
            try:
                countries = [self._validate_country(countries)]
            except (TypeError, ValueError) as e:
                print(f"Bad input:\n  {countries}: {e}")
                return None                           # plot all the countries if none is specificaly selected

        # validate year range (make sure we do not have issue regarding the year (ploting empty graphs))
        if start_year is not None and end_year is not None and end_year < start_year:
            print(f"Error: end_year ({end_year}) must be >= start_year ({start_year})")
            return None
        if start_year is not None and (start_year < 2000 or start_year > 2100):
            print(f"Error: start_year ({start_year}) is outside reasonable range (2000–2100)")
            return None
        if end_year is not None and (end_year < 2000 or end_year > 2100):
            print(f"Error: end_year ({end_year}) is outside reasonable range (2000–2100)")
            return None
        # store the issue
        bad_input = []
        unstable = []
        # plot for the country
        fig, ax = plt.subplots()
        # loop through the countries
        for c in countries:
            try:
                df = self._get_gdp_per_capita(c, start_year, end_year)
                if df.empty:
                    bad_input.append((c, f"No data available for years {start_year}–{end_year}"))
                ax.plot(df["year"], df["gdp_per_capita"], label=c)
            except (TypeError, KeyError) as e:
                bad_input.append((c, str(e)))
                continue
            except ValueError as e:
                unstable.append((c, str(e)))
                continue

        ax.set_title("GDP per capita comparison")
        ax.set_xlabel("Year")
        ax.set_ylabel("GDP per capita")
        if ax.get_lines():
            ax.legend()
        ax.grid(True)

        if show:
            plt.show()

        if bad_input:
            print("\nBad input countries:")
            for c, reason in bad_input:
                print(f"  {c}: {reason}")

        if unstable:
            print("\nUnstable countries:")
            for c, reason in unstable:
                print(f"  {c}: {reason}")

        return fig


    def plot_convergence(self, countries=None, show=True):
        """Plot each country's convergence path toward its Solow steady state.

        For each country, computes the log deviation of actual GDP per
        capita from its predicted open-economy steady-state value y*
        and plots it against years since the start of the sample::

            log(y_t / y_star)

        A horizontal dashed line at zero marks the steady state. A
        value below zero means the country is still below its steady
        state and should be converging upward; a value above zero
        means it has exceeded its predicted steady state.

        This is the core output method for answering the research
        question: it directly shows whether countries are converging
        toward their predicted steady states, and how quickly.

        Parameters
        ----------
        countries : str, list of str, or None, optional
            What to plot.

            - ``None`` (default) - all countries in the database.
            - ``str`` - a single ISO-3 code (case-insensitive),
              wrapped into a one-element list internally.
            - ``list of str`` - several codes (case-insensitive).

            Use ``get_available_countries()`` to see all valid codes.
        show : bool, default True
            If True, call ``plt.show()`` after building the figure.
            Set to False to suppress display and only return the
            figure object.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object, returned so the caller can save or
            further customise it. ``None`` is returned only when the
            top-level country argument is rejected before any plotting
            happens.

        Notes
        -----
        This method does not raise on bad input. Per-country failures
        are silently absorbed and the loop continues:

        - Countries whose calibration is dynamically unstable are
          skipped and listed in an "Unstable countries" summary
          printed after the figure is shown.
        - Countries whose code is invalid or whose data is missing
          are skipped and listed in a "Bad input countries" summary.

        The figure is returned with whatever countries successfully
        produced a convergence path, even if some were skipped.
        """
        # plot convergence toward the steady state 
        # (use the same structure of code then the first method)
        if countries is None:
            countries = self._available_countries
        elif isinstance(countries, list):
            countries = countries
        else:
            try:
                countries = [self._validate_country(countries)]
            except (TypeError, ValueError) as e:
                print(f"Bad input:\n  {countries}: {e}")
                return None

        bad_input = []
        unstable = []

        fig, ax = plt.subplots()

        for c in countries:
            try:
                params = self.build_country_parameters(c)
                df = self._get_gdp_per_capita(c)
                df["log_dev"] = np.log(df["gdp_per_capita"] / params["y_star"])
                df["t"] = df["year"] - df["year"].min()
                ax.plot(df["t"], df["log_dev"], label=c)
            except (TypeError, KeyError) as e:
                bad_input.append((c, str(e)))
                continue
            except ValueError as e:
                unstable.append((c, str(e)))
                continue

        ax.axhline(0, linestyle="--")
        ax.set_title("Convergence")
        ax.set_xlabel("Years since start of sample")
        ax.set_ylabel(r"$\log(y_t / y^*)$")
        if ax.get_lines():
            ax.legend()
        ax.grid(True)

        if show:
            plt.show()

        if bad_input:
            print("\nBad input countries:")
            for c, reason in bad_input:
                print(f"  {c}: {reason}")

        if unstable:
            print("\nUnstable countries:")
            for c, reason in unstable:
                print(f"  {c}: {reason}")

        return fig

    def plot_scenario(self, country, new_s=None, new_n=None, new_r=None, new_g=None, new_delta=None, periods=50, show=True):
        """Compare baseline and counterfactual convergence paths.

        Starting from the country's most recent observed GDP per capita,
        simulates two trajectories: one toward the steady state implied
        by the current calibration, and one toward the steady state
        implied by user-supplied values for any subset of
        ``{s, n, r, g, delta}``. Parameters not overridden keep their
        baseline values.

        Parameters
        ----------
        country : str
            ISO-3 country code (case-insensitive).
        new_s : float, optional
            Counterfactual savings rate (decimal). Must be strictly
            between 0 and 1. If None, the country's own savings rate
            is used.
        new_n : float, optional
            Counterfactual population growth rate (decimal). Must be
            non-negative. If None, the country's own rate is used.
        new_r : float, optional
            Counterfactual world real interest rate (decimal). Must be
            non-negative. If None, the class-level ``world_interest_rate``
            is used.
        new_g : float, optional
            Counterfactual rate of technological progress. If None, the
            class-level ``g`` is used.
        new_delta : float, optional
            Counterfactual depreciation rate. If None, the class-level
            ``delta`` is used.
        periods : int, default 50
            Number of years to simulate forward.
        show : bool, default True
            If True, call ``plt.show()``.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure with two convergence paths (baseline and
            scenario). ``None`` if the country is unknown, the
            counterfactual inputs are invalid, or the chosen scenario
            is dynamically unstable.

        Notes
        -----
        This method does not raise on bad input. Validation errors
        (out-of-range counterfactual parameters, non-integer or
        non-positive ``periods``) and stability failures (``s * r >= n``)
        are reported via print statements and result in ``None`` being
        returned.

        The baseline path is computed with the class-level ``r``, ``g``,
        and ``delta``; only the scenario path uses the counterfactual
        structural parameters when supplied.
        """

        # validate country
        try:
            country = self._validate_country(country)
        except (TypeError, ValueError) as e:
            print(f"Bad input:\n  {country}: {e}")
            return None

    # check for wrong input
        if new_s is not None and not (0 < new_s < 1):
            print(f"Error: new_s must be strictly between 0 and 1 (got {new_s})")
            return None
        if new_n is not None and new_n < 0:
            print(f"Error: new_n must be non-negative (got {new_n})")
            return None
        if new_r is not None and new_r < 0:
            print(f"Error: new_r must be non-negative (got {new_r})")
            return None
        if not isinstance(periods, int) or periods <= 0:
            print(f"Error: periods must be a positive integer (got {periods!r})")
            return None

        # build parameters
        try:
            params = self.build_country_parameters(country)
        except ValueError as e:
            print(f"Unstable country:\n  {country}: {e}")
            return None

        s = new_s if new_s is not None else params["s"]
        n = new_n if new_n is not None else params["n"]
        B = params["B"]

        # structural parameters — use current defaults or counterfactual
        r_orig = self.world_interest_rate
        g_orig = self.g
        delta_orig = self.delta

        r_new = new_r if new_r is not None else r_orig
        g_new = new_g if new_g is not None else g_orig
        delta_new = new_delta if new_delta is not None else delta_orig

    # compute steady states with different interest rates
        try:
        # original steady state uses class-level r
            original_ss = self._compute_steady_state(params["s"], params["n"], B)

        # scenario steady state — compute manually if r changed
            k_star_new = ((self.alpha * B) / r_new) ** (1 / (1 - self.alpha))
            y_star_new = B * k_star_new ** self.alpha
            w_star_new = (1 - self.alpha) * y_star_new
            if s * r_new >= n:
                print(f"Scenario not possible: s*r = {s * r_new:.4f} >= n = {n:.4f}")
                return None
            v_star_new = (s * w_star_new) / (n - s * r_new)
            f_star_new = v_star_new - k_star_new
            yn_star_new = y_star_new + r_new * f_star_new
            c_star_new = (1 - s) * yn_star_new
            scenario_ss = {"y_star": round(y_star_new, 3), "c_star": round(c_star_new, 3)}
        except ValueError as e:
            print(f"Scenario not possible:\n  {e}")
            return None

        current_y = self._get_gdp_per_capita(country)["gdp_per_capita"].iloc[-1]
        years = np.arange(periods)

        lambda_orig = (1 - self.alpha) * (params["n"] + g_orig + delta_orig)
        lambda_new = (1 - self.alpha) * (n + g_new + delta_new)

        path_orig = original_ss["y_star"] + (current_y - original_ss["y_star"]) * np.exp(-lambda_orig * years)
        path_new = scenario_ss["y_star"] + (current_y - scenario_ss["y_star"]) * np.exp(-lambda_new * years)

        fig, ax = plt.subplots()
        ax.plot(years, path_orig, label="Current")
        ax.plot(years, path_new, linestyle="--", label="Scenario")

        ax.set_title(f"Scenario: {country}")
        ax.set_xlabel("Years from now")
        ax.set_ylabel("GDP per capita")
        ax.legend()
        ax.grid(True)

        if show:
            plt.show()

        return fig
    



    def compare_steady_states(self, countries=None, show=True):
        """Bar chart comparing steady-state GDP per capita across countries.

        For each country whose calibration is stable, computes the Solow
        open-economy steady-state GDP per capita (y*) and displays it as
        a vertical bar. Bars appear in the order the countries were
        provided.

        This method is useful for comparing the long-run income levels
        predicted by the Solow model across development groups, and for
        identifying outliers such as China whose calibrated y* diverges
        significantly from actual income levels.

        Parameters
        ----------
        countries : str, list of str, or None, optional
            What to compare.

            - ``None`` (default) - all countries in the database.
            - ``str`` - a single ISO-3 code (case-insensitive),
              wrapped into a one-element list internally.
            - ``list of str`` - several codes (case-insensitive).

            Use ``get_available_countries()`` to see all valid codes.
        show : bool, default True
            If True, call ``plt.show()`` after building the figure.
            Set to False to suppress display and only return the figure
            object.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object, returned so the caller can save or
            further customise it. ``None`` is returned if the top-level
            country argument is rejected, or if no country in the request
            produced a stable calibration (a "No stable countries to
            plot." message is printed in that case).

        Notes
        -----
        This method does not raise on bad input. Per-country failures
        are silently absorbed:

        - Countries whose calibration is dynamically unstable
          (``s * r >= n``) are skipped and listed in an "Unstable
          countries" summary printed after the figure is shown.
        - Countries whose code is invalid are skipped and listed in
          a "Bad input countries" summary.

        The figure is returned (with whatever stable countries
        produced bars), unless *every* requested country failed —
        in which case ``None`` is returned instead.
        """
        if countries is None:
            countries = self._available_countries
        elif isinstance(countries, list):
            countries = countries
        else:
            try:
                countries = [self._validate_country(countries)]
            except (TypeError, ValueError) as e:
                print(f"Bad input:\n  {countries}: {e}")
                return None

        bad_input = []
        unstable = []
        values = []
        valid_countries = []

        for c in countries:
            try:
                params = self.build_country_parameters(c)
                values.append(params["y_star"])
                valid_countries.append(c)
            except (TypeError, KeyError) as e:
                bad_input.append((c, str(e)))
                continue
            except ValueError as e:
                unstable.append((c, str(e)))
                continue

        if not values:
            print("No stable countries to plot.")
            return None

        fig, ax = plt.subplots()
        ax.bar(valid_countries, values)
        ax.set_title("Steady-state comparison")
        ax.set_ylabel("Steady-state GDP per capita")
        ax.grid(True)

        if show:
            plt.show()

        if bad_input:
            print("\nBad input countries:")
            for c, reason in bad_input:
                print(f"  {c}: {reason}")

        if unstable:
            print("\nUnstable countries:")
            for c, reason in unstable:
                print(f"  {c}: {reason}")
        return fig

    def plot_indicator(self, countries, indicator_key, start_year=None, end_year=None, show=True):
        """Plot a raw indicator time series for one or more countries.

        General-purpose plotting tool that retrieves any of the five
        indicators stored in the database and overlays them on a single
        figure. No transformation is
        applied. The raw values from the database are plotted directly.
        Useful for inspecting the underlying data before running model
        calculations, or for comparing indicator trends across
        development groups.

        Parameters
        ----------
        countries : str, list of str, or None
            What to plot.

            - ``str`` - a single ISO-3 code (case-insensitive),
              wrapped into a one-element list internally.
            - ``list of str`` - several codes (case-insensitive).
            - ``None`` - all countries in the database. Must be passed
              explicitly as ``countries=None`` since this argument has
              no default value in the signature.

            Use ``get_available_countries()`` to see all valid codes.
        indicator_key : str
            Indicator to plot (case-insensitive). Must be one of:
            ``"gdp"``, ``"population"``, ``"population_growth"``,
            ``"savings_rate"``, ``"labor_force"``.
        start_year : int, optional
            First year to include. If None, no lower bound is applied.
        end_year : int, optional
            Last year to include. Must be ``>= start_year`` if both
            are given. If None, no upper bound is applied.
        show : bool, default True
            If True, call ``plt.show()`` after building the figure.
            Set to False to suppress display and only return the
            figure object.

        Returns
        -------
        matplotlib.figure.Figure or None
            The figure object, returned so the caller can save or
            further customise it. ``None`` is returned only when input
            validation fails before plotting (invalid indicator, bad
            country argument, or inconsistent year range).

        Notes
        -----
        This method does not raise on bad input. Per-country failures
        (missing data for the requested window, query errors) are
        silently absorbed and listed in a "Bad input countries" or
        "Unstable countries" summary printed after the figure is shown.
        The figure is returned with whatever countries successfully
        produced a line.
        """
        # validate indicator first
        try:
            indicator_key = self._validate_indicator(indicator_key)
        except (TypeError, ValueError) as e:
            print(f"Bad indicator:\n  {indicator_key}: {e}")
            return None

        # normalize countries
        if countries is None:
            countries = self._available_countries
        elif isinstance(countries, list):
            countries = countries
        else:
            try:
                countries = [self._validate_country(countries)]
            except (TypeError, ValueError) as e:
                print(f"Bad input:\n  {countries}: {e}")
                return None

        # validate year range
        if start_year is not None and end_year is not None and end_year < start_year:
            print(f"Error: end_year ({end_year}) must be >= start_year ({start_year})")
            return None

        bad_input = []
        unstable = []

        fig, ax = plt.subplots()

        for c in countries:
            try:
                df = self.get_indicator(c, indicator_key, start_year, end_year)
                if df.empty:
                    bad_input.append((c, f"No data available for years {start_year}–{end_year}"))
                    continue
                ax.plot(df["year"], df["value"], label=c)
            except (TypeError, KeyError) as e:
                bad_input.append((c, str(e)))
                continue
            except ValueError as e:
                unstable.append((c, str(e)))
                continue

        ax.set_title(f"{indicator_key}")
        ax.set_xlabel("Year")
        ax.set_ylabel(indicator_key.replace("_", " ").capitalize())
        if ax.get_lines():
            ax.legend()
        ax.grid(True)

        if show:
            plt.show()

        if bad_input:
            print("\nBad input countries:")
            for c, reason in bad_input:
                print(f"  {c}: {reason}")

        if unstable:
            print("\nUnstable countries:")
            for c, reason in unstable:
                print(f"  {c}: {reason}")

        return fig
