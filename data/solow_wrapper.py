# ============================================================
# IMPORTS
# ============================================================

import os                               # add to read fred_api
import random                           # need to time delay
import time                             # need for pause
import pandas as pd                     # play with data frame
import requests                         # need for the HTTP World Bank API
from dotenv import load_dotenv          # load the key from fred from .env
from fredapi import Fred                # the fred API from FRED

# ============================================================
# SETTINGS
# ============================================================


load_dotenv()                            # load environment variables from .env
API_KEY = os.getenv("FRED_API_KEY")      # load the key for FRED API

if not API_KEY:                          # error message
    raise ValueError("FRED_API_KEY not found in .env file")

# define period over which we analyse data
START_YEAR = 2000
END_YEAR = 2015

# ============================================================
# DATA SOURCES
# ============================================================

# FRED series IDs per country and indicator
FRED_SERIES = {
    "ZAF": {"gdp": "MKTGDPZAA646NWDB", "population": "POPTOTZAA647NWDB", "population_growth": "SPPOPGROWZAF"},
    "KEN": {"gdp": "MKTGDPKEA646NWDB", "population": "POPTOTKEA647NWDB", "population_growth": "SPPOPGROWKEN"},
    "GHA": {"gdp": "MKTGDPGHA646NWDB", "population": "POPTOTGHA647NWDB", "population_growth": "SPPOPGROWGHA"},
    "FRA": {"gdp": "MKTGDPFRA646NWDB", "population": "POPTOTFRA647NWDB", "population_growth": "SPPOPGROWFRA"},
    "GBR": {"gdp": "MKTGDPGBA646NWDB", "population": "POPTOTGBA647NWDB", "population_growth": "SPPOPGROWGBR"},
    "ITA": {"gdp": "MKTGDPITA646NWDB", "population": "POPTOTITA647NWDB", "population_growth": "SPPOPGROWITA"},
    "KOR": {"gdp": "MKTGDPKRA646NWDB", "population": "POPTOTKRA647NWDB", "population_growth": "SPPOPGROWKOR"},
    "CHN": {"gdp": "MKTGDPCNA646NWDB", "population": "POPTOTCNA647NWDB", "population_growth": "SPPOPGROWCHN"},
    "JPN": {"gdp": "MKTGDPJPA646NWDB", "population": "POPTOTJPA647NWDB", "population_growth": "SPPOPGROWJPN"},
    "SGP": {"gdp": "MKTGDPSGA646NWDB", "population": "POPTOTSGA647NWDB", "population_growth": "SPPOPGROWSGP"},
    "BRA": {"gdp": "MKTGDPBRA646NWDB", "population": "POPTOTBRA647NWDB", "population_growth": "SPPOPGROWBRA"},
    "ARG": {"gdp": "MKTGDPARA646NWDB", "population": "POPTOTARA647NWDB", "population_growth": "SPPOPGROWARG"},
    "USA": {"gdp": "MKTGDPUSA646NWDB", "population": "POPTOTUSA647NWDB", "population_growth": "SPPOPGROWUSA"},
    "CAN": {"gdp": "MKTGDPCAA646NWDB", "population": "POPTOTCAA647NWDB", "population_growth": "SPPOPGROWCAN"},
}

# World Bank indicators (used where FRED is missing data)
WB_INDICATORS = {"savings_rate": "NY.GNS.ICTR.ZS", "labor_force": "SL.TLF.TOTL.IN",}

# Maps 3-letter country codes to 2-letter World Bank API codes
COUNTRY_WB_CODES = {
    "ZAF": "ZA", "KEN": "KE", "GHA": "GH",
    "FRA": "FR", "GBR": "GB", "ITA": "IT",
    "KOR": "KR", "CHN": "CN", "JPN": "JP",
    "SGP": "SG", "BRA": "BR", "ARG": "AR",
    "USA": "US", "CAN": "CA",
}

# Defines which source to use for each indicator
INDICATOR_SOURCES = {"gdp": "fred", "population": "fred", "population_growth": "fred", "savings_rate": "wb","labor_force": "wb",}

# ============================================================
# CONNECTIONS
# ============================================================

fred = Fred(api_key=API_KEY)        # FRED API client
session = requests.Session()        # reuse HTTP connection for efficiency (useful due to many request to the same server)

# ============================================================
# HELPERS
# ============================================================
# note that in the instruction set it is not stated that we need to define the function outside the Class thus we are not going to do so here (you will however find a few information inside the code)

# check if there are any gap in the data (expect to have all the year from 2000 - 2015)
def validate_years(df, country, indicator_key, start_year, end_year):
    # Check if any expected years are missing
    expected = set(range(start_year, end_year + 1))
    actual = set(df["year"])
    missing = expected - actual
    if missing:
        print(f"Warning: {country} {indicator_key} missing years {sorted(missing)}")

# ============================================================
# FRED FETCH
# ============================================================
# go and fetch the data from the FRED API

# go for a specific series number for a specific country between the time range set 
# (note that due to server issue (speed of the request) we had to add a process that slowed down the request and that retry if the data did not load the first time hence max_retries and base_delay)
def fetch_fred_indicator(series_id, country, indicator_key, start_year, end_year, max_retries=5, base_delay=2.0):

    for attempt in range(1, max_retries + 1):        # loops for the fives tries so if fails it retires
        try:
            # wait only after first failed attempt
            if attempt > 1:
                time.sleep(base_delay * attempt + random.uniform(0, 1))
                    # note that the time delay is massive because for some reason it still does not work

            # request data from FRED API
            data = fred.get_series(series_id, observation_start=f"{start_year}-01-01", observation_end=f"{end_year}-12-31",)

            # convert to DataFrame (get the data from each data and reomves na)
            df = pd.DataFrame({"year": data.index.year, "value": data.values,}).dropna()
            if df.empty:
                raise ValueError(f"No data for {series_id}")        # just value error if nothing

            # aggregate to yearly frequency (handles monthly/quarterly data)
            df = (df.groupby("year", as_index=False).mean().sort_values("year"))

            # combine the dataframe for countries and indicators so we know which row came from where 
            df["country"] = country
            df["indicator"] = indicator_key
            df["source"] = "fred"

            validate_years(df, country, indicator_key, start_year, end_year)     # run the check define earlier
            return df                                                            # return our dataframe

        
        except Exception as e:                                                   # just error message if we have failed more than 5 times to retrieve data so we know it does not work
            if attempt == max_retries:
                raise RuntimeError(f"FRED failed for {series_id}: {e}") from e

            print(f"Retry FRED {series_id}: {e}")                                # error message if needed to try again to download data

# ============================================================
# WORLD BANK FETCH
# ============================================================

# same system of function than for FRED but this time for the World Bank data
def fetch_world_bank_indicator(country, indicator_key, start_year, end_year, max_retries=5, base_delay=2.0):

    if country not in COUNTRY_WB_CODES:                                         # check that the country is in the database
        raise ValueError(f"No WB code for {country}")

    wb_code = COUNTRY_WB_CODES[country]                                         # go for each country like GBR -> GB and check for the indicator
    indicator = WB_INDICATORS[indicator_key]

    all_values = []                                                             # define so that the loop can use them afterward (start at page 1 for example)
    page = 1

    while True:                                                                 # in World Bank data come on each page so we need to go through them (page 1, page 2, etc...)
        # API request with pagination
        url = (
            f"https://api.worldbank.org/v2/country/{wb_code}/indicator/"
            f"{indicator}?format=json&per_page=200&page={page}"
        )

        for attempt in range(1, max_retries + 1):                               # this is the same loop as before (retry if loading fails)
            try:
                if attempt > 1:
                    time.sleep(base_delay * attempt + random.uniform(0, 1))
                        # massive gap of time in order to solve the issue of extracting the data (now time grows at each attemps )

                response = session.get(url, timeout=15)
                                                                                # handle the HTTP reponse
                if response.status_code == 200:                                 # success = 200 so it breaks
                    break
                elif response.status_code >= 500:                               # if longer then 500 means error so there was a problem
                    raise RuntimeError("Server error")
                else:
                    raise RuntimeError(f"Client error {response.status_code}")

            except Exception as e:                                              # same as with FRED exceed limit so it failed
                if attempt == max_retries:
                    raise RuntimeError(f"WB failed: {e}") from e

        # parse JSON response (issue with the World Bank return - need to turn it into python object)
        try:
            data = response.json()                              # json transform                                             
        except ValueError:
            raise RuntimeError("Invalid JSON from WB")          # if server does not work return error

        if not isinstance(data, list) or len(data) < 2:         # return the two element (page, entry)
            raise ValueError("Invalid WB response")

        all_values.extend(data[1])                              # finaly keep only the data entry ([0] is pagination)

        # stop when last page is reached
        if page >= data[0]["pages"]:                            # handle the page system of world bank and break once reach the last page
            break
        page += 1

    # filter relevant years in time range and remove missing values 
    values = []
    for entry in all_values:
        year = int(entry["date"])
        val = entry["value"]

        if start_year <= year <= end_year and val is not None:
            values.append({"year": year, "value": val})


    # build the dataframe (same logic than with FRED)
    df = pd.DataFrame(values)

    if df.empty:
        raise ValueError(f"No WB data for {country}, {indicator_key}")              

    df = df.sort_values("year")

    # add metadata
    df["country"] = country
    df["indicator"] = indicator_key
    df["source"] = "wb"

    validate_years(df, country, indicator_key, start_year, end_year)
    return df

# ============================================================
# UNIFIED FETCH
# ============================================================

# convenience so it fetch the data for the right country (can use GBR instead of the full list of FRED number)
def fetch_indicator(country, indicator_key, start_year, end_year):

    # check indicator validity
    if indicator_key not in INDICATOR_SOURCES:
        raise ValueError(f"Unknown indicator {indicator_key}")

    source = INDICATOR_SOURCES[indicator_key]

    # route request to correct data source (FRED or World Bank)
    if source == "fred":
        series_id = FRED_SERIES[country][indicator_key]
        return fetch_fred_indicator(series_id, country, indicator_key, start_year, end_year)

    elif source == "wb":
        return fetch_world_bank_indicator(country, indicator_key, start_year, end_year)

    else:
        raise ValueError("Invalid source")
    
