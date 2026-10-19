# ============================================================
# IMPORTS
# ============================================================

import duckdb
from pathlib import Path
from solow_wrapper import fetch_indicator, FRED_SERIES, INDICATOR_SOURCES, START_YEAR , END_YEAR



# ============================================
# Code 
# ============================================

# just make sure the code is exectuted entierly when run as main (if the file is imported then it skip the code)
if __name__ == "__main__":              

    # ============================================================
    # SETTINGS
    # ============================================================

    # list of countries from wrapper
    COUNTRIES = list(FRED_SERIES.keys())

    # database path (stored in /data folder)
    DB_PATH = Path(__file__).resolve().parent / "fred_WB.db"

    # ============================================================
    # CREATE DATABASE
    # ============================================================

    # connect to DuckDB database
    db = duckdb.connect(str(DB_PATH))

    # SQL upperletter to drop the table
    # rebuild tables each time (clean state)
    db.execute("DROP TABLE IF EXISTS observations")
    db.execute("DROP TABLE IF EXISTS indicators")
    db.execute("DROP TABLE IF EXISTS countries")

    # create tables
    # SQL create the table db.execute()
    db.execute("CREATE TABLE countries (country_code VARCHAR PRIMARY KEY)")
    db.execute("CREATE TABLE indicators (indicator_key VARCHAR PRIMARY KEY, description VARCHAR)")
    db.execute("CREATE TABLE observations (country_code VARCHAR, indicator_key VARCHAR, year INTEGER, value DOUBLE, PRIMARY KEY (country_code, indicator_key, year), FOREIGN KEY (country_code) REFERENCES countries(country_code), FOREIGN KEY (indicator_key) REFERENCES indicators(indicator_key))")

    # ============================================================
    # INDICATOR DESCRIPTIONS
    # ============================================================

    # dictionary of the indicator
    DESCRIPTIONS = {"gdp": "GDP, current US dollars","population": "Total population", "population_growth": "Population growth, annual %","savings_rate": "Gross savings, % of GDP","labor_force": "Labor force, total",}
    # insert indicator metadata (? placeholder, even if value call without description return the name twice due to DESCRIPTIONS.get(key,key))
    for key in INDICATOR_SOURCES:
        db.execute("INSERT INTO indicators VALUES (?, ?)", [key, DESCRIPTIONS.get(key, key)])

    # ============================================================
    # FETCH AND STORE DATA
    # ============================================================

    for country in COUNTRIES:
        print(f"\nFetching {country}...")

        # insert country into table
        db.execute(
            """
            INSERT INTO countries VALUES (?)
            """,
            [country]
        )

        for key in INDICATOR_SOURCES:                                       # goes through every indicatory for the country
            try:
                # fetch data from wrapper
                df = fetch_indicator(country, key, START_YEAR, END_YEAR)    # return dataframe with both data from the wrapper FRED and WB

                # insert each observation                                   
                for _, row in df.iterrows():                                # inserte datapoint into table (if two value are the same the new one overwrite the older one)
                    db.execute(
                        """
                        INSERT OR REPLACE INTO observations
                        VALUES (?, ?, ?, ?)
                        """,
                        [country, key, int(row["year"]), float(row["value"]),]
                    )

                print(f"  ✓ {key}: {len(df)} observations")                 # print the result

            except Exception as e:                                          # error message
                print(f"  ✗ {key}: {e}")

    # close connection
    db.close()

    print(f"\nDatabase built: {DB_PATH}")                                   # simply return the link of the dataframe (FRED_WB.db)
