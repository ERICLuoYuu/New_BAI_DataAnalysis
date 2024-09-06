# -*- coding: utf-8 -*-
# Copyright (C) 2018-2021, earthobservations developers.
# Distributed under the MIT License. See LICENSE for more info.
"""
=====
About
=====
Acquire station information from DWD.

"""  # Noqa:D205,D400
import logging
from datetime import datetime
import polars as pl

from wetterdienst.provider.dwd.observation import (
    DwdObservationDataset,
    DwdObservationPeriod,
    DwdObservationRequest,
    DwdObservationResolution,
)

log = logging.getLogger()


def station_example():
    """Retrieve stations_result of DWD that measure temperature."""
    stations = DwdObservationRequest(
        parameter=[DwdObservationDataset.TEMPERATURE_AIR, DwdObservationDataset.TEMPERATURE_EXTREME, DwdObservationDataset.PRECIPITATION, DwdObservationDataset.PRESSURE, DwdObservationDataset.SUN, DwdObservationDataset.WIND, DwdObservationDataset.WIND_EXTREME,DwdObservationDataset.SOLAR, DwdObservationDataset.DEW_POINT, DwdObservationDataset.MOISTURE],
        resolution=DwdObservationResolution.MINUTE_10,
        period=DwdObservationPeriod.HISTORICAL,
        start_date=datetime(1996, 6, 14),
        end_date=datetime(2023, 10, 16),
    )

    result = stations.filter_by_station_id(7374)

    return result.df, result.values.all().df


def main():
    """Run example."""
    logging.basicConfig(level=logging.INFO)
    station_info, values = station_example()
    
    # Print unique parameters
    parameters = values["parameter"].unique()
    print("Available parameters:")
    for param in parameters:
        print(f"- {param}")

    # Print a sample of the raw data
    print("\nSample of raw data:")
    print(values.head(10))

    # Create a dictionary to store our data
    data_dict = {}

    # List of all parameters we're trying to extract
    # Automatically create param_mapping for all available parameters
    param_mapping = {
    "precipitation_duration": "precipitation_duration",
    "wind_direction": "wind_direction",
    "wind_speed": "wind_speed",
    "temperature_air_mean_0_05m": "tair_5cm_mean",
    "temperature_air_min_2m": "tair_2m_min",
    "wind_direction_gust_max": "wind_direction_gust_max",
    "temperature_air_mean_2m": "tair_2m_mean",
    "precipitation_index": "precipitation_index",
    "radiation_sky_long_wave": "LWIN",
    "wind_speed_min": "wind_speed_min",
    "temperature_dew_point_mean_2m": "tair_2m_dp_mean",
    "pressure_air_site": "pressure_air",
    "precipitation_height": "precipitation",
    "temperature_air_max_0_05m": "tair_5cm_max",
    "radiation_sky_short_wave_diffuse": "SWIN_diffuse",
    "temperature_air_max_2m": "tair_2m_max",
    "radiation_global": "SWIN",
    "humidity": "rH",
    "wind_speed_rolling_mean_max": "wind_speed_rolling_mean_max",
    "temperature_air_min_0_05m": "tair_5cm_min",
    "wind_gust_max": "wind_gust_max",
    "sunshine_duration": "sunshine_duration"
}

    # Populate the dictionary, checking for missing data
    for dwd_param, df_column in param_mapping.items():
        filtered_data = values.filter(pl.col("parameter") == dwd_param)["value"]
        if len(filtered_data) > 0:
            data_dict[df_column] = filtered_data
            print(f"Data found for {dwd_param} (Column: {df_column}). Length: {len(filtered_data)}")
        else:
            print(f"Warning: No data found for parameter '{dwd_param}' (Column: {df_column})")

    # Add the data_time column
    data_dict["data_time"] = values["date"].unique()
    print(f"data_time length: {len(data_dict['data_time'])}")

    # Print lengths of all series
    for key, value in data_dict.items():
        print(f"Length of {key}: {len(value)}")

    # Create the DataFrame only with available data
    try:
        df = pl.DataFrame(data_dict)
        print("DataFrame created successfully.")
        print("DataFrame shape:", df.shape)
        print("DataFrame columns:", df.columns)
        df.write_parquet("./ahaus_data_1996_2023.parquet")
        print("Data written to parquet file successfully.")
    except Exception as e:
        print(f"Error creating DataFrame: {str(e)}")

if __name__ == "__main__":
    main()