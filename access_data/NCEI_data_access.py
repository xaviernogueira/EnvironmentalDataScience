"""
Accesses NOAA's National Centers for Environmental Information
Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted
"""
import pandas as pd
import requests
from typing import Union, Tuple, List
from datetime import datetime
from access_data_utilities import prep_datetime_str, get_state_bbox, get_state_fips, \
    validate_date

# set environment variables
TOKEN = 'DSsSMpVbUnklkSjSzeAZFfaNsKsvommI'
STATE_NAME = 'pennsylvania'
START_TIME = '2016-01-01'
END_TIME = '2021-12-31'

# underlying NCEI data access functions
def get_ncei_dataset_codes(
    token: str,
    ) -> List[str]:
    datasets = []
    for d in list(requests.get(
        url='https://www.ncei.noaa.gov/cdo-web/api/v2/datasets',
        headers={'token': token},
        ).json()['results']):
        
        datasets.append(d['id'])
    return datasets

def get_stations_by_state(
    state_name: str,
    state_fips: str,
    date_range: Tuple[str, str],
    token: str,
    ncei_dataset_id: str,
    ) -> pd.DataFrame:

    # verify that the ncei_dataset_id is available
    dataset_ids = get_ncei_dataset_codes(token)
    if ncei_dataset_id not in dataset_ids:
        print(f'ERROR: param:ncei_dataset_id={ncei_dataset_id} is not a valid!\n'
        f'Please select from the following: {dataset_ids}')
        raise KeyError

    base_stations_url = r'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?limit=1000'
    base_stations_url = base_stations_url + f'&datasetid={ncei_dataset_id}'
    state_addition = f'&locationid=FIPS:{state_fips}'

    print(f'Fetching precipitation stations for state={state_name}')
    stations_list = list(requests.get(
        base_stations_url + state_addition,
        headers={'token': token},
        ).json()['results'])
        #TODO: debug this

    print(f'{len(stations_list)} precipitation stations total in {state_name}')
    stations_list = [i for i in stations_list if validate_date(i, date_range)]
    print(f'{len(stations_list)} precipitation stations with valid min/max date ranges')

    # convert to dataframe
    stations_df = pd.DataFrame.from_records(
        stations_list,
        index='id',
        )
    stations_df.index.name = 'station_id'
    del stations_list
    return stations_df

# main data access functions
def get_precipitation_data(
    state_name: str,
    start_time: Union[str, datetime],
    end_time: Union[str, datetime],
    token: str,
    ) -> pd.DataFrame:
    # format dates and statename
    state_name = state_name.lower()
    start_time = prep_datetime_str(start_time)
    end_time = prep_datetime_str(end_time)

    # get state bbox
    bbox_dict = get_state_bbox(state_name)

    # get state fips string code
    state_fips = get_state_fips()
    
    #[state_name]

    # get stations info as a dataframe
    print('Finding valid NCEI stations')
    stations_df = get_stations_by_state(
        state_name,
        state_fips,
        date_range=(start_time, end_time),
        token=token,
        ncei_dataset_id='GHCND',
        )

while __name__ == '__main__':
    get_precipitation_data(
    STATE_NAME,
    START_TIME,
    END_TIME,
    TOKEN,
    )