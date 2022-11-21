"""
Accesses NOAA's National Centers for Environmental Information
Documentation: https://www.ncdc.noaa.gov/cdo-web/webservices/v2#gettingStarted
"""
import pandas as pd
import requests
import gc
from typing import Union, Tuple, List
from datetime import datetime
from access_data_utilities import prep_datetime_str, get_state_bbox, get_state_fips, \
    validate_date

# set environment variables
TOKEN = 'DSsSMpVbUnklkSjSzeAZFfaNsKsvommI'
STATE_NAME = 'pennsylvania'
START_TIME = '2016-01-01'
END_TIME = '2021-12-31'

# use a class to limit API calls for url verifications
class UrlChecker:
    def __init__(self, token: str) -> None:
        self.token = token
        self._dataset_ids = None
    
    @property
    def dataset_ids(self) -> List[str]:
        if self._dataset_ids is None:
            # verify that the ncei_dataset_id is available
            self._dataset_ids = get_ncei_dataset_codes(
                token=self.token,
                )
        return self._dataset_ids

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
    url_checker: UrlChecker,
    base_stations_url: str = None,
    ) -> pd.DataFrame:

    # verify that the ncei_dataset_id is available
    if ncei_dataset_id not in url_checker.dataset_ids:
        print(f'ERROR: param:ncei_dataset_id={ncei_dataset_id} is not a valid!\n'
        f'Please select from the following: {url_checker.dataset_ids}')
        raise KeyError

    if not base_stations_url: base_stations_url = r'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?limit=1000'
    request_url = base_stations_url + f'&datasetid={ncei_dataset_id}'

    print(f'Fetching precipitation stations for state={state_name}')
   # stations_list = list(
    stations_list = requests.get(
        base_stations_url + request_url + f'&locationid=FIPS:{state_fips}',
        headers={'token': token},
        ) #.json() #['results'])
    #TODO: come back, was working a second ago, getting 500 API error.
    stations_list = [i for i in stations_list if validate_date(i, date_range)]
    print(f'{len(stations_list)} precipitation stations with valid min/max date ranges\n Done')

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
    dataset_id: str = 'GHCND',
    datatype_id: str = 'PRCP',
    ) -> pd.DataFrame:

    # format dates and statename
    state_name = state_name.lower()
    start_str, start_dt = prep_datetime_str(start_time)
    end_str, end_dt = prep_datetime_str(end_time)

    # get state bbox
    bbox_dict = get_state_bbox(state_name)

    # get state fips string code
    state_fips = get_state_fips()[state_name]

    # get base API url and URl component checker (for API call efficiency)
    base_stations_url = r'https://www.ncei.noaa.gov/cdo-web/api/v2/stations?limit=1000'
    url_checker = UrlChecker(token)

    # get stations info as a dataframe
    print('Finding valid NCEI stations')
    stations_df = get_stations_by_state(
        state_name,
        state_fips,
        date_range=(start_str, end_str),
        token=token,
        ncei_dataset_id=dataset_id,
        url_checker=url_checker,
        base_stations_url=base_stations_url,
        )

    # get data
    print('Getting data from NCEI API')
    years = list(range(start_dt.year, end_dt.year + 1))

    data_list = []
    for year in years:
        print(f'Processing year={year}')
        for station_id in stations_df.index[:5]:
            station_id = str(station_id)

            url = base_stations_url +  f'&datasetid={dataset_id}' + \
                f'&locationid=FIPS:{state_fips}' + \
                f'&datatypeid={datatype_id}' + \
                f'&stationid={station_id}' + \
                f'&startdate={year}-01-01&enddate={year}-12-31'

            response = dict(requests.get(
                url,
                headers={'token': 'DSsSMpVbUnklkSjSzeAZFfaNsKsvommI'},
                ).json())
            if 'results' in list(response.keys()): data_list.extend(list(response['results']))
        gc.collect()

    # construct a dataframe with all precipitation data
    data_df = pd.DataFrame().from_records(
        data_list,  
        index='station',
        exclude=['attributes'],
    )
    data_df.index.name = 'station_id'
    return data_df

if __name__ == '__main__':
    get_precipitation_data(
    STATE_NAME,
    START_TIME,
    END_TIME,
    TOKEN,
    )
