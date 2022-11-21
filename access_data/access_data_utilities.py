import pandas as pd
import requests
from typing import Union, Tuple, List
from datetime import datetime
from typing import TypedDict

# formalize any relevant types
class BboxDict(TypedDict):
    xmin: float  # min longitude
    xmax: float  # max longitude
    ymin: float  # min latitude
    ymax: float  # max latitude


def prep_datetime_str(
    time: Union[str, datetime],
    format: str = r'%Y-%m-%d',
    ) -> Tuple[str, datetime]:
    """
    Two behaviors: 
        * if type(param:time)=str -> verifies that a string date is in the correct format.
        * if type(param:time)=datetime -> converts to a string formatted date.
    :returns: a tuple with (datetime string, datetime object)
    """
    if isinstance(time, str):
        try:
            dt = datetime.strptime(time, format)
            return (dt.strftime(format), dt)
        except ValueError:
            print(f'Time string {time} does not match format={format}')
            raise ValueError
    elif isinstance(time, datetime): return (time.strftime(format), time)
    else: 
        print('ERROR: param:time must be either a valid date string of a datetime object! '
        f'type(param:time)={type(time)}.')
        raise TypeError

def get_state_bbox(
    state_name: str,
        ) -> BboxDict:
    """
    Get the bounding box of a US state in EPSG4326 given it's name
    :param place: (str) a valid state name in english.
    :returns: (tuple) a tuple w/ coordinates as floats i.e.,
        [[11.777, 53.7253321, -70.2695876, 7.2274985]].
    """
    # create url to pull openstreetmap data
    url_prefix = 'http://nominatim.openstreetmap.org/search?state='

    url = '{0}{1}{2}'.format(
        url_prefix,
        state_name.lower(),
        '&format=json&polygon=0',
        )
    response = requests.get(url).json()[0]

    lst = response['boundingbox']
    coors = [float(i) for i in lst]
    return {
        'xmin': coors[-2],
        'xmax': coors[-1],
        'ymin': coors[0],
        'ymax': coors[1],
        }

def get_state_fips(
    keep_states: Union[str, List[str]] = None
    ) -> dict:
    """
    Builds a dictionary with lowercase statenames as keys, and their FIPS codes (str) as value.
    :param keep_states: (optional, str or list of str) only keeps dictionary records for specified statenames.
    """
    list_of_pairs = list(
        requests.get(
            r'https://api.census.gov/data/2010/dec/sf1?get=NAME&for=state:*'
        ).json()
    )

    if keep_states:
        if isinstance(keep_states, str): keep_states = [keep_states]
        keep_states = [state.lower() for state in keep_states]

    state_fips_dict = {}
    for pair in list_of_pairs:
        if keep_states:
            if pair[0].lower() not in keep_states: continue
        state_fips_dict[pair[0].lower()] = str(pair[-1])

    state_fips_dict.pop('name')
    return state_fips_dict

def validate_date(
    response_dict: dict,
    date_range: Tuple[str, str],
    str_format: str = '%Y-%m-%d',
        ) -> bool:
    # get station date range
    min_date = datetime.strptime(response_dict['mindate'], str_format)
    max_date = datetime.strptime(response_dict['maxdate'], str_format)
    
    # convert our date range to datetime
    range_min_date = datetime.strptime(date_range[0], str_format)
    range_max_date = datetime.strptime(date_range[-1], str_format)

    if min_date <= range_min_date:
        if max_date >= range_max_date:
            return True
    else:
        return False
