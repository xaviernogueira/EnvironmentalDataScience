from NCEI_data_access import get_ncei_data

# set NCEI_data_access environment variables
TOKEN = 'DSsSMpVbUnklkSjSzeAZFfaNsKsvommI'
STATE_NAME = 'pennsylvania'
START_TIME = '2016-01-01'
END_TIME = '2021-12-31'

if __name__ == '__main__':
    output_dict = get_ncei_data(
        STATE_NAME,
        START_TIME,
        END_TIME,
        TOKEN,
        )