import os
import glob

from datetime import datetime
from tqdm import tqdm_notebook
from tqdm import tqdm
import pandas as pd
import numpy as np
import pytz
import time

STATIONNAME = 'Boulder_CO'
YEAR = '2013'


def get_surfrad_data(station_name=STATIONNAME, year=YEAR, timezone=None, force_reload=False, force_redownload=False, reindex=True, clean_flaged_values=False):
    """Load, parse and combine the surfrad data for a particular Year

    Parameters
    ----------
    :param station_name:
        Name of weather station from which to retrieve data
    :param year:
        chosen year
    :param force_reload: bool (optional)
        if True, force reload of data
    :param timezone: string (optional)
        Any of the tz database time zones (eg. 'America/Denver')
    :param reindex: bool (optional)
        reindexes the dataframe to a full year of data (eg. 525600 lines if after 2009 and 175200 if before 2009)
    :param clean_flaged_values: bool (optional)
        if flag > 0, replace with NaN for all columns with flags.


    Returns
    -------
    :returns: pandas.DataFrame
        The surfrad data

    More INFO
    ---------

    Parameter         Parameter      Parameter      Parameter         
    ================  =============  =============  ================
    Year              dt             direct_n       dw_casetemp       
    Day number        zen            direct_n_Flag  dw_casetemp_Flag  
    Month             dw_solar       diffuse        dw_dometemp       
    Day               dw_solar_Flag  diffuse_Flag   dw_dometemp_Flag  
    Hour              uw_solar       dw_ir          uw_ir             
    Minute            uw_solar_Flag  dw_ir_Flag     uw_ir_Flag        
    uw_casetemp       par            totalnet       windspd
    uw_casetemp_Flag  par_Flag       totalnet_Flag  windspd_Flag
    uw_dometemp       netsolar       temp           winddir
    uw_dometemp_Flag  netsolar_Flag  temp_Flag      winddir_Flag
    uvb               netir          rh             pressure
    uvb_Flag          netir_Flag     rh_Flag        pressure_Flag
    ----------------  -------------  -------------  ----------------

    """
    names = ['Year', 'Day number', 'Month', 'Day', 'Hour', 'Minute', 'dt', 'zen', 'dw_solar',
             'dw_solar_Flag', 'uw_solar', 'uw_solar_Flag', 'direct_n', 'direct_n_Flag', 'diffuse', 'diffuse_Flag',
             'dw_ir', 'dw_ir_Flag', 'dw_casetemp', 'dw_casetemp_Flag', 'dw_dometemp', 'dw_dometemp_Flag', 'uw_ir',
             'uw_ir_Flag', 'uw_casetemp', 'uw_casetemp_Flag', 'uw_dometemp', 'uw_dometemp_Flag', 'uvb', 'uvb_Flag',
             'par', 'par_Flag', 'netsolar', 'netsolar_Flag', 'netir', 'netir_Flag', 'totalnet', 'totalnet_Flag',
             'temp', 'temp_Flag', 'rh', 'rh_Flag', 'windspd', 'windspd_Flag', 'winddir', 'winddir_Flag', 'pressure',
             'pressure_Flag']

    # Force reload
    if force_redownload:
        download_surfrad(station_name, year)
    if force_reload or not os.path.exists(station_name + '_' + year + '.gzip'):
        if not os.path.exists(os.getcwd() + '/surfrad/' + station_name + '/' + year):
            download_surfrad(station_name, year)
        files = []
        start_dir = os.getcwd() + '/surfrad/' + station_name
        pattern = "*.*"
        for dir,_,_ in os.walk(start_dir):
            files.extend(glob.glob(os.path.join(dir, pattern)))
        dfs = [pd.read_csv(f, delim_whitespace=True, skipinitialspace=True, skiprows=2,
                           header=None, names=names, encoding='ASCII') for f in tqdm_notebook(files, desc='creating dataframe')]
        data = pd.concat(dfs)
        data.index = pd.to_datetime(data.Year*100000000+data.Month*1000000+data.Day*10000+data.Hour*100+data.Minute, format='%Y%m%d%H%M')
        data.index.name = 'timestamp'
        data = data.sort_index()
        if len(timezone):
            change_timezone(data, timezone)
        data = data.tshift(-1,freq='1min')
        data = data[data.index.year == int(year)]
        data.to_csv(station_name + '_' + year + '.gzip', compression='gzip')
    else:
        data = pd.read_csv(station_name + '_' + year + '.gzip', index_col=[0], compression='gzip')
        data = data.sort_index()
        try:
            data.index = pd.to_datetime(data.index, format='%Y-%m-%d %H:%M:%S')
            data.index.name = 'timestamp'
        except TypeError:
            data.index = pd.to_datetime(data.index)
            data.index.name = 'timestamp'
        if len(timezone):
            change_timezone(data, timezone)
    if clean_flaged_values:
        col_flag = data.columns[data.columns.str.endswith('_Flag')]
        col_noflag = col_flag.str.replace('_Flag', '')
        for i in tqdm_notebook(col_noflag.values, desc='cleaning flags'):
            data.loc[data[i + '_Flag'] > 0, i] = np.nan
    if reindex:
        data = reindex_from_frequency(data, year, timezone, drop=True)
    return data


def change_timezone(data, timezone):
    zone = pytz.timezone(timezone)
    data.index = data.index.tz_localize('UTC').tz_convert(zone)
    data.index.name = 'timestamp'


def reindex_from_frequency(data, year, timezone, drop=True):
    if data.index[0].year >= 2009:
        frequency='1min'
    else:
        frequency='3min'
    newindex = pd.date_range(datetime(int(year), 1, 1, 0, 0), datetime(int(year), 12, 31, 23, 59), freq=frequency, tz=timezone)
    data = data.reindex(newindex, fill_value=np.NaN)
    if drop:
        data = data[data.index.year == int(year)]
    return data


import ftplib
from ftplib import FTP


def download_surfrad(station_name=None, year=None):
    """

    :param station_name:
    :param year:
    """
    original_directory = os.getcwd()
    stationNameArray = ['Alamosa CO', 'Bondville IL', 'Boulder CO', 'Desert Rock NV',
                        'Fort Peck MT', 'Goodwin Creek MS', 'Penn State PA',
                        'Rutland VT', 'Sioux Falls SD', 'Wasco OR']
    # log into FTP
    ftp = FTP('aftp.cmdl.noaa.gov')
    ftp.login()

    ftpURL = 'data/radiation/surfrad/' + station_name + '/'

    try:
        ftp.cwd(ftpURL)
    except ftplib.all_errors as e:
        print(e)

    ftp_URL_withyear = year + '/'

    try:
        ftp.cwd(ftp_URL_withyear)
    except ftplib.all_errors as e:
        print(e)

    if not os.path.exists(os.getcwd() + '/surfrad/' + station_name + '/' + year):
        os.makedirs(os.getcwd() + '/surfrad/' + station_name + '/' + year)
    os.chdir(os.getcwd() + '/surfrad/' + station_name + '/' + year)
    filenames = ftp.nlst()
    for filename in tqdm_notebook(filenames, desc='download'):
        try:
            file = open(filename, 'wb')
            ftp.retrbinary('RETR ' + filename, file.write)
            file.close()
        except ftplib.all_errors as e:
            print(e)
    os.chdir(original_directory)
    ftp.quit()

    # download first day of next year
    # log into FTP
    ftp = FTP('aftp.cmdl.noaa.gov')
    ftp.login()

    ftpURL = 'data/radiation/surfrad/' + station_name + '/'

    try:
        ftp.cwd(ftpURL)
    except ftplib.all_errors as e:
        print(e)

    ftp_URL_nextyear = str(int(year)+1) + '/'
    try:
        ftp.cwd(ftp_URL_nextyear)
    except ftplib.all_errors as e:
        print(e)

    station_path = os.getcwd() + '/surfrad/' + station_name + '/' + str(int(year)+1)
    if not os.path.exists(station_path):
        os.makedirs(station_path)
    os.chdir(station_path)

    filename = ftp.nlst()
    try:
        file = open(filename[0], 'wb')
        ftp.retrbinary('RETR ' + filename[0], file.write)
        file.close()
    except ftplib.all_errors as e:
        print(e)
    ftp.quit()
    print('File transfer complete!')
    os.chdir(original_directory)


def lookup(s):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    """
    dates = {date: pd.to_datetime(date) for date in s.unique()}
    return s.apply(lambda v: dates[v])

def store_data(data,station_name=None,year=None):
    store = pd.HDFStore(station_name + '_' + year + 'h5')
