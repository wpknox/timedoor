# Author - Willis Knox
import sys
sys.path.append('../')
from datetime import datetime
import timedoor.timedoor as timedoor
from env.key import key
import pandas as pd

api_key = key


def try_arima():
    df = pd.read_csv('../data/AirQualityUCI/small_aq.csv', sep=';')
    datetimes = df["Date"] + df["Time"]
    valid_datetimes = [datetime.strptime(
        date, '%d/%m/%Y%H.%M.%S').isoformat() + 'Z' for date in datetimes]
    vals = list(df["PT08.S5(O3)"])
    arima = timedoor.auto_arima(
        dates=valid_datetimes, values=vals, api_key=api_key, error_value=0)
    print(arima)


def try_changepoint():
    df = pd.read_csv('../data/AirQualityUCI/small_aq.csv', sep=';')
    datetimes = df["Date"] + df["Time"]
    valid_datetimes = [datetime.strptime(
        date, '%d/%m/%Y%H.%M.%S').isoformat() + 'Z' for date in datetimes]
    vals = list(df["PT08.S5(O3)"])
    change_point = timedoor.changepoint_detection(
        dates=valid_datetimes, values=vals, api_key=api_key, error_value=-200)
    print(change_point)


def main():
    try_arima()
    try_changepoint()


if __name__ == '__main__':
    main()
