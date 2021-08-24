# Author - Willis Knox
import sys
sys.path.append('../')

from datetime import datetime
import timedoor.timedoor as timedoor
from env.key import key
import pandas as pd

api_key = key

def main():
    df = pd.read_csv('../data/AirQualityUCI/small_aq.csv', sep=';')
    datetimes = df["Date"] + df["Time"]
    valid_datetimes = [datetime.strptime(date, '%d/%m/%Y%H.%M.%S').isoformat() + 'Z' for date in datetimes]
    ozone = df["PT08.S5(O3)"]
    arima = timedoor.auto_arima(dates=valid_datetimes, values=ozone, api_key=key)
    # print(arima)

if __name__ == '__main__':
    main()