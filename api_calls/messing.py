# Author - Willis Knox
from timedoor import timedoor_objects as td
from timedoor import timedoor
from env.key import key
import pandas as pd

api_key = key

def main():
    dates = []
    values = []
    arima = timedoor.auto_arima(dates=dates, values=values, api_key=key)
    print(arima)

if __name__ == '__main__':
    main()