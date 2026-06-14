import os


cwd = os.getcwd()


print("Current working directory: {0}".format(cwd))


print ("os.getcwd() returns an object of type {0}".format(type(cwd)))


# copy the filepath

os.chdir ("C:/Users/thoma/OneDrive/Documents/GitHub/Quant-Researcher-simulation/taskOne")


# let's jump into task 1


import pandas as pd

import numpy as np

from matplotlib import pyplot as plt

from datetime import date, timedelta


date_time = ["10-2020", "11-2020", "12-2020"]

date_time = pd.to_datetime(date_time, format="%m-%Y")

data = [1, 2, 3]

def build_model(csv_path='natgas_R.csv'):

    df = pd.read_csv('natgas_R.csv', parse_dates=['Dates'], date_format="%Y-%m-%d")
    prices = df['Prices'].values
    dates = df['Dates'].values

    start_date = date(2020,10,31)
    end_date = date(2024,9,30)
    months = []
    year = start_date.year
    month = start_date.month + 1

    while True:

        current = date(year, month, 1) + timedelta(days=-1)
        months.append(current)
        if current.month == end_date.month and current.year == end_date.year:
            break
        else:
            month = ((month + 1) % 12) or 12
            if month == 1:
                year += 1

    days_from_start = [(day - start_date ).days for day in months]

    def simple_regression(x, y):
        xbar = np.mean(x)
        ybar = np.mean(y)
        slope = np.sum((x - xbar) * (y - ybar))/ np.sum((x - xbar)**2)
        intercept = ybar - slope*xbar
        return slope, intercept
    
    def bilinear_regression(y, x1, x2):
        slope1 = np.sum(y * x1) / np.sum(x1 ** 2)
        slope2 = np.sum(y * x2) / np.sum(x2 ** 2)
        return(slope1, slope2)

    time = np.array(days_from_start)
    slope, intercept = simple_regression(time, prices)

    sin_prices = prices - (time * slope + intercept)
    sin_time = np.sin(time * 2 * np.pi / (365))
    cos_time = np.cos(time * 2 * np.pi / (365))
    slope1, slope2 = bilinear_regression(sin_prices, sin_time, cos_time)

    amplitude = np.sqrt(slope1 ** 2 + slope2 ** 2)
    shift = np.arctan2(slope2, slope1)

    def interpolate(date):

        days = (date - pd.Timestamp(start_date)).days
        if days in days_from_start:
            return prices[days_from_start.index(days)]
        else:
            return amplitude * np.sin(days * 2 * np.pi / 365 + shift) + days * slope + intercept

    return interpolate
