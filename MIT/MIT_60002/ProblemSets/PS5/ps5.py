# -*- coding: utf-8 -*-
# Problem Set 5: Experimental Analysis
# Name: 
# Collaborators (discussion):
# Time:

import matplotlib.pyplot as plt
import numpy as np
import re
from numpy.polynomial.polynomial import Polynomial
import math
# cities in our weather data
CITIES = [
    'BOSTON',
    'SEATTLE',
    'SAN DIEGO',
    'PHILADELPHIA',
    'PHOENIX',
    'LAS VEGAS',
    'CHARLOTTE',
    'DALLAS',
    'BALTIMORE',
    'SAN JUAN',
    'LOS ANGELES',
    'MIAMI',
    'NEW ORLEANS',
    'ALBUQUERQUE',
    'PORTLAND',
    'SAN FRANCISCO',
    'TAMPA',
    'NEW YORK',
    'DETROIT',
    'ST LOUIS',
    'CHICAGO'
]

TRAINING_INTERVAL = range(1961, 2010)
TESTING_INTERVAL = range(2010, 2016)

"""
Begin helper code
"""
class Climate(object):
    """
    The collection of temperature records loaded from given csv file
    """
    def __init__(self, filename):
        """
        Initialize a Climate instance, which stores the temperature records
        loaded from a given csv file specified by filename.

        Args:
            filename: name of the csv file (str)
        """
        self.rawdata = {}

        f = open(filename, 'r')
        header = f.readline().strip().split(',')
        for line in f:
            items = line.strip().split(',')

            date = re.match('(\d\d\d\d)(\d\d)(\d\d)', items[header.index('DATE')])
            year = int(date.group(1))
            month = int(date.group(2))
            day = int(date.group(3))

            city = items[header.index('CITY')]
            temperature = float(items[header.index('TEMP')])
            if city not in self.rawdata:
                self.rawdata[city] = {}
            if year not in self.rawdata[city]:
                self.rawdata[city][year] = {}
            if month not in self.rawdata[city][year]:
                self.rawdata[city][year][month] = {}
            self.rawdata[city][year][month][day] = temperature
            
        f.close()

    def get_yearly_temp(self, city, year):
        """
        Get the daily temperatures for the given year and city.

        Args:
            city: city name (str)
            year: the year to get the data for (int)

        Returns:
            a 1-d pylab array of daily temperatures for the specified year and
            city
        """
        temperatures = []
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        for month in range(1, 13):
            for day in range(1, 32):
                if day in self.rawdata[city][year][month]:
                    temperatures.append(self.rawdata[city][year][month][day])
        return np.array(temperatures)

    def get_daily_temp(self, city, month, day, year):
        """
        Get the daily temperature for the given city and time (year + date).

        Args:
            city: city name (str)
            month: the month to get the data for (int, where January = 1,
                December = 12)
            day: the day to get the data for (int, where 1st day of month = 1)
            year: the year to get the data for (int)

        Returns:
            a float of the daily temperature for the specified time (year +
            date) and city
        """
        assert city in self.rawdata, "provided city is not available"
        assert year in self.rawdata[city], "provided year is not available"
        assert month in self.rawdata[city][year], "provided month is not available"
        assert day in self.rawdata[city][year][month], "provided day is not available"
        return self.rawdata[city][year][month][day]


def se_over_slope(x, y, estimated, model):
    """
    For a linear regression model, calculate the ratio of the standard error of
    this fitted curve's slope to the slope. The larger the absolute value of
    this ratio is, the more likely we have the upward/downward trend in this
    fitted curve by chance.
    
    Args:
        x: an 1-d np array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d np array of values estimated by a linear
            regression model
        model: a np array storing the coefficients of a linear regression
            model

    Returns:
        a float for the ratio of standard error of slope to slope
    """
    assert len(y) == len(estimated)
    assert len(x) == len(estimated)
    EE = ((estimated - y)**2).sum()
    var_x = ((x - x.mean())**2).sum()
    SE = np.sqrt(EE/(len(x)-2)/var_x)
    return SE/model[0]


"""
End helper code
"""


def generate_models(x: [np.ndarray], y: [np.ndarray], degs: [int]):
    """
    Generate regression models by fitting a polynomial for each degree in degs
    to points (x, y).

    Args:
        x: an 1-d np array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        degs: a list of degrees of the fitting polynomial

    Returns:
        a list of np arrays, where each array is a 1-d array of coefficients
        that minimizes the squared error of the fitting polynomial
    """

    models = []
    for i in degs:
        models.append(np.asarray(np.polyfit(x, y, i)))
    return models


def r_squared(y, estimated):
    """
    Calculate the R-squared error term.
    
    Args:
        y: 1-d np array with length N, representing the y-coordinates of the
            N sample points
        estimated: an 1-d np array of values estimated by the regression
            model

    Returns:
        a float for the R-squared error term
    """
    R_sq_err = 1.0 - (np.sum((y-estimated) **2)) / \
               np.sum((y - np.mean(y))** 2)

    return R_sq_err


def evaluate_models_on_training(x, y, models):
    """
    For each regression model, compute the R-squared value for this model with the
    standard error over slope of a linear regression line (only if the model is
    linear), and plot the data along with the best fit curve.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        R-square of your model evaluated on the given data points,
        and SE/slope (if degree of this model is 1 -- see se_over_slope). 

    Args:
        x: an 1-d np array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a np array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    # infos = {'deg': [float(R_err), 'title', float(SE_err(if deg==1)) }
    infos = {}
    for model in models:
        deg = str(len(model)-1)
        infos[deg] = []

        y_p = np.polyval(model, x)
        r_err = r_squared(y, y_p)

        if deg == '1':
            se = se_over_slope(x, y, y_p, model)
            title = ('Deg: {}, R^2 er: {}, SE/slope: {}'.format(deg, r_err, se) )

        else:
            title = ('Deg: {}, R^2 er: {}'.format(deg, r_err) )

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(x, y, color='blue')
        ax.plot(x, y_p, color='red')
        ax.set(title=title, xlabel="Years", ylabel="Deg C", )
        plt.show()


# x = np.array(range(50))
# y = np.array(range(50))
# degrees = [1]
# models = generate_models(x, y, degrees)
# evaluate_models_on_training(x,y,models)

def gen_cities_avg(climate: Climate, multi_cities: [str], years: [int]):
    """
    Compute the average annual temperature over multiple cities.

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to average over (list of str)
        years: the range of years of the yearly averaged temperature (list of
            int)

    Returns:
        a np 1-d array of floats with length = len(years). Each element in
        this array corresponds to the average annual temperature over the given
        cities for a given year.
    """
    year_avg_temps = []
    for year in years:
        temps = []
        for city in multi_cities:
            temps = np.concatenate((temps, climate.get_yearly_temp(city, year)), axis=None)
        year_avg_temps.append(np.mean(temps))

    return year_avg_temps


def moving_average(y, window_length):
    """
    Compute the moving average of y with specified window length.

    Args:
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        window_length: an integer indicating the window length for computing
            moving average

    Returns:
        an 1-d np array with the same length as y storing moving average of
        y-coordinates of the N sample points
    """
    n = window_length
    avg = []
    for idx, _ in enumerate(y):
        if idx < n:
            myslice = y[:idx+1]
        else:
            myslice = y[idx-n+1: idx+1]

        avg.append(np.mean(myslice))

    return avg


def rmse(y, estimated):
    """
    Calculate the root mean square error term.

    Args:
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        estimated: an 1-d np array of values estimated by the regression
            model

    Returns:
        a float for the root mean square error term
    """

    RMSE = np.sqrt(np.sum((y-estimated)**2 / len(y)))
    return RMSE


def gen_std_devs(climate, multi_cities, years):
    """
    For each year in years, compute the standard deviation over the averaged yearly
    temperatures for each city in multi_cities. 

    Args:
        climate: instance of Climate
        multi_cities: the names of cities we want to use in our std dev calculation (list of str)
        years: the range of years to calculate standard deviation for (list of int)

    Returns:
        a np 1-d array of floats with length = len(years). Each element in
        this array corresponds to the standard deviation of the average annual 
        city temperatures for the given cities in a given year.
    """

    result = []
    for year in years:
        daily_temp_365 = np.zeros(365)
        daily_temp_366 = np.zeros(366)
        for city in multi_cities:
            if len(climate.get_yearly_temp(city, year)) == 365:
                daily_temp_365 += climate.get_yearly_temp(city, year)
            else:
                daily_temp_366 += climate.get_yearly_temp(city, year)
        if sum(daily_temp_365) > sum(daily_temp_366):
            daily_temp = daily_temp_365
        else:
            daily_temp = daily_temp_366
        daily_temp = daily_temp / len(multi_cities)
        mean = np.mean(daily_temp)
        var = 0.0
        for temp in list(daily_temp):
            var += (temp - mean) ** 2
        result.append(math.sqrt(var / len(daily_temp)))
    return np.array(result)

def evaluate_models_on_testing(x, y, models):
    """
    For each regression model, compute the RMSE for this model and plot the
    test data along with the model’s estimation.

    For the plots, you should plot data points (x,y) as blue dots and your best
    fit curve (aka model) as a red solid line. You should also label the axes
    of this figure appropriately and have a title reporting the following
    information:
        degree of your regression model,
        RMSE of your model evaluated on the given data points. 

    Args:
        x: an 1-d np array with length N, representing the x-coordinates of
            the N sample points
        y: an 1-d np array with length N, representing the y-coordinates of
            the N sample points
        models: a list containing the regression models you want to apply to
            your data. Each model is a np array storing the coefficients of
            a polynomial.

    Returns:
        None
    """
    for model in models:
        print(model)
        p = np.poly1d(model)
        plt.figure()
        plt.plot(x, y, 'bo', label = 'Measured points')
        plt.plot(x, p(x), 'r-', label = 'Fit')
        plt.legend(loc = 'best')
        plt.title('Degree of Fit: ' + str(len(model) - 1) + '\n' + 'RMSE: ' + str(round(rmse(y, p(x)), 5)))
        plt.xlabel('Year')
        plt.ylabel('Temperature (Celsius)')
        plt.show()


if __name__ == '__main__':
    # Part A.4
    climate = Climate('data.csv')

    #   4AI
    # data = New York, Jan 10, 1961 - 2009 temps
    temps = []
    years = np.asarray(TRAINING_INTERVAL, dtype=int)
    city = 'NEW YORK'
    month = 1
    day = 10
    for year in years:
        temps.append(climate.get_daily_temp(city, month, day, year))

    temps = np.asarray(temps)
    models = generate_models(years, temps, [1])
    evaluate_models_on_training(years, temps, models)

    # 4AII:
    temps = []
    for year in years:
        year_temps = climate.get_yearly_temp(city, year)
        year_temps = np.asarray(year_temps)
        temps.append(np.mean(year_temps))

    temps = np.asarray(temps)
    models = generate_models(years, temps, [1])
    evaluate_models_on_training(years, temps, models)

    # Part B
    temps = gen_cities_avg(climate, CITIES, years)
    models = generate_models(years, temps, [1])
    evaluate_models_on_training(years, temps, models)

    # Part C
    # cities_average = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    # moving_aver = moving_average(cities_average, 5)
    # model = generate_models(pylab.array(TRAINING_INTERVAL), moving_aver, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), moving_aver, model)

    # The R^2 is better compared to those in part A&B. The previous data used to plot in A&B are noisy. Computing the
    # moving average helps reducing the noise level which improves the fit.

    # Part D.2
    # 2.I

    # cities_average = gen_cities_avg(climate, CITIES, TRAINING_INTERVAL)
    # moving_aver = moving_average(cities_average, 5)
    # model = generate_models(pylab.array(TRAINING_INTERVAL), moving_aver, [1, 2, 20])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), moving_aver, model)

    # Degree fit 20 is the best model with the best R^2 value

    # 2.II
    # cities_average = gen_cities_avg(climate, CITIES, TESTING_INTERVAL)
    # moving_aver = moving_average(cities_average, 5)
    # evaluate_models_on_testing(pylab.array(TESTING_INTERVAL), moving_aver, model)

    # Linear fit gives the best RMSE. This is not the case in part I where degree 20 fit gives the best fit instead.
    # The difference is caused by having fewer data points in part II. The result would be worse ( increase in RMSE) if
    # we generated models using the A.4.II data.

    # Part E
    # std_devs = gen_std_devs(climate, CITIES, TRAINING_INTERVAL)
    # moving_aver = moving_average(std_devs, 5)
    # model = generate_models(pylab.array(TRAINING_INTERVAL), moving_aver, [1])
    # evaluate_models_on_training(pylab.array(TRAINING_INTERVAL), moving_aver, model)

    # The result shows that the 5-year moving averages on the yearly standard deviation decreases over time. The
    # analysis can be improved by using a larger window length and higher degree of fit