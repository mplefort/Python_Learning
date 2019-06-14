import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
# import sklearn
from sklearn.model_selection import train_test_split
from sklearn.model_selection import	StratifiedShuffleSplit
from pandas.tools.plotting import scatter_matrix


import os
import sys

# Set working directory to HandsOnMachineLearing
# os.chdir(os.path.join(os.getcwd(), 'HandsOnMachineLearning', 'mynotes') )
os.chdir("..")
if os.path.basename(os.getcwd()) != "HandsOnMachineLearning":
    print("Current directory not \'HandsOnMachineLearning\', at %s" % os.getcwd())
    exit()
HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)


def split_train_test(data: pd.DataFrame, test_ratio: float) -> object:
    """
        split data into test and train set. Typically 20% Test, 80% Train
    :param data: data frame to split
    :param test_ratio: float percent to keep as test ratio (0.0 - 1.0)
    """

    # create random indicies 0-len(data)
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


### Get the Data
# Load the data
housing = load_housing_data(HOUSING_PATH)

## Explore the data - Notice tail heavy data, Caps in median income, home age, and home value, diff scales of values
# housing.hist(bins=50, figsize=(20,15))


## Limit income range to categories and and reshape data to be more bell shaped pg. 76
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)  # limit income cats to 10 (15.0 / 1.5)
housing["income_cat"].where(housing["income_cat"] < 5,	5.0,	inplace=True)  # rounds down cats > 5.0 (75K) to 75k
# housing["income_cat"].hist()
# plt.show()

## Create a Test set
# train_set, test_set = split_train_test(housing, 0.2)  # Using own split_train_test func
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)  # using scikit-learn
print("%i train data + %i test data" % (len(train_set), len(test_set)) )

# with scikit-learn stratified split for equal weighting income cats
split = StratifiedShuffleSplit(n_splits=1,	test_size=0.2,	random_state=42)
for	train_index, test_index	in	split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]


## Make a chart to show the comparison of stratified vs random samples on pg 78.
# samplecomparedf = pd.DataFrame(
#     data=(
#             {'Overall': housing["income_cat"].value_counts() / len(housing),  # all data
#              'Random': test_set["income_cat"].value_counts() / len(test_set),  # random samples
#              'Strat': strat_test_set["income_cat"].value_counts() / len(strat_test_set)}  # strat samples
#             )
#     )
#
# samplecomparedf['Rand % err'] = np.abs(( samplecomparedf['Overall'] - samplecomparedf['Random'] ) / samplecomparedf['Overall']) * 100.0
# samplecomparedf['Strat % err'] = np.abs(( samplecomparedf['Overall'] - samplecomparedf['Strat'] ) / samplecomparedf['Overall']) * 100.0
#
# # Shows strat error gets a more accurate distribution of income_cat in test/train split with close to
# print(samplecomparedf.sort_index())

for	set_ in	(strat_train_set, strat_test_set):
    set_.drop("income_cat",	axis=1,	inplace=True)  # remove income_cat as we only used it for stratified samples

### Discover and Visualize the Data to Gain Insights Pg. 80

housing	= strat_train_set.copy() # create copy of training set to play/explore with
## scatter plot of the data
# housing.plot(kind="scatter", x="longitude",	y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population", figsize=(10,7),  # radius is population size/100
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,  # color map is housing value
#              )
# plt.legend()
# plt.show()

## Looking for Correlations
# corr_matrix = housing.corr()
# print(corr_matrix['median_house_value'].sort_values(ascending=False))  # see that median income has a linear rel with house price

# attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']  # plot to make sure corr matrix zeros are actual no correlation
# scatter_matrix(housing[attributes])
# housing.plot(kind='scatter', x='median_income', y='median_house_value')
# plt.show()

## Experimenting with Attribute Combinations
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]

# corr_matrix = housing.corr() # shows more correlation with "bedrooms_per_room" and "rooms_per_household"
# print(corr_matrix['median_house_value'].sort_values(ascending=False))  # see that median income has a linear rel with house price

### Prepare the Data for Machine Learning Algorithms


print('done')