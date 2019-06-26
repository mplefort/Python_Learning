import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
import pandas as pd
import os

HOUSING_PATH = os.path.join("datasets", "housing")

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)
# set pycharm to show each column when printing
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',10)

#  Get data
housing = load_housing_data()
## inspect data with:
# housing.describe()
# housing.hist(bins=50, figsize=(20,15))
# plt.show()

"""
Data splitting for test train set
    - if offline ok to use a seeded random num gen to keep data in, if data updating will need to train new data in 
    same order/split so a hash will need to be assigned to old and new data. 

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

"""

"""
Stratified sampling based off income
    - more accurate test/train split data based off of distribution of data instead of just random samples
"""
from sklearn.model_selection import StratifiedShuffleSplit

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)  # scale data max 15 -> 10, 75% < 5, round up
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True) #cut off at 5.0

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

# drop "income_cat from test/train sets as stratified
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

"""
Discover the data 
    - remove test set of data to manipulate only training data. Even human can make overfitting occur if able to 
    manipulate training set.
    - using a scatter plot of map is a good option to see location to price and population values. 
"""

housing = strat_train_set.copy()
# s = radius is population size,
# c = color is housing price using jet scheme
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
# s=housing["population"]/100, label="population", figsize=(10,7),
# c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
# plt.legend()

"""
Correlations 
 - scatter plot hist of most correclated attributes to housing_value
 - most promsing is median income with high corellation. Detailed scatter of it shows high freq of points at the 350,
 450, and 500 
"""
from pandas.plotting import scatter_matrix

print(housing.corr())
attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)

"""
Data attribute combinations
"""
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values())

""""
---------------------- Prepare Data for Machine Learning pg. 87 ----------------------------------------------
- transformation functions to prepare data. Start to build library of utils and prepare for future data updates as well.
"""

# return to original data set
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

"""
Data cleaning
 - remove or fill missing data
"""
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy='median')
# remove non numerical values to calculate median
housing_num = housing.drop("ocean_proximity",axis=1)
imputer.fit(housing_num)
X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)

""" 
Handling Text Categories - with one_hot_encoding
"""
from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)


""" 
Custom Transformers for data cleaning
 - Follow Scikit learns outline include a fit(), transform(), and fit_transform() method.
    - makes easy to make training with and without certain transforms easy.
"""

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# A numpy array of the extra attribs added using a transfromer function
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


"""
Feature scaling
 - There are two common ways to get all attributes to have the same scale: min-max scaling and standardization.
    - min-max = scales to 0-1 by x = x[i] - x_min / (x_max - x_min). Subject to outliers ruining values. 
        - MinMiaxScaler()
    - standardization - normalize to guassian dist. Good for outlier datas, bad for ANNs as they expect 0-1
        - StandardScaler()
 - Always scale to training set and then apply to test.

"""

"""
Transformation Pipelines
    - SciKit pipeline for transforms
    - 
"""
# convert the numpy.array from scikit-learns pipeline tranformers back into a pandas df
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([ ('selector', DataFrameSelector(num_attribs)),
                          ('imputer', Imputer(strategy="median")),
                          ('attribs_adder', CombinedAttributesAdder()),
                          ('std_scaler', StandardScaler()),
                          ])

cat_pipeline = Pipeline([('selector', DataFrameSelector(cat_attribs)),
                         # ('label_binarizer', LabelBinarizer()),    # Cannot get to work, used a separate flow below to
                         # complete issue.
                         ])

# combine the two pipelines
from sklearn.pipeline import FeatureUnion
full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

# Sol for above error: https://stackoverflow.com/questions/46162855/fit-transform-takes-2-positional-arguments-but-3-were-given-with-labelbinarize
housing_prepared = np.delete(housing_prepared,11,1)
encoder = LabelBinarizer()
housing_cat = housing["ocean_proximity"]
housing_cat_1hot = encoder.fit_transform(housing_cat)

housing_prepared = np.concatenate([housing_prepared, housing_cat_1hot], axis=1)


"""
------------------------  Select and Train a model --------------------------------
pg. 96

"""