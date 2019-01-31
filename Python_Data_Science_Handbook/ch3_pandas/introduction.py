import pandas as pd
import os
import seaborn as sns
import numpy as np

# print(pd.__version__)

## US states example page 145
# dir = os.path.join(os.getcwd(),'data-USstates')
# print(dir)
#
# pop = pd.read_csv(os.path.join(dir, 'state-population.csv'))
# areas = pd.read_csv(os.path.join(dir, 'state-areas.csv'))
# abbrevs = pd.read_csv(os.path.join(dir, 'state-abbrevs.csv'))
#
#
# print(pop.head()); print(areas.head()); print(abbrevs.head())

## Planets Data
planets = sns.load_dataset('planets')
# print(planets.shape)
# print(planets.head())
# print(type(planets))

print(planets.dropna().describe())
