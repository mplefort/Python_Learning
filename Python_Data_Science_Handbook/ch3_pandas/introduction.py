import pandas as pd
import os

print(pd.__version__)

dir = os.path.join(os.getcwd(),'data-USstates')
print(dir)

pop = pd.read_csv(os.path.join(dir, 'state-population.csv'))
areas = pd.read_csv(os.path.join(dir, 'state-areas.csv'))
abbrevs = pd.read_csv(os.path.join(dir, 'state-abbrevs.csv'))


print(pop.head()); print(areas.head()); print(abbrevs.head())

