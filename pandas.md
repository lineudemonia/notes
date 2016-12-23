##Data manipulation

~~~
import pandas as pd

titanic_survival 
= pd.read_csv("titanic_survival.csv")

print(titanic_survival.shape)
titanic_survival['age'].describe()


age = titanic_survival["age"]
age_is_null = pandas.isnull(age)
age_null_true = age[age_is_null]
age_null_count = len(age[age_is_null])

age_is_null = pd.isnull(titanic_survival["age"])
correct_mean_age = titanic_survival["age"][age_is_null == False].mean()

correct_mean_fare = titanic_survival["fare"].mean()

passenger_classes = [1, 2, 3]
fares_by_class = {}
fares_by_class = {}
for p_class in passenger_classes:
    fares_by_class[p_class] = titanic_survival[titanic_survival['pclass'] == p_class]['fare'].mean()
    
import numpy as np
port_stats = titanic_survival.pivot_table(index = 'embarked', values = ['fare', 'survived'], aggfunc = np.sum)

drop_na_columns = titanic_survival.dropna(axis = 'index')
new_titanic_survival = titanic_survival.dropna(axis=0, subset = ['age', 'sex'])

# .iloc seletcts the integer locations (top 10 regardless of index after sorting), .loc selects index
first_ten_rows = new_titanic_survival.iloc[:10]
row_position_fifth = new_titanic_survival.iloc[4]
row_index_25 = new_titanic_survival.loc[25]

first_row_first_column = new_titanic_survival.iloc[0,0]
all_rows_first_three_columns = new_titanic_survival.iloc[:,0:3]
row_index_83_age = new_titanic_survival.loc[83,"age"]
row_index_1000_pclass = new_titanic_survival.loc[766,"pclass"] 

titanic_reindexed = new_titanic_survival.reset_index(drop = True)

# This function returns the hundredth item from a series
def hundredth_row(column):
    # Extract the hundredth item
    hundredth_item = column.iloc[99]
    return hundredth_item

# Return the hundredth item from each column
hundredth_row = titanic_survival.apply(hundredth_row)

~~~

Series objects use NumPy arrays for fast computation, but build on them by adding valuable features for analyzing data. For example, while NumPy arrays utilize an integer index, Series objects can utilize other index types, like a string index. Series objects also allow for mixed data types and utilize the NaN Python value for handling missing values.


~~~
# create a sub-df from data frame

from pandas import Series

film_names = data["FILM"].values
rt_scores = data["SCORE"].values

series_custom = Series(rt_scores, index = film_names)
series_film = series_custom[['Minions (2015)', 'Leviathan (2014)']]

print(series_film)


##Pandas preserve the linkage between rows in sorting

sc2 = series_custom.sort_index()
sc3 = series_custom.sort_values()
print(sc2.head(10))
print(sc3.head(10))


films = ["The Lazarus Effect (2015)", "Gett: The Trial of Viviane Amsalem (2015)", "Mr. Holmes (2015)"]

best_movies_ever = fandango_films.loc[films]

# Shows response in a summary table
data['What is typically the main dish at your Thanksgiving dinner?'].value_counts()

~~~

##Visualization


### Line charts
~~~
plt.plot(first_twelve['DATE'], first_twelve['VALUE'])
plt.xticks(rotation = 90)
plt.xlabel('Month')
plt.ylabel('Unemployment Rate')
plt.title('Monthly Unemployment Trends, 1948')
plt.show()
~~~

### Bar chart

~~~
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
fig, ax = plt.subplots()
bar_heights = norm_reviews.ix[0, num_cols].values
bar_positions = arange(5) + 0.75
tick_positions = range(1,6)
left = bar_positions
height = bar_heights
ax.bar(left, height, width = 0.5)
ax.set_xticks(tick_positions)
ax.set_xticklabels(num_cols, rotation = 90)
plt.xlabel('Rating Source')
plt.ylabel('Average Rating')
plt.title('Average User Rating For Avengers: Age of Ultron (2015)')
plt.show()

~~~

### Horizontal bar chart
~~~
import matplotlib.pyplot as plt
from numpy import arange
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue', 'Fandango_Stars']
bar_widths = norm_reviews.ix[0, num_cols].values
bar_positions = arange(5) + 0.75
tick_positions = range(1,6)
fig, ax = plt.subplots()
ax.barh(bar_positions, bar_widths, 0.5)
ax.set_yticks(tick_positions)
ax.set_yticklabels(num_cols)
ax.set_ylabel('Rating Source')
ax.set_xlabel('Average Rating')
ax.set_title('Average User Rating For Avengers: Age of Ultron (2015)')
plt.show()
~~~

### Scatter plot

~~~
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.scatter(norm_reviews['Fandango_Ratingvalue'], norm_reviews['RT_user_norm'])
ax.set_xlabel('Fandango')
ax.set_ylabel('Rotten Tomatoes')
plt.show()
~~~

### Histogram

~~~
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.hist(norm_reviews['Fandango_Ratingvalue'], range = (0, 5))
plt.show()

#  With subplots
fig = plt.figure(figsize=(5,20))
ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(4,1,3)
ax4 = fig.add_subplot(4,1,4)
ax1.hist(norm_reviews['Fandango_Ratingvalue'], bins = 20, range = (0, 5))
ax1.set_title('Distribution of Fandango Ratings')
ax2.hist(norm_reviews['RT_user_norm'], bins = 20, range = (0, 5))
ax2.set_title('Distribution of Rotten Tomatoes Ratings')
ax3.hist(norm_reviews['Metacritic_user_nom'], bins = 20, range = (0, 5))
ax3.set_title('Distribution of Metacritic Ratings')
ax4.hist(norm_reviews['IMDB_norm'], bins = 20, range = (0,5))
ax4.set_title('Distribution of IMBD Ratings')
ax1.set_ylim(0, 50)
ax2.set_ylim(0, 50)
ax3.set_ylim(0, 50)
ax4.set_ylim(0, 50)
ax1.set_ylabel('Frequency')
ax2.set_ylabel('Frequency')
ax3.set_ylabel('Frequency')
ax4.set_ylabel('Frequency')
plt.show()
~~~

### Box chart
~~~
#multiple box charts

import matplotlib.pyplot as plt
fig, ax = plt.subplots()
num_cols = ['RT_user_norm', 'Metacritic_user_nom', 'IMDB_norm', 'Fandango_Ratingvalue']
ax.boxplot(norm_reviews[num_cols].values)
ax.set_ylim(0, 5)
ax.set_xticklabels(num_cols, rotation = 90)
plt.show()
~~~

### Code beautify

~~~

# Take out ticks on axis as well as spine lines
fig, ax = plt.subplots()
ax.plot(women_degrees['Year'], women_degrees['Biology'], label='Women')
ax.plot(women_degrees['Year'], 100-women_degrees['Biology'], label='Men')
ax.tick_params(bottom="off", top="off", left="off", right="off")
for key, spine in ax.spines.items():
    spine.set_visible(False)
ax.legend(loc = 'upper right')
plt.show()

~~~

~~~ 
#Change color to color-blind 10 palette
#change line width from 2 to 3
#change display from 2 by 2 to 1 line and 6 columns

stem_cats = ['Engineering', 'Computer Science', 'Psychology', 'Biology', 'Physical Sciences', 'Math and Statistics']

fig = plt.figure(figsize=(18, 3))

for sp in range(0,6):
    ax = fig.add_subplot(1,6,sp+1)
    ax.plot(women_degrees['Year'], women_degrees[stem_cats[sp]], c=cb_dark_blue, label='Women', linewidth=3)
    ax.plot(women_degrees['Year'], 100-women_degrees[stem_cats[sp]], c=cb_orange, label='Men', linewidth=3)
    for key,spine in ax.spines.items():
        spine.set_visible(False)
    ax.set_xlim(1968, 2011)
    ax.set_ylim(0,100)
    ax.set_title(stem_cats[sp])
    ax.tick_params(bottom="off", top="off", left="off", right="off")

plt.legend(loc='upper right')
plt.show()
~~~

## Seaborn
~~~
import seaborn as sns
import pandas as pd
import matplotlib as plt

~~~
###kernel density plot with histogram
~~~
import seaborn as sns
import matplotlib.pyplot as plt
sns.distplot(titanic['Age'])
plt.show()
~~~

###Just kernel density plot with shading = True
~~~
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.kdeplot(titanic['Age'], shade = True)
plt.xlabel("Age")
plt.plot()

~~~

### Some beautifying

seaborn.set_style() 

- darkgrid: Coordinate grid displayed, dark background color
- whitegrid: Coordinate grid displayed, white background color
- dark: Coordinate grid hidden, dark background color
- white: Coordinate grid hidden, white background color
- ticks: Coordinate grid hidden, white background color, ticks visible

~~~
# Set spine (axis)

sns.set_style('darkgrid')
sns.kdeplot(titanic['Age'], shade = True)
sns.despine(left = True, bottom = True)
plt.xlabel('Age')

~~~

### Conditional distributions
~~~

# Condition on unique values of the "Survived" column.
g = sns.FacetGrid(titanic, col="Survived", size=6)
# For each subset of values, generate a kernel density plot of the "Age" columns.
g.map(sns.kdeplot, "Age", shade=True)
~~~
Seaborn handled:

subsetting the data into rows (facet means subset) )where Survived is 0 and where Survived is 1
creating both Axes objects, ensuring the same axis scales
plotting both kernel density plots.
FacetGrid.map() can take both SNS or PLT arguments

### Multiple conditional distributions

~~~
g = sns.FacetGrid(titanic, col="Survived", row="Pclass")
g.map(sns.kdeplot, "Age", shade=True)
sns.despine(left=True, bottom=True)
plt.show()
~~~

#### it will create 3 * 2 graphs with Pclass = [1, 2, 3] and Survived = [0, 1]

### Add more feature (with different hues) and relevant legend
~~~
import seaborn as sns
g = sns.FacetGrid(titanic, col = 'Survived', row = 'Pclass', hue = 'Sex', size = 3)
g.map(sns.kdeplot, 'Age', shade = True).add_legend()
sns.despine(left = True, bottom = True)
plt.show()
~~~


## Basemap for dealing with geo data

To create a new instance of the basemap class, we call the basemap constructor and pass in values for the required parameters:

- projection: the map projection.
- llcrnrlat: latitude of lower left hand corner of the desired map domain
- urcrnrlat: latitude of upper right hand corner of the desired map domain
- llcrnrlon: longitude of lower left hand corner of the desired map domain
- urcrnrlon: longitude of upper right hand corner of the desired map domain

~~~

import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
m = Basemap(projection='merc',llcrnrlat=-80,urcrnrlat=80,llcrnrlon=-180,urcrnrlon=180)
longitudes = airports["longitude"].tolist()
latitudes = airports["latitude"].tolist()
x, y = m(longitudes, latitudes)
m.scatter(x, y, s = 1) # s is size
m.drawcoastlines() # draw coastline
plt.show()

~~~

### Basemap great circles
Cannot handle situations in which the great circle intersects the edge of the map projection domain, and then re-enters the domain. The following code generates a great circle for the first three routes in the dataframe in the following parameter order: 

- lon1 - longitude of the starting point.
- lat1 - latitude of the starting point.
- lon2 - longitude of the ending point.
- lat2 - latitude of the ending point.

~~~
m.drawgreatcircle(39.956589, 43.449928, 49.278728, 55.606186)
m.drawgreatcircle(48.006278, 46.283333, 49.278728, 55.606186)
m.drawgreatcircle(39.956589, 43.449928, 43.081889 , 44.225072)
~~~

~~~
fig, ax = plt.subplots(figsize=(15,20))
m = Basemap(projection='merc', llcrnrlat=-80, urcrnrlat=80, llcrnrlon=-180, urcrnrlon=180)
m.drawcoastlines()

def create_great_circles(df):
    # Add your code here.
    for index, row in df.iterrows():
        if abs(int(row['end_lat']) - int(row['start_lat'])) < 180 and abs(int(row['end_lon']) - int(row['start_lon'])) < 180:
            m.drawgreatcircle(row['start_lon'], row['start_lat'], row['end_lon'], row['end_lat'])

dfw = geo_routes[geo_routes['source'] == "DFW"]
create_great_circles(dfw)
plt.show()
~~~