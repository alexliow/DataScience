import pandas as pd
import numpy as np
df2 = pd.read_csv('../f1dataWith2014.csv', header=0)
array = []

# Create a dictionary
for i in range(0, 2815):
    current = dict(year=df2['#race_year'][i], raceID=df2['race_id'][i], country=df2['race_country'][i], team=df2['team'][i], time=df2['time_in_milliseconds_measured_or_inferred'][i], startPos=df2['grid_starting_position'][i], rank=df2['finishing_position'][i]) 
    array.append(current)

# Split each category into table with 1's and 0's 
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
newArray = vec.fit_transform(array).toarray()

# Write to a file and give table with each row representing attributes of single driver
csvFile = "/Users/alexliow/Desktop/Data Sci/DataScienceRepo/RandomForest/revisedDataWith2014.csv"
with open(csvFile, "w") as output:
    writer = csv.writer(output, lineterminator='\n')
    writer.writerows(newArray)

#IN BETWEEN I TOOK THE DATA IN revised Data and formatted them into train and test data and removed rank information from test_data

# Create a train data df
df2 = pd.read_csv('it2Train.csv')
train_data = df2.values
train_data

# Random Forest from sklearn
from sklearn.ensemble import RandomForestClassifier 
forest = RandomForestClassifier(n_estimators = 100)
forest = forest.fit(train_data[0::,1::],train_data[0::,0])

# Read in the test_data and put into df then predict
df3 = pd.read_csv('it2Test.csv')
test_data = df2.values
output = forest.predict(test_data)
output