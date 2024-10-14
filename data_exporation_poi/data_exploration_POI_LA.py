import os
import pandas as pd
import matplotlib.pyplot as plt

filename = os.path.join(os.getcwd(), "Documents\BTTAI\data_exporation_poi", "data.csv")
poi_data = pd.read_csv(filename)


print("Basic csv info: ")
print(poi_data.shape)
print(poi_data.columns)
print(poi_data.head(5))

print(poi_data.isnull().sum()) #the data doesn't have any nulls! 

#the data set contains an acacident ID, cordinents, location and point of interest (poi) and the type of poi
print("----------------------------------")
print("Counts unique: ")

print(poi_data['location'].unique())
print(poi_data['poi'].unique())
print(poi_data['poi_type'].unique())

#it seems the location is only in LA, so I removed the column as it doesn't serves any additional information
# As for the poi and type of poi, depends on the application we choose, we might want to eliminate one or use both of them
# to be for spesific.
print("----------------------------------")
print("Changes in columns: ")
poi_data = poi_data.drop('location', axis=1)
print(poi_data.columns)

#Count and Histogram for the poi column - not balanced, there is a lot of bus related accidents (poi and poi type correlates, same inconsistency).
#based on the problem, we should alternate it.

print("----------------------------------")
print("Counts: ")

print(poi_data['poi'].value_counts())

poi_data['poi'].hist()
plt.xlabel('POI')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.show()

# Poi type
print(poi_data['poi'].value_counts())

poi_data['poi_type'].hist()
plt.xlabel('POI Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)

plt.show()
