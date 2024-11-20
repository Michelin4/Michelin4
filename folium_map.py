import folium
import pandas as pd
import webbrowser
import os


class Map:
    def __init__(self, data_point, zoom_start=13):
        self.center = (data_point['LATITUDE'], data_point['LONGITUDE'])
        self.zoom_start = zoom_start

        tiles = 'https://tile.jawg.io/jawg-dark/{z}/{x}/{y}{r}.png?access-token=yKizQwgQd53rog6n0kisnZ6OhqsNUzcMIhPKV7RLevpkLFUEfGrQbgW8SUl9GwKa'
        attr = (
            '<a href="https://jawg.io" title="Tiles Courtesy of Jawg Maps" target="_blank">&copy; <b>Jawg</b>Maps</a> &copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
        )
        self.map = folium.Map(location=self.center, zoom_start=self.zoom_start, tiles=tiles, attr=attr)

        self.dict_color_poi = {
            'highway': 'darkpurple',
            'tourism': 'orange',
            'shop': 'blue',
            'amenity': 'green',
            'surface': 'purple',
            'leisure': 'pink',
            'building': 'black',
        }

        self.dict_icons_poi = {  #from here https://fontawesome.com/v4/icons/
            'highway': 'road',
            'tourism': 'suitcase',
            'shop': 'shopping-bag',
            'amenity': 'bath',
            'surface': 'expand',
            'leisure': 'gift',
            'building': 'building',
        }

    def showMap(self):
        self.map.save("map.html")
        webbrowser.open("map.html")

    def marker(self, point, icon='circle', color='red', popup=False):

        lat, lon = point['LATITUDE'], point['LONGITUDE']
        if popup == True:

            injury_columns = ['FATAL', 'SERIOUS_INJURY', 'MINOR_INJURY', 'NO_INJURY']
            injury_type = point[injury_columns][point[injury_columns] == 1.0].index[0]

            person_involved = ['VH', 'CYC', 'PED']
            pi_type = point[person_involved][point[person_involved] == 1.0].index[0]

            popup = (
                "Latitude: " + str(point['LATITUDE'].round(2)) +
                "<br>Longitude: " + str(point['LONGITUDE'].round(2)) +
                "<br>Year: " + str(int(point['YEAR'])) +
                "<br>Person Involved: " + pi_type +
                "<br>Injury: " + injury_type +
                "<br>POI Type: " + str(point['nearest_poi_type']) +
                "<br>Distance to POI: " + str(point['distance_to_poi'].round(2))
            )


            folium.Marker(
                location=(lat, lon),
                icon=folium.Icon(color=color, prefix='fa', icon=icon),
                popup= folium.Popup(popup, min_width=300, max_width=300)
            ).add_to(self.map)
        else:
            folium.Marker(
                location=(lat, lon),
                icon=folium.Icon(color=color, prefix='fa', icon=icon)
            ).add_to(self.map)


def main():
    url = r'C:\Users\noamk\Documents\BTTAI\Git\Michelin4\processed_data\crash_data.csv'

    if os.path.exists(url):
        df = pd.read_csv(url)
        print(df.head())
    else:
        print(f"File not found: {url}")
        return

    data_point_1 = df.iloc[10]
    data_point_2 = df.iloc[11]
    data_point_3 = df.iloc[15]
    map = Map(data_point_1, zoom_start=13)
    map.marker(data_point_1, icon=map.dict_icons_poi[data_point_1['nearest_poi_type']], color=map.dict_color_poi[data_point_1['nearest_poi_type']], popup=True)
    map.marker(data_point_2, icon=map.dict_icons_poi[data_point_2['nearest_poi_type']], color=map.dict_color_poi[data_point_2['nearest_poi_type']], popup=True)
    map.marker(data_point_3, icon=map.dict_icons_poi[data_point_3['nearest_poi_type']], color=map.dict_color_poi[data_point_3['nearest_poi_type']], popup=True)
    map.showMap()

if __name__ == "__main__":
    main()
