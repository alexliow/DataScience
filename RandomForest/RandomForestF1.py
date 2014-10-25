
# coding: utf-8

import pandas as pd

df = pd.read_csv('f1data.csv', header=0)

df['Teams'] = df['team'].map( {"BAR-Honda" : 0,
"BMW" : 1,
"BMW Sauber" : 2,
"BMW Sauber-Ferrari" : 3,
"Brawn-Mercedes" : 4,
"Caterham-Renault" : 5,
"Ferrari" : 6,
"Force India-Ferrari" : 7,
"Force India-Mercedes" : 8,
"Honda" : 9,
"HRT-Cosworth" : 10,
"Jaguar-Cosworth" : 11,
"Jordan-Ford" : 12,
"Jordan-Toyota" : 13,
"Lotus-Cosworth" : 14,
"Lotus-Renault" : 15,
"Marussia-Cosworth" : 16,
"Marussia-Ferrari" : 17,
"McLaren-Mercedes" : 18,
"Mercedes" : 19,
"Mercedes GP" : 20,
"Minardi-Cosworth" : 21,
"RBR-Cosworth" : 22,
"RBR-Ferrari" : 23,
"RBR-Renault" : 24,
"Red Bull Racing-Renault" : 25,
"Red Bull-Renault" : 26,
"Renault" : 27,
"Sauber-BMW" : 28,
"Sauber-Ferrari" : 29,
"Sauber-Petronas" : 30,
"Spyker-Ferrari" : 31,
"STR-Cosworth" : 32,
"STR-Ferrari" : 33,
"STR-Renault" : 34,
"Super Aguri-Honda" : 35,
"Toyota" : 36,
"Virgin-Cosworth" : 37,
"Williams-BMW" : 38,
"Williams-Cosworth" : 39,
"Williams-Mercedes" : 40,
"Williams-Renault" : 41,
"Williams-Toyota" : 42} ).astype(int)

df['Country'] = df['race_country'].map( {"Abu Dhabi" : 0,
"Australia" : 1,
"Austria" : 2,
"Bahrain" : 3,
"Belgium" : 4,
"Brazil" : 5,
"Canada" : 6,
"China" : 7,
"Europe" : 8,
"France" : 9,
"Germany" : 10,
"Great Britain" : 11,
"Hungary" : 12,
"India" : 13,
"Italy" : 14,
"Japan" : 15,
"Korea" : 16,
"Malaysia" : 17,
"Monaco" : 18,
"Russia" : 19,
"San Marino" : 20,
"Singapore" : 21,
"Spain" : 22,
"Turkey" : 23,
"United States" : 24} ).astype(int)

df = df.drop(['#race_year', 'race_id', 'race_country', 'car_number', 'driver_name', 'team', 'laps'], axis = 1)

cols = df.columns.tolist()

cols = cols[1:] + cols [:1]

df = df[cols]

train_data = df.values

from sklearn.ensemble import RandomForestClassifier 

forest = RandomForestClassifier(n_estimators = 100)

forest = forest.fit(train_data[0::,1::],train_data[0::,0])

df2 = pd.read_csv('test_data.csv', header=0)

df2['Teams'] = df2['team'].map( {"BAR-Honda" : 0,
"BMW" : 1,
"BMW Sauber" : 2,
"BMW Sauber-Ferrari" : 3,
"Brawn-Mercedes" : 4,
"Caterham-Renault" : 5,
"Ferrari" : 6,
"Force India-Ferrari" : 7,
"Force India-Mercedes" : 8,
"Honda" : 9,
"HRT-Cosworth" : 10,
"Jaguar-Cosworth" : 11,
"Jordan-Ford" : 12,
"Jordan-Toyota" : 13,
"Lotus-Cosworth" : 14,
"Lotus-Renault" : 15,
"Marussia-Cosworth" : 16,
"Marussia-Ferrari" : 17,
"McLaren-Mercedes" : 18,
"Mercedes" : 19,
"Mercedes GP" : 20,
"Minardi-Cosworth" : 21,
"RBR-Cosworth" : 22,
"RBR-Ferrari" : 23,
"RBR-Renault" : 24,
"Red Bull Racing-Renault" : 25,
"Red Bull-Renault" : 26,
"Renault" : 27,
"Sauber-BMW" : 28,
"Sauber-Ferrari" : 29,
"Sauber-Petronas" : 30,
"Spyker-Ferrari" : 31,
"STR-Cosworth" : 32,
"STR-Ferrari" : 33,
"STR-Renault" : 34,
"Super Aguri-Honda" : 35,
"Toyota" : 36,
"Virgin-Cosworth" : 37,
"Williams-BMW" : 38,
"Williams-Cosworth" : 39,
"Williams-Mercedes" : 40,
"Williams-Renault" : 41,
"Williams-Toyota" : 42} ).astype(int)

df2['Country'] = df2['race_country'].map( {"Abu Dhabi" : 0,
"Australia" : 1,
"Austria" : 2,
"Bahrain" : 3,
"Belgium" : 4,
"Brazil" : 5,
"Canada" : 6,
"China" : 7,
"Europe" : 8,
"France" : 9,
"Germany" : 10,
"Great Britain" : 11,
"Hungary" : 12,
"India" : 13,
"Italy" : 14,
"Japan" : 15,
"Korea" : 16,
"Malaysia" : 17,
"Monaco" : 18,
"Russia" : 19,
"San Marino" : 20,
"Singapore" : 21,
"Spain" : 22,
"Turkey" : 23,
"United States" : 24} ).astype(int)

df2 = df2.drop(['race_country','team'], axis = 1)

cols2 = df2.columns.tolist()

cols2 = cols2[1:] + cols2[:1]

cols2 = cols2[1:] + cols2[:1]

df2 = df2[cols2]

test_data = df2.values

output = forest.predict(test_data)
