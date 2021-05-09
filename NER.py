# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from google.colab import drive
drive.mount("/content/drive")

#Bsemap Library to show the coordinates obtained on map
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_excel(r'C:\Python practise\lat-and-lon.xlsx','Sheet1')

fig = plt.figure(figsize=(12,9))

m = Basemap(projection='mill',
           llcrnrlat = -90,
           urcrnrlat = 90,
           llcrnrlon = -180,
           urcrnrlon = 180,
           resolution = 'c')

m.drawcoastlines()

m.drawparallels(np.arange(-90,90,10),labels=[True,False,False,False])
m.drawmeridians(np.arange(-180,180,30),labels=[0,0,0,1])

sites_lat_y = df['latitude'].tolist()
sites_lon_x = df['longitude'].tolist()

colors = ['green', 'darkblue', 'yellow', 'red', 'blue', 'orange']

m.scatter(sites_lon_x,sites_lat_y,latlon=True, s=500, c=colors, marker='o', alpha=1, edgecolor='k', linewidth=1, zorder=2)
m.scatter(-135,60,latlon=True, s=5000, c='blue', marker='^', alpha=1, edgecolor='k', linewidth=1, zorder=1)

plt.title('Basemap tutorial', fontsize=20)

plt.show()

import pandas as pd
import spacy
from tkinter import filedialog
from pandas import DataFrame
import random
import matplotlib.pyplot as plt

data=pd.read_csv("/content/drive/MyDrive/sm/disaster.csv")

#NER list we are using
nlp=spacy.load('en_core_web_sm') 

#Removing duplicate messages
messages=data.message.unique() 

#Converting every element to string
messages = [str(message) for message in messages]

docs=nlp.pipe(messages)

#Intialize lists and index
list_indexes=[]
list_entities=[]
list_label=[]
i=0
list_coordinates=[]



#Appending all the entities 
for doc in nlp.pipe(messages, batch_size=30000, n_threads=3): 
  for ent in doc.ents: 
    
    list_entities.append(ent.text)
    list_indexes.append(i)
    list_label.append(ent.label_)
  i+=1
  

#Creating a DataFrame and inserting the List(Indexes,Entity,Label)
entities_message_index_label = pd.DataFrame({'Message': list_indexes, 'Entity': list_entities, 'Label': list_label})

entities_message_index_label.head()

#For finding the clusters of locations using K-mean Clustering
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
from sklearn.datasets.samples_generator import make_blobs


X, y_true = make_blobs(n_samples=40209, centers=23,
                       cluster_std=0.60, random_state=0)
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=23)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
print(centers)

#data.drop(["messages"], axis = 1, inplace = True
#!pip install mysql.connector
#!pip install pymysql
#CONNECTING MYSQL TO RETRIEVE DATA
from sqlalchemy import create_engine

MYSQL_HOSTNAME = 'aika-prod.ckfjfiqb1qvb.us-east-2.rds.amazonaws.com' 
MYSQL_USER = 'FILL_IN'
MYSQL_PASSWORD = 'FILL_IN'
MYSQL_DATABASE = 'FILL_IN'

connection_string = f'mysql+pymysql://{MYSQL_USER}:{MYSQL_PASSWORD}@{MYSQL_HOSTNAME}/{MYSQL_DATABASE}'
connect_args = {'ssl': {'ca': '/content/drive/MyDrive/sm/entity.csv'}}

db = create_engine(connection_string, connect_args=connect_args)

query = """SELECT MAX(DISTINCT Entity) FROM entities_message_index_label
           WHERE Label="GPE";""".format(MYSQL_DATABASE)
query1 = """SELECT * FROM enitities_message_index_label WHERE 
            Label = "DATE";""".format(MYSQL_DATABASE)

events_df = pd.read_sql(query,query1 con=db)
events_df
#Saving the enities to a csv file on drive
entities_message_index_label.to_csv(r'/content/drive/MyDrive/sm/entity.csv', index=False) 
print("New message about any help")
new_message = input()
med_help = input()
new_row = pd.DataFrame({'message':new_message, 'original':'NULL', 'genre':0, 
                        'related':'NULL', 'Pll':33, 'request':'nill', 
                        'offer':0, 'aid_related':'0', 'medical_help':med_help, }, 
                                                            index =[0]) 
data = pd.concat([new_row, data]).reset_index(drop = True) 
