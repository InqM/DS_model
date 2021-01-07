import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from scipy.spatial import distance
import copy
from IPython.display import display
import pandas as pd
import numpy as np
# import seaborn as sns
# import plotly.express as px
# import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import plotly
from sklearn.preprocessing import StandardScaler
from scipy.spatial import distance
import copy
from IPython.display import display
import warnings
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
from joblib import load


def predict(df):
    
    data = pd.read_csv('data.csv')
    song_name, artist = find_word(df[0][0], data, n)
    n = df[0][1]
        
    #Matches Song with the artist
    index = 0
    indices = []
    
    for i, item in enumerate(data['artists']):
        if(item == artist):
            indices.append(i)
    
    for item in indices:
        if(data['name'][item] == song_name):
            index = item
    
    #Gets Predictions with NN
    neighbor_predictions = model = load('app/model.joblib').kneighbors([df_s[index]])
    
    #Finds the prediction's names and who its by in the dictionary
    list_of_predictions = neighbor_predictions[1][0].tolist()

    ten_similar_tracks_title = []
    for item in list_of_predictions:
        track_hash = dictionary['name'].iloc[item]
        ten_similar_tracks_title.append(track_hash)
    
    ten_similar_tracks_artists = []
    for item in list_of_predictions:
        track_hash = dictionary['artists'].iloc[item]
        ten_similar_tracks_artists.append(track_hash)

    ten_suggestions = []
    for i in range(0,n+1):
        ten_suggestions.append(ten_similar_tracks_title[i] + ' by ' + ten_similar_tracks_artists[i])
        
    
    return ten_suggestions


def find_word(word,df,number):
    df.drop_duplicates(inplace=True)
    words=df['name'].values
    artists=df['artists'].values
    t=[]
    count=0
    if word[-1]==' ':
        word=word[:-1]
    for i in words:
        if word.lower() in i.lower():
            t.append([len(word)/len(i),count])
        else:
            t.append([0,count])
        count+=1
    t.sort(reverse=True)
    s=[[words[t[i][1]],artists[t[i][1]].strip('][').split(', ')] for i in range(number)]   
    songs=[words[t[i][1]] for i in range(number)]
    artist=[artists[t[i][1]] for i in range(number)]
    x=[]
    for i in s:
        l=''
        by=''
        for j in i[1]:
            by+=j
        l+=i[0]+' by '+by
        x.append(l)
    tup=[]
    for i in range(number):
        tup.append((x[i],i))

    
    return songs,artist