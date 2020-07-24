# Import libraries
import os
import math
import requests
import urllib.request
from datetime import datetime
import time
import json
import sys
import numpy as np
import pandas as pd
import glob
from bs4 import BeautifulSoup
from helpers import mixrange
from mappings import normalize__type_local

# Set the URL you want to webscrape from
# url = 'https://app.dvf.etalab.gouv.fr/'

departements_list = "1-95,971-976"
departements_list = "44,49,85"
# departements_list = "1"

results = []
count = 0

# Loop over departements
for departement in mixrange(departements_list):
    departement_str = '{}'.format(departement).zfill(2)
    print('> {}'.format(departement_str))
    url = 'https://geo.api.gouv.fr/departements/{}/communes?geometry=contour&format=geojson&type=commune-actuelle'.format(departement_str)

    response = requests.get(url)

    json_data = json.loads(response.text)
    
    # Loop over departement cities
    for city in json_data['features']:
        print(city['properties'])
        code = city['properties']['code']
        print('  > {}'.format(code))
        url = "https://cadastre.data.gouv.fr/bundler/cadastre-etalab/communes/{}/geojson/sections".format(code)
        # print(url)
        response = requests.get(url)
        json_data = json.loads(response.text)

        # Loop over parcelles
        for parcelle in json_data['features']:
            code = parcelle['properties']['code']
            prefixe = parcelle['properties']['prefixe']
            commune = parcelle['properties']['commune']
            print('    > {}'.format(code))

            # url = 'https://geo.api.gouv.fr/departements/44/communes?geometry=contour&format=geojson&type=commune-actuelle'
            url = 'https://app.dvf.etalab.gouv.fr/api/mutations3/{}/{}{}'.format(commune, prefixe, code)
   
            response = requests.get(url)
            json_data = json.loads(response.text)
            # print(json_data)

            # Loop over mutations (operations du cadastre)
            for mutation in json_data['mutations']:
                # print(mutation['surface_terrain'])
                if (mutation['type_local'] == 1 or mutation['type_local'] == 2):
                    break

                if (mutation['longitude'] == 'None' 
                    or mutation['latitude'] == 'None'
                    or mutation['surface_terrain'] == 'None'):
                    break

                result = {
                    # "date_mutation": mutation['date_mutation'],
                    # "date_mutation": time.mktime(datetime.strptime(mutation['date_mutation'], "%Y-%m-%d").timetuple()),
                    "date_mutation": datetime.strptime(mutation['date_mutation'], "%Y-%m-%d").year,
                    # "city": city['properties']['code'],
                    "city": mutation['nom_commune'],
                    "valeur_fonciere": mutation['valeur_fonciere'],
                    # "type_local": normalize__type_local(mutation['type_local']),
                    "type_local": mutation['type_local'],
                    # "type_local": mutation['type_local'],
                    "surface_reelle_bati": mutation['surface_reelle_bati'],
                    "nombre_pieces_principales": mutation['nombre_pieces_principales'],
                    "surface_terrain": float(mutation['surface_terrain']),
                    # "longitude": '%.6f'%float(mutation['longitude']),
                    # "latitude": '%.6f'%float(mutation['latitude']),
                    # "longitude": float(mutation['longitude']),
                    # "latitude": float(mutation['latitude']),
                }
                results.append(result)

            if len(results) > 0:
                df = pd.DataFrame(results)
                # Remove nan values
                # df.dropna(subset=['surface_terrain', 'surface_reelle_bati', 'longitude', 'latitude'], inplace=True)
                df.dropna(subset=['surface_terrain', 'surface_reelle_bati'], inplace=True)
                # Remove duplicates
                # columns_dedup = ['city','date_mutation','latitude','longitude','valeur_fonciere']
                # columns_dedup = ['city','date_mutation','valeur_fonciere']
                # df.drop_duplicates(keep='first', subset=columns_dedup, inplace=True) 
                # Filtering unset data
                df = df[df['nombre_pieces_principales'] != 'None']
                df = df[df['surface_reelle_bati'] != 'None']
                df = df[df['type_local'] != 'None']
                count += df.shape[0]
                outdir = 'data/unmerged/{}'.format(departement_str)
                outfile = '{}/{}-{}{}.csv'.format(outdir, commune, prefixe, code)
                if not os.path.exists(outdir):
                    os.mkdir(outdir)
                df.to_csv(outfile)
                results = []

            if count >= 100:
                break

        else:
            continue
        break

    else:
        continue
    break



# Merge data
path = r'./data' # use your path
all_files = glob.glob(path + "/unmerged/**/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=0, header=0, engine='python')
    li.append(df)
df = pd.concat(li, axis=0, ignore_index=True, sort=True)

df = df.reset_index()
first_column = df.columns[0]
df = df.drop([first_column], axis=1)

# Save merged & filtered data
merge_outfile = 'data/merged/{}.csv'.format(datetime.now(tz=None))
# Shuffle rows
df = df.sample(frac=1).reset_index(drop=True)
df.to_csv(merge_outfile)

train_file = 'data/etalab/train.csv'
test_full_file = 'data/etalab/test_full.csv'
test_file = 'data/etalab/test.csv'

# Populate train.csv
df = pd.read_csv(merge_outfile, index_col=0, header=0, engine='python')
# df = pd.read_csv(merge_outfile, header=0, engine='python')
n = 75
df = df.head(int(len(df)*(n/100)))
df.index.name = 'id'
df.to_csv(train_file)

# Populate test_full.csv
df = pd.read_csv(merge_outfile, index_col=0, header=0, engine='python')
df = df.tail(len(df) - int(len(df)*(n/100)))
df.index.name = 'id'
df.to_csv(test_full_file)

# Populate test.csv
cols = ['city', 'date_mutation', 'latitude', 'longitude', 'nombre_pieces_principales', 'surface_reelle_bati', 'surface_terrain', 'type_local']
df = df[cols]
df.index.name = 'id'
df.to_csv(test_file)
