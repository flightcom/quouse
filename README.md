# Estimation immobilière

https://app.dvf.etalab.gouv.fr

## URLS

1. Choix du département

https://geo.api.gouv.fr/departements/44/communes?geometry=contour&format=geojson&type=commune-actuelle

```json
{
    "type": "Feature",
    "geometry": {
        "type": "Polygon",
        …
    },
    "properties": {
        "code": "69001",
        "codeDepartement": "69",
        "codeRegion": "84",
        "codesPostaux": ["69170"],
        "nom": "Affoux",
        "population": 358,
        "type": "Feature"
    }
}
```

1. Choix de la ville

https://cadastre.data.gouv.fr/bundler/cadastre-etalab/communes/44143/geojson/sections


```json
{
    "type":"Feature",
    "id":"44143000AB",
    "geometry":{
        "type":"MultiPolygon",
        "coordinates":[]
    },
    "properties":{
        "id":"44143000AB",
        "commune":"44143",
        "prefixe":"000",
        "code":"AB",
        "created":"2001-04-24",
        "updated":"2016-03-24"
    }
}
```

3. Choix de la parcelle

https://app.dvf.etalab.gouv.fr/api/mutations3/44143/000AB

{"id_mutation":"2019-177348","date_mutation":"2019-02-04","numero_disposition":"1","nature_mutation":"Vente","valeur_fonciere":"350000.0","adresse_numero":"23.0","adresse_suffixe":"None","adresse_nom_voie":"RUE DE LA CALIFORNIE","adresse_code_voie":"0540","code_postal":"44400","code_commune":"44143","nom_commune":"Rez\u00e9","code_departement":"44","ancien_code_commune":"None","ancien_nom_commune":"None","id_parcelle":"44143000AC0241","ancien_id_parcelle":"None","numero_volume":"None","lot1_numero":"None","lot1_surface_carrez":"nan","lot2_numero":"None","lot2_surface_carrez":"None","lot3_numero":"None","lot3_surface_carrez":"None","lot4_numero":"None","lot4_surface_carrez":"None","lot5_numero":"None","lot5_surface_carrez":"None","nombre_lots":"0","code_type_local":"4","type_local":"Local industriel. commercial ou assimil\u00e9","surface_reelle_bati":"1111.0","nombre_pieces_principales":"0.0","code_nature_culture":"S","nature_culture":"sols","code_nature_culture_speciale":"None","nature_culture_speciale":"None","surface_terrain":"1800.0","longitude":"-1.584724","latitude":"47.190804","section_prefixe":"000AC"},

### Itérations

loop:départements (01 à 95)
    get parcelles
    loop:parcelles
        get mutations
        loop:mutations
            organize data
            if: data.nature_mutation == 'Vente'
            save data# quouse
