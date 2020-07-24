def normalize__type_local(type_local: str):
    if type_local == 'Maison':
        return 1
    elif type_local == 'Appartement':
        return 2
    elif type_local == 'Dépendance':
        return 3
    elif type_local == 'Local industriel. commercial ou assimilé':
        return 4
    else:
        return 'None'