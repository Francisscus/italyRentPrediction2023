import pandas as pd
import numpy as np
import joblib
# import streamlit as st

def main():
    pass

def load_model(path):
    artifact = joblib.load(path)
    return artifact["model"], artifact["features"], artifact["pandas_version"]

def insert_data():
    surface = float(input("Inserisci superfice: " ))
    city = str(input("Inserisci città: "))
    housing_unit = str(input("Inserisci tipo di abitazione: "))
    floor = float(input("A quale piano si trova? ")) 
    num_rooms = float(input("Quante stanze ha? "))

    print(surface)
    print(city)
    print(housing_unit)
    print(floor)
    print(num_rooms)

    df_predict = pd.DataFrame(columns=['Surface', 'City', 'Housing_unit', 'floor',  'num_rooms'])

    df_predict.loc[len(df_predict)] = {
        'Surface' : surface,
        'City': city,
        'Housing_unit': housing_unit,
        'floor': floor,
        'num_rooms':num_rooms
    }

    city_size = {
        "Roma": "big",
        "Milano": "big",
        "Napoli": "big",
        "Torino": "mid-big",
        "Palermo": "mid-big",
        "Genova": "mid-big",
        "Bologna": "mid",
        "Firenze": "mid",
        "Bari": "mid",
        "Catania": "mid",
        "Verona": "mid",
        "Venezia": "mid",
        "Messina": "mid",
        "Padova": "mid",
        "Trieste": "mid",
        "Brescia": "mid",
        "Taranto": "mid",
        "Prato": "mid",
        "Parma": "mid-small",
        "Modena": "mid-small"
    }

    df_predict["city_size"] = df_predict["City"].map(city_size)

    #######################################################################

    city_macroregion = {
        "Milano": "north",
        "Torino": "north",
        "Genova": "north",
        "Bologna": "north",
        "Verona": "north",
        "Venezia": "north",
        "Padova": "north",
        "Trieste": "north",
        "Parma": "north",
        "Brescia": "north",
        "Modena": "north",
        "Prato": "center",
        "Roma": "center",
        "Firenze": "center",
        "Napoli": "south",
        "Palermo": "south",
        "Bari": "south",
        "Catania": "south",
        "Messina": "south",
        "Taranto": "south"
    }

    df_predict["macroregion"] = df_predict["City"].map(city_macroregion)

    #######################################################################

    bins = [-np.inf, 40, 70, 100, 200, 400, np.inf]
    labels = [
        "very small",
        "small",
        "mid-small",
        "mid",
        "large",
        "very large"
    ]

    df_predict["surface_bracket"] = pd.cut(
        df_predict["Surface"],
        bins=bins,
        labels=labels,
        right=False
    )

    #######################################################################

    df_predict = df_predict[['Surface', 'City', 'Housing_unit', 'city_size', 'macroregion', 'floor',
        'num_rooms', 'surface_bracket']]
    
    df_predict[['City', 'Housing_unit', 'city_size', 
                   'macroregion', 'surface_bracket']] = df_predict[['City', 'Housing_unit', 'city_size', 'macroregion', 'surface_bracket']].astype('category')
    
    # print(df_predict)
    return df_predict

def predict_rent(model, X):
    return model.predict(X).item()




if __name__ == "__main__":
    main()

    model, features, version = load_model("models/lgbm_regressor_v1.pkl")
    print(version)

    df = insert_data()
    print(df)
    print("\n")
    
    # prediction = model.predict(df)
    prediction = int(predict_rent(model, df))
    print("\n")
    print("------------- PREDICTION -------------")
    print("\n")
    print(f"Il canone di affitto ideale è di {prediction} €")
    print("\n")
    print("--------------------------------------")

