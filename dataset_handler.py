import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Încărcarea dataset-ului
df = pd.read_csv("autos.csv", sep=",", encoding="utf-8")

# Eliminarea coloanelor irelevante
df = df.drop(['index', 'dateCrawled', 'dateCreated', 'lastSeen', 'nrOfPictures', 'postalCode', 'abtest'], axis=1)

# Completarea valorilor lipsă cu valori default
df['vehicleType'] = df['vehicleType'].fillna("Unknown")
df['gearbox'] = df['gearbox'].fillna("manual")
df['fuelType'] = df['fuelType'].fillna("Unknown")
df['notRepairedDamage'] = df['notRepairedDamage'].replace({"ja": 1, "nein": 0, "NoInfo": -1}).infer_objects(copy=False)

df['model'] = df['model'].fillna("Unknown")
df['price'] = pd.to_numeric(df['price'], errors='coerce')

# Aplicarea Label Encoding pe variabilele categorice
categorical_cols = ['seller', 'offerType', 'vehicleType', 'gearbox', 'fuelType', 'brand', 'model']
label_encoders = {}

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Salvăm encoderul pentru reconversie dacă e necesar

# Normalizarea variabilelor numerice (kilometraj și puterea motorului)
scaler = MinMaxScaler()
df[['kilometer', 'powerPS']] = scaler.fit_transform(df[['kilometer', 'powerPS']])

# Salvăm dataset-ul preprocesat
df.to_csv("processed_used_cars.csv", index=False)
print(df.dtypes)  # Vezi tipurile de date pentru fiecare coloană

print("✅ Preprocesarea datelor finalizată! Dataset-ul procesat a fost salvat ca 'processed_used_cars.csv'.")