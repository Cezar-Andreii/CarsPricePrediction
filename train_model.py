from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import pandas as pd

# Încărcăm dataset-ul procesat
df = pd.read_csv("processed_used_cars.csv")

# Împărțim datele în features și label
df= df.drop(columns=["name"])

X = df.drop("price", axis=1)
y = df["price"]

# Împărțim în set de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creăm modelul RandomForest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# Antrenăm modelul
rf_model.fit(X_train, y_train)

# Facem predicții
y_pred = rf_model.predict(X_test)

# Evaluăm performanța modelului
mae = mean_absolute_error(y_test, y_pred)
print(f"📊 Mean Absolute Error: {mae}")