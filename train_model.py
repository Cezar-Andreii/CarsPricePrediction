import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from lightgbm import LGBMRegressor
import time
import joblib

# Încărcăm dataset-ul procesat
df = pd.read_csv("processed_used_cars.csv")

# Eliminăm numele (dacă nu l-ai eliminat deja) și verificăm pentru valori lipsă
if 'name' in df.columns:
    df = df.drop(columns=["name"])

# Tratăm outlier-ele de preț
q_low = df["price"].quantile(0.01)
q_high = df["price"].quantile(0.99)
df_cleaned = df[(df["price"] > q_low) & (df["price"] < q_high)]

# Feature engineering
df_cleaned['price_per_ps'] = df_cleaned['price'] / (df_cleaned['powerPS'] + 1)  # +1 pentru a evita împărțirea la zero

# Dacă există anul înmatriculării, calculăm vechimea mașinii
if 'yearOfRegistration' in df_cleaned.columns:
    current_year = 2025
    df_cleaned['car_age'] = current_year - df_cleaned['yearOfRegistration']

# Definiți X și y
X = df_cleaned.drop("price", axis=1)
y = df_cleaned["price"]

# Împărțim în set de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Funcție pentru evaluarea și afișarea rezultatelor pentru mai multe modele
def evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    # Măsurăm timpul de antrenare
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time

    # Facem predicții
    y_pred = model.predict(X_test)

    # Calculăm metricile
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    # Cross-validation
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
    cv_mae = -cv_scores.mean()

    # Returnăm rezultatele
    return {
        'Model': model_name,
        'MAE': mae,
        'RMSE': rmse,
        'R²': r2,
        'CV MAE': cv_mae,
        'Training Time (s)': training_time,
        'Predicții': y_pred[:5],  # Doar primele 5 predicții ca exemplu
        'Model Object': model
    }


# Creăm și evaluăm diferite modele
models = [
    (RandomForestRegressor(n_estimators=100, random_state=42), "Random Forest (baseline)"),
    (RandomForestRegressor(n_estimators=200, max_depth=20, min_samples_split=5, random_state=42),
     "Random Forest (tuned)"),
    (xgb.XGBRegressor(objective='reg:squarederror', random_state=42), "XGBoost (default)"),
    (xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, max_depth=7,
                      learning_rate=0.1, subsample=0.8, colsample_bytree=0.8, random_state=42), "XGBoost (tuned)"),
    (LGBMRegressor(random_state=42), "LightGBM (default)")
]

results = []
for model, name in models:
    result = evaluate_model(model, name, X_train, X_test, y_train, y_test)
    results.append(result)
    print(f"Model: {name} | MAE: {result['MAE']:.2f} | RMSE: {result['RMSE']:.2f} | R²: {result['R²']:.4f}")

# Convertim rezultatele în DataFrame pentru o vizualizare mai ușoară
results_df = pd.DataFrame([{k: v for k, v in r.items() if k != 'Model Object' and k != 'Predicții'}
                           for r in results])

# Salvăm cel mai bun model
best_model_idx = results_df['MAE'].idxmin()
best_model = results[best_model_idx]['Model Object']
best_model_name = results[best_model_idx]['Model']
joblib.dump(best_model, f"best_car_price_model_{best_model_name.replace(' ', '_').lower()}.pkl")
print(f"\nCel mai bun model: {best_model_name} (salvat ca model)")

# Calculăm și afișăm importanța featurelor pentru cel mai bun model
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)

    print("\nTop 10 cele mai importante features:")
    print(feature_importance.head(10))

# Vizualizarea rezultatelor
plt.figure(figsize=(14, 8))

# Plot pentru MAE
plt.subplot(2, 2, 1)
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('Mean Absolute Error (MAE) - mai mic = mai bun')
plt.xticks(rotation=45, ha='right')

# Plot pentru RMSE
plt.subplot(2, 2, 2)
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('Root Mean Squared Error (RMSE) - mai mic = mai bun')
plt.xticks(rotation=45, ha='right')

# Plot pentru R²
plt.subplot(2, 2, 3)
sns.barplot(x='Model', y='R²', data=results_df)
plt.title('R² Score - mai mare = mai bun')
plt.xticks(rotation=45, ha='right')

# Plot pentru timpul de antrenare
plt.subplot(2, 2, 4)
sns.barplot(x='Model', y='Training Time (s)', data=results_df)
plt.title('Timpul de antrenare (secunde)')
plt.xticks(rotation=45, ha='right')

plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()

# Comparație între valorile reale și predicții pentru modelul de bază și cel mai bun model
plt.figure(figsize=(12, 6))

# Random Forest (baseline)
rf_baseline = results[0]['Model Object']
rf_predictions = rf_baseline.predict(X_test)

# Cel mai bun model
best_predictions = best_model.predict(X_test)

# Plot pentru comparația dintre predicții
plt.scatter(y_test, rf_predictions, alpha=0.5, label='Random Forest (baseline)')
plt.scatter(y_test, best_predictions, alpha=0.5, label=f'Best Model: {best_model_name}')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Preț real')
plt.ylabel('Preț prezis')
plt.title('Comparație între prețurile reale și cele prezise')
plt.legend()
plt.grid(True)
plt.savefig('predictions_comparison.png')
plt.show()

# Calculăm îmbunătățirea în procente față de baseline
improvement = (results[0]['MAE'] - results[best_model_idx]['MAE']) / results[0]['MAE'] * 100
print(f"\nÎmbunătățire față de baseline: {improvement:.2f}%")

# Exemplu de utilizare a modelului
print("\nExemplu de utilizare a modelului:")
print("Introduceți datele unei mașini:")
sample_car = X_test.iloc[0].values.reshape(1, -1)
predicted_price = best_model.predict(sample_car)[0]
actual_price = y_test.iloc[0]
print(f"Preț prezis: {predicted_price:.2f}")
print(f"Preț real: {actual_price:.2f}")
print(f"Diferență: {abs(predicted_price - actual_price):.2f}")