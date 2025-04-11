:construction:
# STILL WORKING

# Car Price Predictor

## Descriere

Car Price Predictor este o aplicație bazată pe Machine Learning care estimează prețul de vânzare al unei mașini second-hand în funcție de caracteristicile acesteia. Acest proiect demonstrează procesul complet de dezvoltare a unei soluții ML, de la prelucrarea datelor până la crearea unui model predictiv și expunerea acestuia printr-o interfață web.

## Obiectiv

Scopul principal al acestui proiect este de a crea un sistem care poate oferi o estimare de preț precisă pentru o mașină second-hand, ajutând atât vânzătorii cât și cumpărătorii în procesul de evaluare a vehiculelor.

## Dataset

Proiectul utilizează un dataset de pe Kaggle care conține informații despre mașini second-hand din Germania. Setul de date include caracteristici precum:
- Brand și model
- Anul înmatriculării
- Tipul de combustibil
- Kilometraj
- Putere motor
- Tipul cutiei de viteze
- Informații despre daune

## Componente principale

### 1. Analiza și preprocesarea datelor
- Curățarea datelor: eliminarea valorilor aberante și tratarea valorilor lipsă
- Analiza exploratorie: identificarea corelațiilor și a factorilor importanți
- Feature engineering: crearea de caracteristici noi relevante pentru predicție

### 2. Modelarea ML
- Experimentarea cu diferite algoritmi de ML: Random Forest, XGBoost, LightGBM
- Evaluarea și compararea performanței modelelor
- Optimizarea hiperparametrilor pentru modelul ales
- Analiza importanței caracteristicilor

### 3. Implementare API și Interfață
- Dezvoltarea unui API RESTful pentru predicții
- Crearea unei interfețe web intuitive 
- Funcționalități pentru încărcarea detaliilor mașinii
- Vizualizare rezultat cu detalii explicative

## Tehnologii utilizate

- **Limbaj**: Python
- **Biblioteci ML**: Scikit-learn, XGBoost, LightGBM
- **Prelucrare date**: Pandas, NumPy
- **Vizualizare**: Matplotlib, Seaborn
- **Backend**: Flask
- **Frontend**: HTML, CSS, JavaScript


Modelul final obține un Mean Absolute Error (MAE) de aproximativ [X] euro pe setul de test, ceea ce înseamnă că în medie, predicțiile se abat cu [X] euro de la prețul real. Analiza importanței caracteristicilor arată că factorii cei mai importanți în determinarea prețului sunt:

1. Anul fabricației
2. Kilometrajul
3. Marca și modelul
4. Puterea motorului

## Provocări întâmpinate

- Tratarea corectă a valorilor aberante
- Evitarea data leakage în procesul de feature engineering
- Optimizarea performanței modelelor ML
- Crearea unei interfețe intuitive pentru utilizatori

## Îmbunătățiri viitoare

- Integrare cu API-uri pentru detectarea automată a modelului din imagini
- Extinderea setului de date cu mașini din România
- Implementarea unui model de detectare a anomaliilor pentru identificarea ofertelor suspecte
- Adăugarea de prognoza pentru deprecierea viitoare a mașinii

