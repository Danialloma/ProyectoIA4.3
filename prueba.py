import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV # GridSearchCV se mantiene por si se quiere usar más adelante
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# --- Parte 1: Carga y Preparación de Datos ---

path = r"C:\Users\eduar\Documents\GitHub\Hola\Amazon_Unlocked_Mobile.csv"
df1 = pd.read_csv(path)
df1 = df1.dropna(subset=["Reviews"]) # Eliminar filas con Reviews nulas

Rating = df1["Rating"]
Reviews = df1["Reviews"]

def positivo_negativo(rating):
    return [1 if r >= 4 else 0 for r in rating]
df1["Nuevo Rating"] = positivo_negativo(Rating)

def reviewN(reviews):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews)
    numeric_values = tfidf_matrix.mean(axis=1).getA1()
    return numeric_values
df1["Review Numerico"] = reviewN(Reviews)

print("DataFrame con características preparadas:")
print(df1.head())
print("\n" + "="*80 + "\n")


# --- Parte 2: Preparación para los Modelos de Clasificación ---
X = df1[['Price', 'Review Numerico', 'Review Votes']]
y = df1['Nuevo Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("Dimensiones de los conjuntos de datos:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
print("\n" + "="*80 + "\n")


# --- Parte 3: Decision Tree Classifier ---
print("--- Decision Tree Classifier ---")
dt_classifier = DecisionTreeClassifier(max_depth=12, min_samples_leaf=10, random_state=42)
dt_classifier.fit(X_train, y_train)

# Predicciones
y_pred_dt_train = dt_classifier.predict(X_train)
y_pred_dt_test = dt_classifier.predict(X_test)

# Evaluación
print("Resultados de Decision Tree en Entrenamiento:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_dt_train):.4f}")
print("Classification Report:")
print(classification_report(y_train, y_pred_dt_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_dt_train))

print("\nResultados de Decision Tree en Prueba:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_dt_test):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_dt_test))
print("\n" + "="*80 + "\n")


# --- Parte 4: Random Forest Classifier (CON LIMITANTES) ---
print("--- Random Forest Classifier ---")
rf_classifier = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                       class_weight=None, random_state=42) # Ajustado n_estimators y max_depth, añadido class_weight
rf_classifier.fit(X_train, y_train)

# Predicciones
y_pred_rf_train = rf_classifier.predict(X_train)
y_pred_rf_test = rf_classifier.predict(X_test)

# Evaluación
print("Resultados de Random Forest en Entrenamiento:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_rf_train):.4f}")
print("Classification Report:")
print(classification_report(y_train, y_pred_rf_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_rf_train))

print("\nResultados de Random Forest en Prueba:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf_test):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_test))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_rf_test))
print("\n" + "="*80 + "\n")


# --- Parte 5: Comparación Final ---
print("--- Comparación de Modelos Optimizados ---")
print(f"Decision Tree Test Accuracy: {accuracy_score(y_test, y_pred_dt_test):.4f}")
print(f"Random Forest Test Accuracy: {accuracy_score(y_test, y_pred_rf_test):.4f}")

# GridSearchCV para Decision Tree:
#param_grid_dt = {
#    'max_depth': [5, 7, 10, 12],
#    'min_samples_split': [10, 20, 50],
#    'min_samples_leaf': [5, 10, 20]
#}
#grid_search_dt_cls = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid_dt, cv=5, n_jobs=-1)
#grid_search_dt_cls.fit(X_train, y_train)
#print("\nMejores parámetros para Decision Tree (GridSearch Fino):", grid_search_dt_cls.best_params_)
#print("Mejor Accuracy (Train CV) para Decision Tree:", grid_search_dt_cls.best_score_)

# GridSearchCV para Random Forest:
#param_grid_rf = {
#    'n_estimators': [100, 200, 300],
#    'max_depth': [7, 10, 12, 15],
#    'min_samples_leaf': [5, 10, 20],
#    'class_weight': [None, 'balanced']
#}
#grid_search_rf_cls = GridSearchCV(RandomForestClassifier(random_state=42), param_grid_rf, cv=5, n_jobs=-1)
#grid_search_rf_cls.fit(X_train, y_train)
#print("\nMejores parámetros para Random Forest (GridSearch Fino):", grid_search_rf_cls.best_params_)
#print("Mejor Accuracy (Train CV) para Random Forest:", grid_search_rf_cls.best_score_)