import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Parte 1: Carga y Preparación de Datos ---
path = r"C:\Users\eduar\Documents\GitHub\ProyectoIA4.3\Amazon_Unlocked_Mobile.csv"
df1 = pd.read_csv(path) # Vuelve a cargar tu DataFrame real

# Primero, manejar NaNs en la columna 'Reviews' si aún no se ha hecho
df1 = df1.dropna(subset=["Reviews"]).copy() # .copy() para evitar SettingWithCopyWarning

def positivo_negativo(rating):
    return [1 if r >= 4 else 0 for r in rating]
df1["Nuevo Rating"] = positivo_negativo(df1["Rating"])

def reviewN(reviews):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews)
    numeric_values = tfidf_matrix.mean(axis=1).getA1()
    return numeric_values
df1["Review Numerico"] = reviewN(df1["Reviews"])

# --- ¡NUEVA LÍNEA CLAVE PARA MANEJAR NANs EN LAS CARACTERÍSTICAS NUMÉRICAS! ---
# Identifica las columnas que usarás como características
features_to_use = ['Price', 'Review Numerico', 'Review Votes']

# Elimina cualquier fila que contenga un NaN en CUALQUIERA de estas columnas de características
# Esto es crucial para SVC y la mayoría de los estimadores de scikit-learn
df1.dropna(subset=features_to_use, inplace=True)


print("DataFrame con características preparadas (después de manejar NaNs):")
print(df1.head())
print(f"Número de filas después de eliminar NaNs: {len(df1)}") # Para ver cuántas filas se eliminaron
print("\n" + "="*80 + "\n")

# --- Parte 2: Preparación para los Modelos de Clasificación ---
X = df1[features_to_use] # Usar la lista de características definida
y = df1['Nuevo Rating']

# ... el resto de tu código para train_test_split, StandardScaler, y los modelos ...

# Si sigues la recomendación de probar con una muestra más pequeña para SVM:
# X_sample, _, y_sample, _ = train_test_split(X, y, test_size=0.9, random_state=42, stratify=y)
# X_train, X_test, y_train, y_test = train_test_split(X_sample, y_sample, test_size=0.25, random_state=42, stratify=y_sample)

# Si vas a usar el dataset completo (con la advertencia de tiempo para SVC):
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- ESCALADO DE CARACTERÍSTICAS PARA SVM ---
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)



print("Dimensiones de los conjuntos de datos:")
print(f"X_train: {X_train.shape}")
print(f"X_test: {X_test.shape}")
print(f"y_train: {y_train.shape}")
print(f"y_test: {y_test.shape}")
print("\n" + "="*80 + "\n")


# --- Parte 3: Decision Tree Classifier (Usando Mejores Parámetros de GridSearchCV) ---
print("--- Decision Tree Classifier ---")
dt_classifier_optimized = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, min_samples_split=10, random_state=42)
dt_classifier_optimized.fit(X_train_scaled, y_train) # Usar datos escalados para DT también es buena práctica, aunque no tan crítico como para SVM

y_pred_dt_opt_train = dt_classifier_optimized.predict(X_train_scaled)
y_pred_dt_opt_test = dt_classifier_optimized.predict(X_test_scaled)

print("Resultados de Decision Tree en Entrenamiento:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_dt_opt_train):.4f}")
print("Classification Report:")
print(classification_report(y_train, y_pred_dt_opt_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_dt_opt_train))

print("\nResultados de Decision Tree en Prueba:")
dt_test_accuracy = accuracy_score(y_test, y_pred_dt_opt_test)
dt_test_report = classification_report(y_test, y_pred_dt_opt_test, output_dict=True)
dt_test_cm = confusion_matrix(y_test, y_pred_dt_opt_test)

print(f"Accuracy: {dt_test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_dt_opt_test))
print("Confusion Matrix:")
print(dt_test_cm)
print("\n" + "="*80 + "\n")


# --- Parte 4: Random Forest Classifier (Usando Mejores Parámetros de GridSearchCV) ---
print("--- Random Forest Classifier ---")
rf_classifier_optimized = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                                 class_weight=None, random_state=42)
rf_classifier_optimized.fit(X_train_scaled, y_train) # Usar datos escalados para RF también

y_pred_rf_opt_train = rf_classifier_optimized.predict(X_train_scaled)
y_pred_rf_opt_test = rf_classifier_optimized.predict(X_test_scaled)

print("Resultados de Random Forest en Entrenamiento:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_rf_opt_train):.4f}")
print("Classification Report:")
print(classification_report(y_train, y_pred_rf_opt_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_rf_opt_train))

print("\nResultados de Random Forest en Prueba:")
rf_test_accuracy = accuracy_score(y_test, y_pred_rf_opt_test)
rf_test_report = classification_report(y_test, y_pred_rf_opt_test, output_dict=True)
rf_test_cm = confusion_matrix(y_test, y_pred_rf_opt_test)

print(f"Accuracy: {rf_test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf_opt_test))
print("Confusion Matrix:")
print(rf_test_cm)
print("\n" + "="*80 + "\n")


# --- Parte 5: Support Vector Machine (SVC) ---
print("--- Support Vector Machine (SVC) ---")

# Parámetros iniciales para SVC.
# IMPORTANTE: C y gamma son cruciales y deben ajustarse con GridSearchCV
# Pero para una primera prueba, estos valores son un buen punto de partida.
# C (regularización): valores más pequeños = más regularización
# gamma (influencia de una sola muestra de entrenamiento): 'scale' es el valor por defecto
svc_classifier = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True) # probability=True para poder usar predict_proba si fuera necesario
svc_classifier.fit(X_train_scaled, y_train)

y_pred_svc_train = svc_classifier.predict(X_train_scaled)
y_pred_svc_test = svc_classifier.predict(X_test_scaled)

print("Resultados de SVC en Entrenamiento:")
print(f"Accuracy: {accuracy_score(y_train, y_pred_svc_train):.4f}")
print("Classification Report:")
print(classification_report(y_train, y_pred_svc_train))
print("Confusion Matrix:")
print(confusion_matrix(y_train, y_pred_svc_train))

print("\nResultados de SVC en Prueba:")
svc_test_accuracy = accuracy_score(y_test, y_pred_svc_test)
svc_test_report = classification_report(y_test, y_pred_svc_test, output_dict=True)
svc_test_cm = confusion_matrix(y_test, y_pred_svc_test)

print(f"Accuracy: {svc_test_accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred_svc_test))
print("Confusion Matrix:")
print(svc_test_cm)
print("\n" + "="*80 + "\n")


# --- Parte 6: Comparación de Modelos y Visualizaciones (ACTUALIZADO con SVC) ---
print("--- Comparación de Modelos Finales Optimizados ---")
print(f"Decision Tree Test Accuracy: {dt_test_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")
print(f"SVC Test Accuracy: {svc_test_accuracy:.4f}")

# 1. Gráfico de barras de Accuracy
models = ['Decision Tree', 'Random Forest', 'SVC']
accuracies = [dt_test_accuracy, rf_test_accuracy, svc_test_accuracy]

plt.figure(figsize=(10, 7))
sns.barplot(x=models, y=accuracies, palette='viridis')
plt.ylim(0.0, 1.0)
plt.title('Comparación de Accuracy en el Conjunto de Prueba')
plt.ylabel('Accuracy')
for index, value in enumerate(accuracies):
    plt.text(index, value + 0.02, f'{value:.4f}', ha='center')
plt.show()

# 2. Gráficos de Precision, Recall, F1-score por clase
metrics = ['precision', 'recall', 'f1-score']
class_labels = ['0', '1']

fig, axes = plt.subplots(1, 3, figsize=(20, 6), sharey=True) # 3 subgráficos ahora

# Decision Tree
dt_metrics_data = {metric: [dt_test_report['0'][metric], dt_test_report['1'][metric]] for metric in metrics}
df_dt_metrics = pd.DataFrame(dt_metrics_data, index=class_labels)
df_dt_metrics.plot(kind='bar', ax=axes[0], colormap='plasma', rot=0)
axes[0].set_title('Métricas del Decision Tree (Test)')
axes[0].set_ylabel('Score')
axes[0].legend(title='Métrica')
axes[0].set_ylim(0, 1.0)

# Random Forest
rf_metrics_data = {metric: [rf_test_report['0'][metric], rf_test_report['1'][metric]] for metric in metrics}
df_rf_metrics = pd.DataFrame(rf_metrics_data, index=class_labels)
df_rf_metrics.plot(kind='bar', ax=axes[1], colormap='plasma', rot=0)
axes[1].set_title('Métricas del Random Forest (Test)')
axes[1].legend(title='Métrica')
axes[1].set_ylim(0, 1.0)

# SVC
svc_metrics_data = {metric: [svc_test_report['0'][metric], svc_test_report['1'][metric]] for metric in metrics}
df_svc_metrics = pd.DataFrame(svc_metrics_data, index=class_labels)
df_svc_metrics.plot(kind='bar', ax=axes[2], colormap='plasma', rot=0)
axes[2].set_title('Métricas del SVC (Test)')
axes[2].legend(title='Métrica')
axes[2].set_ylim(0, 1.0)

plt.tight_layout()
plt.show()


# 3. Matrices de Confusión Visualizadas
fig, axes = plt.subplots(1, 3, figsize=(21, 6)) # 3 subgráficos ahora

sns.heatmap(dt_test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0], cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
axes[0].set_title('Matriz de Confusión - Decision Tree')
axes[0].set_xlabel('Predicción')
axes[0].set_ylabel('Valor Real')

sns.heatmap(rf_test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[1], cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
axes[1].set_title('Matriz de Confusión - Random Forest')
axes[1].set_xlabel('Predicción')
axes[1].set_ylabel('Valor Real')

sns.heatmap(svc_test_cm, annot=True, fmt='d', cmap='Blues', ax=axes[2], cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['Actual 0', 'Actual 1'])
axes[2].set_title('Matriz de Confusión - SVC')
axes[2].set_xlabel('Predicción')
axes[2].set_ylabel('Valor Real')

plt.tight_layout()
plt.show()