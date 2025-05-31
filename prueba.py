import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

path = r"C:\Users\eduar\Documents\GitHub\ProyectoIA4.3\Amazon_Unlocked_Mobile.csv"
df1 = pd.read_csv(path)

df1 = df1.dropna(subset=["Reviews"])

def positivo_negativo(rating):
    return [1 if r >= 4 else 0 for r in rating]
df1["Nuevo Rating"] = positivo_negativo(df1["Rating"])

def reviewN(reviews):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(reviews)
    numeric_values = tfidf_matrix.mean(axis=1).getA1()
    return numeric_values
df1["Review Numerico"] = reviewN(df1["Reviews"]) # Usamos df1["Reviews"] directamente

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
dt_classifier_optimized = DecisionTreeClassifier(max_depth=12, min_samples_leaf=5, min_samples_split=10, random_state=42)
dt_classifier_optimized.fit(X_train, y_train)

y_pred_dt_opt_train = dt_classifier_optimized.predict(X_train)
y_pred_dt_opt_test = dt_classifier_optimized.predict(X_test)

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


# --- Parte 4: Random Forest Classifier ---
print("--- Random Forest Classifier ---")
rf_classifier_optimized = RandomForestClassifier(n_estimators=200, max_depth=15, min_samples_leaf=10,
                                                 class_weight=None, random_state=42)
rf_classifier_optimized.fit(X_train, y_train)

y_pred_rf_opt_train = rf_classifier_optimized.predict(X_train)
y_pred_rf_opt_test = rf_classifier_optimized.predict(X_test)

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


# --- Parte 5: Comparación de Modelos y Visualizaciones ---
print("--- Comparación de Modelos Finales ---")
print(f"Decision Tree Test Accuracy: {dt_test_accuracy:.4f}")
print(f"Random Forest Test Accuracy: {rf_test_accuracy:.4f}")

# Gráficos de Precision, Recall, F1-score por clase
metrics = ['precision', 'recall', 'f1-score']
class_labels = ['0', '1'] # Clases: 0 (Negativo), 1 (Positivo)

fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)

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

plt.tight_layout()
plt.show()


# Matrices de Confusión Visualizadas
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
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

plt.tight_layout()
plt.show()