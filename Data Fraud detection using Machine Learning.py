# Data Fraud detection using Machine Learning
# Models compared: KNN, Decision tree, ANN and SVM 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier    
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
)

RANDOM_STATE = 42
sns.set(style='whitegrid')

#---------------------------------------------------------------------------------
#  STEP 1: Load the dataset

df = pd.read_csv('mpesa_synthetic.csv')
print(df.head())
print(df.info())
print("Initial dataset shape:", df.shape)

#---------------------------------------------------------------------------------
#  STEP 2: Data Preprocessing

df.dropna(inplace=True)  # Drop rows with missing values
if 'is_fraud' in df.columns:
    df = df[df['is_fraud'] != 'Unknown']  # Remove rows with 'Unknown' in the target variable
print("Dataset shape after preprocessing:", df.shape)


# Simple yes/no encoding for categorical variables
#df['is_fraud'] = df['is_fraud'].map({'Yes': 1, 'No': 0})
# Encode categorical features using one-hot encoding
categorical_features = ['transaction_id', 'transaction_type', 'device_type']
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)  
df = df.select_dtypes(include=[np.number])  # Keep only numeric columns

# Step 3: Train/Test Split + Scaling
X = df.drop('is_fraud', axis=1)
Y = df['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=RANDOM_STATE)
print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)
print("Training labels shape:", y_train.shape)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#---------------------------------------------------------------------------------
# Step 4a: Train and evaluate KNN model

knn_param_grid = {'n_neighbors': [3, 5, 7, 9], 'weights': ['uniform', 'distance']}
knn_grid = GridSearchCV(KNeighborsClassifier(), knn_param_grid, cv=2, scoring='f1', n_jobs=-1
)
knn_grid.fit(X_train, y_train)

best_knn_model = knn_grid.best_estimator_
y_pred_knn = best_knn_model.predict(X_test)
print("\nbest KNN model parameters:", knn_grid.best_params_["n_neighbors"])
print("KNN Model Accuracy:", accuracy_score(y_test, y_pred_knn))
print("KNN Model Precision:", precision_score(y_test, y_pred_knn))
print("KNN Model Recall:", recall_score(y_test, y_pred_knn))
print("KNN Model F1-Score:", f1_score(y_test, y_pred_knn))

#---------------------------------------------------------------------------------
# Step 4b: Train and evaluate Decision Tree model

dt_param_grid = {
    'max_depth': [None, 5, 10, 15], 'min_samples_split': [2, 5, 10]
    }
dt_grid = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_STATE), dt_param_grid, cv=2, scoring='f1', n_jobs=-1
)
dt_grid.fit(X_train, y_train)

best_dt_model = dt_grid.best_estimator_
y_pred_dt = best_dt_model.predict(X_test)
print("\nbest Decision Tree model parameters:", dt_grid.best_params_["max_depth"])
print("Decision Tree Model Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Decision Tree Model Precision:", precision_score(y_test, y_pred_dt))
print("Decision Tree Model Recall:", recall_score(y_test, y_pred_dt))
print("Decision Tree Model F1-Score:", f1_score(y_test, y_pred_dt))

#---------------------------------------------------------------------------------
# Step 4c: Train and evaluate ANN model
mlp_param_grid = {
    'hidden_layer_sizes': [(50,)], 'activation': ['relu', 'tanh'], 'solver': ['adam', 'sgd'], 'max_iter': [200, 300]
}

mlp_grid = GridSearchCV(
    MLPClassifier(random_state=RANDOM_STATE), mlp_param_grid, cv=2, scoring='f1', n_jobs=-1
)
mlp_grid.fit(X_train, y_train)

best_mlp_model = mlp_grid.best_estimator_
y_pred_mlp = best_mlp_model.predict(X_test)
print("\nbest ANN model parameters:", mlp_grid.best_params_)
print("ANN Model Accuracy:", accuracy_score(y_test, y_pred_mlp))
print("ANN Model Precision:", precision_score(y_test, y_pred_mlp))
print("ANN Model Recall:", recall_score(y_test, y_pred_mlp))
print("ANN Model F1-Score:", f1_score(y_test, y_pred_mlp))

#---------------------------------------------------------------------------------
# Step 4d: Train and evaluate SVM model 
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
svm_param_grid = {
    'estimator__C': [0.1, 1, 10],
}

base_svm_model = LinearSVC(random_state=RANDOM_STATE, dual=False, max_iter=2000)
calibrated_svm_model = CalibratedClassifierCV(estimator=base_svm_model, cv=2)

svm_grid = GridSearchCV(calibrated_svm_model, svm_param_grid, cv=2, scoring='f1', n_jobs=-1)
svm_grid.fit(X_train, y_train)

best_svm_model = svm_grid.best_estimator_
y_pred_svm = best_svm_model.predict(X_test)
print("\nbest SVM model parameters:", svm_grid.best_params_)
print("SVM Model Accuracy:", accuracy_score(y_test, y_pred_svm))
print("SVM Model Precision:", precision_score(y_test, y_pred_svm))
print("SVM Model Recall:", recall_score(y_test, y_pred_svm))
print("SVM Model F1-Score:", f1_score(y_test, y_pred_svm))

#---------------------------------------------------------------------------------
# step 5: evaluate models using ROC-AUC

y_pred_proba_knn = best_knn_model.predict_proba(X_test)[:, 1]
y_pred_proba_dt = best_dt_model.predict_proba(X_test)[:, 1]
y_pred_proba_mlp = best_mlp_model.predict_proba(X_test)[:, 1]
y_pred_proba_svm = best_svm_model.predict_proba(X_test)[:, 1]

print("\nROC-AUC Scores:")
print("KNN Model ROC-AUC:", roc_auc_score(y_test, y_pred_proba_knn))
print("Decision Tree Model ROC-AUC:", roc_auc_score(y_test, y_pred_proba_dt))
print("ANN Model ROC-AUC:", roc_auc_score(y_test, y_pred_proba_mlp))
print("SVM Model ROC-AUC:", roc_auc_score(y_test, y_pred_proba_svm))


# ---------------------------------------------------------------------------------
# Step 6: Visualization
# ROC curves for all models
model = {
    'KNN': y_pred_proba_knn,
    'Decision Tree': y_pred_proba_dt,
    'ANN': y_pred_proba_mlp,
    'SVM': y_pred_proba_svm
}
plt.figure(figsize=(10, 8))
for model_name, y_pred_proba in model.items():
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier (AUC = 0.50)')
plt.title('ROC Curves for Fraud Detection Models')    
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.savefig('roc_curves.png')
plt.close()

# confusion matrix for all models
models = {
    'KNN': y_pred_proba_knn,
    'Decision Tree': y_pred_proba_dt,
    'ANN': y_pred_proba_mlp,
    'SVM': y_pred_proba_svm
}
plt.figure(figsize=(12, 10))
for i, (model_name, y_pred_proba) in enumerate(models.items(), 1):
    plt.subplot(2, 2, i)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.title(f'{model_name} Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
plt.tight_layout()
plt.savefig('confusion_matrices.png')
plt.close()

# model comparison table
comparison_df = pd.DataFrame({
    'Model': ['KNN', 'Decision Tree', 'ANN', 'SVM'],
    'Precision': [precision_score(y_test, (y_pred_proba >= 0.5).astype(int)) for y_pred_proba in models.values()],
    'Recall': [recall_score(y_test, (y_pred_proba >= 0.5).astype(int)) for y_pred_proba in models.values()],
    'F1-Score': [f1_score(y_test, (y_pred_proba >= 0.5).astype(int)) for y_pred_proba in models.values()],
    'ROC-AUC': [roc_auc_score(y_test, y_pred_proba) for y_pred_proba in models.values()]
})
print("\nModel Comparison Table:")
print(comparison_df.to_string(index=False))
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='F1-Score', data=comparison_df)
plt.title('Model Comparison based on F1-Score')
plt.savefig('model_comparison_f1_score.png')
plt.close()