
# Predicting Paclitaxel Response in Breast Cancer
# This script loads preprocessed gene expression data (GSE25066),
# applies standardization for Logistic Regression, trains two models,
# evaluates performance, and saves results.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import matplotlib.pyplot as plt

# Load data
expr = pd.read_csv('C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_expression_gene_level.csv', index_col=0)
meta = pd.read_csv('C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_response_metadata.csv', index_col=0)
X = expr.T
y = meta.loc[X.index, 'response_label']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize for Logistic Regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]

# Evaluation Function
def evaluate_and_save(y_test, y_pred, y_prob, model_name):
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{model_name}_classification_report.csv")

    results_df = pd.DataFrame({
        'True_Label': y_test,
        'Predicted_Label': y_pred,
        'Probability': y_prob
    }, index=y_test.index)
    results_df.to_csv(f"{model_name}_predictions.csv")

    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

# Plot ROC curves and save
plt.figure(figsize=(8, 6))
evaluate_and_save(y_test, y_pred_lr, y_prob_lr, "LogisticRegression")
evaluate_and_save(y_test, y_pred_rf, y_prob_rf, "RandomForest")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png")
plt.show()
###########################################################
# 🔍 Feature Importance from Random Forest
importances = rf.feature_importances_
feature_names = X.columns
importance_df = pd.DataFrame({
    'Gene': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

# Save top 500 features
top_genes = importance_df.head(500)
top_genes.to_csv("RandomForest_Top_500_Features.csv", index=False)

# Plot top 30 out of 500 for visualization purposes
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 10))
sns.barplot(data=top_genes.head(30), x='Importance', y='Gene', palette='crest')
plt.title("Top 30 Most Important Genes (Random Forest)")
plt.xlabel("Feature Importance")
plt.ylabel("Gene")
plt.tight_layout()
plt.savefig("RandomForest_Top_30_Features.png")
plt.show()
#########################################################

