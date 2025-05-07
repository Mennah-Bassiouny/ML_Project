
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Load data
X = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_expression_gene_level.csv", index_col=0).T
y = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_response_metadata.csv", index_col=0)["response_label"].astype(int).reindex(X.index)

# Load top genes from Random Forest feature importance
top_genes_df = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/ML Model Training/RandomForest_Top_500_Features.csv")
top_genes = top_genes_df["Gene"].head(100).tolist()  # select top 100 features

# Subset expression data
X_selected = X[top_genes]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25, stratify=y, random_state=42)

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, class_weight='balanced', solver='liblinear')
logreg.fit(X_train, y_train)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
report_logreg = classification_report(y_test, y_pred_logreg, output_dict=True)
auc_logreg = roc_auc_score(y_test, y_prob_logreg)
pd.DataFrame(report_logreg).T.to_csv("logreg_top100_classification_report.csv")
pd.DataFrame({
    "TrueLabel": y_test,
    "PredictedLabel": y_pred_logreg,
    "Probability": y_prob_logreg
}, index=y_test.index).to_csv("logreg_top100_predictions.csv")

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
auc_rf = roc_auc_score(y_test, y_prob_rf)
pd.DataFrame(report_rf).T.to_csv("rf_top100_classification_report.csv")
pd.DataFrame({
    "TrueLabel": y_test,
    "PredictedLabel": y_pred_rf,
    "Probability": y_prob_rf
}, index=y_test.index).to_csv("rf_top100_predictions.csv")

# Plot ROC
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='green')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Top 100 Features")
plt.legend()
plt.tight_layout()
plt.savefig("top100_features_roc_curve.png")
