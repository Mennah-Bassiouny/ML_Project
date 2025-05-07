
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load gene expression and response data
X = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_expression_gene_level.csv", index_col=0).T
y = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_response_metadata.csv", index_col=0)["response_label"].astype(int).reindex(X.index)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Logistic Regression with SMOTE
logreg = LogisticRegression(max_iter=1000, solver='liblinear')
logreg.fit(X_train_res, y_train_res)
y_pred_logreg = logreg.predict(X_test)
y_prob_logreg = logreg.predict_proba(X_test)[:, 1]
report_logreg = classification_report(y_test, y_pred_logreg, output_dict=True)
auc_logreg = roc_auc_score(y_test, y_prob_logreg)

# Save Logistic Regression report and predictions
pd.DataFrame(report_logreg).T.to_csv("logreg_smote_classification_report.csv")
logreg_preds = pd.DataFrame({
    "TrueLabel": y_test,
    "PredictedLabel": y_pred_logreg,
    "Probability": y_prob_logreg
}, index=y_test.index)
logreg_preds.to_csv("logreg_smote_predictions.csv")

# Random Forest with SMOTE
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train_res, y_train_res)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
report_rf = classification_report(y_test, y_pred_rf, output_dict=True)
auc_rf = roc_auc_score(y_test, y_prob_rf)

# Save Random Forest report and predictions
pd.DataFrame(report_rf).T.to_csv("rf_smote_classification_report.csv")
rf_preds = pd.DataFrame({
    "TrueLabel": y_test,
    "PredictedLabel": y_pred_rf,
    "Probability": y_prob_rf
}, index=y_test.index)
rf_preds.to_csv("rf_smote_predictions.csv")

# Plot ROC curves
fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_prob_logreg)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

plt.figure(figsize=(8, 6))
plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})', color='blue')
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})', color='red')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves with SMOTE")
plt.legend()
plt.tight_layout()
plt.savefig("smote_roc_curve.png")
