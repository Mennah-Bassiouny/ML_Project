
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
