
# 🧬 Exploratory Data Analysis for GSE25066 Paclitaxel Response Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
expr = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_expression_gene_level.csv", index_col=0).T
meta = pd.read_csv("C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_response_metadata.csv", index_col=0)
y = meta.loc[expr.index, 'response_label']

# Summary statistics
summary = expr.describe()
summary.to_csv("EDA_expression_summary.csv")

# Class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=y.map({1: "pCR", 0: "RD"}))
plt.title("Class Distribution (Response Labels)")
plt.xlabel("Response")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("EDA_class_distribution.png")
plt.close()

# PCA Analysis (Top 2 Components)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(expr)
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_scaled)
pca_df = pd.DataFrame(pca_result, columns=["PC1", "PC2"], index=expr.index)
pca_df["Response"] = y.map({1: "pCR", 0: "RD"})

plt.figure(figsize=(8, 6))
sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="Response", palette="Set1", s=60, alpha=0.8)
plt.title("PCA of Gene Expression (Top 2 PCs)")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("EDA_pca_plot.png")
plt.close()

# Heatmap of Top 50 Most Variable Genes
top_var_genes = expr.var().sort_values(ascending=False).head(50).index
top_expr = expr[top_var_genes]

plt.figure(figsize=(14, 10))
sns.heatmap(top_expr.T, cmap="vlag", xticklabels=False, yticklabels=True)
plt.title("Heatmap of Top 50 Most Variable Genes")
plt.tight_layout()
plt.savefig("EDA_top_variable_genes_heatmap.png")
plt.close()
