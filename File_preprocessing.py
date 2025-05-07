# Re-import libraries and re-run full script after code execution environment was reset
import pandas as pd
import gzip

# === Step 1: Load expression data ===
expression_file = "C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/Dataset/GSE25066_series_matrix.txt.gz"
expression_data = pd.read_csv(expression_file, sep="\t", comment="!", index_col=0)

# === Step 2: Load and map annotation ===
annotation_file = "C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/Dataset/GPL96.annot.gz"
annotation_data_raw = pd.read_csv(annotation_file, sep="\t", skiprows=27, dtype=str)
annotation_clean = annotation_data_raw[["ID", "Gene symbol"]].dropna()
annotation_clean = annotation_clean[annotation_clean["Gene symbol"] != ""]
annotation_clean["Gene symbol"] = annotation_clean["Gene symbol"].apply(lambda x: x.split("///")[0].strip())
annotation_clean = annotation_clean.drop_duplicates()
annotation_clean.set_index("ID", inplace=True)

# === Step 3: Map probes to genes and aggregate ===
expression_data.index.name = "ID"
expression_annotated = expression_data.merge(annotation_clean, left_index=True, right_index=True)
expression_annotated = expression_annotated.reset_index().set_index("Gene symbol")
expression_annotated = expression_annotated.drop(columns=["ID"])
expression_gene_level = expression_annotated.groupby(expression_annotated.index).mean()

# === Step 4: Extract response labels from GSE25066 metadata ===
with gzip.open(expression_file, "rt", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

response_line = [line for line in lines if line.startswith("!Sample_characteristics_ch1") and "response" in line.lower()][0]
response_values = [
    entry.split(":")[-1].replace('"', '').strip()
    for entry in response_line.strip().split("\t")[1:]
]

metadata = pd.DataFrame({
    "sample_id": expression_gene_level.columns,
    "response_raw": response_values
})
metadata["response_label"] = metadata["response_raw"].apply(
    lambda x: 1 if "pcr" in x.lower() else (0 if "rd" in x.lower() else None)
)
metadata = metadata.dropna()
metadata.set_index("sample_id", inplace=True)

# === Step 5: Match expression and metadata ===
matching_samples = expression_gene_level.columns.intersection(metadata.index)
expression_final = expression_gene_level[matching_samples]
metadata_final = metadata.loc[matching_samples]

# === Step 6: Save outputs ===
expression_out_path = "C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_expression_gene_level.csv"
metadata_out_path = "C:/Users/menna/OneDrive/Documents/IU LUDDY BIOINFORMATICS/Machine learning/Project/output/GSE25066_response_metadata.csv"

expression_final.to_csv(expression_out_path)
metadata_final.to_csv(metadata_out_path)

(expression_out_path, metadata_out_path)
#########################################################


