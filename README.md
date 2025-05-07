# 🧬 Predicting Taxane Response in Breast Cancer Using Machine Learning

This project uses gene expression data from breast cancer patients (GSE25066) to predict response to taxane-based chemotherapy (Paclitaxel and Docetaxel) using machine learning. The aim is to identify predictive biomarkers and build accurate models for precision oncology.

## 📁 Project Structure
├── File_preprocessing.py # Load & preprocess microarray data
├── EDA_Script.py # Exploratory data analysis: class balance, PCA, heatmap
├── ML_class_weight_balanced.py # Logistic regression & random forest with class weights
├── ML_with_SMOTE.py # ML with SMOTE oversampling for class imbalance
├── Paclitaxel_Model_Training_and_RF_feature_selection.py # Full pipeline with feature ranking
├── RandomForest_Top_500_Features.py # Feature importance from Random Forest
├── ML_Top100_Feature_Selection.py # Retrain models on top 100 features
├── Transpose_series_matrix.txt # Transpose raw expression data (if needed)
├── output/ # Folder containing saved reports, predictions, plots

## 🧪 Data Source

- **Dataset**: [GSE25066 - GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE25066)  
- **Samples**: ~500 breast cancer patients  
- **Platform**: Affymetrix Human Genome U133A Array (GPL96)

## 🛠 Methods

- **Preprocessing**: Probe-to-gene mapping, metadata extraction, filtering
- **EDA**: Class imbalance visualization, PCA, top variable genes heatmap
- **Modeling**: 
  - Logistic Regression & Random Forest
  - Class-weight balancing & SMOTE
  - Top 100 variable gene selection for performance boost
- **Evaluation**: ROC-AUC, recall, F1-score, classification report

## 🔬 Key Results

- Logistic regression with SMOTE reached **AUC = 0.86**
- Identified top gene predictors: **CDCA8**, **MCM6**, **DGKI**
- Random forest showed moderate performance boost with top features

## 🧰 Requirements

- Python 3.8+
- `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `imblearn`

Install required packages:
```bash
pip install -r requirements.txt

👩‍💻 Authors
Menatalla Bassiouny – menbass@iu.edu

Raghad Malkawi – ragmalka@iu.edu

