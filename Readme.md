# 🧬 GenomBridge AI

GenomBridge AI is an end-to-end bioinformatics and machine learning system that analyzes Genome-Wide Association Studies (GWAS) data to estimate polygenic disease risk. The project processes large-scale genetic association datasets, engineers disease-risk features, and predicts relative genetic susceptibility based on user-provided gene inputs.

⚠️ Disclaimer: This project is intended for research and educational purposes only. It does not provide medical advice, diagnosis, or treatment recommendations.

---

## 🚀 Features

- GWAS data preprocessing and cleaning
- Polygenic risk feature engineering
- Disease-level aggregation of genetic associations
- Machine learning-based disease risk prediction
- User gene input support
- Personalized disease risk ranking
- Cross-validation for model evaluation
- Bioinformatics + AI integration

---

## 🏗️ Project Architecture

```text
User Gene Input
       │
       ▼
Gene Matching
       │
       ▼
GWAS Dataset
       │
       ▼
Data Preprocessing
       │
       ▼
Feature Engineering
       │
       ▼
Disease Aggregation
       │
       ▼
Machine Learning Model
       │
       ▼
Disease Risk Prediction
```

---

## 📂 Project Structure

```text
GenomBridge-AI/
│
├── data/
│   └── gwas_associations.tsv
│
├── main.py
├── requirements.txt
└── README.md
```

---

## 🧬 Dataset

This project uses the GWAS Catalog "All Associations" dataset containing:

- Disease/Trait associations
- SNP information
- Gene mappings
- Risk allele frequencies
- P-values
- Odds Ratios (OR/BETA)
- Population study metadata

Source:
https://www.ebi.ac.uk/gwas/

---

## ⚙️ Technologies Used

### Programming Language
- Python

### Libraries
- Pandas
- NumPy
- Scikit-Learn

### Machine Learning
- HistGradientBoostingRegressor
- Cross Validation

### Domain
- Bioinformatics
- Genomics
- Genetic Risk Analysis

---

## 🔄 Workflow

### 1. Data Loading
The GWAS dataset is loaded and validated.

### 2. Data Preprocessing
- Remove missing values
- Handle numeric conversion
- Normalize and clean dataset
- Extract relevant genetic information

### 3. Feature Engineering
Generate disease-risk features such as:

- PRS Sum
- PRS Mean
- PRS Maximum
- Odds Ratio Mean
- Odds Ratio Maximum
- Risk Allele Frequency Statistics
- P-value Statistics
- Variant Type Counts

### 4. Disease Aggregation
Aggregate SNP-level data into disease-level representations.

### 5. Model Training
Train a machine learning model on aggregated disease features.

### 6. User Gene Analysis
Users can input a list of genes such as:

```text
APOE, LDLR, BRCA1
```

The system identifies related genetic associations and predicts disease risk rankings.

---

## 📊 Example Input

```text
APOE, LDLR, LPA, PCSK9, BRCA1, BRCA2
```

---

## 📈 Example Output

```text
USER DISEASE RISK REPORT

Alzheimer disease              7.42
Coronary artery disease        5.22
Type 2 diabetes mellitus       3.25
Parkinson disease              2.98
```

---

## 🧠 Machine Learning Pipeline

Features Used:

```python
FEATURE_COLS = [
    "prs_sum",
    "prs_mean",
    "prs_max",
    "or_beta_mean",
    "or_beta_max",
    "raf_mean",
    "raf_max",
    "p_log_mean",
    "p_log_max",
    "sample_size_mean",
    "variant_type_count"
]
```

Model:

```python
HistGradientBoostingRegressor
```

Evaluation Metrics:

- Mean Squared Error (MSE)
- R² Score
- Cross-Validation R²

---

## 🎯 Objectives

- Explore the application of AI in genomics
- Analyze genetic disease associations
- Generate personalized disease risk rankings
- Demonstrate bioinformatics and machine learning integration

---

## 📌 Future Improvements

- FastAPI backend integration
- React frontend dashboard
- Interactive risk visualizations
- PDF report generation
- User authentication
- Real-time genomic analysis
- Polygenic Risk Score (PRS) enhancement

---

## 💻 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/GenomBridge-AI.git
```

Navigate to project directory:

```bash
cd GenomBridge-AI
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the project:

```bash
python main.py
```

---

## 🎓 Educational Value

This project demonstrates:

- Data preprocessing
- Feature engineering
- Machine learning workflows
- Genomic data analysis
- Risk prediction systems
- End-to-end AI project development

---

## 📜 License

This project is intended for educational and research purposes.

---

## 👨‍💻 Author

Shivam Paul

GenomBridge AI – Genetic Disease Risk Prediction System