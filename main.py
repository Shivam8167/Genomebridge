import pandas as pd
import numpy as np

from dataclasses import dataclass
from enum import Enum

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score

FEATURE_COLS = [
    "prs_sum", "prs_mean", "prs_max",
    "or_beta_mean", "or_beta_max",
    "raf_mean", "raf_max",
    "p_log_mean", "p_log_max",
    "sample_size_mean",
    "variant_type_count"
]



class InheritanceType(Enum):
    AUTOSOMAL_RECESSIVE = "AR"
    AUTOSOMAL_DOMINANT = "AD"
    X_LINKED = "XL"
    POLYGENIC = "PG"

@dataclass
class Disease:
    disease_id: str
    inheritance: InheritanceType
    penetrance: float
    prevalence: float
    severity: float

def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep="\t", low_memory=False)
    print(f"Dataset loaded with {df.shape[0]} rows")
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    print("Preprocessing data...")
    # print("Columns in dataset:", df.columns.tolist())

    # Map dataset columns to clean names
    column_map = {
        "DISEASE/TRAIT": "disease",
        "MAPPED_TRAIT": "mapped_disease",
        "REPORTED GENE(S)": "gene",
        "OR or BETA": "or_beta",
        "P-VALUE": "p_value",
        "RISK ALLELE FREQUENCY": "raf",
        "CHR_ID": "chromosome",
        "CHR_POS": "position",
        "CONTEXT": "variant_context",
        "INITIAL SAMPLE SIZE": "sample_size"
    }

    df = df[df["MAPPED_TRAIT"].str.contains(
        "disease|cancer|diabetes|syndrome|asthma|arthritis",
        case=False, na=False
    )]

    # Keep only available columns
    existing_cols = [col for col in column_map if col in df.columns]
    # print("Using columns:", existing_cols)

    df = df[existing_cols].rename(columns=column_map)

    # 🔧 Clean numeric columns using regex
    def extract_number(series):
        return (
            series.astype(str)
            .str.replace(",", "", regex=False)
            .str.extract(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)")[0]
        )

    for col in ["or_beta", "p_value", "raf", "sample_size"]:
        if col in df.columns:
            df[col] = extract_number(df[col])

    # Convert to numeric
    numeric_cols = ["or_beta", "p_value", "raf", "sample_size"]
    numeric_cols = [c for c in numeric_cols if c in df.columns]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Drop only rows missing critical values
    df = df.dropna(subset=["mapped_disease", "or_beta", "p_value", "raf"])

    # Keep valid p-values
    df = df[df["p_value"] > 0]

    # Feature transform
    df["p_log"] = -np.log10(df["p_value"])

    print("Preprocessing complete")
    print("Rows after preprocessing:", df.shape)

    return df

def get_user_genes() -> list:
    print("\nEnter gene names separated by commas (example: APOE, LDLR, LPA):")
    genes = input(">> ")
    return [g.strip().upper() for g in genes.split(",")]


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    print("Engineering features...")

    encoder = LabelEncoder()
    df["variant_context_encoded"] = encoder.fit_transform(df["variant_context"])

    # Polygenic Risk Component
    df["prs_component"] = df["or_beta"] * df["raf"] * df["p_log"]

    print("Feature engineering done")
    return df

def filter_by_user_genes(df: pd.DataFrame, genes: list) -> pd.DataFrame:
    pattern = "|".join(genes)
    user_df = df[df["gene"].str.contains(pattern, case=False, na=False)]

    print(f"Matched {user_df.shape[0]} rows for user genes")
    return user_df

def aggregate_to_disease_level(df: pd.DataFrame) -> pd.DataFrame:
    print("Aggregating disease-level features...")

    disease_df = df.groupby("mapped_disease").agg({
        "prs_component": ["sum", "mean", "max"],
        "or_beta": ["mean", "max"],
        "raf": ["mean", "max"],
        "p_log": ["mean", "max"],
        "sample_size": "mean",
        "variant_context_encoded": "nunique"
    })

    disease_df.columns = [
        "prs_sum", "prs_mean", "prs_max",
        "or_beta_mean", "or_beta_max",
        "raf_mean", "raf_max",
        "p_log_mean", "p_log_max",
        "sample_size_mean",
        "variant_type_count"
    ]

    disease_df = disease_df.reset_index()

    # 🔥 FIX: Log-transform target
    disease_df["severity_score"] = np.log1p(disease_df["prs_sum"])

    # 🔧 Clip extreme outliers
    for col in [
        "prs_sum", "prs_mean", "prs_max",
        "or_beta_mean", "or_beta_max",
        "raf_mean", "raf_max",
        "p_log_mean", "p_log_max",
        "sample_size_mean"
    ]:
        if col in disease_df.columns:
            disease_df[col] = disease_df[col].clip(
                lower=disease_df[col].quantile(0.01),
                upper=disease_df[col].quantile(0.99)
            )

    print("Aggregation complete")
    print("Rows after aggregation:", disease_df.shape)

    return disease_df



def train_ml_model(df: pd.DataFrame):
    print("Training ML model...")

    if df.shape[0] == 0:
        raise ValueError("❌ No data available after preprocessing and aggregation.")

    # 🔹 Updated feature set (from your improved aggregation)
    X = df[FEATURE_COLS]

    y = df["severity_score"]

    # 🔹 Train / Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 🔥 Better model: Histogram Gradient Boosting
    model = HistGradientBoostingRegressor(
        max_iter=500,       # number of trees
        learning_rate=0.05,
        max_depth=8,
        min_samples_leaf=30,
        random_state=42
    )

    # 🔹 Train
    model.fit(X_train, y_train)

    # 🔹 Predict
    predictions = model.predict(X_test)

    # 🔹 Evaluate
    print("MSE:", mean_squared_error(y_test, predictions))
    print("R² Score:", r2_score(y_test, predictions))

    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print("Cross-validated R² scores:", scores)
    print("Mean CV R²:", scores.mean())

    return model



def create_disease_objects(df: pd.DataFrame):
    diseases = []

    for _, row in df.iterrows():
        disease = Disease(
            disease_id=row["mapped_disease"],
            inheritance=InheritanceType.POLYGENIC,

            # Use aggregated columns instead of old ones
            penetrance=min(1.0, row["or_beta_mean"] / 5),
            prevalence=min(1.0, row["raf_mean"]),
            severity=min(1.0, abs(row["severity_score"]) / 10)
        )
        diseases.append(disease)

    return diseases


def main():
    path = "D:\\Genomebridge\\Dataset\\gwas-catalog-download-associations-alt-full.tsv"

    # 1. Load & preprocess
    df = load_data(path)
    df = preprocess_data(df)
    df = feature_engineering(df)

    # 2. User input
    user_genes = get_user_genes()
    user_df = filter_by_user_genes(df, user_genes)

    if user_df.empty:
        print("No matching genes found in GWAS data")
        return

    # 3. Aggregate user-specific disease risks
    user_disease_df = aggregate_to_disease_level(user_df)

    # 4. Train model on full data (population knowledge)
    disease_df = aggregate_to_disease_level(df)
    model = train_ml_model(disease_df)

    # 5. Predict user risks
    feature_cols = [
    "prs_sum",
    "prs_mean",
    "or_beta_mean",
    "raf_mean",
    "p_log_mean",
    "sample_size_mean",
    "variant_type_count"
]


    print("User DF columns:", user_disease_df.columns)

    X_user = user_disease_df[FEATURE_COLS]
    user_predictions = model.predict(X_user)


    user_disease_df["predicted_risk"] = user_predictions

    print("\n🧬 USER DISEASE RISK REPORT")
    print(user_disease_df[["mapped_disease", "predicted_risk"]]
          .sort_values(by="predicted_risk", ascending=False)
          .head(10))
if __name__ == "__main__":
    main()

