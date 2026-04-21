"""
Car Price Prediction — ShadowFox Intermediate Task
====================================================
Predicts car selling prices based on various attributes using
a Random Forest Regressor with hyperparameter tuning.

Dataset: Car Dekho dataset
Features:
  Car_Name       - Name of the car
  Year           - Year of manufacture
  Selling_Price  - Price the car is being sold at (TARGET)
  Present_Price  - Current showroom price
  Kms_Driven     - Total kilometers driven
  Fuel_Type      - Petrol / Diesel / CNG
  Seller_Type    - Dealer / Individual
  Transmission   - Manual / Automatic
  Owner          - Number of previous owners

Requirements:
    pip install pandas numpy scikit-learn matplotlib seaborn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import os
import pickle

SEED = 42
np.random.seed(SEED)
CURRENT_YEAR = 2024


# ─────────────────────────────────────────────
# 1. LOAD DATASET
# ─────────────────────────────────────────────
def load_data(filepath=None):
    """
    Load the car dataset.
    - If a CSV filepath is provided, loads from file.
    - Otherwise creates a realistic sample dataset for demonstration.
    """
    if filepath and os.path.exists(filepath):
        print(f"[INFO] Loading dataset from {filepath}...")
        df = pd.read_csv(filepath)
    else:
        print("[INFO] No dataset file found. Generating sample dataset...")
        df = generate_sample_data()

    print(f"  Shape   : {df.shape}")
    print(f"  Columns : {list(df.columns)}")
    return df


def generate_sample_data(n=500):
    """Generate a realistic sample car dataset matching Car Dekho format."""
    np.random.seed(SEED)

    car_names = [
        "Maruti Swift", "Honda City", "Hyundai i20", "Toyota Innova",
        "Ford EcoSport", "Maruti Baleno", "Tata Nexon", "Hyundai Creta",
        "Maruti Alto", "Honda Amaze", "Volkswagen Polo", "Renault Duster"
    ]

    years        = np.random.randint(2005, 2021, n)
    present_price= np.round(np.random.uniform(3.0, 25.0, n), 2)
    kms_driven   = np.random.randint(5000, 200000, n)
    fuel_type    = np.random.choice(["Petrol", "Diesel", "CNG"], n, p=[0.55, 0.40, 0.05])
    seller_type  = np.random.choice(["Dealer", "Individual"], n, p=[0.55, 0.45])
    transmission = np.random.choice(["Manual", "Automatic"], n, p=[0.75, 0.25])
    owner        = np.random.choice([0, 1, 2, 3], n, p=[0.55, 0.30, 0.10, 0.05])

    # Selling price depends on present price, age, kms
    age = CURRENT_YEAR - years
    noise = np.random.normal(0, 0.5, n)
    selling_price = np.round(
        present_price * 0.6
        - age * 0.3
        - kms_driven / 50000
        + (transmission == "Automatic") * 1.5
        + (fuel_type == "Diesel") * 0.8
        - owner * 0.5
        + noise, 2
    )
    selling_price = np.clip(selling_price, 0.5, None)

    df = pd.DataFrame({
        "Car_Name"      : np.random.choice(car_names, n),
        "Year"          : years,
        "Selling_Price" : selling_price,
        "Present_Price" : present_price,
        "Kms_Driven"    : kms_driven,
        "Fuel_Type"     : fuel_type,
        "Seller_Type"   : seller_type,
        "Transmission"  : transmission,
        "Owner"         : owner
    })
    return df


# ─────────────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS
# ─────────────────────────────────────────────
def eda(df):
    print("\n── Descriptive Statistics ─────────────────────────────")
    print(df.describe())

    print("\n── Missing Values ──────────────────────────────────────")
    print(df.isnull().sum())

    print("\n── Value Counts ────────────────────────────────────────")
    for col in ["Fuel_Type", "Seller_Type", "Transmission", "Owner"]:
        print(f"\n{col}:\n{df[col].value_counts()}")

    # ── Plots ─────────────────────────────────────────────────────────

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle("Car Price — Exploratory Data Analysis", fontsize=16, fontweight="bold")

    # 1. Selling price distribution
    sns.histplot(df["Selling_Price"], bins=30, kde=True, ax=axes[0][0], color="steelblue")
    axes[0][0].set_title("Distribution of Selling Price")
    axes[0][0].set_xlabel("Selling Price (Lakhs)")

    # 2. Selling price vs Present price
    axes[0][1].scatter(df["Present_Price"], df["Selling_Price"], alpha=0.5, color="coral")
    axes[0][1].set_title("Selling Price vs Present Price")
    axes[0][1].set_xlabel("Present Price"); axes[0][1].set_ylabel("Selling Price")

    # 3. Fuel type distribution
    df["Fuel_Type"].value_counts().plot(kind="bar", ax=axes[0][2], color=["#4CAF50","#2196F3","#FF9800"])
    axes[0][2].set_title("Fuel Type Distribution")
    axes[0][2].tick_params(axis="x", rotation=0)

    # 4. Selling price by transmission
    sns.boxplot(x="Transmission", y="Selling_Price", data=df, ax=axes[1][0])
    axes[1][0].set_title("Selling Price by Transmission")

    # 5. Selling price by fuel type
    sns.boxplot(x="Fuel_Type", y="Selling_Price", data=df, ax=axes[1][1])
    axes[1][1].set_title("Selling Price by Fuel Type")

    # 6. Kms driven vs selling price
    axes[1][2].scatter(df["Kms_Driven"], df["Selling_Price"], alpha=0.4, color="purple")
    axes[1][2].set_title("Kms Driven vs Selling Price")
    axes[1][2].set_xlabel("Kms Driven"); axes[1][2].set_ylabel("Selling Price")

    plt.tight_layout()
    plt.savefig("car_eda.png", dpi=150)
    print("[INFO] Saved car_eda.png")
    plt.close()

    # Correlation heatmap (numeric only)
    plt.figure(figsize=(9, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(),
                annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig("car_correlation.png", dpi=150)
    print("[INFO] Saved car_correlation.png")
    plt.close()


# ─────────────────────────────────────────────
# 3. DATA PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    """
    Steps:
      1. Drop Car_Name (high cardinality, not useful directly)
      2. Feature engineering: Years_Since_Purchase
      3. Handle missing values
      4. Encode categorical variables
      5. Split features / target
    """
    df = df.copy()

    # ── Drop Car_Name ────────────────────────
    if "Car_Name" in df.columns:
        df.drop("Car_Name", axis=1, inplace=True)

    # ── Feature engineering ──────────────────
    df["Years_Since_Purchase"] = CURRENT_YEAR - df["Year"]
    df.drop("Year", axis=1, inplace=True)

    # ── Missing values ───────────────────────
    for col in df.columns:
        if df[col].dtype == "object":
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

    # ── Encode categorical variables ─────────
    label_encoders = {}
    for col in ["Fuel_Type", "Seller_Type", "Transmission"]:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
            print(f"  Encoded {col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    print(f"\n  Final features: {list(df.drop('Selling_Price', axis=1).columns)}")

    X = df.drop("Selling_Price", axis=1)
    y = df["Selling_Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED
    )
    print(f"  Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")
    return X_train, X_test, y_train, y_test, label_encoders, X.columns.tolist()


# ─────────────────────────────────────────────
# 4. MODEL TRAINING & COMPARISON
# ─────────────────────────────────────────────
def train_and_compare(X_train, X_test, y_train, y_test):
    """Train multiple models and compare performance."""

    models = {
        "Linear Regression" : LinearRegression(),
        "Random Forest"     : RandomForestRegressor(n_estimators=200, random_state=SEED),
        "Gradient Boosting" : GradientBoostingRegressor(n_estimators=200, random_state=SEED),
    }

    results = []
    print("\n── Model Comparison ────────────────────────────────────")
    print(f"{'Model':<22} {'RMSE':>8} {'MAE':>8} {'R²':>8} {'CV-R²':>8}")
    print("─" * 55)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae  = mean_absolute_error(y_test, y_pred)
        r2   = r2_score(y_test, y_pred)
        cv   = cross_val_score(model, X_train, y_train, cv=5, scoring="r2").mean()

        print(f"{name:<22} {rmse:>8.3f} {mae:>8.3f} {r2:>8.4f} {cv:>8.4f}")
        results.append({"Model": name, "RMSE": rmse, "MAE": mae,
                        "R2": r2, "CV_R2": cv, "model": model, "y_pred": y_pred})

    best = max(results, key=lambda d: d["R2"])
    print(f"\n✅ Best Model: {best['Model']}  (R² = {best['R2']:.4f})")
    return results, best


# ─────────────────────────────────────────────
# 5. HYPERPARAMETER TUNING (RandomizedSearchCV)
# ─────────────────────────────────────────────
def tune_random_forest(X_train, y_train):
    print("\n[INFO] Tuning Random Forest with RandomizedSearchCV...")

    param_dist = {
        "n_estimators"      : [100, 200, 300, 500],
        "max_depth"         : [None, 5, 10, 15, 20],
        "min_samples_split" : [2, 5, 10],
        "min_samples_leaf"  : [1, 2, 4],
        "max_features"      : ["sqrt", "log2", None],
    }

    rf = RandomForestRegressor(random_state=SEED)
    rs = RandomizedSearchCV(
        rf, param_dist, n_iter=30, cv=5,
        scoring="r2", n_jobs=-1, random_state=SEED, verbose=1
    )
    rs.fit(X_train, y_train)

    print(f"  Best params : {rs.best_params_}")
    print(f"  Best CV R²  : {rs.best_score_:.4f}")
    return rs.best_estimator_


# ─────────────────────────────────────────────
# 6. VISUALISATIONS
# ─────────────────────────────────────────────
def plot_results(y_test, y_pred, model_name, feature_names, model):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Results — {model_name}", fontsize=14, fontweight="bold")

    # 1. Actual vs Predicted
    axes[0].scatter(y_test, y_pred, alpha=0.6, color="steelblue", edgecolors="k", lw=0.3)
    mn, mx = min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2)
    axes[0].set_xlabel("Actual Price"); axes[0].set_ylabel("Predicted Price")
    axes[0].set_title("Actual vs Predicted")

    # 2. Residuals
    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5, color="coral", edgecolors="k", lw=0.3)
    axes[1].axhline(0, color="black", lw=1.5, ls="--")
    axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Residuals")
    axes[1].set_title("Residual Plot")

    # 3. Feature Importance
    try:
        imp = model.feature_importances_
        idx = np.argsort(imp)[::-1]
        axes[2].bar(range(len(imp)), imp[idx], color="teal")
        axes[2].set_xticks(range(len(imp)))
        axes[2].set_xticklabels([feature_names[i] for i in idx], rotation=45, ha="right")
        axes[2].set_title("Feature Importances")
    except AttributeError:
        axes[2].text(0.5, 0.5, "Not available\nfor this model",
                     ha="center", va="center", transform=axes[2].transAxes)

    plt.tight_layout()
    fname = f"car_results_{model_name.replace(' ', '_')}.png"
    plt.savefig(fname, dpi=150)
    print(f"[INFO] Saved {fname}")
    plt.close()


# ─────────────────────────────────────────────
# 7. DEPLOYMENT — PREDICTION FUNCTION
# ─────────────────────────────────────────────
def predict_car_price(model, label_encoders, car_details):
    """
    Predict selling price for a new car.

    Args:
        model          : Trained model.
        label_encoders : Dict of LabelEncoder objects.
        car_details    : Dict with keys:
                         Year, Present_Price, Kms_Driven,
                         Fuel_Type, Seller_Type, Transmission, Owner

    Returns:
        predicted_price : Float (in Lakhs)

    Example:
        predict_car_price(model, encoders, {
            "Year": 2017, "Present_Price": 9.85,
            "Kms_Driven": 45000, "Fuel_Type": "Petrol",
            "Seller_Type": "Dealer", "Transmission": "Manual",
            "Owner": 0
        })
    """
    row = car_details.copy()

    # Feature engineering
    row["Years_Since_Purchase"] = CURRENT_YEAR - row.pop("Year")

    # Encode categoricals
    for col, le in label_encoders.items():
        if col in row:
            row[col] = le.transform([row[col]])[0]

    # Build input array in correct order
    feature_order = [
        "Present_Price", "Kms_Driven", "Fuel_Type",
        "Seller_Type", "Transmission", "Owner", "Years_Since_Purchase"
    ]
    x = np.array([[row[f] for f in feature_order]])

    price = model.predict(x)[0]
    return max(round(price, 2), 0.0)


def save_model(model, label_encoders, path="car_price_model.pkl"):
    with open(path, "wb") as f:
        pickle.dump({"model": model, "label_encoders": label_encoders}, f)
    print(f"[INFO] Model saved → {path}")


def load_model(path="car_price_model.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data["model"], data["label_encoders"]


# ─────────────────────────────────────────────
# 8. MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Step 1: Load data ─────────────────────────────────────────────────
    # To use your own dataset: df = load_data("car data.csv")
    df = load_data()

    # ── Step 2: EDA ───────────────────────────────────────────────────────
    eda(df)

    # ── Step 3: Preprocess ────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, label_encoders, feature_names = preprocess(df)

    # ── Step 4: Train & compare models ───────────────────────────────────
    results, best = train_and_compare(X_train, X_test, y_train, y_test)

    # ── Step 5: Visualise best model ─────────────────────────────────────
    plot_results(y_test, best["y_pred"], best["Model"],
                 feature_names, best["model"])

    # ── Step 6: Hyperparameter tuning ────────────────────────────────────
    tuned_rf = tune_random_forest(X_train, y_train)
    y_tuned  = tuned_rf.predict(X_test)

    print("\n── Tuned Random Forest Results ─────────────────────────")
    print(f"  RMSE : {np.sqrt(mean_squared_error(y_test, y_tuned)):.3f}")
    print(f"  MAE  : {mean_absolute_error(y_test, y_tuned):.3f}")
    print(f"  R²   : {r2_score(y_test, y_tuned):.4f}")

    plot_results(y_test, y_tuned, "Tuned_Random_Forest",
                 feature_names, tuned_rf)

    # ── Step 7: Save model ───────────────────────────────────────────────
    save_model(tuned_rf, label_encoders)

    # ── Step 8: Demo prediction ──────────────────────────────────────────
    sample_car = {
        "Year"          : 2017,
        "Present_Price" : 9.85,
        "Kms_Driven"    : 45000,
        "Fuel_Type"     : "Petrol",
        "Seller_Type"   : "Dealer",
        "Transmission"  : "Manual",
        "Owner"         : 0
    }

    predicted = predict_car_price(tuned_rf, label_encoders, sample_car)
    print(f"\n🚗 Sample Prediction:")
    for k, v in sample_car.items():
        print(f"   {k:<18}: {v}")
    print(f"   {'Predicted Price':<18}: ₹ {predicted} Lakhs")

    print("\n✅ Car Price Prediction task complete!")
    print("   Files generated:")
    print("   - car_eda.png")
    print("   - car_correlation.png")
    print("   - car_results_*.png")
    print("   - car_price_model.pkl")
