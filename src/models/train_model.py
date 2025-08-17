import warnings

import joblib
import optuna
import pandas as pd
import xgboost as xgb
from category_encoders import CatBoostEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


def load_and_prepare_data(data_path="data/multisim_dataset.parquet"):
    """Load and prepare data for training"""
    df = pd.read_parquet(data_path)

    # Identify categorical and numerical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Remove target and identifier columns
    if "target" in numerical_cols:
        numerical_cols.remove("target")
    if "telephone_number" in categorical_cols:
        categorical_cols.remove("telephone_number")

    # Feature engineering
    df["tenure_years"] = df["tenure"] / 365.25
    df["age_tenure_ratio"] = pd.to_numeric(df["age"], errors="coerce") / (
        df["tenure_years"] + 1
    )
    df["device_change_frequency"] = df["dev_num"] / (df["tenure_years"] + 1)

    # Add new numerical features
    numerical_cols_final = numerical_cols + [
        "tenure_years",
        "age_tenure_ratio",
        "device_change_frequency",
    ]

    # Prepare feature matrix and target
    X = df[categorical_cols + numerical_cols_final].copy()
    y = df["target"].copy()

    return X, y, categorical_cols, numerical_cols_final


def create_preprocessing_pipeline(categorical_cols, numerical_cols):
    """Create preprocessing pipeline"""
    numerical_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("catboost", CatBoostEncoder(return_df=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        [
            ("num", numerical_pipeline, numerical_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def optimize_hyperparameters(X_train, y_train, preprocessor, n_trials=30):
    """Optimize hyperparameters using Optuna"""

    def objective(trial):
        n_estimators = trial.suggest_int("n_estimators", 100, 500)
        max_depth = trial.suggest_int("max_depth", 3, 10)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
        subsample = trial.suggest_float("subsample", 0.6, 1.0)
        colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
        reg_alpha = trial.suggest_float("reg_alpha", 0, 2)
        reg_lambda = trial.suggest_float("reg_lambda", 0, 2)

        xgb_model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            eval_metric="logloss",
        )

        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", xgb_model)])

        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring="roc_auc")
        return cv_scores.mean()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    return study.best_trial.params


def train_model(X_train, y_train, best_params, preprocessor):
    """Train final model with best parameters"""
    best_xgb_pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                xgb.XGBClassifier(
                    **best_params, random_state=42, eval_metric="logloss"
                ),
            ),
        ]
    )

    best_xgb_pipeline.fit(X_train, y_train)
    return best_xgb_pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate trained model"""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print("Model Evaluation Results:")
    print("=" * 40)
    print(classification_report(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc_score(y_test, y_pred_proba):.4f}")

    return y_pred, y_pred_proba


def save_model(model, model_path="models/multisim_model.joblib"):
    """Save trained model"""
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")


def main():
    """Main training pipeline"""
    print("Starting model training pipeline...")

    # Load and prepare data
    print("Loading and preparing data...")
    X, y, categorical_cols, numerical_cols = load_and_prepare_data()
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

    # Create preprocessing pipeline
    print("Creating preprocessing pipeline...")
    preprocessor = create_preprocessing_pipeline(categorical_cols, numerical_cols)

    # Optimize hyperparameters
    print("Optimizing hyperparameters...")
    best_params = optimize_hyperparameters(X_train, y_train, preprocessor)
    print(f"Best parameters: {best_params}")

    # Train final model
    print("Training final model...")
    model = train_model(X_train, y_train, best_params, preprocessor)

    # Evaluate model
    print("Evaluating model...")
    evaluate_model(model, X_test, y_test)

    # Save model
    save_model(model)

    print("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
