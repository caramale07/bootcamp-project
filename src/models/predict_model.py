import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


def load_model(model_path="models/multisim_model.joblib"):
    """Load trained model"""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = joblib.load(model_path)
    print(f"Model loaded from {model_path}")
    return model


def prepare_prediction_data(data_path):
    """Prepare data for prediction (same preprocessing as training)"""
    df = pd.read_parquet(data_path)

    # Identify categorical and numerical features
    categorical_cols = df.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

    # Remove target and identifier columns if present
    if "target" in numerical_cols:
        numerical_cols.remove("target")
    if "telephone_number" in categorical_cols:
        categorical_cols.remove("telephone_number")

    # Feature engineering (same as training)
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

    # Prepare feature matrix
    X = df[categorical_cols + numerical_cols_final].copy()

    return X, df


def make_predictions(model, X, include_probabilities=True):
    """Make predictions using trained model"""
    predictions = model.predict(X)

    if include_probabilities:
        probabilities = model.predict_proba(X)
        return predictions, probabilities

    return predictions


def save_predictions(
    df_original,
    predictions,
    probabilities=None,
    output_path="predictions/multisim_predictions.csv",
):
    """Save predictions to file"""
    # Create predictions dataframe
    results_df = (
        df_original[["telephone_number"]].copy()
        if "telephone_number" in df_original.columns
        else pd.DataFrame()
    )
    results_df["predicted_multisim"] = predictions

    if probabilities is not None:
        results_df["probability_not_multisim"] = probabilities[:, 0]
        results_df["probability_multisim"] = probabilities[:, 1]

    # Create output directory if it doesn't exist
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Save to file
    results_df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")

    return results_df


def predict_single_sample(model, sample_data):
    """Make prediction for a single sample"""
    prediction = model.predict([sample_data])[0]
    probability = model.predict_proba([sample_data])[0]

    return prediction, probability


def main():
    """Main prediction pipeline"""
    parser = argparse.ArgumentParser(
        description="Run predictions using trained multisim model"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/multisim_dataset.parquet",
        help="Path to data file for prediction",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/multisim_model.joblib",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="predictions/multisim_predictions.csv",
        help="Path to save predictions",
    )
    parser.add_argument(
        "--sample", action="store_true", help="Run prediction on a sample of the data"
    )

    args = parser.parse_args()

    print("Starting prediction pipeline...")

    # Load model
    print("Loading trained model...")
    model = load_model(args.model)

    # Load and prepare data
    print(f"Loading data from {args.data}...")
    X, df_original = prepare_prediction_data(args.data)
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} features")

    # Make predictions
    print("Making predictions...")
    predictions, probabilities = make_predictions(model, X)

    # Print summary
    unique_preds, counts = np.unique(predictions, return_counts=True)
    print("Prediction summary:")
    for pred, count in zip(unique_preds, counts):
        percentage = (count / len(predictions)) * 100
        label = "Multisim" if pred == 1 else "Not Multisim"
        print(f"  {label}: {count} ({percentage:.1f}%)")

    # Save predictions
    results_df = save_predictions(df_original, predictions, probabilities, args.output)

    # Show sample predictions
    if args.sample:
        print("\nSample predictions:")
        print(results_df.head(10))

    print("Prediction pipeline completed successfully!")


if __name__ == "__main__":
    main()
