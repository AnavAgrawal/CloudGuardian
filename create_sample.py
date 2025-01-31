import pandas as pd
import numpy as np
import os
import pickle


def prepare_dataset(df):
    """Prepare features from BETH dataset"""
    df["processId"] = df["processId"].map(lambda x: 0 if x in [0, 1, 2] else 1)
    df["parentProcessId"] = df["parentProcessId"].map(
        lambda x: 0 if x in [0, 1, 2] else 1)
    df["userId"] = df["userId"].map(lambda x: 0 if x < 1000 else 1)
    df["mountNamespace"] = df["mountNamespace"].map(
        lambda x: 0 if x == 4026531840 else 1)
    df["returnValue"] = df["returnValue"].map(
        lambda x: 0 if x == 0 else (1 if x > 0 else 2))
    return df


def create_sample_dataset():
    # Create directories if they don't exist
    os.makedirs('data', exist_ok=True)

    # Load original training dataset
    print("Loading training dataset...")
    train_df = pd.read_csv("data/labelled_training_data.csv")

    # Preprocess features
    print("Preprocessing features...")
    train_df = prepare_dataset(train_df)

    # Load the trained model
    print("Loading trained model...")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Get features for prediction
    features = train_df[["processId", "parentProcessId", "userId",
                        "mountNamespace", "eventId", "argsNum", "returnValue"]]

    # Get model predictions
    print("Getting model predictions...")
    predictions = model.predict(features)
    train_df['is_suspicious'] = np.where(predictions == 1, 1, 0)

    # Count available samples
    suspicious_available = len(train_df[train_df['is_suspicious'] == 1])
    normal_available = len(train_df[train_df['is_suspicious'] == 0])

    print(f"\nAvailable samples:")
    print(f"Suspicious: {suspicious_available}")
    print(f"Normal: {normal_available}")

    # Calculate sample sizes (targeting 20% suspicious)
    # Try to get 40, but take what's available
    suspicious_size = min(30, suspicious_available)
    normal_size = suspicious_size * 5  # Maintain 20% ratio

    print(f"\nSampling {suspicious_size} suspicious and {
          normal_size} normal entries...")

    # Sample from each prediction group
    suspicious_sample = train_df[train_df['is_suspicious'] == 1].sample(
        n=suspicious_size, random_state=40)
    normal_sample = train_df[train_df['is_suspicious']
                             == 0].sample(n=normal_size, random_state=40)

    # Combine and shuffle
    sample_df = pd.concat([suspicious_sample, normal_sample])
    sample_df = sample_df.sample(
        frac=1.0, random_state=42).reset_index(drop=True)

    # # # Keep only necessary columns
    # # columns_to_keep = [
    # #     "processId", "parentProcessId", "userId", "mountNamespace",
    # #     "eventId", "argsNum", "returnValue", "is_suspicious"
    # # ]
    # sample_df = sample_df[columns_to_keep]

    # Save to CSV
    output_path = os.path.join("data", "sample_data.csv")
    sample_df.to_csv(output_path, index=False)

    # Print statistics
    total_rows = len(sample_df)
    suspicious_count = len(sample_df[sample_df['is_suspicious'] == 1])
    suspicious_percentage = (suspicious_count / total_rows) * 100

    print("\nSample Dataset Created:")
    print(f"Total Rows: {total_rows}")
    print(f"Suspicious Entries: {
          suspicious_count} ({suspicious_percentage:.1f}%)")
    print(f"Normal Entries: {
          total_rows - suspicious_count} ({100 - suspicious_percentage:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    create_sample_dataset()
