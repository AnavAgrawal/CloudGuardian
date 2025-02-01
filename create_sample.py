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

    # Instead of random sampling, let's create the pattern
    suspicious_entries = train_df[train_df['is_suspicious'] == 1]
    normal_entries = train_df[train_df['is_suspicious'] == 0]

    # Determine how many complete patterns we can create
    num_suspicious_available = len(suspicious_entries)
    # Let's create up to 30 patterns
    num_patterns = min(30, num_suspicious_available)

    # Create the patterned dataset
    patterned_df = pd.DataFrame()
    for i in range(num_patterns):
        # Get 6 normal entries
        pattern_normal = normal_entries.sample(n=6, random_state=i)
        # Get 1 suspicious entry
        pattern_suspicious = suspicious_entries.sample(n=1, random_state=i)
        # Combine in order
        pattern = pd.concat([pattern_normal, pattern_suspicious])
        patterned_df = pd.concat([patterned_df, pattern])

    # Reset index
    patterned_df = patterned_df.reset_index(drop=True)

    # Save to CSV
    output_path = os.path.join("data", "sample_data.csv")
    patterned_df.to_csv(output_path, index=False)

    # Print statistics
    total_rows = len(patterned_df)
    suspicious_count = len(patterned_df[patterned_df['is_suspicious'] == 1])
    suspicious_percentage = (suspicious_count / total_rows) * 100

    print("\nPatterned Dataset Created:")
    print(f"Total Rows: {total_rows}")
    print(f"Number of complete patterns: {num_patterns}")
    print(f"Suspicious Entries: {
          suspicious_count} ({suspicious_percentage:.1f}%)")
    print(f"Normal Entries: {
          total_rows - suspicious_count} ({100 - suspicious_percentage:.1f}%)")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    create_sample_dataset()
