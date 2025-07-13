import pandas as pd
from autogluon.multimodal import MultiModalPredictor

from pathlib import Path
import argparse

def main(
    dataset_name: str,
    output_path: str,
    leaderboard_path: str,
    time_limit: int = 3600,
    results_dir: str = "autogluon_results"
):
    # Set paths for images and CSVs
    data_root = Path("./data") / dataset_name
    train_csv = data_root / "train.csv"
    test_csv = data_root / "test.csv"
    
    # Read CSVs into DataFrames
    train_df = pd.read_csv(train_csv)
    test_df = pd.read_csv(test_csv)

    # Fix column names for AutoGluon
    # They must be 'image' and 'label' (train) and 'image' (test)
    train_df = train_df.rename(columns={"image_file_name": "image"})
    test_df = test_df.rename(columns={"image_file_name": "image"})
    
    # Add full path for images (required for AG)
    train_df['image'] = train_df['image'].apply(lambda x: str(data_root / f"images_train/{x}"))
    test_df['image'] = test_df['image'].apply(lambda x: str(data_root / f"images_test/{x}"))
    
    # Train AutoGluon ImagePredictor
    predictor = MultiModalPredictor(label = 'label', path=results_dir)
    predictor.fit(train_df, time_limit=time_limit)
    
    # Predict on test set
    preds = predictor.predict(test_df)
    preds['image'] = test_df['image']  # keep image names for clarity
    preds.to_csv(output_path, index=False)
    print(f"Test predictions saved to {output_path}")
    
    # Save leaderboard
    lb = predictor.leaderboard(test_df, extra_info=True, silent=True)
    lb.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard saved to {leaderboard_path}")
    print(lb.head())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset: 'flowers', 'fashion', 'emotions'")
    parser.add_argument("--output-path", type=str, default="ag_preds.csv", help="Where to save predictions")
    parser.add_argument("--leaderboard-path", type=str, default="ag_leaderboard.csv", help="Where to save leaderboard")
    parser.add_argument("--time-limit", type=int, default=3600, help="Time limit in seconds for AG")
    parser.add_argument("--results-dir", type=str, default="autogluon_results", help="Where to save AG models/checkpoints")

    args = parser.parse_args()
    main(
        dataset_name=args.dataset,
        output_path=args.output_path,
        leaderboard_path=args.leaderboard_path,
        time_limit=args.time_limit,
        results_dir=args.results_dir,
    )
