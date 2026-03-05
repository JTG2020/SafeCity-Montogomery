"""
One-off: merge Dataset/sentiment_by_cell_311.csv into Dataset/feature_matrix.csv.
Writes Dataset/feature_matrix_with_sentiment.csv (pipeline unchanged until you wire this in).
"""
import os
import pandas as pd

DATASET_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Dataset")

def main():
    fm_path = os.path.join(DATASET_DIR, "feature_matrix.csv")
    sent_path = os.path.join(DATASET_DIR, "sentiment_by_cell_311.csv")
    out_path = os.path.join(DATASET_DIR, "feature_matrix_with_sentiment.csv")

    fm = pd.read_csv(fm_path)
    sent = pd.read_csv(sent_path)

    fm = fm.merge(sent, on="grid_cell", how="left")
    fm["sentiment_mean"] = fm["sentiment_mean"].fillna(0)
    fm["sentiment_count"] = fm["sentiment_count"].fillna(0)
    fm["sentiment_std"] = fm["sentiment_std"].fillna(0)

    fm.to_csv(out_path, index=False)
    print(f"Wrote {out_path} — shape {fm.shape} (added sentiment_mean, sentiment_count, sentiment_std)")

if __name__ == "__main__":
    main()