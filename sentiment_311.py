"""
Sentiment / complaint-intensity from 311 Request_Type only.
Outputs: Dataset/sentiment_by_cell_311.csv (merge into feature_matrix on grid_cell).
"""
import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "Dataset")
GRID_SIZE = 0.005  # same as auto_pipeline

def assign_grid_cell(lat, lon, grid_size=GRID_SIZE):
    grid_lat = round(round(lat / grid_size) * grid_size, 6)
    grid_lon = round(round(lon / grid_size) * grid_size, 6)
    return f"{grid_lat:.4f}_{grid_lon:.4f}"

# Complaint-intensity: more negative/urgent phrases -> higher score (0 = neutral, 1 = very negative)
NEGATIVE_PHRASES = [
    "nuisance", "stagnant", "mosquito", "ditch", "drainage", "overgrown", "debris",
    "vacant", "illegal dump", "standing water", "sewage", "waste", "weed", "junk",
    "rodent", "flood", "emergency", "hazard", "unsafe", "complaint", "dump", "odor",
]
def score_request_type(text):
    if pd.isna(text) or str(text).strip() == "":
        return 0.0
    t = str(text).lower()
    score = 0.0
    for phrase in NEGATIVE_PHRASES:
        if phrase in t:
            score += 0.15
    return min(1.0, score)

def main():
    path_311 = os.path.join(DATASET_DIR, "311_requests_cleaned.csv")
    df = pd.read_csv(path_311, usecols=["Request_Type", "Latitude", "Longitude"], low_memory=False)
    df = df.dropna(subset=["Latitude", "Longitude", "Request_Type"])

    df["grid_cell"] = [
        assign_grid_cell(lat, lon)
        for lat, lon in zip(df["Latitude"], df["Longitude"])
    ]
    df["sentiment_score"] = df["Request_Type"].map(score_request_type)

    agg = df.groupby("grid_cell").agg(
        sentiment_mean=("sentiment_score", "mean"),
        sentiment_count=("sentiment_score", "count"),
        sentiment_std=("sentiment_score", "std"),
    ).reset_index()
    agg["sentiment_std"] = agg["sentiment_std"].fillna(0)

    out_path = os.path.join(DATASET_DIR, "sentiment_by_cell_311.csv")
    agg.to_csv(out_path, index=False)
    print(f"Wrote {out_path} with {len(agg)} grid cells.")
    return agg

if __name__ == "__main__":
    main()