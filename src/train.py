import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from data_loader import load_data
from features import build_training_data
from live_stats import load_live_team_stats


def main():
    print("Loading data...")
    results, seed_map = load_data()

    print("Loading live stats...")
    live_stats = load_live_team_stats()

    live_stats = live_stats.dropna(subset=["TeamName"])
    live_stats["TeamName"] = live_stats["TeamName"].str.lower().str.strip()
    live_stats = live_stats.groupby("TeamName", as_index=False).mean(numeric_only=True)

    live_map = {
        row["TeamName"]: row.to_dict()
        for _, row in live_stats.iterrows()
    }

    print("Building features...")
    X, y = build_training_data(results, seed_map, live_map)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))

    print(f"Accuracy: {acc:.4f}")

    joblib.dump(model, "models/model.pkl")
    print("Model saved!")
    

if __name__ == "__main__":
    main()