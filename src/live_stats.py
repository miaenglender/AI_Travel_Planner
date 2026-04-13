import pandas as pd

def load_live_team_stats():
    """
    Loads current-season-ish NCAA team stats from SportsReference.
    Used as additional real-world features for prediction.
    """

    url = "https://www.sports-reference.com/cbb/seasons/men/2025-ratings.html"

    try:
        tables = pd.read_html(url)
        df = tables[0]

        # Flatten multi-level columns if they exist
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(-1)

        # Rename common columns safely
        rename_map = {}

        for col in df.columns:
            col_lower = str(col).lower()

            if "school" in col_lower or "team" in col_lower:
                rename_map[col] = "TeamName"
            elif "w-l%" in col_lower or "win%" in col_lower:
                rename_map[col] = "win_pct"
            elif "ortg" in col_lower:
                rename_map[col] = "off_rating"
            elif "drtg" in col_lower:
                rename_map[col] = "def_rating"

        df = df.rename(columns=rename_map)

        # Keep only useful columns if they exist
        keep_cols = [c for c in ["TeamName", "win_pct", "off_rating", "def_rating"] if c in df.columns]
        df = df[keep_cols]

        # Clean team names for matching
        if "TeamName" in df.columns:
            df["TeamName"] = df["TeamName"].astype(str).str.lower()

        # Convert numeric columns safely
        for col in ["win_pct", "off_rating", "def_rating"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        return df

    except Exception as e:
        print("Failed to load live stats:", e)

        # fallback so your app doesn't crash
        return pd.DataFrame(columns=["TeamName", "win_pct", "off_rating", "def_rating"])