import streamlit as st
import pandas as pd

from data_loader import load_data
from live_stats import load_live_team_stats
from features import build_full_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# -----------------------
# UI
# -----------------------
st.set_page_config(page_title="March Madness Predictor", layout="centered")

st.title("🏀 March Madness Predictor")
st.write("Predict NCAA game winners")


# -----------------------
# LOAD DATA (ONCE)
# -----------------------
results, seed_map = load_data()


# -----------------------
# LIVE DATA (SAFE)
# -----------------------
@st.cache_data
def get_live_map():
    df = load_live_team_stats()

    df = df.dropna(subset=["TeamName"])
    df["TeamName"] = df["TeamName"].str.lower().str.strip()
    df = df.groupby("TeamName", as_index=False).mean(numeric_only=True)

    return {
        row["TeamName"]: row.to_dict()
        for _, row in df.iterrows()
    }


live_map = get_live_map()


# -----------------------
# PRECOMPUTE DATA (ONCE ONLY)
# -----------------------
@st.cache_data
def get_data():
    return build_full_dataset(results, seed_map)


X, y, elo, stats = get_data()


# -----------------------
# MODEL (ONCE ONLY)
# -----------------------
@st.cache_resource
def train_model():
    model = LogisticRegression(max_iter=2000)
    model.fit(X, y)
    return model


model = train_model()


# -----------------------
# TEAMS
# -----------------------
teams = pd.read_csv("data/Mteams.csv")
teams["TeamName_clean"] = teams["TeamName"].str.lower()

name_to_id = dict(zip(teams["TeamName_clean"], teams["TeamID"]))
id_to_name = dict(zip(teams["TeamID"], teams["TeamName"]))

team_names = sorted(teams["TeamName"].tolist())


# -----------------------
# UI
# -----------------------
team_a = st.selectbox("Team A", team_names)
team_b = st.selectbox("Team B", team_names)


# -----------------------
# PREDICTION (FAST NOW)
# -----------------------
if st.button("Predict Winner"):

    with st.spinner("Thinking... 🤔"):

        a_id = name_to_id[team_a.lower()]
        b_id = name_to_id[team_b.lower()]

        features = [[
            seed_map.get(a_id, 16) - seed_map.get(b_id, 16),
            elo[a_id] - elo[b_id],
            stats[a_id]["win_pct"] - stats[b_id]["win_pct"],
            stats[a_id]["avg_margin"] - stats[b_id]["avg_margin"],
        ]]

        prob = model.predict_proba(features)[0][1]

        if prob > 0.5:
            st.success(f"🏆 {team_a} wins! ({prob:.1%})")
        else:
            st.success(f"🏆 {team_b} wins! ({1-prob:.1%})")