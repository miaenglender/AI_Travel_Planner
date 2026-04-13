import pandas as pd
from collections import defaultdict
import numpy as np


# -----------------------
# GLOBAL SAFE DEFAULT
# -----------------------
def default_elo():
    return 1500


# -----------------------
# ELO SYSTEM
# -----------------------
def calculate_elo(results, k=20, base=1500):
    elo = defaultdict(default_elo)

    for _, row in results.iterrows():
        w, l = row["WTeamID"], row["LTeamID"]
        w_score, l_score = row["WScore"], row["LScore"]

        ew, el = elo[w], elo[l]

        prob_w = 1 / (1 + 10 ** ((el - ew) / 400))

        margin = w_score - l_score
        margin_multiplier = np.log(abs(margin) + 1) * (
            2.2 / ((ew - el) * 0.001 + 2.2)
        )

        update = k * margin_multiplier * (1 - prob_w)

        elo[w] += update
        elo[l] -= update

    return elo


# -----------------------
# TEAM STATS
# -----------------------
def compute_team_stats(results):
    wins = defaultdict(int)
    losses = defaultdict(int)
    margin = defaultdict(int)

    for _, row in results.iterrows():
        w, l = row["WTeamID"], row["LTeamID"]
        w_score, l_score = row["WScore"], row["LScore"]

        wins[w] += 1
        losses[l] += 1

        margin[w] += (w_score - l_score)
        margin[l] -= (w_score - l_score)

    stats = {}

    teams = set(list(wins.keys()) + list(losses.keys()))

    for team in teams:
        games = wins[team] + losses[team]

        stats[team] = {
            "win_pct": wins[team] / games if games else 0,
            "avg_margin": margin[team] / games if games else 0,
        }

    return stats


# -----------------------
# TRAINING DATA
# -----------------------
def build_training_data(results, seed_map, live_map):
    print("Building features...")

    elo = calculate_elo(results)
    stats = compute_team_stats(results)

    X = []
    y = []

    for _, row in results.iterrows():
        w, l = row["WTeamID"], row["LTeamID"]

        seed_diff = seed_map.get(w, 16) - seed_map.get(l, 16)
        elo_diff = elo[w] - elo[l]

        w_stats = stats.get(w, {"win_pct": 0, "avg_margin": 0})
        l_stats = stats.get(l, {"win_pct": 0, "avg_margin": 0})

        win_pct_diff = w_stats["win_pct"] - l_stats["win_pct"]
        margin_diff = w_stats["avg_margin"] - l_stats["avg_margin"]

        features = [
            seed_diff,
            elo_diff,
            win_pct_diff,
            margin_diff,
        ]

        X.append(features)
        y.append(1)

        X.append([-f for f in features])
        y.append(0)

    return pd.DataFrame(X), pd.Series(y)

def build_full_dataset(results, seed_map):
    from features import calculate_elo, compute_team_stats

    elo = calculate_elo(results)
    stats = compute_team_stats(results)

    X = []
    y = []

    for _, row in results.iterrows():
        w, l = row["WTeamID"], row["LTeamID"]

        seed_diff = seed_map.get(w, 16) - seed_map.get(l, 16)
        elo_diff = elo[w] - elo[l]

        w_stats = stats.get(w, {"win_pct": 0, "avg_margin": 0})
        l_stats = stats.get(l, {"win_pct": 0, "avg_margin": 0})

        features = [
            seed_diff,
            elo_diff,
            w_stats["win_pct"] - l_stats["win_pct"],
            w_stats["avg_margin"] - l_stats["avg_margin"],
        ]

        X.append(features)
        y.append(1)

        X.append([-f for f in features])
        y.append(0)

    return pd.DataFrame(X), pd.Series(y), elo, stats

