import pandas as pd


def load_data():
    # MAIN TRAINING DATA
    results = pd.read_csv("data/regular_season_results.csv")

    # TOURNAMENT 
    tourney = pd.read_csv("data/TourneyCompactResults.csv")

    # SEEDS FILE 
    seeds = pd.read_csv("data/seeds.csv")

    # TEAM SEEDS MAP
    seed_map = dict(
        zip(
            seeds["TeamID"],
            seeds["Seed"].str.extract(r"(\d+)")[0].astype(int)
        )
    )

    return results, seed_map