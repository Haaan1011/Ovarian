import numpy as np
import pandas as pd


def make_synthetic_dataset(n: int = 1000, random_state: int = 777) -> pd.DataFrame:
    """Generate synthetic tabular data to exercise the full ML workflow."""
    rng = np.random.default_rng(random_state)

    df = pd.DataFrame(
        {
            "AMH": np.clip(rng.lognormal(0.5, 0.8, n), 0.01, 20),
            "AFC": np.clip(rng.poisson(10, n).astype(float), 1, 40),
            "FSH": np.clip(rng.normal(7, 3, n), 1, 30),
            "LH": np.clip(rng.normal(5, 2, n), 0.5, 20),
            "Age": np.clip(rng.normal(32, 5, n), 20, 45),
            "P": np.clip(rng.lognormal(-1, 0.5, n), 0.01, 2),
            "Weight": np.clip(rng.normal(58, 10, n), 35, 100),
            "DBP": np.clip(rng.normal(76, 10, n), 50, 110),
            "WBC": np.clip(rng.normal(6, 1.5, n), 2, 15),
            "ALT": np.clip(rng.lognormal(2.8, 0.4, n), 5, 100),
            "RBC": np.clip(rng.normal(4.2, 0.4, n), 2.5, 6),
            "Duration": np.clip(rng.exponential(2, n), 0.5, 15),
            "PLT": np.clip(rng.normal(220, 50, n), 50, 450),
            "POIorDOR": rng.choice(["Yes", "No"], n, p=[0.15, 0.85]),
            "PCOS": rng.choice(["Yes", "No"], n, p=[0.12, 0.88]),
            "Protocol": rng.choice(["Long", "Short", "Antagonist"], n, p=[0.5, 0.2, 0.3]),
            "Initial.FSH": np.clip(rng.normal(150, 50, n), 75, 300),
            "Recombinant": rng.choice(["Yes", "No"], n, p=[0.6, 0.4]),
            "Use.LH": rng.choice(["Yes", "No"], n, p=[0.3, 0.7]),
            "Num.oocytes": np.clip(rng.poisson(10, n).astype(float), 0, 40),
        }
    )

    por_score = (
        -df["AMH"] * 0.5
        - df["AFC"] * 0.3
        + df["FSH"] * 0.4
        + df["Age"] * 0.06
        + (df["POIorDOR"] == "Yes").astype(float) * 1.5
    )
    por_prob = 1 / (1 + np.exp(-por_score + por_score.median()))
    df["POR"] = (rng.random(n) < por_prob * 0.22 / por_prob.mean()).map(
        {True: "Yes", False: "No"}
    )

    hor_score = (
        df["AMH"] * 0.6
        + df["AFC"] * 0.4
        - df["FSH"] * 0.2
        - df["Age"] * 0.05
        + (df["PCOS"] == "Yes").astype(float) * 1.2
    )
    hor_prob = 1 / (1 + np.exp(-hor_score + hor_score.median()))
    df["HOR"] = (rng.random(n) < hor_prob * 0.15 / hor_prob.mean()).map(
        {True: "Yes", False: "No"}
    )

    for col, miss_rate in [
        ("AMH", 0.12),
        ("AFC", 0.08),
        ("LH", 0.05),
        ("P", 0.06),
        ("ALT", 0.04),
    ]:
        mask = rng.random(n) < miss_rate
        df.loc[mask, col] = np.nan

    return df
