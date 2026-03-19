from typing import Dict, Tuple

import pandas as pd

from ovarian_prediction.models import OvarianMLSystem
from ovarian_prediction.preprocessing import OvarianPreprocessor


def train_from_dataframe(
    df: pd.DataFrame,
    tune: bool = True,
    n_trials: int = 50,
    random_state: int = 777,
) -> Tuple[OvarianPreprocessor, OvarianMLSystem, Dict[str, Dict]]:
    proc = OvarianPreprocessor(random_state=random_state)
    data = proc.fit_transform(df)
    ml_system = OvarianMLSystem(n_trials=n_trials, random_state=random_state)
    ml_system.train_all(data, tune=tune)
    results = ml_system.evaluate_all(data)
    return proc, ml_system, results
