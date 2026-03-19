import pandas as pd

from ovarian_prediction.preprocessing import MICEImputer, encode_categoricals


def test_encode_categoricals_creates_dummy_columns():
    df = pd.DataFrame({"Protocol": ["Long", "Antagonist"], "value": [1, 2]})
    encoded = encode_categoricals(df, cat_cols=["Protocol"])
    assert "value" in encoded.columns
    assert any(column.startswith("Protocol_") for column in encoded.columns)


def test_mice_imputer_fills_numeric_and_categorical_missing_values():
    df = pd.DataFrame(
        {
            "AMH": [1.1, None, 2.5, 3.0],
            "POIorDOR": pd.Series(["No", None, "Yes", "No"], dtype="category"),
        }
    )
    imputer = MICEImputer(random_state=42)
    transformed = imputer.fit_transform(df)
    assert transformed["AMH"].isna().sum() == 0
    assert transformed["POIorDOR"].isna().sum() == 0
