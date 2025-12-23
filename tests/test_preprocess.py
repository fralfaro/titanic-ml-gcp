from titanic_ml.data.preprocess import load_data, preprocess


def test_preprocess_returns_dataframe():
    df = load_data()
    df_processed = preprocess(df)

    assert "Survived" in df_processed.columns
    assert df_processed.isnull().sum().sum() == 0
