import numpy as np
import pandas as pd

from qtabgan import (
    QTabGAN,
    QTabGANConfig,
)

from qtabgan.preprocessing import (
    TabularPreprocessor,
)


def create_demo_dataset(
    n_samples=500
):

    rng = np.random.default_rng(42)

    age = rng.normal(
        40,
        10,
        n_samples
    )

    income = (
        age * 2000
        + rng.normal(
            0,
            10000,
            n_samples
        )
    )

    gender = rng.choice(
        ["Male", "Female"],
        n_samples
    )

    education = rng.choice(
        [
            "HighSchool",
            "Bachelor",
            "Master",
        ],
        n_samples
    )

    target = (
        income > np.median(income)
    ).astype(int)

    return pd.DataFrame(
        {
            "Age": age,
            "Income": income,
            "Gender": gender,
            "Education": education,
            "Target": target,
        }
    )


def main():

    # ---------------------------------------------------------
    # Create data
    # ---------------------------------------------------------

    df = create_demo_dataset()

    print(
        df.head()
    )

    # ---------------------------------------------------------
    # Separate target
    # ---------------------------------------------------------

    target = df["Target"]

    features = df.drop(
        columns=["Target"]
    )

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    preprocessor = TabularPreprocessor(
        numerical_columns=[
            "Age",
            "Income"
        ],
        categorical_columns=[
            "Gender",
            "Education"
        ]
    )

    X = preprocessor.fit_transform(
        features
    )

    # ---------------------------------------------------------
    # Condition
    # ---------------------------------------------------------

    condition = pd.get_dummies(
        target
    ).astype(float).values

    # ---------------------------------------------------------
    # QTabGAN
    # ---------------------------------------------------------

    config = QTabGANConfig(
        n_qubits=4,
        n_layers=2,
        epochs=10,
        batch_size=16
    )

    model = QTabGAN(
        data_dim=X.shape[1],
        condition_dim=condition.shape[1],
        config=config
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = model.fit(
        X,
        condition
    )

    # ---------------------------------------------------------
    # Generate synthetic data
    # ---------------------------------------------------------

    synthetic = model.generate(
        condition[:20]
    )

    print(
        "\nGenerated data:"
    )

    print(
        synthetic
    )

    # ---------------------------------------------------------
    # Convert back to original format
    # ---------------------------------------------------------

    synthetic_df = (
        preprocessor.inverse_transform(
            synthetic
        )
    )

    print(
        "\nSynthetic dataframe:"
    )

    print(
        synthetic_df.head()
    )


if __name__ == "__main__":
    main()