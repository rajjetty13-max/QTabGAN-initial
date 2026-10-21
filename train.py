import argparse

import pandas as pd

from qtabgan import (
    QTabGAN,
    QTabGANConfig,
)

from qtabgan.preprocessing import (
    TabularPreprocessor,
)


def main():

    parser = argparse.ArgumentParser(
        description="Train QTabGAN (use benchmark.py for holdout evaluation)"
    )

    parser.add_argument(
        "--data",
        required=True,
        help="Path to CSV dataset"
    )

    parser.add_argument(
        "--target",
        required=True,
        help="Target/condition column"
    )

    parser.add_argument(
        "--numerical",
        nargs="+",
        required=True,
        help="Numerical columns"
    )

    parser.add_argument(
        "--categorical",
        nargs="*",
        default=[],
        help="Categorical columns"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30
    )

    parser.add_argument(
        "--qubits",
        type=int,
        default=8
    )

    parser.add_argument(
        "--task", choices=["classification", "regression"], default="classification",
        help="Classification uses one-hot class labels; regression uses a scalar condition"
    )

    args = parser.parse_args()

    # ---------------------------------------------------------
    # Load data
    # ---------------------------------------------------------

    df = pd.read_csv(
        args.data
    )

    print(
        "Dataset shape:",
        df.shape
    )

    # ---------------------------------------------------------
    # Preprocess
    # ---------------------------------------------------------

    preprocessor = TabularPreprocessor(
        numerical_columns=args.numerical,
        categorical_columns=args.categorical
    )

    # Target is treated as the condition.
    target = df[args.target]

    features = df.drop(
        columns=[args.target]
    )

    processed_data = (
        preprocessor.fit_transform(
            features
        )
    )

    # ---------------------------------------------------------
    # Create condition vector
    # ---------------------------------------------------------

    if args.task == "classification":

        condition = pd.get_dummies(
            target
        ).astype(float).values

    else:

        condition = (
            target
            .astype(float)
            .values
            .reshape(-1, 1)
        )

    # ---------------------------------------------------------
    # Create model
    # ---------------------------------------------------------

    config = QTabGANConfig(
        n_qubits=args.qubits,
        epochs=args.epochs
    )

    model = QTabGAN(
        data_dim=processed_data.shape[1],
        condition_dim=condition.shape[1],
        config=config
    )

    # ---------------------------------------------------------
    # Train
    # ---------------------------------------------------------

    history = model.fit(
        processed_data,
        condition
    )

    print(
        "\nTraining complete."
    )

    print(
        "Final generator loss:",
        history["generator_loss"][-1]
    )

    print(
        "Final discriminator loss:",
        history["discriminator_loss"][-1]
    )


if __name__ == "__main__":
    main()