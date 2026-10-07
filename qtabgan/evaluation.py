import numpy as np

from scipy.spatial.distance import jensenshannon

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
)

from sklearn.metrics import (
    accuracy_score,
    mean_squared_error,
)


def mean_squared_error_metric(
    real,
    synthetic
):
    """
    Mean squared error between two arrays.
    """

    return mean_squared_error(
        real,
        synthetic
    )


def average_categorical_jsd(
    real,
    synthetic,
    categorical_indices
):
    """
    Average Jensen-Shannon divergence
    across categorical columns.
    """

    scores = []

    for index in categorical_indices:

        real_values = real[:, index]
        synthetic_values = synthetic[:, index]

        categories = np.unique(
            np.concatenate(
                [
                    real_values,
                    synthetic_values
                ]
            )
        )

        real_distribution = np.array(
            [
                np.mean(
                    real_values == category
                )
                for category in categories
            ]
        )

        synthetic_distribution = np.array(
            [
                np.mean(
                    synthetic_values == category
                )
                for category in categories
            ]
        )

        score = (
            jensenshannon(
                real_distribution,
                synthetic_distribution
            ) ** 2
        )

        scores.append(score)

    if len(scores) == 0:
        return 0.0

    return float(
        np.mean(scores)
    )


def classification_utility(
    train_real,
    train_synthetic,
    test_real,
    test_labels
):
    """
    Train classifiers using real and synthetic
    training data and evaluate on real test data.
    """

    model_real = RandomForestClassifier(
        random_state=42
    )

    model_synthetic = RandomForestClassifier(
        random_state=42
    )

    model_real.fit(
        train_real,
        test_labels[:len(train_real)]
    )

    model_synthetic.fit(
        train_synthetic,
        test_labels[:len(train_synthetic)]
    )

    predictions_real = model_real.predict(
        test_real
    )

    predictions_synthetic = (
        model_synthetic.predict(
            test_real
        )
    )

    return {
        "real_train_accuracy": accuracy_score(
            test_labels,
            predictions_real
        ),
        "synthetic_train_accuracy": accuracy_score(
            test_labels,
            predictions_synthetic
        )
    }


def regression_utility(
    train_real,
    train_synthetic,
    test_real,
    test_labels
):
    """
    Compare downstream regression performance.
    """

    model_real = RandomForestRegressor(
        random_state=42
    )

    model_synthetic = RandomForestRegressor(
        random_state=42
    )

    model_real.fit(
        train_real,
        test_labels[:len(train_real)]
    )

    model_synthetic.fit(
        train_synthetic,
        test_labels[:len(train_synthetic)]
    )

    predictions_real = model_real.predict(
        test_real
    )

    predictions_synthetic = (
        model_synthetic.predict(
            test_real
        )
    )

    return {
        "real_train_mse": mean_squared_error(
            test_labels,
            predictions_real
        ),
        "synthetic_train_mse": mean_squared_error(
            test_labels,
            predictions_synthetic
        )
    }