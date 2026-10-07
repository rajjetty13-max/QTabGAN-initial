from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


class TabularPreprocessor:
    """
    Preprocessing pipeline for QTabGAN.

    Numerical features:
        1. Clip to 1st/99th percentiles
        2. Z-score normalize
        3. Min-max scale to [0, pi]
        4. Add very small Gaussian noise

    Categorical features:
        One-hot encoding
    """

    def __init__(
        self,
        numerical_columns: Optional[List[str]] = None,
        categorical_columns: Optional[List[str]] = None,
        noise_std: float = 1e-5,
    ):
        self.numerical_columns = numerical_columns or []
        self.categorical_columns = categorical_columns or []

        self.noise_std = noise_std

        self.encoder = None

        self.q1 = {}
        self.q99 = {}

        self.mean = {}
        self.std = {}

        self.min_value = {}
        self.max_value = {}

        self.output_columns = []

    def fit(self, df: pd.DataFrame):
        """
        Learn preprocessing parameters from the dataset.
        """

        # Numerical preprocessing parameters
        for column in self.numerical_columns:

            values = df[column].astype(float).values

            self.q1[column] = np.percentile(values, 1)
            self.q99[column] = np.percentile(values, 99)

            clipped = np.clip(
                values,
                self.q1[column],
                self.q99[column]
            )

            self.mean[column] = np.mean(clipped)

            std = np.std(clipped)

            if std == 0:
                std = 1.0

            self.std[column] = std

            z = (clipped - self.mean[column]) / self.std[column]

            self.min_value[column] = np.min(z)
            self.max_value[column] = np.max(z)

        # Categorical preprocessing
        if len(self.categorical_columns) > 0:

            self.encoder = OneHotEncoder(
                sparse_output=False,
                handle_unknown="ignore"
            )

            self.encoder.fit(
                df[self.categorical_columns]
            )

            encoded_names = self.encoder.get_feature_names_out(
                self.categorical_columns
            )

            self.output_columns.extend(encoded_names.tolist())

        self.output_columns = (
            self.numerical_columns
            + self.output_columns
        )

        return self

    def transform(self, df: pd.DataFrame):
        """
        Transform a dataframe into the representation
        used by QTabGAN.
        """

        parts = []

        # Numerical features
        for column in self.numerical_columns:

            values = df[column].astype(float).values

            clipped = np.clip(
                values,
                self.q1[column],
                self.q99[column]
            )

            z = (
                clipped - self.mean[column]
            ) / self.std[column]

            minimum = self.min_value[column]
            maximum = self.max_value[column]

            if maximum - minimum == 0:
                scaled = np.zeros_like(z)
            else:
                scaled = (
                    (z - minimum)
                    / (maximum - minimum)
                ) * np.pi

            noise = np.random.normal(
                0,
                self.noise_std,
                size=scaled.shape
            )

            scaled = scaled + noise

            parts.append(
                scaled.reshape(-1, 1)
            )

        # Categorical features
        if (
            self.encoder is not None
            and len(self.categorical_columns) > 0
        ):

            categorical = self.encoder.transform(
                df[self.categorical_columns]
            )

            parts.append(categorical)

        if len(parts) == 0:
            return np.empty((len(df), 0))

        return np.concatenate(parts, axis=1)

    def fit_transform(self, df: pd.DataFrame):
        """
        Fit the preprocessor and transform the dataset.
        """

        self.fit(df)

        return self.transform(df)

    def inverse_transform(
        self,
        array: np.ndarray
    ):
        """
        Convert generated data back into a dataframe.
        """

        result = {}

        index = 0

        # Numerical features
        for column in self.numerical_columns:

            value = array[:, index]

            minimum = self.min_value[column]
            maximum = self.max_value[column]

            if maximum - minimum == 0:
                z = np.full_like(
                    value,
                    minimum
                )
            else:
                z = (
                    value / np.pi
                ) * (
                    maximum - minimum
                ) + minimum

            original = (
                z * self.std[column]
            ) + self.mean[column]

            result[column] = original

            index += 1

        # Categorical features
        if (
            self.encoder is not None
            and len(self.categorical_columns) > 0
        ):

            encoded_size = len(
                self.encoder.get_feature_names_out()
            )

            categorical_array = array[
                :,
                index:index + encoded_size
            ]

            decoded = self.encoder.inverse_transform(
                categorical_array
            )

            for i, column in enumerate(
                self.categorical_columns
            ):
                result[column] = decoded[:, i]

        return pd.DataFrame(result)