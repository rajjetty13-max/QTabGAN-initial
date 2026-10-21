"""Prepare downloaded public data without guessing the paper's unreported row choices."""
import argparse
import io
from pathlib import Path
import zipfile
import pandas as pd

p = argparse.ArgumentParser(__doc__)
p.add_argument('--adult-zip', type=Path, required=True)
p.add_argument('--loan-csv', type=Path, required=True)
p.add_argument('--output', type=Path, default=Path('data'))
a = p.parse_args(); a.output.mkdir(exist_ok=True, parents=True)
columns = ['age','workclass','fnlwgt','education','education-num','marital-status',
           'occupation','relationship','race','sex','capital-gain','capital-loss',
           'hours-per-week','native-country','income']
with zipfile.ZipFile(a.adult_zip) as z:
    frames = [pd.read_csv(io.BytesIO(z.read(name)), names=columns, skipinitialspace=True,
                          comment='|') for name in ['adult.data','adult.test']]
adult = pd.concat(frames, ignore_index=True)
adult['income'] = adult['income'].str.rstrip('.')
# Preserve '?' as an explicit category, retain all 48,842 rows (paper says ~48,000).
adult.to_csv(a.output / 'adult.csv', index=False)
loan = pd.read_csv(a.loan_csv)
loan = loan.drop(columns=['ID'])  # Identifier is not a predictor.
loan.to_csv(a.output / 'loan_prepared.csv', index=False)
print('Adult:', adult.shape, 'Loan:', loan.shape)
