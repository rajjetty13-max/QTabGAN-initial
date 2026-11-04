# Demonstration outputs

These belong to the short teaching run in `../../QTabGAN_step_by_step.ipynb`, not the full research benchmark. Re-running the lesson replaces the generated files here.

| Artifact | What to inspect |
| --- | --- |
| `class_balance.png`, `split_counts.csv`, `*_rows.csv` | Imbalance and exact original CSV rows assigned to each split |
| `input_histograms.png`, `input_correlations.png` | Training inputs and feature relationships |
| `one_row_numeric_encoding.csv`, `encoding_checks.csv` | One raw-to-scaled example, category counts and unseen-category fractions |
| `quantum_probabilities.png`, `latent_angles.csv` | Random quantum inputs and their probability output |
| `training_circuit.png`, `.txt` | Actual shared circuit used by this lesson |
| `initial_quantum_gradients.png`, `*_quantum_angles.csv` | Gradient flow and quantum parameter changes |
| `learning_dashboard.png`, `training_log.csv` | BCE losses, gradients, validation utility, calibration MSE and diversity |
| `test_confusion_matrices.png`, `test_auc_curves.png` | Held-out prediction errors and ranking behavior |
| `final_test_scores.csv`, `test_predictions.csv` | Metrics and underlying per-row labels/probabilities |
| `generated_vs_real.png`, `correlation_checks.png` | Distributional collapse and lost feature relationships |
| `raw_numeric_diagnostics.csv`, `category_coverage.csv` | Numeric spread, range failures and category coverage |
| `synthetic_demo.csv` | The selected generator's rows and their requested labels (not verified label consistency) |
| `manifest.json`, `summary.json` | Configuration, versions, provenance, selected epoch and actual final numbers |

Start with **[learning_dashboard.png](learning_dashboard.png)** and **[generated_vs_real.png](generated_vs_real.png)**. The first shows changes over training; the second can expose failure even when a numerical metric appears to improve.
