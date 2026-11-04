# Start here: understand the whole QTabGAN experiment

**Meeting soon? Start with [MEETING_GUIDE.md](MEETING_GUIDE.md)** for a short speaking summary, exact execution flow and organized diagnostic plots.

Open **[QTabGAN_step_by_step.ipynb](QTabGAN_step_by_step.ipynb)**. It already contains executed tables, numerical checks and figures. Read it top-to-bottom before changing any model settings.

The purpose is understanding, not another architecture search. This folder reuses `qtabgan/model.py`, `preprocessing.py` and the shared quantum circuit. No existing research results or training implementation were replaced.

## What to look at

1. Real data → split row IDs → training-fitted numerical/one-hot encoding.
2. Random angles → quantum circuit → a normalized vector of 256 probabilities.
3. Probability vector + class label → mapper → generated row.
4. Real/fake logits → two BCE losses → gradients → Adam training updates.
5. Generate data → fit Random Forest → choose a checkpoint using validation macro F1.
6. Freeze that checkpoint → score the test split → confusion matrix, ROC/PR and calibration MSE.
7. Inspect distributions and diversity → follow one final classifier prediction.

**The final predictor is the Random Forest. The GAN supplies its synthetic training data; it does not classify the final customer.**

## Run without installing Jupyter

From the repository root:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python simple_walkthrough/run_notebook.py
```

The runner executes the notebook's trusted code cells in order in one Python process and saves their outputs back into the notebook. It captures tables and plots, but it is not a Jupyter kernel/server and does not implement notebook magics. The lesson uses ordinary Python. It needs the already-installed project dependencies and local `data/loan_prepared.csv`; see the root README to prepare data if missing. It makes no cloud or hardware calls.

The default is a quick teaching run: 1,000 stratified Loan rows, 600/200/200 split, 5 epochs, 8 qubits, 2 original-paper layers, small classical widths and analytic simulation. Re-running replaces only this lesson's `outputs/demo/` files. Retained research experiments are untouched.

## Run interactively

Open the `.ipynb` in VS Code or Jupyter and select the project's `.venv` Python kernel. This environment did not include Jupyter/IPython kernel packages during preparation. If your editor cannot use that environment, install the optional kernel into it:

```bash
.venv/bin/python -m pip install ipykernel
```

Then restart/select the `.venv` kernel and run all cells in order. No kernel installation is required for the bundled runner above. The saved notebook was executed with that runner, not with a Jupyter kernel.

## Files

| File | Purpose |
| --- | --- |
| `QTabGAN_step_by_step.ipynb` | Main lesson, explanations, executable cells and saved outputs |
| `helpers.py` | Small RNG-isolation, classifier-scoring and diversity functions; no duplicate GAN |
| `run_notebook.py` | Minimal dependency-free notebook-cell runner |
| [PAPER_SETTINGS.md](PAPER_SETTINGS.md) | What the supplied paper specifies, our choices, reference scores and author questions |
| [TEAM_TASKS.md](TEAM_TASKS.md) | Proposed group ownership and a personal evidence log for grading |
| [outputs/demo/README.md](outputs/demo/README.md) | Map of plots, CSVs and manifests from the teaching run |

## What the MSE numbers mean

- **Brier MSE:** squared error between the classifier's positive probability and the actual 0/1 label. Lower is better. This evaluates calibration; it is not a GAN loss.
- **Feature-mean MSE:** squared differences between encoded numerical column means. Low error can hide collapse.
- We do **not** pair an arbitrary fake customer with an unrelated real customer and call their row MSE a reconstruction score. This GAN has no paired reconstruction target.

## Recorded demonstration result

The initial 5-epoch run selected epoch 1 by validation macro F1. Synthetic-trained test accuracy was **80.5%**, macro F1 **0.4460**, AUROC **0.5288**, and Brier MSE **0.22385**. The real-trained reference achieved **97.0%** accuracy. Always predicting the majority class gets **90.5%** on this test split. The generated data shows severe collapse, despite finite gradients and parameter updates.

Those are teaching-run results, **not a paper reproduction or improvement claim**. If you change the setup, use the newly generated manifest and summary rather than treating these recorded numbers as the new output. The source paper reports a different Loan split and dataset configuration; the exact author settings remain unresolved.
