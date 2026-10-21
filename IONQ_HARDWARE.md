# Running a trained QTabGAN generator on IonQ

This workflow targets **IonQ Quantum Cloud directly**. It loads a saved QTabGAN checkpoint, runs its quantum circuits on the selected IonQ backend, and turns the returned probabilities into synthetic CSV records using the frozen classical mapper.

It supports checkpoint generation. It does not train parameters on hardware. No cloud jobs were submitted while developing or testing this adapter.

## What runs where?

```text
Your computer                           IonQ Quantum Cloud

Load trained circuit angles
Choose fixed random inputs
Export H / RY / RZ / CNOT circuits ----> Compile and execute circuits
                                        Measure outcomes
Receive measurement probabilities <---- Return probability distributions
Correct the output ordering
Append requested class labels
Run the saved classical mapper
Decode and save synthetic CSV
```

The mapper and preprocessing stay on your computer. The submitted payload contains the circuit, angles, backend, shot count, and job identifiers; it does not upload the original dataset or the class labels.

The adapter exports the existing PennyLane gate definition, including the saved Hadamard-placement option. It does not replace the trained circuit with a new architecture. QIS rotation angles are sent in radians; IonQ compiles the QIS circuit to its hardware gate set.

## What you need

- The repository’s Python environment and dependencies, including `requests`.
- A completed CSV-backed classification benchmark checkpoint and its fitted preprocessor.
- An IonQ Quantum Cloud API key with access to the desired backend.

The existing Loan run can be used:

```text
results/loan_100/manifest.json
results/loan_100/seed_42/qtabgan/model.pt
results/loan_100/seed_42/preprocessor.joblib
```

Checkpoint and preprocessing files must be trusted local artifacts. The preprocessor is a serialized Python object.

The implementation and tests use Linux/POSIX file locking. Run commands below from the repository root. If using another Python environment, replace `.venv/bin/python` with that environment’s Python command.

## 1. Validate a small bundle locally

A **bundle** is a directory holding a frozen copy of the checkpoint, preprocessing, random inputs, class choices, IonQ requests, and execution journal. Keeping those together makes later collection independent of changes to the original model files.

Prepare four circuits for the cloud simulator:

```bash
.venv/bin/python scripts/ionq_generate.py prepare \
  --run results/loan_100 \
  --rows 4 \
  --shots 1024 \
  --backend simulator \
  --output hardware_runs/loan_simulator_trial
```

This command is completely offline. It also replays the exported JSON locally and checks that its ideal probabilities match the checkpoint.

Then test generation from the exact exported circuits:

```bash
.venv/bin/python scripts/ionq_generate.py local hardware_runs/loan_simulator_trial
```

This writes:

- `local_ideal_synthetic.csv`: generated records using exact local probabilities.
- `local_shots_synthetic.csv`: generated records using locally sampled finite-shot probabilities.
- Corresponding probability arrays and diagnostics measuring changes from the ideal reference.

The bundle prepared during development is `hardware_runs/loan_simulator`; the example above uses a fresh directory. Preparation refuses to overwrite an existing bundle. Choose a new directory for a new experiment, and preserve journals for jobs already submitted.

By default, requested classes cycle through the saved label order. For a binary problem, four rows request classes `0, 1, 0, 1`. These are balanced pilot inputs, not an attempt to reproduce the original class frequencies. To choose labels explicitly, add, for example:

```bash
--class-indices 0 0 1 1
```

Supply exactly one class index per requested row. The actual label names and their order appear in `bundle.json`.

## 2. Set your API key and discover the backend

For Bash, enter the key without displaying it or putting its value in a command:

```bash
read -rs -p "IonQ API key: " IONQ_API_KEY
export IONQ_API_KEY
printf '\n'
```

The client reads `IONQ_API_KEY` from the environment. It does not read `.env` files automatically and does not write the key to bundles or logs.

List the backends visible to your key:

```bash
.venv/bin/python scripts/ionq_generate.py backends
```

Use the exact backend identifier available to your account. A `qpu.*` backend executes on hardware; `simulator` executes in IonQ’s cloud simulator. The code does not hard-code a particular QPU as available to your account.

You can first submit the simulator bundle with the commands in step 4. The ideal cloud simulator can ignore the requested shot count and return ideal probabilities; `local_shots` explicitly samples finite-shot probabilities for the local comparison.

## 3. Prepare your QPU bundle

Replace `qpu.YOUR_BACKEND` below with the exact identifier from your account:

```bash
.venv/bin/python scripts/ionq_generate.py prepare \
  --run results/loan_100 \
  --rows 4 \
  --shots 1024 \
  --backend qpu.YOUR_BACKEND \
  --output hardware_runs/loan_qpu_trial

.venv/bin/python scripts/ionq_generate.py local hardware_runs/loan_qpu_trial
```

Open `hardware_runs/loan_qpu_trial/bundle.json`. It contains the exact requests that will be sent, the requested backend, and the shot count.

Each generated record uses a separate circuit and job. Four records at 1,024 shots request 4,096 circuit-shots in total. That is a requested execution count, not a dollar estimate or a guarantee about billable work. Increasing `--rows` increases the number of submitted jobs.

Debiasing is disabled by default so the setting is explicit. To request it, prepare a new bundle with `--debias` and at least 500 shots. Availability and behavior depend on the selected backend; this option has not been live-validated here.

## 4. Submit the prepared requests

This is the command that sends jobs to IonQ:

```bash
.venv/bin/python scripts/ionq_generate.py submit hardware_runs/loan_qpu_trial
```

It prints each accepted job ID and saves it immediately in `state.json`.

Rerunning `submit` skips rows which already have job IDs. It does not intentionally submit those rows again. A lock prevents two commands from operating on the same bundle concurrently.

The adapter never automatically retries a POST request. If the connection drops after IonQ accepted a job, repeating that POST could otherwise create an extra job.

## 5. Collect the results

Check the saved jobs:

```bash
.venv/bin/python scripts/ionq_generate.py collect hardware_runs/loan_qpu_trial
```

This performs one status check for each unfinished job. If jobs are still queued or running, rerun the same command later. Collection never submits jobs.

Once every row is complete, it writes:

```text
ionq_synthetic.csv
ionq_probabilities.npz
ionq_diagnostics.json
row_0000_job.json
row_0000_result.json
...
```

The CSV contains records decoded by the saved preprocessor, with the requested labels appended. The diagnostics compare the returned quantum probabilities and mapper outputs with the same inputs’ local ideal reference.

Failed or canceled jobs are recorded and reported. The code does not substitute local simulation data for missing hardware results and does not write a complete table while any row is unfinished.

## Recovering a submission after a lost response

The journal records `submitting` before making each POST. If the response is lost, the code refuses to automatically submit that row again.

Find the matching uniquely named job in IonQ Quantum Cloud. Then attach its job ID, using the row number from the journal:

```bash
.venv/bin/python scripts/ionq_generate.py attach hardware_runs/loan_qpu_trial \
  --row 0 \
  --job-id YOUR_JOB_UUID
```

The adapter checks that the retrieved job’s name and backend match that bundle row. You can then rerun `submit` to submit any remaining pending rows, followed by `collect`.

If the API rejected the request and no job exists, resolve the rejection before preparing a corrected bundle. Preserve the original journal and check the cloud account before creating replacement submissions. Do not erase an uncertain submission’s journal just to make the command run again.

## Output ordering and result formats

The mapper expects wire 0 to be the most significant bit of its probability-vector index. IonQ’s flat QIS result keys are decimal integers with qubit `i` stored at bit `2**i`.

For example, on three qubits, IonQ outcome `1` means only qubit 0 is 1. That corresponds to the mapper’s `|100>` column, index 4. The adapter performs this conversion and fills unreported outcomes with zero probability.

The client uses a v1 probability artifact when supplied, or the documented QIS `/results/probabilities` endpoint. It accepts flat decimal-key probability objects. It rejects arbitrary register-nested formats, malformed probabilities, and counts supplied in place of probabilities rather than guessing how to interpret them.

## What has been tested?

The local test suite covers:

- Exported circuit probabilities and directed CNOT order, with both Hadamard options.
- Asymmetric basis states that expose bit-order mistakes.
- Sparse probability decoding and malformed results.
- Submission journaling, resuming, and lost-response recovery.
- Pending and failed jobs without simulated-data fallback.
- Bundle and snapshot integrity checks.
- HTTP request format and credential placement.
- A complete prepare → submit → collect → mapper → CSV round trip using a mocked IonQ service.

Run the suite with:

```bash
MPLCONFIGDIR=/tmp/qtabgan-mpl .venv/bin/python -m unittest discover -s tests -v
```

The existing trained Loan checkpoint was also exported and tested locally in a four-row bundle. **Authenticated simulator access, actual QPU execution, and device-native compilation have not been tested here.** Those require your account and selected backend.

## What this does not change

Running a checkpoint on hardware does not repair its learned data distribution. The current 100-epoch QTabGAN checkpoints showed mode collapse in the previous benchmarks. This workflow makes those checkpoints executable on IonQ and lets you measure hardware effects; it does not establish paper reproduction or useful synthetic-data quality.

This adapter covers direct IonQ Quantum Cloud checkpoint inference. AWS Braket, Azure Quantum, hardware training, multicircuit batching, and hardware-noise retraining are separate extensions.

## Source files and API references

- [qtabgan/ionq.py](qtabgan/ionq.py): circuit export, probability conversion, HTTP client, and job journal.
- [scripts/ionq_generate.py](scripts/ionq_generate.py): bundle preparation, local validation, submission/collection commands, and mapper inference.
- [tests/test_ionq.py](tests/test_ionq.py) and [tests/test_ionq_workflow.py](tests/test_ionq_workflow.py): offline verification.
- [IonQ direct API guide](https://docs.ionq.com/guides/direct-api-submission): request format, QIS results, and little-endian output convention.
- [Create job API](https://docs.ionq.com/api-reference/v0.4/jobs/create-job): backend, shots, and submission options.
- [Result formats](https://docs.ionq.com/api-reference/v0.4/schemas/results-formats) and [artifact retrieval](https://docs.ionq.com/api-reference/v0.4/jobs/get-job-artifact): result contracts.
- [QIS gate representation](https://docs.ionq.com/api-reference/v0.3/writing-quantum-programs): rotation units and supported gate names, retained in the v0.4 QIS input used here.
