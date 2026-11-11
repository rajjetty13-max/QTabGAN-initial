"""Explain QTabGAN's mechanics and print explicit diagnostics, end to end.

Built for THIS repo (current main / ojas-changes, incl. the Nov 2026 model
improvements). It uses only existing APIs:
  - qtabgan.QTabGAN / QTabGANConfig          (model.py, config.py)
  - qtabgan.preprocessing.TabularPreprocessor (preprocessing.py)
  - qtabgan.evaluation helpers                (evaluation.py)

Default run = the original paper-style model (linear mapper head, feature-only
discriminator, gain 1). The improvement flags mirror extended_benchmark.py:
    --ansatz paper_readout --condition-discriminator --quantum-init-gain 4
    --tabular-output --moment-loss-weight 10

Short forms (presets fill in the dataset columns):
    python scripts/explain_and_diagnose.py --dataset adult --epochs 10
    python scripts/explain_and_diagnose.py --dataset adult --improved --epochs 10
    python scripts/explain_and_diagnose.py --dataset loan --improved --epochs 10

Or spell everything out, e.g.:
    python scripts/explain_and_diagnose.py --data data/adult.csv \
        --target income \
        --numerical age fnlwgt education-num capital-gain capital-loss hours-per-week \
        --categorical workclass education marital-status occupation relationship \
                      race sex native-country \
        --epochs 10

Sections: [1] data  [2] mechanics (circuit + shape trace)  [3] quantum
diagnostics  [4] training (live, per epoch)  [5] generated-data fidelity
[6] ML utility. Classification only, like benchmark.py.
"""
import argparse
import sys
from pathlib import Path

# Allow `python scripts/explain_and_diagnose.py` from the repo root
# (same fix as scripts/diagnose_tables.py).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import pennylane as qml
import torch
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split

from qtabgan import QTabGAN, QTabGANConfig
from qtabgan.preprocessing import TabularPreprocessor
from qtabgan.evaluation import (
    average_categorical_jsd,
    classification_utility,
    correlation_difference,
)


def banner(n, title):
    print("\n" + "=" * 72)
    print(f"[{n}] {title}")
    print("=" * 72)


GATE_ORDER = {
    "paper": (
        "  H on every qubit -> RY(noise) per qubit   [noise encodes the sample]\n"
        "  then per layer: RY(weights) -> RZ(weights) -> ring CNOT(i, i+1 mod n)\n"
        "  probabilities are read out in the Z basis"
    ),
    "paper_readout": (
        "  as 'paper' (H -> RY(noise) -> per layer RY/RZ + ring CNOT),\n"
        "  then a final H on every qubit: X-basis readout, so the final\n"
        "  layer's RZ phases become visible in the probabilities"
    ),
    "reuploading": (
        "  H on every qubit, then per layer: RY(noise) re-uploaded ->\n"
        "  RZ(weights[..., 0]) -> RY(weights[..., 1]) -> CNOT matchings\n"
        "  (two disjoint pairings, direction alternates per layer)"
    ),
}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=["adult", "loan"], default=None,
                   help="Preset that fills in --data/--target/--numerical/"
                        "--categorical (explicit flags still win)")
    p.add_argument("--data", default=None)
    p.add_argument("--target", default=None)
    p.add_argument("--numerical", nargs="+", default=None)
    p.add_argument("--categorical", nargs="*", default=None)
    p.add_argument("--epochs", type=int, default=10,
                   help="Short diagnostic run, NOT a benchmark run")
    p.add_argument("--qubits", type=int, default=8)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    # --- architecture / improvement flags (mirror extended_benchmark.py) ---
    p.add_argument("--ansatz", choices=["paper", "reuploading", "paper_readout"],
                   default="paper")
    p.add_argument("--condition-discriminator", action="store_true",
                   help="Discriminator sees features + one-hot label: D(x, y)")
    p.add_argument("--classical-init-seed", type=int, default=None)
    p.add_argument("--quantum-init-gain", type=float, default=1.0,
                   help="Amplify quantum variation in the mapper's first layer "
                        "around its calibrated mean (1.0 = off)")
    p.add_argument("--tabular-output", action="store_true",
                   help="Mixed-type head: pi*sigmoid numerics + straight-through "
                        "Gumbel-softmax one-hot categoricals")
    p.add_argument("--categorical-temperature", type=float, default=0.5)
    p.add_argument("--moment-loss-weight", type=float, default=0.0,
                   help="Weight of the train-only conditional moment penalty "
                        "added to the generator loss (0 = off)")
    p.add_argument("--improved", action="store_true",
                   help="Preset for the TABULAR_OUTPUT.md configuration: "
                        "paper_readout + D(x, y) + init gain 4 + tabular output "
                        "+ moment loss 10 + batch size 64")
    p.add_argument("--skip-utility", action="store_true",
                   help="Skip the RandomForest ML-utility section (slowest part)")
    args = p.parse_args()

    # Fill any unset data arguments from the --dataset preset.
    PRESETS = {
        "adult": dict(
            data="data/adult.csv", target="income",
            numerical=["age", "fnlwgt", "education-num", "capital-gain",
                       "capital-loss", "hours-per-week"],
            categorical=["workclass", "education", "marital-status", "occupation",
                         "relationship", "race", "sex", "native-country"]),
        "loan": dict(
            data="data/loan_prepared.csv", target="PersonalLoan",
            numerical=["Age", "Experience", "Income", "Family", "CCAvg", "Mortgage"],
            categorical=["ZIP Code", "Education", "SecuritiesAccount",
                         "CDAccount", "Online", "CreditCard"]),
    }
    if args.dataset:
        preset = PRESETS[args.dataset]
        args.data = args.data or preset["data"]
        args.target = args.target or preset["target"]
        args.numerical = args.numerical or preset["numerical"]
        if args.categorical is None:
            args.categorical = preset["categorical"]
    if not (args.data and args.target and args.numerical):
        p.error("--data, --target and --numerical are required "
                "(or pass --dataset adult|loan)")
    if args.categorical is None:
        args.categorical = []
    if args.improved:  # one flag instead of six
        args.ansatz = "paper_readout"
        args.condition_discriminator = True
        args.quantum_init_gain = 4.0
        args.tabular_output = True
        args.moment_loss_weight = 10.0
        if args.batch_size == 32:  # untouched default -> use the study's 64
            args.batch_size = 64

    rng = np.random.default_rng(args.seed)

    # ---------------------------------------------------------------- [1] DATA
    banner(1, "DATA — what goes in")
    df = pd.read_csv(args.data)
    print(f"rows x columns: {df.shape}")
    print(f"missing values total: {int(df.isna().sum().sum())}")
    print("class balance (target = condition):")
    print(df[args.target].value_counts(normalize=True).rename("fraction"))

    # Hold out a test set exactly so section [6] is an honest utility check.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=args.seed, stratify=df[args.target])
    print(f"train rows: {len(train_df)}   test rows: {len(test_df)} (80:20)")

    pre = TabularPreprocessor(numerical_columns=args.numerical,
                              categorical_columns=args.categorical)
    # Fit on TRAIN only, so preprocessing statistics never see the test set.
    X_train = pre.fit_transform(train_df.drop(columns=[args.target]))
    X_test = pre.transform(test_df.drop(columns=[args.target]))

    classes = sorted(df[args.target].astype(str).unique())
    to_int = {c: i for i, c in enumerate(classes)}
    y_train = train_df[args.target].astype(str).map(to_int).values
    y_test = test_df[args.target].astype(str).map(to_int).values
    condition = np.eye(len(classes))[y_train]  # one-hot label = the condition y

    print(f"\nencoded feature width: {X_train.shape[1]} columns "
          f"({len(args.numerical)} numeric + one-hot categoricals)")
    # Paper preprocessing check (preprocessing.py docstring): numerics are
    # clipped 1/99 pct -> z-scored -> min-maxed to [0, pi] -> +N(0, 1e-5) noise.
    for j, col in enumerate(args.numerical):
        print(f"  numeric {col:16s} encoded range "
              f"[{X_train[:, j].min():.3f}, {X_train[:, j].max():.3f}]  (expect ~[0, pi])")
    if pre.encoder is not None:  # every one-hot group must sum to exactly 1 per row
        pos = len(args.numerical)
        for col, cats in zip(args.categorical, pre.encoder.categories_):
            block = X_train[:, pos:pos + len(cats)]
            print(f"  one-hot {col:16s} {len(cats)} categories, "
                  f"row sums all == 1: {bool(np.allclose(block.sum(1), 1))}")
            pos += len(cats)

    # Config is built AFTER preprocessing: the tabular head / moment loss need
    # the output schema (numeric width + per-feature category counts), derived
    # from the fitted preprocessor exactly as extended_benchmark.py does.
    use_schema = args.tabular_output or args.moment_loss_weight > 0
    cfg = QTabGANConfig(
        n_qubits=args.qubits, n_layers=args.layers, epochs=args.epochs,
        batch_size=args.batch_size, seed=args.seed, ansatz=args.ansatz,
        condition_discriminator=args.condition_discriminator,
        classical_init_seed=args.classical_init_seed,
        quantum_init_gain=args.quantum_init_gain,
        tabular_output=args.tabular_output,
        numerical_dim=len(args.numerical) if use_schema else None,
        categorical_dims=(tuple(len(c) for c in pre.encoder.categories_)
                          if use_schema and pre.encoder is not None else ()),
        categorical_temperature=args.categorical_temperature,
        moment_loss_weight=args.moment_loss_weight)

    # ----------------------------------------------------------- [2] MECHANICS
    banner(2, "MECHANICS — what the model actually is")
    model = QTabGAN(data_dim=X_train.shape[1], condition_dim=len(classes), config=cfg)
    q = model.quantum_generator
    n_q_params = q.weights.numel()
    n_map = sum(t.numel() for t in model.mapper.parameters())
    n_disc = sum(t.numel() for t in model.discriminator.parameters())
    print(f"config: {cfg}")
    print(f"\nquantum weights shape (layers, qubits, [RY, RZ]): {tuple(q.weights.shape)}"
          f"  -> {n_q_params} quantum parameters")
    print(f"mapper parameters: {n_map:,}   discriminator parameters: {n_disc:,}")
    print(f"probability vector width: 2^{cfg.n_qubits} = {2 ** cfg.n_qubits} outcomes")
    print(f"discriminator input: {X_train.shape[1]} encoded features"
          + (f" + {len(classes)} one-hot label columns  -> D(x, y)"
             if cfg.condition_discriminator else
             " only (feature-only D(x); the label is NOT shown to D)"))
    if cfg.quantum_init_gain != 1.0:
        print(f"quantum init gain: {cfg.quantum_init_gain} — the mapper's first layer "
              f"was re-centred on the mean probability vector and its quantum block "
              f"scaled x{cfg.quantum_init_gain} at init "
              f"(calibration mean stored: {model.quantum_initialization_mean is not None})")
    if cfg.tabular_output:
        print(f"tabular output head: numerics -> pi*sigmoid (bounded to (0, pi)); "
              f"each categorical group -> straight-through Gumbel-softmax "
              f"(tau={cfg.categorical_temperature}), exactly one-hot in forward pass")
    if cfg.moment_loss_weight > 0:
        print(f"moment loss: generator loss = adversarial + "
              f"{cfg.moment_loss_weight} x conditional moment penalty "
              f"(targets from the TRAIN split only)")
    print(f"\ngate order in quantum_generator.operations() (ansatz='{cfg.ansatz}'):")
    print(GATE_ORDER[cfg.ansatz])
    print("\nthe circuit, drawn by PennyLane for one noise input:")
    noise_row = torch.rand(cfg.n_qubits) * np.pi
    print(qml.draw(q.circuit)(noise_row, q.weights))

    print("\nshape trace of ONE forward pass (batch of 4):")
    noise = torch.rand(4, cfg.n_qubits) * np.pi
    probs = q(noise)                                   # (4, 256)
    cond4 = condition[:4].astype(np.float32)
    with torch.no_grad():
        raw4 = model.mapper(probs, torch.tensor(cond4))      # linear head output
        gen4 = model.generate(cond4)                         # full generator output
        logits = model.discriminate(torch.tensor(gen4), torch.tensor(cond4))
    print(f"  noise            {tuple(noise.shape)}   angles in [0, pi)")
    print(f"  -> probs         {tuple(probs.shape)}   each row sums to {probs[0].sum():.6f}")
    print(f"  -> + condition   {cond4.shape}   one-hot class label")
    print(f"  -> mapper output {tuple(raw4.shape)}   synthetic encoded rows (linear head)")
    if cfg.tabular_output:
        print(f"  -> generator out {gen4.shape}   after the tabular adapter "
              f"(one-hot categoricals, numerics in (0, pi))")
    d_label = "D(x, y) logit" if cfg.condition_discriminator else "D(x) logit   "
    print(f"  -> {d_label} {tuple(logits.shape)}   >0 means 'looks real'")

    # ----------------------------------------------------- [3] QUANTUM CHECKS
    banner(3, "QUANTUM DIAGNOSTICS — is the circuit doing anything?")
    with torch.no_grad():
        noise = torch.rand(512, cfg.n_qubits) * np.pi
        probs = q(noise).numpy()
    entropy = -(probs * np.log(probs + 1e-12)).sum(1)
    print(f"probability rows sum to 1: {bool(np.allclose(probs.sum(1), 1))}")
    print(f"entropy of p(noise): mean {entropy.mean():.3f} nats "
          f"(uniform over {2 ** cfg.n_qubits} outcomes = {np.log(2 ** cfg.n_qubits):.3f})")
    print(f"effective outcomes exp(entropy): {np.exp(entropy).mean():.1f} "
          f"of {2 ** cfg.n_qubits}  <- small = distribution collapsed onto few outcomes")
    print(f"outcomes with mean p < 1e-4 ('dead'): {(probs.mean(0) < 1e-4).mean():.1%}")
    print(f"variation across different noise inputs: mean per-outcome std "
          f"{probs.std(0).mean():.4f}  <- ~0 would mean the noise input is ignored")
    if cfg.quantum_init_gain != 1.0:
        print(f"note: init gain x{cfg.quantum_init_gain} amplifies exactly this "
              f"variation inside the mapper; the probabilities themselves are unchanged")
    with torch.no_grad():  # analytic circuit is deterministic: same noise, same p
        again = q(noise[:8]).numpy()
    print(f"same noise twice -> identical probs: "
          f"{bool(np.allclose(again, probs[:8]))} (analytic, no shots)")
    with torch.no_grad():  # the sampler's finite-shot version of the same p
        sampled = q.sampled_probabilities(noise[:64]).numpy()
    print(f"shot-sampled p ({cfg.shots} shots) vs exact p: mean |diff| "
          f"{np.abs(sampled - probs[:64]).mean():.4f}  <- this gap is pure shot noise")

    # ------------------------------------------------------------ [4] TRAINING
    banner(4, f"TRAINING — {args.epochs} epochs, live (diagnostic run, not a benchmark)")

    def on_epoch(epoch, values):  # QTabGAN.fit calls this after every epoch
        line = (f"  epoch {epoch:3d}  G loss {values['generator_loss']:.4f}  "
                f"D loss {values['discriminator_loss']:.4f}  "
                f"|quantum grad| {values['quantum_gradient_norm']:.2e}")
        if "moment_loss" in values:  # present only when the penalty is enabled
            line += (f"  (adversarial {values['adversarial_loss']:.4f} + "
                     f"{cfg.moment_loss_weight:g} x moment "
                     f"{values['moment_loss']:.4f})")
        print(line, flush=True)

    history = model.fit(X_train, condition, verbose=False, epoch_callback=on_epoch)
    g, d, qg = (np.array(history[k]) for k in
                ("generator_loss", "discriminator_loss", "quantum_gradient_norm"))
    # With the moment penalty on, generator_loss mixes two objectives; the
    # adversarial component alone is the one comparable to the 0.69 balance mark.
    adv = np.array(history.get("adversarial_loss", history["generator_loss"]))
    k = max(1, len(g) // 3)
    print(f"\nfirst-third means:  G {g[:k].mean():.4f}   D {d[:k].mean():.4f}")
    print(f"last-third means:   G {g[-k:].mean():.4f}   D {d[-k:].mean():.4f}")
    if "moment_loss" in history:
        mom = np.array(history["moment_loss"])
        print(f"moment penalty:     first third {mom[:k].mean():.4f}   "
              f"last third {mom[-k:].mean():.4f}  <- should fall as the generator "
              f"matches per-class means/stds/marginals")
    print(f"quantum grad norm:  mean {qg.mean():.2e}  max {qg.max():.2e}  "
          f"last {qg[-1]:.2e}  <- 0 or NaN here means the circuit is not learning")
    print("how to read this: adversarial G and D losses near 0.69 (= ln 2) = balanced game.")
    if d[-k:].mean() < 0.35 and adv[-1] > adv[0]:
        print("  -> discriminator is winning and the generator is falling behind.")
    elif adv[-k:].mean() < 0.35:
        print("  -> discriminator is barely learning; check its LR / capacity.")
    else:
        print("  -> no runaway winner in this window; check fidelity below, not just losses.")

    # ------------------------------------------------- [5] GENERATED-DATA CHECKS
    banner(5, "GENERATED DATA — does synthetic look like real?")
    shuffled = condition[rng.permutation(len(condition))]  # keep real class balance
    synth_labels = shuffled.argmax(1)
    fake = model.generate(shuffled)  # encoded space, same row count as train

    cols = pre.output_columns
    mean_gap = np.abs(fake.mean(0) - X_train.mean(0))
    std_ratio = fake.std(0) / (X_train.std(0) + 1e-12)
    print(f"encoded columns: mean |mean gap| {mean_gap.mean():.4f}, "
          f"mean std ratio {std_ratio.mean():.3f} (1.0 = same spread as real)")
    print("worst 5 encoded columns by mean gap:")
    for j in np.argsort(-mean_gap)[:5]:
        print(f"  {cols[j]:28s} real mean {X_train[:, j].mean():.3f}  "
              f"synth mean {fake[:, j].mean():.3f}  std ratio {std_ratio[j]:.3f}")

    if args.categorical and pre.encoder is not None:
        # Support check, the failure TABULAR_OUTPUT.md was written about: the
        # linear head emits raw scores, so synthetic 'one-hot' coordinates can
        # be negative or > 1 and a row need not pick exactly one category.
        pos = len(args.numerical)
        neg, over, onehot = [], [], []
        for cats in pre.encoder.categories_:
            block = fake[:, pos:pos + len(cats)]
            neg.append((block < 0).mean())
            over.append((block > 1).mean())
            onehot.append((np.isclose(block.sum(1), 1)
                           & (block.max(1) == 1) & (block.min(1) == 0)).mean())
            pos += len(cats)
        print(f"\ncategorical support in encoded synthetic rows:")
        print(f"  coordinates < 0: {np.mean(neg):.1%}   coordinates > 1: "
              f"{np.mean(over):.1%}   (real data: 0% / 0% by construction)")
        print(f"  rows that are exactly one-hot per group: {np.mean(onehot):.1%}"
              + ("  <- tabular head guarantees this in the forward pass"
                 if cfg.tabular_output else
                 "  <- the linear head does NOT guarantee this; the rest is "
                 "only fixed up when rows are decoded"))

    real_dec = pre.inverse_transform(X_train)   # round-trip, original units
    fake_dec = pre.inverse_transform(fake)
    if args.numerical:
        print("\nnumeric, original units (KS near 0 = same distribution):")
        for col in args.numerical:
            ks = ks_2samp(real_dec[col], fake_dec[col]).statistic
            print(f"  {col:16s} KS {ks:.3f}   real mean {real_dec[col].mean():,.1f}  "
                  f"synth mean {fake_dec[col].mean():,.1f}")
        # The studies' headline diversity metric: per-class spread of the
        # synthetic numerics vs the real per-class spread (1.0 = matched).
        ratios = []
        for c in range(len(classes)):
            for col in args.numerical:
                r = real_dec.loc[y_train == c, col].std()
                s = fake_dec.loc[synth_labels == c, col].std()
                if r and np.isfinite(r) and r > 0:
                    ratios.append(s / r)
        if ratios:
            print(f"numeric within-class std ratio: mean {np.mean(ratios):.3f} "
                  f"(1.0 = same per-class spread as real; near 0 = collapsed "
                  f"to the class mean)")
        cd = correlation_difference(real_dec[args.numerical].to_numpy(),
                                    fake_dec[args.numerical].to_numpy())
        print(f"correlation difference (paper metric, lower = better): {cd}")
    if args.categorical:
        cover = []
        for col in args.categorical:
            real_cats = set(train_df[col].astype(str).unique())
            fake_cats = set(fake_dec[col].astype(str).unique())
            cover.append(len(fake_cats & real_cats) / max(1, len(real_cats)))
        print(f"category coverage (decoded): mean {np.mean(cover):.1%} of real "
              f"categories appear in synthetic data")
        jsd = average_categorical_jsd(
            train_df[args.categorical].astype(str).to_numpy(),
            fake_dec[args.categorical].astype(str).to_numpy(),
            range(len(args.categorical)))
        print(f"average categorical JSD (paper metric, lower = better): {jsd:.4f}")

    # ------------------------------------------------------------ [6] ML UTILITY
    if not args.skip_utility:
        banner(6, "ML UTILITY — train on synthetic, test on real (repo's own helper)")
        util = classification_utility(
            X_train, fake, X_test, y_test,
            real_labels=y_train, synthetic_labels=synth_labels, seed=cfg.seed)
        ra, sa = util["real_train_accuracy"], util["synthetic_train_accuracy"]
        rf, sf = util["real_train_f1_macro"], util["synthetic_train_f1_macro"]
        print(f"RandomForest trained on REAL data:      accuracy {ra:.4f}  F1-macro {rf:.4f}")
        print(f"RandomForest trained on SYNTHETIC data: accuracy {sa:.4f}  F1-macro {sf:.4f}")
        print(f"gaps: accuracy {abs(ra - sa):.4f}   F1 {abs(rf - sf):.4f}   "
              f"(paper Table 3, Adult: 0.0020 / 0.0080 — for a full benchmark run, "
              f"use benchmark.py, not this short diagnostic run)")
    print("\nDone.")


if __name__ == "__main__":
    main()
