# QTabGAN

A hybrid quantum-classical Generative Adversarial Network (GAN) for synthetic tabular data generation.

This repository contains a research-oriented implementation inspired by:

> QTabGAN: A Hybrid Quantum-Classical GAN for Tabular Data Synthesis

The model combines a variational quantum circuit (VQC) with classical neural networks to generate synthetic tabular data.

---

## Overview

Traditional GANs generate data entirely using classical neural networks.

QTabGAN instead uses a hybrid architecture:

```text
                 Random Noise
                      |
                      v
              Variational Quantum
                   Circuit
                      |
                      v
              Quantum Probability
                   Vector
                      |
                      +---------+
                      |         |
                      |    Class Condition
                      |         |
                      v         v
                 Classical Mapper
                      |
                      v
                Synthetic Data
                      |
                      v
                 Discriminator
                   /       \
                Real       Fake