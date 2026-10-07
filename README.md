# Monotonicity Constraints in Bayesian Neural Networks

This repository contains the code developed for my master's thesis, “Predictive Accuracy and Uncertainty under Monotonicity Constraints in Bayesian Neural Networks,” at the University of Trier.

## Overview

This study investigates how incorporating monotonicity information into Bayesian neural networks (BNNs) affects predictive accuracy and uncertainty quantification. A controlled simulation study compares an unconstrained BNN with soft-constrained BNNs using three penalty strengths and a hard-constrained BNN with an architecture that guarantees monotonicity.

The models are evaluated under correctly specified, partially misspecified, and misspecified monotonicity assumptions. The experimental design additionally varies sample size and observation noise and distinguishes between interpolation within the training domain and extrapolation beyond it. This design makes it possible to examine how the effects of monotonicity constraints on predictive accuracy and uncertainty quantification vary with constraint validity, data availability, noise level, and prediction region.


Posterior inference for all models is performed using the No-U-Turn Sampler (NUTS), implemented in Python with JAX and NumPyro. Performance is assessed using measures of point prediction, probabilistic prediction, latent-function uncertainty, epistemic and aleatoric uncertainty, monotonicity adherence, and MCMC diagnostics.

## Key findings

- Under correct specification, the hard-constrained BNN achieved the lowest
  RMSE and NLPD and narrower latent intervals during interpolation while maintaining
  predictive coverage close to the nominal 95% level. These benefits were
  also observed in the small-sample, high-noise setting, where the valid
  constraint provided useful structural information despite the limited
  information available from the data. During extrapolation, the hard-constrained model produced greater epistemic
  uncertainty and latent interval width than the other models but also achieved lower RMSE and NLPD and higher
  latent-function coverage. The valid constraint therefore supported a more
  appropriate representation of uncertainty beyond the observed data rather
  than uniformly narrowing posterior uncertainty.

- Under misspecification, predictive coverage could remain close to the nominal
  95% level because the constrained models partially compensated for the
  structural discrepancy by overestimating the observation-noise variance,
  even when latent-function coverage deteriorated substantially.

- Under misspecification, more informative data could reinforce confidence in
  an incorrect latent relationship: epistemic uncertainty decreased as the
  posterior concentrated within the constrained but misspecified function
  class, while latent-function coverage deteriorated.

- Under partial misspecification, local monotonicity violations reduced
  interpolation performance, although the broader directional information
  could still benefit extrapolation in some settings.



## Repository structure


```text
monotonic-bnn/
├── configs/
│   ├── prior_check.yaml
│   ├── smoke.yaml
│   ├── pilot.yaml
│   └── full.yaml
│
├── src/
│   ├── __init__.py
│   ├── data.py
│   ├── experiment.py
│   ├── metrics.py
│   ├── models.py
│   └── prior_checks.py
│
├── tests/
│   ├── test_data_metrics.py
│   └── test_prior_checks.py
│
├── run_prior_checks.py
├── run_simulation.py
├── requirements.txt
├── README.md
└── .gitignore
```

The main files and their roles are:

- configs/*.yaml — Contains configuration files for the prior checks, smoke tests, pilot runs, and full simulation study.
- src/data.py — Generates the simulated datasets for the correctly specified, partially misspecified, and misspecified scenarios, and handles data standardization.

- src/models.py — Defines the unconstrained, soft-constrained, and hard-constrained Bayesian neural network models.

- src/experiment.py — Handles posterior inference, posterior prediction, and the main simulation workflow across experimental conditions and replications.

- src/metrics.py — Computes predictive accuracy and uncertainty measures, interpolation/extrapolation summaries, monotonicity adherence, and MCMC diagnostics.

- src/prior_checks.py — Implements the prior predictive checks used to examine the behavior of the BNN priors under different prior-scale settings.

- tests/ — Contains automated tests for data generation, metric calculations, and prior-check functionality.

- run_prior_checks.py — Entry point for running the prior predictive checks.

- run_simulation.py — Entry point for running the simulation experiments.

## Installation

Python 3.11 is recommended.

```bash
git clone https://github.com/kenta-yashiro-stat/monotonic-bnn.git
cd monotonic-bnn
python -m pip install -r requirements.txt
```

## Quick start

Run the prior predictive checks:

```bash
python run_prior_checks.py --config configs/prior_check.yaml
```

Validate the simulation pipeline using short MCMC chains:

```bash
python run_simulation.py --config configs/smoke.yaml
```

Run the full simulation:

```bash
python run_simulation.py --config configs/full.yaml
```

The full experiment involves 3,000 NUTS fits and is computationally intensive.

## Configurations

- `prior_check.yaml`: prior predictive checks
- `smoke.yaml`: short pipeline validation
- `pilot.yaml`: preliminary runtime and convergence assessment
- `full.yaml`: complete simulation study

## Experimental design

The simulation varies:

- Monotonicity assumption: correct, partially misspecified, or misspecified
- Sample size: $n=50$ or $n=200$
- Observation noise: $\sigma=0.20$ or $\sigma =0.60$
- Model: unconstrained BNN, soft-constrained BNN with $\lambda \in \{10, 100, 1000\}$, or hard-constrained BNN
- Prediction region: interpolation or extrapolation

The design comprises 12 experimental conditions:

- 3 monotonicity scenarios
- 2 sample sizes
- 2 observation-noise levels

Each condition is evaluated using five models and 50 paired simulation
replications, resulting in $12 \times 5 \times 50 = 3{,}000$ model fits.

## Constraint implementations

The soft-constrained models penalize negative partial derivatives with respect to $x_1$. The hard-constrained model guarantees monotonicity in $x_1$ through nonnegative path weights. No monotonicity is imposed on $x_2$.

## Posterior inference

Posterior inference is performed using four NUTS chains with:

- 1,000 warm-up iterations per chain
- 1,000 posterior samples per chain
- Target acceptance probability of 0.90
- Maximum tree depth of 10

The simulation uses a fixed master seed to ensure reproducibility.

## CPU chains

To run four MCMC chains in parallel on a CPU, set the following environment
variable before starting the simulation:

```bash
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
```

The number of host devices should generally match the number of MCMC chains.

## Outputs

Results are written to the directory specified by `output_dir` in the selected
configuration file. The principal outputs are:

- `raw_metrics.csv`: Replication-level evaluation metrics
- `summary_metrics.csv`: Aggregated results and Monte Carlo standard errors
- `run_config.yaml`: Configuration used for the run

The reported metrics cover predictive accuracy, interval calibration,
epistemic and aleatoric uncertainty, monotonicity adherence, and MCMC diagnostics.