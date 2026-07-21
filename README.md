# hypergraph-epidemic-sensitivity

This repository contains the code used for the experiments in the paper:

**"Structural Sensitivity in Epidemic Agent-Based Models"**

The repository implements a full pipeline for running structural
sensitivity experiments in the STRIDE epidemic agent-based simulator,
extracting hypergraph-based diagnostics, and generating the analysis
figures used in the paper.

The pipeline consists of three main stages:

1.  Simulation -- run epidemic simulations for population perturbations.
2.  Processing -- extract structural diagnostics and aggregate results.
3.  Analysis -- generate plots and statistical summaries.

# Overview of the Pipeline

### Select an intervention plan

Copy the desired intervention configuration to STRIDE's active configuration
file. For example, to reproduce the general-intervention analysis:

```bash
cp data/PLAN2_mixed_intervention.xml config/run_default.xml
```

The five XML files in `data/` correspond to the five intervention regimes
described in the paper. Repeat the simulation and processing pipeline for each
plan when reproducing the cross-policy robustness analysis. 

## 1. Simulation

Simulation experiments are executed using:

    simulation/run_sensitivity_all.py

This script:

-   runs the STRIDE simulator
-   iterates over all population perturbations
-   runs multiple stochastic seeds
-   executes epidemic simulations for one intervention plan
-   extracts daily simulation outputs
-   computes intermediate diagnostics
-   stores results as compressed archives

The script expects the STRIDE simulator to be located in the parent
directory of the repository:

    ../stride

The simulator produces daily agent state files that are processed by the
pipeline.

------------------------------------------------------------------------

## 2. Processing

After simulations are completed, the extracted outputs are processed
using:

    processing/run_structural_sensitivity_pipeline.py

This stage:

-   extracts structural diagnostics
-   aggregates results across seeds
-   computes outcome metrics
-   builds summary datasets
-   prepares data used in the final analysis

The processing stage produces the datasets used in the paper's figures
and robustness analyses.

------------------------------------------------------------------------

## 3. Analysis

The final plots and analyses are generated using scripts in:

    analysis/

Main scripts:

    analysis/analysis_plot.py
    analysis/robustness_plots.py

These scripts generate the figures used in the paper, including:

-   structural sensitivity comparisons
-   hypergraph diagnostic associations
-   robustness analyses across intervention regimes

------------------------------------------------------------------------

# Data

The `data/` directory contains:

### Population Perturbations

    Populations.zip

This archive contains the synthetic population variants used in the
structural sensitivity experiments. These perturbations modify
structural properties such as:

-   household composition
-   community mixing
-   age distribution
-   school assignments
-   workplace assignments

These correspond to the population variants described in the paper.

### Intervention Plans

The XML files define the STRIDE simulation configurations for the
intervention scenarios:

    PLAN1_no_interventions.xml
    PLAN2_mixed_intervention.xml
    PLAN3_work_focused.xml
    PLAN4_social_focused.xml
    PLAN5_broad_NPI_plus_TTI.xml

Each plan represents a different policy scenario used in the robustness
analysis.

------------------------------------------------------------------------

# STRIDE Simulator

The experiments use a modified version of the STRIDE COVID-19 simulator:

https://github.com/NWils98/stride_covid19_v1

## Building STRIDE

STRIDE requires CMake, a C++ compiler, and the Boost filesystem, thread,
date-time, and system libraries.

Clone and build the simulator as follows:

```bash
git clone https://github.com/NWils98/stride_covid19_v1.git
cd stride_covid19_v1
git checkout c0429a773e4515cc674fa721df614d53a94bff61

cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=<absolute-path-to-hypergraph-epidemic-sensitivity>

cmake --build build --parallel
cmake --install build
```

Installing STRIDE directly into the structural-sensitivity repository creates
the `bin/`, `config/`, and `data/` directories expected by
`simulation/run_sensitivity_all.py`.

After installation, extract the baseline and perturbed populations:

```bash
cd <absolute-path-to-hypergraph-epidemic-sensitivity>
unzip data/pop_belgium600k_c500_teachers_censushh.zip -d data
unzip data/Populations.zip -d data
```

------------------------------------------------------------------------

# Running the Full Experiment

### Step 1 --- Run simulations

    cd simulation
    python run_sensitivity_all.py

This runs the full structural sensitivity experiment for the selected
intervention plan.

### Step 2 --- Process results

    cd processing
    python run_structural_sensitivity_pipeline.py

This extracts diagnostics and builds aggregated datasets.

### Step 3 --- Generate figures

    cd analysis
    python analysis_plot.py

Optional robustness analysis:

    python robustness_plots.py

------------------------------------------------------------------------

# Output

The pipeline produces:

-   processed structural metrics
-   aggregated epidemic outcomes
-   hypergraph diagnostic summaries
-   plots used in the paper

Intermediate outputs are stored as **Parquet files** for efficient
processing.

------------------------------------------------------------------------

# Reproducibility

All experiments are deterministic conditional on:

-   population variant
-   stochastic seed
-   intervention plan configuration

Simulation seeds are controlled in the experiment scripts.

------------------------------------------------------------------------

# License

This repository is released under the license specified in `LICENSE`.

