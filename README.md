# hypergraph-epidemic-sensitivity

This repository contains the code used for the experiments in the paper:

**"Explaining Epidemic Sensitivity to Population Structure using
Hypergraph Diagnostics."**

The repository implements a full pipeline for running structural
sensitivity experiments in the STRIDE epidemic agent-based simulator,
extracting hypergraph-based diagnostics, and generating the analysis
figures used in the paper.

The pipeline consists of three main stages:

1.  Simulation -- run epidemic simulations for population perturbations.
2.  Processing -- extract structural diagnostics and aggregate results.
3.  Analysis -- generate plots and statistical summaries.

# Overview of the Pipeline

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

The STRIDE simulator is **not included in this repository for the
double-blind submission**.

STRIDE is a publicly available epidemic agent-based simulator. After the
review process, the repository will be updated to include the correct
link and integration instructions.

Expected structure during reproduction:

    project_root/
        stride/
        structural-sensitivity-repo/
            simulation/
            processing/
            analysis/

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

