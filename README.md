# Experimental Analysis of Hybrid Quantum-Classical Variational Circuits for Classification Under NISQ Constraints

![Project overview](fig3_hybrid_workflow.png)

This repository contains the experimental outputs for a system-level
study of hybrid quantum-classical workflows under noisy
intermediate-scale quantum (NISQ) constraints.

The study evaluates variational quantum circuits (VQCs), QAOA
simulations, hybrid workflow overhead, scheduling strategies, noise
sensitivity, scalability, and deployment-readiness indicators. The
accompanying paper frames these components as complementary evidence
across algorithmic, circuit, hardware, and system levels.

> **Scope note:** The repository contains experimental outputs and the
> supplied notebook. Results should be interpreted within the tested
> configurations and simulation assumptions. They should not be treated
> as evidence of universal quantum advantage.

## 1. Project Highlights

-   VQC simulations across multiple qubit counts and circuit depths.
-   Depolarizing-noise experiments at several error levels.
-   QAOA simulation experiments.
-   Hybrid workflow timing analysis.
-   Classical baseline comparison.
-   QPU scheduling simulation.
-   Circuit scalability analysis.
-   Readiness-oriented analysis.
-   Publication-quality PNG figures and CSV result files.
-   Reproducible notebook-based experimental workflow.

The supplied experimental summary reports 80 VQC experiments, 12 QAOA
experiments, 25 hybrid-workflow runs, 50 classical-baseline experiments,
24 scheduling runs, and 33 noise experiments.

## 2. Repository Structure

``` text
.
├── notebookb074434e1a.ipynb
├── README.md
│
├── executive_summary.txt
│
├── data_vqc_experiments.csv
├── data_qaoa_experiments.csv
├── data_hybrid_workflow.csv
├── data_classical_baseline.csv
├── data_scheduling.csv
├── data_noise_impact.csv
├── data_scalability.csv
├── data_readiness.csv
│
├── fig1_circuit_complexity.png
├── fig2_noise_impact.png
├── fig3_hybrid_workflow.png
├── fig4_scheduling.png
├── fig5_scalability.png
├── fig6_readiness.png
└── supplementary_figures.png
```

## 3. Experimental Workflow

![Circuit complexity](fig1_circuit_complexity.png)

The experimental workflow follows a hybrid quantum-classical pipeline:

1.  Prepare and normalize the experimental inputs.
2.  Construct the quantum circuit.
3.  Vary qubit count and circuit depth.
4.  Apply the selected noise model.
5.  Execute the circuit with the simulator.
6.  Record circuit depth, gate count, parameters, and execution time.
7.  Compare hybrid workflow components with classical processing.
8.  Evaluate scheduling, scalability, noise, and readiness-related
    metrics.
9.  Store numerical results in CSV files.
10. Generate the figures used for analysis.

The paper describes the conceptual hybrid pipeline as classical
preprocessing, quantum state encoding, VQC execution, measurement
aggregation, and classical optimization. The reported workflow also
considers resource contention and scheduling.

## 4. VQC Experiments

The supplied VQC results contain the following fields:

  Field              Description
  ------------------ ------------------------------
  `qubits`           Number of qubits
  `layers`           Circuit layers
  `depth`            Circuit depth
  `gates`            Gate count
  `noise`            Depolarizing-noise level
  `execution_time`   Circuit execution time
  `params`           Number of circuit parameters

The supplied result data cover qubit counts from 4 to 12 and circuit
layers from 1 to 4, with noise levels of 0, 0.01, 0.02, and 0.05.

![Noise impact](fig2_noise_impact.png)

The supplied summary reports a strong degradation in the recorded
noise-impact metric as noise increases. The repository should therefore
be used to inspect the underlying CSV values rather than relying only on
aggregated statements.

## 5. QAOA Experiments

QAOA experiments are included as a supporting quantum-optimization
study.

The supplied experimental summary reports:

-   12 QAOA experiments.
-   4--10 qubits.
-   1--3 QAOA layers (`p` layers).
-   Average circuit depth of 28.
-   Average gate count of 65.
-   Average execution time of approximately 0.0041 s.

The detailed measurements are available in:

``` text
data_qaoa_experiments.csv
```

## 6. Hybrid Workflow Analysis

![Hybrid workflow](fig3_hybrid_workflow.png)

The hybrid workflow separates:

-   Quantum processing time.
-   Classical processing time.
-   Data-transfer time.
-   Total workflow time.

The supplied summary reports an average quantum-processing contribution
of approximately 18.4%, classical processing of approximately 65.4%, and
data transfer of approximately 16.1% for the summarized workload.

The raw measurements are available in:

``` text
data_hybrid_workflow.csv
```

These measurements are workflow-level simulation results. They should
not be interpreted as a matched classical-vs-quantum inference
benchmark.

## 7. Classical Baseline

The repository includes classical timing and accuracy measurements in:

``` text
data_classical_baseline.csv
```

The supplied summary reports an average classical time of approximately
0.0176 s and an average hybrid time of approximately 0.0198 s for the
summarized workload.

The comparison is intended to provide context for the hybrid workflow
rather than establish a general quantum speedup.

## 8. Scheduling Analysis

![Scheduling](fig4_scheduling.png)

The scheduling experiment compares queue-management strategies under
simulated workloads.

The supplied summary includes:

-   Naive/FIFO scheduling.
-   Priority-based scheduling.
-   Multiple core configurations.
-   Waiting time.
-   Turnaround time.

The summarized results report a small improvement for the priority-based
strategy. The detailed measurements are stored in:

``` text
data_scheduling.csv
```

This is a scheduling simulation. It does not reproduce or measure a
cloud provider's internal QPU scheduler.

## 9. Scalability Analysis

![Scalability](fig5_scalability.png)

Scalability experiments examine how circuit resources change with qubit
count.

The supplied summary reports polynomial-fit approximations for:

-   Gates: approximately `O(n^0.98)`.
-   Depth: approximately `O(n^0.74)`.
-   Parameters: approximately `O(n^1.00)`.

The detailed values are available in:

``` text
data_scalability.csv
```

These fitted exponents describe the supplied experimental dataset and
should not be generalized beyond the evaluated circuit construction.

## 10. Readiness Analysis

![Readiness](fig6_readiness.png)

The repository contains a readiness-oriented analysis in:

``` text
data_readiness.csv
```

and the corresponding visualization:

``` text
fig6_readiness.png
```

The readiness analysis is intended as a study-specific exploratory
assessment. It should not be interpreted as a universally validated
deployment-readiness score.

## 11. Supplementary Visualization

![Supplementary figures](supplementary_figures.png)

Additional plots are consolidated in:

``` text
supplementary_figures.png
```

## 12. Noise Model

The experimental notebook uses Qiskit Aer noise-model functionality,
including depolarizing errors.

The supplied paper methodology distinguishes:

-   Ideal simulation.
-   Synthetic depolarizing-noise simulation.
-   Backend-calibrated simulation.
-   Real-hardware execution.

These conditions should not be treated as interchangeable evidence.

## 13. Software Environment

The supplied notebook imports the following major Python packages:

``` text
numpy
pandas
matplotlib
seaborn
scipy
qiskit
qiskit-aer
```

It also uses standard Python modules such as:

``` text
json
time
os
warnings
datetime
collections
math
```

A practical environment can be created with:

``` bash
python -m venv .venv
```

Activate the environment:

### Linux / macOS

``` bash
source .venv/bin/activate
```

### Windows

``` powershell
.venv\Scripts\activate
```

Install the main dependencies:

``` bash
pip install numpy pandas matplotlib seaborn scipy qiskit qiskit-aer jupyter
```

> **Version note:** Use the exact package versions from the experimental
> environment when reproducing numerical results. The supplied paper
> text refers to Qiskit Aer 0.13 for the simulation methodology, while
> the uploaded notebook itself does not contain a complete environment
> lock file.

## 14. Running the Notebook

Start Jupyter:

``` bash
jupyter notebook
```

Open:

``` text
notebookb074434e1a.ipynb
```

Then run the notebook cells from top to bottom.

For a clean reproduction:

1.  Use a fresh Python environment.
2.  Install the required dependencies.
3.  Open the notebook.
4.  Run all cells sequentially.
5.  Preserve the same random seeds and experimental parameters when
    applicable.
6.  Export generated figures and CSV files.
7.  Compare the regenerated outputs with the supplied `data_*.csv` and
    `fig*.png` files.

## 15. Results Included in This Repository

### Figures

  Figure                          Purpose
  ------------------------------- ------------------------------
  `fig1_circuit_complexity.png`   Circuit complexity
  `fig2_noise_impact.png`         Noise sensitivity
  `fig3_hybrid_workflow.png`      Hybrid workflow timing
  `fig4_scheduling.png`           Scheduling comparison
  `fig5_scalability.png`          Scalability
  `fig6_readiness.png`            Readiness-oriented analysis
  `supplementary_figures.png`     Supplementary visual results

### Data

  File                            Purpose
  ------------------------------- ---------------------------------
  `data_vqc_experiments.csv`      VQC circuit experiments
  `data_qaoa_experiments.csv`     QAOA experiments
  `data_hybrid_workflow.csv`      Hybrid timing measurements
  `data_classical_baseline.csv`   Classical baseline measurements
  `data_scheduling.csv`           Scheduling experiments
  `data_noise_impact.csv`         Noise analysis
  `data_scalability.csv`          Scaling analysis
  `data_readiness.csv`            Readiness analysis

## 16. Reproducibility and Interpretation

The supplied paper emphasizes that the experimental evidence has
different levels:

``` text
Analytical proxy
      ↓
Controlled simulation
      ↓
Noise simulation
      ↓
Backend-calibrated simulation
      ↓
Real hardware
      ↓
System/deployment assessment
```

An analytical gate-survival proxy is not a measured physical fidelity.
Simulation-side timing is not inference latency. A single-backend
hardware experiment is not representative of all quantum hardware.

The paper also identifies missing reproducibility information that
should be addressed for a submission-grade release, including exact data
splits, trial-level outputs, transpiler settings, calibration snapshots,
fold-local preprocessing, and complete circuit metadata.

## 17. Current Experimental Scope

The supplied materials evaluate small benchmark workloads and controlled
circuit configurations. The reported scope includes VQC classification,
QAOA simulation, hybrid workflow timing, scheduling, noise, scalability,
and readiness-oriented analyses.

The supplied paper reports that its conclusions are restricted to the
evaluated datasets, configurations, noise assumptions, and hardware
conditions. It does not claim universal quantum advantage or universal
deployment readiness.

## 18. Citation

If you use this repository in academic work, cite the associated
manuscript:

``` text
Anurag Bhattacharjee, Sayan Ghosh, and Anjan Bandyopadhyay,
"Experimental Analysis of Hybrid Quantum-Classical Variational Circuits
for Classification Under NISQ Constraints."
```

## 19. License

No license information was supplied with the uploaded experimental
files. Add an appropriate license before public release.

## 20. Contact

**Anurag Bhattacharjee**

For the final public repository, consider adding:

-   `requirements.txt` or `environment.yml`
-   exact Qiskit and Python versions
-   random seeds
-   dataset acquisition instructions
-   train/test split definitions
-   raw trial-level results
-   calibration snapshots
-   transpiler configuration
-   hardware job identifiers where appropriate
-   a formal open-source license
