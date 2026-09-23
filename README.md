# MMC Dynamic Performance Verification Framework

This repository provides an automated Python-based workflow for running and managing PSCAD simulations to assess the 
dynamic performance compliance of modular multilevel converters (MMCs) with the functional requirements specified in 
InterOPERA D2.1. It also includes a sensitivity analysis methodology based on the Sobol’ algorithm and a surrogate model 
using Bayesian Sparse Polynomial Chaos Expansion (BSPCE).

The framework supports parameter sweeps with systematic data export and post-processing. For each test, time-series data 
are exported as .csv files for further analysis, while a summary indicates whether the converter passes or fails the 
test. Control responses can also be visualised where appropriate.

---

## 1. Overview

The scripts in this project automate:

- Connection to PSCAD projects via the Python API
- Configuration of MMC control modes (active power, voltage control, droop control)
- Execution of different test scenarios:
  - Step changes in voltage reference
  - Step changes in active power reference
  - Power disturbances
  - Voltage disturbances
- Parameter sweeps over RLC grid equivalents
- Automatic extraction and storage of simulation results

The goal is to enable **reproducible, large-scale simulation studies** for HVDC and MMC-based grid-connected systems.

---

## 2. Project Structure
```text
MMC_Verification_Framework/
│
├── src/
│   ├── mmc_sim/
│   │   │
│   │   ├── core/
│   │   │   ├── __init__.py
│   │   │   ├── config.py
│   │   │   ├── pscad.py
│   │   │   ├── logger.py
│   │   │   ├── parameter_sweep.py
│   │   │   ├── config_components.py
│   │   │   ├── misc.py
│   │   │   └── simulation.py
│   │   │
│   │   └── tests/
│   │       ├── step_voltage_referemce/
│   │       │   ├── pscad_model/
│   │       │   ├── output/
│   │       │   └── scripts/
│   │       │       ├── config.yaml
│   │       │       ├── run_step_voltage_reference_response.py
│   │       │       └── svr_methods.py
│   │       ├── step_power_referemce/
│   │       │   ├── pscad_model/
│   │       │   ├── output/
│   │       │   └── scripts/
│   │       │       ├── config.yaml
│   │       │       ├── run_step_power_reference_response.py
│   │       │       └── spr_methods.py
│   │       ├── step_voltage_disturbance/
│   │       │   ├── pscad_model/
│   │       │   ├── output/
│   │       │   └── scripts/
│   │       │       ├── config.yaml
│   │       │       ├── run_step_voltage_disturbance_response.py
│   │       │       └── svd_methods.py
│   │       └─── step_power_disturbance/
│   │            ├── pscad_model/
│   │            ├── output/
│   │            └── scripts/
│   │               ├── config.yaml
│   │               ├── run_step_power_disturbance_response.py
│   │               └── spd_methods.py
│   │    
│   └── sensitivity_analysis/
│       ├── scripts/
│       │   ├── SA_config.yaml
│       │   ├── SA_input_data_generator.py
│       │   ├── SA_methods.py
│       │   ├── step_voltage_reference_response_sim_for_SA.py
│       │   └── SA_step_voltage_reference_response.py
│       └── output/
├── poetry.lock                    # Locked dependencies (Poetry)
├── pyproject.toml                 # Project configuration & dependencies
└── README.md                      # Project documentation
```

---
## 3. Experiment Types
- N.B: Manually launch PSCAD before initiating any test. This approach is adopted to avoid license and windows defender issues.
### 3.1 Step Voltage Disturbance Response
- File: `run_step_voltage_disturbance_response.py`
- Purpose: To assess the open-loop quasi-static behavior of active power response to changes in DC voltage
- Control mode: Voltage droop control (`idmode = 2`)
- In the `config.py` file, set voltage level after a step (`step_voltage`)
---

### 3.2 Step Power Disturbance Response
- File: `run_step_power_disturbance_response.py`
- Purpose: 
  - To evaluate the closed-loop dynamics of the device under test (DUT) under various possible operational conditions. 
  - To assess whether the DUT possesses sufficient disturbance rejection capability to handle worst- case disturbances.
- Model includes full RLC equivalent of DC grid
---

### 3.3 Step Voltage Reference Response
- File: `run_step_voltage_reference_response.py`
- Purpose: To ensure the capability of the AC/DC converter unit to adhere to the DC voltage set-point at its DC-PoC at steady state
- Control mode: DC voltage control (`idmode = 0`)
---

### 3.4 Step Power Reference Response
- File: `run_step_power_reference_response.py`
- Purpose: Analyze active power tracking performance under reference changes
- Control mode: Active power control (`idmode = 1`)
---

### 3.5 Sensitivity analysis
- Sensitivity analysis assess the impact of the DC and AC grid parameters on control response.
- File: `SA_input_data_generator.py` generates sample data
- File: `step_voltage_reference_response_sim_for_SA.py` performs EMT simulations in PSCAD using the sample data.
- The KPI's from the EMT simulations are processed using file: `step_voltage_ref_response_result_analysis.ipynb` in the result analysis directory.
- Sobol sensitivity analysis is then performed using a surrogate PCE model with file: `SA_step_voltage_ref_response.py`.
---

## 4. Parameter Sweeps

All experiments support automated parameter sweeps over:

### Electrical parameters
- Inductance (L)
- Resistance (R)
- Capacitance (C)

### Sweep structure
- Full Cartesian product of selected parameter ranges
- Each simulation is executed independently
- Results are stored with structured file names

Example:
```
x_L400_C1300_R6_.csv
```
---

## 5. Configuration (config.yaml)

The simulation settings are centralized in `config.yaml` with the file required to run the simulation needing minimal 
updates. This ensures:

Reproducibility
Easy modification of test cases
Separation of code and experiment setup
---

## 6. Important!
Given the automation constraints in PSCAD,
- The DC grid RLC requivalent parameters should be named R_dc, L_dc, C_dc for the resistor, inductor, and capacitor respectively.
- Real constants in PSCAD for the power references should be named Pref_init, and P_ref_step for the initial and final 
power after a step respectively.
- Real constants in PSCAD for the voltage reference should be named "uref" and the final voltage after a step, "u_step".
- Ensure that any library objects referenced in the scripts are named appropriately.
- The source code should be adapted accordingly for different choices of parameter/component names and these parameters 
should be accessible on the main canvas of PSCAD.
---

## 7. Requirements & Installation

This project uses Poetry for dependency management and environment isolation. It is compatible with Python 3.10.
Clone the project using

`git clone <repo-url>`

Install Poetry (if not already installed):

`pip install poetry`

Then install all dependencies:

`poetry install`

Activate the environment:

`.venv\Scripts\activate` or `poetry env activate`

Run the tests in the "mmc_sim" directory as well as the sensitivity analysis in "sensitivity_analysis" directory as desired


---

## 8. License

For academic and research use only (update as needed).