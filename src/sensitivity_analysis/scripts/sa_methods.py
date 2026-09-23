import glob

import yaml
import chaospy as cp
from sklearn.metrics import r2_score
from SALib.sample.latin import sample

from src.mmc_sim.tests.step_voltage_reference.scripts.svr_methods import *
from src.mmc_sim.core.logger import setup_logger

CONFIG_FILE = Path(".\\SA_config.yaml")

emt_sim_logger = setup_logger(
    "simulation"
)

pce_logger = setup_logger("pce_analysis")


def load_config_file(config_file: Path):
    """
    Load YAML configuration file.
    """
    with open(config_file) as file:
        return yaml.safe_load(file)


# Problem definition
def define_sampling_problem():
    """
    Define uncertain parameters for sensitivity analysis.
    Returns
    -------
    dict
        SALib problem definition.
    """
    return {"num_vars": 5,
            "names": ["L", "C", "R", "SCR", "XR"],
            "bounds": [
                [400.0, 800.0],  # L [mH]
                [800.0, 2100.0],  # C [uF]
                [6.0, 10.0],  # R [Ohm]
                [5.0, 15.0],  # SCR
                [5.0, 20.0]  # X/R ratio
            ]
            }


# Grid parameter calculation
def calculate_grid_parameters(dataframe: pd.DataFrame, nominal_power: float = 1200,
                              voltage: float = 400, frequency: float = 50):
    """
    Calculate equivalent grid impedance.
    Parameters
    ----------
    dataframe:
        Sample dataframe containing SCR and XR.
    Returns
    -------
    DataFrame
        Updated dataframe with Lg and Rg.
    """
    dataframe["Lg"] = (voltage ** 2 / (dataframe["SCR"] * nominal_power * 2 * math.pi * frequency))
    dataframe["Rg"] = (voltage ** 2 / (dataframe["SCR"] * nominal_power * dataframe["XR"]))
    return dataframe


# Sample generation
def generate_samples(number_of_samples: int, random_seed: int = None):
    """
    Generate Latin Hypercube samples.

    Parameters
    ----------
    number_of_samples:
        Number of samples.

    random_seed:
        Reproducibility seed.

    Returns
    -------
    DataFrame
    """

    problem = define_sampling_problem()
    pce_logger.info(f"Generating {number_of_samples} samples")
    samples = sample(problem, number_of_samples, seed=random_seed)
    dataframe = pd.DataFrame(samples, columns=problem["names"])
    dataframe = calculate_grid_parameters(dataframe)
    return dataframe


# Save samples
def save_samples(dataframe: pd.DataFrame, output_directory: Path, filename="sample_data.csv"):
    """
    Save generated samples.
    """

    output_directory.mkdir(parents=True, exist_ok=True)
    output_file = (output_directory / filename)
    dataframe.round(3).to_csv(output_file, index=False)
    return output_file


def post_process_pscad_sim_results(file_path: Path) -> pd.DataFrame:
    files = glob.glob(f"{file_path}\\*.csv")
    l_df = [pd.read_csv(file) for file in files]
    df = pd.concat(l_df)
    df.reset_index(drop=True, inplace=True)
    return df


def load_dataset(sample_file: Path, config_file: Path):
    """
    Load and merge simulation input/output data.

    Parameters
    ----------
    sample_file:
        Generated sensitivity samples.

    config_file:
        YAML file containing file paths.

    Returns
    -------
    DataFrame
        Prepared input-output dataset.
    """
    pce_logger.info("Loading sample and simulation data")
    samples = pd.read_csv(sample_file)

    cfg = load_config_file(config_file)
    sim_results_folder = Path(cfg["sensitivity_analysis"]["output"]["sim_data_dir"])
    sim_summary_df = post_process_pscad_sim_results(sim_results_folder)

    # Select required simulation outputs
    sim_summary_df = sim_summary_df[["L_mH", "C_uF", "R_ohms", "Xm", "Tcr", "Tcs"]]
    dataset = samples.merge(
        sim_summary_df,
        left_on=["L", "C", "R"],
        right_on=["L_mH", "C_uF", "R_ohms"],
        how="inner"
    )
    dataset.drop(columns=["L_mH", "C_uF", "R_ohms"], inplace=True)

    # Convert mH to H
    dataset["L"] /= 1000
    pce_logger.info(f"Dataset size: {dataset.shape}")
    return dataset


# Polynomial Chaos Model
def create_distribution():
    """
    Define uncertain parameter distributions.
    """
    return cp.J(
        cp.Uniform(0.4, 0.8),  # L
        cp.Uniform(800, 2100),  # C
        cp.Uniform(6, 10),  # R
        cp.Uniform(5, 15),  # SCR
        cp.Uniform(5, 20)  # XR
    )


def train_pce(X_train, Y_train, order, distribution):
    """
    Train polynomial chaos expansion surrogate.
    """
    pce_logger.info(f"Training PCE order={order}")
    expansion = cp.generate_expansion(order, distribution)
    model = cp.fit_regression(expansion, X_train.T, Y_train)
    return model


# Validation
def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    """
    Evaluate surrogate model accuracy.
    """
    Y_train_pred = model(*X_train.T)
    Y_test_pred = model(*X_test.T)
    metrics = {
        "r2_train": r2_score(Y_train, Y_train_pred),
        "r2_test":
            r2_score(Y_test, Y_test_pred),
        "relative_l2_error": np.linalg.norm(Y_test - Y_test_pred) / np.linalg.norm(Y_test)
    }
    return metrics


# Sobol sensitivity
def calculate_sobol_indices(model, distribution):
    """
    Calculate Sobol sensitivity indices.
    """
    pce_logger.info("Calculating Sobol indices")
    return {
        "first_order": cp.Sens_m(model, distribution),
        "second_order": cp.Sens_m2(model, distribution),
        "total": cp.Sens_t(model, distribution)
    }


def sample_data_emt_run(sample_data_path: str, plot_results: bool = False):
    cfg = load_configuration(CONFIG_FILE)

    model = PSCADModel(
        cfg["project_path"],
        cfg["project_name"],
    )

    project = model.get_project()

    project.component(cfg["mmc_id"]).parameters(idmode="0")
    project.parameters(time_step=cfg["time_step"])
    project.parameters(time_duration=cfg["time_duration"])
    project.parameters(sample_step=cfg["sample_step"])

    configure_ac_grid(
        project,
        cfg["ac_grid_id"],
        cfg["scr"],
        cfg["xr"],
        cfg["mva"],
        cfg["fn"],
        cfg["ac_voltage"],
    )

    model.set_output(cfg["output_file"])

    canvas_components = model.canvas_components()

    set_power_and_voltage_values(
        canvas_components,
        cfg["ref_power"],
        cfg["uref"],
        cfg["u_step"],
        cfg["step_time"]
    )

    dc_grid_components = find_dc_components(canvas_components)
    dc_network = ConfigDCGridComponents(dc_grid_components)
    sample_data_df = pd.read_csv(sample_data_path)
    for _, row in sample_data_df.iterrows():
        dc_grid_params = {"R": row["R"], "L": round(row["L"] / 1000, 6), "C": row["C"]}  # convert inductance to H
        dc_network.set_dc_network(
            **dc_grid_params
        )

        simulation = RunSimulation(model)

        emt_sim_logger.info(f"Running simulation for {dc_grid_params}")

        result_df = simulation.run(cfg["result_file"])

        # Ensure grid parameter values are exactly the same as those in the sample data
        new_file_name = format_rlc_filename(**dc_grid_params, file_name=cfg["output_file"])

        move_result_file(result_df, cfg["save_path"] / "sim_timeseries", new_file_name)
        if cfg["u_step"] > cfg["uref"]:
            result_summary = analyse_step_up_voltage_signal_for_sa(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                                   step_time=cfg["step_time"],
                                                                   voltage_reference=cfg["uref"])
            file_path = cfg["save_path"] / "sim_summary" / new_file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            result_summary.to_csv(file_path)
            if plot_results:
                plot_step_up_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                            step_time=cfg["step_time"], voltage_reference=cfg["uref"])
        elif cfg["u_step"] < cfg["uref"]:
            result_summary = analyse_step_down_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                              step_time=cfg["step_time"], voltage_reference=cfg["uref"])
            file_path = cfg["save_path"] / "sim_summary" / new_file_name
            file_path.parent.mkdir(parents=True, exist_ok=True)
            result_summary.to_csv(file_path, index=False)
            if plot_results:
                plot_step_down_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                              step_time=cfg["step_time"], voltage_reference=cfg["uref"])
    return


def analyse_step_up_voltage_signal_for_sa(
        file_path,
        step_time,
        time_col="TIME",
        signal_col="Vdc",
        step_pu=0.02,
        tol_factor=0.05,
        voltage_reference=640.0,
        mean_window=100
):
    """
    Analyse a step-up voltage signal from a CSV file.

    Tcr:
        Time from the voltage step until the signal first
        enters the tolerance band.

    Tcs:
        Time from the voltage step until the signal enters
        the tolerance band and remains there for the rest
        of the simulation.

    Xm:
        Maximum voltage deviation above the target voltage.
    """

    # LOAD DATA
    df = pd.read_csv(file_path)
    start_points = df.index[df[time_col] >= step_time]

    if len(start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time - 0.1: .3f} s."
        )

    steady_state_point = df.index.get_loc(start_points[0])
    df = df.iloc[steady_state_point:, :].copy()

    # EXTRACT PARAMETERS FROM FILE NAME
    file_name = str(file_path).split("\\")[-1]
    try:
        df["R_ohms"] = float(file_name.split("_")[-2][1:])
        df["L_mH"] = float(file_name.split("_")[-4][1:])
        df["C_uF"] = float(file_name.split("_")[-3][1:])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract R, L and C from file name: {file_name}")

    # SIGNAL
    t = df[time_col].to_numpy()
    y = df[signal_col].to_numpy()
    if len(y) == 0:
        raise ValueError("Signal contains no data.")

    # TOLERANCE BAND
    target = (1.0 + step_pu) * voltage_reference
    step_voltage = step_pu * voltage_reference
    tol = tol_factor * step_voltage
    lower = target - tol
    upper = target + tol * 0.5  # Reducing the tolerance improves the sensitivity analysis

    y_smooth = smooth_signal(y, window_size=mean_window)
    within_band = ((y_smooth >= lower) & (y_smooth <= upper))

    # POST-STEP REGION
    post_step_indices = np.where(t >= step_time)[0]

    if len(post_step_indices) == 0:
        raise ValueError(
            f"No samples found after step time "
            f"{step_time: .3f} s."
        )

    # PEAK
    peak_idx_local = np.argmax(y_smooth[post_step_indices])
    peak_idx = post_step_indices[peak_idx_local]
    peak_value = y_smooth[peak_idx]
    peak_time = t[peak_idx]

    # Maximum deviation above target
    Xm = peak_value - target

    # FIRST ENTRY INTO BAND
    first_entry_idx = None
    for i in post_step_indices:
        if within_band[i]:
            first_entry_idx = i
            break

    if first_entry_idx is not None:
        first_entry_time = t[first_entry_idx]
        Tcr = first_entry_time - step_time
    else:
        first_entry_time = None
        Tcr = None

    # SETTLING TIME
    settling_idx = None

    for i in post_step_indices:
        if within_band[i] and np.all(within_band[i:]):
            settling_idx = i
            break

    if settling_idx is not None:
        settling_time = t[settling_idx]
        Tcs = settling_time - step_time
    else:
        settling_time = None
        Tcs = None

    # STEADY-STATE VOLTAGE
    Vdc_ss = df[signal_col].iloc[-50:].mean()

    # RESULT
    Xm_limit = 0.004 * voltage_reference
    passed = (
            pd.notna(Tcr) and
            pd.notna(Tcs) and
            Tcr <= 0.2 and    # Tcr <= 0.2 s
            Tcs <= 0.3 and    # Tcs <= 0.3 s
            Xm <= Xm_limit
    )

    # SUMMARISE RESULTS
    result_df = df.iloc[[-1]].copy()
    result_df["Tcr"] = (round(Tcr, 3) if Tcr is not None else np.nan)
    result_df["Tcs"] = (round(Tcs, 3) if Tcs is not None else np.nan)
    result_df["Xm"] = round(Xm, 2)
    result_df["Vdc_ss"] = round(Vdc_ss, 2)
    result_df["Result"] = "Pass" if passed else "Fail"
    result_df = result_df.drop(columns=[time_col, signal_col], errors="ignore")
    result_df.reset_index(drop=True, inplace=True)
    return result_df


def analyse_step_down_voltage_signal_for_sa(
        file_path,
        step_time,
        time_col="TIME",
        signal_col="Vdc",
        step_pu=0.02,
        tol_factor=0.05,
        voltage_reference=640.0,
        mean_window=100
):
    """
    Analyse a step-down voltage signal from a CSV file.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.
    step_time : float
        Time at which the voltage step is applied.
    time_col : str
        Name of the time column.
    signal_col : str
        Name of the voltage signal column.
    step_pu : float
        Magnitude of the voltage step in per-unit.
    tol_factor : float
        Fraction of the step magnitude used as the settling tolerance.
    voltage_reference : float
        Initial/reference DC voltage.
    mean_window : int
        Window size used for signal smoothing.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing R, L, C, Tcr, Tcs, Xm, Vdc_ss and Result.
    """

    # LOAD DATA
    df = pd.read_csv(file_path)
    start_idx = df.index[df[time_col] >= step_time]

    if len(start_idx) == 0:
        raise ValueError(
            f"No data found at or after {step_time: .3f} s."
        )

    start_position = df.index.get_loc(start_idx[0])
    df = df.iloc[start_position:].copy()

    # EXTRACT R, L AND C FROM FILE NAME
    file_name = str(file_path).replace("\\", "/").split("/")[-1]

    try:
        parts = file_name.split("_")

        df["R_ohms"] = float(parts[-2][1:])
        df["L_mH"] = float(parts[-4][1:])
        df["C_uF"] = float(parts[-3][1:])
    except (IndexError, ValueError):
        raise ValueError(f"Could not extract R, L and C from file name: {file_name}")

    # SIGNAL DATA
    t = df[time_col].to_numpy()
    y = df[signal_col].to_numpy()

    if len(y) == 0:
        raise ValueError("The signal contains no data.")

    y_smooth = smooth_signal(y, window_size=mean_window)

    # TARGET VALUE
    target = (1.0 - step_pu) * voltage_reference

    # Step magnitude
    step_magnitude = step_pu * voltage_reference

    # Tolerance = tol_factor × step magnitude
    tol = tol_factor * step_magnitude

    lower = target - tol * 0.5   # Reducing the tolerance improves the sensitivity analysis
    upper = target + tol

    # TOLERANCE BAND
    within_band = ((y_smooth >= lower) & (y_smooth <= upper))

    # MINIMUM VOLTAGE / OVERSHOOT
    min_idx = np.argmin(y_smooth)

    min_value = y_smooth[min_idx]
    min_time = t[min_idx]

    # Voltage deviation below target
    Xm = target - min_value

    # TIME TO ENTER TOLERANCE BAND
    fall_idx = None

    for i in range(len(y_smooth)):
        if within_band[i]:
            fall_idx = i
            break

    if fall_idx is not None:
        fall_time = t[fall_idx]
        Tcr = fall_time - step_time
    else:
        Tcr = np.nan

    # SETTLING TIME
    settling_idx = None

    for i in range(len(y_smooth)):
        if within_band[i] and np.all(within_band[i:]):
            settling_idx = i
            break

    if settling_idx is not None:
        settling_time = t[settling_idx]
        Tcs = settling_time - step_time
    else:
        Tcs = np.nan

    # STEADY-STATE VOLTAGE
    Vdc_ss = df[signal_col].iloc[-50:].mean()

    # RESULT
    Xm_limit = 0.004 * voltage_reference
    passed = (
            pd.notna(Tcr) and
            pd.notna(Tcs) and
            Tcr <= 0.2 and  # Tcr <= 0.2 s
            Tcs <= 0.3 and  # Tcs <= 0.3 s
            Xm <= Xm_limit
    )

    # CREATE RESULT DATAFRAME
    result_df = df.iloc[[-1]].copy()
    result_df["Tcr"] = round(Tcr, 3) if pd.notna(Tcr) else np.nan
    result_df["Tcs"] = round(Tcs, 3) if pd.notna(Tcs) else np.nan
    result_df["Xm"] = round(Xm, 2)
    result_df["Vdc_ss"] = round(Vdc_ss, 2)
    result_df["Result"] = "Pass" if passed else "Fail"
    result_df.drop(columns=[time_col, signal_col], inplace=True, errors="ignore")
    result_df.reset_index(drop=True, inplace=True)
    return result_df
