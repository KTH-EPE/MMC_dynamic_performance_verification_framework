import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from typing import Dict, List

from src.mmc_sim.core.config import Config
from src.mmc_sim.core.pscad import PSCADModel
from src.mmc_sim.core.config_components import ConfigDCGridComponents
from src.mmc_sim.core.run_simulation import RunSimulation
from src.mmc_sim.core.parameter_sweep import ParameterSweep
from src.mmc_sim.core.misc import *
from src.mmc_sim.core.logger import setup_logger

CONFIG_FILE = Path("config.yaml")

logger = setup_logger(
    "simulation"
)


def apply_scr_and_xr(scr: float, xr: float, ac_voltage: float = 400.,
                     sn: float = 1200., fn: float = 50.) -> (float, float):
    """
    Compute equivalent grid inductance (Lg) and resistance (Rg)
    from Short Circuit Ratio (SCR) and X/R ratio.

    Assumes:
    - Base voltage = 400 kV
    - Nominal power = sn (MVA)
    - Frequency = fn (Hz)
    """
    lg = ac_voltage ** 2 / (scr * sn * 2 * fn * math.pi)
    rg = ac_voltage ** 2 / (scr * sn * xr)
    return lg, rg


def configure_ac_grid(project, component_id: str, scr: float, xr: float, mva: float, fn: float, u_ac: float):
    lg, rg = apply_scr_and_xr(scr=scr, xr=xr, sn=mva, fn=fn, ac_voltage=u_ac)

    grid = project.component(component_id)
    grid.parameters(Rg=rg, Lg=lg)


def set_power_and_voltage_values(components, ref_power, uref, u_step, step_time):
    for comp in components:
        try:
            name = comp.parameters().get("Name")
        except Exception:
            continue

        if name == "Pref":
            comp.parameters(Value=ref_power)
        elif name == "uref":
            comp.parameters(Value=uref)
        elif name == "u_step":
            comp.parameters(Value=u_step)
        elif name == "step_obj":
            comp.parameters(X=step_time)


def find_dc_components(components):
    required = {
        "R_dc": "R",
        "L_dc": "L",
        "C_dc": "C",
    }

    dc_components = {}

    for comp in components:
        try:
            name = comp.parameters().get("Name")
        except Exception:
            continue

        if name in required:
            dc_components[required[name]] = comp

    missing = set(required.values()) - set(dc_components)

    if missing:
        raise ValueError(
            f"Missing DC components in PSCAD model: {sorted(missing)}"
        )

    return dc_components


def load_configuration(config_file):
    cfg = Config(config_file)

    sim_cfg = "step_voltage_ref"

    return {
        "project_path": Path(cfg.get(sim_cfg, "model")),
        "project_name": cfg.get(sim_cfg, "name"),
        "mva": cfg.get(sim_cfg, "mva"),
        "fn": cfg.get(sim_cfg, "fn"),
        "step_time": cfg.get(sim_cfg, "step_time"),
        "sample_step": cfg.get(sim_cfg, "sample_step"),
        "time_step": cfg.get(sim_cfg, "time_step"),
        "time_duration": cfg.get(sim_cfg, "time_duration"),
        "output_file": cfg.get(sim_cfg, "results", "file_name"),
        "save_path": Path(cfg.get(sim_cfg, "results", "save_path")),
        "result_file": Path(
            f"{cfg.get(sim_cfg, 'results', 'result_file')}/"
            f"{cfg.get(sim_cfg, 'results', 'file_name')}"
        ),
        "ref_power": cfg.get(sim_cfg, "ref_power"),
        "uref": cfg.get(sim_cfg, "reference_voltage"),
        "u_step": cfg.get(sim_cfg, "step_voltage"),
        "scr": cfg.get(sim_cfg, "SCR"),
        "xr": cfg.get(sim_cfg, "XR"),
        "ac_voltage": cfg.get(sim_cfg, "ac_voltage"),
        "mmc_id": cfg.get(sim_cfg, "components", "mmc_id"),
        "ac_grid_id": cfg.get(sim_cfg, "components", "ac_grid_id"),
    }


def single_run(rlc_params: Dict[str, float], plot_results: bool = True):
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
    dc_network.set_dc_network(**rlc_params)
    logger.info(f"Running simulation for {rlc_params}")
    simulation = RunSimulation(model)
    result_df = simulation.run(cfg["result_file"])
    new_file_name = format_rlc_filename(**rlc_params, file_name=cfg["output_file"])
    move_result_file(result_df, cfg["save_path"] / "sim_timeseries", new_file_name)
    if cfg["u_step"] > cfg["uref"]:
        result_summary = analyse_step_up_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                        step_time=cfg["step_time"], voltage_reference=cfg["uref"])
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
        result_summary.to_csv(file_path)
        if plot_results:
            plot_step_down_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                          step_time=cfg["step_time"], voltage_reference=cfg["uref"])
    return


def parameter_sweep_run(rlc_params: Dict[str, List[float]], plot_results: bool = False):
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

    dc_grid_parameters = ParameterSweep(rlc_params)

    dc_network = ConfigDCGridComponents(dc_grid_components)
    for params in dc_grid_parameters.combinations():
        dc_network.set_dc_network(
            **params
        )

        simulation = RunSimulation(model)
        logger.info(f"Running simulation for {params}")
        result_df = simulation.run(cfg["result_file"])
        new_file_name = format_rlc_filename(**params, file_name=cfg["output_file"])
        move_result_file(result_df, cfg["save_path"] / "sim_timeseries", new_file_name)
        if cfg["u_step"] > cfg["uref"]:
            result_summary = analyse_step_up_voltage_signal(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                            step_time=cfg["step_time"], voltage_reference=cfg["uref"])
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


def analyse_step_up_voltage_signal(
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
        df["L_H"] = float(file_name.split("_")[-4][1:])
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
    upper = target + tol

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


def smooth_signal(y, window_size=50):
    """
    Apply moving average (running average) to smooth signal.

    Parameters:
    - y: input signal (numpy array)
    - window_size: number of samples in averaging window

    Returns:
    - smoothed signal (same length as input)
    """
    return pd.Series(y).rolling(window=window_size, center=True, min_periods=1).mean().values


def plot_step_up_voltage_signal(
        file_path,
        step_time=3,
        time_col="TIME",
        signal_col="Vdc",
        step_pu=0.02,
        voltage_reference=640,
        tol_factor=0.05,
        mean_window=100
):
    # LOAD DATA
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    start_points = df.index[
        df[time_col] >= (step_time - 0.5)
    ]

    step_start_points = df.index[
        df[time_col] >= step_time
    ]

    if len(start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time - 0.5: .3f} s."
        )

    if len(step_start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time: .3f} s."
        )

    # Start analysis 0.5 s before the step
    steady_state_point = df.index.get_loc(start_points[0])
    df = df.iloc[steady_state_point:].copy()

    # Find step location again after slicing df
    step_point = np.searchsorted(df[time_col].to_numpy(), step_time, side="left")

    # SIGNAL
    t = df[time_col].to_numpy()
    y = df[signal_col].to_numpy()

    if len(y) == 0:
        raise ValueError("The signal contains no data.")

    # POST-STEP SIGNAL FOR SMOOTHING
    avg_y = y[step_point:]

    if len(avg_y) == 0:
        raise ValueError(
            f"No signal samples found at or after "
            f"{step_time: .3f} s."
        )

    # SMOOTH POST-STEP SIGNAL
    y_smooth_post = smooth_signal(avg_y, window_size=mean_window)

    # TOLERANCE BAND
    target = (1.0 + step_pu) * voltage_reference
    step_voltage = step_pu * voltage_reference
    tol = tol_factor * step_voltage

    lower = target - tol
    upper = target + tol

    # Create a full-length smoothed signal for plotting
    y_smooth = y.copy()
    y_smooth_post_length = len(y_smooth_post)
    y_smooth[step_point:step_point + y_smooth_post_length] = y_smooth_post

    # Within tolerance band
    within_band = (
        (y_smooth >= lower) &
        (y_smooth <= upper)
    )

    # POST-STEP REGION
    post_step_indices = np.where(
        t >= step_time
    )[0]

    if len(post_step_indices) == 0:
        raise ValueError(
            f"No samples found after step time "
            f"{step_time:.3f} s."
        )

    # Tcr
    rise_idx = None

    for i in post_step_indices:
        if within_band[i]:
            rise_idx = i
            break

    if rise_idx is not None:
        rise_time = t[rise_idx]
        Tcr = rise_time - step_time
    else:
        rise_time = None
        Tcr = None

    # PEAK / Xm
    post_step_y = y_smooth[post_step_indices]
    peak_idx_local = np.argmax(post_step_y)
    peak_idx = post_step_indices[peak_idx_local]
    peak_value = y_smooth[peak_idx]
    peak_time = t[peak_idx]
    Xm = max(0.0, peak_value - target)  # Maximum deviation above target

    # Tcs
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

    # PLOTTING
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(
        t,
        y / voltage_reference,
        label="Udc",
        color="blue"
    )

    ax.plot(
        t,
        y_smooth / voltage_reference,
        linestyle="--",
        label="Udc average"
    )

    # Target
    ax.axhline(
        target / voltage_reference,
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Reference"
    )

    # Tolerance band
    ax.axhline(
        lower / voltage_reference,
        linestyle="-.",
        color="brown",
        linewidth=1.5,
        label="Lower tolerance"
    )

    ax.axhline(
        upper / voltage_reference,
        linestyle=":",
        color="brown",
        linewidth=1.5,
        label="Upper tolerance"
    )

    # Tcr MARKER
    if rise_idx is not None:
        rise_y = y_smooth[rise_idx] / voltage_reference

        ax.scatter(
            rise_time,
            rise_y,
            color="indigo"
        )

        ax.annotate(
            f"Tcr = {Tcr: .3f} s",
            (rise_time, rise_y),
            xytext=(
                rise_time - 0.01,
                rise_y - 0.008
            ),
            arrowprops=dict(arrowstyle="->")
        )

    # Xm MARKER
    peak_y = peak_value / voltage_reference
    ax.scatter(
        peak_time,
        peak_y,
        color="green"
    )

    ax.annotate(
        f"Xm = {Xm: .2f} kV",
        (peak_time, peak_y),
        xytext=(
            peak_time + 0.005,
            peak_y + 0.005
        ),
        arrowprops=dict(arrowstyle="->")
    )

    # Tcs MARKER
    if settling_idx is not None:
        settling_y = (
            y_smooth[settling_idx]
            / voltage_reference
        )

        ax.scatter(
            settling_time,
            settling_y,
            color="violet"
        )

        ax.annotate(
            f"Tcs = {Tcs: .3f} s",
            (settling_time, settling_y),
            xytext=(
                settling_time + 0.01,
                settling_y + 0.007
            ),
            arrowprops=dict(arrowstyle="->")
        )

    # LABELS / FORMAT
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Vdc [p.u.]")
    ax.set_title("Voltage reference step response")
    ax.legend(loc=4)
    ax.grid()
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )
    ax.set_ylim([0.99, 1.03])
    ax.set_xlim([
        step_time - 0.2,
        step_time + 0.5
    ])
    fig.tight_layout()

    # SAVE FIGURE
    fig_path = file_path.parent.parent / "sim_figures"
    fig_name = file_path.stem
    output_path = fig_path / f"{fig_name}.pdf"
    fig_path.mkdir(
        parents=True,
        exist_ok=True
    )
    fig.savefig(
        output_path,
        bbox_inches="tight"
    )
    plt.close(fig)


def analyse_step_down_voltage_signal(
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
        df["L_H"] = float(parts[-4][1:])
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

    lower = target - tol
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


def plot_step_down_voltage_signal(
        file_path,
        time_col="TIME",
        signal_col="Vdc",
        step_time=3,
        step_pu=0.02,
        voltage_reference=640,
        tol_factor=0.05,
        mean_window=100
):
    # LOAD DATA
    file_path = Path(file_path)
    df = pd.read_csv(file_path)

    start_points = df.index[
        df[time_col] >= (step_time - 0.5)
    ]

    step_start_points = df.index[
        df[time_col] >= step_time
    ]

    if len(start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time - 0.5: .3f} s."
        )

    if len(step_start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time: .3f} s."
        )

    # Start analysis 0.5 s before the step
    steady_state_point = df.index.get_loc(start_points[0])
    df = df.iloc[steady_state_point:].copy()

    # Find step location again after slicing df
    step_point = np.searchsorted(
        df[time_col].to_numpy(),
        step_time,
        side="left"
    )

    # SIGNAL
    t = df[time_col].to_numpy()
    y = df[signal_col].to_numpy()

    if len(y) == 0:
        raise ValueError("The signal contains no data.")

    # POST-STEP SIGNAL FOR SMOOTHING
    avg_y = y[step_point:]

    if len(avg_y) == 0:
        raise ValueError(
            f"No signal samples found at or after "
            f"{step_time: .3f} s."
        )

    # SMOOTH POST-STEP SIGNAL
    y_smooth_post = smooth_signal(avg_y, window_size=mean_window)

    # TOLERANCE BAND
    target = (1.0 - step_pu) * voltage_reference
    step_voltage = step_pu * voltage_reference
    tol = tol_factor * step_voltage
    lower = target - tol
    upper = target + tol

    # Create full-length smoothed signal
    y_smooth = y.copy()
    smooth_length = len(y_smooth_post)

    y_smooth[step_point:step_point + smooth_length] = y_smooth_post[:smooth_length]

    # Within tolerance band
    within_band = ((y_smooth >= lower) & (y_smooth <= upper))

    # POST-STEP REGION
    post_step_indices = np.where(t >= step_time)[0]

    if len(post_step_indices) == 0:
        raise ValueError(
            f"No samples found after step time "
            f"{step_time: .3f} s."
        )

    # Tcr
    fall_idx = None

    for i in post_step_indices:
        if within_band[i]:
            fall_idx = i
            break

    if fall_idx is not None:
        fall_time = t[fall_idx]
        Tcr = fall_time - step_time
    else:
        fall_time = None
        Tcr = None

    #  Xm
    post_step_y = y_smooth[post_step_indices]

    min_idx_local = np.argmin(post_step_y)
    min_idx = post_step_indices[min_idx_local]
    min_value = y_smooth[min_idx]
    min_time = t[min_idx]
    Xm = max(0.0, target - min_value)  # Maximum deviation below target

    # Tcs
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

    # PLOTTING
    fig, ax = plt.subplots(figsize=(7, 4))

    ax.plot(
        t,
        y / voltage_reference,
        color="blue",
        label="Vdc"
    )

    ax.plot(
        t,
        y_smooth / voltage_reference,
        linestyle="--",
        label="Vdc average"
    )

    # Target
    ax.axhline(
        target / voltage_reference,
        linestyle="--",
        color="red",
        linewidth=1.5,
        label="Reference"
    )

    # Tolerance band
    ax.axhline(
        lower / voltage_reference,
        linestyle="-.",
        color="brown",
        linewidth=1.5,
        label="Lower tolerance"
    )

    ax.axhline(
        upper / voltage_reference,
        linestyle=":",
        color="brown",
        linewidth=1.5,
        label="Upper tolerance"
    )

    # Tcr MARKER
    if fall_idx is not None:
        fall_y = y_smooth[fall_idx] / voltage_reference

        ax.scatter(
            fall_time,
            fall_y
        )

        ax.annotate(
            f"Tcr = {Tcr: .3f} s",
            (
                fall_time,
                fall_y
            ),
            xytext=(
                fall_time - 0.09,
                fall_y + 0.008
            ),
            arrowprops=dict(arrowstyle="->")
        )

    # Xm MARKER
    min_y = min_value / voltage_reference

    ax.scatter(
        min_time,
        min_y
    )

    ax.annotate(
        f"Xm = {Xm: .2f} kV",
        (
            min_time,
            min_y
        ),
        xytext=(
            min_time + 0.005,
            min_y + 0.006
        ),
        arrowprops=dict(arrowstyle="->")
    )

    # Tcs MARKER
    if settling_idx is not None:
        settling_y = (
            y_smooth[settling_idx]
            / voltage_reference
        )

        ax.scatter(
            settling_time,
            settling_y
        )

        ax.annotate(
            f"Tcs = {Tcs: .3f} s",
            (
                settling_time,
                settling_y
            ),
            xytext=(
                settling_time + 0.01,
                settling_y + 0.007
            ),
            arrowprops=dict(arrowstyle="->")
        )

    # FORMAT PLOT
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Udc [p.u.]")
    ax.set_title("Voltage reference step response")
    ax.yaxis.set_major_formatter(
        FormatStrFormatter("%.3f")
    )

    ax.legend(loc=1)
    ax.set_ylim([
        0.97,
        1.005
    ])

    ax.set_xlim([
        step_time - 0.2,
        step_time + 0.5
    ])

    ax.grid()
    fig.tight_layout()

    # SAVE FIGURE
    fig_path = file_path.parent.parent / "sim_figures"
    fig_name = file_path.stem
    output_path = fig_path / f"{fig_name}.pdf"
    fig_path.mkdir(
        parents=True,
        exist_ok=True
    )
    fig.savefig(
        output_path,
        bbox_inches="tight"
    )
    plt.close(fig)
