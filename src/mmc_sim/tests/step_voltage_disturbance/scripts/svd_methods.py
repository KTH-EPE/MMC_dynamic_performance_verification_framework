import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def configure_power_and_voltage_values(components, initial_power, step_power, uref, u_step, step_time):
    for comp in components:
        try:
            name = comp.parameters().get("Name")
        except Exception:
            continue

        if name == "Pref_init":
            comp.parameters(Value=initial_power)

        elif name == "Pref_step":
            comp.parameters(Value=step_power)
        elif name == "u_step":
            comp.parameters(Value=u_step)
        elif name == "uref":
            comp.parameters(Value=uref)
        elif name == "step_obj1":
            comp.parameters(X=step_time)
        elif name == "step_obj2":
            comp.parameters(X=step_time)


def find_dc_components(components):
    required = {
        "R_dc": "R",
        "L_dc": "L",
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

    sim_cfg = "step_voltage_dist"

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
        "initial_power": cfg.get(sim_cfg, "initial_power"),
        "final_power": cfg.get(sim_cfg, "final_power"),
        "uref": cfg.get(sim_cfg, "voltage_reference"),
        "u_step": cfg.get(sim_cfg, "voltage_step"),
        "scr": cfg.get(sim_cfg, "SCR"),
        "xr": cfg.get(sim_cfg, "XR"),
        "ac_voltage": cfg.get(sim_cfg, "ac_voltage"),
        "mmc_id": cfg.get(sim_cfg, "components", "mmc_id"),
        "ac_grid_id": cfg.get(sim_cfg, "components", "ac_grid_id"),
    }


def single_run(rl_params: Dict[str, float], plot_results: bool = True):
    cfg = load_configuration(CONFIG_FILE)

    model = PSCADModel(
        cfg["project_path"],
        cfg["project_name"],
    )

    project = model.get_project()

    project.component(cfg["mmc_id"]).parameters(idmode="2")
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

    configure_power_and_voltage_values(
        canvas_components,
        cfg["initial_power"],
        cfg["final_power"],
        cfg["uref"],
        cfg["u_step"],
        cfg["step_time"]
    )

    dc_grid_components = find_dc_components(canvas_components)

    dc_network = ConfigDCGridComponents(dc_grid_components)
    dc_network.set_dc_network(**rl_params)
    logger.info(f"Running simulation for {rl_params}")
    simulation = RunSimulation(model)
    result_df = simulation.run(cfg["result_file"])
    new_file_name = format_rl_filename(**rl_params, file_name=cfg["output_file"])
    move_result_file(result_df, cfg["save_path"] / "sim_timeseries", new_file_name)
    result_summary, timeseries_df = analyse_step_response(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                          cfg["final_power"], step_time=cfg["step_time"])
    summary_path = cfg["save_path"] / "sim_summary" / new_file_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    result_summary.to_csv(summary_path)
    if plot_results:
        result_fig = plot_step_response(timeseries_df, cfg["step_time"], result_summary)
        fig_path = cfg["save_path"] / "sim_figures" / (Path(new_file_name).stem + ".pdf")
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        result_fig.savefig(fig_path)
        plt.close(result_fig)
    return


def parameter_sweep_run(rl_params: Dict[str, List[float]], plot_results: bool = True):
    cfg = load_configuration(CONFIG_FILE)

    model = PSCADModel(
        cfg["project_path"],
        cfg["project_name"],
    )

    project = model.get_project()

    project.component(cfg["mmc_id"]).parameters(idmode="2")
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

    configure_power_and_voltage_values(
        canvas_components,
        cfg["initial_power"],
        cfg["final_power"],
        cfg["uref"],
        cfg["u_step"],
        cfg["step_time"]
    )

    dc_grid_components = find_dc_components(canvas_components)
    dc_grid_parameters = ParameterSweep(rl_params)
    dc_network = ConfigDCGridComponents(dc_grid_components)

    for params in dc_grid_parameters.combinations():
        dc_network.set_dc_network(
            **params
        )

        simulation = RunSimulation(model)
        logger.info(f"Running simulation for {params}")
        result_df = simulation.run(cfg["result_file"])
        new_file_name = format_rl_filename(**params, file_name=cfg["output_file"])
        move_result_file(result_df, cfg["save_path"] / "sim_timeseries", new_file_name)
        result_summary, timeseries_df = analyse_step_response(cfg["save_path"] / "sim_timeseries" / new_file_name,
                                                              cfg["final_power"], step_time=cfg["step_time"])
        summary_path = cfg["save_path"] / "sim_summary" / new_file_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        result_summary.to_csv(summary_path)
        if plot_results:
            result_fig = plot_step_response(timeseries_df, cfg["step_time"], result_summary)
            fig_path = cfg["save_path"] / "sim_figures" / (Path(new_file_name).stem + ".pdf")
            fig_path.parent.mkdir(parents=True, exist_ok=True)
            result_fig.savefig(fig_path)
            plt.close(result_fig)
    return


def analyse_step_response(
        res_file: str,
        pref: float,
        signal_name: str = "Pac",
        step_time: float = 3.0
):
    """
    Analyse the active power step response.

    Parameters
    ----------
    res_file : str
        Path to the PSCAD result file.
    pref : float
        Final active power reference (MW).
    signal_name : str
        Name of the AC power column in the result DataFrame.
    step_time : float
        Time at which the power step is applied.

    Returns
    -------
    summary_df : pandas.DataFrame
        One-row DataFrame containing the extracted metrics.
    p_df : pandas.DataFrame
        Full simulation results.
    """

    # LOAD DATA
    p_df = pd.read_csv(res_file)

    # CHECK REQUIRED COLUMNS
    if "TIME" not in p_df.columns:
        raise ValueError(
            "The result file does not contain a 'TIME' column."
        )

    if signal_name not in p_df.columns:
        raise ValueError(
            f"The result file does not contain the "
            f"'{signal_name}' column."
        )

    # EXTRACT PARAMETERS FROM FILE NAME
    filename = Path(res_file).stem

    try:
        R = float(filename.split("_")[-2][1:])
        L = float(filename.split("_")[-3][1:])
    except (IndexError, ValueError):
        raise ValueError(
            f"Could not extract R and L from file name: "
            f"{filename}"
        )

    # SIGNAL
    pac_col = p_df[signal_name]

    start_points = p_df.index[
        p_df["TIME"] >= (step_time - 0.3)
    ]

    if len(start_points) == 0:
        raise ValueError(
            f"No data found at or after "
            f"{step_time - 0.3:.3f} s."
        )

    steady_state_point = p_df.index.get_loc(
        start_points[0]
    )

    search_region = pac_col.iloc[steady_state_point:]

    # 50% AND 90% RESPONSE LEVELS
    pac = -pref
    if pac > 0:
        p50_candidates = search_region[
            search_region >= 0.5 * pac
        ]

        p90_candidates = search_region[
            search_region >= 0.9 * pac
        ]
    else:
        p50_candidates = search_region[
            search_region <= 0.5 * pac
        ]
        p90_candidates = search_region[
            search_region <= 0.9 * pac
        ]

    # T50
    if not p50_candidates.empty:
        p50_index = p50_candidates.index[0]
        t50 = round(p_df.loc[p50_index, "TIME"] - step_time, 3)
    else:
        t50 = np.nan

    # T90
    if not p90_candidates.empty:
        p90_index = p90_candidates.index[0]
        t90 = round(p_df.loc[p90_index, "TIME"] - step_time, 3)
    else:
        t90 = np.nan

    # PASS / FAIL
    result = "Fail"
    if (
        pd.notna(t50)
        and pd.notna(t90)
        and t50 <= 0.05
        and t90 <= 0.3
    ):
        result = "Pass"

    # SUMMARY
    summary_df = pd.DataFrame({
        "L_H": [L],
        "R_ohms": [R],
        "t50_s": [t50],
        "t90_s": [t90],
        "Result": [result]
    })
    return summary_df, p_df


def plot_step_response(
        p_df,
        step_time,
        summary_df,
        signal_col="Pac",
):
    """
    Plot the active power step response highlighting the 50% and 90% response times.

    Parameters
    ----------
    p_df : pandas.DataFrame
        Simulation results.
    step_time : float
        Time at which the power step is applied (s).
    summary_df : pandas.DataFrame
        Output from analyse_step_response().
    signal_col : str, optional
        Name of the signal to plot.

    Returns
    -------
    matplotlib.figure.Figure
        Figure containing the step response plot.
    """

    fig, ax = plt.subplots(figsize=(6, 4))

    time = p_df["TIME"]
    signal = p_df[signal_col]

    # Response times
    t50 = summary_df.loc[0, "t50_s"] + step_time
    t90 = summary_df.loc[0, "t90_s"] + step_time

    # Plot window
    mask = (time >= step_time - 0.5) & (time <= step_time + 0.5)

    ax.plot(time, signal, label=signal_col)

    ax.axvline(
        t50,
        linestyle="--",
        linewidth=1.2,
        label="50% provision",
        color="salmon"
    )

    ax.axvline(
        t90,
        linestyle="--",
        linewidth=1.2,
        label="90% provision",
        color="red"
    )

    ax.set_xlim(step_time - 0.2, step_time + 0.5)
    ax.set_ylim(-2, abs(signal.loc[mask]).max() * 1.1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pac (MW)")
    ax.grid(True, linestyle="--")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig
