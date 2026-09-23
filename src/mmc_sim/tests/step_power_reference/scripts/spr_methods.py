import math
from typing import Dict, List
import pandas as pd

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


def configure_rate_limiter(project, component_id: str, ramp_rate: float):
    limiter = project.component(component_id)

    params = {
        "IR": f"{ramp_rate} [1/s]",
        "DR": f"{ramp_rate} [1/s]",
    }

    limiter.parameters(**params)


def configure_ac_grid(project, component_id: str, scr: float, xr: float, mva: float, fn: float, u_ac: float):
    lg, rg = apply_scr_and_xr(scr=scr, xr=xr, ac_voltage=u_ac, sn=mva, fn=fn)

    grid = project.component(component_id)
    grid.parameters(Rg=rg, Lg=lg)


def configure_power_references(components, initial_power, final_power, step_time):
    for comp in components:
        try:
            name = comp.parameters().get("Name")
        except Exception:
            continue

        if name == "Pref_init":
            comp.parameters(Value=initial_power)

        elif name == "Pref_step":
            comp.parameters(Value=final_power)
        elif name == "step_obj":
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

    sim_cfg = "step_power_ref"

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
        "ramp_rate": cfg.get(sim_cfg, "ramp_rate"),
        "scr": cfg.get(sim_cfg, "SCR"),
        "xr": cfg.get(sim_cfg, "XR"),
        "ac_voltage": cfg.get(sim_cfg, "ac_voltage"),
        "mmc_id": cfg.get(sim_cfg, "components", "mmc_id"),
        "rate_limiter_id": cfg.get(sim_cfg, "components", "rate_limiter_id"),
        "ac_grid_id": cfg.get(sim_cfg, "components", "ac_grid_id"),
    }


def single_run(rl_params: Dict[str, float]):
    cfg = load_configuration(CONFIG_FILE)

    model = PSCADModel(
        cfg["project_path"],
        cfg["project_name"],
    )

    project = model.get_project()

    project.component(cfg["mmc_id"]).parameters(idmode="1")
    project.parameters(time_step=cfg["time_step"])
    project.parameters(time_duration=cfg["time_duration"])
    project.parameters(sample_step=cfg["sample_step"])

    configure_rate_limiter(
        project,
        cfg["rate_limiter_id"],
        cfg["ramp_rate"],
    )

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

    configure_power_references(
        canvas_components,
        cfg["initial_power"],
        cfg["final_power"],
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
    summary_df = summarise_results(cfg["save_path"] / "sim_timeseries" / new_file_name, cfg["step_time"],
                                   cfg["final_power"], cfg["mva"])
    summary_path = cfg["save_path"] / "sim_summary" / new_file_name
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path)
    return


def parameter_sweep_run(rl_params: Dict[str, List[float]]):
    cfg = load_configuration(CONFIG_FILE)

    model = PSCADModel(
        cfg["project_path"],
        cfg["project_name"],
    )

    project = model.get_project()

    project.component(cfg["mmc_id"]).parameters(idmode="1")
    project.parameters(time_step=cfg["time_step"])
    project.parameters(time_duration=cfg["time_duration"])
    project.parameters(sample_step=cfg["sample_step"])

    configure_rate_limiter(
        project,
        cfg["rate_limiter_id"],
        cfg["ramp_rate"],
    )

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

    configure_power_references(
        canvas_components,
        cfg["initial_power"],
        cfg["final_power"],
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
        summary_df = summarise_results(cfg["save_path"] / "sim_timeseries" / new_file_name, cfg["step_time"],
                                       cfg["final_power"], cfg["mva"])
        summary_path = cfg["save_path"] / "sim_summary" / new_file_name
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(summary_path)
    return


def summarise_results(
    res_file: str,
    step_time: float,
    p_set: float,
    nominal_mva: float
) -> pd.DataFrame:
    """
    Check the steady-state active power error following a power step.

    Parameters
    ----------
    res_file : str
        Path to the PSCAD result file.
    step_time : float
        Time at which the power step is applied (s).
    p_set : float
        Active power reference (MW).
    nominal_mva : float
        Converter rated power (MVA).

    Returns
    -------
    pandas.DataFrame
        One-row DataFrame containing the test results.
    """

    df = pd.read_csv(res_file)

    # Check required columns
    required_columns = {"TIME", "Pac"}
    missing_columns = required_columns - set(df.columns)

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {', '.join(sorted(missing_columns))}"
        )

    if nominal_mva <= 0:
        raise ValueError("nominal_mva must be greater than zero.")

    # Extract L and R from filename
    filename = Path(res_file).stem
    parts = filename.split("_")

    try:
        L = float(parts[-3][1:])
        R = float(parts[-2][1:])
    except (IndexError, ValueError):
        raise ValueError(
            f"Could not extract R and L from file name: {filename}"
        )

    # Find the first sample 0.75 s after the step
    steady_state_points = df.index[
        df["TIME"] >= (step_time + 0.75)
    ]

    if len(steady_state_points) == 0:
        raise ValueError(
            f"No data found at or after {step_time + 0.75:.3f} s."
        )

    steady_state_idx = df.index.get_loc(steady_state_points[0])

    # Calculate average steady-state active power
    p_ss = df["Pac"].iloc[steady_state_idx:].mean()

    # Steady-state error in pu
    delta_p_ss = (
        abs(abs(p_ss) - abs(p_set)) / nominal_mva
    )

    # Pass/fail criterion
    result = "Pass" if delta_p_ss <= 0.002 else "Fail"

    return pd.DataFrame({
        "L_mH": [L],
        "R_ohms": [R],
        "Pss_pu": [round(abs(p_ss) / nominal_mva, 3)],
        "Delta_Pss": [round(delta_p_ss, 3)],
        "Result": [result]
    })
