"""
Polynomial Chaos Expansion based sensitivity analysis.
This analysis is implemented based on a pre-defined uncertainty parameter distribution based on the sample data problem.
See `create_distribution()' in `sa_methods.py`. It should be updated to match the sample data problem.
The EMT simulation summary, containing the KPIs should be concatenated into a single csv file whose location is included
in the config file. The sensitivity of each KPI is evaluated separately, so update as required.
"""

from sklearn.model_selection import train_test_split

from sa_methods import load_dataset, Path, load_config_file, create_distribution, train_pce, evaluate_model, \
    calculate_sobol_indices, pce_logger, CONFIG_FILE


# Main analysis pipeline
def run_analysis(config_file: Path, test_kpi: str):
    """
    Each KPI ("Tcr", "Tcs", "Xm") is evaluated separately. So, update accordingly.
    """
    cfg = load_config_file(config_file)
    sample_data = Path(cfg["sensitivity_analysis"]["output"]["sample_data"])
    dataset = load_dataset(sample_data, config_file)

    # Inputs
    X = dataset[["L", "C", "R", "SCR", "XR"]].values

    # Output response
    Y = dataset[test_kpi].values

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=13)

    distribution = create_distribution()

    model = train_pce(X_train, Y_train, order=3, distribution=distribution)
    metrics = evaluate_model(model, X_train, X_test, Y_train, Y_test)

    sobol = calculate_sobol_indices(model, distribution)
    pce_logger.info(f"Accuracy: {metrics}")
    return {
        "model": model,
        "metrics": metrics,
        "sobol": sobol
    }


if __name__ == "__main__":

    kpi = "Xm"  # "Tcr", "Tcs", "Xm"  Update as necessary.

    results_summary = run_analysis(CONFIG_FILE, kpi)
    print(results_summary["sobol"])
