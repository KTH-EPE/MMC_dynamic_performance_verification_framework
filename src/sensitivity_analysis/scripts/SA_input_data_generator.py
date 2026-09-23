"""
Sensitivity analysis sample generation.
This analysis has been implemented with a pre-defined sample problem (see function in `sa_methods.py`:
`define_sampling_problem()`. It should be updated for different sample sets.
"""

from sa_methods import Path, load_config_file, generate_samples, save_samples


# Main execution
def run_sampling(config_file: Path, samples: int = 400):
    cfg = load_config_file(config_file)
    output_directory = Path(cfg["sensitivity_analysis"]["output"]["sample_data_dir"])
    dataframe = generate_samples(number_of_samples=samples, random_seed=13)
    save_samples(dataframe, output_directory)
    return dataframe


if __name__ == "__main__":
    cfg_file = Path("SA_config.yaml")
    samples_no = 400
    data_samples = run_sampling(cfg_file, samples_no)
    print(data_samples.head())
