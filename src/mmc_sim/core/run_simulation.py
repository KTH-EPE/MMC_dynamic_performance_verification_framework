from pathlib import Path
from mhi.pscad.utilities.file import OutFile


class RunSimulation:

    def __init__(
            self,
            model
    ):
        self.model = model

    def run(
            self,
            result_file
    ):
        self.model.project.run()

        outfile = OutFile(
            str(result_file)
        )

        outfile.toCSV()

        return Path(
            f"{result_file}.csv"
        )
