import mhi.pscad


class PSCADModel:

    def __init__(
            self,
            project_path,
            project_name
    ):
        self.app = mhi.pscad.launch(settings={"ws_restore": False}, version="5.1.0")
        self.app.settings(
            fortran_version="GFortran 13.2 (64-bit)",
        )

        self.app.load(
            str(project_path)
        )

        self.project = (
            self.app.project(project_name)
        )

        self.project.focus()

    def set_output(self, name):
        self.project.parameters(
            output_filename=f"{name}.out"
        )

    def canvas_components(self):
        return self.project.canvas(
            "Main"
        ).components()

    def get_project(self):
        return self.project
