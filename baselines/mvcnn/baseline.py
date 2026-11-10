from baselines.fixed_views import render_fixed_views


class MVCNNViewsBaseline:
    def __init__(self, name):
        steps = {"mvcnn12": 1, "mvcnn6": 2}
        self.name = name
        self.indices = range(0, 12, steps[name])

    def run(self, input_path, output_dir=None):
        views = [
            {
                "index": index,
                "azimuth_deg": 30 * index,
                "elevation_deg": 30,
                "radius": 3.0,
                "image": f"selected/view_{index:02d}.png",
            }
            for index in self.indices
        ]
        return render_fixed_views(self.name, views, input_path, output_dir)
