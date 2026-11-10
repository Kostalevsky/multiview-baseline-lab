import math

from baselines.fixed_views import render_fixed_views


class GIFTFibonacciBaseline:
    name = "gift_fibonacci6"

    def run(self, input_path, output_dir=None):
        count = 6
        golden_angle = math.pi * (3 - math.sqrt(5))
        views = []
        for index in range(count):
            z = 1 - 2 * (index + 0.5) / count
            azimuth = (index * golden_angle) % (2 * math.pi)
            elevation = math.asin(z)
            views.append(
                {
                    "index": index,
                    "azimuth_deg": math.degrees(azimuth),
                    "elevation_deg": math.degrees(elevation),
                    "radius": 3.0,
                    "image": f"selected/view_{index:02d}.png",
                }
            )
        return render_fixed_views(
            self.name,
            views,
            input_path,
            output_dir,
            details={
                "method": "GIFT-inspired RGB camera baseline",
                "sampling": "midpoint spherical Fibonacci, golden angle",
                "coordinate_frame": "Blender world Z-up after OBJ import",
                "normalization": "world bounding-box center, maximum extent 1",
                "sources": {
                    "gift_section_2_1": "https://songbai.site/files/GIFT_CVPR.pdf",
                    "spherical_fibonacci": "https://doi.org/10.1145/2816795.2818131",
                },
                "adaptations": [
                    "6 views instead of the 64 views used in GIFT experiments",
                    "Fibonacci sampling is our choice for spherical camera placement",
                    "laboratory RGB perspective rendering and bounding-box normalization",
                ],
            },
        )
