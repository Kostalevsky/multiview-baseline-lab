import math

NUM_SHELLS = 3
VIEWS_PER_SHELL = 40

SHELL_RADII = [
    2.5,
    3.0,
    3.5,
]

MIN_ELEVATION_DEG = 10.0
MAX_ELEVATION_DEG = 80.0


def generate_hemisphere_views():
    """
    Generate 120 deterministic camera poses distributed across
    three hemispherical shells.

    Each shell contains 40 cameras.

    Returns
    -------
    list[dict]
        Camera metadata dictionaries containing:
        - index
        - shell
        - radius
        - azimuth
        - elevation
        - camera_position
    """

    views = []

    index = 0

    for shell_index, radius in enumerate(SHELL_RADII):
        for local_index in range(VIEWS_PER_SHELL):
            fraction = (local_index + 0.5) / VIEWS_PER_SHELL

            elevation = MIN_ELEVATION_DEG + fraction * (
                MAX_ELEVATION_DEG - MIN_ELEVATION_DEG
            )

            golden_angle = 137.50776405003785

            azimuth = (local_index * golden_angle + shell_index * 17.0) % 360.0

            azimuth_rad = math.radians(azimuth)

            elevation_rad = math.radians(elevation)

            horizontal_radius = radius * math.cos(elevation_rad)

            x = horizontal_radius * math.sin(azimuth_rad)

            y = -horizontal_radius * math.cos(azimuth_rad)

            z = radius * math.sin(elevation_rad)

            views.append(
                {
                    "index": index,
                    "shell": shell_index,
                    "radius": radius,
                    "azimuth": azimuth,
                    "elevation": elevation,
                    "camera_position": [
                        x,
                        y,
                        z,
                    ],
                }
            )

            index += 1

    if len(views) != 120:
        raise RuntimeError(
            "VIN-NBV candidate generator "
            f"produced {len(views)} views "
            "instead of 120."
        )

    return views
