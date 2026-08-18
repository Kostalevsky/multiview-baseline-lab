from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.manifold import Isomap


def load_candidate_features(
    candidate_paths,
    image_size=(64, 64),
):

    features = []

    for path in candidate_paths:
        image = Image.open(path).convert("RGB")
        image = image.resize(image_size)

        array = np.asarray(
            image,
            dtype=np.float32,
        )

        array /= 255.0

        features.append(array.reshape(-1))

    return np.stack(features)


def compute_isomap_embedding(features):
    n_neighbors = min(
        5,
        len(features) - 1,
    )

    isomap = Isomap(
        n_neighbors=n_neighbors,
        n_components=2,
    )

    return isomap.fit_transform(features)


def farthest_point_selection(
    points,
    num_points=6,
):
    if num_points > len(points):
        raise ValueError("num_points cannot exceed number of candidates")

    center = points.mean(axis=0)

    first_index = np.argmax(
        np.linalg.norm(
            points - center,
            axis=1,
        )
    )

    selected = [int(first_index)]

    while len(selected) < num_points:
        best_index = None
        best_distance = -1.0

        for i, point in enumerate(points):
            if i in selected:
                continue

            distances = [np.linalg.norm(point - points[j]) for j in selected]

            min_distance = min(distances)

            if min_distance > best_distance:
                best_distance = min_distance
                best_index = i

        selected.append(int(best_index))

    return selected


def select_views(
    candidates_dir,
    num_views=6,
):
    candidates_dir = Path(candidates_dir)

    candidate_paths = sorted(candidates_dir.glob("view_*.png"))

    if len(candidate_paths) != 28:
        raise RuntimeError(f"Expected 28 candidates, " f"got {len(candidate_paths)}.")

    print(f"[CAD2Tech] Loading {len(candidate_paths)} " "candidate views")

    features = load_candidate_features(candidate_paths)

    print(f"[CAD2Tech] Feature matrix: " f"{features.shape}")

    embedding = compute_isomap_embedding(features)

    print(f"[CAD2Tech] Isomap embedding: " f"{embedding.shape}")

    selected_indices = farthest_point_selection(
        embedding,
        num_points=num_views,
    )

    selected_paths = [candidate_paths[index] for index in selected_indices]

    return (
        selected_indices,
        selected_paths,
        embedding,
    )
