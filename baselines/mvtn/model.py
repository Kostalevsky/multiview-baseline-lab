from itertools import pairwise

import torch
from torch import nn


def mlp(channels, *, activate=True, normalize=True, dropout=0.5):
    layers = []
    for source, target in pairwise(channels):
        layers.append(nn.Linear(source, target))
        if normalize:
            layers.append(nn.BatchNorm1d(target))
        if activate:
            layers.append(nn.ReLU())
        if dropout:
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


def conv1d(channels, *, activate=True, normalize=True):
    layers = []
    for source, target in pairwise(channels):
        layers.append(nn.Conv1d(source, target, 1))
        if normalize:
            layers.append(nn.BatchNorm1d(target))
        if activate:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class Transformation(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.dimensions = dimensions
        self.convs = nn.Sequential(
            conv1d([dimensions, 64]),
            conv1d([64, 128]),
            conv1d([128, 1024], activate=False, normalize=False),
        )
        self.fcs = nn.Sequential(
            conv1d([1024, 512]),
            conv1d([512, 256]),
            conv1d([256, dimensions * dimensions], activate=False, normalize=False),
        )

    def forward(self, points):
        features = self.convs(points).max(dim=2, keepdim=True).values
        matrix = self.fcs(features).reshape(-1, self.dimensions, self.dimensions)
        return matrix + torch.eye(
            self.dimensions, device=points.device, dtype=points.dtype
        ).unsqueeze(0)


class PointNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_transform = Transformation(3)
        self.conv1 = conv1d([3, 64])
        self.feature_transform = Transformation(64)
        self.conv2s = conv1d([64, 128, 1024])
        self.classifier = mlp([1024, 512, 256, 40])

    def forward(self, points):
        points = torch.bmm(self.input_transform(points), points)
        features = self.conv1(points)
        features = torch.bmm(self.feature_transform(features), features)
        global_features = self.conv2s(features).max(dim=2).values
        return self.classifier(global_features)


class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fe_model = PointNet()

    def forward(self, points):
        return self.fe_model(points.transpose(1, 2))


class LearnedOffsetViewSelector(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.num_views = num_views
        self.view_transformer = nn.Sequential(
            mlp([40 + 2 * num_views, 40, 40, 5 * num_views, 2 * num_views]),
            mlp(
                [2 * num_views, 2 * num_views],
                activate=False,
                normalize=False,
                dropout=0,
            ),
            nn.Tanh(),
        )
        self.register_buffer("views_azim", torch.zeros(num_views))
        self.register_buffer("views_elev", torch.zeros(num_views))
        self.register_buffer("views_dist", torch.zeros(num_views))

    def forward(self, features):
        azimuth = self.views_azim.expand(len(features), -1)
        elevation = self.views_elev.expand(len(features), -1)
        distance = self.views_dist.expand(len(features), -1)
        offsets = self.view_transformer(
            torch.cat([features, azimuth, elevation], dim=1)
        )
        azimuth_offset, elevation_offset = offsets.chunk(2, dim=1)
        return (
            azimuth + azimuth_offset * 180.0 / self.num_views,
            elevation + elevation_offset * 90.0,
            distance,
        )


class ViewSelector(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.chosen_view_selector = LearnedOffsetViewSelector(num_views)

    def forward(self, features):
        return self.chosen_view_selector(features)


class OffsetMVTN(nn.Module):
    def __init__(self, num_views):
        super().__init__()
        self.view_selector = ViewSelector(num_views)
        self.feature_extractor = FeatureExtractor()

    def forward(self, points):
        return self.view_selector(self.feature_extractor(points))
