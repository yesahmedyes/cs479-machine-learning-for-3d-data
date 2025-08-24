import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class STNKd(nn.Module):
    # T-Net a.k.a. Spatial Transformer Network
    def __init__(self, k: int):
        super().__init__()
        self.k = k
        self.conv1 = nn.Sequential(nn.Conv1d(k, 64, 1), nn.BatchNorm1d(64))
        self.conv2 = nn.Sequential(nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024))

        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, k * k),
        )

    def forward(self, x):
        """
        Input: [B,k,N]
        Output: [B,k,k]
        """
        B = x.shape[0]
        device = x.device
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.max(x, 2)[0]

        x = self.fc(x)

        # Followed the original implementation to initialize a matrix as I.
        identity = (
            Variable(torch.eye(self.k, dtype=torch.float))
            .reshape(1, self.k * self.k)
            .expand(B, -1)
            .to(device)
        )
        x = x + identity
        x = x.reshape(-1, self.k, self.k)
        return x


class PointNetFeat(nn.Module):
    """
    Corresponds to the part that extracts max-pooled features.
    """

    def __init__(
        self,
        input_transform: bool = False,
        feature_transform: bool = False,
    ):
        super().__init__()
        self.input_transform = input_transform
        self.feature_transform = feature_transform

        if self.input_transform:
            self.stn3 = STNKd(k=3)
        if self.feature_transform:
            self.stn64 = STNKd(k=64)

        # point-wise mlp

        self.conv1 = nn.Sequential(nn.Conv1d(3, 64, 1), nn.BatchNorm1d(64), nn.ReLU())

        self.conv2 = nn.Sequential(
            nn.Conv1d(64, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv1d(128, 1024, 1), nn.BatchNorm1d(1024), nn.ReLU()
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - Global feature: [B,1024]
            - Local features: [B, 64, N]
            - Input transformation: [B, 3, 3]
            - Feature transformation: [B, 64, 64]
        """
        x = pointcloud.transpose(2, 1)  # [B, 3, N]

        input_trans = None
        if self.input_transform:
            input_trans = self.stn3(x)  # [B, 3, 3]
            x = torch.bmm(input_trans, x)  # [B, 3, 3] @ [B, 3, N] -> [B, 3, N]

        x = self.conv1(x)  # [B, 64, N]

        feature_trans = None
        if self.feature_transform:
            feature_trans = self.stn64(x)  # [B, 64, 64]
            x = torch.bmm(feature_trans, x)  # [B, 64, 64] @ [B, 64, N] -> [B, 64, N]

        local_features = x

        x = self.conv2(x)  # [B, 128, N]
        x = self.conv3(x)  # [B, 1024, N]

        global_features = torch.max(x, 2)[0]  # [B, 1024]

        return global_features, local_features, input_trans, feature_trans


class PointNetCls(nn.Module):
    def __init__(self, num_classes, input_transform, feature_transform):
        super().__init__()
        self.num_classes = num_classes

        # extracts max-pooled features
        self.pointnet_feat = PointNetFeat(input_transform, feature_transform)

        # returns the final logits from the max-pooled features.
        self.classifier = nn.Sequential(
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - logits [B,num_classes]
            - ...
        """
        global_features, _, input_trans, feature_trans = self.pointnet_feat(pointcloud)

        logits = self.classifier(global_features)

        return logits, input_trans, feature_trans


class PointNetPartSeg(nn.Module):
    def __init__(self, m=50):
        super().__init__()

        self.m = m

        self.pointnet_feat = PointNetFeat(input_transform=True, feature_transform=True)

        self.seg_conv1 = nn.Sequential(
            nn.Conv1d(1088, 512, 1), nn.BatchNorm1d(512), nn.ReLU()
        )

        self.seg_conv2 = nn.Sequential(
            nn.Conv1d(512, 256, 1), nn.BatchNorm1d(256), nn.ReLU()
        )

        self.seg_conv3 = nn.Sequential(
            nn.Conv1d(256, 128, 1), nn.BatchNorm1d(128), nn.ReLU()
        )

        self.seg_conv4 = nn.Conv1d(128, m, 1)

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud: [B,N,3]
        Output:
            - logits: [B,50,N] | 50: # of point labels
            - ...
        """

        B, N, _ = pointcloud.shape

        global_features, local_features, input_trans, feature_trans = (
            self.pointnet_feat(pointcloud)
        )

        # Global features: [B, 1024]
        # Local features: [B, 64, N]

        global_features_expanded = global_features.unsqueeze(2).expand(-1, -1, N)

        concatenated_features = torch.cat(
            [local_features, global_features_expanded], dim=1
        )  # [B, 1088, N]

        x = self.seg_conv1(concatenated_features)  # [B, 512, N]
        x = self.seg_conv2(x)  # [B, 256, N]
        x = self.seg_conv3(x)  # [B, 128, N]

        logits = self.seg_conv4(x)  # [B, 50, N]

        return logits, input_trans, feature_trans


class PointNetAutoEncoder(nn.Module):
    def __init__(self, num_points):
        super().__init__()

        self.pointnet_feat = PointNetFeat()

        # Decoder is just a simple MLP that outputs N x 3 (x,y,z) coordinates.
        self.decoder = nn.Sequential(
            nn.Linear(1024, num_points // 4),
            nn.BatchNorm1d(num_points // 4),
            nn.ReLU(),
            nn.Linear(num_points // 4, num_points // 2),
            nn.BatchNorm1d(num_points // 2),
            nn.ReLU(),
            nn.Linear(num_points // 2, num_points),
            nn.BatchNorm1d(num_points),
            nn.Linear(num_points, num_points * 3),
        )

    def forward(self, pointcloud):
        """
        Input:
            - pointcloud [B,N,3]
        Output:
            - reconstructed_pointcloud [B,N,3]
            - input_trans [B,3,3] or None
            - feat_trans [B,64,64] or None
        """
        B, N, _ = pointcloud.shape

        global_features, _, _, _ = self.pointnet_feat(pointcloud)

        reconstructed_flat = self.decoder(global_features)  # [B, N*3]

        reconstructed_pointcloud = reconstructed_flat.view(B, N, 3)  # [B, N, 3]

        return reconstructed_pointcloud


def get_orthogonal_loss(feat_trans, reg_weight=1e-3):
    """
    a regularization loss that enforces a transformation matrix to be a rotation matrix.
    Property of rotation matrix A: A*A^T = I
    """
    if feat_trans is None:
        return 0

    B, K = feat_trans.shape[:2]
    device = feat_trans.device

    identity = torch.eye(K).to(device)[None].expand(B, -1, -1)
    mat_square = torch.bmm(feat_trans, feat_trans.transpose(1, 2))

    mat_diff = (identity - mat_square).reshape(B, -1)

    return reg_weight * mat_diff.norm(dim=1).mean()
