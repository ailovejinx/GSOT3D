import numpy as np
import torch
import copy
from pyquaternion import Quaternion
from transforms3d.euler import euler2mat, mat2euler
import torchmetrics.utilities.data
from shapely.geometry import Polygon
from torchmetrics import Metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def cal_3dbb_vertices(center, dim, rot_mat):
    """Calculate 3D bounding box vertices from center, dimension and rotation matrix.
    Args:
      center: Center of the box (3).
      dim: Dimension of the box (3).
      rot_mat: Rotation matrix (3x3).
    Returns:
      Box vertices (8, 3).
    """
    w, l, h = dim / 2
    delta_dim = np.array([
        [l, w, h],
        [l, -w, h],
        [l, -w, -h],
        [l, w, -h],
        [-l, w, h],
        [-l, -w, h],
        [-l, -w, -h],
        [-l, w, -h]
    ])
    vertices = np.dot(delta_dim, rot_mat) + center
    return vertices


def get_rotated_box(box_point_3d, angle):
    """Rotate a box along its vertical axis.
    Args:
      box_point_3d: Input box.
      angle: Rotation angle in rad.
    Returns:
      A rotated box
    """
    res_box_point_3d = copy.deepcopy(box_point_3d)
    roll, pitch, yaw = mat2euler(box_point_3d.rotation_matrix)
    # print(f"roll: {roll}")

    rotated_quan = box_point_3d.orientation * Quaternion(axis=[1, 0, 0], radians=angle)

    res_box_point_3d.orientation = rotated_quan
    roll_real, _, _ = mat2euler(res_box_point_3d.rotation_matrix)
    # print(f"real roll: {roll_real}")

    return res_box_point_3d


def estimateAccuracy(box_a, box_b, dim=3, up_axis=(0, -1, 0)):
    if dim == 3:
        return np.linalg.norm(box_a.center - box_b.center, ord=2)
    elif dim == 2:
        up_axis = np.array(up_axis)
        return np.linalg.norm(
            box_a.center[up_axis == 0] - box_b.center[up_axis == 0], ord=2)


def fromBoxToPoly(box, up_axis=(0, -1, 0)):
    """

    :param box:
    :param up_axis: the up axis must contain only one non-zero component
    :return:
    """
    if up_axis[1] != 0:
        return Polygon(tuple(box.corners()[[0, 2]].T[[0, 1, 5, 4]]))
    elif up_axis[2] != 0:
        return Polygon(tuple(box.bottom_corners().T))


def estimateOverlap(box_a, box_b, dim=2, up_axis=(0, -1, 0)):
    # if box_a == box_b:
    #     return 1.0

    Poly_anno = fromBoxToPoly(box_a, up_axis)
    Poly_subm = fromBoxToPoly(box_b, up_axis)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area

    else:
        up_axis = np.array(up_axis)
        up_max = min(box_a.center[up_axis != 0], box_b.center[up_axis != 0])
        up_min = max(box_a.center[up_axis != 0] - box_a.wlh[2],
                     box_b.center[up_axis != 0] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, up_max[0] - up_min[0])
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]

        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
        return overlap


def fromWaymoBoxToPoly(box):
    return Polygon(tuple(box.corners()[[0, 1]].T[[0, 1, 5, 4]]))


def estimateWaymoOverlap(box_a, box_b, dim=2):

    Poly_anno = fromWaymoBoxToPoly(box_a)
    Poly_subm = fromWaymoBoxToPoly(box_b)

    box_inter = Poly_anno.intersection(Poly_subm)
    box_union = Poly_anno.union(Poly_subm)
    if dim == 2:
        return box_inter.area / box_union.area
    else:
        zmax = min(box_a.center[2], box_b.center[2])
        zmin = max(box_a.center[2] - box_a.wlh[2],
                   box_b.center[2] - box_b.wlh[2])
        inter_vol = box_inter.area * max(0, zmax - zmin)
        anno_vol = box_a.wlh[0] * box_a.wlh[1] * box_a.wlh[2]
        subm_vol = box_b.wlh[0] * box_b.wlh[1] * box_b.wlh[2]
        overlap = inter_vol * 1.0 / (anno_vol + subm_vol - inter_vol)
    return overlap


class TorchPrecision(Metric):
    """Computes and stores the Precision using torchMetrics"""

    def __init__(self, n=21, max_accuracy=2, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_accuracy = max_accuracy
        self.Xaxis = torch.linspace(0, self.max_accuracy, steps=n)
        self.add_state("accuracies", default=[])

    def value(self, accs):
        prec = [
            torch.sum((accs <= thres).float()) / len(accs)
            for thres in self.Xaxis
        ]
        return torch.tensor(prec)

    def update(self, val):
        self.accuracies.append(val)

    def compute(self):
        accs = torchmetrics.utilities.data.dim_zero_cat(self.accuracies)
        if accs.numel() == 0:
            return 0.0
        return torch.trapz(self.value(accs), x=self.Xaxis * 100 / self.max_accuracy)


class TorchSuccess(Metric):
    """Computes and stores the Success using torchMetrics"""

    def __init__(self, n=21, max_overlap=1, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.max_overlap = max_overlap
        self.Xaxis = torch.linspace(0, self.max_overlap, steps=n)
        self.add_state("overlaps", default=[])

    def value(self, overlaps):
        succ = [
            torch.sum((overlaps >= thres).float()) / len(overlaps)
            for thres in self.Xaxis
        ]
        return torch.tensor(succ)

    def compute(self):
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return 0
        return torch.tensor(np.trapz(self.value(overlaps), x=self.Xaxis) * 100 / self.max_overlap)

    def update(self, val):
        self.overlaps.append(val)


class TorchMAO(Metric):
    """Computes and stores the mean Average Overlap (mAO) using torchMetrics"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        # save overlaps and sequence IDs
        self.add_state("overlaps", default=[])
        self.add_state("seq_ids", default=[])

    def update(self, overlaps: torch.Tensor, seq_ids, category_name):
        """
        Args:
            overlaps: 重叠度值的张量
            seq_ids: 对应的序列ID列表
        """
        self.overlaps.append(overlaps)
        self.seq_ids.extend(seq_ids)

    def compute(self):
        # concat all overlaps
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return torch.tensor(0.0)

        # create a dictionary to group overlaps by sequence IDs
        seq_dict = {}
        for overlap, seq_id in zip(overlaps.cpu().numpy(), self.seq_ids):
            if seq_id not in seq_dict:
                seq_dict[seq_id] = []
            seq_dict[seq_id].append(overlap)

        # calculate the mean overlap for each sequence
        sequence_means = []
        for seq_id in seq_dict:
            seq_overlaps = seq_dict[seq_id]
            sequence_means.append(np.mean(seq_overlaps))

        # calculate the mean of the sequence means
        mao = np.mean(sequence_means) if sequence_means else 0.0

        return torch.tensor(mao * 100)  # convert to percentage


class TorchMSR50(Metric):
    """Computes and stores the mean Success Rate at overlap 0.5 (mSR50) using torchMetrics"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = 0.5
        # save overlaps and sequence IDs
        self.add_state("overlaps", default=[])
        self.add_state("seq_ids", default=[])

    def update(self, overlaps: torch.Tensor, seq_ids):
        """
        Args:
            overlaps: 重叠度值的张量
            seq_ids: 对应的序列ID列表
        """
        self.overlaps.append(overlaps)
        self.seq_ids.extend(seq_ids)

    def compute(self):
        # concat all overlaps
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return torch.tensor(0.0)

        # create a dictionary to group overlaps by sequence IDs
        seq_dict = {}
        for overlap, seq_id in zip(overlaps.cpu().numpy(), self.seq_ids):
            if seq_id not in seq_dict:
                seq_dict[seq_id] = []
            seq_dict[seq_id].append(overlap)

        # calculate the Success Rate (SR) for each sequence
        sequence_sr = []
        for seq_id in seq_dict:
            seq_overlaps = np.array(seq_dict[seq_id])

            # calculate the proportion of frames in the sequence with overlap >= threshold
            sr = np.mean(seq_overlaps >= self.threshold)
            sequence_sr.append(sr)

        # calculate the mean of the sequence SRs
        msr = np.mean(sequence_sr) if sequence_sr else 0.0

        return torch.tensor(msr * 100)  # 转换为百分比


class TorchMSR75(Metric):
    """Computes and stores the mean Success Rate at overlap 0.75 (mSR75) using torchMetrics"""

    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.threshold = 0.75
        # save overlaps and sequence IDs
        self.add_state("overlaps", default=[])
        self.add_state("seq_ids", default=[])

    def update(self, overlaps: torch.Tensor, seq_ids):
        """
        Args:
            overlaps: 重叠度值的张量
            seq_ids: 对应的序列ID列表
        """
        self.overlaps.append(overlaps)
        self.seq_ids.extend(seq_ids)

    def compute(self):
        # 合并所有的overlaps
        overlaps = torchmetrics.utilities.data.dim_zero_cat(self.overlaps)

        if overlaps.numel() == 0:
            return torch.tensor(0.0)

        # 将overlaps和seq_ids组合成字典，按序列ID分组
        seq_dict = {}
        for overlap, seq_id in zip(overlaps.cpu().numpy(), self.seq_ids):
            if seq_id not in seq_dict:
                seq_dict[seq_id] = []
            seq_dict[seq_id].append(overlap)

        # 计算每个序列的SR75
        sequence_sr = []
        for seq_id in seq_dict:
            seq_overlaps = np.array(seq_dict[seq_id])
            # 计算单个序列中重叠度大于阈值的帧数占比
            sr = np.mean(seq_overlaps >= self.threshold)
            sequence_sr.append(sr)

        # 计算所有序列SR的平均值
        msr = np.mean(sequence_sr) if sequence_sr else 0.0

        return torch.tensor(msr * 100)  # 转换为百分比


class TorchRuntime(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("sum_runtime", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')
        self.add_state("num_runs", default=torch.tensor(0, dtype=torch.int),
                       dist_reduce_fx='sum')

    def update(self, runtime, n_runs):
        self.sum_runtime += runtime
        self.num_runs += n_runs

    def compute(self):
        return self.sum_runtime / self.num_runs


class TorchNumFrames(Metric):

    def __init__(self):
        super().__init__()
        self.add_state("n_frames", default=torch.tensor(0.0, dtype=torch.float),
                       dist_reduce_fx='sum')

    def update(self, n_frames):
        self.n_frames += n_frames

    def compute(self):
        return self.n_frames
