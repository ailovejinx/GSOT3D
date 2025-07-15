import os.path as osp
import pandas as pd
import pickle as pkl
import numpy as np
from pyquaternion import Quaternion
from tqdm import tqdm
import bisect
import torch

from .utils import *
from .base_dataset import BaseDataset, EvalDatasetWrapper
from utils import pl_ddp_rank
import random


def sample_data(data_list, percentage=0.1, seed=42):
    # randomly get 10% of the data
    random.seed(seed)
    sample_size = int(len(data_list) * percentage)
    return random.sample(data_list, sample_size)


class GSOT3DMem(BaseDataset):

    def __init__(self, split_type, cfg, log):
        super().__init__(split_type, cfg, log)

        assert cfg.category_name in [
            'backpack', 'balloon', 'basketball', 'basketball_player', 'bench',
            'bicycle', 'book', 'bottle', 'bus', 'can',
            'car', 'cat', 'chair', 'charger', 'computer_case',
            'cup', 'cyclist', 'displayer', 'dog', 'doll',
            'earphone', 'electric_bike', 'electro-tricycle', 'fire_extinguisher', 'football',
            'football_player', 'frisbee', 'gamepad', 'handbag', 'handcart',
            'headwear', 'instant_noodles', 'keyboard', 'laptop', 'medicine_box',
            'melon', 'mineral_water', 'motorbike', 'pedestrian', 'pillow',
            'printer', 'shampoo', 'shoes', 'sign', 'sink',
            'stool', 'suitcase', 'table', 'tools', 'toy_transportation',
            'trash_bin', 'umbrella', 'van', 'warning_post', 'All'
        ]
        # read train/test split
        train_split = osp.join(cfg.data_root_dir, 'unseen_train_set.txt')
        test_split = osp.join(cfg.data_root_dir, 'unseen_test_set.txt')
        with open(train_split, 'r') as f:
            train_scene_ids = f.readlines()
            train_scene_ids = [train_scene.strip() for train_scene in train_scene_ids]
        with open(test_split, 'r') as f:
            test_scene_ids = f.readlines()
            test_scene_ids = [test_scene.strip() for test_scene in test_scene_ids]

        if cfg.scale == 'FULL':
            split_type_to_scene_ids = dict(
                train=train_scene_ids,
                val=test_scene_ids,
                test=test_scene_ids
            )
        elif cfg.scale == 'TINY':
            train_scene_ids_sampled = sample_data(train_scene_ids, percentage=0.3)
            test_scene_ids_sampled = sample_data(test_scene_ids, percentage=0.3)
            split_type_to_scene_ids = dict(
                train=train_scene_ids_sampled,
                val=test_scene_ids_sampled,
                test=test_scene_ids_sampled
            )
        else:
            print(f'scale must be FULL or TINY, got{cfg.scale} instead')
            raise NotImplementedError

        self.preload_offset = cfg.train_cfg.preload_offset if split_type == 'train' else cfg.eval_cfg.preload_offset
        self.cache = cfg.train_cfg.cache if split_type == 'train' else cfg.eval_cfg.cache
        self.calibration_info = {}

        scene_ids = split_type_to_scene_ids[split_type]
        self.tracklet_annotations = self._build_tracklet_annotations(scene_ids)

        if self.cache:
            if not cfg.debug:
                cache_file_dir = osp.join(
                    f'GSOT3D_{cfg.scale}_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            else:
                cache_file_dir = osp.join(
                    f'GSOT3D_{cfg.scale}_DEBUG_{self.cfg.category_name}_{split_type}_{self.cfg.coordinate_mode}_{self.preload_offset}.cache')
            if osp.exists(cache_file_dir):
                self.log.info(f'Loading data from cache file {cache_file_dir}')
                with open(cache_file_dir, 'rb') as f:
                    tracklets = pkl.load(f)
            else:
                tracklets = []
                for tracklet_id in tqdm(range(len(self.tracklet_annotations)), desc='[%6s]Loading pcds ' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
                    frames = []
                    for frame_anno in self.tracklet_annotations[tracklet_id]:
                        frames.append(self._build_frame(frame_anno))

                    comp_template_pcd = merge_template_pcds(
                        [frame['pcd'] for frame in frames],
                        [frame['bbox'] for frame in frames],
                        offset=cfg.target_offset,
                        scale=cfg.target_scale
                    )
                    if self.preload_offset > 0:
                        for frame in frames:
                            frame['pcd'] = crop_pcd_axis_aligned(
                                frame['pcd'], frame['bbox'], offset=self.preload_offset)

                    tracklets.append({
                        'comp_template_pcd': comp_template_pcd,
                        'frames': frames
                    })

                with open(cache_file_dir, 'wb') as f:
                    self.log.info(
                        f'Saving data to cache file {cache_file_dir}')
                    pkl.dump(tracklets, f)
            self.tracklets = tracklets
        else:
            self.tracklets = None

        if split_type == 'train':
            if self.tracklets:
                self.tracklet_annotations, self.tracklets = self.filter_tracklet(
                    self.tracklet_annotations, self.tracklets, cfg.num_smp_frames_per_tracklet)

        self.tracklet_num_frames = [len(tracklet_anno)
                                    for tracklet_anno in self.tracklet_annotations]
        self.tracklet_st_frame_id = []
        self.tracklet_ed_frame_id = []
        last_ed_frame_id = 0
        for num_frames in self.tracklet_num_frames:
            assert num_frames > 0
            self.tracklet_st_frame_id.append(last_ed_frame_id)
            last_ed_frame_id += num_frames
            self.tracklet_ed_frame_id.append(last_ed_frame_id)

    def filter_tracklet(self, t_annos, ts, min_tracklet_length=0):
        t_annos_new, ts_new = [], []
        for t_anno, t in zip(t_annos, ts):
            if len(t_anno) >= min_tracklet_length:
                t_annos_new.append(t_anno)
                ts_new.append(t)
        return t_annos_new, ts_new

    def get_dataset(self):
        if self.split_type == 'train':
            return TrainDatasetWrapper(self, self.cfg, self.log)
        else:
            return EvalDatasetWrapper(self, self.cfg, self.log)

    def num_frames(self):
        return self.tracklet_ed_frame_id[-1]

    def num_tracklets(self):
        return len(self.tracklet_annotations)

    def num_tracklet_frames(self, tracklet_id):
        return self.tracklet_num_frames[tracklet_id]

    def get_frame(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.tracklets[tracklet_id]['frames'][frame_id]
            return frame
        else:
            frame_anno = self.tracklet_annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno)
            if self.preload_offset > 0:
                frame['pcd'] = crop_pcd_axis_aligned(
                    frame['pcd'], frame['bbox'], offset=self.preload_offset)
            return frame

    def get_frame_bbox(self, tracklet_id, frame_id):
        if self.tracklets:
            frame = self.tracklets[tracklet_id]['frames'][frame_id]
        else:
            frame_anno = self.tracklet_annotations[tracklet_id][frame_id]
            frame = self._build_frame(frame_anno)
        return frame['bbox']

    def get_comp_pcd(self, tracklet_id):
        comp_template_pcd = self.tracklets[tracklet_id]['comp_template_pcd']
        return comp_template_pcd

    def get_tracklet_frame_id(self, idx):
        tracklet_id = bisect.bisect_right(
            self.tracklet_ed_frame_id, idx)
        assert self.tracklet_st_frame_id[
            tracklet_id] <= idx and idx < self.tracklet_ed_frame_id[tracklet_id]
        frame_id = idx - \
            self.tracklet_st_frame_id[tracklet_id]
        return tracklet_id, frame_id

    def _build_tracklet_annotations(self, scene_ids):
        tracklet_annotations = []
        for scene_id in tqdm(
                scene_ids, desc='[%6s]Loading annos' % self.split_type.upper(), disable=pl_ddp_rank() != 0):
            annotation_file_dir = osp.join(
                self.cfg.data_root_dir, f'data/{scene_id}/gt.txt')
            data = pd.read_csv(
                annotation_file_dir,
                sep=' ',
                names=[
                    "classname", "pointN", "out-of-view", "occlusion",
                    "length", "width", "height",
                    "center_x", "center_y", "center_z",
                    "roll", "pitch", "yaw"
                ]
            )
            data.insert(0, 'frame', range(1, len(data) + 1))  # add frame id
            if self.cfg.category_name == 'All':
                # we don't have 'DontCare'
                data = data
            else:
                data = data[data["classname"] == self.cfg.category_name]

            data.insert(loc=0, column='scene', value=scene_id)

            data_tracklet = data.copy()
            data_tracklet = data_tracklet.reset_index(drop=True)
            tracklet_anno = [anno for index, anno in data_tracklet.iterrows()]
            tracklet_annotations.append(tracklet_anno)
        return tracklet_annotations

    @staticmethod
    def _read_calibration_file(filepath):
        """Read in a calibration file and parse into a dictionary."""
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                values = line.split()
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[values[0]] = np.array(
                        [float(x) for x in values[1:]]).reshape(3, 4)
                except ValueError:
                    pass
        return data

    def _build_frame(self, frame_anno):
        scene_id = frame_anno['scene']
        frame_id = frame_anno['frame']
        if scene_id in self.calibration_info:
            calib = self.calibration_info[scene_id]  # TODO
        else:
            calib = self._read_calibration_file(
                osp.join(self.cfg.data_root_dir, f'data/{scene_id}/calib.json'))
            self.calibration_info[scene_id] = calib
        # velo_to_cam = np.vstack((calib["Tr_velo_cam"], np.array([0, 0, 0, 1])))   TODO

        assert self.cfg.coordinate_mode in ['camera', 'velodyne']

        if self.cfg.coordinate_mode == 'camera':  # TODO
            bbox_center = [frame_anno["x"], frame_anno["y"] -
                           frame_anno["height"] / 2, frame_anno["z"]]
            size = [frame_anno["width"],
                    frame_anno["length"], frame_anno["height"]]
            orientation = Quaternion(
                axis=[0, 1, 0], radians=frame_anno["rotation_y"]) * Quaternion(
                axis=[1, 0, 0], radians=np.pi / 2)
            bbox = BoundingBox(bbox_center, size, orientation)
        else:
            # coordinate is in velodyne, which is different from kitti(camera coordinate)
            box_center_velo = np.array(
                [frame_anno["center_x"], frame_anno["center_y"], frame_anno["center_z"]])
            size = [frame_anno["width"], frame_anno["length"], frame_anno["height"]]
            # orientation = Quaternion(
            #     axis=[0, 0, -1], radians=frame_anno["yaw"])
            orientation = (Quaternion(axis=[0, 0, -1], radians=frame_anno["yaw"]) *
                           Quaternion(axis=[0, -1, 0], radians=frame_anno["pitch"]) *
                           Quaternion(axis=[-1, 0, 0], radians=frame_anno["roll"]))
            bbox = BoundingBox(box_center_velo, size, orientation)
        try:
            pcd_file_dir = osp.join(
                self.cfg.data_root_dir, f'data/{scene_id}/point_cloud_bin/{str(frame_id).zfill(5)}.bin')
            pcd = PointCloud(np.fromfile(
                pcd_file_dir, dtype=np.float32).reshape(-1, 4).T)
            if self.cfg.coordinate_mode == 'camera':
                pass
                # pcd.transform(velo_to_cam)    TODO
        except Exception as e:
            # in case the Point cloud is missing
            pcd = PointCloud(np.array([[0, 0, 0]]).T)
            print(f'Error: {e}')

        return {'pcd': pcd, 'bbox': bbox, 'anno': frame_anno}


def print_np(**kwargs):
    for k, v in kwargs.items():
        print(k, np.concatenate((v[:5], v[-5:]), axis=0))


class TrainDatasetWrapper(torch.utils.data.Dataset):
    def __init__(self, dataset: BaseDataset, cfg, log):
        super().__init__()
        self.dataset = dataset
        self.cfg = cfg
        self.log = log

    def __len__(self):
        # return 100
        return self.dataset.num_frames()

    def _generate_item(self, frames, prev_frame_bboxes):

        frame_pcds = [f['pcd'] for f in frames]
        frame_bboxes = [f['bbox'] for f in frames]

        pcds = []
        mask_gts = []
        bbox_gts = []
        is_dynamic_gts = []
        wlh = None
        for i, (bbox, pcd, prev_bbox) in enumerate(zip(frame_bboxes, frame_pcds, prev_frame_bboxes)):
            if self.cfg.train_cfg.use_z:
                if self.cfg.use_smp_aug:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=4)
                    bbox_offset[3] = bbox_offset[3] * \
                        (5 if self.cfg.degree else np.deg2rad(5))
                else:
                    bbox_offset = np.zeros(4)
            else:
                if self.cfg.use_smp_aug:
                    bbox_offset = np.random.uniform(low=-0.3, high=0.3, size=3)
                    bbox_offset[2] = bbox_offset[2] * \
                        (5 if self.cfg.degree else np.deg2rad(5))
                else:
                    bbox_offset = np.zeros(3)

            if i == 0:
                # for the first frame, crop the pcd with the given gt bbox
                base_bbox = bbox
                wlh = bbox.wlh
            else:
                # for other frames, crop the pcd with previous pred bbox
                base_bbox = get_offset_box(
                    prev_bbox, bbox_offset, use_z=self.cfg.train_cfg.use_z, offset_max=self.cfg.offset_max, degree=self.cfg.degree, is_training=True)

            bbox = transform_box(bbox, base_bbox)
            pcd = crop_and_center_pcd(pcd, base_bbox, offset=self.cfg.frame_offset,
                                      offset2=self.cfg.frame_offset2, scale=self.cfg.frame_scale)
            mask_gt = get_pcd_in_box_mask(pcd, bbox, scale=1.25).astype(int)
            bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                bbox.orientation.degrees if self.cfg.degree else bbox.orientation.radians) * bbox.orientation.axis[-1]])

            assert pcd.nbr_points() >= 5
            # if i == 0:
            #     # target pcd in the first frame mustn't be empty
            #     assert np.sum(mask_gt) >= 5

            pcd, idx = resample_pcd(
                pcd, self.cfg.frame_npts, return_idx=True, is_training=True)
            mask_gt = mask_gt[idx]

            pcds.append(pcd.points.T)
            mask_gts.append(mask_gt)
            bbox_gts.append(bbox_gt)
            if i == 0:
                is_dynamic_gts.append(False)
            else:
                if np.linalg.norm(bbox_gts[i][:3]-bbox_gts[i-1][:3], ord=2) > self.cfg.dynamic_threshold:
                    is_dynamic_gts.append(True)
                else:
                    is_dynamic_gts.append(False)

        first_mask_gt = mask_gts[0]
        first_bbox_gt = bbox_gts[0]

        data = dict(
            wlh=wlh,
            lwh=np.array([wlh[1], wlh[0], wlh[2]]),
            pcds=pcds,
            mask_gts=mask_gts,
            bbox_gts=bbox_gts,
            first_mask_gt=first_mask_gt,
            first_bbox_gt=first_bbox_gt,
            is_dynamic_gts=is_dynamic_gts
        )
        return self._to_float_tensor(data)

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.FloatTensor(v)
        return tensor_data

    def __getitem__(self, idx):

        tracklet_id, frame_id = self.dataset.get_tracklet_frame_id(idx)
        tracklet_length = self.dataset.num_tracklet_frames(tracklet_id)
        assert tracklet_length >= self.cfg.num_smp_frames_per_tracklet

        frame_ids = [frame_id]
        acceptable_set = set(range(
            max(0, frame_id-self.cfg.max_frame_dis),
            min(tracklet_length, frame_id+self.cfg.max_frame_dis+1)
        )).difference(set(frame_ids))
        while len(frame_ids) < self.cfg.num_smp_frames_per_tracklet:
            idx = np.random.choice(list(acceptable_set))
            frame_ids.append(idx)
            new_set = set(range(max(0, idx-self.cfg.max_frame_dis),
                          min(tracklet_length, idx+self.cfg.max_frame_dis+1)))
            acceptable_set = acceptable_set.union(
                new_set).difference(set(frame_ids))

        frame_ids = sorted(frame_ids)
        if np.random.rand() < 0.5:
            # Reverse time
            frame_ids = frame_ids[::-1]
            prev_frame_ids = [min(f_id+1, tracklet_length-1)
                              for f_id in frame_ids]
        else:
            prev_frame_ids = [max(f_id-1, 0)
                              for f_id in frame_ids]

        frames = [self.dataset.get_frame(tracklet_id, id) for id in frame_ids]
        prev_frame_bboxes = [self.dataset.get_frame_bbox(
            tracklet_id, id) for id in prev_frame_ids]
        # comp_template_pcd = self.dataset.get_comp_pcd(tracklet_id)

        try:
            return self._generate_item(frames, prev_frame_bboxes)
        except AssertionError:
            return self[torch.randint(0, len(self), size=(1,)).item()]
