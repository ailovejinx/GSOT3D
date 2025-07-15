import torch
import numpy as np
import pytorch_lightning as pl
import os.path as osp
import json
import time

from optimizers import create_optimizer
from schedulers import create_scheduler
from models import create_model
from utils import *

from pytorch3d.ops import box3d_overlap
import os


class BaseTask(pl.LightningModule):

    def __init__(self, cfg, log):
        super().__init__()
        self.cfg = cfg
        self.txt_log = log
        self.model = create_model(cfg.model_cfg, log)
        self.txt_log.info('Model size = %.2f MB' % self.compute_model_size())
        if 'Waymo' in cfg.dataset_cfg.dataset_type:
            self.succ_total = AverageMeter()
            self.prec_total = AverageMeter()
            self.succ_easy = AverageMeter()
            self.prec_easy = AverageMeter()
            self.succ_medium = AverageMeter()
            self.prec_medium = AverageMeter()
            self.succ_hard = AverageMeter()
            self.prec_hard = AverageMeter()
            self.n_frames_total = 0
            self.n_frames_easy = 0
            self.n_frames_medium = 0
            self.n_frames_hard = 0
        elif 'NuScenes' in cfg.dataset_cfg.dataset_type:
            self.prec = TorchPrecision()
            self.succ = TorchSuccess()
            self.n_frames_total = 0
            self.n_frames_key = 0
        else:
            self.prec = TorchPrecision()
            self.succ = TorchSuccess()
            self.mao = TorchMAO()
            self.msr50 = TorchMSR50()
            self.msr75 = TorchMSR75()
            self.runtime = TorchRuntime()
            self.n_frames = 0
            if cfg.save_test_result:
                self.pred_bboxes = []
                self.gt_bboxes = []
                self.scene_list = []
                self.frame_list = []
                self.track_id_list = []
                self.class_list = []

        if 'GSOT3D' in cfg.dataset_cfg.dataset_type:
            symmetric_file = osp.join(cfg.dataset_cfg.data_root_dir, 'symmetric.txt')
            self.symmetric = np.loadtxt(symmetric_file, dtype=np.int32)

    def _cal_suc_prec_step(self, gt_bbox, pred_bbox, is_symm):
        if self.cfg.eval_cfg.iou_mode == "7-DoF":
            this_overlap = estimateOverlap(gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space,
                                           up_axis=self.cfg.dataset_cfg.up_axis)
            this_accuracy = estimateAccuracy(gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space,
                                             up_axis=self.cfg.dataset_cfg.up_axis)
            return this_overlap, this_accuracy
        elif self.cfg.eval_cfg.iou_mode == "9-DoF":
            # print(angle)
            origin_center = pred_bbox.center
            gt_bbox_vertices = cal_3dbb_vertices(gt_bbox.center, gt_bbox.wlh, gt_bbox.rotation_matrix)
            gt_bbox_vertices = torch.tensor(gt_bbox_vertices, dtype=torch.float).unsqueeze(0)
            if is_symm:
                angle_num = 120
                this_overlap_max = 0.0
                this_accuracy_max = 0.0
                for angle in np.linspace(0, np.pi * 2, angle_num):
                    box_3d_rotated = get_rotated_box(pred_bbox, angle)
                    # print(f"rotate angle: {angle}")
                    # print(box_3d_rotated.center - origin_center)
                    # this_overlap = estimateOverlap(gt_bbox, box_3d_rotated, dim=self.cfg.eval_cfg.iou_space,
                    #                                            up_axis=self.cfg.dataset_cfg.up_axis)
                    box_3d_rotated_vertices = cal_3dbb_vertices(
                        box_3d_rotated.center, box_3d_rotated.wlh, box_3d_rotated.rotation_matrix)
                    box_3d_rotated_vertices = torch.tensor(box_3d_rotated_vertices, dtype=torch.float).unsqueeze(0)

                    _, this_overlap = box3d_overlap(gt_bbox_vertices, box_3d_rotated_vertices)
                    this_overlap = this_overlap.item()
                    this_accuracy = estimateAccuracy(
                        gt_bbox,
                        box_3d_rotated,
                        dim=self.cfg.eval_cfg.iou_space,
                        up_axis=self.cfg.dataset_cfg.up_axis)

                    if this_overlap > this_overlap_max:
                        this_overlap_max = this_overlap
                    if this_accuracy > this_accuracy_max:
                        this_accuracy_max = this_accuracy

                return this_overlap_max, this_accuracy_max
            else:
                pred_bbox_vertices = cal_3dbb_vertices(pred_bbox.center, pred_bbox.wlh, pred_bbox.rotation_matrix)
                pred_bbox_vertices = torch.tensor(pred_bbox_vertices, dtype=torch.float).unsqueeze(0)

                _, this_overlap = box3d_overlap(gt_bbox_vertices, pred_bbox_vertices)
                this_accuracy = estimateAccuracy(
                    gt_bbox,
                    pred_bbox,
                    dim=self.cfg.eval_cfg.iou_space,
                    up_axis=self.cfg.dataset_cfg.up_axis)

            return this_overlap, this_accuracy
        else:
            raise ValueError(f"iou_mode should be 7-DoF or 9-DoF, got{self.cfg.eval_cfg.iou_mode}")

    def compute_model_size(self):
        num_param = sum([p.numel() for p in self.model.parameters()])
        param_size = num_param * 4 / 1024 / 1024  # MB
        return param_size

    def configure_optimizers(self):
        optimizer = create_optimizer(self.cfg.optimizer_cfg, self.parameters())
        scheduler = create_scheduler(self.cfg.scheduler_cfg, optimizer)
        return dict(
            optimizer=optimizer,
            lr_scheduler=scheduler
        )

    def training_step(self, *args, **kwargs):
        raise NotImplementedError(
            'Training_step has not been implemented!')

    def on_validation_epoch_start(self):
        self.prec.reset()
        self.succ.reset()
        # self.mao.reset()
        # self.msr50.reset()
        # self.msr75.reset()
        self.runtime.reset()
        # self.n_frames.reset()
        self.n_frames = 0

    def forward_on_tracklet(self, tracklet):
        raise NotImplementedError(
            'Forward_on_tracklet has not been implemented!')

    def validation_step(self, batch, batch_idx):
        tracklet = batch[0]
        # get symmetric flag
        if 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            seq_num = int(tracklet[0]['anno']['scene'].split('_')[-1])
            symmetric_flag = self.symmetric[seq_num - 1]
        # else:  # KITTI
        #     seq_num = f"{tracklet[0]['anno']['scene']}_{tracklet[0]['anno']['track_id']}"
        # # else:
        #     pass

        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            if 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
                overlap, accuracy = self._cal_suc_prec_step(gt_bbox, pred_bbox, symmetric_flag)
            else:
                overlap, accuracy = self._cal_suc_prec_step(gt_bbox, pred_bbox, False)
            overlaps.append(overlap)
            accuracies.append(accuracy)

        self.succ(torch.tensor(overlaps, device=self.device))
        # self.mao(torch.tensor(overlaps, device=self.device), str(seq_num))
        # self.msr50(torch.tensor(overlaps, device=self.device), str(seq_num))
        # self.msr75(torch.tensor(overlaps, device=self.device), str(seq_num))

        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        # self.n_frames(torch.tensor(n_frames, device=self.device))
        self.n_frames = n_frames

    def on_validation_epoch_end(self):

        self.log('precision', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        self.log('n_frames', self.n_frames, prog_bar=True)

    def _on_test_epoch_start_kitti_format(self):
        self.prec.reset()
        self.succ.reset()
        self.mao.reset()
        self.runtime.reset()
        self.n_frames = 0
        if 'MBPTrack' in self.cfg.model_cfg.model_type:
            with torch.no_grad():
                for _ in range(100):
                    backbone_input = dict(
                        pcds=torch.randn(1, 1, 1024, 3).cuda(),
                    )
                    trfm_input = dict(
                        xyz=torch.randn(1, 128, 3).cuda(),
                        feat=torch.randn(1, 128, 128).cuda(),
                        memory=dict(
                            feat=torch.randn(2, 1, 128, 3, 128).cuda(),
                            xyz=torch.randn(1, 3, 128, 3).cuda(),
                            mask=torch.randn(1, 3, 128).cuda(),
                        ),
                    )
                    loc_input = dict(
                        xyz=torch.randn(1, 128, 3).cuda(),
                        geo_feat=torch.randn(1, 128, 128).cuda(),
                        mask_feat=torch.randn(1, 128, 128).cuda(),
                        lwh=torch.ones(1, 3).cuda()
                    )
                    _ = self.model(backbone_input, 'embed')
                    _ = self.model(trfm_input, 'propagate')
                    _ = self.model(loc_input, 'localize')

    def _on_test_epoch_start_waymo_format(self):
        self.succ_total.reset()
        self.prec_total.reset()
        self.succ_easy.reset()
        self.prec_easy.reset()
        self.succ_medium.reset()
        self.prec_medium.reset()
        self.succ_hard.reset()
        self.prec_hard.reset()
        self.n_frames_total = 0
        self.n_frames_easy = 0
        self.n_frames_medium = 0
        self.n_frames_hard = 0

    def _on_test_epoch_start_nuscenes_format(self):
        self.prec.reset()
        self.succ.reset()
        # self.mao.reset()
        self.n_frames_total = 0
        self.n_frames_key = 0

    def on_test_epoch_start(self):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_kitti_format()
        elif 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_kitti_format()
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_waymo_format()
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            self._on_test_epoch_start_nuscenes_format()

    def _test_step_kitti_format(self, batch, batch_idx):
        tracklet = batch[0]
        # get symmetric flag
        if 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            seq_num = int(tracklet[0]['anno']['scene'].split('_')[-1])
            symmetric_flag = self.symmetric[seq_num - 1]

        torch.cuda.synchronize()
        start_time = time.time()
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        # save this sequence's results to txt
        if 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            result_save_path = osp.join(f'/data/code/evaluation_results/{self.cfg.model_cfg.model_type}',
                                        str(tracklet[0]['anno']['scene']) + f'_{self.cfg.dataset_cfg.category_name}.txt')
            with open(result_save_path, 'w') as f:
                for bb in pred_bboxes:
                    f.write(f"{bb.center[0]:.2f} {bb.center[1]:.2f} {bb.center[2]:.2f} "
                            f"{bb.wlh[0]:.2f} {bb.wlh[1]:.2f} {bb.wlh[2]:.2f} {bb.rotation_matrix[0, 0]:.2f} "
                            f"{bb.rotation_matrix[0, 1]:.2f} {bb.rotation_matrix[0, 2]:.2f} "
                            f"{bb.rotation_matrix[1, 0]:.2f} {bb.rotation_matrix[1, 1]:.2f} {bb.rotation_matrix[1, 2]:.2f} "
                            f"{bb.rotation_matrix[2, 0]:.2f} {bb.rotation_matrix[2, 1]:.2f} {bb.rotation_matrix[2, 2]:.2f}\n")
        elif 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            result_save_dir = (f'/data/code/{self.cfg.dataset_cfg.dataset_type}_results/prot3d_'
                               f'{self.cfg.dataset_cfg.dataset_type}/{self.cfg.dataset_cfg.category_name}/result')
            if not osp.exists(result_save_dir):
                os.makedirs(result_save_dir)

            gt_save_dir = (f'/data/code/{self.cfg.dataset_cfg.dataset_type}_results/prot3d_'
                           f'{self.cfg.dataset_cfg.dataset_type}/{self.cfg.dataset_cfg.category_name}/gt')
            if not osp.exists(gt_save_dir):
                os.makedirs(gt_save_dir)

            result_save_path = osp.join(result_save_dir, str(batch_idx) + f'_{self.cfg.dataset_cfg.category_name}.txt')
            gt_save_path = osp.join(gt_save_dir, str(batch_idx) + f'_{self.cfg.dataset_cfg.category_name}_gt.txt')
            with open(result_save_path, 'w') as f:
                for bb in pred_bboxes:
                    f.write(f"{bb.center[0]:.2f} {bb.center[1]:.2f} {bb.center[2]:.2f} "
                            f"{bb.wlh[0]:.2f} {bb.wlh[1]:.2f} {bb.wlh[2]:.2f} {bb.rotation_matrix[0, 0]:.2f} "
                            f"{bb.rotation_matrix[0, 1]:.2f} {bb.rotation_matrix[0, 2]:.2f} "
                            f"{bb.rotation_matrix[1, 0]:.2f} {bb.rotation_matrix[1, 1]:.2f} {bb.rotation_matrix[1, 2]:.2f} "
                            f"{bb.rotation_matrix[2, 0]:.2f} {bb.rotation_matrix[2, 1]:.2f} {bb.rotation_matrix[2, 2]:.2f}\n")

            with open(gt_save_path, 'w') as f:
                for bb in gt_bboxes:
                    f.write(f"{bb.center[0]:.2f} {bb.center[1]:.2f} {bb.center[2]:.2f} "
                            f"{bb.wlh[0]:.2f} {bb.wlh[1]:.2f} {bb.wlh[2]:.2f} {bb.rotation_matrix[0, 0]:.2f} "
                            f"{bb.rotation_matrix[0, 1]:.2f} {bb.rotation_matrix[0, 2]:.2f} "
                            f"{bb.rotation_matrix[1, 0]:.2f} {bb.rotation_matrix[1, 1]:.2f} {bb.rotation_matrix[1, 2]:.2f} "
                            f"{bb.rotation_matrix[2, 0]:.2f} {bb.rotation_matrix[2, 1]:.2f} {bb.rotation_matrix[2, 2]:.2f}\n")

        torch.cuda.synchronize()
        end_time = time.time()
        runtime = end_time-start_time
        n_frames = len(tracklet)
        if self.cfg.save_test_result:
            self.pred_bboxes.append((batch_idx, pred_bboxes))
        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            if 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
                overlap, accuracy = self._cal_suc_prec_step(gt_bbox, pred_bbox, symmetric_flag)
            else:
                overlap, accuracy = self._cal_suc_prec_step(gt_bbox, pred_bbox, False)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        self.succ(torch.tensor(overlaps, device=self.device))
        # self.mao.update(torch.tensor(overlaps, device=self.device), str(seq_num))
        # self.msr50.update(torch.tensor(overlaps, device=self.device), str(seq_num))
        # self.msr75.update(torch.tensor(overlaps, device=self.device), str(seq_num))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.runtime(torch.tensor(runtime, device=self.device),
                     torch.tensor(n_frames, device=self.device))
        # self.n_frames = torch.tensor(n_frames, device=self.device)
        # self.n_frames = n_frames
        # self.txt_log.info('Prec=%.3f Succ=%.3f Frames=%d RunTime=%.6f' % (
        #     self.prec.compute(), self.succ.compute(), self.n_frames, self.runtime.compute()))
        # self.log('precision', self.prec.compute(), prog_bar=True, logger=False)
        # self.log('success', self.succ.compute(), prog_bar=True, logger=False)
        # self.log('n_frames', self.n_frames,
        #          prog_bar=True, logger=False)

    def _test_step_waymo_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        tracklet_length = len(tracklet) - 1
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        n_frames = len(tracklet)

        success = TorchSuccess()
        precision = TorchPrecision()

        overlaps, accuracies = [], []
        for pred_bbox, gt_bbox in zip(pred_bboxes, gt_bboxes):
            overlap = estimateWaymoOverlap(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space)
            accuracy = estimateAccuracy(
                gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
            overlaps.append(overlap)
            accuracies.append(accuracy)
        success(torch.tensor(overlaps, device=self.device))
        precision(torch.tensor(accuracies, device=self.device))
        success = success.compute() if type(
            success.compute()) == float else success.compute().item()
        precision = precision.compute() if type(
            precision.compute()) == float else precision.compute().item()

        self.succ_total.update(success, n=tracklet_length)
        self.prec_total.update(precision, n=tracklet_length)
        self.n_frames_total += n_frames
        if tracklet[0]['mode'] == 'easy':
            self.succ_easy.update(success, n=tracklet_length)
            self.prec_easy.update(precision, n=tracklet_length)
            self.n_frames_easy += n_frames
        elif tracklet[0]['mode'] == 'medium':
            self.succ_medium.update(success, n=tracklet_length)
            self.prec_medium.update(precision, n=tracklet_length)
            self.n_frames_medium += n_frames
        elif tracklet[0]['mode'] == 'hard':
            self.succ_hard.update(success, n=tracklet_length)
            self.prec_hard.update(precision, n=tracklet_length)
            self.n_frames_hard += n_frames

        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def _test_step_nuscenes_format(self, batch, batch_idx):
        # if batch_idx != 0:
        #     return
        tracklet = batch[0]
        if tracklet[0]['anno']['num_lidar_pts'] == 0:
            return
        n_frames = len(tracklet)
        self.n_frames_total += n_frames
        pred_bboxes, gt_bboxes = self.forward_on_tracklet(tracklet)
        overlaps, accuracies = [], []
        for i, (pred_bbox, gt_bbox) in enumerate(zip(pred_bboxes, gt_bboxes)):
            anno = tracklet[i]['anno']
            if anno['is_key_frame'] == 1:
                self.n_frames_key += 1
                overlap = estimateOverlap(
                    gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
                accuracy = estimateAccuracy(
                    gt_bbox, pred_bbox, dim=self.cfg.eval_cfg.iou_space, up_axis=self.cfg.dataset_cfg.up_axis)
                overlaps.append(overlap)
                accuracies.append(accuracy)

        self.succ(torch.tensor(overlaps, device=self.device))
        # self.mao(torch.tensor(overlaps, device=self.device), torch.tensor(batch_idx, device=self.device))   # TODO
        # self.msr50(torch.tensor(overlaps, device=self.device), torch.tensor(batch_idx, device=self.device))
        # self.msr75(torch.tensor(overlaps, device=self.device), torch.tensor(batch_idx, device=self.device))
        self.prec(torch.tensor(accuracies, device=self.device))
        self.txt_log.info('Key: Prec=%.3f Succ=%.3f Key Frames=(%d/%d)' % (
            self.prec.compute(), self.succ.compute(), self.n_frames_key, self.n_frames_total))

    def test_step(self, batch, batch_idx):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_kitti_format(batch, batch_idx)
        elif 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_kitti_format(batch, batch_idx)
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_waymo_format(batch, batch_idx)
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            return self._test_step_nuscenes_format(batch, batch_idx)

    def _on_test_epoch_end_kitti_format(self):
        self.log('precision', self.prec.compute(), prog_bar=True)
        self.log('success', self.succ.compute(), prog_bar=True)
        # self.log('mao', self.mao.compute(), prog_bar=True)
        # self.log('msr50', self.msr50.compute(), prog_bar=True)
        # self.log('msr75', self.msr75.compute(), prog_bar=True)
        self.log('runtime', self.runtime.compute(), prog_bar=True)
        self.txt_log.info('Avg Prec/Succ=%.3f/%.3f Frames=%d Runtime=%.6f' % (
            self.prec.compute(), self.succ.compute(), self.n_frames, self.runtime.compute()))
        if self.cfg.save_test_result:
            # self.scene_list.sort(key=lambda x: x[0])
            # self.frame_list.sort(key=lambda x: x[0])
            # self.track_id_list.sort(key=lambda x: x[0])
            # self.class_list.sort(key=lambda x: x[0])
            self.pred_bboxes.sort(key=lambda x: x[0])
            self.gt_bboxes.sort(key=lambda x: x[0])

            data = []
            for idx, bbs in self.pred_bboxes:
                pred_bboxes = []
                for bb in bbs:
                    pred_bboxes.append([idx, bb.encode()])
                data.append(pred_bboxes)
            with open(osp.join(self.cfg.work_dir, 'result.json'), 'w') as f:
                json.dump(data, f)

            data = []
            for idx, bbs in self.gt_bboxes:
                gt_bboxes = []
                for bb in bbs:
                    gt_bboxes.append([idx, bb.encode()])
                data.append(gt_bboxes)
            with open(osp.join(self.cfg.work_dir, 'gt.json'), 'w') as f:
                json.dump(data, f)

    def _on_test_epoch_end_waymo_format(self):
        self.txt_log.info('============ Final ============')
        self.txt_log.info('Total: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_total.avg, self.succ_total.avg,  self.n_frames_total))
        self.txt_log.info('easy: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_easy.avg, self.succ_easy.avg,  self.n_frames_easy))
        self.txt_log.info('medium: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_medium.avg, self.succ_medium.avg,  self.n_frames_medium))
        self.txt_log.info('hard: Prec=%.3f Succ=%.3f Frames=%d' % (
            self.prec_hard.avg, self.succ_hard.avg,  self.n_frames_hard))

    def _on_test_epoch_end_nuscenes_format(self):
        self.txt_log.info('============ Final ============')
        self.txt_log.info('Key: Prec=%.3f Succ=%.3f Key Frames=(%d/%d)' % (
            self.prec.compute(), self.succ.compute(), self.n_frames_key, self.n_frames_total))

    def on_test_epoch_end(self):
        if 'KITTI' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_kitti_format()
        elif 'GSOT3D' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_kitti_format()
        elif 'Waymo' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_waymo_format()
        elif 'NuScenes' in self.cfg.dataset_cfg.dataset_type:
            return self._on_test_epoch_end_nuscenes_format()
        