import torch.nn.functional as F
import torch
import time
import json
import os.path as osp

from datasets.utils.pcd_utils import *
from .base_task import BaseTask


class PROT3D(BaseTask):

    def __init__(self, cfg, log):
        super().__init__(cfg, log)

        # for name, param in self.model.named_parameters():
        #     if param.requires_grad:
        #         print(f"Parameter {name} will be updated during training.")

    def build_mask_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        return F.binary_cross_entropy_with_logits(
            pred,
            gt,
        )

    def build_objectness_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.binary_cross_entropy_with_logits(
            pred,
            gt,
            pos_weight=torch.tensor([2.0], device=self.device),
            reduction='none'
        )
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_bbox_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.smooth_l1_loss(pred, gt, reduction='none')
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-6)
        return loss

    def build_center_loss(self, input):
        pred = input['pred']
        gt = input['gt']
        mask = input['mask']
        loss = F.mse_loss(pred, gt, reduction='none')
        loss = (loss.mean(2) * mask).sum() / (mask.sum() + 1e-06)
        return loss

    def training_step(self, batch, batch_idx):

        pcds = batch['pcds']  # b,t,n,3
        mask_gts = batch['mask_gts']  # b,t,n, n for point number
        bbox_gts = batch['bbox_gts']  # b,t,4
        first_mask_gt = batch['first_mask_gt']  # b,n
        first_bbox_gt = batch['first_bbox_gt']  # b,4
        is_dynamic_gts = batch['is_dynamic_gts']  # b,t
        lwh = batch['lwh']  # b,3

        # backbone
        embed_output = self.model(dict(
            pcds=pcds
        ), mode='embed')
        xyzs, geo_feats, idxs = embed_output['xyzs'], embed_output['feats'], embed_output['idxs']
        # geo_feats: B, t, N, C

        # DeFPM
        propagate_output = self.model(dict(
            feat=geo_feats[:, 0, :, :],
            xyz=xyzs[:, 0, :, :],
            # first_mask_gt=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
        ), mode='propagate')

        layer_feats = propagate_output['layer_feats']

        # update memory
        update_output = self.model(dict(
            layer_feats=layer_feats,
            xyz=xyzs[:, 0, :, :],
            # mask=torch.gather(first_mask_gt, 1, idxs[:, 0, :]),
        ), mode='update')
        memory = update_output['memory']

        n_smp_frame = self.cfg.dataset_cfg.num_smp_frames_per_tracklet

        mask_loss_1, crs_obj_loss_1, rfn_obj_loss_1, center_loss_1, bbox_loss_1 = 0.0, 0.0, 0.0, 0.0, 0.0
        mask_loss_2, crs_obj_loss_2, rfn_obj_loss_2, center_loss_2, bbox_loss_2 = 0.0, 0.0, 0.0, 0.0, 0.0

        for i in range(1, n_smp_frame):
            propagate_output = self.model(dict(
                memory=memory,
                feat=geo_feats[:, i, :, :],
                xyz=xyzs[:, i, :, :]
            ), mode='propagate')
            # geo_feat, mask_feat = propagate_output['geo_feat'], propagate_output['mask_feat']
            geo_feat = propagate_output['geo_feat']
            layer_feats = propagate_output['layer_feats']

            mem_xyz = propagate_output['mem_xyz']
            mem_feat = propagate_output['mem_feat']

            localize_output_1 = self.model(dict(
                geo_feat=geo_feat,
                # mask_feat=mask_feat,
                xyz=xyzs[:, i, :, :],
                lwh=lwh,
                center_gt=bbox_gts[:, i, :3]
            ), mode='localize')

            # stage 1 output feats
            proposal_feat = localize_output_1['proposal_feat']  # B, C, n_proposals
            proposal_xyz_1 = localize_output_1['proposal_xyz']
            bboxes_pred_1 = localize_output_1['bboxes_pred']  # B, n_proposals, 7
            # best_bbox_idx_1 = bboxes_pred_1[:, :, -1]
            fps_idxs_1 = localize_output_1['fps_idxs']

            # stage 2
            fusion_input = dict(
                feat=proposal_feat,
                m_feat=mem_feat,
                xyz=proposal_xyz_1,
                m_xyz=mem_xyz
            )
            fusion_output = self.model(fusion_input, mode='fusion')

            # location 2
            localize_output_2 = self.model(dict(
                fusion_feats=fusion_output['geo_feat'],
                fusion_xyz=proposal_xyz_1,
                lwh=lwh,
                center_gt=bbox_gts[:, i, :3]
            ), mode='localize_2stage')

            # stage 1 loss
            mask_pred_1 = localize_output_1['mask_pred']
            # b,n
            mask_loss_1 += self.build_mask_loss(dict(
                pred=mask_pred_1,
                gt=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])
            ))
            center_pred_1 = localize_output_1['center_pred']
            center_loss_1 += self.build_center_loss(dict(
                pred=center_pred_1,
                gt=bbox_gts[:, i, :3].unsqueeze(
                    1).expand_as(center_pred_1),
                mask=torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])
            ))

            dist = torch.sum(
                (center_pred_1 - bbox_gts[:, i, None, :3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask_1 = torch.ones_like(
                objectness_label, dtype=torch.float)
            objectness_pred_1 = localize_output_1['objectness_pred']
            crs_obj_loss_1 += self.build_objectness_loss(dict(
                pred=objectness_pred_1,
                gt=objectness_label,
                mask=objectness_mask_1
            ))

            dist = torch.sum(
                (proposal_xyz_1 - bbox_gts[:, i, None, :3]) ** 2, dim=-1)

            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_pred_1 = bboxes_pred_1[:, :, -1]  # B, K
            objectness_mask_1 = torch.ones_like(
                objectness_label, dtype=torch.float)
            rfn_obj_loss_1 += self.build_objectness_loss(dict(
                pred=objectness_pred_1,
                gt=objectness_label,
                mask=objectness_mask_1
            ))
            bbox_loss_1 += self.build_bbox_loss(dict(
                pred=bboxes_pred_1[:, :, :9],
                gt=bbox_gts[:, i, None, :9].expand_as(
                    bboxes_pred_1[:, :, :9]),
                mask=objectness_label
            ))

            # stage 2 loss
            mask_pred_2 = localize_output_2['mask_pred']
            center_pred_2 = localize_output_2['center_pred']
            objectness_pred_2 = localize_output_2['objectness_pred']
            bboxes_pred_2 = localize_output_2['bboxes_pred']
            proposal_xyz_2 = localize_output_2['proposal_xyz']

            # b,n
            mask_gts_1 = torch.gather(mask_gts[:, i, :], 1, idxs[:, i, :])
            mask_loss_2 += self.build_mask_loss(dict(
                pred=mask_pred_2,
                gt=torch.gather(mask_gts_1, 1, fps_idxs_1)
            ))

            center_loss_2 += self.build_center_loss(dict(
                pred=center_pred_2,
                gt=bbox_gts[:, i, :3].unsqueeze(
                    1).expand_as(center_pred_2),
                mask=torch.gather(mask_gts_1, 1, fps_idxs_1)
            ))

            dist = torch.sum(
                (center_pred_2 - bbox_gts[:, i, None, :3]) ** 2, dim=-1)
            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_mask_2 = torch.ones_like(
                objectness_label, dtype=torch.float)

            crs_obj_loss_2 += self.build_objectness_loss(dict(
                pred=objectness_pred_2,
                gt=objectness_label,
                mask=objectness_mask_2
            ))

            dist = torch.sum(
                (proposal_xyz_2 - bbox_gts[:, i, None, :3]) ** 2, dim=-1)

            dist = torch.sqrt(dist + 1e-6)  # B, K
            objectness_label = torch.zeros_like(dist, dtype=torch.float)
            objectness_label[dist < 0.3] = 1
            objectness_pred_2 = bboxes_pred_2[:, :, -1]  # B, K
            objectness_mask_2 = torch.ones_like(
                objectness_label, dtype=torch.float)
            rfn_obj_loss_2 += self.build_objectness_loss(dict(
                pred=objectness_pred_2,
                gt=objectness_label,
                mask=objectness_mask_2
            ))
            # use wlh offset
            bboxes_pred_2 = torch.cat([
                bboxes_pred_2[:, :, :6],
                bboxes_pred_2[:, :, 6:7] + lwh[:, 1:2].unsqueeze(1).expand_as(bboxes_pred_2[:, :, 6:7]),
                bboxes_pred_2[:, :, 7:8] + lwh[:, 0:1].unsqueeze(1).expand_as(bboxes_pred_2[:, :, 7:8]),
                bboxes_pred_2[:, :, 8:9] + lwh[:, 2:3].unsqueeze(1).expand_as(bboxes_pred_2[:, :, 8:9]),
                bboxes_pred_2[:, :, 9:]
            ], dim=2)
            bbox_loss_2 += self.build_bbox_loss(dict(
                pred=bboxes_pred_2[:, :, :9],
                gt=bbox_gts[:, i, None, :9].expand_as(
                    bboxes_pred_2[:, :, :9]),
                mask=objectness_label
            ))

            if i < n_smp_frame - 1:
                update_output = self.model(dict(
                    layer_feats=layer_feats,
                    xyz=xyzs[:, i, :, :],
                    mask=mask_pred_1.sigmoid(),
                    memory=memory
                ), mode='update')
                memory = update_output['memory']

        loss = (self.cfg.loss_cfg.mask_weight * mask_loss_2 +
                self.cfg.loss_cfg.crs_obj_weight * crs_obj_loss_2 +
                self.cfg.loss_cfg.rfn_obj_weight * rfn_obj_loss_2 +
                self.cfg.loss_cfg.bbox_weight * bbox_loss_2 +
                self.cfg.loss_cfg.center_weight * center_loss_2)

        self.logger.experiment.add_scalars(
            'loss',
            {
                'loss_total': loss,
                'loss_bbox': bbox_loss_1,
                'loss_center': center_loss_1,
                'loss_mask': mask_loss_1,
                'loss_rfn_objectness': rfn_obj_loss_1,
                'loss_crs_objectness': crs_obj_loss_1,
            },
            global_step=self.global_step
        )

        return loss

    def _to_float_tensor(self, data):
        tensor_data = {}
        for k, v in data.items():
            tensor_data[k] = torch.tensor(
                v, device=self.device, dtype=torch.float32).unsqueeze(0)
        return tensor_data

    def forward_on_tracklet(self, tracklet):

        pred_bboxes = []
        gt_bboxes = []

        memory = None
        lwh = None
        lwh_base = None

        last_bbox_cpu = np.array([0.0, 0.0, 0.0, 0.0])
        last_wlh = np.array([0.0, 0.0, 0.0])

        for frame_id, frame in enumerate(tracklet):

            gt_bboxes.append(frame['bbox'])

            if frame_id == 0:
                base_bbox = frame['bbox']
                lwh = np.array(
                    [base_bbox.wlh[1], base_bbox.wlh[0], base_bbox.wlh[2]])
                lwh_base = lwh
            else:
                base_bbox = pred_bboxes[-1]

            pcd = crop_and_center_pcd(
                frame['pcd'], base_bbox, offset=self.cfg.dataset_cfg.frame_offset,
                offset2=self.cfg.dataset_cfg.frame_offset2, scale=self.cfg.dataset_cfg.frame_scale)
            if frame_id == 0:
                # print(pcd.nbr_points())
                if pcd.nbr_points() == 0:
                    pcd.points = np.array([[0.0], [0.0], [0.0]])
                bbox = transform_box(frame['bbox'], base_bbox)
                mask_gt = get_pcd_in_box_mask(
                    pcd, bbox, scale=1.25).astype(int)
                # print(pcd.nbr_points(), mask_gt.shape)
                bbox_gt = np.array([bbox.center[0], bbox.center[1], bbox.center[2], (
                    bbox.orientation.degrees if self.cfg.dataset_cfg.degree else bbox.orientation.radians) *
                                    bbox.orientation.axis[-1]])
                pcd, idx = resample_pcd(
                    pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)
                mask_gt = mask_gt[idx]
                # print(mask_gt.shape, pcd.nbr_points())
            else:
                if pcd.nbr_points() <= 1:
                    bbox = get_offset_box(
                        pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    pred_bboxes.append(bbox)
                    continue

                pcd, idx = resample_pcd(
                    pcd, self.cfg.dataset_cfg.frame_npts, return_idx=True, is_training=False)

            embed_output = self.model(dict(
                pcds=torch.tensor(pcd.points.T, device=self.device,
                                  dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            ), mode='embed')

            xyzs, geo_feats, idxs = embed_output['xyzs'], embed_output['feats'], embed_output['idxs']

            if frame_id == 0:
                first_bbox_gt = torch.tensor(
                    bbox_gt, device=self.device, dtype=torch.float32).unsqueeze(0)
                propagate_output = self.model(dict(
                    feat=geo_feats[:, 0, :, :],
                    xyz=xyzs[:, 0, :, :]
                ), mode='propagate')
                layer_feats = propagate_output['layer_feats']
                update_output = self.model(dict(
                    layer_feats=layer_feats,
                    xyz=xyzs[:, 0, :, :],
                ), mode='update')
                memory = update_output['memory']

                pred_bboxes.append(frame['bbox'])
            else:
                propagate_output = self.model(dict(
                    memory=memory,
                    feat=geo_feats[:, 0, :, :],
                    xyz=xyzs[:, 0, :, :]
                ), mode='propagate')
                # geo_feat, mask_feat = propagate_output['geo_feat'], propagate_output['mask_feat']
                geo_feat = propagate_output['geo_feat']
                layer_feats = propagate_output['layer_feats']

                mem_xyz = propagate_output['mem_xyz']
                mem_feat = propagate_output['mem_feat']

                localize_output_1 = self.model(dict(
                    geo_feat=geo_feat,
                    # mask_feat=mask_feat,
                    xyz=xyzs[:, 0, :, :],
                    lwh=torch.tensor(lwh, device=self.device,
                                     dtype=torch.float32).unsqueeze(0),
                ), mode='localize')
                # mask_pred = localize_output_1['mask_pred']

                # stage 1 output feats
                proposal_feat = localize_output_1['proposal_feat']  # B, C, n_proposals
                proposal_xyz_1 = localize_output_1['proposal_xyz']

                # stage 2
                fusion_input = dict(
                    feat=proposal_feat,
                    m_feat=mem_feat,
                    xyz=proposal_xyz_1,
                    m_xyz=mem_xyz
                )
                fusion_output = self.model(fusion_input, mode='fusion')

                # location 2
                localize_output_2 = self.model(dict(
                    fusion_feats=fusion_output['geo_feat'],
                    fusion_xyz=proposal_xyz_1,
                    lwh=torch.tensor(lwh, device=self.device,
                                     dtype=torch.float32).unsqueeze(0)
                ), mode='localize_2stage')

                mask_pred_2 = localize_output_2['mask_pred']
                bboxes_pred_2 = localize_output_2['bboxes_pred']
                bboxes_pred_cpu_2 = bboxes_pred_2.squeeze(
                    0).detach().cpu().numpy()

                bboxes_pred_cpu_2[np.isnan(bboxes_pred_cpu_2)] = -1e6
                # remove bboxes whose objectness pred is nan
                # it may happen at the early stage of training

                best_box_idx_2 = bboxes_pred_cpu_2[:, -1].argmax()
                bbox_cpu = bboxes_pred_cpu_2[best_box_idx_2, 0:6]
                if torch.max(mask_pred_2.sigmoid()) < self.cfg.missing_threshold:
                    bbox = get_offset_box(
                        pred_bboxes[-1], last_bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    bbox.wlh[0] = last_wlh[0]
                    bbox.wlh[1] = last_wlh[1]
                    bbox.wlh[2] = last_wlh[2]

                else:
                    bbox = get_offset_box(
                        pred_bboxes[-1], bbox_cpu, use_z=self.cfg.dataset_cfg.eval_cfg.use_z, is_training=False)
                    bbox.wlh[0] = bboxes_pred_cpu_2[best_box_idx_2, 6] + lwh_base[1]
                    bbox.wlh[1] = bboxes_pred_cpu_2[best_box_idx_2, 7] + lwh_base[0]
                    bbox.wlh[2] = bboxes_pred_cpu_2[best_box_idx_2, 8] + lwh_base[2]
                    last_bbox_cpu = bbox_cpu
                    last_wlh = bbox.wlh

                lwh = np.array([bbox.wlh[1], bbox.wlh[0], bbox.wlh[2]])

                pred_bboxes.append(bbox)
                if frame_id < len(tracklet) - 1:
                    update_output = self.model(dict(
                        layer_feats=layer_feats,
                        xyz=xyzs[:, 0, :, :],
                        # mask=mask_pred_2.sigmoid(),
                        memory=memory
                    ), mode='update')
                    memory = update_output['memory']

        return pred_bboxes, gt_bboxes

    def load_checkpoint_with_partial_weights(self, checkpoint_path):
        # load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

        # prepare a partial state dict
        partial_state_dict = {k: v for k, v in checkpoint['state_dict'].items() if k in self.state_dict()}

        # debug the partial_state_dict here
        # print("Keys to load:", partial_state_dict.keys())

        # load partial state dict into the model
        self.load_state_dict(partial_state_dict, strict=False)
