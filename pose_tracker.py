import sys
sys.path.append('./FoundationPose')

import trimesh
import numpy as np
from typing import Union

# foundationpose
from estimater import *
from datareader import *
from Utils import draw_posed_3d_box, draw_xyz_axis

# disable logging
logging.disable(logging.WARNING)

def visualize_pose_2d(rgb: np.array, pose_in_cam: np.array, cam_K: np.array, draw_bbox: bool=False, bbox: Union[np.array, None]=None, bboxCenter_T_localOrigin: Union[np.array, None]=None) -> np.array:
    r"""
    Visualize local axis and bounding box in 2D image
    return:
    vis[np.array]: (h, w, 3), RGB
    """
    rgb = rgb.copy()
    # bounding box
    if draw_bbox:
        if bbox is None:
            default_sizes = np.array([0.04, 0.04, 0.02])
            bbox = np.stack([-default_sizes/2, default_sizes/2], axis=0).reshape(2,3)
        if bboxCenter_T_localOrigin is None:
            bboxCenter_T_localOrigin = np.eye(4)
        center_pose = pose_in_cam @ np.linalg.inv(bboxCenter_T_localOrigin)
        vis = draw_posed_3d_box(cam_K, img=rgb, ob_in_cam=center_pose, bbox=bbox)
    # axis
    vis = draw_xyz_axis(rgb, ob_in_cam=pose_in_cam, scale=0.1, K=cam_K, thickness=3, transparency=0, is_input_rgb=True)
    return vis

class PoseTracker:
    def __init__(self, mesh: trimesh.Trimesh, cam_K: np.array):
        self.mesh = mesh
        self.cam_K = cam_K

        self.to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)

        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        self.fp = FoundationPose(model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh, scorer=self.scorer, refiner=self.refiner, debug_dir=None, debug=0, glctx=self.glctx)

    def pose_estimate(self, rgb: np.array, depth: np.array, mask: np.array, refine_iter: int=5):
        mask = mask.astype(bool)
        pose = self.fp.register(K=self.cam_K, rgb=rgb, depth=depth, ob_mask=mask, iteration=refine_iter)
        scores, vis = self.scorer.predict(mesh=self.fp.mesh, rgb=rgb, depth=depth, K=self.cam_K, ob_in_cams=pose[np.newaxis, ...], normal_map=None, mesh_tensors=self.fp.mesh_tensors, glctx=self.glctx, mesh_diameter=self.fp.diameter, get_vis=False)
        return pose, scores[0].float()

    def pose_track(self, rgb: np.array, depth: np.array, refine_iter: int=2):
        pose = self.fp.track_one(K=self.cam_K, rgb=rgb, depth=depth, iteration=refine_iter)
        scores, vis = self.scorer.predict(mesh=self.fp.mesh, rgb=rgb, depth=depth, K=self.cam_K, ob_in_cams=pose[np.newaxis, ...], normal_map=None, mesh_tensors=self.fp.mesh_tensors, glctx=self.glctx, mesh_diameter=self.fp.diameter, get_vis=False)
        return pose, scores[0].float()
    
    def pose_visualize(self, rgb: np.array, pose: np.array) -> np.array:
        vis = visualize_pose_2d(rgb, pose, self.cam_K, draw_bbox=True, bbox=self.bbox, bboxCenter_T_localOrigin=self.to_origin)
        return vis