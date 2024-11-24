import sys
sys.path.append('./FoundationPose')

# local
from pose_tracker import PoseTracker
# foundationpose
from Utils import trimesh_add_pure_colored_texture
# lib
import numpy as np
import trimesh
import os
import zmq

class PoseTrackerServer:
    r"""
    Data types of arrays:
        - int: np.int32
        - float: np.float64
        - image: np.uint8
    4 Modes
        1. "INIT": Initialize a PoseTracker for each object
            - Receive
                * Mode (str) "INIT"
                * Object names (str) e.g. "nut2 bolt2 wrench2_head stick"
                * Camera intrinsics (np.array 3x3) 
            - Do
                * Check if the object meshes exist
                * Initialize a PoseTracker for each existing object
            - Send
                * Names of the objects that have been initialized (str) e.g. "nut2 wrench2_head"
                * bboxCenter_T_localOrigin (np.array NUM_OBJECTSx4x4)
                * bbox (np.array NUM_OBJECTSx2x3)
        2. "ESTIMATE": Estimate the pose of each object
            - Receive
                * Mode (str) "ESTIMATE"
                * HxW (np.array 2x1)
                * Image (np.array HxWx3)
                * Depth (np.array HxW)
                * Object Names (str) e.g. "nut2 bolt2 wrench2_head stick"
                * Masks (np.array NUM_OBJECTSxHxW)
            - Do
                * Estimate the pose of each object
            - Send
                * Pose of each object (np.array NUM_OBJECTSx4x4)
        3. "TRACK": Track the pose of each object
            - Receive
                * Mode (str) "TRACK"
                * HxW (np.array 2x1)
                * Image (np.array HxWx3)
                * Depth (np.array HxW)
                * Object Names (str) e.g. "nut2 bolt2 wrench2_head stick"
            - Do
                * Track the pose of each object
            - Send
                * Pose of each object (np.array NUM_OBJECTSx4x4)
        4. "CLOSE": Close the server
            - Receive
                * Mode (str) "CLOSE"
            - Do
                * Close the server
            - Send
                * "closed"
    """
    def __init__(self, port="8080", assets_folder="./assets"):
        # zmq setup
        context = zmq.Context()
        self.socket = context.socket(zmq.REP)
        self.socket.bind(f"tcp://*:{port}")

        self.assets_folder = assets_folder
        self.obj_tracker_dict = None

    def run_service(self):
        msgs = self.socket.recv_multipart(flags=0)
        msgs_dict = self.interpret_msgs(msgs)
        mode = msgs_dict["mode"]
        if mode == "INIT":
            # clear the previous obj_tracker_dict
            if self.obj_tracker_dict is not None:
                for obj_name, pose_tracker in self.obj_tracker_dict.items():
                    del pose_tracker
                self.obj_tracker_dict = None

            obj_tracker_dict = self.do_init(msgs_dict)
            self.obj_tracker_dict = obj_tracker_dict

            obj_names_list = []
            bboxCenter_T_localOrigin_list = []
            bbox_list = []
            for obj_name, pose_tracker in obj_tracker_dict.items():
                obj_names_list.append(obj_name)
                bboxCenter_T_localOrigin_list.append(pose_tracker.to_origin) # 4x4
                bbox_list.append(pose_tracker.bbox) # 2x3
            obj_names_str = " ".join(obj_names_list)
            bboxCenter_T_localOrigin = np.stack(bboxCenter_T_localOrigin_list, axis=0) # NUM_OBJECTSx4x4
            bbox = np.stack(bbox_list, axis=0) # NUM_OBJECTSx2x3
            msgs_back = [
                        obj_names_str.encode("utf-8"), 
                        bboxCenter_T_localOrigin.astype(np.float64).tobytes(), 
                        bbox.astype(np.float64).tobytes()
                        ]
            self.socket.send_multipart(msgs_back, flags=0)
            print(f"Server initialized with {len(self.obj_tracker_dict)} trackers: {self.obj_tracker_dict.keys()}.")
        elif mode == "ESTIMATE":
            poses = self.do_estimate(msgs_dict, self.obj_tracker_dict)
            msg_back = poses.astype(np.float64).tobytes()
            self.socket.send(msg_back, flags=0)
        elif mode == "TRACK":
            poses = self.do_track(msgs_dict, self.obj_tracker_dict)
            msg_back = poses.astype(np.float64).tobytes()
            self.socket.send(msg_back, flags=0)
        elif mode == "CLOSE":
            self.socket.send(b"closed", flags=0)
            self.socket.close()
            print("Server closed.")
            exit(0)

    def run(self):
        print("Server running...")
        while True:
            try:
                self.run_service()
            except KeyboardInterrupt:
                print("Closing server...")
                self.socket.close()
                break
        print("Server closed.")

    #########################
    #### Helper Functions ###
    #########################
    def interpret_msgs(self, msgs) -> dict:
        mode = msgs[0].decode("utf-8") # str
        msgs_dict = {
            "mode": "UNKNOWN"
        }
        if mode == "INIT":
            object_names = msgs[1].decode("utf-8").split()
            camera_intrinsics = np.frombuffer(msgs[2], dtype=np.float64).reshape(3, 3)
            msgs_dict = {
                "mode": mode,
                "object_names": object_names,
                "camera_intrinsics": camera_intrinsics
            }
        elif mode == "ESTIMATE":
            H, W = np.frombuffer(msgs[1], dtype=np.int32)
            image = np.frombuffer(msgs[2], dtype=np.uint8).reshape(H, W, 3)
            depth = np.frombuffer(msgs[3], dtype=np.float64).reshape(H, W) / 1e3
            object_names = msgs[4].decode("utf-8").split()
            masks = np.frombuffer(msgs[5], dtype=np.uint8).reshape(len(object_names), H, W)
            msgs_dict = {
                "mode": mode,
                "H": H,
                "W": W,
                "image": image,
                "depth": depth,
                "object_names": object_names,
                "masks": masks
            }
        elif mode == "TRACK":
            H, W = np.frombuffer(msgs[1], dtype=np.int32)
            image = np.frombuffer(msgs[2], dtype=np.uint8).reshape(H, W, 3)
            depth = np.frombuffer(msgs[3], dtype=np.float64).reshape(H, W) / 1e3
            object_names = msgs[4].decode("utf-8").split()
            msgs_dict = {
                "mode": mode,
                "H": H,
                "W": W,
                "image": image,
                "depth": depth,
                "object_names": object_names
            }
        elif mode == "CLOSE":
            msgs_dict = {
                "mode": mode
            }
        return msgs_dict
    
    def do_init(self, msgs_dict):
        object_names = msgs_dict["object_names"]
        cam_K = msgs_dict["camera_intrinsics"]

        obj_tracker_dict = {}
        for obj_name in object_names:
            obj_mesh_folder = f"{self.assets_folder}/{obj_name}"

            if os.path.exists(obj_mesh_folder):
                
                # load the first obj mesh if there are multiple
                obj_mesh_filename_list= [filename for filename in os.listdir(obj_mesh_folder) if filename.lower().endswith(".obj")]
                mesh = trimesh.load(os.path.join(obj_mesh_folder, obj_mesh_filename_list[0]), force='mesh')
                # use default color if color.txt does not exist
                if os.path.exists(f"{obj_mesh_folder}/color.txt"):
                    color = np.loadtxt(f"{obj_mesh_folder}/color.txt").astype(np.uint8).flatten()
                else:
                    color = np.array([128, 128, 128], dtype=np.uint8)
                mesh = trimesh_add_pure_colored_texture(mesh, color=color, resolution=4096)

                pose_tracker = PoseTracker(mesh, cam_K)
                obj_tracker_dict[obj_name] = pose_tracker

        return obj_tracker_dict
    
    def do_estimate(self, msgs_dict, obj_tracker_dict) -> np.ndarray:
        image = msgs_dict["image"]
        depth = msgs_dict["depth"]
        object_names = msgs_dict["object_names"]
        masks = msgs_dict["masks"]

        poses = []
        for obj_i, obj_name in enumerate(object_names):
            pose_tracker = obj_tracker_dict[obj_name]
            mask = masks[obj_i] # HxW
            pose, score = pose_tracker.pose_estimate(image.copy(), depth, mask)
            poses.append(pose)
        poses = np.stack(poses, axis=0) # NUM_OBJECTSx4x4
        return poses
    
    def do_track(self, msgs_dict, obj_tracker_dict) -> np.ndarray:
        image = msgs_dict["image"]
        depth = msgs_dict["depth"]
        object_names = msgs_dict["object_names"]

        poses = []
        for obj_name in object_names:
            pose_tracker = obj_tracker_dict[obj_name]
            pose, score = pose_tracker.pose_track(image.copy(), depth)
            poses.append(pose)
        poses = np.stack(poses, axis=0) # NUM_OBJECTSx4x4
        return poses
    
    
if __name__ == "__main__":
    port = "8080"
    assets_folder = "./assets"
    pose_tracker_server = PoseTrackerServer(port=port, assets_folder=assets_folder)
    pose_tracker_server.run()