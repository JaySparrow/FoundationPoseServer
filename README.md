# FoundationPoseServer
## Prerequisites
- CUDA 11.8

## Install
1. Clone and install FoundationPose: https://github.com/NVlabs/FoundationPose. 
    - Download the pretrained weights. 
    - Go to `estimater.py`, at line 24 right before `os.makedirs(debug_dir, exist_ok=True)`, add `if debug > 0:`.

## Mesh Assets
To add a mesh for a new object, create a folder under `assets` with the folder name as the object name. In the object folder, include the `.obj` mesh file (and a `.mtl` material file, if there is one) and a `color.txt`. The `color.txt` contains one line of RGB color values, e.g. `0 255 255`.

## Folder Structure
```
.
+- FoundationPose
+- assets
|  +- obj_name
|  |  +- *.obj
|  |  +- color.txt
│  +- ...
+- pose_tracker_server..py
+- pose_tracker.py
+- README.md
```

## Use
```
# Change TCP port and assets folder in the script.
python pose_tracker_server.py
```

## Troubleshoot
### 1. "fatal error: Eigen/Dense: No such file or directory"
Go to `"bundlesdf/mycuda/setup.py"` and change the path at line 35 `include_dirs` to `"/path/to/conda/include/eigen3"`.