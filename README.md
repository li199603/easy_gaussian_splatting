# Easy Gaussian Splatting
Gaussian Splatting implementation based on [gsplat](https://github.com/nerfstudio-project/gsplat). Easier to install and use than the [official implementation](https://github.com/graphdeco-inria/gaussian-splatting).  

## Setup
You need to install pytorch first. We run the project in python3.10 + pytroch2.1 + cuda11.8 environment. But the project is not strict about which version to rely on.  
In linux, simple use ```pip install gsplat==1.0.0``` to install gsplat. You can install the newer version, but the compilation process takes up more memory (could be more than 16G RAM). To install gsplat on Windows, please check [this instruction](https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md). Run the following command to install the other dependencies.  
```bash
pip install -r requirements.txt
pip install matplotlib open3d  # optional
```  
You can use the following two formats of data to get your project up. Training with this data requires at least 16G RAM and 8G VRAM. The 16G RAM requirement is primarily for the ability to compile gsplat. It will build the CUDA code on the first run (JIT). You can also modify the configuration file to reduce gaussians densification, resulting in less VRAM usage.  
- [tandt_db](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) colmap format  
- [nerf_synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) blender format  

## Usage
### training
```bash
python train.py -c path_to_config -d path_to_colmap_or_blender_format_data -o path_to_output(default: ./output)
# eg:
python train.py -c configs/tandt_db.yaml -d data/tandt_db/tandt/truck
python train.py -c configs/nerf_synthetic.yaml -d data/nerf_synthetic/lego
```  
### evaluation
After the training is complete, the script train.py performs the evaluation by default. We can still evaluate the gaussian splatting model by use  
```bash
python eval.py -p path_to_training_output
python eval.py -p path_to_training_output -i selected_iterations
```  
### viewer
We provide a simple viewer to visualize the results of the training. You can use the following command to run the viewer. You can control with ```W, A, S, D, Q, E``` for camera translation and ```↑ ↓ ← →``` for rotation. In viewer, the arrow keys are used to control the camera's rotation around a point. We also provide GUI buttons ```roll+, roll-, pitch+, pitch-, yaw+, yaw-``` to control the camera's rotation around itself.  
```bash
python launch_viewer.py -p path_to_training_output
python launch_viewer.py -p path_to_training_output -i selected_iterations
```  
This viewer allows you to view the scene being constructed while training. Using the ```--view_online``` in running train.py.  
