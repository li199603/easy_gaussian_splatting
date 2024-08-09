# Easy Gaussian Splatting
[Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) implementation based on [gsplat](https://github.com/nerfstudio-project/gsplat). Easy to install and use.  

## Implementation Result
PSNR scores for some scenes, compare with official implementation.  
|               | Truck  | Train  | Dr Johnson | Playroom | Drums  | Lego    |
|:-------------:|:------:|:------:|:----------:|:--------:|:------:|:-------:|
| Official Impl | 25.19  | 21.10  | 28.77      | 30.04    | 26.15  | 35.78   |
| Ours          | 25.16  | 21.26  | 28.12      | 30.20    | 25.75  | 36.27   |

<div align="center">
  <video src="https://github.com/user-attachments/assets/86ba1847-e29c-491d-8dd1-0c9a020fc7ab" width="50%" />
</div>

## Setup
You need to install pytorch first. We run the project in python3.10 + pytroch2.1 environment. But the project is not strict about which version to rely on.  
In linux, simple use ```pip install gsplat==1.0.0``` to install gsplat. You can install the newer version, but the compilation process takes up more memory (could be more than 16G RAM). To install gsplat on Windows, please check [this instruction](https://github.com/nerfstudio-project/gsplat/blob/main/docs/INSTALL_WIN.md). Run the following command to install the other dependencies.  
```bash
pip install -r requirements.txt
pip install matplotlib open3d  # optional
```  
You can use the following two formats of data to get your project up. Training with this data requires at least 16G RAM and 16G VRAM. The 16G RAM requirement is primarily for the ability to compile gsplat. It will build the CUDA code on the first run (JIT). You can also modify the configuration file to reduce gaussians densification, resulting in less VRAM usage.  
- [tandt_db](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/datasets/input/tandt_db.zip) colmap format  
- [nerf_synthetic](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) blender format  

## Usage
### Training
```bash
python train.py -c path_to_config -d path_to_data -o path_to_output(default: ./output)
# eg:
python train.py -c configs/tandt_db.yaml -d data/tandt_db/tandt/truck
python train.py -c configs/nerf_synthetic.yaml -d data/nerf_synthetic/lego
```  
### Evaluation
After the training is complete, the script train.py performs the evaluation by default. We can still evaluate the gaussian splatting model by use  
```bash
python eval.py -p path_to_training_output
python eval.py -p path_to_training_output -i selected_iterations
```  
### Viewer
We provide a simple viewer to visualize the results of the training. You can use the following command to run the viewer. You can control with ```W, A, S, D, Q, E``` for camera translation and ```↑ ↓ ← →``` for rotation. In viewer, the arrow keys are used to control the camera's rotation around a point. We also provide GUI buttons ```roll+, roll-, pitch+, pitch-, yaw+, yaw-``` to control the camera's rotation around itself.  
```bash
python launch_viewer.py -p path_to_training_output
python launch_viewer.py -p path_to_training_output -i selected_iterations
```  
This viewer allows you to view the scene being constructed while training. Using the ```--view_online``` in running train.py.  
The viewer supports exporting rendered videos. You need to first add some cameras and set the parameters of the exported video. After clicking the 'export' button, the viewer will use cameras you added to interpolate multiple camera poses. Then it render multiple frames about these poses to synthesize the video.  

## Funny Masks
We take the mask into account in the calculation of loss, by this ```render_img = mask * gt_img + (1.0 - mask) * render_img```. In this way, the masked region of the image will not have any influence on the generation of the gaussian points. Because the gradient cannot be backpropagated to the gaussian points from the subsequent calculation of either L1 loss or SSIM loss, which means that the mask region does not produce a gradient. We can use masks to control the generated area of the gaussian points, so that the target object is removed from the constructed scene.  
We use the drums scene in the nerf synthetic dataset to show this funny feature. We will remove all cymbals from the scene. Raw data for drums scenes is still available [here](https://drive.google.com/drive/folders/1JDdLGDruGNXWnM1eqY1FNL9PlStjaKWi) and masks data is available [here](https://github.com/user-attachments/files/16555816/drums_masks.zip). Put the masks data into the drums data folder as follows:  

```text
drums
├── test
├── train
├── train_masks
    ├── r_0.png
    ├── r_1.png
    ├── ...
├── val
├── ...
```  
Replace the corresponding parameter in configs/ nerf_synthet.yaml with  
```yaml
use_masks: true
mask_expand_pixels: 4
```  
Run ```python train.py -c configs/nerf_synthetic.yaml -d path_to_drums_data```, and you will see the cymbals are removed from the scene. It should be noted that the evaluation on val set and test set are not accurate due to the lack of masks for these two sets.  

<div align="center">
  <video src="https://github.com/user-attachments/assets/298ae520-6d39-4297-8611-d871e79d6e48" width="50%" />
</div>  
