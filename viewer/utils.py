from viser.transforms import SE3
from typing import Tuple, Callable, List
import numpy as np
import threading
import imageio
from datetime import datetime
from pathlib import Path


def fov2focal(fov: float, pixels: float) -> float:
    return pixels / (2 * np.tan(fov / 2))


def focal2fov(focal: float, pixels: float) -> float:
    return 2 * np.arctan(pixels / (2 * focal))


def radians_norm(radians: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(radians), np.cos(radians))


class CameraState:
    def __init__(self, w2c: np.ndarray, K: np.ndarray, width: int, height: int) -> None:
        self.w2c = w2c  # colmap/opencv (X right, Y down, Z forward)
        self.K = K
        self.width = width
        self.height = height

    def fov(self) -> Tuple[float, float]:
        fov_x = focal2fov(self.K[0, 0], self.width)
        fov_y = focal2fov(self.K[1, 1], self.height)
        return fov_x, fov_y

    def roll_pitch_yaw(self) -> np.ndarray:
        rpy = SE3.from_matrix(self.w2c).inverse().rotation().as_rpy_radians()
        return np.array([rpy.roll, rpy.pitch, rpy.yaw], dtype=np.float32)

    def distance_to(self, other: "CameraState") -> float:
        self_c2w = np.linalg.inv(self.w2c)
        other_c2w = np.linalg.inv(other.w2c)
        dist = np.linalg.norm(self_c2w[:3, 3] - other_c2w[:3, 3], ord=2)
        return float(dist)

    def copy(self) -> "CameraState":
        return CameraState(self.w2c.copy(), self.K.copy(), self.width, self.height)


class DelayRender:
    def __init__(self, render_func: Callable[[CameraState], np.ndarray]) -> None:
        self.camera_states: List[CameraState] = []
        self.lock = threading.Lock()
        self.render_img = np.ones((1080, 1920, 3), np.float32)
        self.render_func = render_func

    def get_render_image(self, camera_state: CameraState) -> np.ndarray:
        with self.lock:
            self.camera_states.append(camera_state)
        return self.render_img

    def update_render_image(self):
        camera_state = None
        with self.lock:
            if len(self.camera_states) != 0:
                camera_state = self.camera_states[-1]
                self.camera_states.clear()
        if camera_state is not None:
            self.render_img = self.render_func(camera_state)


def camera_interpolation(
    camera_states: List[CameraState], duration: float, fps: float
) -> List[CameraState]:
    n = len(camera_states)
    total_frames = int(duration * fps)
    if total_frames < n:
        return camera_states

    dist_arr = np.empty((n - 1,))
    for i in range(n - 1):
        dist_arr[i] = camera_states[i].distance_to(camera_states[i + 1])
    num_frames_arr = dist_arr / dist_arr.sum() * total_frames

    default_camera_state = camera_states[0].copy()
    new_camera_states: List[CameraState] = [camera_states[0]]
    for i in range(n - 1):
        num_frames = int(num_frames_arr[i])
        if num_frames == 0:
            camera_state = default_camera_state.copy()
            camera_state.w2c = camera_states[i + 1].w2c
            new_camera_states.append(camera_state)
            continue
        start_c2w = SE3.from_matrix(camera_states[i].w2c).inverse()
        end_c2w = SE3.from_matrix(camera_states[i + 1].w2c).inverse()
        ec2sc = start_c2w.inverse() @ end_c2w
        for j in range(1, num_frames + 1):
            c2w = start_c2w @ SE3.exp(SE3.log(ec2sc) * j / num_frames)
            camera_state = default_camera_state.copy()
            camera_state.w2c = c2w.inverse().as_matrix()
            new_camera_states.append(camera_state)

    return new_camera_states


class RecordManager:
    def __init__(
        self,
        render_func: Callable[[CameraState], np.ndarray],
        duration: float,
        fps: float,
        output_dir: Path,
    ) -> None:
        self.render_func: Callable[[CameraState], np.ndarray] = render_func
        self.duration = duration
        self.fps = fps
        self.output_dir = output_dir
        self.camera_states: List[CameraState] = []

    def export_video(self) -> None:
        print("Exporting video...")
        if len(self.camera_states) <= 1:
            print("ERROR: No enough camera states to export video")
            return
        camera_states = camera_interpolation(
            self.camera_states, self.duration, self.fps
        )
        image_lst: List[np.ndarray] = []
        for camera_state in camera_states:
            image = self.render_func(camera_state) * 255.0
            image = np.floor(image).astype(np.uint8)
            image_lst.append(image)
        vedio_name = datetime.now().strftime(r"%m-%d_%H-%M-%S") + ".mp4"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        vedio_path = self.output_dir / vedio_name
        imageio.mimsave(vedio_path, image_lst, fps=self.fps)  # type: ignore
        print(f"Exported video saved at {vedio_path}")
