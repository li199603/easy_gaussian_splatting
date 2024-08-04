import viser
from viser.transforms import SE3, SO3
from typing import Callable, Dict, List, Literal, Tuple
import numpy as np
import threading
from .utils import CameraState, fov2focal, radians_norm, RecordManager
import time
from pathlib import Path

ActionType = Literal["static", "move", "update"]
RenderPolicyType = Literal["none", "normal"]


class RenderTask:
    def __init__(self, camera_state: CameraState, action: ActionType) -> None:
        self.camera_state = camera_state
        self.action = action


class ViewerRuntime(threading.Thread):
    def __init__(
        self,
        render_func: Callable[[CameraState], np.ndarray],
        client: viser.ClientHandle,
        target_camera_states: List[CameraState],
        in_training_mode: bool,
        video_output_dir: Path,
        default_fov: float = np.pi / 2,
        default_hw: Tuple[int, int] = (1080, 1920),
        static_time: float = 0.2,
        update_time: float = 1.0,
    ) -> None:
        super().__init__()
        self.render_func = render_func
        self.client = client
        self.target_camera_states = target_camera_states
        self.in_training_mode = in_training_mode
        self.static_time = static_time
        self.update_time = update_time
        self.running = False

        self.tasks: List[RenderTask] = []
        self.tasks_lock = threading.Lock()
        self.tasks_cond = threading.Condition(self.tasks_lock)
        self.render_policy: RenderPolicyType = "normal"
        self.record_manager = RecordManager(
            self.render_func, duration=5, fps=30, output_dir=video_output_dir
        )

        # these member could be modified through the gui
        self.fov_x, self.fov_y = default_fov, default_fov
        self.height, self.width = default_hw
        self.rotation_radian = 5 / 180 * np.pi

        self.state_machine: Dict[
            RenderPolicyType, Dict[ActionType, RenderPolicyType]
        ] = {
            "none": {"static": "none", "move": "normal", "update": "normal"},
            "normal": {"static": "none", "move": "normal", "update": "normal"},
        }
        if len(self.target_camera_states) != 0:
            camera_state = self.target_camera_states[0]
            target_c2w = SE3.from_matrix(camera_state.w2c).inverse()
            self.client.camera.wxyz = target_c2w.rotation().wxyz
            self.client.camera.position = target_c2w.translation()
            self.fov_x, self.fov_y = camera_state.fov()
            self.height, self.width = camera_state.height, camera_state.width
        self._define_gui()
        self.client.camera.on_update(self._on_update)

    def _on_update(self, camera: viser.CameraHandle):
        self.submit_task(RenderTask(self._get_camera_state(camera), "move"))

    def run(self):
        self.running = True
        elapsed_time = 0.0
        while self.running:
            t0 = time.time()
            with self.tasks_cond:
                self.tasks_cond.wait(self.static_time)
                if not self.running:
                    break
                if len(self.tasks) == 0:
                    self.tasks.append(
                        RenderTask(self._get_camera_state(self.client.camera), "static")
                    )
                if self.in_training_mode and elapsed_time > self.update_time:
                    elapsed_time = 0.0
                    self.tasks.append(
                        RenderTask(self._get_camera_state(self.client.camera), "update")
                    )
                while len(self.tasks) != 0:
                    task = self.tasks.pop(0)
                    action_trans = self.state_machine[self.render_policy]
                    self.render_policy = action_trans[task.action]
                t1 = time.time()
                elapsed_time += t1 - t0
            if self.render_policy == "none":
                continue
            image = self.render_func(task.camera_state)
            image = self.adjust_image_aspect(image, self.client.camera.aspect)
            self.client.scene.set_background_image(image)

    def adjust_image_aspect(self, image: np.ndarray, aspect: float) -> np.ndarray:
        h, w = image.shape[:2]
        if w / h < aspect:
            new_h = h
            new_w = int(h * aspect)
        elif w / h > aspect:
            new_h = int(w / aspect)
            new_w = w
        else:
            new_h, new_w = h, w
        new_image = np.zeros((new_h, new_w, 3), dtype=np.float32)
        new_image[:h, :w] = image
        return new_image

    def stop(self):
        self.running = False
        with self.tasks_cond:
            self.tasks_cond.notify()

    def submit_task(self, task: RenderTask):
        with self.tasks_cond:
            self.tasks.append(task)
            self.tasks_cond.notify()

    def _skip_to_target_camera(self, camera_state: CameraState):
        target_c2w = SE3.from_matrix(camera_state.w2c).inverse()
        with self.client.atomic():
            self.client.camera.wxyz = target_c2w.rotation().wxyz
            self.client.camera.position = target_c2w.translation()

    def _get_camera_state(self, camera: viser.CameraHandle) -> CameraState:
        with self.client.atomic():
            c2w = SE3.from_rotation_and_translation(
                SO3(camera.wxyz), camera.position
            ).as_matrix()
            w2c = np.linalg.inv(c2w)
            height, width = self.height, self.width
            fx = fov2focal(self.fov_x, width)
            fy = fov2focal(self.fov_y, height)
            K = np.array(
                [[fx, 0, width / 2], [0, fy, height / 2], [0, 0, 1]], dtype=np.float32
            )
            return CameraState(w2c, K, width, height)

    def get_closest_index(self, camera_state: CameraState) -> int:
        closest_index = 0
        min_dist = float("inf")
        for i in range(len(self.target_camera_states)):
            dist = camera_state.distance_to(self.target_camera_states[i])
            if dist < min_dist:
                min_dist = dist
                closest_index = i
        return closest_index

    def _camera_rotation(
        self, direction: Literal["roll+", "roll-", "pitch+", "pitch-", "yaw+", "yaw-"]
    ):
        if direction == "roll+":
            so3 = SO3.from_z_radians(-self.rotation_radian)
        elif direction == "roll-":
            so3 = SO3.from_z_radians(self.rotation_radian)
        elif direction == "pitch+":
            so3 = SO3.from_x_radians(-self.rotation_radian)
        elif direction == "pitch-":
            so3 = SO3.from_x_radians(self.rotation_radian)
        elif direction == "yaw+":
            so3 = SO3.from_y_radians(-self.rotation_radian)
        elif direction == "yaw-":
            so3 = SO3.from_y_radians(self.rotation_radian)
        else:
            raise ValueError(f"Unknown direction: {direction}")

        cur_camera_state = self._get_camera_state(self.client.camera)
        tagete_w2c = SE3.from_rotation(so3).as_matrix() @ cur_camera_state.w2c
        target_camera_state = CameraState(
            tagete_w2c,
            cur_camera_state.K,
            cur_camera_state.width,
            cur_camera_state.height,
        )
        self._skip_to_target_camera(target_camera_state)

    def _define_gui(self):
        with self.client.gui.add_folder("Camera Params"):
            gui_number_height = self.client.gui.add_number(
                "height", initial_value=self.height, min=1, step=1
            )
            gui_number_width = self.client.gui.add_number(
                "width", initial_value=self.width, min=1, step=1
            )
            gui_number_fov_x = self.client.gui.add_number(
                "fov x", initial_value=self.fov_x / np.pi * 180, min=0, max=180, step=1
            )
            gui_number_fov_y = self.client.gui.add_number(
                "fov y", initial_value=self.fov_y / np.pi * 180, min=0, max=180, step=1
            )
        with self.client.gui.add_folder("Rotation"):
            gui_number_rotation_angle = self.client.gui.add_number(
                "Rotation Angle",
                initial_value=self.rotation_radian / np.pi * 180,
                min=0,
                max=360,
                step=1,
            )
            gui_button_roll_pos = self.client.gui.add_button("roll+")
            gui_button_roll_neg = self.client.gui.add_button("roll-")
            gui_button_pitch_pos = self.client.gui.add_button("pitch+")
            gui_button_pitch_neg = self.client.gui.add_button("pitch-")
            gui_button_yaw_pos = self.client.gui.add_button("yaw+")
            gui_button_yaw_neg = self.client.gui.add_button("yaw-")

        @gui_number_height.on_update
        def _(_):
            self.height = max(int(gui_number_height.value), 1)
            self.submit_task(
                RenderTask(self._get_camera_state(self.client.camera), "update")
            )

        @gui_number_width.on_update
        def _(_):
            self.width = max(int(gui_number_width.value), 1)
            self.submit_task(
                RenderTask(self._get_camera_state(self.client.camera), "update")
            )

        @gui_number_fov_x.on_update
        def _(_):
            self.fov_x = min(max(gui_number_fov_x.value, 0), 180) / 180 * np.pi
            self.submit_task(
                RenderTask(self._get_camera_state(self.client.camera), "update")
            )

        @gui_number_fov_y.on_update
        def _(_):
            self.fov_y = min(max(gui_number_fov_y.value, 0), 180) / 180 * np.pi
            self.submit_task(
                RenderTask(self._get_camera_state(self.client.camera), "update")
            )

        @gui_number_rotation_angle.on_update
        def _(_):
            self.rotation_radian = (
                min(max(gui_number_rotation_angle.value, 0), 360) / 180 * np.pi
            )

        @gui_button_roll_pos.on_click
        def _(_):
            self._camera_rotation("roll+")

        @gui_button_roll_neg.on_click
        def _(_):
            self._camera_rotation("roll-")

        @gui_button_pitch_pos.on_click
        def _(_):
            self._camera_rotation("pitch+")

        @gui_button_pitch_neg.on_click
        def _(_):
            self._camera_rotation("pitch-")

        @gui_button_yaw_pos.on_click
        def _(_):
            self._camera_rotation("yaw+")

        @gui_button_yaw_neg.on_click
        def _(_):
            self._camera_rotation("yaw-")

        with self.client.gui.add_folder(
            "Target Cameras", visible=len(self.target_camera_states) != 0
        ):
            gui_number_target_camera_index = self.client.gui.add_number(
                "camera index",
                initial_value=1,
                min=1,
                max=len(self.target_camera_states),
                step=1,
            )
            gui_button_skip_to_closest = self.client.gui.add_button("skip to closest")

        @gui_number_target_camera_index.on_update
        def _(_):
            index = min(
                max(gui_number_target_camera_index.value, 1),
                len(self.target_camera_states),
            )
            target_camera_state = self.target_camera_states[index - 1]
            gui_number_height.value = target_camera_state.height
            gui_number_width.value = target_camera_state.width
            fox_x, fov_y = target_camera_state.fov()
            gui_number_fov_x.value = fox_x / np.pi * 180
            gui_number_fov_y.value = fov_y / np.pi * 180
            self._skip_to_target_camera(target_camera_state)

        @gui_button_skip_to_closest.on_click
        def _(_):
            cur_camera_state = self._get_camera_state(self.client.camera)
            closest_index = self.get_closest_index(cur_camera_state)
            target_camera_state = self.target_camera_states[closest_index]
            gui_number_height.value = target_camera_state.height
            gui_number_width.value = target_camera_state.width
            fox_x, fov_y = target_camera_state.fov()
            gui_number_fov_x.value = fox_x / np.pi * 180
            gui_number_fov_y.value = fov_y / np.pi * 180
            gui_number_target_camera_index.value = closest_index + 1
            self._skip_to_target_camera(target_camera_state)

        with self.client.gui.add_folder(
            "Export Video", visible=not self.in_training_mode, expand_by_default=False
        ):
            gui_number_export_camera_index = self.client.gui.add_number(
                "camera index",
                initial_value=0,
                min=0,
                step=1,
            )
            gui_button_add_camera = self.client.gui.add_button("add camera")
            gui_button_clear_camera = self.client.gui.add_button("clear cameras")
            gui_number_record_duration = self.client.gui.add_number(
                "record duration",
                initial_value=self.record_manager.duration,
                min=1,
                step=0.1,
            )
            gui_number_record_fps = self.client.gui.add_number(
                "record fps",
                initial_value=self.record_manager.fps,
                min=1,
                step=1,
            )
            gui_button_export = self.client.gui.add_button("export")

        @gui_number_export_camera_index.on_update
        def _(_):
            if len(self.record_manager.camera_states) == 0:
                return
            index = gui_number_export_camera_index.value
            if index <= 0 or index > len(self.record_manager.camera_states):
                return

            target_camera_state = self.record_manager.camera_states[index - 1]
            gui_number_height.value = target_camera_state.height
            gui_number_width.value = target_camera_state.width
            fox_x, fov_y = target_camera_state.fov()
            gui_number_fov_x.value = fox_x / np.pi * 180
            gui_number_fov_y.value = fov_y / np.pi * 180
            self._skip_to_target_camera(target_camera_state)

        @gui_button_add_camera.on_click
        def _(_):
            self.record_manager.camera_states.append(
                self._get_camera_state(self.client.camera)
            )

        @gui_button_clear_camera.on_click
        def _(_):
            self.record_manager.camera_states.clear()

        @gui_number_record_duration.on_update
        def _(_):
            self.record_manager.duration = max(gui_number_record_duration.value, 1)

        @gui_number_record_fps.on_update
        def _(_):
            self.record_manager.fps = max(gui_number_record_fps.value, 1)

        @gui_button_export.on_click
        def _(_):
            self.record_manager.export_video()
