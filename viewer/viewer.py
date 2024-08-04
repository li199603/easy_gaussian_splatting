import viser
from typing import Callable, Dict, List
import numpy as np
import threading
from .utils import CameraState, DelayRender
from .viewer_runtime import ViewerRuntime
from pathlib import Path


class Viewer:
    def __init__(
        self,
        render_func: Callable[[CameraState], np.ndarray],
        target_camera_states: List[CameraState],
        host: str = "localhost",
        port: int = 9981,
        in_training_mode: bool = False,
        video_output_dir: Path = Path("./output"),
    ) -> None:
        """
        render_func return a image array of shape (H, W, 3) with values ranging from 0 to 1
        """
        render_func_lock = threading.Lock()

        def render_with_lock(camera_state: CameraState) -> np.ndarray:
            with render_func_lock:
                return render_func(camera_state)

        self.render_func = render_with_lock
        self.target_camera_states = target_camera_states
        self.in_training_mode = in_training_mode
        self.video_output_dir = video_output_dir
        self.video_output_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_map: Dict[int, ViewerRuntime] = {}
        self.delay_render_map: Dict[int, DelayRender] = {}

        self.server = viser.ViserServer(host, port, verbose=False)
        self.server.on_client_connect(self._on_connect)
        self.server.on_client_disconnect(self._on_disconnect)

    def _on_connect(self, client: viser.ClientHandle):
        render_func = self.render_func
        if self.in_training_mode:
            delay_render = DelayRender(self.render_func)
            self.delay_render_map[client.client_id] = delay_render
            render_func = delay_render.get_render_image

        runtime = ViewerRuntime(
            render_func,
            client,
            self.target_camera_states,
            self.in_training_mode,
            self.video_output_dir,
        )
        runtime.start()
        self.runtime_map[client.client_id] = runtime

    def _on_disconnect(self, client: viser.ClientHandle):
        if client.client_id in self.runtime_map:
            self.runtime_map[client.client_id].stop()
            self.runtime_map[client.client_id].join()
            self.runtime_map.pop(client.client_id)

    def update_render_image(self):
        for delay_render in self.delay_render_map.values():
            delay_render.update_render_image()
