import asyncio
from pathlib import Path

import cv2
import flet as ft
import numpy as np

from models import YoloTFLiteDetector
import base64

import platform
import flet_camera as fc

class DetectorView:
    """Flet source controls; shared detector logic remains in models.py."""

    def __init__(self, page: ft.Page):
        self._page = page
        model_path = Path(__file__).parent.parent / "assets" / "yolo11n_float32.tflite"
        self.detector = YoloTFLiteDetector(model_path, confidence=0.2) # due to 320 size
        self.is_android = platform.system().lower() not in ["windows", "darwin", "linux"] 
        self.image_picker = ft.FilePicker()
        self.video_picker = ft.FilePicker()
        self.mobile_camera = fc.Camera(
            preview_enabled=False,   # hide the preview UI, but keep the control mounted
            width=1, height=1,       # or fold it into normal layout
            on_stream_image=self._on_android_frame,
        )
        self.available_cameras: list[fc.CameraDescription] = []
        self.current_camera: fc.CameraDescription | None = None
        self.camera_lens = fc.CameraLensDirection.BACK  # default facing
        self.swap_camera_button = ft.Button(
            content="Swap Camera",
            bgcolor=ft.Colors.BLUE,
            color=ft.Colors.WHITE,
            on_click=self.toggle_camera_facing,
            visible=False,  # shown only once webcam is running on Android
        )
        self.preview = ft.Image(
            src="icon.png",
            visible=False,
            width=1280,
            height=720,
            fit=ft.BoxFit.CONTAIN,
            gapless_playback=True,
        )
        self.status = ft.Text("Ready")
        self.video_button = ft.Button(
            content="Select Video", bgcolor=ft.Colors.RED, color=ft.Colors.WHITE, on_click=self.toggle_video
        )
        self.webcam_button = ft.Button(
            content="Start Webcam", bgcolor=ft.Colors.BLUE, color=ft.Colors.WHITE, on_click=self.toggle_webcam
        )
        self.running = False
        self.active_source: str | None = None
        self.source_task: asyncio.Task | None = None

    def build(self) -> ft.Column:
        preview_frame = ft.Container(
            content=self.preview,
            width=1280,
            height=720,
            alignment=ft.Alignment.CENTER,
            border=ft.Border.all(1, ft.Colors.BLUE),
            clip_behavior=ft.ClipBehavior.HARD_EDGE,
        )
        return ft.Column(
            controls=[
                ft.Text("📟 Makers - YOLOv11 TFLite Detector", color=ft.Colors.WHITE),
                self.status,
                ft.Container(content=preview_frame, alignment=ft.Alignment.CENTER, expand=True),
                ft.Slider(
                    min=0.1,
                    max=0.95,
                    divisions=17,
                    value=0.2,
                    label="Confidence: {value}",
                    round=2,
                    active_color=ft.Colors.BLUE,
                    inactive_color=ft.Colors.WHITE,
                    thumb_color=ft.Colors.RED,
                    on_change=self.change_confidence,
                ),
                ft.Row(
                    controls=[
                        ft.Button(content="Select Image", bgcolor=ft.Colors.RED, color=ft.Colors.WHITE, on_click=self.select_image),
                        self.video_button,
                        self.webcam_button,
                        self.swap_camera_button,
                        ft.Button(content="Clear", bgcolor=ft.Colors.WHITE, color=ft.Colors.BLUE, on_click=self.clear),
                    ],
                    alignment=ft.MainAxisAlignment.CENTER,
                    wrap=True,
                ),
                self.mobile_camera if self.is_android else ft.Container()
            ],
            expand=True,
        )

    def change_confidence(self, event) -> None:
        self.detector.confidence = float(event.control.value)

    async def select_image(self, _event) -> None:
        self._stop_current()
        files = await self.image_picker.pick_files(
            dialog_title="Select an image", file_type=ft.FilePickerFileType.IMAGE, with_data=True
        )
        if not files:
            return
        selected = files[0]
        self.status.value = f"Running inference on {selected.name}..."
        self._page.update()
        try:
            if selected.path:
                frame = await asyncio.to_thread(cv2.imread, selected.path)
            else:
                frame = cv2.imdecode(np.frombuffer(selected.bytes or b"", dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("The selected image could not be decoded.")
            annotated, count = await asyncio.to_thread(self.detector.annotate, frame)
            self._show_frame(annotated)
            self.status.value = f"{count} object(s) detected in {selected.name}."
        except Exception as error:
            self.status.value = f"Inference failed: {error}"
        self._page.update()

    async def toggle_video(self, _event) -> None:
        if self.active_source == "video":
            self._stop_current("Video stopped.")
            self._page.update()
            return
        self._stop_current()
        files = await self.video_picker.pick_files(dialog_title="Select a video", file_type=ft.FilePickerFileType.VIDEO)
        if not files:
            return
        if not files[0].path:
            self.status.value = "Video inference needs a local video path on this backend."
            self._page.update()
            return
        self.running = True
        self.active_source = "video"
        self.video_button.content = "Stop video"
        self.status.value = f"Running video inference: {files[0].name}."
        self._page.update()
        self.source_task = asyncio.create_task(self._video_loop(files[0].path, files[0].name))

    async def toggle_webcam(self, _event) -> None:
        if self.active_source == "webcam":
            self._stop_current("Webcam stopped.")
            self._page.update()
            return
        self._stop_current()
        self.running = True
        self.active_source = "webcam"
        self.webcam_button.content = "Stop webcam"
        self.status.value = "Opening webcam..."
        self._page.update()
        self.source_task = asyncio.create_task(self._camera_loop())

    def clear(self, _event) -> None:
        self._stop_current("Canvas cleared.")
        self.preview.visible = False
        self._page.update()
        
    def _stop_current(self, message: str | None = None) -> None:
        self.running = False

        if self.is_android and hasattr(self, 'mobile_camera'):
            try:
                asyncio.create_task(self.mobile_camera.stop_image_stream())
            except Exception:
                pass

        if self.source_task and not self.source_task.done():
            self.source_task.cancel()

        self.source_task = None
        self.active_source = None
        self.swap_camera_button.visible = False
        self.video_button.content = "Select Video"
        self.webcam_button.content = "Start Webcam"
        if message:
            self.status.value = message

    async def _video_loop(self, path: str, name: str) -> None:
        capture = None
        try:
            capture = cv2.VideoCapture(path)
            if not capture.isOpened():
                self._stop_current(f"Could not open {name}. Check the video's codec.")
                self._page.update()
                return

            while self.running and self.active_source == "video":
                ok, frame = await asyncio.to_thread(capture.read)
                if not ok:
                    self._stop_current("Video finished or could not be decoded.")
                    self._page.update()
                    break
                
                # --- CROSS-PLATFORM ANDROID COLOR PATCH ---
                #if self.is_android:
                    # Android VideoCapture usually decodes as RGBA or inverse RGB channels
                    # If colors are still slightly off (e.g. blue faces), switch to cv2.COLOR_RGBA2BGR
                    #frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # ------------------------------------------

                annotated, count = await asyncio.to_thread(self.detector.annotate, frame)
                self._show_frame(annotated)
                self.status.value = f"Video inference: {count} object(s) detected."
                self._page.update()

                await asyncio.sleep(0.05)
        except asyncio.CancelledError:
            pass
        except Exception as error:
            self._stop_current(f"Video inference failed: {error}")
            self._page.update()
        finally:
            if capture is not None:
                capture.release()

    async def _camera_loop(self) -> None:
        if not self.is_android:
            # --- DESKTOP WINDOWS PIPELINE ---
            capture = cv2.VideoCapture(0)
            if not capture.isOpened():
                self._stop_current("Could not open the default webcam.")
                self._page.update()
                return
            try:
                while self.running and self.active_source == "webcam":
                    ok, frame = await asyncio.to_thread(capture.read)
                    if not ok:
                        self._stop_current("Webcam frame could not be read.")
                        break
                    annotated, count = await asyncio.to_thread(self.detector.annotate, frame)
                    self._show_frame(annotated)
                    self.status.value = f"Webcam inference: {count} object(s) detected."
                    self._page.update()
            except asyncio.CancelledError:
                pass
            except Exception as error:
                self._stop_current(f"Webcam inference failed: {error}")
                self._page.update()
            finally:
                capture.release()            
        else:
            # --- MOBILE ANDROID PIPELINE ---
            try:
                self.status.value = "Initializing camera..."
                self._page.update()

                if not self.available_cameras:
                    self.available_cameras = await self.mobile_camera.get_available_cameras()
                if not self.available_cameras:
                    self._stop_current("No camera found on this device/emulator.")
                    self._page.update()
                    return

                await self._start_camera_with_lens(self.camera_lens)

                if self.is_android:
                    self.swap_camera_button.visible = True

                self.status.value = "Starting object detector stream..."
                self._page.update()
                await self.mobile_camera.start_image_stream()

            except Exception as init_error:
                self._stop_current(f"Camera setup failed: {init_error}")
                self._page.update()

    async def _start_camera_with_lens(self, lens: fc.CameraLensDirection) -> None:
        camera = next(
            (c for c in self.available_cameras if c.lens_direction == lens),
            self.available_cameras[0],  # fall back to whatever exists
        )
        self.current_camera = camera
        await self.mobile_camera.initialize(
            description=camera,
            resolution_preset=fc.ResolutionPreset.MEDIUM,
            enable_audio=False,
        )
            
    def _show_frame(self, frame: np.ndarray) -> None:
        ok, encoded = cv2.imencode(".jpg", frame)
        if not ok:
            raise ValueError("Could not encode the annotated frame.")

        image_data = base64.b64encode(encoded).decode("ascii")

        self.preview.src = f"data:image/jpeg;base64,{image_data}"
        self.preview.visible = True
            
    async def toggle_camera_facing(self, _event) -> None:
        if self.active_source != "webcam" or not self.is_android:
            return
        self.camera_lens = (
            fc.CameraLensDirection.FRONT
            if self.camera_lens == fc.CameraLensDirection.BACK
            else fc.CameraLensDirection.BACK
        )
        self.status.value = f"Switching to {self.camera_lens.value} camera..."
        self._page.update()
        try:
            await self.mobile_camera.stop_image_stream()
            await self._start_camera_with_lens(self.camera_lens)
            await self.mobile_camera.start_image_stream()
            self.status.value = f"{self.camera_lens.value.capitalize()} camera active."
        except Exception as error:
            self.status.value = f"Camera swap failed: {error}"
        self._page.update()
        
    def _rotate_for_sensor(self, frame: np.ndarray) -> np.ndarray:
        camera = self.current_camera
        if camera is None:
            return frame

        angle = camera.sensor_orientation % 360  # portrait-device assumption; see note below
        if angle == 90:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif angle == 180:
            frame = cv2.rotate(frame, cv2.ROTATE_180)
        elif angle == 270:
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

        if camera.lens_direction == fc.CameraLensDirection.FRONT:
            frame = cv2.flip(frame, 1)  # mirror, so it matches what the user sees in a mirror

        return frame

    def _on_android_frame(self, e: fc.CameraImageEvent) -> None:
        if not self.running or self.active_source != "webcam":
            return
        try:
            nparr = np.frombuffer(e.bytes, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is not None:
                frame = self._rotate_for_sensor(frame)
                annotated, count = self.detector.annotate(frame)
                self._show_frame(annotated)
                self.status.value = f"Webcam inference: {count} object(s) detected."
                self._page.update()
        except Exception as error:
            print(f"Mobile frame processing error: {error}")
