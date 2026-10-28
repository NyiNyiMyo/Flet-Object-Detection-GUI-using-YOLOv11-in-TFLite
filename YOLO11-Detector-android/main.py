import flet as ft

from views.detector_view import DetectorView

async def main(page: ft.Page):
    page.title = "YOLOv11-Detector"

    # maximize only on desktop
    if page.platform in (ft.PagePlatform.WINDOWS, ft.PagePlatform.LINUX, ft.PagePlatform.MACOS):
        page.window.maximized = True
    # linux dark mode
    page.theme_mode = ft.ThemeMode.DARK  # Forces dark mode layout
    page.dark_theme = ft.Theme(color_scheme_seed="blue") # Optional: sets a nice accent color
    
    detector_view = DetectorView(page)
    page.services.append(detector_view.image_picker)
    page.services.append(detector_view.video_picker)
    page.controls.append(detector_view.build())
    page.update()

ft.run(main)
