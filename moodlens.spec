# moodlens.spec  —  PyInstaller build spec for MoodLens-CV
# Run with:  pyinstaller moodlens.spec

import sys
from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# ── Extra data files bundled into the app ────────────────────────────────────
added_files = [
    ("face_landmarker.task", "."),
    ("stress_alert.mp3",     "."),
]

# Collect mediapipe model/data files
added_files += collect_data_files("mediapipe")

# Collect deepface model configs
added_files += collect_data_files("deepface")

a = Analysis(
    ["moodlens_gui.py"],
    pathex=[str(Path(".").resolve())],
    binaries=collect_dynamic_libs("cv2") + collect_dynamic_libs("mediapipe"),
    datas=added_files,
    hiddenimports=[
        # mediapipe
        "mediapipe.tasks.python",
        "mediapipe.tasks.python.vision",
        "mediapipe.python.solutions",
        # deepface backends
        "deepface",
        "deepface.modules",
        "deepface.models",
        "tensorflow",
        # sklearn
        "sklearn.ensemble",
        "sklearn.preprocessing",
        "sklearn.utils._cython_blas",
        "sklearn.neighbors._typedefs",
        "sklearn.neighbors._quad_tree",
        "sklearn.tree._utils",
        # PyQt5 multimedia
        "PyQt5.QtMultimedia",
        # pynput
        "pynput",
        "pynput.mouse",
        "pynput.keyboard",
        # misc
        "psutil",
    ],
    hookspath=[],
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MoodLens",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,          # no terminal window
    icon="icon.ico",        # optional — remove line if no icon file
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MoodLens",
)
