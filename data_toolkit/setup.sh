pip install pillow imageio imageio-ffmpeg tqdm easydict opencv-python-headless pandas open3d objaverse huggingface_hub[cli] open_clip_torch

# dump_mesh.py/dump_pbr.py auto-download Blender and try `sudo apt-get install` its
# runtime libs (libxrender1 libxi6 libxkbcommon-x11-0 libsm6 libxfixes3 libgl1). If you
# don't have sudo, install the equivalents into this conda env instead (no sudo needed) —
# both scripts automatically pick them up via LD_LIBRARY_PATH:
#   conda install -c conda-forge libxkbcommon xorg-libxrender xorg-libxi xorg-libsm xorg-libxfixes
