# 01_setup.py -- run every session
# Paste this whole file into ONE Colab cell and run it.
# Mounts Drive and sets the dataset path used by all other cells.

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
DATA = Path("/content/drive/MyDrive/capstone/dataset_final")   # has train/ test/ brain_subplanes/
