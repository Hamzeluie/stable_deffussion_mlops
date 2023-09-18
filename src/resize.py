import sys
import os
from pathlib import Path
import cv2
import yaml



params = yaml.safe_load(open(sys.argv[1]))["resize"]

input_dir = params["input_dir"]
output_dir = params["output_dir"]
size = params["size"]
img_type = params["img_type"]
os.makedirs(output_dir, exist_ok=True)
img_list = Path(input_dir).glob(f"**/*.{img_type}")
for p in img_list:
    img = cv2.imread(p.as_posix())
    img = cv2.resize(img, (size, size))
    out_path = f"{output_dir}/{p.name}"
    cv2.imwrite(out_path, img)
