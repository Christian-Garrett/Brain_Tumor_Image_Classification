import os
import sys
from pathlib import Path

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

IMAGE_FOLDERS = ["glioma_tumor",
                 "meningioma_tumor",
                 "no_tumor",
                 "pituitary_tumor"]

CAP_CHECKPT_LOG_DIR = \
    os.path.join(module_path, "Output/Checkpointing/Capsule/Capsule")

CONV_CHECKPT_LOG_DIR = \
    os.path.join(module_path, "Output/Checkpointing/Convolutional")

AUGMENT_DIFF_THRESH = 50  # count diff threshold for increasing the size of an 
                          # image set through data augmentation

AUG_DATA_PREFIX = 'aug_scan_'  # the prefix added to augmented data to differentiate 
                               # from originals
