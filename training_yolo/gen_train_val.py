import pathlib
import random
from pathlib import Path

# the ratio of val set over whole dataset
# MANUALLY EDIT
RATIO = 0.25

YOLO_DATA_DIR = Path('./data')
YOLO_IMG_DIR = YOLO_DATA_DIR / 'img'

if __name__ == '__main__':
    # get all filename in folder
    name_list = [(fn.as_posix() + '\n') for fn in YOLO_IMG_DIR.glob('*.jpg')]

    # shuffle the name list
    random.shuffle(name_list)

    # get number of val set
    num_val = int(len(name_list) * RATIO)
    train_list = name_list[:-num_val]
    val_list = name_list[-num_val:]

    with open(str(YOLO_DATA_DIR / 'train.txt'), 'w') as f:
        f.writelines(train_list)

    with open(str(YOLO_DATA_DIR / 'val.txt'), 'w') as f:
        f.writelines(val_list)

    print('Done!')
