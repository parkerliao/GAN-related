import os
import glob
from shutil import copy2
from PIL import Image
import json
import numpy as np


def copy_file(src, src_ext, dst):
    # find all files ends up with ext
    flist = sorted(glob.glob(os.path.join(src, src_ext)))
    for fname in flist:
        # src_path = os.path.join(src, fname)
        print(fname)
        copy2(fname, dst)
        print('copied %s to %s' % (fname, dst))


def isRGBequal(pixel, rgb):
    if pixel[0] is rgb[0] and pixel[1] is rgb[1] and pixel[2] is rgb[2]:
        return True
    else:
        return False


color_id = {(0, 0, 0): 0, (153, 108, 6): 1, (112, 105, 191): 2, (89, 121, 72): 3, (190, 225, 64): 4, (206, 190, 59): 5,
            (81, 13, 36): 6, (115, 176, 195): 7, (161, 171, 27): 8, (135, 169, 180): 9, (29, 26, 199): 10,
            (102, 16, 239): 11, (242, 107, 146): 12, (156, 198, 23): 13, (49, 89, 160): 14, (68, 218, 116): 15,
            (11, 236, 9): 16, (196, 30, 8): 17, (121, 67, 28): 18, (0, 53, 65): 19, (146, 52, 70): 20,
            (226, 149, 143): 21, (151, 126, 171): 22, (194, 39, 7): 23, (205, 120, 161): 24, (212, 51, 60): 25,
            (211, 80, 208): 26, (189, 135, 188): 27, (54, 72, 205): 28, (103, 252, 157): 29, (124, 21, 123): 30,
            (19, 132, 69): 31, (195, 237, 132): 32, (94, 253, 175): 33, (182, 251, 87): 34, (90, 162, 242): 35}


def color2label(clabel_root, label_root, inst_root):
    clabel_list = sorted(glob.glob(os.path.join(clabel_root, '*.png')))

    for clabel in clabel_list:

        clabel_map = Image.open(clabel).convert('RGB')
        clabel_map = np.array(clabel_map, dtype=np.int32)
        label_map = np.zeros((clabel_map.shape[0], clabel_map.shape[1]))
        instance_map = np.zeros((clabel_map.shape[0], clabel_map.shape[1]),dtype=np.int32)
        for i in range(clabel_map.shape[0]):
            for j in range(clabel_map.shape[1]):
                try:
                    label_map[i][j] = color_id[(clabel_map[i][j][0], clabel_map[i][j][1], clabel_map[i][j][2])]
                    if label_map[i][j] == 34:
                        instance_map[i][j] = 34000
                    else:
                        instance_map[i][j] = 0
                except:
                    label_map[i][j] = 0
        temp_label_img = Image.fromarray(label_map).convert('L')
        temp_inst_img = Image.fromarray(instance_map).convert('I')

        filename = os.path.splitext(os.path.basename(clabel))[0]
        filename = filename.split('_')[0]

        label_savename = os.path.join(label_root, filename + "_label" + '.png')
        inst_savename = os.path.join(inst_root, filename + '_inst' + '.png')
        temp_label_img.save(label_savename, 'png')
        temp_inst_img.save(inst_savename, 'png')
        print(filename + " was successfully converted")


# def color2instance():


def construct_box(inst_root, cls_root, dst):
    inst_list = sorted(glob.glob(os.path.join(inst_root, '*.png')))
    cls_list = sorted(glob.glob(os.path.join(cls_root, '*.png')))
    for inst, cls in zip(*(inst_list, cls_list)):
        inst_map = Image.open(inst)
        inst_map = np.array(inst_map, dtype=np.int32)
        cls_map = Image.open(cls)
        cls_map = np.array(cls_map, dtype=np.int32)
        H, W = inst_map.shape
        # get a list of unique instances
        inst_info = {'imgHeight': H, 'imgWidth': W, 'objects': {}}
        inst_ids = np.unique(inst_map)
        print(inst_ids)
        for iid in inst_ids:
            if int(iid) < 1000:  # filter out non-instance masks
                continue
            ys, xs = np.where(inst_map == iid)
            ymin, ymax, xmin, xmax = \
                ys.min(), ys.max(), xs.min(), xs.max()
            cls_label = np.median(cls_map[inst_map == iid])
            inst_info['objects'][str(iid)] = {'bbox': [xmin.item(), ymin.item(), xmax.item(), ymax.item()],
                                              'cls': int(cls_label)}
        # write a file to path
        filename = os.path.splitext(os.path.basename(inst))[0]
        savename = os.path.join(dst, filename + '.json')
        with open(savename, 'w') as f:
            json.dump(inst_info, f)
        print('wrote a bbox summary of %s to %s' % (inst, savename))


# organize image
if __name__ == '__main__':

    folder_name = 'datasets/airsim/'
    train_img_dst = os.path.join(folder_name, 'train_img')
    train_clabel_dst = os.path.join(folder_name, 'train_clabel')
    train_label_dst = os.path.join(folder_name, 'train_label')
    train_inst_dst = os.path.join(folder_name, 'train_inst')
    train_bbox_dst = os.path.join(folder_name, 'train_bbox')
    val_img_dst = os.path.join(folder_name, 'val_img')
    val_label_dst = os.path.join(folder_name, 'val_label')
    val_inst_dst = os.path.join(folder_name, 'val_inst')
    val_bbox_dst = os.path.join(folder_name, 'val_bbox')

    if not os.path.exists(train_img_dst):
        os.makedirs(train_img_dst)
    if not os.path.exists(train_clabel_dst):
        os.makedirs(train_clabel_dst)
    if not os.path.exists(train_label_dst):
        os.makedirs(train_label_dst)
    if not os.path.exists(train_inst_dst):
        os.makedirs(train_inst_dst)
    if not os.path.exists(val_img_dst):
        os.makedirs(val_img_dst)
    if not os.path.exists(val_label_dst):
        os.makedirs(val_label_dst)
    if not os.path.exists(val_inst_dst):
        os.makedirs(val_inst_dst)

    # train_image
    copy_file('./datasets/airsim/gt/train', \
              '*_gt.png', train_img_dst)
    copy_file('./datasets/airsim/clabel/train', \
              '*_clabel.png', train_clabel_dst)

    color2label('./datasets/airsim/clabel/train', './datasets/airsim/train_label', './datasets/airsim/train_inst')

    # train_label
    # copy_file('./datasets/airsim/label/train', \
    #           '*.png', train_label_dst)
    # # train_inst
    # copy_file('./datasets/cityscape/gtFine/train', \
    #           '*_instanceIds.png', train_inst_dst)
    # # val_image
    # copy_file('./datasets/cityscape/leftImg8bit/val', \
    #           '*_leftImg8bit.png', val_img_dst)
    # # val_label
    # copy_file('./datasets/cityscape/gtFine/val', \
    #           '*_labelIds.png', val_label_dst)
    # # val_inst
    # copy_file('./datasets/cityscape/gtFine/val', \
    #           '*_instanceIds.png', val_inst_dst)

    if not os.path.exists(train_bbox_dst):
        os.makedirs(train_bbox_dst)
    # if not os.path.exists(val_bbox_dst):
    #     os.makedirs(val_bbox_dst)
    # wrote a bounding box summary

    construct_box('datasets/airsim/train_inst', 'datasets/airsim/train_label',
                  train_bbox_dst)
    # construct_box('datasets/cityscape/gtFine/val', \
    #               '*_instanceIds.png', '*_labelIds.png', val_bbox_dst)
