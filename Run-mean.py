import csv
import glob
import os
from os.path import join, isdir
from Evaluation.Test_GTOT import val_GTOT, get_exp
from Evaluation.Test_RGBT234 import val_RGBT234, get_txt
import tools
from tracker_mean_mutil import *
import numpy as np
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import math


def set_seed(seed_a, seed_b, seed_c):
    np.random.seed(seed_a)
    torch.manual_seed(seed_b)
    torch.cuda.manual_seed(seed_c)


global logger


def genConfig(seq_path, set_type):
    path, seqname = os.path.split(seq_path)

    if set_type == 'RGBT234':
        ############################################  have to refine #############################################

        img_list_v = sorted(
            [seq_path + '/visible/' + p for p in os.listdir(seq_path + '/visible') if os.path.splitext(p)[1] == '.jpg'])
        img_list_t = sorted([seq_path + '/infrared/' + p for p in os.listdir(seq_path + '/infrared') if
                             os.path.splitext(p)[1] == '.jpg'])
        gt = np.loadtxt(seq_path + '/visible.txt', delimiter=',')
        ir = np.loadtxt(seq_path + '/infrared.txt', delimiter=',')
        ##polygon to rect
    if set_type == 'GTOT':
        img_list_v = sorted(
            [seq_path + '/v/' + p for p in os.listdir(seq_path + '/v') if
             os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        img_list_t = sorted([seq_path + '/i/' + p for p in os.listdir(seq_path + '/i') if
                             os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']])
        gt = np.loadtxt(seq_path + '/init.txt', delimiter='\t')
        ir = np.loadtxt(seq_path + '/groundTruth_i.txt', delimiter=' ')
        ir[:, 2] = ir[:, 2] - ir[:, 0]
        ir[:, 3] = ir[:, 3] - ir[:, 1]
    if gt.shape[1] == 8:
        x_min = np.min(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_min = np.min(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        x_max = np.max(gt[:, [0, 2, 4, 6]], axis=1)[:, None]
        y_max = np.max(gt[:, [1, 3, 5, 7]], axis=1)[:, None]
        gt = np.concatenate((x_min, y_min, x_max - x_min, y_max - y_min), axis=1)

    return img_list_v, img_list_t, gt, ir


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default='RGBT234')
    parser.add_argument("-result_path", default='./result.npy')
    parser.add_argument("-visual_log", default=False, action='store_true')
    parser.add_argument("-visualize", default=False, action='store_true')
    parser.add_argument("-adaptive_align", default=True, action='store_false')
    parser.add_argument("-padding", default=1.2, type=float)
    parser.add_argument("-jitter", default=True, action='store_false')
    parser.add_argument("-exp", default=2)
    parser.add_argument("-weight", default='DAPNet_91_0.9948')
    parser.add_argument("-seed_a", default=123, type=int)
    parser.add_argument("-seed_b", default=456, type=int)
    parser.add_argument("-seed_c", default=789, type=int)
    parser.add_argument("-lr_mult", default=10, type=float)
    parser.add_argument("-scale_large", default=False, action='store_true')
    args = parser.parse_args()

    ##################################################################################
    #########################Just modify opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ## option setting
    model_result = args.weight
    num = len(glob.glob('./runs/track/*/'))
    if not os.path.exists(os.path.join('./runs/track/exp' + str(num))):
        os.mkdir(os.path.join('./runs/track/exp' + str(num)))
        os.mkdir(os.path.join('./runs/track/exp' + str(num), model_result))
    opts['result_path'] = args.result_path
    opts['visual_log'] = args.visual_log
    opts['set_type'] = args.set_type
    opts['visualize'] = args.visualize
    opts['adaptive_align'] = args.adaptive_align
    opts['padding'] = args.padding
    opts['jitter'] = args.jitter
    opts['model_path_all'] = './runs/train/exp' + str(args.exp) + '/weights/' + model_result + '.pth'
    opts['save_proj'] = './runs/track/exp' + str(num)
    opts['lr_mult']['branches'] = args.lr_mult
    opts['lr_mult']['instance_conv'] = 10
    opts['scale_large'] = args.scale_large
    seed_a = args.seed_a
    seed_b = args.seed_b
    seed_c = args.seed_c
    info = "lr:" + str(opts['lr_mult']['branches']) + "_seed_init: " + str(seed_a) + ',' + str(seed_b) + ',' + str(
        seed_c) + ',' + str(args.scale_large)
    opts['info'] = info
    tools.format_dictionary(opts)

    channel = [96, 256, 512]
    rate = [1, 1, 1]
    topk = [int((1 - rate[i]) * channel[i]) for i in range(3)]
    print(str(rate))

    tools.log_csv(opts)

    dataset_path = '/media/lx/Disk_data/LRJ_data/DATA/'

    seq_home = dataset_path + opts['set_type']
    seq_list = sorted([f for f in os.listdir(seq_home) if isdir(join(seq_home, f))])
    iou_list = []
    dis_list = []
    len_iou_list = 0
    fps_list = dict()
    bb_result = dict()
    result = dict()

    iou_list_nobb = []
    bb_result_nobb = dict()
    set_seed(seed_a, seed_b, seed_c)
    print(seed_a, seed_b, seed_c)
    for num_id, seq in enumerate(seq_list):

        if num_id < 0:
            continue
        seq_path = seq_home + '/' + seq
        img_list_v, img_list_t, gt, ir = genConfig(seq_path, opts['set_type'])
        if os.path.exists(os.path.join(opts['save_proj'] + '/' + model_result + '/Ours' + '_' + seq + '.txt')):
            # os.remove(os.path.join(opts['save_proj'] + model_result + '/Ours' + '_' + seq + '.txt'))
            continue
        iou_result, result_bb, fps, result_nobb, dis_result, result_mean, result_wbf, result_best = run_mdnet(
            img_list_v, img_list_t,
            gt[0],
            gt,
            display=opts[
                'visualize'],
            topk=topk,
            scale_large=args.scale_large)
        # iou_result, result_bb, fps, result_nobb, dis_result = run_mdnet(img_list_v, img_list_t, gt[0], ir[0], gt,
        #                                                                 display=opts['visualize'], topk=topk)

        enable_frameNum = 0.
        for iidx in range(len(iou_result)):
            if (math.isnan(iou_result[iidx]) == False):
                enable_frameNum += 1.
            else:
                ## gt is not alowed
                iou_result[iidx] = 0.

        iou_list += iou_result.flatten().tolist()
        dis_result = dis_result.flatten().tolist()
        if opts['set_type'] == "RGBT234":
            dis_result = [1 if dis <= 20 else 0 for dis in dis_result]
        else:
            dis_result = [1 if dis <= 5 else 0 for dis in dis_result]
        dis_list += dis_result
        len_iou_list += enable_frameNum

        bb_result[seq] = result_bb
        fps_list[seq] = fps

        bb_result_nobb[seq] = result_nobb
        print('{} {} ({}) : {} , total mIoU:{}, total mDis:{}, total fps:{}'.format(num_id, seq,len(iou_result), iou_result.mean(),
                                                                               sum(iou_list) / len_iou_list,
                                                                               sum(dis_list) / len_iou_list,
                                                                               sum(fps_list.values()) / len(
                                                                                   fps_list)))
        with open(os.path.join(opts['save_proj'], 'result.csv'), 'a', newline='') as csv_file :
            writer = csv.writer(csv_file)
            writer.writerow([str(num_id), str(seq), str(iou_result.mean()),
                             str(sum(iou_list) / len_iou_list),
                             str(sum(dis_list) / len_iou_list),
                             str(sum(fps_list.values()) / len(
                                 fps_list))])

        res = {}
        res['nobb'] = result_nobb.round().tolist()
        res['ori'] = result_bb.round().tolist()
        res['mean'] = result_mean.round().tolist()
        res['wbf'] = result_wbf.round().tolist()
        res['best'] = result_best.round().tolist()
        res['type'] = 'rect'
        res['fps'] = fps

        res_txt = ['nobb', 'ori', 'mean', 'wbf', 'best']
        for txt in res_txt:
            loc_8 = []
            for loc in res[txt]:
                loc_8.append(loc[0])
                loc_8.append(loc[1])
                loc_8.append(loc[0] + loc[2])
                loc_8.append(loc[1])
                loc_8.append(loc[0] + loc[2])
                loc_8.append(loc[1] + loc[3])
                loc_8.append(loc[0])
                loc_8.append(loc[1] + loc[3])
                with open(os.path.join(opts['save_proj'] + "/" + model_result + '/Ours_' + txt + '_' + seq + '.txt'),
                          'a') as f:
                    count = 0
                    for k in loc_8:
                        count += 1
                        f.write(str(k))
                        if count < 8:
                            f.write(' ')
                    f.write('\n')
                loc_8 = []

    if opts['set_type'] == "RGBT234":
        get_txt(num)
        val_RGBT234([str(num) + ext for ext in ['_nobb', '_ori', '_mean', '_wbf', '_best']], opts['save_proj'])
    else:
        get_exp(num)
        val_GTOT([str(num) + ext for ext in ['_nobb', '_ori', '_mean', '_wbf', '_best']], opts['save_proj'])
