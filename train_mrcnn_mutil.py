
import glob
import pickle
import sys
from math import floor

import torch.optim as optim

import tools

sys.path.insert(0, './modules')
import argparse
from modules.model_train_best6 import *
from modules.roi_align.modules.roi_align import RoIAlignAdaMax
from modules.data_prov import *
from modules.img_cropper import *


def set_seed(a, b, c):
    np.random.seed(a)
    torch.manual_seed(b)
    torch.cuda.manual_seed(c)


global logger


def convert_seconds(seconds):
    seconds = floor(seconds)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def set_optimizer(model, lr_base, lr_mult=pretrain_opts['lr_mult'], momentum=pretrain_opts['momentum'],
                  w_decay=pretrain_opts['w_decay']):
    params = model.get_learnable_params(model)
    param_list = []
    for k, p in params.items():
        lr = lr_base
        for l, m in lr_mult.items():
            if k.startswith(l):
                lr = lr_base * m
        param_list.append({'params': [p], 'lr': lr})
    # optimizer = optim.SGD(param_list, lr=lr, momentum=momentum, weight_decay=w_decay)
    optimizer = optim.Adam(param_list, lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=w_decay)
    return optimizer


def train_mdnet(topk, num):
    ## set image directory
    if pretrain_opts['set_type'] == 'RGBT234':
        img_home = '/media/lx/Disk_data/LRJ_data/DATA/RGBT234/'
        data_path = './data/RGBT234.pkl'
    if pretrain_opts['set_type'] == 'GTOT':
        img_home = '/media/lx/Disk_data/LRJ_data/DATA/Multi_Modal_RGBT_dataset_CSR/'
        data_path = './data/GTOT.pkl'
    ## Init dataset ##
    with open(data_path, 'rb') as fp:
        data = pickle.load(fp)

    K = len(data)

    ## Init model ##
    model = MDNet(pretrain_opts['init_model_path'], K=K, topK=topk)
    if pretrain_opts['adaptive_align']:
        align_h = model.roi_align_model.aligned_height
        align_w = model.roi_align_model.aligned_width
        spatial_s = model.roi_align_model.spatial_scale
        model.roi_align_model = RoIAlignAdaMax(align_h, align_w, spatial_s)
        model.roi_align_model_mid = RoIAlignAdaMax(7, 7, spatial_s)
        model.roi_align_model_high = RoIAlignAdaMax(11, 11, spatial_s)

    if pretrain_opts['use_gpu']:
        model = model.cuda()

    model.set_learnable_params(model, pretrain_opts['ft_layers'], True)
    model.train()

    dataset = [None] * K
    for k, (seqname, seq) in enumerate(data.items()):
        img_list_v = seq['RGB_image']
        gt_v = seq['RGB_gt']
        img_list_i = seq['T_image']
        gt_i = gt_v
        if pretrain_opts['set_type'] == 'RGBT234':
            img_dir = img_home + seqname
        if pretrain_opts['set_type'] == 'GTOT':
            img_dir = img_home + seqname

        dataset[k] = RegionDataset(img_dir, img_list_v, img_list_i, gt_v, gt_i, model.receptive_field, pretrain_opts)

    ## Init criterion and optimizer ##
    binaryCriterion = BinaryLoss()
    # binaryCriterion=BinaryLossTriple()
    interDomainCriterion = nn.CrossEntropyLoss()
    evaluator = Precision()
    optimizer = set_optimizer(model, pretrain_opts['lr'])

    best_score = 0.
    best_cur_interloss = 999
    batch_cur_idx = 0
    time_start = time.time()
    ClsLoss = []
    SCORE = []
    CLSLOSS = []
    INTERLOSS = []
    for i in range(pretrain_opts['n_cycles']):
        print("==== Start Cycle %d ====" % (i))
        # 检查参数是否参与训练

        k_list = np.random.permutation(K)
        prec = np.zeros(K)
        totalTripleLoss = np.zeros(K)
        totalInterClassLoss = np.zeros(K)
        for j, k in enumerate(k_list):
            tic = time.time()

            cropped_scenes1, cropped_scenes2, pos_rois, neg_rois = dataset[k].next()

            for sidx in range(0, len(cropped_scenes1)):
                cur_scene1 = cropped_scenes1[sidx]
                cur_scene2 = cropped_scenes2[sidx]
                cur_pos_rois = pos_rois[sidx]
                cur_neg_rois = neg_rois[sidx]

                cur_scene1 = Variable(cur_scene1)
                cur_scene2 = Variable(cur_scene2)
                cur_pos_rois = Variable(cur_pos_rois)
                cur_neg_rois = Variable(cur_neg_rois)
                if pretrain_opts['use_gpu']:
                    cur_scene1 = cur_scene1.cuda()
                    cur_scene2 = cur_scene2.cuda()
                    cur_pos_rois = cur_pos_rois.cuda()
                    cur_neg_rois = cur_neg_rois.cuda()
                cur_feat_map = model(cur_scene1, cur_scene2, k, out_layer='conv3')

                cur_pos_feats = model.roi_align_model(cur_feat_map, cur_pos_rois)
                cur_pos_feats_mid = model.roi_align_model_mid(cur_feat_map, cur_pos_rois)
                cur_pos_feats_high = model.roi_align_model_high(cur_feat_map, cur_pos_rois)
                # cur_pos_feats = cur_pos_feats.view(cur_pos_feats.size(0), -1)
                cur_neg_feats = model.roi_align_model(cur_feat_map, cur_neg_rois)
                cur_neg_feats_mid = model.roi_align_model_mid(cur_feat_map, cur_neg_rois)
                cur_neg_feats_high = model.roi_align_model_high(cur_feat_map, cur_neg_rois)
                # cur_neg_feats = cur_neg_feats.view(cur_neg_feats.size(0), -1)

                if sidx == 0:
                    pos_feats = [cur_pos_feats]
                    neg_feats = [cur_neg_feats]
                    pos_feats_mid = [cur_pos_feats_mid]
                    neg_feats_mid = [cur_neg_feats_mid]
                    pos_feats_high = [cur_pos_feats_high]
                    neg_feats_high = [cur_neg_feats_high]
                else:
                    pos_feats.append(cur_pos_feats)
                    neg_feats.append(cur_neg_feats)
                    pos_feats_mid.append(cur_pos_feats_mid)
                    neg_feats_mid.append(cur_neg_feats_mid)
                    pos_feats_high.append(cur_pos_feats_high)
                    neg_feats_high.append(cur_neg_feats_high)
            feat_dim = cur_neg_feats.size(1)
            # pos_feats = torch.stack(pos_feats, dim=0).view(-1, feat_dim)
            # neg_feats = torch.stack(neg_feats, dim=0).view(-1, feat_dim)
            pos_feats = torch.stack(pos_feats, dim=0).view(-1, feat_dim, 3, 3)
            neg_feats = torch.stack(neg_feats, dim=0).view(-1, feat_dim, 3, 3)
            pos_feats_mid = torch.stack(pos_feats_mid, dim=0).view(-1, feat_dim, 5, 5)
            neg_feats_mid = torch.stack(neg_feats_mid, dim=0).view(-1, feat_dim, 5, 5)
            pos_feats_high = torch.stack(pos_feats_high, dim=0).view(-1, feat_dim, 7, 7)
            neg_feats_high = torch.stack(neg_feats_high, dim=0).view(-1, feat_dim, 7, 7)

            pos_score = model(pos_feats, pos_feats, k, in_layer='fc4', x1_m=pos_feats_mid, x1_h=pos_feats_high)
            neg_score = model(neg_feats, neg_feats, k, in_layer='fc4', x1_m=neg_feats_mid, x1_h=neg_feats_high)

            ##################################################################
            # fusion_score = model(pos_feats, neg_feats, k, in_layer='fc4')
            ##################################################################

            cls_loss = binaryCriterion(pos_score, neg_score)
            # cls_loss_rgb = binaryCriterion(pos_rgb, neg_rgb)
            # cls_loss_ir = binaryCriterion(pos_ir, neg_ir)
            # cls_loss += cls_loss_rgb + cls_loss_ir
            ## inter frame classification

            interclass_label = Variable(torch.zeros((pos_score.size(0))).long())
            if pretrain_opts['use_gpu']:
                interclass_label = interclass_label.cuda()
            total_interclass_score = pos_score[:, 1].contiguous()
            total_interclass_score = total_interclass_score.view((pos_score.size(0), 1))

            K_perm = np.random.permutation(K)
            K_perm = K_perm[0:100]
            for cidx in K_perm:
                if k == cidx:
                    continue
                else:
                    interclass_score = model(pos_feats, pos_feats, cidx, in_layer='fc4', x1_m=pos_feats_mid, x1_h=pos_feats_high)
                    total_interclass_score = torch.cat((total_interclass_score,
                                                        interclass_score[:, 1].contiguous().view(
                                                            (interclass_score.size(0), 1))), dim=1)

            interclass_loss = interDomainCriterion(total_interclass_score, interclass_label)
            totalInterClassLoss[k] = interclass_loss.data

            (cls_loss + 0.1 * interclass_loss).backward()
            ClsLoss.append(cls_loss.data)
            batch_cur_idx += 1
            if (batch_cur_idx % pretrain_opts['seqbatch_size']) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), pretrain_opts['grad_clip'])
                optimizer.step()
                model.zero_grad()
                batch_cur_idx = 0

            ## evaulator
            prec[k] = evaluator(pos_score, neg_score)
            ## computation latency
            toc = time.time() - tic

            print("Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" % \
                  (i, j, k, cls_loss.data, prec[k], totalInterClassLoss[k], toc))
            logger.info("Cycle %2d, K %2d (%2d), BinLoss %.3f, Prec %.3f, interLoss %.3f, Time %.3f" % \
                        (i, j, k, cls_loss.data, prec[k], totalInterClassLoss[k], toc))

        cur_score = prec.mean()
        cur_interloss = totalInterClassLoss.mean()
        try:
            total_miou = sum(total_iou) / len(total_iou)
        except:
            total_miou = 0.
        print("Mean Precision: %.3f BinLoss: %.3f Inter Loss: %.3f IoU: %.3f  Time: %s" % (
            cur_score, sum(ClsLoss) / len(ClsLoss), cur_interloss, total_miou,
            convert_seconds(time.time() - time_start)))
        SCORE.append(cur_score)
        CLSLOSS.append(sum(ClsLoss).cpu() / len(ClsLoss))
        INTERLOSS.append(cur_interloss)
        tools.plot_pic(SCORE, CLSLOSS, INTERLOSS, num)
        logger.info("Mean Precision: %.3f BinLoss: %.3f Inter Loss: %.3f IoU: %.3f  Time: %s" % (
            cur_score, sum(ClsLoss) / len(ClsLoss), cur_interloss, total_miou,
            convert_seconds(time.time() - time_start)))
        ClsLoss.clear()
        if cur_score > best_score:
            best_score = cur_score
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('branches')}

            print("Save model to %s" % pretrain_opts['model_path_all'])
            logger.info("Save model to %s" % pretrain_opts['model_path_all'])
            torch.save(state_dict, pretrain_opts['model_path_all'])

            if pretrain_opts['use_gpu']:
                model = model.cuda()
        if cur_score > 0.95:
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('branches')}

            new_p = pretrain_opts['model_path_all'].replace('.pth', '_{}_{:.4f}.pth'.format(i, cur_score))
            print("Save model to %s" % new_p)
            logger.info("Save model to %s" % new_p)
            torch.save(state_dict, new_p)

            if pretrain_opts['use_gpu']:
                model = model.cuda()
        if cur_interloss < best_cur_interloss:
            best_cur_interloss = cur_interloss
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('branches')}
            new_p = pretrain_opts['model_path_all'].replace('.pth', '_inter.pth')
            print("Save model to %s" % new_p)
            logger.info("Save model to %s" % new_p)
            torch.save(state_dict, new_p)
            if pretrain_opts['use_gpu']:
                model = model.cuda()
        if 170 >= i >= 140:
            if pretrain_opts['use_gpu']:
                model = model.cpu()
            state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('branches')}
            new_p = pretrain_opts['model_path_all'].replace('.pth', '_{}.pth'.format(i))
            print("Save model to %s" % new_p)
            logger.info("Save model to %s" % new_p)
            torch.save(state_dict, new_p)
            if pretrain_opts['use_gpu']:
                model = model.cuda()

        # if i % 10 == 0:
        #     if pretrain_opts['use_gpu']:
        #         model = model.cpu()
        #
        #     state_dict = {k: v for k, v in model.state_dict().items() if not k.startswith('branches')}
        #     new_p = pretrain_opts['model_path_all'].replace('.pth', '_{}.pth'.format(i))
        #     print("Save model to %s" % new_p)
        #     logger.info("Save model to %s" % new_p)
        #     torch.save(state_dict, new_p)
        #
        #     if pretrain_opts['use_gpu']:
        #         model = model.cuda()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-set_type", default='GTOT')  # RGBT234,GTOT
    parser.add_argument("-padding_ratio", default=5., type=float)
    parser.add_argument("-frame_interval", default=1, type=int,
                        help="frame interval in batch. ex) interval=1 -> [1 2 3 4 5], interval=2 ->[1 3 5]")
    parser.add_argument("-init_model_path", default='./models/init_models/imagenet-vgg-m.mat')
    parser.add_argument("-resume_train", default=False)
    parser.add_argument("-batch_frames", default=8, type=int)
    parser.add_argument("-lr", default=0.0001, type=float)
    parser.add_argument("-batch_pos", default=64, type=int)
    parser.add_argument("-batch_neg", default=196, type=int)
    parser.add_argument("-n_cycles", default=300, type=int)
    parser.add_argument("-adaptive_align", default=True, action='store_false')
    parser.add_argument("-seqbatch_size", default=50, type=int)
    args = parser.parse_args()
    num = len(glob.glob('./runs/train/*/'))
    ##################################################################################
    #########################Just modify pretrain_opts in this script.#########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    ##option setting
    pretrain_opts['set_type'] = args.set_type
    pretrain_opts['padding_ratio'] = args.padding_ratio
    pretrain_opts['padded_img_size'] = pretrain_opts['img_size'] * int(pretrain_opts['padding_ratio'])
    pretrain_opts['frame_interval'] = args.frame_interval
    pretrain_opts['init_model_path'] = args.init_model_path
    pretrain_opts['batch_frames'] = args.batch_frames
    pretrain_opts['lr'] = args.lr
    pretrain_opts['batch_pos'] = args.batch_pos  # original = 64
    pretrain_opts['batch_neg'] = args.batch_neg  # original = 192
    pretrain_opts['n_cycles'] = args.n_cycles
    pretrain_opts['adaptive_align'] = args.adaptive_align
    pretrain_opts['seqbatch_size'] = args.seqbatch_size
    pretrain_opts['resume_train'] = args.resume_train
    pretrain_opts['model_path_all'] = './runs/train/exp' + str(num) + '/weights/DAPNet.pth'

    ##################################################################################
    ############################Do not modify pretrain_opts anymore.###########################
    ######################Becuase of synchronization of options#######################
    ##################################################################################
    seed_a = 123
    seed_b = 456
    seed_c = 123
    info = "seed_init: " + str(seed_a) + ',' + str(seed_b) + ',' + str(seed_c)
    set_seed(seed_a, seed_b, seed_c)
    pretrain_opts['info'] = info
    tools.format_dictionary(pretrain_opts)
    logger = tools.train_log(info)
    if not os.path.exists('./runs/train/exp' + str(num) + '/weights/'):
        os.mkdir('./runs/train/exp' + str(num) + '/weights/')
    channel = [96, 256, 512]
    rate = [1, 1, 1]
    topk = [int((1 - rate[i]) * channel[i]) for i in range(3)]
    print(str(rate))
    logger.info(str(rate))
    train_mdnet(topk, num)
    # 0.5 0.75 0.25
    # 0.75 0.5 0.25
    # 0.25 1 0.5
    # 0.8 0.6 0.4
    # 0.7 0.6 0.5
    # 0.5,0.5,0.5
    # 0.9, 0.2, 0.5
    # 0.9, 0.3, 0.9
    # 0.01, 0.01, 0.01
    # 0.3, 0.6, 0.9
