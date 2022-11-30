from __future__ import print_function

from miscc.utils import mkdir_p, truncated_noise
from miscc.config import cfg, cfg_from_file

from datasets import TextDatasetBirds, TextDatasetCOCO, prepare_data
from DAMSM import RNN_ENCODER, CNN_ENCODER
import clip

import os
import sys
import time
import random
import pprint
import datetime
import dateutil.tz
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from model_sproj import NetG, NetD  # , D_NET256
import torchvision.utils as vutils

import re
import uuid
from collections import OrderedDict

dir_path = (os.path.abspath(os.path.join(os.path.realpath(__file__), '..')))
sys.path.append(dir_path)
import multiprocessing

UPDATE_INTERVAL = 200


def parse_args():
    parser = argparse.ArgumentParser(description='Train a DAMSM network')
    parser.add_argument('--cfg', dest='cfg_file',
                        help='optional config file',
                        default='cfg/bird.yml', type=str)
    parser.add_argument('--gpu', dest='gpu_id', type=int, default=0)
    parser.add_argument('--data_dir', dest='data_dir', type=str, default='')
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    args = parser.parse_args()
    return args


def clip_text_embedding(text, clip_model, device):
    text = clip.tokenize(text).to(device)
    text_features = clip_model.encode_text(text)
    return text, text_features.float()


def clip_image_embedding(image, clip_model, device):
    image = image.to(device)
    image_features = clip_model.encode_image(image)
    return image_features.float()


def load_clip(device):
    clip_model, preprocess = clip.load('ViT-B/32', device)
    clip_pool = torch.nn.AdaptiveAvgPool2d((224, 224))
    clip_trans = transforms.Compose(
        [transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    for param in clip_model.parameters():
        param.requires_grad = False
    return clip_model, clip_pool, clip_trans


def sampling_test(dataloader, dataset, clip_model, clip_pool, clip_trans, device, n=3000):
    import tensorflow as tf
    from ssd_tf import ssd, cfid

    batch_size = cfg.TRAIN.BATCH_SIZE
    cnt = 0
    R_count = 0
    n = n
    real_sim_scores = np.zeros(n)
    fake_sim_scores = np.zeros(n)
    cont = True
    ixtoword = dataset.ixtoword

    all_real = np.zeros((n, 512))
    all_caps = np.zeros((n, 512))
    all_f_caps = np.zeros((n, 512))

    for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        if (cont == False):
            break
        for step, data in enumerate(dataloader, 0):
            if (cont == False):
                break
            imgs, class_ids, sorted_cap_indices, keys, \
            captions, cap_lens, ns, attrs, vs, \
            mis_cap, cap_len, mis_ns, mis_attrs, mis_vs, = prepare_data(data)

            cnt += batch_size

            # clip embedding for global
            words = []
            sent = []
            # mis_sent = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                words.append(tmp_w)
                sent.append(tmp_s)
                # here for cutting words
                # mis_cap = tmp_w[1:]
                # mis_sent.append(mis_cap)
            sent, sent_emb = clip_text_embedding(text=sent, clip_model=clip_model, device=device)

            # here for replacing words
            mis_words = []
            mis_sent = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in mis_cap[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                mis_words.append(tmp_w)
                mis_sent.append(tmp_s)
            mis_sent, mis_sent_emb = clip_text_embedding(text=mis_sent, clip_model=clip_model, device=device)

            imgs = imgs[0].to(device)
            real_for_clip = clip_trans(clip_pool(imgs))
            real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)

            for j in range(batch_size):
                real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                sent_emb[j] /= sent_emb[j].norm(dim=-1, keepdim=True)
                mis_sent_emb[j] /= mis_sent_emb[j].norm(dim=-1, keepdim=True)

                all_real[R_count] = real_for_clip[j].cpu().numpy()
                all_caps[R_count] = sent_emb[j].cpu().numpy()
                all_f_caps[R_count] = mis_sent_emb[j].cpu().numpy()

                fake_sim_score = torch.cosine_similarity(real_for_clip[j], mis_sent_emb[j], dim=0)
                fake_sim_scores[R_count] = fake_sim_score.item()
                # print(fake_sim_score.item())

                real_sim_score = torch.cosine_similarity(real_for_clip[j], sent_emb[j], dim=0)
                real_sim_scores[R_count] = real_sim_score.item()
                # print(real_sim_score.item())

                R_count += 1
                # print('R_count: ', R_count)

                if R_count >= n:
                    print('data loaded')
                    all_real = tf.convert_to_tensor(all_real)
                    all_caps = tf.convert_to_tensor(all_caps)
                    all_f_caps = tf.convert_to_tensor(all_f_caps)

                    ssd_scores = ssd(all_caps, all_f_caps, all_real, )
                    print(f'SSD:{ssd_scores[0]}, SS:{ssd_scores[1]}, dSV:{ssd_scores[2]}, TrSV:{ssd_scores[3]}')

                    cfif = cfid(all_caps, all_f_caps, all_real, )
                    print(f'cfif:{cfif}')
                    cont = False
                    break

    print('data loaded')
    all_real = tf.convert_to_tensor(all_real)
    all_caps = tf.convert_to_tensor(all_caps)
    all_f_caps = tf.convert_to_tensor(all_f_caps)

    ssd_scores = ssd(all_caps, all_f_caps, all_real, )
    print(f'SSD:{ssd_scores[0]}, SS:{ssd_scores[1]}, dSV:{ssd_scores[2]}, TrSV:{ssd_scores[3]}')

    cfid = cfid(all_caps, all_f_caps, all_real, )
    print(f'cfid:{cfid}')


if __name__ == "__main__":
    # unique identifier
    uid = uuid.uuid4().hex

    args = parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)

    if args.gpu_id == -1:
        cfg.CUDA = False
    else:
        cfg.GPU_ID = args.gpu_id

    if args.data_dir != '':
        cfg.DATA_DIR = args.data_dir
    print('Using config:')
    pprint.pprint(cfg)

    if not cfg.TRAIN.FLAG:
        args.manualSeed = 100
    elif args.manualSeed is None:
        args.manualSeed = 100
        # args.manualSeed = random.randint(1, 10000)
    print("seed now is : ", args.manualSeed)
    random.seed(args.manualSeed)
    np.random.seed(args.manualSeed)
    torch.manual_seed(args.manualSeed)
    if cfg.CUDA:
        torch.cuda.manual_seed_all(args.manualSeed)

    torch.cuda.set_device(cfg.GPU_ID)
    cudnn.benchmark = True

    now = datetime.datetime.now(dateutil.tz.tzlocal())
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')
    output_dir = '../output/%s_%s_%s' % \
                 (cfg.DATASET_NAME, cfg.CONFIG_NAME, timestamp)

    # Get data loader ##########################################################
    imsize = cfg.TREE.BASE_SIZE
    batch_size = cfg.TRAIN.BATCH_SIZE
    image_transform = transforms.Compose([
        transforms.Resize(int(imsize * 76 / 64)),
        transforms.RandomCrop(imsize),
        transforms.RandomHorizontalFlip()])
    if cfg.B_VALIDATION:
        if cfg.DATASET_NAME == 'birds':
            print('Loading birds')
            use_tz = False
            print(f'tz {use_tz}')
            dataset = TextDatasetBirds(cfg.DATA_DIR, 'test',
                                       base_size=cfg.TREE.BASE_SIZE,
                                       transform=image_transform)
            print(dataset.n_words, dataset.embeddings_num)
        else:
            print('Loading COCO')
            use_tz = True
            print(f'tz {use_tz}')
            dataset = TextDatasetCOCO(cfg.DATA_DIR, 'test',
                                      base_size=cfg.TREE.BASE_SIZE,
                                      transform=image_transform)
            print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))
    else:
        if cfg.DATASET_NAME == 'birds':
            print('Loading birds')
            use_tz = False
            print(f'tz {use_tz}')
            dataset = TextDatasetBirds(cfg.DATA_DIR, 'train',
                                       base_size=cfg.TREE.BASE_SIZE,
                                       transform=image_transform)
            print(dataset.n_words, dataset.embeddings_num)
        else:
            print('Loading COCO')
            use_tz = True
            print(f'tz {use_tz}')
            dataset = TextDatasetCOCO(cfg.DATA_DIR, 'train',
                                      base_size=cfg.TREE.BASE_SIZE,
                                      transform=image_transform)
            print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    clip_model, clip_pool, clip_trans, = load_clip(device)
    if cfg.B_VALIDATION:
        nn = [100, 1000, 5000, 10000, 20000, 30000]
        for n in nn:
            print('-' * 20)
            print(n)
            count = sampling_test(dataloader, dataset, clip_model, clip_pool, clip_trans, device, n)
