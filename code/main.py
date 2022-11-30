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


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if 'module.' in k:
            k = k.replace('module.', '')
            new_state_dict[k] = v
        else:
            new_state_dict = state_dict
            break
    return new_state_dict


def R_sampling(text_encoder, image_encoder, netG, dataloader, dataset, device, use_tz=False):
    model_dir = cfg.TRAIN.NET_G
    split_dir = 'valid'
    # Build and load the generator
    # netG.load_state_dict(torch.load('models/%s/netG.pth' % (cfg.CONFIG_NAME)))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    s_tmp = model_dir[:-4]
    save_dir = '%s/%s' % (s_tmp, split_dir)
    print(f'saving imgs to {save_dir}')
    mkdir_p(save_dir)
    cnt = 0

    R_count = 0
    n = 10000
    R = np.zeros(n)
    cont = True
    ixtoword = dataset.ixtoword

    for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        if (cont == False):
            break
        for step, data in enumerate(dataloader, 0):
            imgs, class_ids, sorted_cap_indices, keys, \
            captions, cap_lens, ns, attrs, vs, \
            mis_cap, cap_len, mis_ns, mis_attrs, mis_vs, = prepare_data(data)

            cnt += batch_size
            #######################################################
            # (2) Generate fake images
            ######################################################
            words = []
            sent_txt = []
            aspects_list = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                words.append(tmp_w)
                sent_txt.append(tmp_s)

                tmp_a_list = []
                for a in aspects[b]:
                    tmp_a = [ixtoword[k] for k in a.to('cpu').numpy().tolist() if k != 0]
                    tmp_a = ' '.join(tmp_a)
                    tmp_a_list.append(tmp_a)
                aspects_list.append(tmp_a_list)

            aspects = []
            aspects_emb = []
            for a in aspects_list:
                a, a_emb = clip_text_embedding(text=a, clip_model=clip_model, device=device)
                aspects.append(a)
                aspects_emb.append(a_emb)

            aspects = torch.stack(aspects).to(device)
            aspects_emb = torch.stack(aspects_emb).to(device)

            sent, sent_emb = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

            with torch.no_grad():
                if use_tz:
                    noise = truncated_noise(batch_size, 100, 0.88)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                fake_imgs = netG(noise, sent_emb, aspects_emb)

            hidden = text_encoder.init_hidden(batch_size)
            # words_embs: batch_size x nef x seq_len
            # sent_emb: batch_size x nef
            words_embs, sent_emb = text_encoder(captions, cap_lens, hidden)
            words_embs, sent_emb = words_embs.detach(), sent_emb.detach()
            print(fake_imgs.shape)
            _, cnn_code = image_encoder(fake_imgs)
            print(cnn_code.shape)

            for j in range(batch_size):
                mis_captions, mis_captions_len = dataset.get_mis_caption(class_ids[j])
                hidden = text_encoder.init_hidden(99)
                _, sent_emb_t = text_encoder(mis_captions, mis_captions_len, hidden)
                rnn_code = torch.cat((sent_emb[i, :].unsqueeze(0), sent_emb_t), 0)
                ### cnn_code = 1 * nef
                ### rnn_code = 100 * nef
                scores = torch.mm(cnn_code[i].unsqueeze(0), rnn_code.transpose(0, 1))  # 1* 100
                cnn_code_norm = torch.norm(cnn_code[i].unsqueeze(0), 2, dim=1, keepdim=True)
                rnn_code_norm = torch.norm(rnn_code, 2, dim=1, keepdim=True)
                norm = torch.mm(cnn_code_norm, rnn_code_norm.transpose(0, 1))
                scores0 = scores / norm.clamp(min=1e-8)
                print(torch.argmax(scores0))
                if torch.argmax(scores0) == 0:
                    R[R_count] = 1
                R_count += 1
                print('R_count: ', R_count)

                if R_count >= n:
                    print(f'R_count >= 30000 {R_count} ')
                    sum = np.zeros(10)
                    R = R[:R_count]
                    print(len(R))
                    np.random.shuffle(R)
                    q = n // 10
                    for i in range(10):
                        sum[i] = np.average(R[i * q:(i + 1) * q - 1])
                    R_mean = np.average(sum)
                    R_std = np.std(sum)
                    print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
                    cont = False

    print(f'R_count >= 30000 {R_count} ')
    sum = np.zeros(10)
    R = R[:R_count]
    print(len(R))
    np.random.shuffle(R)
    q = n // 10
    for i in range(10):
        sum[i] = np.average(R[i * q:(i + 1) * q - 1])
    R_mean = np.average(sum)
    R_std = np.std(sum)
    print("R mean:{:.4f} std:{:.4f}".format(R_mean, R_std))
    cont = False


def sampling_ssd(netG, dataloader, dataset, clip_model, clip_pool, clip_trans, device, use_tz=False, n=30000,
                 sample_dir=''):
    import tensorflow as tf
    from ssd_tf import ssd
    from eval.FID.fid_score import calculate_fid_given_paths
    from eval.IS.inception_score import inception_score
    gpu_ori_setting = os.environ['CUDA_VISIBLE_DEVICES']
    print(gpu_ori_setting)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_ori_setting.split(',')[-1]
    print(os.environ['CUDA_VISIBLE_DEVICES'])

    os.environ["MKL_NUM_THREADS"] = "1"
    gpu = tf.config.list_physical_devices('GPU')
    # tf.config.gpu_options.allow_growth = True
    # tf.config.experimental.set_memory_growth(device=gpu[0], enable=True)
    split_dir = 'valid'
    target_device = 'cpu'
    # Build and load the generator
    # netG.load_state_dict(torch.load('models/%s/netG.pth' % (cfg.CONFIG_NAME)))
    netG.eval()

    batch_size = cfg.TRAIN.BATCH_SIZE
    save_dir = '%s/%s' % (sample_dir, split_dir)
    print(f'saving imgs to {save_dir}')
    mkdir_p(save_dir)

    cnt = 0
    R_count = 0
    n = n

    cont = True
    ixtoword = dataset.ixtoword

    all_real = np.zeros((n, 512))
    all_fake = np.zeros((n, 512))
    all_caps = np.zeros((n, 512))

    for i in range(cfg.TEXT.CAPTIONS_PER_IMAGE + 1):  # (cfg.TEXT.CAPTIONS_PER_IMAGE):
        if (cont == False):
            break
        for step, data in enumerate(dataloader, 0):
            if (cont == False):
                break

            # imgs, captions, cap_lens, class_ids, keys = prepare_data(data)
            imgs, class_ids, sorted_cap_indices, keys, \
            captions, cap_lens, ns, attrs, vs, \
            mis_cap, cap_len, mis_ns, mis_attrs, mis_vs, = prepare_data(data)

            cnt += batch_size

            words = []
            sent_txt = []
            aspects_list = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                words.append(tmp_w)
                sent_txt.append(tmp_s)

                tmp_a_list = []
                for a in ns[b]:
                    tmp_a = [ixtoword[k] for k in a.to('cpu').numpy().tolist() if k != 0]
                    tmp_a = ' '.join(tmp_a)
                    tmp_a_list.append(tmp_a)
                aspects_list.append(tmp_a_list)

            aspects = []
            aspects_emb = []
            for a in aspects_list:
                a, a_emb = clip_text_embedding(text=a, clip_model=clip_model, device=device)
                aspects.append(a)
                aspects_emb.append(a_emb)

            aspects = torch.stack(aspects).to(device)
            aspects_emb = torch.stack(aspects_emb).to(device)

            sent, sent_emb = clip_text_embedding(text=sent_txt, clip_model=clip_model, device=device)

            #######################################################
            # (2) Generate fake images
            ######################################################
            with torch.no_grad():
                if use_tz:
                    noise = truncated_noise(batch_size, 100, 0.88)
                    noise = torch.tensor(noise, dtype=torch.float).to(device)
                else:
                    noise = torch.randn(batch_size, 100)
                    noise = noise.to(device)
                fake_imgs = netG(noise, sent_emb, aspects_emb)

            imgs = imgs[0].to(device)
            real_for_clip = clip_trans(clip_pool(imgs))
            real_for_clip = clip_image_embedding(real_for_clip, clip_model, device)

            fake_for_clip = clip_trans(clip_pool(fake_imgs))
            fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)

            for j in range(batch_size):
                caps_txt = ' '.join([ixtoword[k] for k in captions[j].to('cpu').numpy().tolist() if k != 0])
                s_tmp = '%s/single/%s' % (save_dir, keys[j])
                folder = s_tmp[:s_tmp.rfind('/')]
                if not os.path.isdir(folder):
                    print('Make a new folder: ', folder)
                    mkdir_p(folder)
                im = fake_imgs[j].data.cpu().numpy()
                # [-1, 1] --> [0, 255]
                im = (im + 1.0) * 127.5
                im = im.astype(np.uint8)
                im = np.transpose(im, (1, 2, 0))
                im = Image.fromarray(im)
                fullpath = '%s_%3d_%s.png' % (s_tmp, i, caps_txt)
                im.save(fullpath)

                fake_for_clip[j] /= fake_for_clip[j].norm(dim=-1, keepdim=True)
                real_for_clip[j] /= real_for_clip[j].norm(dim=-1, keepdim=True)
                sent_emb[j] /= sent_emb[j].norm(dim=-1, keepdim=True)
                all_real[R_count] = real_for_clip[j].to(target_device).numpy()
                all_fake[R_count] = fake_for_clip[j].to(target_device).numpy()
                all_caps[R_count] = sent_emb[j].to(target_device).numpy()
                R_count += 1
                print('R_count: ', R_count)

                if R_count >= n:
                    cont = False
                    break
    print('data loaded')
    all_real = tf.convert_to_tensor(all_real)
    all_fake = tf.convert_to_tensor(all_fake)
    all_caps = tf.convert_to_tensor(all_caps)
    ssd_scores = ssd(all_real, all_fake, all_caps, )

    result = f'SSD:{ssd_scores[0]}, SS:{ssd_scores[1]}, dSV:{ssd_scores[2]}, TrSV:{ssd_scores[3]}, '
    if cfg.CONFIG_NAME =='bird':
        paths = [f'{save_dir}/single', '/data1/phd21_zhaorui_tan/data_raw/birds/CUB_200_2011/images']
    else:
        paths = [f'{save_dir}/single', '/data1/phd21_zhaorui_tan/data_raw/coco/train/val/val2014']
    fid_value = calculate_fid_given_paths(paths, batch_size, gpu[-1], 2048)
    result += f'FID: {fid_value}'
    print(result)



def train(dataloader, dataset,
          netG, netD_S, netD_A, optimizerG, optimizerD_S, optimizerD_A,
          state_epoch, batch_size, ixtoword,
          clip_model, clip_pool, clip_trans, text_encoder, image_encoder, device, use_tz=False):
    real_labels, fake_labels, match_labels = dataset.prepare_labels(batch_size)

    for epoch in range(state_epoch + 1, cfg.TRAIN.MAX_EPOCH + 1):
        for step, data in enumerate(dataloader, 0):
            torch.cuda.empty_cache()
            imgs, class_ids, sorted_cap_indices, keys, \
            captions, cap_lens, ns, attrs, vs, \
            mis_caps, mis_cap_lens, mis_ns, mis_attrs, mis_vs = prepare_data(data)

            # clip embedding for global
            words = []
            sent = []
            aspects_list = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in captions[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                words.append(tmp_w)
                sent.append(tmp_s)

                tmp_a_list = []
                for a in ns[b]:
                    tmp_a = [ixtoword[k] for k in a.to('cpu').numpy().tolist() if k != 0]
                    tmp_a = ' '.join(tmp_a)
                    tmp_a_list.append(tmp_a)
                aspects_list.append(tmp_a_list)

            sent, sent_emb = clip_text_embedding(text=sent, clip_model=clip_model, device=device)

            aspects = []
            aspects_emb = []
            for a in aspects_list:
                a, a_emb = clip_text_embedding(text=a, clip_model=clip_model, device=device)
                aspects.append(a)
                aspects_emb.append(a_emb)
            # aspects = torch.stack(aspects).to(device)
            aspects_emb = torch.stack(aspects_emb).to(device)

            # hard neg clip embedding
            mis_words = []
            mis_sent = []
            for b in range(batch_size):
                tmp_w = [ixtoword[k] for k in mis_caps[b].to('cpu').numpy().tolist() if k != 0]
                tmp_s = ' '.join(tmp_w)
                mis_words.append(tmp_w)
                mis_sent.append(tmp_s)
            mis_sent, mis_sent_emb = clip_text_embedding(text=mis_sent, clip_model=clip_model, device=device)

            ##########################################################
            # update D
            ##########################################################

            imgs = imgs[0].to(device)
            # synthesize fake images
            # noise = torch.randn(batch_size, 100)
            # print('using truncated_noise')
            if use_tz:
                noise = truncated_noise(batch_size, 100, 0.88)
                noise = torch.tensor(noise, dtype=torch.float).to(device)
            else:
                noise = torch.randn(batch_size, 100)
                noise = noise.to(device)
            fake = netG(noise, sent_emb, aspects_emb)

            # ————————————————————————————————————————————————————————
            # D: global
            # ————————————————————————————————————————————————————————
            real_features = netD_S(imgs)
            fake_features = netD_S(fake.detach())
            # D global loss
            output = netD_S.module.COND_DNET(real_features, sent_emb)
            errD_real = torch.nn.ReLU()(1.0 - output).mean()
            output = netD_S.module.COND_DNET(real_features, mis_sent_emb)
            errD_real_mismatch = torch.nn.ReLU()(1.0 + output).mean()
            output = netD_S.module.COND_DNET(fake_features, sent_emb)
            errD_fake = torch.nn.ReLU()(1.0 + output).mean()
            errD_sent = errD_real + (errD_fake + errD_real_mismatch) / 2.0
            # D global update
            optimizerD_S.zero_grad()
            errD_sent.backward()
            optimizerD_S.step()

            # ————————————————————————————————————————————————————————
            # D: local
            # ————————————————————————————————————————————————————————
            real_features = netD_A(imgs)
            fake_features = netD_A(fake.detach())
            # D local loss
            errD_aspects = 0
            for na in range(0, cfg.MAX_ATTR_NUM):
                a = aspects_emb[:, na:na + 1, :].squeeze()
                output = netD_A.module.COND_DNET(real_features, a)
                errD_real = torch.nn.ReLU()(1.0 - output).mean()
                output = netD_A.module.COND_DNET(real_features[:(batch_size - 1)], a[1:batch_size])
                errD_real_mismatch = torch.nn.ReLU()(1.0 + output).mean()
                output = netD_A.module.COND_DNET(fake_features, a)
                errD_fake = torch.nn.ReLU()(1.0 + output).mean()
                errD_aspects += errD_real + (errD_fake + errD_real_mismatch) / 2.0
            errD_aspects /= cfg.MAX_ATTR_NUM
            # D global update
            optimizerD_A.zero_grad()
            errD_aspects.backward()
            optimizerD_A.step()

            # ————————————————————————————————————————————————————————
            # D: global MA-GP
            # ————————————————————————————————————————————————————————
            interpolated = (imgs.data).requires_grad_()
            sent_inter = (sent_emb.data).requires_grad_()
            features = netD_S(interpolated)
            out = netD_S.module.COND_DNET(features, sent_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, sent_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD_S.zero_grad()
            d_loss.backward()
            optimizerD_S.step()

            interpolated = (imgs.data).requires_grad_()
            aspect_inter = (torch.mean(aspects_emb, dim=1).squeeze().data).requires_grad_()
            features = netD_A(interpolated)
            out = netD_A.module.COND_DNET(features, aspect_inter)
            grads = torch.autograd.grad(outputs=out,
                                        inputs=(interpolated, aspect_inter),
                                        grad_outputs=torch.ones(out.size()).cuda(),
                                        retain_graph=True,
                                        create_graph=True,
                                        only_inputs=True)
            grad0 = grads[0].view(grads[0].size(0), -1)
            grad1 = grads[1].view(grads[1].size(0), -1)
            grad = torch.cat((grad0, grad1), dim=1)
            grad_l2norm = torch.sqrt(torch.sum(grad ** 2, dim=1))
            d_loss_gp = torch.mean((grad_l2norm) ** 6)
            d_loss = 2.0 * d_loss_gp
            optimizerD_A.zero_grad()
            d_loss.backward()
            optimizerD_A.step()

            ##########################################################
            # update G
            ##########################################################

            # ————————————————————————————————————————————————————————
            # G: task0 image global
            # ————————————————————————————————————————————————————————
            # G:  image global
            features = netD_S(fake)
            output = netD_S.module.COND_DNET(features, sent_emb)
            G_loss_i_g = - output.mean()
            # G:  semantic global
            fake_for_clip = clip_trans(clip_pool(fake))
            fake_for_clip = clip_image_embedding(fake_for_clip, clip_model, device)
            sent_probs = torch.cosine_similarity(fake_for_clip, sent_emb, dim=1)
            G_loss_s_g = (1 - sent_probs).mean()
            # G: contrastive_loss
            mis_probs = torch.cosine_similarity(fake_for_clip, mis_sent_emb, dim=1)
            contrastive_loss = ((mis_probs + 1e-8) / ((mis_probs + sent_probs) + 1e-8)).mean()

            # ————————————————————————————————————————————————————————
            # G: task0 image local
            # ————————————————————————————————————————————————————————
            # G: image local
            features = netD_A(fake)
            G_loss_i_l = 0
            for na in range(0, cfg.MAX_ATTR_NUM):
                a = aspects_emb[:, na:na + 1, :].squeeze()
                output = netD_A.module.COND_DNET(features, a)
                G_loss_i_l += - output.mean()
            G_loss_i_l /= cfg.MAX_ATTR_NUM

            aspects_prob = torch.cosine_similarity(fake_for_clip.unsqueeze(1), aspects_emb, dim=2)
            G_loss_s_l = (1 - aspects_prob).mean()

            # ————————————————————————————————————————————————————————
            # G: Gproj
            # ————————————————————————————————————————————————————————
            # task0
            errG_1 = G_loss_i_g + G_loss_i_l

            # task1
            semantic_loss = G_loss_s_g + G_loss_s_l
            errG_2 = 10 * ( semantic_loss +  contrastive_loss)
            optimizerG.zero_grad()
            if epoch <= cfg.WARM_UP:
                errG_1.backward()
            else:
                errG_1.backward(retain_graph=True)
                netG.module.observe(0, proj=False)
                errG_2.backward()
                # check task1 grads
                netG.module.observe(1)
            optimizerG.step()

            if (step % 5 == 0):
                try:
                    print(
                        f'[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(dataloader)}] '
                        f'Loss_D_sent {round(errD_sent.item(), 2)} '
                        f'Loss_D_aspect {round(errD_aspects.item(), 2)} '
                        f'Loss_G_1 {round(errG_1.item(), 2)} '
                        f'G_loss_i_g {round(G_loss_i_g.item(), 2)} '
                        f'G_loss_i_l {round(G_loss_i_l.item(), 2)} '
                        f'Loss_G_2 {round(errG_2.item(), 2)} '
                        f'G_loss_s_g {round(G_loss_s_g.item(), 2)} '
                        f'G_loss_s_l {round(G_loss_s_l.item(), 2)} '
                        f's_loss {round(semantic_loss.item(), 2)} '
                        f'c_loss {round(contrastive_loss.item(), 2)} ')

                except Exception:
                    print(
                        f'[{epoch}/{cfg.TRAIN.MAX_EPOCH}][{step}/{len(dataloader)}] '
                        f'Loss_D_sent {round(errD_sent.item(), 2)} '
                        f'Loss_D_aspect {round(errD_aspects.item(), 2)} '
                        f'Loss_G_1 {round(errG_1.item(), 2)} '
                        f'G_loss_i_g {round(G_loss_i_g.item(), 2)} '
                        f'G_loss_i_l {round(G_loss_i_l.item(), 2)} '
                    )

        vutils.save_image(fake.data,
                          '%s/fake_samples_epoch_%03d.png' % (f'{output_dir}/imgs', epoch),
                          normalize=True)
        if epoch % 1 == 0:
            torch.save(netG.state_dict(),
                       '%s/models/%s/netG_%03d.pth' % (output_dir, cfg.CONFIG_NAME, epoch))
            torch.save(netG.module.grads,
                       '%s/models/%s/netGgrads_%03d.pth' % (output_dir, cfg.CONFIG_NAME, epoch))
            torch.save(netD_S.state_dict(),
                       '%s/models/%s/netD_S_%03d.pth' % (output_dir, cfg.CONFIG_NAME, epoch))
            torch.save(netD_A.state_dict(),
                       '%s/models/%s/netD_A_%03d.pth' % (output_dir, cfg.CONFIG_NAME, epoch))


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
                                       transform=image_transform, HNSC_r=cfg.HNSC.R)
            print(dataset.n_words, dataset.embeddings_num)
        else:
            print('Loading COCO')
            use_tz = True
            print(f'tz {use_tz}')
            dataset = TextDatasetCOCO(cfg.DATA_DIR, 'train',
                                      base_size=cfg.TREE.BASE_SIZE,
                                      transform=image_transform, HNSC_r=cfg.HNSC.R)
            print(dataset.n_words, dataset.embeddings_num)
        assert dataset
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, drop_last=True,
            shuffle=True, num_workers=int(cfg.WORKERS))

    # validation data ##########################################################

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    netG = NetG(cfg.TRAIN.NF, nz=100, ).to(device)
    netD_S = NetD(cfg.TRAIN.NF, ncond=512).to(device)
    netD_A = NetD(cfg.TRAIN.NF, ncond=512).to(device)

    state_epoch = 0
    if len(cfg.TRAIN.NET_G) >= 0:
        try:
            print(f'using {cfg.TRAIN.NET_G}')
            netG_path = cfg.TRAIN.NET_G
            print(netG_path)
            state_dict = torch.load(netG_path, map_location=lambda storage, loc: storage)
            state_dict = clean_state_dict(state_dict)
            netG.load_state_dict(state_dict)
            print('loaded G')

            netD_S_path = netG_path.replace('netG', 'netD_S')
            print(netD_S_path)
            state_dict = torch.load(netD_S_path, map_location=lambda storage, loc: storage)
            state_dict = clean_state_dict(state_dict)
            netD_S.load_state_dict(state_dict)

            netD_A_path = netG_path.replace('netG', 'netD_A')
            print(netD_A_path)
            state_dict = torch.load(netD_A_path, map_location=lambda storage, loc: storage)
            state_dict = clean_state_dict(state_dict)
            netD_A.load_state_dict(state_dict)

            num = re.findall('\d+', netG_path)[-1]
            state_epoch = int(num)
            print('Loading model epoch', state_epoch)
            netDgrads_path = netG_path.replace('netG', 'netGgrads')
            try:
                netGgrads = torch.load(netDgrads_path)
                netG.grads = netGgrads
                print('Loaded netGgrads')
            except Exception:
                print('++++++++++++++++++++++++')
                print('Unable to load netGgrads')
                print('++++++++++++++++++++++++')
                pass

        except Exception:
            print('++++++++++++++++++++++++')
            print('Unable to load the model')
            print('++++++++++++++++++++++++')
            pass

    print('CUDA count', torch.cuda.device_count())
    netG = nn.DataParallel(netG, device_ids=range(torch.cuda.device_count()))
    netD_S = nn.DataParallel(netD_S, device_ids=range(torch.cuda.device_count()))
    netD_A = nn.DataParallel(netD_A, device_ids=range(torch.cuda.device_count()))
    print('using parallel training')

    clip_model, clip_pool, clip_trans, = load_clip(device)
    text_encoder = RNN_ENCODER(dataset.n_words, nhidden=cfg.TEXT.EMBEDDING_DIM)
    state_dict = torch.load(cfg.TEXT.DAMSM_NAME, map_location=lambda storage, loc: storage)
    text_encoder.load_state_dict(state_dict)
    text_encoder.cuda()
    text_encoder.eval()

    image_encoder = CNN_ENCODER(cfg.TEXT.EMBEDDING_DIM)
    img_encoder_path = cfg.TEXT.DAMSM_NAME.replace('text_encoder', 'image_encoder')
    state_dict = torch.load(img_encoder_path, map_location=lambda storage, loc: storage)
    image_encoder.load_state_dict(state_dict)
    image_encoder.cuda()
    image_encoder.eval()

    optimizerG = torch.optim.Adam(netG.parameters(), lr=0.0001, betas=(0.0, 0.9))
    optimizerD_S = torch.optim.Adam(netD_S.parameters(), lr=0.0004, betas=(0.0, 0.9))
    optimizerD_A = torch.optim.Adam(netD_A.parameters(), lr=0.0004, betas=(0.0, 0.9))

    if cfg.B_VALIDATION:
        # nn = [100, 1000, 5000, 10000, 20000]
        nn = [30000]
        for n in nn:
            print('--------------------------------------------------------------------------')
            print(n)
            count = sampling_ssd(netG, dataloader, dataset, clip_model, clip_pool, clip_trans, device, use_tz,n, cfg.TRAIN.NET_G[:-4])
            print('state_epoch:  %d' % (state_epoch))

    else:
        # create dir for output
        print(f'saving output to {output_dir}')
        img_dir = f'{output_dir}/imgs'
        model_dir = f'{output_dir}/models/{cfg.CONFIG_NAME}'
        mkdir_p(img_dir)
        mkdir_p(model_dir)
        ixtoword = dataset.ixtoword
        count = train(dataloader, dataset, netG, netD_S, netD_A, optimizerG, optimizerD_S, optimizerD_A, state_epoch,
                      batch_size, ixtoword, clip_model, clip_pool,
                      clip_trans, text_encoder, image_encoder, device, use_tz)
