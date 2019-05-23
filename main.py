from __future__ import print_function
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import math
import util 
import numpy as np
import classifier
import classifier2
import sys
import traceback
import model
from tensorboardX import SummaryWriter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='FLO', help='FLO')
    parser.add_argument('--dataroot', default='/home/zwl/data/zsl-data/data', help='path to dataset')
    parser.add_argument('--matdataset', default=True, help='Data in matlab format')
    parser.add_argument('--image_embedding', default='res101')
    parser.add_argument('--class_embedding', default='att')
    parser.add_argument('--syn_num', type=int, default=100, help='number features to generate per class')
    parser.add_argument('--gzsl', action='store_true', default=False, help='enable generalized zero-shot learning')
    parser.add_argument('--preprocessing', action='store_true', default=False, help='enbale MinMaxScaler on visual features')
    parser.add_argument('--standardization', action='store_true', default=False)
    parser.add_argument('--val_mi', action='store_true', default=False, help='only train for one epoch')
    parser.add_argument('--validation', action='store_true', default=False, help='validation mode')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size')
    parser.add_argument('--resSize', type=int, default=2048, help='size of visual features')
    parser.add_argument('--attSize', type=int, default=1024, help='size of semantic features')
    parser.add_argument('--nz', type=int, default=312, help='size of the latent z vector')
    parser.add_argument('--ngh', type=int, default=4096, help='size of the hidden units in generator')
    parser.add_argument('--ndh', type=int, default=1024, help='size of the hidden units in discriminator')
    parser.add_argument('--nepoch', type=int, default=2000, help='number of epochs to train for')
    parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
    parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
    parser.add_argument('--cls_weight', type=float, default=1, help='weight of the classification loss')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train GANs ')
    parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--cuda', action='store_true', default=False, help='enables cuda')
    parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
    parser.add_argument('--pretrain_classifier', default='', help="path to pretrain classifier (to continue training)")
    parser.add_argument('--netG', default='', help="path to netG (to continue training)")
    parser.add_argument('--netD', default='', help="path to netD (to continue training)")
    parser.add_argument('--netM', default='', help="path to netM (to continue training)")
    parser.add_argument('--netG_name', default='')
    parser.add_argument('--netD_name', default='')
    parser.add_argument('--outf', default='./checkpoint/', help='folder to output data and model checkpoints')
    parser.add_argument('--outname', help='folder to output data and model checkpoints')
    parser.add_argument('--save_every', type=int, default=100)
    parser.add_argument('--print_every', type=int, default=1)
    parser.add_argument('--val_every', type=int, default=10)
    parser.add_argument('--start_epoch', type=int, default=0)
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--nclass_all', type=int, default=200, help='number of all classes')
    parser.add_argument('--mine', action='store_true', default=False, help='whether use mine or not')
    parser.add_argument('--mi_w', type=float, default=0.1, help='weight of mi')
    parser.add_argument('--comment', type=str, default=None, help='add your commment')
    parser.add_argument('--tanh', action='store_true', default=False, help='whether to use tanh')
    parser.add_argument('--clipping', action='store_true', default=False, help='whether to use adaptive clipping')
    parser.add_argument('--visual_mi', action='store_true', default=False, help='whether to use visual_mi')
    parser.add_argument('--only_mi', action='store_true', default=False, help='no discriminator')
    parser.add_argument('--method', type=str, default='KL', help='MINE method: : KL, JSD, NCE')
    parser.add_argument('--NCE', type=int, default=5, help='number of negative samples in NCE')
    parser.add_argument('--c2epoch', type=int, default=25, help='number of classifier2 epochs')
    parser.add_argument('--lrelu', type=float, default=0.01, help='negative slope of LeakyReLU')
    return parser


def sampleNCE(num):
    for i in range(num):
        resample()
        input_atts[i].copy_(input_att_)

def sample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def resample():
    batch_feature, batch_label, batch_att = data.next_batch(opt.batch_size)
    input_res_.copy_(batch_feature)
    input_att_.copy_(batch_att)

def generate_syn_feature(netG, classes, attribute, num, opt):
    with torch.no_grad():
        nclass = classes.size(0)
        syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
        syn_label = torch.LongTensor(nclass*num) 
        syn_att = torch.FloatTensor(num, opt.attSize)
        syn_noise = torch.FloatTensor(num, opt.nz)
        if opt.cuda:
            syn_att = syn_att.cuda()
            syn_noise = syn_noise.cuda()
            
        for i in range(nclass):
            iclass = classes[i]
            iclass_att = attribute[iclass]
            syn_att.copy_(iclass_att.repeat(num, 1))
            syn_noise.normal_(0, 1)
            output = netG(syn_noise, syn_att)
            syn_feature.narrow(0, i*num, num).copy_(output.detach().cpu())
            syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

def calc_gradient_penalty(netD, real_data, fake_data, input_att):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()

    interpolates = alpha * real_data + ((1 - alpha) * fake_data)

    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)

    disc_interpolates = netD(interpolates, input_att)

    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def adaptive_clipping(g_u, g_m, netG):
    n_u = np.linalg.norm([torch.norm(i).cpu().item() for i in g_u])
    n_m = np.linalg.norm([torch.norm(i).cpu().item() for i in g_m])
    k = np.min([n_u, n_m]) / n_m 
    for i, p in enumerate(netG.parameters()):
        p.grad = g_u[i] + opt.mi_w * k * g_m[i]

def train_D(netG, netD, optimizerD): 
    ############################
    # (1) Update D network: optimize WGAN-GP objective, Equation (2)
    ###########################
    for p in netD.parameters(): # reset requires_grad
        p.requires_grad = True # they are set to False below in netG update

    for iter_d in range(opt.critic_iter):
        sample()
        netD.zero_grad()
        # train with realG
        # sample a mini-batch
        sparse_real = opt.resSize - input_res[1].gt(0).sum()

        criticD_real = netD(input_res, input_att)
        criticD_real = criticD_real.mean()
        criticD_real.backward(mone)

        # train with fakeG
        noise.normal_(0, 1)
        fake = netG(noise, input_att)
        fake_norm = fake.data[0].norm()
        sparse_fake = fake.data[0].eq(0).sum()
        criticD_fake = netD(fake.detach(), input_att)
        criticD_fake = criticD_fake.mean()
        criticD_fake.backward(one)

        # gradient penalty
        gradient_penalty = calc_gradient_penalty(netD, input_res, fake.detach(), input_att)
        gradient_penalty.backward()

        Wasserstein_D = criticD_real - criticD_fake
        D_cost = criticD_fake - criticD_real + gradient_penalty
        optimizerD.step()
    return D_cost, Wasserstein_D

def sp(x):
    return torch.log(1 + torch.exp(x))

def calc_mi(netM, x, z, z_, term='all'):
    mi = {}
    if opt.method == 'KL':
        mi = {
                1: torch.mean(netM(x, z)), 
                2: torch.mean(torch.exp(netM(x, z_)))
                }
        mi['all'] = mi[1] - torch.log(mi[2])
    elif opt.method == 'JSD':
        mi = {
                1: torch.mean(-sp(-netM(x, z))), 
                2: torch.exp(netM(x, z_)) + 1
                }
        mi['all'] = mi[1] - torch.mean(torch.log(mi[2]))
    elif opt.method == 'NCE':
        s = torch.zeros(opt.batch_size, 1).cuda()
        if term != 1:
            sampleNCE(opt.NCE)
            for i in range(opt.NCE):
                s += torch.exp(netM(x, input_atts[i]))
            s /= opt.NCE
        mi = {
                1: torch.mean(netM(x, z)),
                2: s
                }
        mi['all'] = mi[1] - torch.mean(torch.log(mi[2]))
    return mi[term]

def train_G(netG, netD, netM, optimizerG, mi_obj=False):
    ############################
    # (2) Update G network: optimize WGAN-GP objective, Equation (2)
    ###########################
    for p in netD.parameters(): # reset requires_grad
        p.requires_grad = False # avoid computation

    netG.zero_grad()
    noise.normal_(0, 1)
    fake = netG(noise, input_att)
    criticG_fake = netD(fake, input_att)
    criticG_fake = criticG_fake.mean()
    G_cost = -criticG_fake
    # classification loss
    c_errG = cls_criterion(pretrain_cls.model(fake), input_label)
    errG = G_cost + opt.cls_weight * c_errG if not opt.only_mi else 0

    if mi_obj:
        if not opt.clipping:
            resample()
            # mi = torch.mean(netM(fake, input_att)) - torch.log(torch.mean(torch.exp(netM(fake, input_att_))))
            if opt.visual_mi:
                mi = calc_mi(netM, fake, input_res, input_res_)
            else:
                mi = calc_mi(netM, fake, input_att, input_att_)
            errG -= opt.mi_w * mi
            errG.backward()
        else:
            resample()
            # with clipping
            errG.backward(retain_graph=True)
            for (i, p) in enumerate(netG.parameters()):
                if i == len(g_u):
                    g_u.append(torch.zeros_like(p.grad))
                g_u[i].copy_(p.grad)

            # calculate mi
            fake = netG(noise, input_att)
            if opt.visual_mi:
                mi = calc_mi(netM, fake, input_res, input_res_)
            else:
                mi = calc_mi(netM, fake, input_att, input_att_)
            # mi = torch.mean(netM(fake, input_att)) - torch.log(torch.mean(torch.exp(netM(fake, input_att_))))

            optimizerG.zero_grad()
            errM = -mi
            errM.backward()
            for (i, p) in enumerate(netG.parameters()):
                if i == len(g_m):
                    g_m.append(torch.zeros_like(p.grad))
                g_m[i].copy_(p.grad)

            adaptive_clipping(g_u, g_m, netG)
            errG = errG + opt.mi_w * errM
    else:
        errG.backward()

    optimizerG.step()
    return G_cost, errG

def train_M(netG, netM, optimizerM, ma_rate=0.001):
    noise.normal_(0, 1)
    x_tilde = netG(noise, input_att)
    resample()
    z_ = input_res_ if opt.visual_mi else input_att_
    z = input_res if opt.visual_mi else input_att
    et = calc_mi(netM, x_tilde, z, z_, term=2)
    if netM.ma_et is None:
        netM.ma_et = torch.mean(et).detach().item()

    netM.ma_et += ma_rate * (torch.mean(et).detach().item() - netM.ma_et)
    if abs(netM.ma_et) < 1e-7:
        print('netM ma et too small')
        mi = calc_mi(netM, x_tilde, z, z_, term=1) - torch.mean(torch.log(et))
    else:
        mi = calc_mi(netM, x_tilde, z, z_, term=1) - torch.mean(torch.log(et)) * torch.mean(et).detach() / netM.ma_et
    loss = -mi
    optimizerM.zero_grad()
    loss.backward()
    optimizerM.step()
    return mi.item()

def update_best(acc, epoch, best_acc, best_epoch):
    if isinstance(acc, tuple): # gzsl
        acc_or_h = acc[2]
        best_acc_or_h = best_acc[2]
    else:
        acc_or_h = acc
        best_acc_or_h = best_acc

    if acc_or_h > best_acc_or_h:
        best_acc = acc
        best_epoch = epoch

    return best_acc, best_epoch

if __name__ == '__main__':
    parser = get_args()
    opt = parser.parse_args()
    print(opt)
    opt.save_path = os.path.join(opt.outf, opt.outname)
    print(opt.save_path)
    if not opt.comment:
        comment = input('please input comment: ')
    else:
        comment = opt.comment

    try:
        os.makedirs(opt.save_path)
    except OSError:
        pass


    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    print("Random Seed: ", opt.manualSeed)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    if opt.cuda:
        torch.cuda.manual_seed_all(opt.manualSeed)

    cudnn.benchmark = True

    if torch.cuda.is_available() and not opt.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    # load data
    data = util.DATA_LOADER(opt)
    print("# of training samples: ", data.ntrain)

    # initialize generator and discriminator
    netG = model.MLP_G(opt)
    if opt.netG != '':
        netG.load_state_dict(torch.load(os.path.join(opt.outf, opt.outname, opt.netG)))
    print(netG)

    netD = model.MLP_CRITIC(opt)
    if opt.netD != '':
        netD.load_state_dict(torch.load(os.path.join(opt.outf, opt.outname, opt.netD)))
    print(netD)

    opt.m_nh = opt.ngh # hidden size
    opt.m_nz = opt.nz # noise size
    opt.m_no = 1 # output size
    opt.m_ns = opt.resSize # sample size
    netM = model.MINE(opt)

    if opt.netM != '':
        netM.load_state_dict(torch.load(os.path.join(opt.outf, opt.outname, opt.netM)))
    print(netM)


    # classification loss, Equation (4) of the paper
    cls_criterion = nn.NLLLoss()

    input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att = torch.FloatTensor(opt.batch_size, opt.attSize)
    # for resample
    input_res_ = torch.FloatTensor(opt.batch_size, opt.resSize)
    input_att_ = torch.FloatTensor(opt.batch_size, opt.attSize)

    noise = torch.FloatTensor(opt.batch_size, opt.nz)
    one = torch.FloatTensor([1])
    mone = one * -1
    input_label = torch.LongTensor(opt.batch_size)

    # gradient buffer for adaptive clipping
    g_u = []
    g_m = []
    if opt.method == 'NCE':
        input_atts = []
        for i in range(opt.NCE):
            input_atts.append(torch.FloatTensor(opt.batch_size, opt.attSize).cuda())


    if opt.cuda:
        netD.cuda()
        netG.cuda()
        netM.cuda()
        input_res = input_res.cuda()
        input_res_ = input_res_.cuda()
        input_att_ = input_att_.cuda()
        noise, input_att = noise.cuda(), input_att.cuda()
        one = one.cuda()
        mone = mone.cuda()
        cls_criterion.cuda()
        input_label = input_label.cuda()


    # setup optimizer
    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizerM = optim.Adam(netM.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))




    # train a classifier on seen classes, obtain \theta of Equation (4)
    pretrain_cls = classifier.CLASSIFIER(data.train_feature, util.map_label(data.train_label, data.seenclasses), data.seenclasses.size(0), opt.resSize, opt.cuda, 0.001, 0.5, 50, 100, opt.pretrain_classifier)

    # freeze the classifier during the optimization
    for p in pretrain_cls.model.parameters(): # set requires_grad to False
        p.requires_grad = False


    if opt.gzsl:
        best_acc = (0, 0, 0)
    else:
        best_acc = 0
    best_epoch = 0
    suffix = '_g' if opt.gzsl else '' 
    suffix += f'_{opt.method}' if opt.mine else ''
    save_name = f'netG_epoch{suffix}_{comment}.pth'
    print(save_name)
    writer = SummaryWriter(f'log/{opt.outname}_{opt.mine}{suffix}')
    try:
        for epoch in range(opt.nepoch if not opt.val_mi else 1):
            FP = 0 
            Wasserstein_list = []
            D_cost_list = []
            mi_list = []
            for i in range(0, data.ntrain, opt.batch_size):
                D_cost, Wasserstein_D = train_D(netG, netD, optimizerD)

                Wasserstein_list.append(Wasserstein_D.item())
                D_cost_list.append(D_cost.item())
                if opt.mine:
                    G_cost, c_errG = train_G(netG, netD, netM, optimizerG, opt.mine)
                else:
                    G_cost, c_errG = train_G(netG, netD, None, optimizerG, opt.mine)

                if opt.mine or opt.val_mi:
                    mi = train_M(netG, netM, optimizerM)
                    mi_list.append(mi)
            writer.add_scalar('mi', np.mean(mi_list), epoch)

            print('[%d/%d] Loss_D: %.4f Loss_G: %.4f, Wasserstein_dist: %.4f, c_errG:%.4f'
                      % (epoch, opt.nepoch, np.mean(D_cost_list), G_cost.item(), np.mean(Wasserstein_list), c_errG.item()))
            if opt.mine or opt.val_mi:
                print(f'mi = {np.mean(mi_list):.4f}')

            # evaluate the model, set G to evaluation mode
            netG.eval()
            # Generalized zero-shot learning
            if opt.gzsl:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, opt)
                train_X = torch.cat((data.train_feature, syn_feature), 0)
                train_Y = torch.cat((data.train_label, syn_label), 0)
                nclass = opt.nclass_all
                cls = classifier2.CLASSIFIER(train_X, train_Y, data, nclass, opt.cuda, opt.classifier_lr, 0.5, opt.c2epoch, opt.syn_num, True)
                best_acc, best_epoch = update_best((cls.acc_unseen, cls.acc_seen, cls.H), epoch, best_acc, best_epoch)
                print('unseen=%.4f, seen=%.4f, h=%.4f' % (cls.acc_unseen, cls.acc_seen, cls.H))
                print(f'best_h = {best_acc[2]:.4f}, best_epoch={best_epoch}')
                writer.add_scalar('acc_unseen', cls.acc_unseen, epoch)
                writer.add_scalar('acc_seen', cls.acc_seen, epoch)
                writer.add_scalar('H', cls.H, epoch)
                writer.add_scalar('best_H', best_acc[2], epoch)
            # Zero-shot learning
            else:
                syn_feature, syn_label = generate_syn_feature(netG, data.unseenclasses, data.attribute, opt.syn_num, opt) 
                cls = classifier2.CLASSIFIER(syn_feature, util.map_label(syn_label, data.unseenclasses), data, data.unseenclasses.size(0), opt.cuda, opt.classifier_lr, 0.5, opt.c2epoch, opt.syn_num, False)
                acc = cls.acc
                best_acc, best_epoch = update_best(acc, epoch, best_acc, best_epoch)
                print(f'unseen class accuracy = {acc:.6f} best acc = {best_acc:.6f} best_epoch = {best_epoch}')
                writer.add_scalar('acc', acc, epoch)
                writer.add_scalar('best_acc', best_acc, epoch)
                 
            # reset G to training mode
            netG.train()
    except:
        traceback.print_exc()

    if opt.gzsl:
        summary = f'seed: {opt.manualSeed} best_acc_unseen: {best_acc[0]: .6f}, best_acc_seen: {best_acc[1]: .6f} best_h: {best_acc[2]: .6f} best_epoch: {best_epoch} ' + comment 
    else:
        summary = f'seed: {opt.manualSeed} best_acc: {best_acc: .6f}, best_epoch: {best_epoch} ' + comment 

    writer.close()
    if not opt.val_mi:
        save_name = f'_{epoch}{suffix}_{comment}.pth'
        torch.save(netG.state_dict(), os.path.join(opt.save_path, 'netG' + save_name))
        torch.save(netD.state_dict(), os.path.join(opt.save_path, 'netD' + save_name))
        if opt.mine:
            torch.save(netM.state_dict(), os.path.join(opt.save_path, 'netM' + save_name))
        print(summary)
        with open(f'{opt.dataset}_log{suffix}.txt', 'a+') as f:
            f.write(summary + '\n')
        f.close()

