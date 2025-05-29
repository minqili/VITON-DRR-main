import torch
import torch.nn as nn
import torchgeometry as tgm

from torchvision.utils import make_grid, save_image

import argparse
import os
import time
from cp_dataset import CPDatasetTest, CPDataLoader
from networks import ConditionGenerator, load_checkpoint, define_D
from tqdm import tqdm
from tensorboardX import SummaryWriter
from utils import *
from get_norm_const import D_logit


def get_opt():
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_ids", default="0")
    parser.add_argument('-j', '--workers', type=int, default=4)
    parser.add_argument('-b', '--batch-size', type=int, default=1)
    parser.add_argument('--fp16', action='store_true', help='use amp')

    parser.add_argument("--dataroot", default='\\path\\to\\dataroot\\')
    parser.add_argument("--datamode", default='test')
    parser.add_argument("--data_list", default='test_pairs.txt')
    parser.add_argument("--datasetting", default="unpaired")
    parser.add_argument("--fine_width", type=int, default=96)
    parser.add_argument("--fine_height", type=int, default=128)

    parser.add_argument('--tensorboard_dir', type=str, default='tensorboard', help='save tensorboard infos')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints', help='save checkpoint infos')
    parser.add_argument('--tocg_checkpoint', type=str, default='.\checkpoints\\tocg_final.pth', help='tocg checkpoint')
    parser.add_argument('--D_checkpoint', type=str, default='.\checkpoints\\D_final.pth', help='D checkpoint')
    
    parser.add_argument("--tensorboard_count", type=int, default=100)
    parser.add_argument("--shuffle", action='store_true', help='shuffle input data')
    parser.add_argument("--semantic_nc", type=int, default=13)
    parser.add_argument("--output_nc", type=int, default=13)
    
    # network
    parser.add_argument("--warp_feature", choices=['encoder', 'T1'], default="T1")
    parser.add_argument("--out_layer", choices=['relu', 'conv'], default="conv")
    
    # training
    parser.add_argument("--clothmask_composition", type=str, choices=['no_composition', 'detach', 'warp_grad'], default='warp_grad')
        

    # Hyper-parameters
    parser.add_argument('--upsample', type=str, default='bilinear', choices=['nearest', 'bilinear'])
    parser.add_argument('--occlusion', action='store_true', help="Occlusion handling")
    parser.add_argument('--cuda', default="cuda", help='cuda or cpu')
    
    # Discriminator
    parser.add_argument('--Ddownx2', action='store_true', help="Downsample D's input to increase the receptive field")  
    parser.add_argument('--Ddropout', action='store_true', help="Apply dropout to D")
    parser.add_argument('--num_D', type=int, default=2, help='Generator ngf')
    parser.add_argument('--spectral', action='store_true', help="Apply spectral normalization to D")
    parser.add_argument('--norm_const', type=float, default=2,help='Normalizing constant for rejection sampling')
    
    opt = parser.parse_args()
    return opt


def mapping(input):
    output = torch.FloatTensor((input.cpu().numpy() == 1).astype(np.float64)).cuda() * 2
    output = torch.FloatTensor((input.cpu().numpy() == 2).astype(np.float64)).cuda() * 13 + output
    output = torch.FloatTensor((input.cpu().numpy() == 3).astype(np.float64)).cuda() * 5 + output
    output = torch.FloatTensor((input.cpu().numpy() == 6).astype(np.float64)).cuda() * 15 + output
    output = torch.FloatTensor((input.cpu().numpy() == 5).astype(np.float64)).cuda() * 14 + output
    output = torch.FloatTensor((input.cpu().numpy() == 4).astype(np.float64)).cuda() * 9 + output
    output = torch.FloatTensor((input.cpu().numpy() == 8).astype(np.float64)).cuda() * 17 + output
    output = torch.FloatTensor((input.cpu().numpy() == 7).astype(np.float64)).cuda() * 16 + output
    output = torch.FloatTensor((input.cpu().numpy() == 10).astype(np.float64)).cuda() * 19 + output
    output = torch.FloatTensor((input.cpu().numpy() == 9).astype(np.float64)).cuda() * 18 + output
    return output

def mapping_2(input):
    output = torch.FloatTensor((input.cpu().numpy() == 1).astype(np.float64)).cuda() * 1
    output = torch.FloatTensor((input.cpu().numpy() == 2).astype(np.float64)).cuda() * 12 + output
    output = torch.FloatTensor((input.cpu().numpy() == 3).astype(np.float64)).cuda() * 4 + output
    output = torch.FloatTensor((input.cpu().numpy() == 6).astype(np.float64)).cuda() * 11 + output
    output = torch.FloatTensor((input.cpu().numpy() == 5).astype(np.float64)).cuda() * 13 + output
    output = torch.FloatTensor((input.cpu().numpy() == 4).astype(np.float64)).cuda() * 8 + output
    output = torch.FloatTensor((input.cpu().numpy() == 8).astype(np.float64)).cuda() * 9 + output
    output = torch.FloatTensor((input.cpu().numpy() == 10).astype(np.float64)).cuda() * 5 + output
    output = torch.FloatTensor((input.cpu().numpy() == 7).astype(np.float64)).cuda() * 10 + output
    output = torch.FloatTensor((input.cpu().numpy() == 9).astype(np.float64)).cuda() * 6 + output
    return output

def Re_size(input):
    face = torch.FloatTensor((input.cpu().numpy() == 2).astype(np.float))
    face = F.interpolate(face, scale_factor=2, mode='bilinear')
    output = torch.FloatTensor((face.cpu().numpy() > 0).astype(np.float)) * 2

    cloth = torch.FloatTensor((input.cpu().numpy() == 3).astype(np.float))
    cloth = F.interpolate(cloth, scale_factor=2, mode='bilinear')
    cloth = torch.FloatTensor((cloth.cpu().numpy() > 0).astype(np.float))
    temp_c = output + cloth
    output += torch.FloatTensor((temp_c.cpu().numpy() == 1).astype(np.float)) * 3

    arm_l = torch.FloatTensor((input.cpu().numpy() == 6).astype(np.float))
    arm_l = F.interpolate(arm_l, scale_factor=2, mode='bilinear')
    arm_l = torch.FloatTensor((arm_l.cpu().numpy() > 0).astype(np.float))
    temp_al = output + arm_l
    output += torch.FloatTensor((temp_al.cpu().numpy() == 1).astype(np.float)) * 6

    arm_r = torch.FloatTensor((input.cpu().numpy() == 5).astype(np.float))
    arm_r = F.interpolate(arm_r, scale_factor=2, mode='bilinear')
    arm_r = torch.FloatTensor((arm_r.cpu().numpy() > 0).astype(np.float))
    temp_ar = output + arm_r
    output += torch.FloatTensor((temp_ar.cpu().numpy() == 1).astype(np.float)) * 5

    pants = torch.FloatTensor((input.cpu().numpy() == 4).astype(np.float))
    pants = F.interpolate(pants, scale_factor=2, mode='bilinear')
    pants = torch.FloatTensor((pants.cpu().numpy() > 0).astype(np.float))
    temp_p = output + pants
    output += torch.FloatTensor((temp_p.cpu().numpy() == 1).astype(np.float)) * 4

    leg_l = torch.FloatTensor((input.cpu().numpy() == 8).astype(np.float))
    leg_l = F.interpolate(leg_l, scale_factor=2, mode='bilinear')
    leg_l = torch.FloatTensor((leg_l.cpu().numpy() > 0).astype(np.float))
    temp_ll = output + leg_l
    output += torch.FloatTensor((temp_ll.cpu().numpy() == 1).astype(np.float)) * 8

    leg_r = torch.FloatTensor((input.cpu().numpy() == 7).astype(np.float))
    leg_r = F.interpolate(leg_r, scale_factor=2, mode='bilinear')
    leg_r = torch.FloatTensor((leg_r.cpu().numpy() > 0).astype(np.float))
    temp_lr = output + leg_r
    output += torch.FloatTensor((temp_lr.cpu().numpy() == 1).astype(np.float)) * 7

    foot_l = torch.FloatTensor((input.cpu().numpy() == 10).astype(np.float))
    foot_l = F.interpolate(foot_l, scale_factor=2, mode='bilinear')
    foot_l = torch.FloatTensor((foot_l.cpu().numpy() > 0).astype(np.float))
    temp_fl = output + foot_l
    output += torch.FloatTensor((temp_fl.cpu().numpy() == 1).astype(np.float)) * 10

    foot_r = torch.FloatTensor((input.cpu().numpy() == 9).astype(np.float))
    foot_r = F.interpolate(foot_r, scale_factor=2, mode='bilinear')
    foot_r = torch.FloatTensor((foot_r.cpu().numpy() > 0).astype(np.float))
    temp_fr = output + foot_r
    output += torch.FloatTensor((temp_fr.cpu().numpy() == 1).astype(np.float)) * 9

    hair = torch.FloatTensor((input.cpu().numpy() == 1).astype(np.float))
    hair = F.interpolate(hair, scale_factor=2, mode='bilinear')
    hair = torch.FloatTensor((hair.cpu().numpy() > 0).astype(np.float))
    temp_h = output + hair
    output += torch.FloatTensor((temp_h.cpu().numpy() == 1).astype(np.float))

    return output.cuda()


def test(opt, test_loader, board, tocg, D=None):
    # Model
    up = nn.Upsample(size=(256, 192), mode='bilinear')
    gauss = tgm.image.GaussianBlur((15, 15), (3, 3))
    gauss.cuda()
    tocg.cuda()
    tocg.eval()
    if D is not None:
        D.cuda()
        D.eval()
    
    a = os.makedirs(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'multi-task'), exist_ok=True)
    num = 0
    iter_start_time = time.time()
    if D is not None:
        D_score = []
    for inputs in test_loader.data_loader:
        
        # input1
        c_paired = inputs['cloth'][opt.datasetting].cuda()
        cm_paired = inputs['cloth_mask'][opt.datasetting].cuda()
        cm_paired = torch.FloatTensor((cm_paired.detach().cpu().numpy() > 0.5).astype(np.float64)).cuda()
        # input2
        parse_agnostic = inputs['parse_agnostic'].cuda()
        densepose = inputs['densepose'].cuda()
        # openpose = inputs['pose'].cuda()
        # GT
        label_onehot = inputs['parse_onehot'].cuda()  # CE
        label = inputs['parse'].cuda()  # GAN loss
        parse_cloth_mask = inputs['pcm'].cuda()  # L1
        im_c = inputs['parse_cloth'].cuda()  # VGG
        # visualization
        im = inputs['image']

        with torch.no_grad():
            # inputs
            cm_paired = cm_paired.squeeze()
            if cm_paired.shape[-1] == 3:
                print(c_paired.shape, cm_paired.shape)
                cm_paired = cm_paired[:,:,0].squeeze().unsqueeze(0).unsqueeze(0)
            else :
                cm_paired = cm_paired.squeeze().unsqueeze(0).unsqueeze(0)

            input1 = torch.cat([c_paired, cm_paired], 1)
            # input1 = cm_paired
            input2 = torch.cat([parse_agnostic, densepose], 1)

            # forward
            flow_list, fake_segmap, warped_cloth_paired, warped_clothmask_paired = tocg(opt, input1, input2)
            
            # warped cloth mask one hot 
            warped_cm_onehot = torch.FloatTensor((warped_clothmask_paired.detach().cpu().numpy() > 0.5).astype(np.float64)).cuda()
            
            if opt.clothmask_composition != 'no_composition':
                if opt.clothmask_composition == 'detach':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_cm_onehot
                    fake_segmap = fake_segmap * cloth_mask

                if opt.clothmask_composition == 'warp_grad':
                    cloth_mask = torch.ones_like(fake_segmap)
                    cloth_mask[:,3:4, :, :] = warped_clothmask_paired
                    fake_segmap = fake_segmap * cloth_mask
            if D is not None:
                fake_segmap_softmax = F.softmax(fake_segmap, dim=1)
                pred_segmap = D(torch.cat((input1.detach(), input2.detach(), fake_segmap_softmax), dim=1))
                score = D_logit(pred_segmap)
                # score = torch.exp(score) / opt.norm_const
                score = (score / (1 - score)) / opt.norm_const
                print("prob0", score)
                for i in range(cm_paired.shape[0]):
                    name = inputs['c_name']['unpaired'][i].replace('.jpg', '.png')
                    D_score.append((name, score[i].item()))
            
            
            # generated fake cloth mask & misalign mask
            fake_clothmask = (torch.argmax(fake_segmap.detach(), dim=1, keepdim=True) == 3).long()
            misalign = fake_clothmask - warped_cm_onehot
            misalign[misalign < 0.0] = 0.0
        
        for i in range(c_paired.shape[0]):
            grid = make_grid([(c_paired[i].cpu() / 2 + 0.5), (cm_paired[i].cpu()).expand(3, -1, -1), visualize_segmap(parse_agnostic.cpu(), batch=i), ((densepose.cpu()[i]+1)/2),
                            (im_c[i].cpu() / 2 + 0.5), parse_cloth_mask[i].cpu().expand(3, -1, -1),  (warped_cm_onehot[i].cpu().detach()).expand(3, -1, -1),
                            visualize_segmap(label.cpu(), batch=i), visualize_segmap(fake_segmap.cpu(), batch=i), (im[i]/2 +0.5), (misalign[i].cpu().detach()).expand(3, -1, -1)],
                                nrow=4)
            save_image(grid, os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                             opt.datamode, opt.datasetting, 'multi-task',
                             (inputs['c_name']['unpaired'][i].split('.')[0] + '_' +
                              inputs['c_name']['unpaired'][i].split('.')[0] + '.png')))
            fake_segmap = gauss(up(fake_segmap))
            pre_seg = torch.argmax(F.log_softmax(fake_segmap, dim=1), dim=1, keepdim=True)
            pre_seg = torch.tensor(pre_seg, dtype=torch.float32)
            # pre_seg = Re_size(pre_seg)
            pre_seg = mapping_2(pre_seg).squeeze().detach().cpu().numpy()
            # pre_seg = F.interpolate(pre_seg, scale_factor=2, mode='bilinear').squeeze().detach().numpy()
            iname = inputs['im_name'][0].split('=')
            path = os.path.join("../dataset" + '/seg_pre',(iname[0] + '.png'))
            cv2.imwrite(path, pre_seg)

        num += c_paired.shape[0]
        print(num)
    if D is not None:
        D_score.sort(key=lambda x: x[1], reverse=True)
        # Save D_score
        for name, score in D_score:
            f = open(os.path.join('./output', opt.tocg_checkpoint.split('/')[-2], opt.tocg_checkpoint.split('/')[-1],
                                opt.datamode, opt.datasetting, 'multi-task', 'rejection_prob.txt'), 'a')
            f.write(name + ' ' + str(score) + '\n')
            f.close()
    print(f"Test time {time.time() - iter_start_time}")


def main():
    opt = get_opt()
    print(opt)
    print("Start to test %s!")
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_ids
    
    # create test dataset & loader
    test_dataset = CPDatasetTest(opt)
    test_loader = CPDataLoader(opt, test_dataset)
    
    # visualization
    if not os.path.exists(opt.tensorboard_dir):
        os.makedirs(opt.tensorboard_dir)
    board = SummaryWriter(log_dir=os.path.join(opt.tensorboard_dir, opt.tocg_checkpoint.split('/')[-1], opt.tocg_checkpoint.split('/')[-1], opt.datamode, opt.datasetting))

    # Model
    input1_nc = 4  # cloth + cloth-mask
    input2_nc = opt.semantic_nc + 3  # parse_agnostic + densepose
    tocg = ConditionGenerator(opt, input1_nc=input1_nc, input2_nc=input2_nc, output_nc=opt.output_nc, ngf=96, norm_layer=nn.BatchNorm2d)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        if opt.norm_const is None:
            raise NotImplementedError
        D = define_D(input_nc=input1_nc + input2_nc + opt.output_nc, Ddownx2 = opt.Ddownx2, Ddropout = opt.Ddropout, n_layers_D=3, spectral = opt.spectral, num_D = opt.num_D)
    else:
        D = None
    # Load Checkpoint
    load_checkpoint(tocg, opt.tocg_checkpoint,opt)
    if not opt.D_checkpoint == '' and os.path.exists(opt.D_checkpoint):
        load_checkpoint(D, opt.D_checkpoint, opt)
    # Train
    test(opt, test_loader, board, tocg, D=D)

    print("Finished testing!")


if __name__ == "__main__":
    main()
