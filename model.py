import torch
import torch.nn as nn
import network as nt
import dataset as dt
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import math
import matplotlib.pyplot as plt
from skimage import measure
from torch.utils.tensorboard import SummaryWriter
import torchvision.models as models
import torch
import torch.autograd as autograd
import torch.nn.functional as F
import time
from collections import OrderedDict
from matplotlib import cm

class TTI(object):
    def __init__(self,args):
        self.epoch = args.epoch
        self.phase = args.phase
        self.model = args.model
        self.batch_size = args.batch_size
        self.dataset = dt.CtDataset(args)
        self.gpu_no = args.gpu_no
        self.loss = args.loss
        self.data_loader = DataLoader(self.dataset,batch_size=self.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.dataset,batch_size =1, shuffle= False )


        if self.loss == 'mse':
            self.loss = nn.MSELoss().cuda()
        #net
        if self.model =='red_cnn':
            self.net = nn.DataParallel(nt.RED_CNN().cuda())
            if self.phase =='train':
                self.net.train()
            else:
                self.net.eval()

        elif self.model =='wgan':

            if self.phase =='train':
                self.g = nn.DataParallel(nt.W_Generator()).cuda()
                self.d = nn.DataParallel(nt.W_Discriminator()).cuda()
                self.v = models.vgg19(pretrained=True).cuda()

            else:
                self.g = nt.W_Generator()
                self.d = nt.W_Discriminator().cuda()
                self.v = models.vgg19(pretrained=True).cuda()
            self.g_state = {'epoch': self.epoch, 'state_dict': self.g.state_dict()}
        else:
            print('안 만들었다 아직')




        if args.optimizer == 'adam':
            if self.model == 'red_cnn':
                self.optimizer = optim.Adam(self.net.parameters(),lr=0.000001)
            elif self.model == 'wgan':
                self.g_optimizer = optim.Adam(self.g.parameters(),lr=0.000001,betas=(0.5, 0.9))
                self.d_optimizer = optim.Adam(self.d.parameters(), lr=0.000001,betas=(0.5, 0.9))
                #self.v_optimizer = optim.Adam(self.v.parameters(), lr=0.01)


        else:
            print('안 만들었다 아직')









    def train(self):

        writer = SummaryWriter()
        def xavier_weights_init(m):
            classname = m.__class__.__name__
            if classname.find('Conv') != -1:
                nn.init.xavier_normal_(m.weight, 1)


        if self.model == 'red_cnn':
            lr_sc = lr_scheduler.StepLR(self.optimizer, step_size=10)

            self.net.apply(xavier_weights_init)
            for e in range(self.epoch+1):

                startTime = time.time()
                filepath = '/home/boh001/save_model/{}/{}.pth'.format(self.model, e)
                total_loss = 0
                print('epoch :',e)

                for x, y in self.data_loader:
                    input = x.cuda()
                    output = self.net(input).cuda()
                    target = y.cuda()


                    l = self.loss(output,target)
                    self.optimizer.zero_grad()
                    l.backward()
                    self.optimizer.step()
                    lr_sc.step()
                    total_loss += l.item()

                endTime = time.time() - startTime
                print('{} seconds per Epoch :'.format(endTime))

                if e % 10 == 0:
                    print(f"Training loss: {total_loss}")
                    writer.add_scalar('Loss/train', total_loss, e)
                    writer.close()

                    torch.save(self.net.state_dict(), filepath)
                    print('Save done : epoch {}'.format(e))


        elif self.model =='wgan':

            self.g.apply(xavier_weights_init)
            self.d.apply(xavier_weights_init)


            def calc_gradient_penalty(netD, real_data, fake_data):
                alpha = torch.rand(real_data.shape)
                alpha = alpha.cuda()

                interpolates = alpha * real_data + ((1 - alpha) * fake_data)


                interpolates = interpolates.cuda()
                interpolates = autograd.Variable(interpolates, requires_grad=True)

                disc_interpolates = netD(interpolates)

                gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                          grad_outputs=torch.ones(disc_interpolates.size()).cuda() ,create_graph=True, retain_graph=True, only_inputs=True)[0]

                gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
                return gradient_penalty

            for e in range(self.epoch+1):
                filepath = '/home/boh001/save_model/{}/{}.pth'.format(self.model, e)
                for p in self.d.parameters():
                    p.requires_grad = True
                d_steps = 4
                g_steps = 1
                Wasserstein_D = 0
                one = torch.FloatTensor([1]).cuda()
                mone = one * -1

                print('epoch :', e)
                for x, y in self.data_loader:
                    for d_index in range(d_steps):
                        #print(d_index)
                        self.d.zero_grad()
                        for p in self.d.parameters():
                            p.requires_grad = True
                        for p in self.g.parameters():
                            p.requires_grad = False
                        x = x.cuda()
                        y = y.cuda()
                        g_out = self.g(x).cuda()
                        d_out_fake = self.d(g_out).cuda()
                        #print('d_out_fake:',d_out_fake.shape)
                        d_loss_fake = d_out_fake.mean()
                        #print(d_loss_fake)
                        d_loss_fake.backward(one)

                        d_out_x = self.d(y).cuda()
                        d_loss_real = d_out_x.mean()
                        #print(d_loss_real)
                        d_loss_real.backward(mone)


                        gradient_penalty = calc_gradient_penalty(self.d,y, g_out)*10
                        gradient_penalty.backward(one)

                        d_loss = d_loss_fake - d_loss_real + gradient_penalty
                        Wasserstein_D = d_loss_real - d_loss_fake

                        self.d_optimizer.step()

                    for g_index in range(g_steps):
                        self.g.zero_grad()
                        for p in self.d.parameters():
                            p.requires_grad = False

                        for p in self.g.parameters():
                            p.requires_grad = True


                        g_fake_data = self.g(x)
                        dg_fake_decision = self.d(g_fake_data)
                        g_error = dg_fake_decision.mean()

                        g_error.backward(mone)




                       # g_fake_data = g_fake_data * 255.0
                       # y = y * 255.0

                        v = nn.DataParallel(self.v.features).cuda()

                        for p in self.v.parameters():
                            p.requires_grad = False
                        g_fake_data = self.g(x)
                        g_fake_data = F.upsample(g_fake_data,size = (224,224))

                        concat_fake = torch.cat((g_fake_data,g_fake_data),1)
                        concat_fake = torch.cat((concat_fake,g_fake_data),1)
                        gv_fake_decision = v(concat_fake)

                        y = F.upsample(y,size = (224,224))
                        concat_real = torch.cat((y,y),1)
                        concat_real  = torch.cat((concat_real,y),1)
                        gv_real_decision = v(concat_real)
                        #print((gv_fake_decision-gv_real_decision).norm(2,1).sum()*0.1)
                        per_loss = ((gv_fake_decision-gv_real_decision).norm(2,1).sum()*10)/(gv_fake_decision.shape[1]*gv_fake_decision.shape[2]*gv_fake_decision.shape[3])
                        per_loss.backward()
                        self.g_optimizer.step()


                print('d_loss :',d_loss)
                print('fake :',d_loss_fake)
                print('real :',d_loss_real)
                print('GP :',gradient_penalty)
                print('g_loss : ', per_loss-g_error)
                print('per_loss :',per_loss)
                print('g_error :',g_error)

                writer.add_scalar('d_loss :',d_loss, e)
                writer.add_scalar('g_loss : ', per_loss-g_error, e)

                writer.close()

                if e % 10 == 0:
                    torch.save(self.g_state, filepath)
                    print('Save done : epoch {}'.format(e))



    # evaluate
    def test(self):
        i = 0
        def show_img(input, output, target):
            plt.figure(figsize=(20, 40))
            plt.subplot(1, 3, 1).title.set_text('input')
            plt.imshow(input[0][0].cpu().detach().numpy(),cmap=cm.gray, vmin=0, vmax=1)
            plt.subplot(1, 3, 2).title.set_text('output')
            plt.imshow(output[0][0].cpu().detach().numpy(),cmap=cm.gray, vmin=0, vmax=1)
            plt.subplot(1, 3, 3).title.set_text('target')
            plt.imshow(target[0][0].cpu().detach().numpy(),cmap=cm.gray, vmin=0, vmax=1)

        def psnr(img1, img2, PIXEL_MAX=255.0):
            mse = np.mean((img1 - img2) ** 2)
            if mse == 0:
                return 100
            return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

        def rmse(img1, img2):
            mse = np.mean((img1 - img2) ** 2)
            return math.sqrt(mse)

        if self.model == 'red_cnn':


            for x, y in self.test_loader:
                filepath = '/home/boh001/test/{}/{}'.format(self.model,i)
                print(i)
                i += 1
                input = x

                self.net.load_state_dict(torch.load('/home/boh001/save_model/red_cnn/50.pth'))
                output = self.net(input)
                # print(input[0].shape)
                target = y

                PSNR = psnr(output[0][0].cpu().detach().numpy(), target[0][0].cpu().detach().numpy())
                RMSE = rmse(output[0][0].cpu().detach().numpy(), target[0][0].cpu().detach().numpy())
                SSIM = measure.compare_ssim(output[0][0].cpu().detach().numpy(), target[0][0].cpu().detach().numpy())

                show_img(input, output, target)
                plt.savefig(filepath)
                print('PSNR :', PSNR)
                print('RMSE :', RMSE)
                print('SSIM :', SSIM)

        elif self.model =='wgan':
            i = 0
            for x, y in self.test_loader:
                i += 1
                print(i)

                filepath = '/home/boh001/test/{}/{}'.format(self.model,i)
                input = x['image'].float()


                self.g.load_state_dict(self.g_state['state_dict'])
                output = self.g(input)
                target = y['labels']

                show_img(input, output, target)
                plt.savefig(filepath)
                plt.close()








