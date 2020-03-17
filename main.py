import argparse
import os
import sys
import model as md
#os.chdir(os.getcwd())
#os.chdir(r'/home/yhg/Desktop/ct-denoising/RED_CNM')
#sys.path.extend([os.path.abspath("."), os.path.abspath("./..")])

#from time import time
#os.chdir(os.getcwd() + '/..')
#print('pwd : {}'.format(os.getcwd()))

parser = argparse.ArgumentParser(description='')
#set load directory

parser.add_argument('--dcm_path', dest='dcm_path', default='/data1/SNUH-denoising/Force LDCT/Force LDCT(241-360)', help='dicom file directory')
#parser.add_argument('--input_path', dest='input_path', default='DE_Thorax 1.0 Br59 M_0.8', help='LDCT image folder name')
#parser.add_argument('--target_path', dest='target_path', default='DE_Thorax 1.0 Br40 M_0.8', help='NDCT image folder name')
parser.add_argument('--input_path', dest='input_path', default='/home/boh001/image/M_LDCT', help='LDCT image folder name')
parser.add_argument('--target_path', dest='target_path', default='/home/boh001/image/M_NDCT', help='NDCT image folder name')
parser.add_argument('--extension', dest='extension', default= 'DCM', help='extension, [IMA, DCM]')
parser.add_argument('--patch_size', dest='patch_size', type = int ,default= '52', help='patch_size')
parser.add_argument('--image_size', dest='image_size', type = int , default= '256', help='image_size')
parser.add_argument('--phase', dest='phase', default='train', help='train or test')
parser.add_argument('--model', dest='model', default='red_cnn', help='model')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='batch size')
parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='epoch')
parser.add_argument('--loss', dest = 'loss', default = 'mse', help = 'loss')
parser.add_argument('--optimizer', dest = 'optimizer', default= 'adam', help = 'optimizer')
parser.add_argument('--gpu_no', dest='gpu_no',  default=0, help='gpu no')
parser.add_argument('--train_ratio', dest='train_ratio', type=float, default=0.7, help='# of train samples')
parser.add_argument('--inf_ratio', dest='inf_ratio', type=float, default=0.2, help='# of train samples')
parser.add_argument('--seed', dest='seed', type=int, default=0, help='ramdom sampling seed num(for train/test)')
# -------------------------------------
args = parser.parse_args()
print(args)
#os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_no)

model = md.TTI(args)

if args.phase == 'train':
    model.train()
else:
    model.test()

