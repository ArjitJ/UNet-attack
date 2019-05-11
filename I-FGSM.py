import torch
import torch.nn as nn
from torch.autograd.gradcheck import zero_gradients
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torchvision import transforms


def ConvReLUBN(in_channels, out_channels, kernel_size=3, padding = 0):
    return nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = out_channels, kernel_size=kernel_size, padding = padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels))

def UpConv(in_channels, out_channels, mode='transpose', kernel_size=3): #Mode used in the
    #paper is transpose
    if mode == 'transpose':
        return nn.ConvTranspose2d(in_channels = in_channels, out_channels = out_channels, kernel_size=3, \
                                  stride=2, padding = 1, output_padding = 1)
    else:
    #Use Bilinear Upsampling
        return nn.Sequential(nn.Upsample(mode='bilinear', scale_factor=kernel_size), \
                             ConvReLUBN(in_channels, out_channels, kernel_size=1))
class UNet(nn.Module):
    def __init__(self, out_channels, in_channels):
        super(UNet, self).__init__()
        self.conv_block1 = ConvReLUBN(in_channels, 64)
        self.conv_block2 = ConvReLUBN(64, 64)
        self.pool_1 = nn.MaxPool2d(kernel_size=2)
        self.conv_block3 = ConvReLUBN(64, 128)
        self.conv_block4 = ConvReLUBN(128, 128)
        self.pool_2 = nn.MaxPool2d(kernel_size=2)
        self.conv_block5 = ConvReLUBN(128, 256)
        self.conv_block6 = ConvReLUBN(256, 256)
        self.pool_3 = nn.MaxPool2d(kernel_size=2)
        self.conv_block7 = ConvReLUBN(256, 512)
        self.conv_block8 = ConvReLUBN(512, 512)
        self.upconv_block1 = UpConv(512, 256, kernel_size=3)
        self.conv_block9 = ConvReLUBN(512, 256)
        self.conv_block10 = ConvReLUBN(256, 256)
        self.upconv_block2 = UpConv(256, 128, kernel_size=3)
        self.conv_block11 = ConvReLUBN(256, 128)
        self.conv_block12 = ConvReLUBN(128, 128)
        self.upconv_block3 = UpConv(128, 64, kernel_size=3)
        self.conv_block13 = ConvReLUBN(128, 64)
        self.conv_block14 = ConvReLUBN(64, 64)
        self.conv_block15 = ConvReLUBN(64, out_channels, padding = 1)
    
    def copyandcrop(self, upsampled, bypass, crop=False):    
        if crop:
            c1 = (bypass.size()[2] - upsampled.size()[2])
            c2 = (bypass.size()[3] - upsampled.size()[3])
            bypass = bypass[:, :, c1//2:-(c1-c1//2), c2//2:-(c2-c2//2)]
        return torch.cat((upsampled, bypass), 1)
               
    def forward(self, x):
        layer1 = self.conv_block1(x)
        layer2 = self.conv_block2(layer1)
        layer3 = self.pool_1(layer2)
        layer4 = self.conv_block3(layer3)
        layer5 = self.conv_block4(layer4)
        layer6 = self.pool_2(layer5)
        layer7 = self.conv_block5(layer6)
        layer8 = self.conv_block6(layer7)
        layer9 = self.pool_3(layer8)
        layer10 = self.conv_block7(layer9)
        layer11 = self.conv_block8(layer10)
        layer12 = self.upconv_block1(layer11)
        layer13 = self.copyandcrop(layer12, layer8, crop=True)
        layer14 = self.conv_block9(layer13)
        layer15 = self.conv_block10(layer14)
        layer16 = self.upconv_block2(layer15)
        layer17 = self.copyandcrop(layer16, layer5, crop=True)
        layer18 = self.conv_block11(layer17)
        layer19 = self.conv_block12(layer18)
        layer20 = self.upconv_block3(layer19)
        layer21 = self.copyandcrop(layer20, layer2, crop=True)
        layer22 = self.conv_block13(layer21)
        layer23 = self.conv_block14(layer22)
        layer24 = self.conv_block15(layer23)
        return layer24


batch_size = 1

model = UNet(in_channels = 1, out_channels = 1).cuda()
model.load_state_dict(torch.load('./Lung-BCE.pt'))

input_img = Image.open("./MCUCXR_0013_0.png")
imgtransform = transforms.Compose([transforms.Resize((572, 572)), transforms.ToTensor()])
input_img = imgtransform(input_img).unsqueeze(0).cuda()
img_variable = Variable(input_img, requires_grad = True).cuda()

output_true = model(img_variable)

target = Image.open("./mm.png")
target_arr = np.array(target)
target_arr = target_arr / (target_arr.max() - target_arr.min())
target  = transforms.ToTensor()(Image.fromarray(target_arr)).unsqueeze(0)


target_variable = Variable(target, requires_grad = True).cuda()
epsilon = 1e-1
num_steps = 25
alpha = 1e-3

for i in range(num_steps):
  zero_gradients(img_variable)
  output = model(img_variable)
  loss = torch.nn.BCEWithLogitsLoss()
  loss_cal = loss(output, target_variable)
  img_variable.retain_grad()
  loss_cal.backward()
  x_grad = alpha * torch.sign(img_variable.grad.data)
  diff = img_variable.data - x_grad
  grad = diff - input_img
  grad = torch.clamp(grad, -epsilon, epsilon)
  x_adv = input_img + grad
  img_variable.data = x_adv
  
output_adv = model(img_variable)

plt.imshow(input_img.cpu().detach().numpy()[0, 0, :, :].T)

plt.imshow(img_variable.cpu().detach().numpy()[0, 0, :, :].T)

plt.imshow(torch.sigmoid(output_adv).cpu().detach().numpy()[0, 0, :, :].T > 0.5)

plt.imshow(torch.sigmoid(output_true).cpu().detach().numpy()[0, 0, :, :].T > 0.5)

plt.imshow(abs(input_img.cpu().detach().numpy()[0, 0, :, :].T - img_variable.cpu().detach().numpy()[0, 0, :, :].T))

