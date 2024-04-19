import torch
import torch.nn as nn
import torch.nn.functional as function
import numpy as np

""" segmentation model example
"""

class Unet(nn.Module):
    def __init__(self):
        super().__init__()

        """ Encoder """
        self.e1 = encoder_block(3, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)

        """ Classifier """
        self.outputs = nn.Conv2d(64, 34, kernel_size=1, padding=0)

    def forward(self, inputs):
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        """ Segmentation output """
        outputs = self.outputs(d4)
        return outputs


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        #self.drop = nn.Dropout(0.25)

        self.relu = nn.ReLU()

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #x = self.drop(x)

        return x

""" Encoder block:
    It consists of an conv_block followed by a max pooling.
    Here the number of filters doubles and the height and width half after every block.
"""
class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)

        return x, p

""" Decoder block:
    The decoder block begins with a transpose convolution, followed by a concatenation with the skip
    connection from the encoder block. Next comes the conv_block.
    Here the number filters decreases by half and the height and width doubles.
"""
class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)

        return x


class SegNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(kernel_size=2, stride=2)

        # Encode stage 1
        self.enc_1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_1b = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.norm_enc_1a = nn.BatchNorm2d(64, momentum=0.5)
        self.norm_enc_1b = nn.BatchNorm2d(64, momentum=0.5)

        # Encode stage 2
        self.enc_2a = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.enc_2b = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.norm_enc_2a = nn.BatchNorm2d(128, momentum=0.5)
        self.norm_enc_2b = nn.BatchNorm2d(128, momentum=0.5)

        # Encode stage 3
        self.enc_3a = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.enc_3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.enc_3c = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.norm_enc_3a = nn.BatchNorm2d(256, momentum=0.5)
        self.norm_enc_3b = nn.BatchNorm2d(256, momentum=0.5)
        self.norm_enc_3c = nn.BatchNorm2d(256, momentum=0.5)
        
        # Encode stage 4
        self.enc_4a = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.enc_4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_4c = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm_enc_4a = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_enc_4b = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_enc_4c = nn.BatchNorm2d(512, momentum=0.5)

        # Encode stage 5
        self.enc_5a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.enc_5c = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm_enc_5a = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_enc_5b = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_enc_5c = nn.BatchNorm2d(512, momentum=0.5) 

        # Decode stage 5
        #self.up_5 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0)    
        self.dec_5a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_5b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_5c = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.norm_dec_5a = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_dec_5b = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_dec_5c = nn.BatchNorm2d(512, momentum=0.5) 

        # Decode stage 4    
        #self.up_4 = nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2, padding=0) 
        self.dec_4a = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_4b = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.dec_4c = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.norm_dec_4a = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_dec_4b = nn.BatchNorm2d(512, momentum=0.5)
        self.norm_dec_4c = nn.BatchNorm2d(256, momentum=0.5)

        # Decode stage 3    
        #self.up_3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0) 
        self.dec_3a = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_3b = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.dec_3c = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.norm_dec_3a = nn.BatchNorm2d(256, momentum=0.5)
        self.norm_dec_3b = nn.BatchNorm2d(256, momentum=0.5)
        self.norm_dec_3c = nn.BatchNorm2d(128, momentum=0.5)

        # Decode stage 2  
        #self.up_2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)   
        self.dec_2a = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.dec_2b = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.norm_dec_2a = nn.BatchNorm2d(128, momentum=0.5)
        self.norm_dec_2b = nn.BatchNorm2d(64, momentum=0.5)

        # Decode stage 1  
        #self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)   
        self.dec_1a = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.dec_1b = nn.Conv2d(64, 19, kernel_size=1, padding=0)
        self.norm_dec_1 = nn.BatchNorm2d(64, momentum=0.5)     

    def forward(self, x):
        # Encode stage 1
        x = function.relu(self.norm_enc_1a(self.enc_1a(x))) 
        x1 = function.relu(self.norm_enc_1b(self.enc_1b(x))) 
        x, ind1 = self.pool(x1)
        size1 = x.size()

        # Encode stage 2
        x = function.relu(self.norm_enc_2a(self.enc_2a(x))) 
        x2 = function.relu(self.norm_enc_2b(self.enc_2b(x))) 
        x, ind2 = self.pool(x2)
        size2 = x.size()

        # Encode stage 3
        x = function.relu(self.norm_enc_3a(self.enc_3a(x))) 
        x = function.relu(self.norm_enc_3b(self.enc_3b(x))) 
        x3 = function.relu(self.norm_enc_3c(self.enc_3c(x)))
        x, ind3 = self.pool(x3)
        size3 = x.size()

        # Encode stage 4
        x = function.relu(self.norm_enc_4a(self.enc_4a(x))) 
        x = function.relu(self.norm_enc_4b(self.enc_4b(x))) 
        x4 = function.relu(self.norm_enc_4c(self.enc_4c(x)))
        x, ind4 = self.pool(x4)
        size4 = x.size()

        # Encode stage 5
        x = function.relu(self.norm_enc_5a(self.enc_5a(x))) 
        x = function.relu(self.norm_enc_5b(self.enc_5b(x))) 
        x5 = function.relu(self.norm_enc_5c(self.enc_5c(x)))
        x, ind5 = self.pool(x5)
        size5 = x.size()

        # Decode Stage 5
        x = self.unpool(x, ind5, output_size=size4)
        #x = self.up_5(x)
        #x = torch.cat([x, x5], axis=1)
        x = function.relu(self.norm_dec_5a(self.dec_5a(x)))
        x = function.relu(self.norm_dec_5b(self.dec_5b(x)))
        x = function.relu(self.norm_dec_5c(self.dec_5c(x)))

        # Decode Stage 4
        x = self.unpool(x, ind4, output_size=size3)
        #x = self.up_4(x)
        #x = torch.cat([x, x4], axis=1)
        x = function.relu(self.norm_dec_4a(self.dec_4a(x)))
        x = function.relu(self.norm_dec_4b(self.dec_4b(x)))
        x = function.relu(self.norm_dec_4c(self.dec_4c(x)))

        # Decode Stage 3
        x = self.unpool(x, ind3, output_size=size2)
        #x = self.up_3(x)
        #x = torch.cat([x, x3], axis=1)
        x = function.relu(self.norm_dec_3a(self.dec_3a(x)))
        x = function.relu(self.norm_dec_3b(self.dec_3b(x)))
        x = function.relu(self.norm_dec_3c(self.dec_3c(x)))

        # Decode Stage 2
        x = self.unpool(x, ind2, output_size=size1)
        #x = self.up_2(x)
        #x = torch.cat([x, x2], axis=1)
        x = function.relu(self.norm_dec_2a(self.dec_2a(x)))
        x = function.relu(self.norm_dec_2b(self.dec_2b(x)))

        # Decode Stage 1
        x = self.unpool(x, ind1)
        #x = self.up_1(x)
        #x = torch.cat([x, x1], axis=1)
        x = function.relu(self.norm_dec_1(self.dec_1a(x)))
        x = self.dec_1b(x)

        return x   


class Efficiency_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.enc_1a = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.enc_1b = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4)
        self.norm_enc_1a = nn.BatchNorm2d(64)
        self.norm_enc_1b = nn.BatchNorm2d(64)

        self.enc_2a = nn.Conv2d(64, 128, kernel_size=3, padding=1, groups=4)
        self.enc_2b = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=8)
        self.norm_enc_2a = nn.BatchNorm2d(128)
        self.norm_enc_2b = nn.BatchNorm2d(128)

        self.enc_3a = nn.Conv2d(128, 256, kernel_size=3, padding=1, groups=8)
        self.enc_3b = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=16)
        self.norm_enc_3a = nn.BatchNorm2d(256)
        self.norm_enc_3b = nn.BatchNorm2d(256)
        
        self.conv_latent_a = nn.Conv2d(256, 512, kernel_size=3, padding=1, groups=16)
        self.conv_latent_b = nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=32)
        self.norm_lat_a = nn.BatchNorm2d(512)
        self.norm_lat_b = nn.BatchNorm2d(512)

        self.up_3 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=3, padding=0)
        self.dec_3a = nn.Conv2d(512, 256, kernel_size=3, padding=1, groups=16)
        self.dec_3b = nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=16)
        self.norm_dec_3a = nn.BatchNorm2d(256)
        self.norm_dec_3b = nn.BatchNorm2d(256)

        self.up_2 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=3, padding=0)
        self.dec_2a = nn.Conv2d(256, 128, kernel_size=3, padding=1, groups=8)
        self.dec_2b = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=8)
        self.norm_dec_2a = nn.BatchNorm2d(128)
        self.norm_dec_2b = nn.BatchNorm2d(128)

        self.up_1 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=3, padding=0)
        self.dec_1a = nn.Conv2d(128, 64, kernel_size=3, padding=1, groups=4)
        self.dec_1b = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=4)
        self.norm_dec_1a = nn.BatchNorm2d(64)
        self.norm_dec_1b = nn.BatchNorm2d(64)

        self.out = nn.Conv2d(64, 19, kernel_size=1, padding=0)

        self.pool = nn.MaxPool2d((3, 3))
        self.dropout = nn.Dropout(0.25)
        self.dequant = torch.quantization.DeQuantStub()

    def forward(self, x):
        # Encode
        x = self.quant(x)
        print(np.shape(x))
        x = self.norm_enc_1a(function.relu(self.enc_1a(x)))
        x1 = self.norm_enc_1b(function.relu(self.enc_1b(x)))
        x = self.dropout(self.pool(x1))

        # Encode
        print(np.shape(x))
        x = self.norm_enc_2a(function.relu(self.enc_2a(x)))
        x2 = self.norm_enc_2b(function.relu(self.enc_2b(x)))
        x = self.dropout(self.pool(x2))

        # Encode
        print(np.shape(x))
        x = self.norm_enc_3a(function.relu(self.enc_3a(x)))
        x3 = self.norm_enc_3b(function.relu(self.enc_3b(x)))
        x = self.dropout(self.pool(x3))

        # Latent
        print(np.shape(x))
        x = self.norm_lat_a(function.relu(self.conv_latent_a(x)))
        x = self.norm_lat_b(function.relu(self.conv_latent_b(x)))

        # Decode
        x = self.up_3(x)
        print(np.shape(x))
        x = torch.cat([x, x3], dim=1)
        x = self.norm_dec_3a(function.relu(self.dec_3a(x)))
        x = self.dropout(self.norm_dec_3b(function.relu(self.dec_3b(x))))

        # Decode
        x = self.up_2(x)
        print(np.shape(x))
        x = torch.cat([x, x2], dim=1)
        x = self.norm_dec_2a(function.relu(self.dec_2a(x)))
        x = self.dropout(self.norm_dec_2b(function.relu(self.dec_2b(x))))

        # Decode
        x = self.up_1(x)
        
        x = torch.cat([x, x1], dim=1)
        x = self.norm_dec_1a(function.relu(self.dec_1a(x)))
        x = self.dropout(self.norm_dec_1b(function.relu(self.dec_1b(x))))

        # Out
        x = self.out(x)
        x = self.dequant(x)

        return x