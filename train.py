"""
This file needs to contain the main training loop. The training code should be encapsulated in a main() function to
avoid any global variables.
"""
import torch.ao.quantization
import torch.quantization.quantize_fx
from torchvision.datasets import Cityscapes
import torchvision.transforms.v2 as transforms
from argparse import ArgumentParser
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch
from matplotlib.colors import ListedColormap, BoundaryNorm
from model import SegNet, Unet, Efficiency_model
import os
import numpy as np
import matplotlib.pyplot as plt
import utils
import torch.nn.utils.prune as prune
#from thop import profile
#import time
#from torchsummary import summary
#import torchvision.models as models

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_arg_parser():
    parser = ArgumentParser()
    parser.add_argument("--data_path", type=str, default=".", help="Path to the data")
    """add more arguments here and change the default values to your needs in the run_container.sh file"""
    return parser

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):
        """
        m = torch.nn.Softmax(dim=1)
        prediction_soft = m(predictions)
        prediction_max = torch.nn.functional.gumbel_softmax(prediction_soft, tau=1, hard=True, dim=1)
        print(prediction_max.unique())
        print(np.shape(prediction_max))
        print(targets.unique())
        """
        # Ignore index 255
        mask = targets != 255
        targets = targets[mask]
        predictions = predictions.squeeze(1)
        predictions = predictions[mask]

        # Flatten label and prediction tensors
        predictions = predictions.view(-1)
        targets = targets/18
        targets = targets.view(-1)
        
        # Determine Dice loss
        intersection = (predictions * targets).sum()                       
        dice = (2.*intersection + self.smooth)/(predictions.sum() + targets.sum() + self.smooth)  
        return 1 - dice

def freeze_layers(model, layers_not_to_freeze):
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze the required layers
    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layers_not_to_freeze):
            param.requires_grad = True

def main(args):
    """define your model, trainingsloop optimitzer etc. here"""

    regular_transform = transforms.Compose([transforms.Resize((270, 270)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # define transform
    complete_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            #transforms.RandomVerticalFlip(p=0.25),
                                            transforms.RandomResizedCrop(size=(256,256), scale=(0.25, 0.75), ratio=(0.5, 1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_x = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # data loading
    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(args.data_path, split='train', mode='fine', target_type='semantic', transforms=regular_transform) #args.data_path
    validation_ratio = 0.1
    val_size = int(validation_ratio*len(dataset))
    train_size = len(dataset)-val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    batch_size = 25
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)#, num_worker=8)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)#, num_worker=8)

    # define model
    model = SegNet()#.cuda()
    #model.load_state_dict(torch.load("SegNet model"))

    #layers_not_to_freeze = ['dec_1', 'dec_2', 'dec_3', 'dec_4', 'dec_5']
    #freeze_layers(model, layers_not_to_freeze)

    # define optimizer and loss function (don't forget to ignore class index 255)
    #weights = torch.tensor([1.0, 1.0, 1.0, 1.5, 1.5, 2.0, 2.0, 1.5, 1.0, 1.5, 1.0, 1.5, 2.0, 1.0, 2.0, 1.5, 2.0, 2.0, 1.5])
    criterion = torch.nn.CrossEntropyLoss(ignore_index=255).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 0.9)

    # training/validation loop
    epochs = 25

    train_loss = []
    val_loss = []
    for i in range(epochs):
        train_loss_epoch = 0
        val_loss_epoch = 0
        for X, Y in train_loader:
            target = (Y*255).long().squeeze(1)
            target = utils.map_id_to_train_id(target).to(device)
            optimizer.zero_grad()
            predictions = model(X).to(device)
            loss = criterion(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss_epoch += loss
            print("Loss of batch at epoch " + str(i+1) + " : " + str(float(loss)))

        # Make a scheduler step at end of each epoch
        scheduler.step()

        # Determine validation loss
        for X, Y in val_loader:
            target = (Y*255).long().squeeze(1)
            target = utils.map_id_to_train_id(target).to(device)
            predictions = model(X).to(device)
            loss_val = criterion(predictions, target)
            val_loss_epoch += loss_val

        train_loss.append(float(train_loss_epoch/len(train_loader)))
        val_loss.append(float(val_loss_epoch/len(val_loader)))
        print("Average train loss of epoch " + str(i+1) + ": " + str(float(train_loss_epoch/len(train_loader))))
        print("Average validation loss of epoch " + str(i+1) + ": " + str(float(val_loss_epoch/len(val_loader))))

    # save model
    torch.save(model.state_dict(), 'SegNet meets Unet')

    # visualize training data
    plt.plot(range(1, epochs+1), train_loss, color='r', label='train loss')
    plt.plot(range(1, epochs+1), val_loss, color='b', label='validation loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss of neural network")
    plt.legend()
    plt.savefig('Train performance of Segnet meets Unet')

    pass

def postprocess_dice(prediction, shape):
    prediction = prediction*18
    prediction = prediction.int()
    return prediction

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)
    processed = prediction.cpu().detach().numpy()
    processed = processed.squeeze()
    return processed

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""
    regular_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                                            
    img = regular_transform(img)
    img = img.unsqueeze(0)
    return img

def visualize():
    model_SegNet = SegNet()
    model_SegNet.load_state_dict(torch.load("models\\Segnet meets Unet"))
    model_SegNet.eval()

    model_Unet = Efficiency_model()
    model_Unet.load_state_dict(torch.load("models\\quantized efficiency net +"))
    model_Unet.eval()

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Define your custom colormap for specific color for each number
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'gray', 'purple', 
              'orange', 'pink', 'olive', 'navy', 'teal', 'coral', 'lime', 'indigo', 'peru']
    custom_cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = BoundaryNorm(bounds, len(colors))

    # define transform
    regular_transform = transforms.Compose([transforms.Resize((270, 270)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    # define transform
    complete_transform = transforms.Compose([transforms.Resize((256, 256)),
                                            #transforms.RandomVerticalFlip(p=0.25),
                                            transforms.RandomResizedCrop(size=(256,256), scale=(0.25, 0.75), ratio=(0.5, 1.5)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transform_x = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(path_local, split='train', mode='fine', target_type='semantic', transforms=regular_transform)#, target_transform=complete_transform) #args.data_path

    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for X, Y in train_loader:
        prediction_Segnet = model_SegNet(X)
        processed_SegNet = postprocess(prediction_Segnet, shape=(256, 256))
        print("Unique classes in SegNet prediction: ", np.unique(processed_SegNet))

        prediction_Unet = model_Unet(X)
        processed_Unet = postprocess(prediction_Unet, shape=(256, 256))
        print("Unique classes in Unet prediction:   ", np.unique(processed_Unet))

        Y = (Y*255).squeeze(1)
        Y = utils.map_id_to_train_id(Y)
        Y = Y.cpu().detach().numpy().transpose(1, 2, 0)
        Y[Y == 255] = 0
        X = X.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
        X = X * std + mean

        fig, axs = plt.subplots(1, 4, figsize=(12, 6))  # 1 row, 2 columns
        axs[0].imshow(processed_SegNet, cmap=custom_cmap, norm=norm)
        axs[0].set_title('efficiency +')
        axs[1].imshow(processed_Unet, cmap=custom_cmap, norm=norm)
        axs[1].set_title('SegNet')
        axs[2].imshow(Y, cmap=custom_cmap, norm=norm)
        axs[2].set_title('Y')
        axs[3].imshow(X)
        axs[3].set_title('X')
        fig.suptitle('Segmentation')
        plt.savefig("Images\\segmented images of Unet and SegNet model.png")
        break

def visualize_report():
    model = Efficiency_model()
    model.load_state_dict(torch.load("models\\Efficiency net +"))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Define your custom colormap for specific color for each number
    colors = ['black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'orange', 'gray', 'purple', 
              'orange', 'pink', 'olive', 'navy', 'teal', 'coral', 'lime', 'indigo', 'peru']
    custom_cmap = ListedColormap(colors)
    bounds = np.arange(len(colors) + 1)
    norm = BoundaryNorm(bounds, len(colors))

    # define transform
    regular_transform = transforms.Compose([transforms.Resize((270, 270)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=mean, std=std)])
    
    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(path_local, split='train', mode='fine', target_type='semantic', transforms=regular_transform) #args.data_path

    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for X, Y in train_loader:
        prediction = model(X)
        processed = postprocess(prediction, shape=(256, 256))

        fig, axs = plt.subplots(1, 3, figsize=(12, 6))  # 1 row, 2 columns
        X = X.cpu().detach().numpy().squeeze(0).transpose(1, 2, 0)
        X = X * std + mean
        Y = (Y*255).squeeze(1)#.cpu().detach().numpy().transpose(1, 2, 0)
        Y = utils.map_id_to_train_id(Y)
        Y = Y.cpu().detach().numpy().transpose(1, 2, 0)
        Y[Y == 255] = 0

        axs[0].imshow(X)
        axs[0].set_title('X')
        axs[1].imshow(Y, cmap=custom_cmap, norm=norm)
        axs[1].set_title('Y')
        axs[2].imshow(processed, cmap=custom_cmap, norm=norm)
        axs[2].set_title('prediction')
        fig.suptitle('Report visualization')
        plt.savefig("Images\\Report visualization.png")
        break

def prune_model():
    model = Efficiency_model()
    model.load_state_dict(torch.load("models\\efficiency net +"))

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            prune.l1_unstructured(module, 'weight', 0.25)
            prune.remove(module, 'weight')

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(params)

    # save model
    torch.save(model.state_dict(), 'models\\pruned efficiency net +')    

    pass

def quantize_model():
    model = Efficiency_model()
    model.load_state_dict(torch.load("models\\Efficiency net +"))
    model.eval()

    # attach a global qconfig, which contains information about what kind
    # of observers to attach. Use 'x86' for server inference and 'qnnpack'
    # for mobile inference. Other quantization configurations such as selecting
    # symmetric or asymmetric quantization and MinMax or L2Norm calibration techniques
    # can be specified here.
    # Note: the old 'fbgemm' is still available but 'x86' is the recommended default
    # for server inference.
    # model_fp32.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
    model.qconfig = torch.ao.quantization.get_default_qconfig('x86')

    # Fuse the activations to preceding layers, where applicable.
    # This needs to be done manually depending on the model architecture.
    # Common fusions include `conv + relu` and `conv + batchnorm + relu`
    model_fused = torch.ao.quantization.fuse_modules(model, [['enc_1a', 'norm_enc_1a'], ['enc_1b', 'norm_enc_1b'],
                                                             ['enc_2a', 'norm_enc_2a'], ['enc_2b', 'norm_enc_2b'],
                                                             ['enc_3a', 'norm_enc_3a'], ['enc_3b', 'norm_enc_3b'],
                                                             ['conv_latent_a', 'norm_lat_a'], ['conv_latent_b', 'norm_lat_b'], 
                                                             ['dec_1a', 'norm_dec_1a'], ['dec_1b', 'norm_dec_1b'],
                                                             ['dec_2a', 'norm_dec_2a'], ['dec_2b', 'norm_dec_2b'],
                                                             ['enc_3a', 'norm_dec_3a'], ['enc_3b', 'norm_dec_3b'],])

    # Prepare the model for static quantization. This inserts observers in
    # the model that will observe activation tensors during calibration.
    model_prepared = torch.ao.quantization.prepare(model_fused, allow_list=['enc_1a', 'norm_enc_1a','enc_1b', 'norm_enc_1b',
                                                                           'enc_2a', 'norm_enc_2a', 'enc_2b', 'norm_enc_2b',
                                                                           'enc_3a', 'norm_enc_3a', 'enc_3b', 'norm_enc_3b',
                                                                           'conv_latent_a', 'norm_lat_a', 'conv_latent_b', 'norm_lat_b', 
                                                                           'dec_1a', 'norm_dec_1a', 'dec_1b', 'norm_dec_1b',
                                                                           'dec_2a', 'norm_dec_2a', 'dec_2b', 'norm_dec_2b',
                                                                           'enc_3a', 'norm_dec_3a', 'enc_3b', 'norm_dec_3b'])

    # calibrate the prepared model to determine quantization parameters for activations
    # in a real world setting, the calibration would be done with a representative dataset
        # define transform
    regular_transform = transforms.Compose([transforms.Resize((270, 270)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(path_local, split='train', mode='fine', target_type='semantic', transforms=regular_transform)#, target_transform=complete_transform) #args.data_path
    batch_size = 15
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    X, Y = train_loader[0]
    model_prepared(X)

    # Convert the observed model to a quantized model. This does several things:
    # quantizes the weights, computes and stores the scale and bias value to be
    # used with each activation tensor, and replaces key operators with quantized
    # implementations.
    model_quantized = torch.ao.quantization.convert(model_prepared)

    # run the model, relevant calculations will happen in int8
    res = model_quantized(X)

    torch.save(model_quantized.state_dict(), 'models\\quantized efficiency net +')  


def count_flops():
    model = Efficiency_model()
    model.load_state_dict(torch.load("models\\quantized efficiency net +"))


    # define transform
    regular_transform = transforms.Compose([transforms.Resize((270, 270)),
                                            transforms.ToTensor(),
                                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    path_local = "C:\\Users\\20192326\\Documents\\YEAR 1 AIES\\Neural networks for computer vision\\Assignment\\data"
    dataset = Cityscapes(path_local, split='train', mode='fine', target_type='semantic', transforms=regular_transform)#, target_transform=complete_transform) #args.data_path
    batch_size = 1
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    i = 0
    inference_time = 0
    gflops_per_second = 0
    for X, Y in train_loader:
        # Measure the inference time
        start_time = time.time()
        output = model(X)
        inference_time_instance = time.time() - start_time
        inference_time += inference_time_instance

        X = X.unsqueeze(0) 
        flops, params = profile(model=model, inputs=X)
        
        # Calculate GFLOPs per second
        gflops_per_second += (flops/1e9) / inference_time_instance
        i+=1
        if (i == 50):
            break
    
    
    print("Inference time: "+ str(inference_time/i))
    print("Amounf of flops in model: " + str(flops))
    print("Amount of GFLOPs: " + str(gflops_per_second/i))
    print("Amount of parameters: " + str(params))


if __name__ == "__main__":
    # Get the arguments
    parser = get_arg_parser()
    args = parser.parse_args()
    main(args)

    #visualize()
    #visualize_report()
    
    #count_flops()
    #prune_model()
    #quantize_model()

    #model = Model()
    #params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    #print(params)
    
