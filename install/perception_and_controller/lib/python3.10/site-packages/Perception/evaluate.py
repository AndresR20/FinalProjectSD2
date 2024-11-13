def evaluate(model,image):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader
    from torch.utils.data import Dataset
    import cv2
    import numpy as np
    import os
    import PIL
    import matplotlib.pyplot as plt
    import random
    import math
    import tqdm

    from Perception.model.unet import UNet
    from Perception.utils.utils import find_edge_channel
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"

    trained_model = model#"./epoch_39.pt"

    #print(image.shape)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges,edges_inv = find_edge_channel(image)
    output_image = np.zeros((gray.shape[0],gray.shape[1],3),dtype=np.uint8)
    output_image[:,:,0] = gray
    output_image[:,:,1] = edges
    output_image[:,:,2] = edges_inv
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.Resize((180,330)),
                                    transforms.ToTensor()])
    test_img = transform(output_image).unsqueeze(0).to(device)
    output = model(test_img) #trained model
    pred = torch.sigmoid(output) #numbers between 0 and 1, treating model predictions as probabilities
    # pred = torch.where(pred > 0.5 , 255, 0).type(torch.uint8)
    pred = torch.where(pred > 0.5 , 1, 0) # if pred>0.5, 1 is returned. else 0
    pred = pred.detach().cpu().squeeze().numpy() # squeeze returns a tensor with all specified dimensions of input of size 1 removed


    pred = cv2.resize(pred.astype(float) , (1080, 720),cv2.INTER_AREA) #pred.astype is source input image array
    #print(pred.shape)

    #image[pred.astype(bool),0] = 155
    #image[pred.astype(bool),1:] = 0 


    #print(image.shape)
    #print(pred.shape)



    #cv2.imshow("asd",image)
    #cv2.imshow("pred",pred)
    #cv2.waitKey()

    return pred
