# import functions
import sys
import os
import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt
import random
import threading

from yolact import Yolact
from data import cfg, set_cfg, set_dataset
from torch.autograd import Variable
from eval import image_inference
from eval import evalimage_v2 as visimg
from datetime import datetime

class Inference:
    def __init__(self, name):
        self.name = name

    def load_network(self, weights):
        with torch.no_grad():
            print('Loading model...', end='')
            set_cfg(self.name + "_config")
            cudnn.fastest = True
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
            net = Yolact()
            try:
                net.load_weights(weights)
                net.eval()
                print('done')
            except (FileNotFoundError):
                print("cannot load the weights... close the program")
                exit(1)
        return net


    def prepare_inference(self, net):
        with torch.no_grad():
            net = net.cuda()
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = True
            cfg.mask_proto_debug = False
        return net


    def load_image(self, imagepath):
        try:
            img_np = cv2.imread(imagepath)
        except (FileNotFoundError):
            print("cannot load the image... close the program")
            exit(1)
        return img_np


    def inference(self, net, imgnp, log_thr=0.1, conf_thr=0.3, max_dets_per_image=20):
        with torch.no_grad():
            try:
                classes, scores, boxes, masks, logimg = image_inference(net, imgnp, log_thr, conf_thr, max_dets_per_image)
            except:
                print("cannot execute the image inference... close the program")
                exit(1)
        return classes, scores, boxes, masks


    # do not use this function in the final code (it's not very fast) but use it purely to visualize the intermediate steps for debugging
    def visualize(self, img_np, masks):
        max_height = 1024
        max_width = 1024

        if img_np.shape[-1] < 3:
           img_np = cv2.cvtColor(img_np,cv2.COLOR_GRAY2RGB)

        if masks.any():
            maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
            emptymask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.float32)

            for i in range (maskstransposed.shape[-1]):
                mask = maskstransposed[:,:,i]*255
                emptymask = cv2.add(emptymask,np.expand_dims(mask,axis=2))

            alpha_channel = emptymask.astype(np.uint8) # creating a dummy alpha channel image.

            maskimg = cv2.cvtColor(np.expand_dims(alpha_channel,axis=2),cv2.COLOR_GRAY2RGB)
            maskimg[np.where((maskimg==[255, 255, 255]).all(axis=2))] = [0,0,255] # change white to red color

            height, width = img_np.shape[:2]
            img_mask = cv2.addWeighted(img_np,1,maskimg,0.5,0)

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

            cv2.imshow("RGB image with mask(s)", img_mask) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
            cv2.waitKey(1)
        else:
            height, width = img_np.shape[:2]
            
            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_np = cv2.resize(img_np, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

            cv2.imshow("RGB image", img_np) # Show image, run "export DISPLAY=:0" if it doesn't work in visual code
            cv2.waitKey(1)


    def processdepth(self, dimg, classes, masks):
        max_height = 700
        max_width = 700

        z3 = np.expand_dims(dimg, axis=2).astype(np.uint8)
        z3 = cv2.cvtColor(z3, cv2.COLOR_GRAY2RGB)

        distances = np.multiply(np.ones(classes.shape[0]), 65535).astype(np.uint16)

        if masks.any():
            maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
            emptymask = np.zeros((maskstransposed.shape[0],maskstransposed.shape[1],1),dtype=np.float32)

            for i in range (maskstransposed.shape[-1]):
                mask = maskstransposed[:,:,i]
                emptymask = cv2.add(emptymask,np.expand_dims(np.multiply(mask, 255),axis=2))
                curclass = classes[i]
                depthmask = np.multiply(dimg, mask)
                depthmask_filtered = depthmask[depthmask != 0]

                if depthmask_filtered.size > 1:
                    hist, bin_edges = np.histogram(depthmask_filtered, density=False) # create a histogram of 10 bins within the filtered tomato mask
                    hist_peak = np.argmax(hist) # get the depth-value with the highest number of bin_counts (peak)
                    lb = bin_edges[hist_peak]
                    ub = bin_edges[hist_peak+1]

                    depth_final = depthmask_filtered[np.where(np.logical_and(depthmask_filtered >= lb, depthmask_filtered <= ub))]

                # plt.hist(depthmask_filtered, 100)
                # plt.show()
                # print(curclass)
                distances[i] = np.average(depth_final)

            alpha_channel = emptymask.astype(np.uint8) # creating a dummy alpha channel image.

            maskimg = cv2.cvtColor(np.expand_dims(alpha_channel,axis=2),cv2.COLOR_GRAY2RGB)
            maskimg[np.where((maskimg==[255, 255, 255]).all(axis=2))] = [0,0,255] # change white to red color

            height, width = z3.shape[:2]
            img_mask = cv2.addWeighted(z3,1,maskimg,0.5,0)

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA) # resize image

        else:
            height, width = z3.shape[:2]
            img_mask = z3

            if max_height < height or max_width < width: # only shrink if img is bigger than required
                scaling_factor = max_height / float(height) # get scaling factor
                if max_width/float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                img_mask = cv2.resize(img_mask, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)

        return img_mask, distances

def analyse(rgbimg, rgbdimg, network, networkinstance, log_thr, conf_mask_thr):
    start = time.time()
    classes, scores, boxes, masks = network.inference(networkinstance, rgbimg, log_thr, conf_mask_thr)
    #self.yolact.visualize(self.lastrgb, masks)

    #self.rgbmask = visimg(self.net, self.lastrgb, self.conf_mask_thr)           

    end = time.time()
    print("Image inference time: {:.3f} s".format(end-start))
    print()
    
    if masks.any():
        maskstransposed = masks.transpose(1,2,0) # transform the mask in the same format as the input image array (h,w,num_dets)
        invmasks = 1-maskstransposed
        rgbimg[np.where((invmasks==1).all(axis=2))] = [0,0,0]
        rgbdimg[np.where((invmasks==1).all(axis=2))] = [0,0,0,0]
        
        return rgbimg, rgbdimg
    else:
        return None, None



if __name__ == "__main__":
	#startup
    if torch.cuda.is_available():
        print("CUDA AVAILABLE")
        # be aware YOLACT++ (specifically the DCN2 Deformable Convolutional Networks) can only work when there is CUDA
        yolact = Inference('yolact_plus_resnet50_jente')

        net = yolact.load_network("weights/yolact_plus_resnet50_jente_config_2348_110383_interrupt.pth")        
        net = yolact.prepare_inference(net)
        log_thr = 0.1
        conf_mask_thr = 0.3
    else:
        print("CUDA ERROR")
        quit()
        
	#iterate over images
    for dirpath, subdirs, files in os.walk("./Images"):
        #Read folder (from dirpath), find images
        file_names = os.listdir(dirpath)
        rgbimgs = [fn for fn in file_names if fn.endswith("_rgb.tiff")]
        for rgbimg in rgbimgs:
            rgbdimgname = rgbimg[:-5]+"d.tiff"
            rgbmaskimgname = rgbimg[:-5]+"_segmented.tiff"
            rgbdmaskimgname = rgbimg[:-5]+"d_segmented.tiff"
            
            rgbimg = cv2.imread(os.path.join(dirpath,rgbimg), -1)
            rgbdimg = cv2.imread(os.path.join(dirpath,rgbdimgname), -1)
            #analyse
            rgbmaskimg, rgbdmaskimg = analyse(rgbimg, rgbdimg, yolact, net, log_thr, conf_mask_thr)
            
            
            if not type(rgbmaskimg) == type(None):
                cv2.imwrite(os.path.join(dirpath,rgbmaskimgname), rgbmaskimg)
                cv2.imwrite(os.path.join(dirpath,rgbdmaskimgname), rgbdmaskimg)

