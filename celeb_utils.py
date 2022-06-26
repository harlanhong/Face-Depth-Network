import glob
import numpy as np
import os
import pdb
import random
def produce_txt_list():
    videos = glob.glob('/data/fhongac/origDataset/Voxceleb2/vox2_train_frames/mp4/*/*/*')
    train_num = 0
    test_num = 0
    trainfile = open('./celeb2/train_files.txt', 'w', encoding='UTF-8')
    train = True
    testfile = open('./celeb2/test_file.txt', 'w', encoding='UTF-8')
    for v in videos:
        print(v)
        if train:
            imgs = os.listdir(v)
            num = len(imgs)
            if num<=2:
                continue
            imglist = range(2,num)
            choice = random.sample(imglist,int(num/3))
            for ch in choice:
                string = v+' '+str(ch)+'\n'
                trainfile.write(string)
                train_num+=1
        else:
            imgs = os.listdir(v)
            num = len(imgs)
            if num<=2:
                continue
            imglist = range(2,num)
            choice = random.sample(imglist,int(num/3))
            for ch in choice:
                string = v+' '+str(ch)+'\n'
                testfile.write(string)
                test_num+=1
            if test_num>1000:
                testfile.close()
                exit(0) 
    trainfile.close()

def produce_txt_list_celeb1():
    train_num = 0
    test_num = 0
    train = True
    if train: 
        videos = glob.glob('/data/fhongac/origDataset/vox1_frames/train/*')
        trainfile = open('./celeb1/train_files.txt', 'w', encoding='UTF-8')
    else:
        videos = glob.glob('/data/fhongac/origDataset/vox1_frames/test/*')
        testfile = open('./celeb1/val_files.txt', 'w', encoding='UTF-8')
    for v in videos:
        print(v)
        if train:
            imgs = os.listdir(v)
            num = len(imgs)
            if num<=2:
                continue
            imglist = range(2,num-1)
            choice = random.sample(imglist,int(num/3))
            for ch in choice:
                string = v+' '+str(ch)+'\n'
                trainfile.write(string)
                train_num+=1
        else:
            imgs = os.listdir(v)
            num = len(imgs)
            if num<=2:
                continue
            imglist = range(2,num-1)
            choice = random.sample(imglist,int(num/3))
            for ch in choice:
                string = v+' '+str(ch)+'\n'
                testfile.write(string)
                test_num+=1
            if test_num>1000:
                testfile.close()
                exit(0) 
    trainfile.close()

if __name__ == '__main__':
    produce_txt_list_celeb1()
