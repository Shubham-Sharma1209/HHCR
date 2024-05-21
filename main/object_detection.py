from imutils import paths
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import ImageFont, ImageDraw, Image
import os
from .char_predictor_single_img import predict_images
def rescale_img(img):
        img_shape=img.shape
        if abs((img.shape[0]-img_shape[0])/img_shape[0])<abs((img.shape[1]-img_shape[1])/img_shape[1]):
            print(0)
            scale=img_shape[0]/img.shape[0]
            new_shape=[int(img.shape[1]*scale),int(img.shape[0]*scale)]
        else:
            print(1)
            scale=img_shape[1]/img.shape[1]
            new_shape=[int(img.shape[1]*scale),int(img.shape[0]*scale)]

        img=cv.resize(img,new_shape)
        blank=np.ones(img_shape,'uint8')*255
        print(img.shape,blank.shape)
        if img.shape[0]==img_shape[0]:
            differ=img_shape[1]-img.shape[1]
            blank[:,differ//2:-differ//2,:]=img
        else:
            differ=img_shape[0]-img.shape[0]
            blank[differ//2:-differ//2,:,:]=img
        return blank

def segment_and_predict():

    dir_path="D:\Programming\Django\\hhcr\\uploads\\"
    path=list(paths.list_images(dir_path))
    if len(path)==0:
        return FileNotFoundError
    path=path[0]
    img=cv.imread(path)
    characters=pd.read_csv("D:\Programming\Django\hhcr\characters.csv",index_col="index")

    fontpath="D:\Programming\Django\hhcr\MartelSans-Light.ttf"
    font = ImageFont.truetype(fontpath, 32)




    # feat_file = open("character.feat",'w')
    # print(img.shape)
    def convert_to_grayscale(img):
        median=cv.medianBlur(img, 3, 15)
        gray=cv.cvtColor(median,cv.COLOR_BGR2GRAY)
        return gray
    def get_adaptive_thresh(gray):
        adaptive_thresh=cv.adaptiveThreshold(gray,255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY,15, 11)
        return adaptive_thresh
    gray=convert_to_grayscale(img)
    adaptive_thresh=get_adaptive_thresh(gray)

    def erode(img):
        ero_shape=cv.MORPH_RECT
        ero_size=2
        element=cv.getStructuringElement(ero_shape,(2*ero_size+1,2*ero_size+1))
        return cv.erode(img,element)
    eroded_img=erode(adaptive_thresh)

    def get_otsu(img):
        blur = cv.GaussianBlur(img,(5,5),0)
        grayscale=cv.cvtColor(blur,cv.COLOR_BGR2GRAY)
        ret2,th2 = cv.threshold(grayscale,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
        cv.imshow('Otsu',th2)
        return th2


    # cv.imshow('eroded',eroded_img)
    contours, hierarchy = cv.findContours(eroded_img, cv.RETR_CCOMP,cv.CHAIN_APPROX_SIMPLE)

    print(f'{len(contours)} Contours found!')
    cont_img=cv.drawContours(eroded_img,contours,-1,0,2)
    # cv.imshow('contours image',cont_img)

    img_with_contours=gray.copy()
    word_contours=[]
    words=[]


    def com_area(rect1,rect2):
        widthoverlap = min(rect1[2],rect2[2]) >= max(rect1[0],rect2[0]) 
        heightoverlap = min(rect1[3],rect2[3]) >= max( rect1[1],rect2[1])
        return (widthoverlap or ( (max(rect1[0],rect2[0]) - min(rect1[2],rect2[2]) ) < 0) ) and heightoverlap


    def merge_rect(rect1,rect2):
        return (min(rect1[0],rect2[0]),min(rect1[1],rect2[1]),max(rect1[2],rect2[2]),max(rect1[3],rect2[3]))

    def area(word):
        return (word[2]-word[0])*(word[3]-word[1])

    def add_padding(img):
        return cv.copyMakeBorder(img,2,2,2,2,cv.BORDER_REPLICATE,value=(255,255,255))


    possible_words=[]

    for c,h in zip(contours,hierarchy[0]):
        x,y,w,h=cv.boundingRect(c)
        x,y,x_r,y_b=x,y,x+w,y+h
        if max((x_r-x)/(y_b-y),(y_b-y)/(x_r-x))>10  or cv.contourArea(c)<100  or cv.contourArea(c)>(img.shape[0]*img.shape[1])/80:
            continue
        img_with_contours = cv.rectangle(img_with_contours,(x,y),(x_r,y_b),(0,0,255),2)
        possible_words.append((x,y,x_r,y_b))

    # cv.imshow('All contours',img_with_contours)
    while True:
        check=0
        for i,word1 in enumerate(possible_words):
            for word2 in possible_words[i+1:]:
                if com_area(word1,word2):
                    check=1
                    possible_words.remove(word2)
                    word1 = merge_rect(word1,word2)
                    possible_words[i]=word1
            if area(word1)>8000:
                possible_words.remove(word1)
                check=1
        if check==0:
            break


    possible_words=list(set(possible_words))


    word_contours=possible_words
    contour_img=img.copy()
    k=len(word_contours)
    def see_words():
        words=img.copy()
        for word in word_contours:
            x,y,x_r,y_b=word
            cv.rectangle(words,(x,y),(x_r,y_b),(0,0,255),1,0)
        cv.imshow('words',words)
    # see_words()
    # print(word_contours)
    # cv.imshow('gray',gray)
    out_img=img.copy()
    img_pil = Image.fromarray(out_img)
    draw = ImageDraw.Draw(img_pil)

    def char_splitter(out_img):
        k=0
        for x,y,x_r,y_b in word_contours:
            word=img[y:y_b,x:x_r]
            word=get_adaptive_thresh(convert_to_grayscale(word))
            # deskew(word)
            if 0 in word.shape:
                continue
            word=255-word
            horizontal_histogram=word.sum(axis=1)
            height,width=word.shape
            horizontal_histogram//=255    
            # print(word.shape,horizontal_histogram.shape)
            upper_end=-1
            no_upper=False
            cv.imwrite(f'./words/{x}-{y}-{x_r}-{y_b}.png',word)
            for loc, v in enumerate(horizontal_histogram):
                # v=v//255
                if v>=width*0.85:
                    # print(k,1)
                    upper_end=loc+1
                    break
                if loc<height-2:
                    # print(    k,2)
                    # if (horizontal_histogram[loc-1]+horizontal_histogram[loc-2]+v)//3>width*1.5:
                    if (horizontal_histogram[loc+1]+horizontal_histogram[loc+2]+v)>width:
                        upper_end=loc+1
                        break
                if loc>=height*0.3:
                    # print(k,3)
                    upper_end=loc
                    no_upper=True
                    break
            if no_upper or loc<height*0.15:
                lower_start=height*0.8
            else:
                lower_start=height*0.7

            # word=word[int(upper_end)+1:int(lower_start),:]
            # word=word[int(upper_end)+1:,:]
            # y+=int(upper_end)+1
            vertical_histogram=word.sum(axis=0)
            vertical_histogram//=255
            start=0
            cheight,width=word.shape
            n=1

            # cv.imwrite(f'./without_upper/{x}-{y}-{w}-{h}-{loc}.png',word)
            word=255-word
            for bel,v in enumerate(vertical_histogram):
                # print(start,v,bel)
                if v<=1:
                    # if v<5 and bel>0:
                    #     if abs(vertical_histogram[bel]-v)<=1:
                    #         continue
                    # if v>=5:
                    #     continue
                    if (bel-start)>=cheight*0.7:
                        # print(bel)
                        if word[:,start:bel].shape[0]*word[:,start:bel].shape[1]>300:
                            # cv.imwrite(f'./characters/{n}_{x}_{y}_{x_r}_{y_b}.png',img[y:y_b,x+start:x+bel+2])
                            # cv.imwrite(f'./characters/{x+start}_{y}_{x+bel+2}_{y_b}.png',img[y:y_b,x+start:x+bel+2])
                            # try:
                                # print(x+start,y,x_r+bel+5,y)
                            result=predict_images(add_padding(img[y-1:y_b+1,x+start-2:x+bel+2]))
                            out_img=cv.rectangle(out_img,(x+start,y),(x+bel,y_b),(0,0,255),1)
                            cv.putText(out_img,characters.iloc[int(result),2] , (x+start,y-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.6, (212,20  ,12), 1)

                            # except:
                            #     pass
                            n+=1
                            start=bel+1
                # else:
                #     start+=1
                
        
            # if bel-start<=5:
            #     continue
            # cv.imwrite(f'./characters/{n}_{x}_{y}_{w}_{h}.png',word[:,start:bel+2])
            # cv.imshow('word-{k}',word)
            k+=1
            # print(k)   

    char_splitter(out_img)
    # for x,y,x_r,y_b in word_contours:
    #     # result=predict_images((img[y:y_b,x:x_r]))
    #     result=predict_images(add_padding(img[y:y_b,x:x_r]))
    #     out_img=cv.rectangle(out_img,(x,y),(x_r,y_b),(0,0,255),1)
    #     cv.putText(out_img,characters.iloc[int(result),2] , (x,y-10 ), cv.FONT_HERSHEY_SIMPLEX, 0.6, (26,4,135), 1)
    #     draw.rectangle((x,y,x_r,y_b),(0,0,255))
    #     draw.text((x,y-10),characters.iloc[int(result),1],font=font,fill=(12,255,212,0))
    # out_img=np.array(img_pil)



    # from char_predictor import *

    # path='./characters'
    # # path='characters'
    # imagePaths=(list(paths.list_files(path)))
    # characters=pd.read_csv("characters.csv",index_col="index")

    # predict_images(imagePaths)
    # for i,imagePath in enumerate(imagePaths):
    #     print(imagePath)
    #     char=imagePath.split('\\')[-1].split('-')[0]
    #     loc=imagePath.split('\\')[-1].split('-')[1]
    #     print(loc)
    #     loc=loc.split('.')[0]
    #     print(loc)
    #     loc=list(map(lambda x:int(x),loc.split('_')))
    #     print(loc)
    #     out_img=cv.rectangle(out_img,(loc[0],loc[1]),(loc[2],loc[3]),(255,0,0),1)
        # img=cv.imread(imagePath)
        # if img is None:
        #     continue
        # if 0 in img.shape:
        #     continue
        # img=cv.cvtColor(img,cv.COLOR_BGR2GRAY)



    # cv.imshow("Output image",out_img)
    # plt.show()
    cv.imwrite('out.jpg',out_img)
    os.remove(path)
    # cv.waitKey(0)