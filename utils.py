import numpy as np
import os
import cv2

#Utilities and pre-processing

def get_content_image(content_name):
    path = os.path.join('./contents/', content_name)
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    img = img.astype(np.float32)
    img = preprocess(img)
    return img

def get_style_images(content_img, style_names):
    _, ch, cw, cd = content_img.shape
    style_imgs = []
    for style_fn in style_names:
        path = os.path.join('./styles/', style_fn)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = img.astype(np.float32)
        img = cv2.resize(img, dsize=(cw, ch), interpolation=cv2.INTER_AREA)
        img = preprocess(img)
        style_imgs.append(img)
    return style_imgs

def write_image(path, img):
    img = postprocess(img)
    cv2.imwrite(path, img)

def preprocess(img):
    imgpre = np.copy(img)
    # bgr to rgb
    imgpre = imgpre[...,::-1]
    # shape (h, w, d) to (1, h, w, d)
    imgpre = imgpre[np.newaxis,:,:,:]
    imgpre -= np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    return imgpre

def postprocess(img):
    imgpost = np.copy(img)
    imgpost += np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))
    # shape (1, h, w, d) to (h, w, d)
    imgpost = imgpost[0]
    imgpost = np.clip(imgpost, 0, 255).astype('uint8')
    # rgb to bgr
    imgpost = imgpost[...,::-1]
    return imgpost
