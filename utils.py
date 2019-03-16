import numpy as np
import os
import cv2

#Utilities and pre-processing
color_convert_type = 'yuv'
init_type = 'random'
noise_ratio = 1.0
seed = 1

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
    cv2.imwrite(path + 'png', img)

def write_images(path, imgs):
    for i in range(len(imgs)):
        write_image(path + '_' + str(i), imgs[i])

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
"""
def write_image_output(output_img, content_img, style_imgs, init_img):
    out_dir = './outputs/'
    img_path = os.path.join(out_dir, 'result.png')
    content_path = os.path.join(out_dir, 'content.png')
    init_path = os.path.join(out_dir, 'init.png')

    write_image(img_path, output_img)
    write_image(content_path, content_img)
    write_image(init_path, init_img)
    for i in range(len(style_imgs)):
        path = os.path.join(out_dir, 'style_' + str(i) + '.png')
        write_image(path, style_imgs[i])
"""
def convert_to_original_colors(content_img, stylized_img):
    content_img  = postprocess(content_img)
    stylized_img = postprocess(stylized_img)
    if color_convert_type == 'yuv':
        cvt_type = cv2.COLOR_BGR2YUV
        inv_cvt_type = cv2.COLOR_YUV2BGR
    elif color_convert_type == 'ycrcb':
        cvt_type = cv2.COLOR_BGR2YCR_CB
        inv_cvt_type = cv2.COLOR_YCR_CB2BGR
    elif color_convert_type == 'luv':
        cvt_type = cv2.COLOR_BGR2LUV
        inv_cvt_type = cv2.COLOR_LUV2BGR
    elif color_convert_type == 'lab':
        cvt_type = cv2.COLOR_BGR2LAB
        inv_cvt_type = cv2.COLOR_LAB2BGR
    content_cvt = cv2.cvtColor(content_img, cvt_type)
    stylized_cvt = cv2.cvtColor(stylized_img, cvt_type)
    c1, _, _ = cv2.split(stylized_cvt)
    _, c2, c3 = cv2.split(content_cvt)
    merged = cv2.merge((c1, c2, c3))
    dst = cv2.cvtColor(merged, inv_cvt_type).astype(np.float32)
    dst = preprocess(dst)
    return dst

def get_init_image(content_img, style_imgs):
    if init_type == 'content':
        return content_img
    elif init_type == 'style':
        return style_imgs[0]
    elif init_type == 'random':
        init_img = get_noise_image(noise_ratio, content_img)
        return init_img

def get_noise_image(noise_ratio, content_img):
    np.random.seed(seed)
    noise_img = np.random.uniform(-20., 20., content_img.shape).astype(np.float32)
    img = noise_ratio * noise_img + (1.-noise_ratio) * content_img
    return img
