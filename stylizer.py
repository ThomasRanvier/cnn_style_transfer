import tensorflow as tf
import numpy as np
import time
from vgg19 import *
from utils import *
from losses import *

verbose = True
print_iterations = 50
original_colors = False
opt = 'adam'
max_iterations = 700
learning_rate = 1e0
style_mask = False
out_dir = './outputs/'

#Weights for the loss functions
content_weight = 1.0#5e0
style_weight = 5000.0#1e4
tv_weight = 0.1#1e-3

def render_image(content_name, style_names):
    content_img = get_content_image(content_name)
    style_imgs = get_style_images(content_img, style_names)
    with tf.Graph().as_default():
        print('\n---- RENDERING SINGLE IMAGE ----\n')
        init_img = get_init_image(content_img, style_imgs)
        tick = time.time()
        stylize(content_img, style_imgs, init_img)
        tock = time.time()
        print('Single image elapsed time: {}'.format(tock - tick))

def stylize(content_img, style_imgs, init_img, frame=None):
    write_image(out_dir + 'init', init_img)
    write_images(out_dir + 'style', style_imgs)
    write_image(out_dir + 'content', content_img)
    with tf.Session() as sess:
        # setup network
        net = build_model(content_img)
        
        # style loss
        if style_mask:
            L_style = sum_masked_style_losses(sess, net, style_imgs)
        else:
            L_style = sum_style_losses(sess, net, style_imgs)
        
        # content loss
        L_content = sum_content_losses(sess, net, content_img)
        
        # denoising loss
        L_tv = tf.image.total_variation(net['input'])
        
        # loss weights
        alpha = content_weight
        beta  = style_weight
        theta = tv_weight
        
        # total loss
        L_total  = alpha * L_content
        L_total += beta  * L_style
        L_total += theta * L_tv
        
        # optimization algorithm
        optimizer = get_optimizer(L_total)

        if opt == 'adam':
            minimize_with_adam(sess, net, optimizer, init_img, L_total)
        elif opt == 'lbfgs':
            minimize_with_lbfgs(sess, net, optimizer, init_img)
        
        output_img = sess.run(net['input'])
        
        if original_colors:
            output_img = convert_to_original_colors(np.copy(content_img), output_img)

        write_image(out_dir + 'output_final', output_img)

def get_optimizer(loss):
    if opt == 'lbfgs':
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(loss, method='L-BFGS-B', options={'maxiter': max_iterations, 'disp': print_iterations})
    elif opt == 'adam':
        optimizer = tf.train.AdamOptimizer(learning_rate)
    return optimizer

#Minimizers

def minimize_with_lbfgs(sess, net, optimizer, init_img):
    if verbose: print('\nMINIMIZING LOSS USING: L-BFGS OPTIMIZER')
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    optimizer.minimize(sess)

def minimize_with_adam(sess, net, optimizer, init_img, loss):
    if verbose: print('\nMINIMIZING LOSS USING: ADAM OPTIMIZER')
    train_op = optimizer.minimize(loss)
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    sess.run(net['input'].assign(init_img))
    iterations = 0
    while (iterations < max_iterations):
        sess.run(train_op)
        if iterations % print_iterations == 0:
            if verbose:
                curr_loss = loss.eval()
                print("At iterate {}\tf=  {}".format(iterations, curr_loss))
            output_img = sess.run(net['input'])
            if original_colors:
                output_img = convert_to_original_colors(np.copy(content_img), output_img)
            write_image(out_dir + 'output_ite_' + str(iterations), output_img)
        iterations += 1
