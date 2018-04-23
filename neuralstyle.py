from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
from PIL import Image
import argparse
import os
from functools import reduce

import vgg

IMAGE_DIM = 500

def create_parser():
    parser = argparse.ArgumentParser(description='Neural Style')
    
    parser.add_argument('style', help='Image used for style', type=str)
    parser.add_argument('content', help='Image used for content', type=str)
    parser.add_argument('-o', '--output', help='Output folder name or file, defaults to output.jpg', type=str,
                        default='output.jpg', action='store', dest='output')
    parser.add_argument('-i', '--iterations', help='Number of iterations optimizer iterates, defaults to 2,000',
                        type=int, default=2000, action='store', dest='iterations')
    parser.add_argument('-d', '--dimension', help='Dimension of output image, as a square image', type=int, default=500,
                        action='store', dest='dim')
    parser.add_argument('-wc', '--weight-content', help='Weight of the content loss function for each layer, default 1.0e-3',
                        type=float, default=0.001, action='store', dest='weight_c')
    parser.add_argument('-ws', '--weight-style', help='Weight of the style loss functions for each layer, default 2.0e5',
                        type=float, default=0.2e6, action='store', dest='weight_s')
    parser.add_argument('-wt', '--weight-total', help='Weight of total variation loss function, default 1.0e-8',
                        type=float, default=0.1e-7, action='store', dest='weight_total')
    parser.add_argument('--noise_weight', help='Use noise weights for layers instead of normalized weights', 
                        action='store_true', default=False, dest='noise_weight')
    parser.add_argument('-lr', '--learning-rate', help='Learning rate for Adam Optimizer function', type=int, default=2,
                        action='store', dest='learning_rate')
    parser.add_argument('--verbose', help='Increase output verbosity', action='store_true', default=False)

    return parser


def main(argv):
    if argv.verbose:
        print(
        '''
        Style Image Name:       {}
        Content Image Name:     {}
        Output Name:            {}
        Number of Iterations:   {}
        Image Dimension:        {}
        Content Loss Weight:    {:.4e}
        Style Loss Weight:      {:.4e}
        Total Variation Weight: {:.4e}
        Learning Rate:          {}
        Use Noise Weight:       {}
        '''.format(argv.style, argv.content, argv.output, argv.iterations, argv.dim, argv.weight_c,
         argv.weight_s, argv.weight_total, argv.learning_rate, argv.noise_weight)
        )

    global IMAGE_DIM
    IMAGE_DIM = argv.dim
    STYLE_LAYERS = ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1']
    CONTENT_LAYERS = ['relu4_2', 'relu5_2']
    weights = None

    if not argv.noise_weight:
        weights = np.load('vgg_normalized.npy')

    input_t = tf.placeholder(shape=(1, IMAGE_DIM, IMAGE_DIM, 3), dtype=tf.float32)
    style = vgg.prep_image(argv.style, IMAGE_DIM)
    content = vgg.prep_image(argv.content, IMAGE_DIM)
    placeholder = np.random.normal(size=(1, IMAGE_DIM, IMAGE_DIM, 3), scale=np.std(content)*0.1)

    g = tf.Graph()
    with g.device('/cpu:0'), tf.Session() as sess:
        cnn = vgg.build_cnn(IMAGE_DIM, input_image=input_t, weights=weights)
        layers = {i: cnn[i] for i in CONTENT_LAYERS}
        # Features of the content image
        content_features = {k: v.eval(feed_dict={input_t: content}, session=sess)
                        for k, v in layers.items()}

    g = tf.Graph()
    with g.device('/cpu:0'), tf.Session() as sess:
        cnn = vgg.build_cnn(IMAGE_DIM, input_image=input_t, weights=weights)
        layers = {i: cnn[i] for i in STYLE_LAYERS}
        # Features of the style image
        tmp_features = {k: v.eval(feed_dict={input_t: style}, session=sess)
                            for k, v in layers.items()}
        tmp_features = {k: np.reshape(v, (-1, v.shape[3])) for k, v, in tmp_features.items()}
        style_features = {k: np.matmul(v.T, v) / v.size for k, v in tmp_features.items()}


    image = tf.Variable(placeholder, dtype=tf.float32)
    cnn = vgg.build_cnn(IMAGE_DIM, input_image=image, weights=weights)
    layers = {i: cnn[i] for i in (CONTENT_LAYERS + STYLE_LAYERS)}

    # This one will be fed the noise image
    gen_features = {k: v for k, v in layers.items()}

    losses = []

    for i in range(len(CONTENT_LAYERS)):
        losses.append(argv.weight_c * vgg.content_loss(content_features, gen_features, CONTENT_LAYERS[i]))

    for i in range(len(STYLE_LAYERS)):
        losses.append(argv.weight_s * vgg.style_loss(style_features, gen_features, STYLE_LAYERS[i]))

    total_loss = reduce(tf.add, losses)
    total_loss = tf.add(total_loss, (argv.weight_total * vgg.total_variation_loss(image, IMAGE_DIM)))

    optimizer = tf.train.AdamOptimizer(argv.learning_rate).minimize(total_loss)

    lowest_loss = float('inf')
    output_image = None
    g = tf.Graph()
    with g.device('/gpu:0'), tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(argv.iterations):
            optimizer.run()
            print("{}/{}: {}".format(i+1, argv.iterations, total_loss.eval()))
            this_loss = total_loss.eval()
            if this_loss < lowest_loss:
                lowest_loss = this_loss
                output_image = image.eval()

        final = vgg.deprocess(output_image)
        if argv.output.endswith('.jpg') or argv.output.endswith('.bmp') or argv.output.endswith('.png'):
            Image.fromarray(final).save(argv.output)
        else:
            path = None
            if not os.path.exists(argv.output):
                os.mkdir(argv.output)

            if argv.output.endswith('/'):
                print("File name not specified; saving file as output.jpg")
                path = argv.output + 'output.jpg'
            else:
                print("File name not specified; saving file as output.jpg")
                path = argv.output + '/' + 'output.jpg'
            Image.fromarray(path).save()

if __name__ == '__main__':
    parser = create_parser()
    
    argv = parser.parse_args()
    
    main(argv)