import tensorflow as tf
import numpy as np

def conv_layer(input, filters, bias):
    conv = tf.nn.conv2d(
            input=input,
            filter=filters,
            strides=(1,1,1,1),
            padding="SAME",
            )
    return tf.nn.bias_add(conv, bias)

def pool_layer(input):
    pool = tf.nn.avg_pool(
        value=input,
        ksize=(1, 2, 2, 1),
        strides=(1, 2, 2, 1),
        padding='SAME'
        )
    return pool

def build_cnn(dim, input_image, weights=None):
    """Creates VGG-19 CNN Network
    Input:
        dim:         Dimension of images, assumed to be square images
        input_image: Image to be fed into network
        weights:     Weights for layers, if None noise is used
    Output:
        returns hash of all the tensors in network
    """
    net = {}
    cnn_layers = (
    'conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',

    'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',

    'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
    'relu3_3', 'conv3_4', 'relu3_4', 'pool3',

    'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
    'relu4_3', 'conv4_4', 'relu4_4', 'pool4',

    'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
    'relu5_3', 'conv5_4', 'relu5_4', 'pool5'
    )
    channels = (3, 64, 128, 256, 512, 512)

    net['input'] = input_image
    current = net['input']
    current_weight = 0

    for i, name in enumerate(cnn_layers):
        layer = name[:4]

        if layer == 'conv':
            weight = None
            bias = None
            
            if weights is None:
                out_channels = channels[int(name[4])]
                in_channels = channels[int(name[4])-1] if int(name[6]) == 1 else out_channels
                weight = np.random.uniform(size=(3, 3, in_channels, out_channels))
            else:
                weight, bias = weights[current_weight]
                weight = np.transpose(weight, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current_weight += 1
            current = conv_layer(current, weight, bias)
        elif layer == 'relu':
            current = tf.nn.relu(current)
        elif layer == 'pool':
            current = pool_layer(current)
        net[name] = current

    return net

MEAN_VALUES = np.array([123.68, 116.779, 103.939])

def prep_image(image, dim):
    '''
    Scales image down and center crops it
    Normalizes it by using mean pixel values (123.68, 116.779, 103.939)
    '''
    from PIL import Image
    im = Image.open(image)

    if im.height < im.width:
        im1 = im.resize((dim, im.width*dim//im.height))
    else:
        im1 = im.resize((im.width*dim//im.height, dim))

    im1 = im1.crop((im1.width//2-dim//2, im1.height//2-dim//2, im1.width//2+dim//2, im1.height//2+dim//2))
    
    im1 = im1 - MEAN_VALUES
    

    return im1[np.newaxis]

def deprocess(im):
    '''
    Undoes the Normalization
    '''
    im = np.copy(im[0])
    im = im + MEAN_VALUES

    im = np.clip(im, 0, 255).astype('uint8')
    return im

def gram_matrix(mat):
    mat_shape = mat.shape.as_list()

    mat = tf.reshape(mat, (-1, mat_shape[3]))
    size = mat_shape[1] * mat_shape[2] * mat_shape[3]

    g = tf.matmul(tf.transpose(mat), mat) / size
    return g

def content_loss(P, X, layer):
    p = np.asarray(P[layer])
    x = X[layer]

    loss = tf.nn.l2_loss(x-p) / p.size
    return tf.cast(loss, tf.float32)

def style_loss(A, X, layer):
    a = np.asarray(A[layer])
    x = X[layer]

    G = gram_matrix(x)

    N = a.shape[0]
    M = a.shape[1]

    loss = 1. / (
        4 * N**2 * M**2) \
           * \
           2 * tf.nn.l2_loss(G - a)
    return tf.cast(loss, tf.float32)

def total_variation_loss(x, dim):
    N = np.prod(x[:, 1:, :, :].shape.as_list())
    M = np.prod(x[:, :, 1:, :].shape.as_list())

    return 2 * (
        (tf.nn.l2_loss(x[:, 1:, :, :] - x[:, :dim-1, :, :]) / N)
        +
        (tf.nn.l2_loss(x[:, :, 1:, :] - x[:, :, :dim-1, :]) / M)
        )