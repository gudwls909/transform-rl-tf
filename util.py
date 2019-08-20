import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageFilter, ImageEnhance

'''
np2pil input img shape: [28,28,1]
pil3np output img shpae: [28,28,1]
'''
np2pil = lambda img: Image.fromarray((img.squeeze(2)*255).astype(np.uint8))
pil2np = lambda img: np.expand_dims((np.array(img) / 255.), axis=2)
pil_rotate = lambda img, angle: img.rotate(angle)
pil_blur = lambda img, radius: img.filter(ImageFilter.GaussianBlur(radius))
pil_sharpen = lambda img, radius: img.filter(ImageFilter.UnsharpMask(radius))
pil_affine = lambda img, theta: img.transform(img.size, Image.AFFINE,
                                              theta, resample=Image.BICUBIC)


def theta2mtx(theta):
    """
    Args:
        theta(np.array): 6 parameters for affine transformation, size=(6,)
    Returns:
        affine_mtx(np.array): size=(3,3)
    """
    affine_mtx = np.eye(3)
    affine_mtx[:2, :] = theta.reshape([2, 3])
    return affine_mtx


def get_affine_theta(method, param=None, a_bound=None):
    """
    Args:
        method: one in ['translation','rotate','shear','scale']
        param:
        a_bound(list): action boundary
    Returns:
        theta(np.array): 6 parameters for affine transformation, size=(6,)
    """

    if method == 'translation':
        sign = np.random.choice([-1, 1], 2)
        a = -6 + sign * np.random.uniform(a_bound[0], a_bound[1], 2) \
            if param is None else param

        theta = np.array((1, 0, a[0],
                          0, 1, a[1]))

    elif method == 'rotate':
        sign = np.random.choice([-1, 1])
        a = sign * np.random.uniform(a_bound[0], a_bound[1])\
            if param is None else param
        a = np.radians(a)
        theta = np.array((np.cos(a), -np.sin(a), 0,
                          np.sin(a), np.cos(a), 0))

    elif method == 'shear':
        sign = np.random.choice([-1, 1], 2)
        a = sign * np.random.uniform(a_bound[0], a_bound[1], 2) \
            if param is None else param
        theta = np.array((1, a[0], 0,
                          a[1], 1, 0))

    elif method == 'scale':
        sign = np.random.choice([-1, 1], 2)
        a = 1 + sign * np.random.uniform(a_bound[0], a_bound[1], 2) \
            if param is None else param
        theta = np.array((a[0], 0, 0,
                          0, a[1], 0))

    else:
        raise Exception("`method` should be one of the ['translation','rotate','shear','scale']")

    return theta


def random_affine_image(img, env_type, r_bound=[20, 50], sh_bound=[0.2, 0.5], sc_bound=[0.1, 0.2], t_bound=[3, 6]):
    """
    Args:
        img(np.array): HWC format
    Returns:
        img(np.array): HWC format
    """
    # translation : move center of the image to (0,0)
    t1_mtx = theta2mtx(get_affine_theta('translation', param=[img.shape[1]/2, img.shape[0]/2]))

    # rotate, shear, scale
    r_mtx = theta2mtx(get_affine_theta('rotate', a_bound=r_bound))
    sh_mtx = theta2mtx(get_affine_theta('shear', a_bound=sh_bound))
    sc_mtx = theta2mtx(get_affine_theta('scale', a_bound=sc_bound))

    # translation : move back (0,0) to be the left-upper corner of the image
    t2_mtx = theta2mtx(get_affine_theta('translation', param=[-img.shape[1]/2, -img.shape[0]/2]))

    # translation : move mnist in (40,40) size black image
    t3_mtx = theta2mtx(get_affine_theta('translation', a_bound=t_bound))

    # integrated affine transformation
    if env_type == 'r':
        affine_mtx = t1_mtx @ r_mtx @ t2_mtx
    elif env_type == 'rsc':
        affine_mtx = t1_mtx @ r_mtx @ sc_mtx @ t2_mtx
    elif env_type == 'rsh':
        affine_mtx = t1_mtx @ r_mtx @ sh_mtx @ t2_mtx
    elif env_type == 'rss':
        affine_mtx = t1_mtx @ r_mtx @ sh_mtx @ sc_mtx @ t2_mtx
    elif env_type == 'rsst':
        affine_mtx = t1_mtx @ r_mtx @ sh_mtx @ sc_mtx @ t2_mtx @ t3_mtx
    else:
        raise TypeError('env type error')

    # transform image
    aff_theta = affine_mtx[:2, :].flatten()
    pil_img = np2pil(img)
    if env_type == 'rsst':
        pil_img = pil_img.transform((40,40), Image.AFFINE, aff_theta, resample=Image.BICUBIC)
    else:
        pil_img = pil_img.transform((28,28), Image.AFFINE, aff_theta, resample=Image.BICUBIC)
    img = pil2np(pil_img)

    return img


def param2theta(param, env):
    """
    Args:
        param(np.array): [r, sh1, sh2, sc1, sc2].shape = (5,)
    """
    img_size = 40 if env == 'rsst' else 28

    # translation : move center of the image to (0,0)
    t1_mtx = theta2mtx(get_affine_theta('translation', param=[img_size/2, img_size/2]))

    # rotate, shear, scale, translate
    t_mtx = theta2mtx(get_affine_theta('translation', param=param[5:])) if env == 'rsst' else np.eye(3)
    if env in ['rss','rsst']:
        sc_mtx = theta2mtx(get_affine_theta('scale', param=param[3:5]))
    elif env == 'rsc':
        sc_mtx = theta2mtx(get_affine_theta('scale', param=param[1:3]))
    else:
        sc_mtx = np.eye(3)
    sh_mtx = theta2mtx(get_affine_theta('shear', param=param[1:3])) if env in ['rsh','rss','rsst'] else np.eye(3)
    r_mtx = theta2mtx(get_affine_theta('rotate', param=param[0]))

    # translation : move back (0,0) to be the left-upper corner of the image
    t2_mtx = theta2mtx(get_affine_theta('translation', param=[-img_size/2, -img_size/2]))

    # integrated affine transformation
    affine_mtx = t1_mtx @ t_mtx @ sc_mtx @ sh_mtx @ r_mtx @ t2_mtx
    theta = affine_mtx[:2, :].flatten()

    return theta


def theta2affine_img(img, theta, resize=None):
    """
    Args:
        img(np.array): HWC format
        theta(np.array): 6 parameters for affine transformation, size=(6,)
    Returns:
        img(np.array): HWC format with size (40,40,1) if resize=None
    """
    pil_img = np2pil(img)
    pil_img = pil_img.transform(pil_img.size, Image.AFFINE, theta, resample=Image.BICUBIC)
    if resize is not None:
        pil_img = pil_img.resize(resize, resample=Image.BICUBIC)
    img = pil2np(pil_img)
    return img


def integrate_thetas(thetas):
    mtxs = [theta2mtx(theta) for theta in thetas]
    int_mtx = theta2mtx(np.array((1, 0, 0, 0, 1, 0)))
    for mtx in mtxs:
        int_mtx = int_mtx @ mtx
    int_theta = int_mtx[:2, :].flatten()
    return int_theta


# def np_rotate(img, angle):
#     img = np2pil(img)
#     img = pil_rotate(img, angle)
#     img = pil2np(img)
#     return img
#     
# 
# def np_sharpen(img, radius):
#     img = np2pil(img)
#     img = pil_sharpen(img, radius)
#     img = pil2np(img)
#     return img
# 
# 
# def random_degrade(img, angle=None, radius=None):
#     '''
#     img: 28*28*1 with each value ranged in [0,1]
#     '''
#     img = np2pil(img)
#     
#     if angle==None:
#         angle = np.random.randint(-80,80)
#     img = pil_rotate(img, angle)
# 
#     if radius==None:
#         radius = np.random.uniform(0.1,0.5)
#     img = pil_blur(img, radius)
#     
#     img = pil2np(img)
#     return img


def make_grid(batch, nrow=8, padding=2):
    '''
    batch: size = [batch_size, 28, 28 ,1]
    nrow: Number of images displayed in each row of the grid
    '''
    batch = batch.squeeze(axis=3)
    ds = batch.shape[1] # data_size
    ncol = np.ceil(batch.shape[0]/nrow).astype(np.int)
    grid = np.ones([(ds+padding)*ncol-padding, (ds+padding)*nrow-padding])

    for i in range(batch.shape[0]):
        row_idx, col_idx = i%nrow, i//nrow
        grid[col_idx*(padding+ds):col_idx*(padding+ds) + ds,
             row_idx*(padding+ds):row_idx*(padding+ds) + ds] = batch[i]

    return grid


def save_batch_fig(fname, batch_grid, img_width, tick_labels):
    '''
    batch_grid: output of `make_grid` function
    img_width: width of original image
    tick_labels: x_axis tick labels
    '''
    xticks_position = np.arange(img_width//2, batch_grid.shape[1], img_width+2)

    fig = plt.figure(figsize=(12,3))
    plt.imshow(batch_grid, cmap='gray')
    plt.xticks(xticks_position, tick_labels)
    plt.yticks([])
    for direction in ['bottom','left','top','right']:
        plt.gca().spines[direction].set_visible(False)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close(fig)


# ================================================================ #
# ================================================================ #

def all_prob(model, batch, mc):
    """
    get all the probabilities of the images repectively, given multiple network
    return size = [# of network, # of images, # of class]
    """
    prob_set = np.zeros([mc, len(batch), 10])
    for i in range(mc):
        prob_set[i] = model.test(batch)
    return prob_set


def get_predictive_entropies(prob_set):
    """
    return shape: (batch_size)
    """
    expected_prob = prob_set.mean(axis=0)
    minus_p_logp = -1 * expected_prob * np.log(expected_prob + 1e-15)
    predictive_entropies = minus_p_logp.sum(axis=1)
    return predictive_entropies


def get_mutual_informations(prob_set):
    """
    return shape: (batch_size)
    """
    minus_p_logp = -1. * prob_set * np.log(prob_set + 1e-15)
    entropies = minus_p_logp.sum(axis=2)
    expected_entropies = entropies.mean(axis=0)

    predictive_entropies = get_predictive_entropies(prob_set)

    mutual_informations = predictive_entropies - expected_entropies
    mutual_informations = np.clip(mutual_informations, a_min=0., a_max=None)
    return mutual_informations



