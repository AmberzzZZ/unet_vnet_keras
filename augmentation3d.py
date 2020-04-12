import numpy as np
from nilearn.image import new_img_like, resample_to_img


def augment_data(data, truth, affine, scale_deviation=None, flip=True):
    n_dim = len(data.shape)
    if scale_deviation:
        scale_factor = random_scale_factor(n_dim, std=scale_deviation)
    else:
        scale_factor = None
    if flip:
        flip_axis = random_flip_dimensions(n_dim)
    else:
        flip_axis = None
    data_list = []
    for i in range(data.shape[0]):
        image = get_image(data[i], affine)
        data_list.append(resample_to_img(distort_image(image, flip_axis=flip_axis, scale_factor=scale_factor),
                                         image, interpolation="continuous").get_data())
    data = np.asarray(data_list)
    truth_image = get_image(truth, affine)
    truth_data = resample_to_img(distort_image(truth_image, flip_axis=flip_axis, scale_factor=scale_factor),
                                 truth_image, interpolation="nearest").get_data()
    return data, truth_data


def get_image(data, affine, nib_class=nib.Nifti1Image):
    return nib_class(dataobj=data, affine=affine)


def random_scale_factor(n_dim=3, mean=1, std=0.25):
    return np.random.normal(mean, std, n_dim)


def random_flip_dimensions(n_dimensions):
    axis = []
    for dim in range(n_dimensions):
        if np.random.choice([True, False]):
            axis.append(dim)
    return axis


def distort_image(image, flip_axis=None, scale_factor=None):
    if flip_axis:
        image = flip_image(image, flip_axis)
    if scale_factor is not None:
        image = scale_image(image, scale_factor)
    return image


def flip_image(image, axis):
    new_data = np.copy(image)
    for axis_index in axis:
        new_data = np.flip(new_data, axis=axis_index)
    return new_data


def scale_image(image, scale_factor):
    scale_factor = np.asarray(scale_factor)
    new_affine = np.copy(image.affine)
    new_affine[:3, :3] = image.affine[:3, :3] * scale_factor
    new_affine[:, 3][:3] = image.affine[:, 3][:3] + (image.shape * np.diag(image.affine)[:3] * (1 - scale_factor)) / 2
    return new_img_like(image, data=image.get_data(), affine=new_affine)



