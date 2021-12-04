from .wide_resnet import wide_resnet20


def Net():
    return wide_resnet20(5, zero_init_residual=True)
