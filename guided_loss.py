from torch import torch


def image_basic_loss(images, target_image):
    """
    Given a target image, return a loss for how far away on average
    the images' pixels are from that image.
    """
    error = torch.abs(images - target_image).mean()
    return error


# dict of all loss for guided sampling, compare img to img sample
# key: loss name
# value: loss function
loss_dict = {
    "BASE": image_basic_loss,
    "MSE": torch.nn.MSELoss(),
    "L1": torch.nn.L1Loss(),
    "CROSS_ENTROPY": torch.nn.CrossEntropyLoss(),
}
