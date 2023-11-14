from torch import torch, nn
from torch.nn.modules import loss


class ImageBasicLoss(loss._Loss):
    def __init__(self):
        super(ImageBasicLoss, self).__init__()

    @staticmethod
    def forward(images, target_image):
        """
        Given a target image, return a loss for how far away on average
        the images' pixels are from that image.
        """
        error = torch.abs(images - target_image).mean()
        return error


def get_all_losses(omit=None):
    """
    Generate a dictionary of all loss functions available in torch.nn.
    Args:
        omit (list): A list of loss function names to omit from the dictionary.
    Returns:
        dict: A dictionary where keys are loss function names and values are instances of the loss functions.
    """
    loss_dict = {}

    for name, obj in nn.__dict__.items():
        if isinstance(obj, type) and issubclass(obj, loss._Loss) and obj is not loss._Loss:
            if omit is None or name not in omit:
                loss_dict[name] = obj()

    return loss_dict


omit_list = ['NLLLoss2d']

loss_dict = get_all_losses(omit=omit_list)
loss_dict = {**loss_dict, **{'ImageBasicLoss': ImageBasicLoss()}}
