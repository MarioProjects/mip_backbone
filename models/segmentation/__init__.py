from .custom_pspnet import *
from .resnet import *
from .efficientnet import efficientnet_model_selector
from .small_segmentation_models import small_segmentation_model_selector


def model_selector_segmentation(model_name, num_classes=2, in_channels=3):
    """

    :param model_name:
    :param num_classes:
    :param in_channels:
    :return:
    """
    classification = "classification" in model_name

    if "resnet34" in model_name or "resnet18" in model_name:
        model = resnet_model_selector(model_name, num_classes, classification, in_channels)

    elif "efficientnet" in model_name:
        model = efficientnet_model_selector(model_name, num_classes, classification, in_channels)

    elif "unet" in model_name:
        model = small_segmentation_model_selector(model_name, num_classes)
    else:
        assert False, "Unknown model selected: {}".format(model_name)

    return model