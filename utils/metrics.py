import os
from utils.general import dict2df, convert_multiclass_mask, reshape_masks, plot_save_pred
import numpy as np
import torch
import monai

AVAILABLE_METRICS = ("accuracy", "iou", "dice", "assd", "hausdorff")


class MetricsAccumulator:
    """
    Tendremos una lista de metricas que seran un dicccionario
    al hacer el print mostrara el promedio para cada metrica
    Ejemplo metrics 2 epochs con 2 clases:
       {"iou": [[0.8, 0.3], [0.9, 0.7]], "dice": [[0.76, 0.27], [0.88, 0.66]]}
    """

    def __init__(self, problem_type, metric_list, num_classes,
                 include_background=True, average="mean", mask_reshape_method="padd"):
        """

        Args:
            problem_type:
            metric_list:
            num_classes:
            include_background:
            average:
        """
        if average not in ["mean", "none"]:
            assert False, f"Unknown average method '{average}'"

        if problem_type not in ["segmentation", "classification"]:
            assert False, f"Unknown problem type: '{problem_type}', please specify a valid one!"
        self.problem_type = problem_type

        if metric_list is None or not isinstance(metric_list, list):
            assert False, "Please, you need to specify a metric [list]"

        diffmetrics = np.setdiff1d(metric_list, AVAILABLE_METRICS)
        if len(diffmetrics):
            assert False, f"'{diffmetrics}' metric(s) not implemented."

        self.include_background = include_background
        self.metric_list = metric_list
        self.num_classes = (num_classes-1) if num_classes > 1 and not include_background else num_classes
        self.metric_methods_args = {}
        self.metrics_helpers = {}
        self.metric_methods = self.__metrics_init()
        self.metrics = {metric_name: [] for metric_name in metric_list}
        self.is_updated = True
        self.average = average
        self.mask_reshape_method = mask_reshape_method

    def __metrics_init(self):
        metric_methods = []
        for metric_str in self.metric_list:
            if metric_str in ["accuracy"]:
                metric_methods.append()
                self.metrics_helpers["accuracy_best_method"] = "max"
                self.metrics_helpers["accuracy_best_value"] = -1
                assert False, f"Accuracy metric not implemented"
            elif metric_str in ["iou"]:
                self.metric_methods_args[metric_str] = {}
                metric_methods.append(jaccard_coef)
                self.metrics_helpers["iou_best_method"] = "max"
                self.metrics_helpers["iou_best_value"] = -1
            elif metric_str in ["dice"]:
                self.metric_methods_args[metric_str] = {}
                metric_methods.append(dice_coef)
                self.metrics_helpers["dice_best_method"] = "max"
                self.metrics_helpers["dice_best_value"] = -1
            elif metric_str in ["hausdorff"]:
                self.metric_methods_args[metric_str] = {"label_idx": 1}
                metric_methods.append(monai.metrics.compute_hausdorff_distance)
                self.metrics_helpers["hausdorff_best_method"] = "min"
                self.metrics_helpers["hausdorff_best_value"] = 10e8
            elif metric_str in ["assd"]:
                self.metric_methods_args[metric_str] = {"label_idx": 1}
                metric_methods.append(monai.metrics.compute_average_surface_distance)
                self.metrics_helpers["assd_best_method"] = "min"
                self.metrics_helpers["assd_best_value"] = 10e8
        return metric_methods

    def record(self, prediction, target, original_img=None, generated_overlays=-1, overlays_path="", img_id=[]):

        if self.is_updated:
            for key in self.metrics:
                self.metrics[key].append([[] for _ in range(self.num_classes)])
            self.is_updated = False

        if self.problem_type == "segmentation":
            """
            prediction and target should be (h, w) with class indices, not probabilities or one channel per class!
            """

            for pred_indx, single_pred in enumerate(prediction):

                if torch.is_tensor(target[pred_indx]):
                    original_mask = target[pred_indx].data.cpu().numpy().astype(np.uint8).squeeze()
                else:  # numpy array
                    original_mask = target[pred_indx].astype(np.uint8)

                # Calculate metrics resizing prediction to original mask shape
                if not self.include_background and self.num_classes == 1:  # Single class -> sigmoid
                    pred_mask = reshape_masks(
                        torch.sigmoid(single_pred).squeeze(0).data.cpu().numpy(),
                        original_mask.shape, self.mask_reshape_method
                    )
                    pred_mask = np.where(pred_mask > 0.5, 1, 0).astype(np.int32)
                else:
                    pred_mask = convert_multiclass_mask(single_pred.unsqueeze(0)).data.cpu().numpy()
                    pred_mask = reshape_masks(pred_mask.squeeze(0), original_mask.shape, self.mask_reshape_method)
                    pred_mask = pred_mask.astype(np.uint8)

                for current_class in np.unique(np.concatenate((original_mask, pred_mask))):

                    if current_class > self.num_classes:
                        assert False, f"Label index '{current_class}' greater than num classes '{self.num_classes}'. " \
                                      f"Please count background if include_background is True."

                    if not self.include_background and current_class == 0:
                        continue

                    y_true = np.where(original_mask == current_class, 1, 0).astype(np.int32)
                    y_pred = np.where(pred_mask == current_class, 1, 0).astype(np.int32)

                    for indx, metric in enumerate(self.metric_methods):
                        self.metrics[self.metric_list[indx]][-1][
                            current_class if self.include_background else (current_class - 1)] += [
                            metric(y_true, y_pred, **self.metric_methods_args[self.metric_list[indx]])]

                if generated_overlays > 0 and len(os.listdir(overlays_path)) < generated_overlays:
                    plot_save_pred(original_img[pred_indx], original_mask, pred_mask, overlays_path, img_id[pred_indx])

        elif self.problem_type == "classification":
            assert False, f"To be done record for classification"
        else:
            assert False, f"Not implemented record for '{self.problem_type}'"

    def update(self):
        """
        CALL THIS METHOD AFTER RECORD ALL SAMPLES / AFTER EACH EPOCH
        We have accumulated metrics along different samples/batches and want to average accross that same epoch samples:
        {'iou': [[[0.8, 0.6], [0.3, 0.5]]]} -> {'iou': [[0.7, 0.4]]}
        """
        for key in self.metrics:
            for i in range(len(self.metrics[key][-1])):
                self.metrics[key][-1][i] = np.mean(self.metrics[key][-1][i])

            if self.average == "mean" or self.average == "none":
                metric_value = np.mean(self.metrics[key][-1])
            min_classes = 2 if self.include_background else 1  # no sense if only one class to calculate average metric
            if self.average != "none" and len(self.metrics[key][-1]) > min_classes:
                self.metrics[key][-1].append(metric_value)

            if self.metrics_helpers[f"{key}_best_method"] == "max":
                if self.metrics_helpers[f"{key}_best_value"] < metric_value:
                    self.metrics_helpers[f"{key}_best_value"] = metric_value
                    self.metrics_helpers[f"{key}_is_best"] = True
                else:
                    self.metrics_helpers[f"{key}_is_best"] = False

            elif self.metrics_helpers[f"{key}_best_method"] == "min":
                if self.metrics_helpers[f"{key}_best_value"] > metric_value:
                    self.metrics_helpers[f"{key}_best_value"] = metric_value
                    self.metrics_helpers[f"{key}_is_best"] = True
                else:
                    self.metrics_helpers[f"{key}_is_best"] = False

            else:
                assert False, "What happened?!"

        self.is_updated = True

    def report_best(self):
        for key in self.metrics:
            print("\t- {}: {}".format(key, self.metrics_helpers[f"{key}_best_value"]))

    def mean_value(self, metric_name):
        if self.average != "none":
            return self.metrics[metric_name][-1][-1]
        return np.mean(self.metrics[metric_name][-1])

    def save_progress(self, output_dir, identifier=""):
        nested_metrics = {
            metric_name:
                [item for sublist in self.metrics[metric_name] for item in sublist]
            for metric_name in self.metrics
        }
        dict2df(nested_metrics, os.path.join(output_dir, f'{identifier}_progress.csv'))

    def __str__(self, precision=3):
        output_str = ""
        for metric_key in self.metric_list:
            output_str += ''.join(['{:{align}{width}.{prec}f} | '.format(
                itemc, align='^', width=9, prec=3) for itemc in self.metrics[metric_key][-1]]
            )
        return output_str


SMOOTH = 1e-10


def jaccard_coef(y_true, y_pred):
    """
    Size of the intersection divided by the size of the union of two label sets
    :param y_true: Numpy ground truth (correct) labels
    :param y_pred: Numpy predicted labels
    :return: Jaccard similarity coefficient
    """
    intersection = np.sum(y_true * y_pred, axis=None)
    union = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None) - intersection
    return float(intersection + SMOOTH) / float(union + SMOOTH)


def dice_coef(y_true, y_pred):
    """
    Computes the Dice coefficient, a measure of set similarity.
    :param y_true: Numpy ground truth (correct) labels
    :param y_pred: Numpy predicted labels
    :return: Dice similarity coefficient
    """
    intersection = np.sum(y_true * y_pred, axis=None)
    summation = np.sum(y_true, axis=None) + np.sum(y_pred, axis=None)
    return (2.0 * intersection + SMOOTH) / (summation + SMOOTH)
