from torch.optim.swa_utils import SWALR
from utils.general import *
from utils.losses import *
from utils.metrics import MetricsAccumulator
from utils.radam import *


def defrost_model(model):
    """
    Unfreeze model parameters
    :param model: Instance of model
    :return: (void)
    """
    for param in model.parameters():  # Defrost model
        param.requires_grad = True


def check_defrost(model, defrosted, current_epoch, defrost_epoch):
    """

    Args:
        model:
        defrosted:
        current_epoch:
        defrost_epoch:

    Returns:

    """
    if defrost_epoch != -1 and current_epoch >= defrost_epoch and not defrosted:
        print("\n---------- Unfreeze Model Weights ----------")
        defrost_model(model)
        defrosted = True
    return defrosted


def get_current_lr(optimizer):
    """
    Gives the current learning rate of optimizer
    :param optimizer: Optimizer instance
    :return: Learning rate of specified optimizer
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']


def get_optimizer(optmizer_type, model, lr=0.1):
    """
    Create an instance of optimizer
    :param optmizer_type: (string) Optimizer name
    :param model: Model that optimizer will use
    :param lr: Learning rate
    :return: Instance of specified optimizer
    """
    if optmizer_type == "sgd":
        optim = torch.optim.SGD(model.parameters(), lr=lr)
    elif optmizer_type == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr)
    elif optmizer_type == "over9000":
        optim = Over9000(filter(lambda p: p.requires_grad, model.parameters()))
    else:
        assert False, "No optimizer named: {}".format(optmizer_type)

    return optim


def get_scheduler(scheduler_name, optimizer, epochs=40, min_lr=0.002, max_lr=0.01, scheduler_steps=None):
    """
    Gives the specified learning rate scheduler
    :param scheduler_name: Scheduler name
    :param optimizer: Optimizer which is changed by the scheduler
    :param epochs: Total training epochs
    :param min_lr: Minimum learning rate for OneCycleLR Scheduler
    :param max_lr: Maximum learning rate for OneCycleLR Scheduler
    :param scheduler_steps: If scheduler steps is selected, which steps perform
    :return: Instance of learning rate scheduler
    """
    if scheduler_name == "steps":
        if scheduler_steps is None:
            assert False, "Please specify scheduler steps."
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=scheduler_steps, gamma=0.1)
    elif scheduler_name == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, cooldown=6, factor=0.1, patience=12)
    elif scheduler_name == "one_cycle_lr":
        return torch.optim.lr_scheduler.OneCycleLR(optimizer, total_steps=epochs, max_lr=max_lr)
    elif scheduler_name == "cyclic":
        return torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=min_lr, max_lr=max_lr, step_size_up=scheduler_steps)
    elif scheduler_name == "constant":
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[9999], gamma=1)
    elif scheduler_name == "swa":
        return SWALR(optimizer, swa_lr=max_lr, anneal_epochs=int(epochs * 0.15))
    else:
        assert False, "Unknown scheduler: {}".format(scheduler_name)


def scheduler_step(optimizer, scheduler, metric, args):
    """
    Perform a step of a scheduler
    :param optimizer: Optimizer used during training
    :param scheduler: Scheduler instance
    :param metric: Metric to minimize
    :param args: Training list of arguments with required fields
    :return: (void) Apply scheduler step
    """
    if args.swa_start != -1:
        optimizer.step()
    if args.scheduler == "steps":
        scheduler.step()
    elif args.scheduler == "plateau":
        scheduler.step(metric.mean_value(args.plateau_metric))
    elif args.scheduler == "one_cycle_lr":
        scheduler.step()
    elif args.scheduler == "constant":
        pass  # No modify learning rate


def get_criterion(criterion_type, weights_criterion='default'):
    """
    Gives a list of subcriterions and their corresponding weight
    :param criterion_type: Name of created criterion
    :param weights_criterion: (optional) Weight for each subcriterion
    :return:
        (list) Subcriterions
        (list) Weights for each criterion
    """

    if weights_criterion == "":
        assert False, "Please specify weights for criterion"
    else:
        weights_criterion = [float(i) for i in weights_criterion.split(',')]

    if criterion_type == "bce":
        criterion1 = nn.BCEWithLogitsLoss()
        criterion = [criterion1]
        multiclass = [False]
    elif criterion_type == "ce":
        criterion1 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1]
        multiclass = [True]
    elif criterion_type == "bce_dice":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion = [criterion1, criterion2, criterion3]
        multiclass = [False, False, False]
    elif criterion_type == "bce_dice_border":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, False]
    elif criterion_type == "bce_dice_ac":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = ActiveContourLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, False]
    elif criterion_type == "bce_dice_border_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion5 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4, criterion5]
        multiclass = [False, False, False, False, True]
    elif criterion_type == "bce_dice_border_haus_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = BCEDicePenalizeBorderLoss().cuda()
        criterion5 = HDDTBinaryLoss().cuda()
        criterion6 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4, criterion5, criterion6]
        multiclass = [False, False, False, False, False, True]
    elif criterion_type == "bce_dice_ce":
        criterion1 = nn.BCEWithLogitsLoss().cuda()
        criterion2 = SoftDiceLoss().cuda()
        criterion3 = SoftInvDiceLoss().cuda()
        criterion4 = nn.CrossEntropyLoss().cuda()
        criterion = [criterion1, criterion2, criterion3, criterion4]
        multiclass = [False, False, False, True]
    else:
        assert False, "Unknown criterion: {}".format(criterion_type)

    return criterion, weights_criterion, multiclass


def create_checkpoint(metrics, model, model_name, output_dir):
    """
    Iterar sobre las diferentes metricas y comprobar si la ultima medicion es mejor que las anteriores
    (debemos comprobar que average!=none - si no tendremos que calcular el average)
    Para cada metrica guardar un checkpint como abajo iterativamente
    Args:
        metrics:
        model:

    Returns:

    """
    dict_metrics = metrics.metrics
    for metric_key in dict_metrics:
        # check if last epoch mean metric value is the best
        if metrics.metrics_helpers[f"{metric_key}_is_best"]:
            torch.save(model.state_dict(), output_dir + f"/model_{model_name}_best_{metric_key}.pt")

    torch.save(model.state_dict(), output_dir + "/model_" + model_name + "_last.pt")


def finish_swa(optimizer, swa_model, train_loader, val_loader, args):
    if args.swa_start == -1:  # If swa was not used, do not perform nothing
        return

    optimizer.bn_update(train_loader, swa_model, device='cuda')

    """
    update_bn() assumes that each batch in the dataloader loader is either a tensors or a list of tensors 
    where the first element is the tensor that the network swa_model should be applied to. 
    If your dataloader has a different structure, you can update the batch normalization statistics of the swa_model
    by doing a forward pass with the swa_model on each element of the dataset.
    """
    for indx, batch in enumerate(train_loader):
        image = batch["image"].type(torch.float).cuda()
        _ = swa_model(image)

    swa_epochs = args.epochs - args.swa_start

    torch.save(
        swa_model.state_dict(),
        os.path.join(args.output_dir, f"model_{args.model_name}_{swa_epochs}swa_lr{args.swa_lr}.pt")
    )

    swa_metrics = MetricsAccumulator(
        args.problem_type, args.metrics, args.num_classes, include_background=False, average="mean"
    )

    swa_metrics = val_step(
        val_loader, swa_model, swa_metrics, generated_overlays=args.generated_overlays, overlays_path="results/overlays"
    )

    print("SWA validation metrics")
    swa_metrics.report_best()


def calculate_loss(y_true, y_pred, criterion, weights_criterion, multiclass_criterion, num_classes):
    """

    Args:
        y_true:
        y_pred:
        criterion:
        weights_criterion:
        multiclass_criterion:

    Returns:

    """
    loss = 0

    if num_classes == 1:  # Single class case
        for indx, crit in enumerate(criterion):
            loss += weights_criterion[indx] * crit(y_pred, y_true)

    else:  # Multiclass case

        # Case Multiclass criterions
        multiclass_indices = [i for i, x in enumerate(multiclass_criterion) if x]
        for indx in multiclass_indices:
            loss += weights_criterion[indx] * criterion[indx](y_pred, y_true)

        singleclass_indices = [i for i, x in enumerate(multiclass_criterion) if not x]
        # Single class criterions => calculate criterions per class
        for current_class in y_true.unique():
            tmp_loss, tmp_mask = 0, 1 - (y_true != current_class) * 1.0

            for indx in singleclass_indices:  # Acumulamos todos los losses para una clase
                tmp_loss += (weights_criterion[indx] * criterion[indx](y_pred[:, current_class, :, :], tmp_mask))

            # Average over the number of classes
            loss += (tmp_loss / len(y_true.unique()))

    return loss


def train_step(train_loader, model, criterion, weights_criterion, multiclass_criterion, optimizer, train_metrics):
    """

    Args:
        train_loader:
        model:
        criterion:
        weights_criterion:
        multiclass_criterion:
        optimizer:
        train_metrics:

    Returns:

    """
    model.train()
    for indx, batch in enumerate(train_loader):
        image, label = batch["image"].cuda(), batch["label"].cuda()
        optimizer.zero_grad()
        prob_preds = model(image)
        loss = calculate_loss(
            label, prob_preds, criterion, weights_criterion, multiclass_criterion,
            num_classes=train_loader.dataset.num_classes
        )
        loss.backward()
        optimizer.step()

        train_metrics.record(prob_preds, label)

    train_metrics.update()
    return train_metrics


def val_step(val_loader, model, val_metrics, generated_overlays=1, overlays_path=""):
    if generated_overlays != 1 and overlays_path != "":
        os.makedirs(overlays_path, exist_ok=True)

    model.eval()
    with torch.no_grad():
        for sample_indx, batch in enumerate(val_loader):
            image = batch["image"].cuda()
            prob_preds = model(image)
            original_masks = batch["original_mask"]

            img_id = batch["img_id"][0]

            if torch.is_tensor(batch["original_img"]):
                original_img = batch["original_img"].data.cpu().numpy().squeeze()
            else:  # numpy array
                original_img = np.squeeze(batch["original_img"], axis=0)

            #original_img = batch["original_img"].data.cpu().numpy().squeeze()
            val_metrics.record(prob_preds, original_masks, original_img, generated_overlays, overlays_path, img_id)

    val_metrics.update()
    return val_metrics
