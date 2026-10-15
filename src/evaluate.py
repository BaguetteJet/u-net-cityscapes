import torch
from tqdm import tqdm
from cityscapesscripts.helpers.labels import labels

# Based on Cityscapes benchmark evaluation metrics
# Source: https://www.cityscapes-dataset.com/benchmarks/
def metrics_from_confusion_matrix(confusion_matrix, num_classes=19):
    confusion_matrix = confusion_matrix.double()
    true_positives = confusion_matrix.diag()
    ground_truth_pixels_per_class = confusion_matrix.sum(dim=1)
    predicted_pixels_per_class = confusion_matrix.sum(dim=0)

    false_positives = predicted_pixels_per_class - true_positives
    false_negatives = ground_truth_pixels_per_class - true_positives

    iou_denominator_per_class = (
        true_positives + false_positives + false_negatives
    )

    iou_per_class = torch.where(
        iou_denominator_per_class > 0,
        true_positives / iou_denominator_per_class.clamp(min=1),
        float("nan"),
    )

    accuracy_per_class = torch.where(
        ground_truth_pixels_per_class > 0,
        true_positives / ground_truth_pixels_per_class.clamp(min=1),
        float("nan"),
    )

    total_ground_truth_pixels = ground_truth_pixels_per_class.sum()

    pixel_accuracy = (
        true_positives.sum() / total_ground_truth_pixels
        if total_ground_truth_pixels > 0
        else torch.tensor(float("nan"), device=confusion_matrix.device)
    )

    class_names = [label.name
        for label in sorted(labels, key=lambda label: label.trainId)
        if 0 <= label.trainId < num_classes
    ]

    return {
        "pixel_accuracy": pixel_accuracy.item(),
        "mean_iou_class": torch.nanmean(iou_per_class).item(),
        "mean_class_accuracy": torch.nanmean(accuracy_per_class).item(),
        "class_names": class_names,
        "iou_per_class": iou_per_class.cpu().tolist(),
        "accuracy_per_class": accuracy_per_class.cpu().tolist(),
    }

@torch.no_grad()
def evaluate(model, loader, device, num_classes=19, ignore_index=255, criterion=None):
    model.eval()

    total_loss = 0.0

    # confusion matrix over the entire set
    # rows = ground-truth
    # cols = predicted
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device)

    for images, truth_masks in tqdm(loader, leave=False):
        # move images to device
        images = images.to(device, non_blocking=True)
        truth_masks = truth_masks.to(device, non_blocking=True).long()

        logits = model(images)

        # optional validation loss from the same forward pass
        if criterion is not None:
            total_loss += criterion(logits, truth_masks).item()

        # get predicted class for each pixel
        predicted_masks = logits.argmax(dim=1)

        # exclude ignored classes
        valid_pixels = (
            (truth_masks >= 0)
            & (truth_masks < num_classes)
            & (truth_masks != ignore_index)
        )

        # pair truth/prediction
        encoded_class_pairs = (num_classes * truth_masks[valid_pixels] + predicted_masks[valid_pixels])

        # add batch pixel counts to the confusion matrix
        confusion_matrix += torch.bincount(encoded_class_pairs, minlength=num_classes**2).reshape(num_classes, num_classes)

    # evaluation metrics based on confusion matrix
    metrics = metrics_from_confusion_matrix(confusion_matrix, num_classes)

    if criterion is not None:
        metrics["loss"] = total_loss / len(loader)

    return metrics
