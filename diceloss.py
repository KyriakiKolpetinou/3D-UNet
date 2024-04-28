def dice_loss(pred, target, num_classes=3, smooth=1.):
    pred = F.softmax(pred, dim=1)
    target = F.one_hot(target, num_classes=num_classes)
    target = target.permute(0, 4, 1, 2, 3)

    intersection = (pred * target).sum(dim=(2, 3, 4))
    union = pred.sum(dim=(2, 3, 4)) + target.sum(dim=(2, 3, 4))
    dice = (2. * intersection + smooth) / (union + smooth)
    return (1 - dice).mean()