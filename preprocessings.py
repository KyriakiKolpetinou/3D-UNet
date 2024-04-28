process = tio.Compose([
    tio.ToCanonical(),
    tio.CropOrPad((368, 368, 400)) ,
    tio.RescaleIntensity((-1,1)),

])

augmentation = tio.RandomAffine(scales=(0.9,1.1),degrees = (-10, 10))
val_transform= process
train_transform = tio.Compose([process, augmentation])
