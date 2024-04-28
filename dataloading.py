train_dataset = tio.SubjectsDataset(subjects[:105], transform = train_transform)
val_dataset = tio.SubjectsDataset(subjects[105:] , transform = val_transform)

sampler = tio.data.LabelSampler(patch_size = 96 , label_name = "Label" , label_probabilities = {0:0.2 , 1:0.3 , 2:0.6})


train_patches_queue = tio.Queue(
    train_dataset,
    max_length = 40,
    samples_per_volume = 5,
    sampler = sampler,
    num_workers = 2,
    shuffle_patches=True

)

val_patches_queue = tio.Queue(
    val_dataset,
    max_length = 40,
    samples_per_volume = 5,
    sampler = sampler ,
    num_workers = 2

)

train_loader = torch.utils.data.DataLoader(train_patches_queue , batch_size =4 , num_workers = 0)
val_loader = torch.utils.data.DataLoader(val_patches_queue, batch_size= 4 , num_workers = 0)