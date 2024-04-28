class Segmenter(pl.LightningModule):
  def __init__(self):
    super().__init__()

    self.model = UNet()
    self.optimizer = torch.optim.Adam(self.model.parameters(), lr = 1e-4)
    self.loss_fn = torch.nn.CrossEntropyLoss()

    self.training_step_outputs = []
    self.validation_step_outputs = []

  def forward(self,data):
    return self.model(data)

  def log_images(self, images, masks, preds, global_step, tag):
    colors = [(0, 0, 0), (1, 1, 0), (1, 0, 0)]  # RGB for background, liver, and tumor
    cmap = ListedColormap(colors)
    my_cmap = ListedColormap(['black', 'yellow', 'red'])  # Background: black, Liver: yellow, Tumor: red

    for sample_idx in range(2):
      slice_idx = random.randint(0, images.shape[-1] - 1)  # depth is the last dimension

    # Extract the specified slice from each volume in the batch
      image_slice = images[sample_idx, ..., slice_idx].cpu().numpy()
      mask_slice = masks[sample_idx, ..., slice_idx].cpu().numpy()
      pred_slice = preds[sample_idx].argmax(dim=0)[...,slice_idx].cpu().numpy()


      image_slice = image_slice.squeeze()  # Squeezes all singleton dimensions
      mask_slice = mask_slice.squeeze()
      pred_slice = pred_slice.squeeze()

    # Normalize the image slice for better visualization
      image_slice_min = image_slice.min()
      image_slice_max = image_slice.max()
      image_normalized = (image_slice - image_slice_min) / (image_slice_max - image_slice_min)

    # Convert grayscale image slice to RGB
      image_rgb = np.stack([image_normalized]*3, axis=-1)

    # Prepare the overlay for ground truth mask and prediction
      gt_overlay = my_cmap(mask_slice)  # Convert class integers to RGBA colors
      pred_overlay = my_cmap(pred_slice)  # Convert class integers to RGBA colors

    # Prepare figure
      fig, axs = plt.subplots(1, 3, figsize=(15, 5))

      axs[0].imshow(image_rgb)  # Image RGB
      axs[0].set_title('Image')
      axs[0].axis('off')

      axs[1].imshow(image_rgb)  # Overlay ground truth
      axs[1].imshow(gt_overlay, alpha=0.5)  # gt_overlay is RGBA
      axs[1].set_title('Ground Truth')
      axs[1].axis('off')

      axs[2].imshow(image_rgb)  # Overlay prediction
      axs[2].imshow(pred_overlay, alpha=0.5)  #pred_overlay is RGBA
      axs[2].set_title('Prediction')
      axs[2].axis('off')

       # Determine the directory based on the phase, e.g., 'Train' or 'Val'
      fig_dir = os.path.join(save_dir, tag, 'Images')  #tag for 'Train' or 'Val'

        # Ensure the directory exists
      os.makedirs(fig_dir, exist_ok=True)

        # Filename now includes only sample and step information
      fig_filename = f"{tag}_Images_sample_{sample_idx}_step_{global_step}.png"

        # Save the figure to the appropriate directory
      fig.savefig(os.path.join(fig_dir, fig_filename))

    # Convert the matplotlib figure to an image tensor for TensorBoard logging
      fig.canvas.draw()
      fig_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
      fig_image = fig_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
      fig_image_tensor = torch.from_numpy(fig_image.copy()).permute(2, 0, 1).float() / 255  # Added .copy() here
      self.logger.experiment.add_image(f"{tag}/compare_sample_{sample_idx}", fig_image_tensor, global_step)

      plt.close(fig)  # Close the figure to free memory



    # Create the grid of images
    #grid = make_grid(torch.cat([image_slice.unsqueeze(0),
                                #mask_slice.unsqueeze(0).unsqueeze(0),
                               # pred_slice.unsqueeze(0).unsqueeze(0)], dim=0))
   # self.logger.experiment.add_image(tag + "/grid", grid, global_step)

  def training_step(self, batch, batch_idx):
    img = batch["CT"]["data"]
    mask = batch["Label"]["data"][:,0]
    mask = mask.long()

    pred = self(img)
    loss= self.loss_fn(pred, mask)
    self.training_step_outputs.append(loss)  # Store loss for the epoch

    dice_loss_value = dice_loss(pred, mask)
    self.log("Train Dice Loss", dice_loss_value)

    if batch_idx % 14 == 0:
      self.log_images(img, mask, pred, self.global_step, "Train")

    self.log("Train Loss", loss)
    return loss

  def validation_step(self, batch, batch_idx):
    img = batch["CT"]["data"]
    mask = batch["Label"]["data"][:,0]
    mask = mask.long()

    pred = self(img)
    loss= self.loss_fn(pred, mask)
    self.validation_step_outputs.append(loss)
    self.log("Val Loss", loss)

    dice_loss_value = dice_loss(pred, mask)
    self.log("Val Dice Loss", dice_loss_value)


    if batch_idx % 5 == 0:
      self.log_images(img, mask, pred, self.global_step, "Val")

    return loss

  def on_train_epoch_end(self):
        # Calculate average training loss for the entire epoch
        avg_train_loss = torch.stack(self.training_step_outputs).mean()
        # Log the average training loss for the epoch
        self.log("Epoch Train Loss", avg_train_loss)
        self.training_step_outputs.clear()

  def on_validation_epoch_end(self):
        epoch_val_average = torch.stack(self.validation_step_outputs).mean()
        self.log("validation_epoch_average", epoch_val_average)
        self.validation_step_outputs.clear()  # free memory

  def configure_optimizers(self):
    scheduler = {
        'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10, verbose=True),
        'monitor': 'Val Loss',  # Ensure this is the metric you want to monitor
        'interval': 'epoch',
        'frequency': 1,
        'strict': True,
    }
    return [self.optimizer], [scheduler]