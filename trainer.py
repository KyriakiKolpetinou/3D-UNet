trainer = pl.Trainer(accelerator ="auto" , logger = TensorBoardLogger(save_dir="/content/logs"), log_every_n_steps=1,
                     callbacks=[checkpoint_callback],max_epochs = 100)

trainer.fit(model , train_loader , val_loader)
