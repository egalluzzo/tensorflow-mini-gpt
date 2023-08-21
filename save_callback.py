import tensorflow as tf
from tensorflow import keras
import os

class SaveCallback(keras.callbacks.Callback):
    """A callback to save the model every 2^n batches (and every N epochs).

    Arguments:
        model: The model to save
        checkpoints_dir: The directory in which to save batches and epochs
        save_early_batches: Boolean, save after 2^n batches (to demonstrate training progress)
        epoch_save_interval: Integer, save after this many epochs
    """

    def __init__(
            self,
            model: keras.Model,
            checkpoints_dir: str,
            save_early_batches=True,
            batch_save_interval=20,
            epoch_save_interval=1
    ):
        self.model = model
        self.checkpoints_dir = checkpoints_dir
        self.save_early_batches = save_early_batches
        self.batch_save_interval = batch_save_interval
        self.epoch_save_interval = epoch_save_interval
        self.hit_first_epoch = False


    def on_train_batch_end(self, batch, logs=None):
        if not self.save_early_batches:
            return
        
        if self.hit_first_epoch:
            return
        
        if (batch + 1) % self.batch_save_interval != 0:
            return
        
        self.model.save("%s/%s-batch-%d" % (self.checkpoints_dir, "ckpt", (batch + 1)))


    def on_epoch_end(self, epoch, logs=None):
        self.hit_first_epoch = True
        if (epoch + 1) % self.epoch_save_interval != 0:
            return
                
        self.model.save("%s/%s-epoch-%d" % (self.checkpoints_dir, "ckpt", (epoch + 1)))
