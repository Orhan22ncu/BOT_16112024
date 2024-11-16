import tensorflow as tf
import horovod.tensorflow as hvd
from tensorflow.keras.optimizers import Adam
import ray
from ray import tune

class DistributedTrainer:
    def __init__(self, model_builder, num_workers=4):
        hvd.init()
        self.model_builder = model_builder
        self.num_workers = num_workers
        
    def setup_distributed_training(self):
        # Horovod configuration
        compression = hvd.Compression.fp16
        optimizer = Adam(learning_rate=0.001 * hvd.size())
        optimizer = hvd.DistributedOptimizer(
            optimizer, compression=compression
        )
        
        return optimizer
    
    def train_distributed(self, train_dataset, validation_dataset, epochs=100):
        optimizer = self.setup_distributed_training()
        model = self.model_builder()
        
        # Shard the data between workers
        train_dataset = train_dataset.shard(hvd.size(), hvd.rank())
        validation_dataset = validation_dataset.shard(hvd.size(), hvd.rank())
        
        callbacks = [
            hvd.callbacks.BroadcastGlobalVariablesCallback(0),
            hvd.callbacks.MetricAverageCallback(),
        ]
        
        # Train the model
        model.compile(optimizer=optimizer, loss='huber_loss', metrics=['mae'])
        model.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return model
    
    def hyperparameter_search(self, config):
        ray.init()
        analysis = tune.run(
            self.train_distributed,
            config=config,
            num_samples=10,
            resources_per_trial={"cpu": 2, "gpu": 0.5}
        )
        return analysis.best_config