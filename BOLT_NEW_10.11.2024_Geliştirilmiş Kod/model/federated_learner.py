import tensorflow as tf
import tensorflow_federated as tff
import numpy as np
from typing import List, Dict

class FederatedLearner:
    def __init__(self, model_fn, num_clients=10):
        self.model_fn = model_fn
        self.num_clients = num_clients
        self.client_data = {}
        
    def preprocess_data(self, dataset):
        def batch_format_fn(element):
            return collections.OrderedDict(
                x=tf.reshape(element['x'], [-1, 28, 28, 1]),
                y=tf.reshape(element['y'], [-1, 1])
            )
        return dataset.repeat(NUM_EPOCHS).shuffle(SHUFFLE_BUFFER).batch(
            BATCH_SIZE).map(batch_format_fn)
    
    def create_federated_data(self, data, labels):
        # Split data among clients
        split_size = len(data) // self.num_clients
        for i in range(self.num_clients):
            start_idx = i * split_size
            end_idx = (i + 1) * split_size
            client_data = tf.data.Dataset.from_tensor_slices({
                'x': data[start_idx:end_idx],
                'y': labels[start_idx:end_idx]
            })
            self.client_data[i] = self.preprocess_data(client_data)
    
    def build_federated_averaging_process(self):
        # Create sample client data spec
        sample_data = next(iter(self.client_data[0]))
        def client_computation(model, dataset):
            # Client training loop
            num_examples = tf.constant(0, dtype=tf.int32)
            loss_sum = tf.constant(0.0, dtype=tf.float32)
            
            for batch in dataset:
                with tf.GradientTape() as tape:
                    outputs = model(batch['x'])
                    loss = tf.keras.losses.SparseCategoricalCrossentropy()(
                        batch['y'], outputs)
                
                grads = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))
                
                batch_size = tf.shape(batch['x'])[0]
                num_examples += batch_size
                loss_sum += loss * tf.cast(batch_size, tf.float32)
            
            return model.trainable_variables, num_examples, loss_sum
        
        # Build federated computation
        self.iterative_process = tff.learning.build_federated_averaging_process(
            self.model_fn,
            client_optimizer_fn=lambda: tf.keras.optimizers.SGD(0.1)
        )
    
    def train_federated_model(self, num_rounds=10):
        state = self.iterative_process.initialize()
        
        for round_num in range(num_rounds):
            # Select random subset of clients
            selected_clients = np.random.choice(
                self.num_clients,
                size=max(1, self.num_clients // 2),
                replace=False
            )
            client_datasets = [self.client_data[i] for i in selected_clients]
            
            # Perform federated training round
            state, metrics = self.iterative_process.next(state, client_datasets)
            
            print(f'Round {round_num}:')
            print(f'Loss: {metrics["loss"]:.4f}')
    
    def evaluate_global_model(self, test_data, test_labels):
        # Convert test data to TF dataset
        test_dataset = tf.data.Dataset.from_tensor_slices({
            'x': test_data,
            'y': test_labels
        }).batch(32)
        
        # Get global model from federated state
        global_model = self.model_fn()
        global_model.set_weights(self.state.model)
        
        # Evaluate
        results = global_model.evaluate(test_dataset)
        return dict(zip(global_model.metrics_names, results))