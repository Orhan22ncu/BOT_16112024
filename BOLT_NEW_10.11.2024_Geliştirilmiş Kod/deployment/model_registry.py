import mlflow
from datetime import datetime
import json
import os

class ModelRegistry:
    def __init__(self, registry_uri):
        mlflow.set_tracking_uri(registry_uri)
        self.client = mlflow.tracking.MlflowClient()
    
    def register_model(self, model, metrics, tags=None):
        with mlflow.start_run():
            # Log metrics
            for key, value in metrics.items():
                mlflow.log_metric(key, value)
            
            # Log model
            mlflow.tensorflow.log_model(model, "model")
            
            # Log tags
            if tags:
                for key, value in tags.items():
                    mlflow.set_tag(key, value)
            
            # Register model
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/model"
            mv = mlflow.register_model(model_uri, "trading_model")
            
            return mv.version
    
    def load_model(self, version="latest"):
        if version == "latest":
            model_version = self.client.get_latest_versions("trading_model")[0]
        else:
            model_version = self.client.get_model_version("trading_model", version)
        
        return mlflow.tensorflow.load_model(model_version.source)
    
    def get_model_metrics(self, version):
        run_id = self.client.get_model_version(
            "trading_model",
            version
        ).run_id
        return self.client.get_run(run_id).data.metrics