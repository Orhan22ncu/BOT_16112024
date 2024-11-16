import tensorflow as tf
from fastapi import FastAPI, BackgroundTasks
from pydantic import BaseModel
import numpy as np
import json
import redis
import asyncio

class ModelServer:
    def __init__(self, model_path, cache_config=None):
        self.model = tf.saved_model.load(model_path)
        self.app = FastAPI()
        self.setup_routes()
        self.cache = redis.Redis(**cache_config) if cache_config else None
        
    def setup_routes(self):
        @self.app.post("/predict")
        async def predict(data: dict, background_tasks: BackgroundTasks):
            try:
                # Check cache
                cache_key = self._generate_cache_key(data)
                if self.cache:
                    cached_result = self.cache.get(cache_key)
                    if cached_result:
                        return json.loads(cached_result)
                
                # Make prediction
                features = self._prepare_features(data)
                prediction = await self._async_predict(features)
                
                # Cache result
                if self.cache:
                    background_tasks.add_task(
                        self._cache_prediction,
                        cache_key,
                        prediction
                    )
                
                return {"prediction": prediction.tolist()}
                
            except Exception as e:
                return {"error": str(e)}
    
    async def _async_predict(self, features):
        return await asyncio.get_event_loop().run_in_executor(
            None, self._predict, features
        )
    
    def _predict(self, features):
        return self.model(features).numpy()
    
    def _prepare_features(self, data):
        # Implement feature preparation
        return np.array(data["features"])
    
    def _generate_cache_key(self, data):
        return f"pred:{hash(str(data))}"
    
    def _cache_prediction(self, key, prediction, expire=300):
        if self.cache:
            self.cache.setex(
                key,
                expire,
                json.dumps(prediction.tolist())
            )
    
    def start(self, host="0.0.0.0", port=8000):
        import uvicorn
        uvicorn.run(self.app, host=host, port=port)