import docker
import kubernetes
from kubernetes import client, config
import yaml
import tempfile

class ModelDeployer:
    def __init__(self, registry_url, namespace="default"):
        self.registry_url = registry_url
        self.namespace = namespace
        self.docker_client = docker.from_env()
        
        # Setup Kubernetes
        try:
            config.load_incluster_config()
        except:
            config.load_kube_config()
        self.k8s_client = client.CoreV1Api()
        self.k8s_apps = client.AppsV1Api()
    
    def build_and_push(self, model_path, version):
        # Build Docker image
        dockerfile = self._generate_dockerfile(model_path)
        image_tag = f"{self.registry_url}/trading-model:{version}"
        
        with tempfile.TemporaryDirectory() as tmpdir:
            with open(f"{tmpdir}/Dockerfile", "w") as f:
                f.write(dockerfile)
            
            self.docker_client.images.build(
                path=tmpdir,
                tag=image_tag,
                buildargs={"MODEL_PATH": model_path}
            )
            
            # Push to registry
            self.docker_client.images.push(image_tag)
        
        return image_tag
    
    def deploy_model(self, image_tag, replicas=3):
        # Create deployment
        deployment = self._create_deployment_manifest(image_tag, replicas)
        service = self._create_service_manifest()
        
        # Apply to cluster
        try:
            self.k8s_apps.create_namespaced_deployment(
                body=deployment,
                namespace=self.namespace
            )
            self.k8s_client.create_namespaced_service(
                body=service,
                namespace=self.namespace
            )
        except kubernetes.client.rest.ApiException as e:
            if e.status == 409:  # Already exists
                self.k8s_apps.patch_namespaced_deployment(
                    name="trading-model",
                    namespace=self.namespace,
                    body=deployment
                )
    
    def _generate_dockerfile(self, model_path):
        return f"""
        FROM python:3.9-slim
        
        WORKDIR /app
        COPY {model_path} /app/model
        
        RUN pip install tensorflow fastapi uvicorn redis
        
        COPY deployment/model_server.py /app/
        
        CMD ["python", "-m", "uvicorn", "model_server:app", "--host", "0.0.0.0"]
        """
    
    def _create_deployment_manifest(self, image_tag, replicas):
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "trading-model"},
            "spec": {
                "replicas": replicas,
                "selector": {
                    "matchLabels": {"app": "trading-model"}
                },
                "template": {
                    "metadata": {
                        "labels": {"app": "trading-model"}
                    },
                    "spec": {
                        "containers": [{
                            "name": "model-server",
                            "image": image_tag,
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {
                                    "memory": "1Gi",
                                    "cpu": "500m"
                                },
                                "limits": {
                                    "memory": "2Gi",
                                    "cpu": "1000m"
                                }
                            }
                        }]
                    }
                }
            }
        }
    
    def _create_service_manifest(self):
        return {
            "apiVersion": "v1",
            "kind": "Service",
            "metadata": {"name": "trading-model-service"},
            "spec": {
                "selector": {"app": "trading-model"},
                "ports": [{
                    "protocol": "TCP",
                    "port": 80,
                    "targetPort": 8000
                }],
                "type": "LoadBalancer"
            }
        }