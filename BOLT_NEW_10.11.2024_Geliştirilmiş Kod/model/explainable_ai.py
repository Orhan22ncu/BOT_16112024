import shap
import lime
import lime.lime_tabular
import tensorflow as tf
import numpy as np
from typing import Dict, List, Any

class ExplainableAI:
    def __init__(self, model, feature_names=None):
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        self.shap_values = None
        
    def initialize_explainer(self, X_train):
        """Initialize SHAP and LIME explainers"""
        # SHAP Explainer
        self.shap_explainer = shap.KernelExplainer(
            self.model.predict,
            shap.sample(X_train, 100)
        )
        
        # LIME Explainer
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(
            X_train,
            feature_names=self.feature_names,
            mode='regression'
        )
    
    def explain_prediction(self, X, method='shap'):
        """Generate explanations for predictions"""
        if method == 'shap':
            return self._explain_shap(X)
        elif method == 'lime':
            return self._explain_lime(X)
        else:
            raise ValueError(f"Unknown explanation method: {method}")
    
    def _explain_shap(self, X):
        """Generate SHAP explanations"""
        shap_values = self.shap_explainer.shap_values(X)
        
        explanations = {
            'shap_values': shap_values,
            'feature_importance': np.abs(shap_values).mean(0),
            'interaction_values': self.shap_explainer.shap_interaction_values(X)
        }
        
        return explanations
    
    def _explain_lime(self, X):
        """Generate LIME explanations"""
        explanations = []
        for i in range(len(X)):
            exp = self.lime_explainer.explain_instance(
                X[i],
                self.model.predict,
                num_features=len(self.feature_names)
            )
            explanations.append(exp)
        
        return explanations
    
    def generate_feature_importance_report(self, X):
        """Generate comprehensive feature importance report"""
        shap_explanations = self._explain_shap(X)
        
        report = {
            'global_importance': {
                'shap': shap_explanations['feature_importance'],
                'interactions': self._analyze_interactions(
                    shap_explanations['interaction_values']
                )
            },
            'local_importance': self._analyze_local_importance(
                shap_explanations['shap_values']
            )
        }
        
        return report
    
    def _analyze_interactions(self, interaction_values):
        """Analyze feature interactions"""
        n_features = len(self.feature_names)
        interactions = np.zeros((n_features, n_features))
        
        for i in range(n_features):
            for j in range(n_features):
                if i != j:
                    interactions[i, j] = np.abs(
                        interaction_values[:, i, j]
                    ).mean()
        
        return interactions
    
    def _analyze_local_importance(self, shap_values):
        """Analyze local feature importance"""
        return {
            'per_instance': shap_values,
            'summary_stats': {
                'mean': np.mean(np.abs(shap_values), axis=0),
                'std': np.std(np.abs(shap_values), axis=0),
                'max': np.max(np.abs(shap_values), axis=0)
            }
        }
    
    def plot_explanations(self, X, method='shap'):
        """Plot feature importance explanations"""
        if method == 'shap':
            shap.summary_plot(
                self.shap_values,
                X,
                feature_names=self.feature_names
            )
        elif method == 'lime':
            exp = self._explain_lime(X[0])
            exp.as_pyplot_figure()
            
    def explain_model_behavior(self, X, y_true=None):
        """Comprehensive model behavior analysis"""
        predictions = self.model.predict(X)
        
        analysis = {
            'feature_importance': self.generate_feature_importance_report(X),
            'prediction_analysis': self._analyze_predictions(
                predictions, y_true
            ),
            'decision_boundaries': self._analyze_decision_boundaries(X)
        }
        
        return analysis
    
    def _analyze_predictions(self, predictions, y_true=None):
        """Analyze prediction patterns and behavior"""
        analysis = {
            'distribution': {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'quantiles': np.percentile(predictions, [25, 50, 75])
            }
        }
        
        if y_true is not None:
            analysis['error_analysis'] = {
                'residuals': predictions - y_true,
                'error_distribution': self._analyze_error_distribution(
                    predictions, y_true
                )
            }
        
        return analysis
    
    def _analyze_error_distribution(self, predictions, y_true):
        """Analyze error distribution patterns"""
        errors = predictions - y_true
        return {
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'error_quantiles': np.percentile(errors, [25, 50, 75]),
            'skewness': self._calculate_skewness(errors),
            'kurtosis': self._calculate_kurtosis(errors)
        }
    
    def _analyze_decision_boundaries(self, X):
        """Analyze model decision boundaries"""
        # Implement decision boundary analysis
        pass
    
    @staticmethod
    def _calculate_skewness(data):
        """Calculate distribution skewness"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 3)
    
    @staticmethod
    def _calculate_kurtosis(data):
        """Calculate distribution kurtosis"""
        return np.mean(((data - np.mean(data)) / np.std(data)) ** 4)