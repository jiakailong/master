import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import importlib
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc, 
    precision_recall_curve, average_precision_score, accuracy_score,
    precision_score, recall_score, f1_score
)
from sklearn.preprocessing import label_binarize
from itertools import cycle
import warnings
warnings.filterwarnings('ignore')

from data_utils.LungDataLoader import LungDataLoader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('evaluation')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size in evaluation')
    parser.add_argument('--model', default='pointnet2_cls_msg_intensity', help='model name')
    parser.add_argument('--num_category', default=4, type=int, help='number of categories')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--checkpoint_path', type=str, required=True, help='path to model checkpoint')
    parser.add_argument('--use_intensity', action='store_true', default=True, help='use intensity information')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normal information')
    parser.add_argument('--process_data', action='store_true', default=False, help='save data offline')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampling')
    parser.add_argument('--output_dir', type=str, default='./evaluation_results', help='output directory for results')
    return parser.parse_args()

class ModelEvaluator:
    def __init__(self, args):
        self.args = args
        self.num_class = args.num_category
        self.class_names = ['Class 0', 'Class 1', 'Class 2', 'Class 3']
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Setup device
        if not args.use_cpu:
            os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def load_model(self):
        """Load the trained model"""
        model = importlib.import_module(self.args.model)
        classifier = model.get_model(
            self.num_class, 
            normal_channel=self.args.use_normals, 
            use_intensity=self.args.use_intensity
        )
        
        # Load checkpoint
        checkpoint = torch.load(self.args.checkpoint_path, map_location=self.device)
        classifier.load_state_dict(checkpoint['model_state_dict'])
        classifier = classifier.to(self.device)
        classifier.eval()
        
        print(f"Model loaded from {self.args.checkpoint_path}")
        print(f"Model accuracy: {checkpoint.get('instance_acc', 'N/A')}")
        
        return classifier
    
    def load_data(self):
        """Load test dataset"""
        data_path = '/media/jiang/jkl/project/dataset/point_cloud_3dgcn_backup'
        test_dataset = LungDataLoader(
            root=data_path, 
            args=self.args, 
            split='test', 
            process_data=self.args.process_data
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=4
        )
        
        return test_loader
    
    def evaluate_model(self, model, test_loader):
        """Evaluate model and collect predictions"""
        all_predictions = []
        all_probabilities = []
        all_targets = []
        
        with torch.no_grad():
            for points, targets in tqdm(test_loader, desc="Evaluating"):
                if not self.args.use_cpu:
                    points, targets = points.to(self.device), targets.to(self.device)
                
                points = points.transpose(2, 1)
                pred_logits, _ = model(points)
                
                # Get probabilities (convert from log probabilities)
                probabilities = torch.exp(pred_logits)
                predictions = pred_logits.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_probabilities), np.array(all_targets)
    
    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # Add accuracy for each class
        for i in range(self.num_class):
            accuracy = cm[i, i] / cm[i, :].sum() if cm[i, :].sum() > 0 else 0
            plt.text(i + 0.5, i - 0.3, f'Acc: {accuracy:.3f}', 
                    ha='center', va='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'confusion_matrix.pdf', bbox_inches='tight')
        plt.show()
        
        return cm
    
    def plot_roc_curves(self, y_true, y_prob):
        """Plot ROC curves for each class and micro/macro averages"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_class)))
        
        # Compute ROC curve and ROC area for each class
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(self.num_class):
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_prob.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        
        # Compute macro-average ROC curve and ROC area
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(self.num_class)]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(self.num_class):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= self.num_class
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
        
        # Plot ROC curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        
        # Plot ROC curve for each class
        for i, color in zip(range(self.num_class), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AUC = {roc_auc[i]:.3f})')
        
        # Plot micro and macro averages
        plt.plot(fpr["micro"], tpr["micro"], 
                label=f'Micro-average (AUC = {roc_auc["micro"]:.3f})',
                color='deeppink', linestyle=':', linewidth=4)
        
        plt.plot(fpr["macro"], tpr["macro"],
                label=f'Macro-average (AUC = {roc_auc["macro"]:.3f})',
                color='navy', linestyle=':', linewidth=4)
        
        plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Multi-class Classification')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'roc_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'roc_curves.pdf', bbox_inches='tight')
        plt.show()
        
        return roc_auc
    
    def plot_pr_curves(self, y_true, y_prob):
        """Plot Precision-Recall curves for each class"""
        # Binarize the output
        y_true_bin = label_binarize(y_true, classes=list(range(self.num_class)))
        
        # Compute PR curve and average precision for each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        
        for i in range(self.num_class):
            precision[i], recall[i], _ = precision_recall_curve(y_true_bin[:, i], y_prob[:, i])
            average_precision[i] = average_precision_score(y_true_bin[:, i], y_prob[:, i])
        
        # Compute micro-average PR curve and average precision
        precision["micro"], recall["micro"], _ = precision_recall_curve(
            y_true_bin.ravel(), y_prob.ravel())
        average_precision["micro"] = average_precision_score(y_true_bin, y_prob, average="micro")
        
        # Plot PR curves
        plt.figure(figsize=(12, 8))
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green'])
        
        # Plot PR curve for each class
        for i, color in zip(range(self.num_class), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2,
                    label=f'{self.class_names[i]} (AP = {average_precision[i]:.3f})')
        
        # Plot micro-average PR curve
        plt.plot(recall["micro"], precision["micro"],
                label=f'Micro-average (AP = {average_precision["micro"]:.3f})',
                color='gold', linestyle=':', linewidth=4)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves for Multi-class Classification')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_curves.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.output_dir / 'pr_curves.pdf', bbox_inches='tight')
        plt.show()
        
        return average_precision
    
    def generate_classification_report(self, y_true, y_pred, y_prob):
        """Generate detailed classification report"""
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro = precision_score(y_true, y_pred, average='macro')
        recall_macro = recall_score(y_true, y_pred, average='macro')
        f1_macro = f1_score(y_true, y_pred, average='macro')
        
        precision_micro = precision_score(y_true, y_pred, average='micro')
        recall_micro = recall_score(y_true, y_pred, average='micro')
        f1_micro = f1_score(y_true, y_pred, average='micro')
        
        # Per-class metrics
        precision_per_class = precision_score(y_true, y_pred, average=None)
        recall_per_class = recall_score(y_true, y_pred, average=None)
        f1_per_class = f1_score(y_true, y_pred, average=None)
        
        # Classification report
        report = classification_report(y_true, y_pred, target_names=self.class_names)
        
        # Save detailed report
        with open(self.output_dir / 'classification_report.txt', 'w') as f:
            f.write("=" * 50 + "\n")
            f.write("LUNG POINT CLOUD CLASSIFICATION EVALUATION REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write("OVERALL METRICS:\n")
            f.write("-" * 20 + "\n")
            f.write(f"Accuracy: {accuracy:.4f}\n")
            f.write(f"Macro-average Precision: {precision_macro:.4f}\n")
            f.write(f"Macro-average Recall: {recall_macro:.4f}\n")
            f.write(f"Macro-average F1-score: {f1_macro:.4f}\n")
            f.write(f"Micro-average Precision: {precision_micro:.4f}\n")
            f.write(f"Micro-average Recall: {recall_micro:.4f}\n")
            f.write(f"Micro-average F1-score: {f1_micro:.4f}\n\n")
            
            f.write("PER-CLASS METRICS:\n")
            f.write("-" * 20 + "\n")
            for i in range(self.num_class):
                f.write(f"{self.class_names[i]}:\n")
                f.write(f"  Precision: {precision_per_class[i]:.4f}\n")
                f.write(f"  Recall: {recall_per_class[i]:.4f}\n")
                f.write(f"  F1-score: {f1_per_class[i]:.4f}\n\n")
            
            f.write("DETAILED CLASSIFICATION REPORT:\n")
            f.write("-" * 35 + "\n")
            f.write(report)
        
        # Print summary
        print("\n" + "=" * 50)
        print("EVALUATION SUMMARY")
        print("=" * 50)
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")
        print(f"Micro F1-score: {f1_micro:.4f}")
        print("\nPer-class F1-scores:")
        for i, f1 in enumerate(f1_per_class):
            print(f"  {self.class_names[i]}: {f1:.4f}")
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_micro': precision_micro,
            'recall_micro': recall_micro,
            'f1_micro': f1_micro,
            'per_class_metrics': {
                'precision': precision_per_class,
                'recall': recall_per_class,
                'f1': f1_per_class
            }
        }
    
    def run_evaluation(self):
        """Run complete evaluation pipeline"""
        print("Loading model...")
        model = self.load_model()
        
        print("Loading test data...")
        test_loader = self.load_data()
        
        print("Evaluating model...")
        predictions, probabilities, targets = self.evaluate_model(model, test_loader)
        
        print("Generating confusion matrix...")
        cm = self.plot_confusion_matrix(targets, predictions)
        
        print("Generating ROC curves...")
        roc_auc = self.plot_roc_curves(targets, probabilities)
        
        print("Generating PR curves...")
        average_precision = self.plot_pr_curves(targets, probabilities)
        
        print("Generating classification report...")
        metrics = self.generate_classification_report(targets, predictions, probabilities)
        
        # Save numerical results
        results = {
            'confusion_matrix': cm.tolist(),
            'roc_auc': roc_auc,
            'average_precision': average_precision,
            'metrics': metrics
        }
        
        import json
        with open(self.output_dir / 'evaluation_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\nEvaluation complete! Results saved to {self.output_dir}")
        
        return results

def main():
    args = parse_args()
    evaluator = ModelEvaluator(args)
    results = evaluator.run_evaluation()

if __name__ == '__main__':
    main()