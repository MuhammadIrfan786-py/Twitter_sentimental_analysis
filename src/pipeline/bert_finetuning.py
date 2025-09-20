#!/usr/bin/env python3
"""
BERT Fine-tuning Script for Twitter Sentiment Analysis
Reusable script with DagsHub MLflow integration
"""

import os
import sys
import json
import argparse
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Handle different versions of transformers
try:
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        get_linear_schedule_with_warmup
    )
except ImportError as e:
    print(f"Import error with transformers: {e}")
    print("Please install transformers: pip install transformers")
    sys.exit(1)

# Handle AdamW import (moved in newer versions)
try:
    from torch.optim import AdamW
except ImportError:
    try:
        from transformers import AdamW
    except ImportError:
        print("Could not import AdamW. Please update PyTorch and transformers.")
        sys.exit(1)

import mlflow
try:
    import mlflow.pytorch
except ImportError:
    print("MLflow PyTorch integration not available")

from tqdm.auto import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TwitterSentimentBERTDataset(Dataset):
    """
    BERT-compatible Dataset class for Twitter Sentiment Analysis
    Adapted from your existing TwitterSentimentDataset
    """
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        
        # BERT tokenization
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class BERTSentimentClassifier:
    """
    Main class for BERT fine-tuning with MLflow tracking
    """
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
        self.model = AutoModelForSequenceClassification.from_pretrained(
            config['model_name'],
            num_labels=config['num_labels']
        )
        self.model.to(self.device)
        
    def setup_mlflow(self):
        """Setup DagsHub MLflow tracking"""
        # Configure DagsHub MLflow
        if self.config.get('dagshub_url'):
            mlflow.set_tracking_uri(self.config['dagshub_url'])
        
        # Set experiment name
        experiment_name = self.config.get('experiment_name', 'twitter_sentiment_bert')
        mlflow.set_experiment(experiment_name)
        
        # Start MLflow run
        run_name = f"bert_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        mlflow.start_run(run_name=run_name)
        
        # Log parameters
        mlflow.log_params({
            'model_name': self.config['model_name'],
            'learning_rate': self.config['learning_rate'],
            'batch_size': self.config['batch_size'],
            'epochs': self.config['epochs'],
            'max_length': self.config['max_length'],
            'warmup_steps': self.config['warmup_steps'],
            'weight_decay': self.config['weight_decay']
        })
        
    def load_data(self, file_path):
        """Load and preprocess data from CSV"""
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Loaded data shape: {df.shape}")
            
            # Handle your specific dataset format (target, text columns)
            texts = df['text'].tolist()
            labels = df['target'].tolist()
            
            # Convert labels to binary if needed (0,4) -> (0,1)
            if set(labels) == {0, 4}:
                labels = [1 if label == 4 else 0 for label in labels]
                logger.info("Converted labels from (0,4) to (0,1)")
            
            return texts, labels
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_data_loaders(self, texts, labels):
        """Create train and validation data loaders"""
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, 
            test_size=self.config['val_split'], 
            random_state=42,
            stratify=labels
        )
        
        # Create datasets
        train_dataset = TwitterSentimentBERTDataset(
            train_texts, train_labels, self.tokenizer, self.config['max_length']
        )
        val_dataset = TwitterSentimentBERTDataset(
            val_texts, val_labels, self.tokenizer, self.config['max_length']
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False
        )
        
        logger.info(f"Train dataset size: {len(train_dataset)}")
        logger.info(f"Validation dataset size: {len(val_dataset)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        predictions = []
        actual_labels = []
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            # Collect predictions
            preds = torch.argmax(outputs.logits, dim=1)
            predictions.extend(preds.cpu().numpy())
            actual_labels.extend(labels.cpu().numpy())
            
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(train_loader)
        accuracy = accuracy_score(actual_labels, predictions)
        
        return avg_loss, accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        predictions = []
        actual_labels = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
                actual_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        accuracy = accuracy_score(actual_labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            actual_labels, predictions, average='weighted'
        )
        
        return avg_loss, accuracy, precision, recall, f1, predictions, actual_labels
    
    def train(self, data_file_path):
        """Main training loop with MLflow tracking"""
        # Setup MLflow
        self.setup_mlflow()
        
        try:
            # Load data
            texts, labels = self.load_data(data_file_path)
            train_loader, val_loader = self.create_data_loaders(texts, labels)
            
            # Setup optimizer and scheduler
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config['learning_rate'],
                weight_decay=self.config['weight_decay']
            )
            
            total_steps = len(train_loader) * self.config['epochs']
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.config['warmup_steps'],
                num_training_steps=total_steps
            )
            
            # Training loop
            best_accuracy = 0
            best_model_state = None
            
            for epoch in range(self.config['epochs']):
                logger.info(f"Epoch {epoch + 1}/{self.config['epochs']}")
                
                # Train
                train_loss, train_acc = self.train_epoch(train_loader, optimizer, scheduler)
                
                # Validate
                val_loss, val_acc, val_precision, val_recall, val_f1, predictions, actual_labels = self.validate(val_loader)
                
                # Log metrics to MLflow
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'train_accuracy': train_acc,
                    'val_loss': val_loss,
                    'val_accuracy': val_acc,
                    'val_precision': val_precision,
                    'val_recall': val_recall,
                    'val_f1': val_f1
                }, step=epoch)
                
                logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
                logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")
                
                # Save best model
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    best_model_state = self.model.state_dict().copy()
                    
                    # Save model checkpoint
                    model_dir = os.path.join('models', 'bert_sentiment')
                    os.makedirs(model_dir, exist_ok=True)
                    
                    # Save model and tokenizer
                    self.model.save_pretrained(model_dir)
                    self.tokenizer.save_pretrained(model_dir)
                    
                    # Log model to MLflow
                    mlflow.pytorch.log_model(self.model, "best_model")
                    
                    logger.info(f"New best model saved with accuracy: {best_accuracy:.4f}")
            
            # Final evaluation
            logger.info(f"Training completed. Best validation accuracy: {best_accuracy:.4f}")
            
            # Log confusion matrix
            cm = confusion_matrix(actual_labels, predictions)
            mlflow.log_text(str(cm), "confusion_matrix.txt")
            
            return best_accuracy
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        finally:
            mlflow.end_run()
    
    def predict(self, texts):
        """Predict sentiment for new texts"""
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for text in texts:
                encoding = self.tokenizer(
                    text,
                    truncation=True,
                    padding='max_length',
                    max_length=self.config['max_length'],
                    return_tensors='pt'
                )
                
                input_ids = encoding['input_ids'].to(self.device)
                attention_mask = encoding['attention_mask'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probs = torch.softmax(outputs.logits, dim=1)
                pred = torch.argmax(outputs.logits, dim=1)
                
                predictions.append(pred.cpu().numpy()[0])
                probabilities.append(probs.cpu().numpy()[0])
        
        return predictions, probabilities

def load_config(config_path):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    parser = argparse.ArgumentParser(description='BERT Fine-tuning for Twitter Sentiment Analysis')
    parser.add_argument('--config', type=str, default='config/bert_config.json', 
                       help='Path to configuration file')
    parser.add_argument('--data', type=str, default='data/processed/processed.csv',
                       help='Path to training data CSV')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='train',
                       help='Mode: train or predict')
    parser.add_argument('--text', type=str, help='Text to predict (for predict mode)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Initialize classifier
    classifier = BERTSentimentClassifier(config)
    
    if args.mode == 'train':
        # Train the model
        logger.info("Starting BERT fine-tuning...")
        best_accuracy = classifier.train(args.data)
        logger.info(f"Training completed with best accuracy: {best_accuracy:.4f}")
        
    elif args.mode == 'predict':
        if not args.text:
            logger.error("Please provide text to predict using --text argument")
            sys.exit(1)
        
        # Load trained model
        model_dir = os.path.join('models', 'bert_sentiment')
        if not os.path.exists(model_dir):
            logger.error(f"No trained model found in {model_dir}")
            sys.exit(1)
        
        # Predict
        predictions, probabilities = classifier.predict([args.text])
        sentiment = "Positive" if predictions[0] == 1 else "Negative"
        confidence = max(probabilities[0])
        
        print(f"Text: {args.text}")
        print(f"Predicted Sentiment: {sentiment}")
        print(f"Confidence: {confidence:.4f}")

if __name__ == "__main__":
    main()