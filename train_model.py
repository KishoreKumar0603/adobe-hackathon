import json
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import joblib
import re

DATASET_PATH = Path("dataset")
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

def extract_text_features(text):
    """Extract text-based features for classification"""
    features = {}
    
    # Basic text features
    features['text_length'] = len(text)
    features['word_count'] = len(text.split())
    features['avg_word_length'] = np.mean([len(word) for word in text.split()]) if text.split() else 0
    
    # Pattern features
    features['has_numbers'] = bool(re.search(r'\d', text))
    features['starts_with_number'] = bool(re.match(r'^\d+', text))
    features['has_colon'] = ':' in text
    features['has_period'] = '.' in text
    features['is_all_caps'] = text.isupper()
    features['is_title_case'] = text.istitle()
    
    # Numbering patterns
    features['numbered_h1'] = bool(re.match(r'^\d+\.?\s', text))  # 1. or 1 
    features['numbered_h2'] = bool(re.match(r'^\d+\.\d+\.?\s', text))  # 1.1. or 1.1
    features['numbered_h3'] = bool(re.match(r'^\d+\.\d+\.\d+\.?\s', text))  # 1.1.1.
    
    # Common heading words
    heading_words = {'introduction', 'conclusion', 'summary', 'overview', 'chapter', 
                    'section', 'background', 'methodology', 'results', 'discussion',
                    'abstract', 'references', 'appendix', 'bibliography'}
    features['has_heading_words'] = any(word.lower() in heading_words for word in text.split())
    
    return features

def prepare_features(data_item):
    """Prepare feature vector for a single data item"""
    text = data_item['text']
    features = data_item['features']
    
    # Text features
    text_features = extract_text_features(text)
    
    # Combine all features
    feature_vector = {
        # Original features
        'font_size': features.get('font_size', 12),
        'bold': int(features.get('bold', False)),
        'font_ratio': features.get('font_ratio', 1.0),
        'y_pos': features.get('y_pos', 0),
        'x_start': features.get('x_start', 0),
        'width': features.get('width', 0),
        
        # Text features
        **text_features
    }
    
    return feature_vector

def load_and_prepare_data():
    """Load training data and prepare features"""
    data_file = DATASET_PATH / "train_data.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"Training data not found at {data_file}")
    
    data = []
    labels = []
    
    print("Loading training data...")
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                item = json.loads(line.strip())
                if not item.get('text', '').strip():
                    continue
                    
                feature_vector = prepare_features(item)
                data.append(feature_vector)
                labels.append(item['label'])
                
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping malformed line {line_num}: {e}")
                continue
    
    print(f"Loaded {len(data)} training examples")
    
    # Convert to arrays
    feature_names = list(data[0].keys()) if data else []
    X = np.array([[item[feat] for feat in feature_names] for item in data])
    y = np.array(labels)
    
    return X, y, feature_names

def train_model():
    """Train the heading classification model"""
    print("Preparing training data...")
    X, y, feature_names = load_and_prepare_data()
    
    # Check class distribution
    unique_labels, counts = np.unique(y, return_counts=True)
    print(f"\nClass distribution:")
    for label, count in zip(unique_labels, counts):
        print(f"  {label}: {count} ({count/len(y)*100:.1f}%)")
    
    # Handle class imbalance by filtering out some non-headings
    if 'non-heading' in unique_labels:
        non_heading_indices = np.where(y == 'non-heading')[0]
        heading_indices = np.where(y != 'non-heading')[0]
        
        # Keep all headings, but sample non-headings
        max_non_headings = len(heading_indices) * 3  # 3:1 ratio
        
        if len(non_heading_indices) > max_non_headings:
            np.random.seed(42)
            sampled_non_headings = np.random.choice(
                non_heading_indices, max_non_headings, replace=False
            )
            selected_indices = np.concatenate([heading_indices, sampled_non_headings])
            X = X[selected_indices]
            y = y[selected_indices]
            print(f"Balanced dataset: {len(X)} samples")
    
    # Split data (remove stratify if still causing issues)
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError as e:
        print(f"⚠️ Stratified split failed: {e}")
        print("Using simple random split instead...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"Training set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'  # Handle remaining class imbalance
    )
    
    model.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    
    print(f"Training accuracy: {train_score:.3f}")
    print(f"Test accuracy: {test_score:.3f}")
    
    # Detailed evaluation
    y_pred = model.predict(X_test)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = list(zip(feature_names, model.feature_importances_))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Most Important Features:")
    for feat, importance in feature_importance[:10]:
        print(f"  {feat}: {importance:.3f}")
    
    # Save model and metadata
    model_file = MODEL_PATH / "heading_classifier.joblib"
    metadata_file = MODEL_PATH / "model_metadata.json"
    
    joblib.dump(model, model_file)
    
    metadata = {
        "feature_names": feature_names,
        "classes": model.classes_.tolist(),
        "train_accuracy": train_score,
        "test_accuracy": test_score,
        "feature_importance": dict(feature_importance),
        "training_samples": len(X_train)
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Model saved to: {model_file}")
    print(f"📋 Metadata saved to: {metadata_file}")
    
    return model, feature_names

if __name__ == "__main__":
    train_model()