from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Use this model: light, fast, 2-class classifier
model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# Download and save the model locally
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Save to your project directory
save_path = "./local_model"
tokenizer.save_pretrained(save_path)
model.save_pretrained(save_path)

print("✅ Model saved to ./local_model")
