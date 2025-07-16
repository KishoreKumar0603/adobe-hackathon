from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

class HeadingClassifier:
    def __init__(self, model_path="./local_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path, local_files_only=True)
        self.model.eval()

    def score_candidates(self, candidates):
        results = []
        for c in candidates:
            text = c["text"]
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                logits = self.model(**inputs).logits
                score = logits.squeeze().item()  # regression score
            results.append((text, score))
        return results
