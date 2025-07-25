# heading_classifier.py

import os
import torch
from transformers import DistilBertTokenizerFast, DistilBertForSequenceClassification

class HeadingClassifier:
    def __init__(self, model_path: str = "./model"):
        if not os.path.exists(model_path):
            raise RuntimeError(f"Model path does not exist: {model_path}")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        self.model.eval()

    def classify_span(self, span: dict, context: str = "") -> str:
        input_text = span['text'] + " [SEP] " + f"font_size:{span['size']} bold:{span['bold']} y_pos:{span['y']} page:{span['page']} context:{context}"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=128,
            padding=True
        )
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            pred_id = torch.argmax(logits).item()
        
        return self._id_to_label(pred_id)

    def _id_to_label(self, pred_id: int) -> str:
        label_map = {
            0: "non-heading",
            1: "H1",
            2: "H2",
            3: "H3"
        }
        return label_map.get(pred_id, "non-heading")
