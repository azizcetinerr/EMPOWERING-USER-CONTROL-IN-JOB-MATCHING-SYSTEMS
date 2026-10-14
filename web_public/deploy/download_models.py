"""Downloads the transformer models in advance while the Docker image is built (the models in config.yaml)."""
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification

CLASSIFIER_MODELS = ["roberta-base-openai-detector", "cross-encoder/nli-distilroberta-base"]
EMBEDDING_MODELS = ["xlm-roberta-base"]

for m in CLASSIFIER_MODELS:
    AutoTokenizer.from_pretrained(m)
    AutoModelForSequenceClassification.from_pretrained(m)
    print("downloaded:", m)
for m in EMBEDDING_MODELS:
    AutoTokenizer.from_pretrained(m)
    AutoModel.from_pretrained(m)
    print("downloaded:", m)
