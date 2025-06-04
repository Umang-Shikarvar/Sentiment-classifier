# Sentiment Classifier with QLoRA (4-bit GPT-Neo)

This project presents a lightweight and efficient sentiment classifier built by fine-tuning the [`EleutherAI/gpt-neo-1.3B`](https://huggingface.co/EleutherAI/gpt-neo-1.3B) language model using **QLoRA (Quantized Low-Rank Adaptation)** with **4-bit quantization**. Trained on the [`tweet_eval/sentiment`](https://huggingface.co/datasets/cardiffnlp/tweet_eval/viewer/sentiment) dataset, the model classifies tweets into Positive, Negative, or Neutral sentiments. It leverages a prompt-based generation approach with output constrained to a fixed set of sentiment tokens using a custom LogitsProcessor, enabling fast and deterministic inference even on low-resource hardware. This makes it ideal for real-time sentiment analysis applications in resource-constrained environments.

---

## Model Details

- **Base Model**: `EleutherAI/gpt-neo-1.3B`
- **Quantization**: 4-bit (`nf4`) via `BitsAndBytesConfig`
- **Fine-tuning**: QLoRA using `peft`
- **Adapter Format**: Compatible with `peft`
- **Dataset**: [`tweet_eval/sentiment`](https://huggingface.co/datasets/cardiffnlp/tweet_eval/viewer/sentiment)
- **Classes**: `Positive`, `Negative`, `Neutral`

---

## How It Works

This model is treated as a text generation model but constrained using:

- a fixed prompt format

- Output generation restricted to a **single token**

- Vocabulary limited to only **3 sentiment tokens**: `["Positive", "Negative", "Neutral"]` using a custom `LogitsProcessor`:

```python
from transformers import LogitsProcessor

class RestrictVocabLogitsProcessor(LogitsProcessor):
  def __init__(self, allowed_token_ids):
      self.allowed_token_ids = allowed_token_ids

  def __call__(self, input_ids, scores):
      mask = torch.full_like(scores, float("-inf"))
      mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
      return mask
```
This design makes the model extremely lightweight and deterministic for sentiment classification tasks.

---

## 🔗 Try It Out

[![Sentiment Classifier Demo](https://cdn-uploads.huggingface.co/production/uploads/67f01e586dca39c70694e116/952tyeAW48A6Gq7XvJr8A.png)](https://huggingface.co/spaces/umangshikarvar/Sentiment_classifier)

Run the model interactively in your browser using Gradio — no setup needed by clicking the image above.

---

## How to Use

**Install dependencies**
```bash
pip install torch transformers peft
```

**Load the model and run inference**
```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel, PeftConfig

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token  # Required for generation

# Load adapter config and base model
checkpoint = "umangshikarvar/sentiment-qlora-gptneo"
peft_config = PeftConfig.from_pretrained(checkpoint)

torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
base_model = AutoModelForCausalLM.from_pretrained(
    peft_config.base_model_name_or_path,
    torch_dtype=torch_dtype
)

# Load QLoRA adapter
model = PeftModel.from_pretrained(base_model, checkpoint)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

# Restrict output to sentiment tokens
class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

sentiment_words = ["Positive", "Negative", "Neutral"]
allowed_ids = [
    tokenizer(word, add_special_tokens=False)["input_ids"][0]
    for word in sentiment_words
]
logits_processor = LogitsProcessorList([
    RestrictVocabLogitsProcessor(allowed_ids)
])

# Inference function
def predict_sentiment(tweet: str) -> str:
    prompt = f"Tweet: {tweet}\nSentiment:"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=False,
        logits_processor=logits_processor
    )

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    prediction = response.replace(prompt, "").strip().split()[0]

    if prediction.lower().startswith("pos"):
        return "Positive"
    elif prediction.lower().startswith("neg"):
        return "Negative"
    else:
        return "Neutral"
```

**Example usage**
```python
print(predict_sentiment("I absolutely love this new feature!"))
# Output: Positive