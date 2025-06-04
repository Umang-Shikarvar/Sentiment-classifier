import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from peft import PeftModel, PeftConfig
import gradio as gr

# === Load Tokenizer ===
tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
tokenizer.pad_token = tokenizer.eos_token

# === Load Model + QLoRA Adapter ===
checkpoint_dir = "/Users/umangshikarvar/Desktop/QLORA/Fine-tuned model"  # Update if needed
peft_config = PeftConfig.from_pretrained(checkpoint_dir)
base_model = AutoModelForCausalLM.from_pretrained(peft_config.base_model_name_or_path, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, checkpoint_dir)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.eval().to(device)

# === Define Custom LogitsProcessor ===
class RestrictVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids, scores):
        mask = torch.full_like(scores, float("-inf"))
        mask[:, self.allowed_token_ids] = scores[:, self.allowed_token_ids]
        return mask

# === Set Allowed Sentiment Tokens ===
sentiment_words = ["Positive", "Negative", "Neutral"]
allowed_ids = [
    tokenizer(word, add_special_tokens=False)["input_ids"][0]
    for word in sentiment_words
]
logits_processor = LogitsProcessorList([
    RestrictVocabLogitsProcessor(allowed_ids)
])

# === Inference Function ===
def predict_sentiment(tweet):
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

# === Gradio Interface ===
gr.Interface(
    fn=predict_sentiment,
    inputs=gr.Textbox(lines=2, placeholder="Enter the text", label="Statement"),
    outputs="text",
    title="Sentiment Classifier",
    description="Classifies the sentiment of a statement, as Positive, Negative, or Neutral."
).launch(share=True)