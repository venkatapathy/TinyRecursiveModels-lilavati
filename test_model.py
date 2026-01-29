import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "checkpoints/slm/gemma_270m_vanilla/final"
print(f"Loading model from {model_path}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="cpu")

prompt = "572+896="
inputs = tokenizer(prompt, return_tensors="pt")

print(f"Prompt: {prompt}")
print(f"Input IDs: {inputs['input_ids']}")

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=False)
print(f"Full output: {generated_text}")

new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
print(f"New tokens: {new_tokens}")
print(f"New tokens decoded: {tokenizer.decode(new_tokens)}")
