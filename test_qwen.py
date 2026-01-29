import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re

model_id = "Qwen/Qwen2.5-Math-1.5B-Instruct"
print(f"Loading model {model_id}...")

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True)

eq = "100-45"
prompt = f"Please calculate the result of the following arithmetic operation: {eq}. Provide only the final numerical answer inside \\boxed{{}}."
messages = [{"role": "user", "content": prompt}]

inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(inputs, max_new_tokens=100, do_sample=False)

generated_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
print(f"Prompt: {eq}")
print(f"Response: {generated_text}")

# Extract prediction using the same logic as evaluate_qwen.py
match = re.search(r'\\boxed\{([-?\d,.]+)\}', generated_text)
if match:
    prediction = match.group(1).replace(',', '').strip()
else:
    clean_text = generated_text.replace(',', '')
    numbers = re.findall(r'-?\d+', clean_text)
    prediction = numbers[-1] if numbers else ""

print(f"Extracted prediction: {prediction}")
print(f"Expectation: 55")
