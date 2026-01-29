from transformers import AutoTokenizer

tokenizer_name = "alpindale/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

text = "572+896=1468"
outputs = tokenizer(text, padding="max_length", truncation=True, max_length=20)
input_ids = outputs["input_ids"]

labels = list(input_ids)
eq_token_id = tokenizer.encode("=", add_special_tokens=False)[-1]
eq_index = labels.index(eq_token_id)

for i in range(eq_index + 1):
    labels[i] = -100

print(f"Input : {input_ids}")
print(f"Labels: {labels}")

for i in range(len(input_ids)):
    token = tokenizer.decode([input_ids[i]])
    label_token = tokenizer.decode([labels[i]]) if labels[i] != -100 else "MASK"
    print(f"{i:2}: ID={input_ids[i]:6} Token={token:10} Label={label_token}")
