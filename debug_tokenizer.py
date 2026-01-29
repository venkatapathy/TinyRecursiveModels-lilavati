from transformers import AutoTokenizer

tokenizer_name = "alpindale/gemma-2b"
tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

text = "123+456=579"
tokens = tokenizer.encode(text)
token_strings = [tokenizer.decode([t]) for t in tokens]

print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Token strings: {token_strings}")

eq_token_id = tokenizer.encode("=", add_special_tokens=False)[-1]
print(f"EQ token ID: {eq_token_id}")

# Check if '=' is in the tokens
if eq_token_id in tokens:
    print(f"EQ token index in tokens: {tokens.index(eq_token_id)}")
else:
    print("EQ token NOT found in tokens!")

# Check BOS
print(f"BOS token ID: {tokenizer.bos_token_id}")
print(f"BOS token: {tokenizer.decode([tokenizer.bos_token_id])}")
