from tokenizer.tokenizer import SimpleTokenizer, GPT2Tokenizer, SPTokenizer

data_file = "./data/the-verdict.txt"
tokenizer = SimpleTokenizer.from_file(data_file)

text = """"It's the last he painted, you know,"
            Mrs. Gisburn Jiang said with pardonable pride."""
ids = tokenizer.encode(text)
print("origin:", text)
print("ids:", ids)
print("decode:", tokenizer.decode(ids))
print("tokenizer eos_id:", tokenizer.eos_id)
print("tokenizer vocab_size:", len(tokenizer.int_to_str))

with open(data_file, "r", encoding="utf-8") as f:
    raw_text = f.read()
gpt2_tokenizer = GPT2Tokenizer()
enc_text = gpt2_tokenizer.encode(raw_text)
print("gpt2 tokenizer:", enc_text)
print("gpt2 tokenizer eos_id:", gpt2_tokenizer.eos_id)
print("gpt2 tokenizer eos_text:", gpt2_tokenizer.decode([gpt2_tokenizer.eos_id]))

dec_text = gpt2_tokenizer.decode(enc_text)
print("gpt2 decode text:", dec_text)

print(enc_text == dec_text)

model_file = "./data/ChatGLMTokenizer/tokenizer.model"
sp_tokenizer = SPTokenizer(model_file)
text = "我爱北京天安门"
ids = sp_tokenizer.encode(text)
print(ids)
print("ChatGLM vocab_size:", sp_tokenizer.vocab_size)
decode_text = sp_tokenizer.decode(ids)
print("decode text:", decode_text)
print(text == decode_text)

