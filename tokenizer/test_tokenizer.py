from tokenizer import SimpleTokenizer, GPT2Tokenizer

tokenizer = SimpleTokenizer.from_file("the-verdict.txt")

text = """"It's the last he painted, you know,"
            Mrs. Gisburn Jiang said with pardonable pride."""
ids = tokenizer.encode(text)
print("origin:", text)
print("ids:", ids)
print("decode:", tokenizer.decode(ids))
print("tokenizer eos_id:", tokenizer.eos_id())
print("tokenizer vocab_size:", len(tokenizer.int_to_str))

with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()
gpt2_tokenizer = GPT2Tokenizer()
enc_text = gpt2_tokenizer.encode(raw_text)
print("gpt2 tokenizer:", enc_text)
print("gpt2 tokenizer eos_id:", gpt2_tokenizer.eos_id())
print("gpt2 tokenizer eos_text:", gpt2_tokenizer.decode([gpt2_tokenizer.eos_id()]))

dec_text = gpt2_tokenizer.decode(enc_text)
print("gpt2 decode text:", dec_text)

print(enc_text == dec_text)

