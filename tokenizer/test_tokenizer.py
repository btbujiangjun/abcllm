# Tokenizer Testing Script
# This script demonstrates the usage of SimpleTokenizer, GPT2Tokenizer, and SPTokenizer.

import os
from tokenizer.tokenizer import SimpleTokenizer, GPT2Tokenizer, SPTokenizer


def test_simple_tokenizer(data_file: str):
    """Test the SimpleTokenizer with sample text."""
    assert os.path.isfile(data_file), f"File not found: {data_file}"
    tokenizer = SimpleTokenizer.from_file(data_file)

    text = """\"It's the last he painted, you know,\"
                Mrs. Gisburn Jiang said with pardonable pride."""
    ids = tokenizer.encode(text)

    print("SimpleTokenizer Results:")
    print("Original text:", text)
    print("Encoded IDs:", ids)
    print("Decoded text:", tokenizer.decode(ids))
    print("EoS ID:", tokenizer.eos_id)
    print("Vocabulary size:", tokenizer.vocab_size)
    print("-" * 50)


def test_gpt2_tokenizer(data_file: str):
    """Test the GPT2Tokenizer with sample text."""
    assert os.path.isfile(data_file), f"File not found: {data_file}"
    with open(data_file, "r", encoding="utf-8") as f:
        raw_text = f.read()

    gpt2_tokenizer = GPT2Tokenizer()
    enc_text = gpt2_tokenizer.encode(raw_text)

    print("GPT2Tokenizer Results:")
    print("Encoded IDs (truncated):", enc_text[:20], "...")  # Display first 20 tokens for brevity
    print("EoS ID:", gpt2_tokenizer.eos_id)
    print("EoS token:", gpt2_tokenizer.decode([gpt2_tokenizer.eos_id]))
    dec_text = gpt2_tokenizer.decode(enc_text)
    print("Decoded text matches original:", raw_text == dec_text)
    print("-" * 50)


def test_sp_tokenizer(model_file: str, text: str):
    """Test the SPTokenizer with sample text."""
    assert os.path.isfile(model_file), f"Model file not found: {model_file}"
    sp_tokenizer = SPTokenizer(model_file)

    ids = sp_tokenizer.encode(text)
    decoded_text = sp_tokenizer.decode(ids)

    print("SPTokenizer Results:")
    print("Original text:", text)
    print("Encoded IDs:", ids)
    print("Vocabulary size:", sp_tokenizer.vocab_size)
    print("Decoded text:", decoded_text)
    print("Decoded text matches original:", text == decoded_text)
    print("-" * 50)


def main():
    """Main function to test all tokenizers."""
    # File paths
    data_file = "./data/the-verdict.txt"
    model_file = "./data/ChatGLMTokenizer/tokenizer.model"

    # Test texts
    chinese_text = "我爱北京天安门"

    # Run tests
    test_simple_tokenizer(data_file)
    test_gpt2_tokenizer(data_file)
    test_sp_tokenizer(model_file, chinese_text)


if __name__ == "__main__":
    main()
