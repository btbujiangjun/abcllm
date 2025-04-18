import os
import json
from dataset.dataset import (
    GPTDataset,
    GPTDataLoader,
    ABCDataLoader,
    LabeledDataset,
    InstructionDataset,
    PreferenceDataset,
)
from tokenizer.tokenizer import GPT2Tokenizer, SPTokenizer

def main():
    # Paths for data and preprocessed files
    raw_file_path = "./data/the-verdict.txt"
    preprocessed_file_path = "./data/pretrain_train_data.bin"

    # Initialize tokenizer
    gpt2_tokenizer = GPT2Tokenizer()

    # Load dataset from raw text files
    print("Loading dataset from raw files...")
    dataset = GPTDataset.from_files([raw_file_path], gpt2_tokenizer, stride=2560)

    print("Dataset statistics:")
    print(f"Number of sequences: {len(dataset)}")
    print(f"Token size: {dataset.token_size}")

    # Iterate over a subset of the dataset
    for idx, (batch, target) in enumerate(dataset):
        if idx >= 5:  # Limit output for brevity
            break
        print(f"Batch {idx}: {batch}")
        print(f"Target {idx}: {target}")

    # Load preprocessed dataset with SPTokenizer
    print("\nLoading preprocessed dataset...")
    sp_tokenizer = SPTokenizer("./data/ChatGLMTokenizer/tokenizer.model")
    preprocessed_dataset = GPTDataset.from_preprocess_files(
        [preprocessed_file_path],
        seq_len=1024,
        stride=1024,
        memmap=True,
    )

    print(f"Preprocessed dataset length: {len(preprocessed_dataset)}")
    for idx, (batch, target) in enumerate(preprocessed_dataset):
        if idx >= 500:  # Limit output for brevity
            break
        print(f"Train batch {idx}: {batch.tolist()} -> {sp_tokenizer.decode(batch.tolist())}")
        print(f"Target batch {idx}: {target.tolist()} -> {sp_tokenizer.decode(target.tolist())}")

    pfile = "./data/finetune/instruction-data-with-preference.json"
    pds = PreferenceDataset(pfile, sp_tokenizer)

    print(f"dataset size:{len(pds)}")
    print(f"max_length:{pds.max_length}")
    print(pds[0])

    batch_size = 2
    num_workers = 1
    train_loader = ABCDataLoader(
        dataset=pds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    for i, input_batch in enumerate(train_loader):
        print(f"I:{i}")
        print(input_batch)
        #print(f"prompt:{sp_tokenizer.decode(input_batch['prompt'].squeeze(0).tolist())}")
        #print(f"chosen:{sp_tokenizer.decode(input_batch['chosen'].squeeze(0).tolist())}")
        #print(f"rejected:{sp_tokenizer.decode(input_batch['rejected'].squeeze(0).tolist())}")

'''

    # Create train and validation dataloaders
    print("\nCreating train/validation dataloaders...")
    loader = GPTDataLoader(gpt2_tokenizer)
    train_loader, val_loader = loader.file_train_val_dataloader(
        raw_file_path, train_ratio=0.8, seq_len=8, stride=7
    )

    # Display train batches
    print("\nTrain DataLoader:")
    for inputs, targets in train_loader:
        for input_ids, target_ids in zip(inputs.tolist(), targets.tolist()):
            print(f"Train Input: {gpt2_tokenizer.decode(input_ids).replace('\n', ' ')}")
            print(f"Train Target: {gpt2_tokenizer.decode(target_ids).replace('\n', ' ')}")

    # Display validation batches
    print("\nValidation DataLoader:")
    for inputs, targets in val_loader:
        for input_ids, target_ids in zip(inputs.tolist(), targets.tolist()):
            print(f"Validation Input: {gpt2_tokenizer.decode(input_ids).replace('\n', ' ')}")
            print(f"Validation Target: {gpt2_tokenizer.decode(target_ids).replace('\n', ' ')}")

    # Generate text dataloader example
    example_text = """"It's the last he painted, you know," Mrs. Gisburn Jiang said with pardonable pride."""
    text_dataloader = loader.text_dataloader(example_text, batch_size=4, shuffle=True, seq_len=4, stride=1)
    print("\nText DataLoader Example:")
    for i,(inputs, targets) in enumerate(text_dataloader):
        for input_ids, target_ids in zip(inputs.tolist(), targets.tolist()):
            print(f"{i+1} Inputs: {inputs}, {gpt2_tokenizer.decode(input_ids)}")
            print(f"{i+1} Targets: {targets}, {gpt2_tokenizer.decode(target_ids)}")

    # Optional: Fine-tuning datasets
    print("\nLoading labeled datasets...")
    train_dataset = LabeledDataset(
        csv_file="./data/finetune/train.csv",
        tokenizer=gpt2_tokenizer,
        seq_len=None,
    )

    val_dataset = LabeledDataset(
        csv_file="./data/finetune/validation.csv",
        tokenizer=gpt2_tokenizer,
        seq_len=train_dataset.seq_len,
    )

    test_dataset = LabeledDataset(
        csv_file="./data/finetune/test.csv",
        tokenizer=gpt2_tokenizer,
        seq_len=train_dataset.seq_len,
    )

    # Example: Iterate through test dataset
    print("\nTest Dataset Example:")
    for text, label in test_dataset:
        print(f"Text: {text}")
        print(f"Label: {label}")

    # Instruction dataset processing
    print("\nSplitting instruction dataset...")
    with open("./data/finetune/instruction-data.json", "r", encoding="utf-8") as f:
        instruction_data = json.load(f)

    split_idx = int(len(instruction_data) * 0.9)
    with open("./data/finetune/train-instruction-data.json", "w") as train_file:
        json.dump(instruction_data[:split_idx], train_file, indent=4)

    with open("./data/finetune/val-instruction-data.json", "w") as val_file:
        json.dump(instruction_data[split_idx:], val_file, indent=4)

    instruction_dataset = InstructionDataset("./data/finetune/instruction-data.json", gpt2_tokenizer)
    print("\nInstruction Dataset Example:")
    for entry in instruction_dataset.data[:5]:  # Limit output
        print(InstructionDataset.format_input(entry))
'''

# Entry point
if __name__ == "__main__":
    main()
