import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import AutoTokenizer

def train_custom_bpe(jsonl_file, vocab_size=2000):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=["<unk>"])
    
    def get_training_corpus():
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    if "svg" in data:
                        yield data["svg"]
                except: pass

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    return tokenizer

if __name__ == "__main__":
    print("Training BPE on dataset...")
    custom_tokenizer = train_custom_bpe("dataset_raw.jsonl", vocab_size=3000)
    
    # Get the vocabulary
    vocab = custom_tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Extract the tokens (skip the first few which are single characters/specials)
    new_tokens = [t[0] for t in sorted_vocab if len(t[0]) > 2]
    
    print(f"Learned {len(new_tokens)} multi-character tokens.")
    print("Top 50 examples:", new_tokens[-50:])
    
    # Load Qwen tokenizer
    qwen_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B", trust_remote_code=True)
    initial_size = len(qwen_tokenizer)
    
    # Filter tokens that are already efficiently tokenized
    # Wait, we can just add all of them, the tokenizer will handle duplicates
    added = qwen_tokenizer.add_tokens(new_tokens)
    print(f"Added {added} new tokens to Qwen tokenizer.")
    print(f"New vocab size: {len(qwen_tokenizer)} (was {initial_size})")

    # Save the token list for train.py to use
    with open("svg_tokens.json", "w") as f:
        json.dump(new_tokens, f)
