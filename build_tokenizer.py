import json
from tokenizers import Tokenizer, models, trainers, pre_tokenizers
from transformers import AutoTokenizer

def train_custom_bpe(jsonl_file, vocab_size=16384):
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    # Use whitespace pre-tokenizer to allow BPE to form within words/numbers
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Add explicit control tokens as special tokens so they are kept atomic
    special_tokens = [
        "<unk>", "<|im_start|>", "<|im_end|>", 
        "<svg", "</svg>", "<path", "/>", "<g>", "</g>", "<rect",
        "d=", "fill=", "stroke=", "viewBox=", "transform=",
        "M", "L", "C", "S", "Q", "T", "Z", "m", "l", "c", "s", "q", "t", "z"
    ]
    
    trainer = trainers.BpeTrainer(vocab_size=vocab_size, special_tokens=special_tokens)
    
    def get_training_corpus():
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    data = json.loads(line)
                    caption = data.get("caption", "")
                    svg = data.get("svg", "")
                    yield f"<|im_start|>user\nGenerate an SVG for: {caption}<|im_end|>\n<|im_start|>assistant\n{svg}<|im_end|>"
                except: pass

    tokenizer.train_from_iterator(get_training_corpus(), trainer=trainer)
    return tokenizer

if __name__ == "__main__":
    print("Training BPE on dataset...")
    custom_tokenizer = train_custom_bpe("dataset_raw.jsonl", vocab_size=16384)
    
    vocab = custom_tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    
    # Extract the tokens (skip the first few which are single characters/specials)
    new_tokens = [t[0] for t in sorted_vocab if len(t[0]) > 1]
    
    print(f"Learned {len(new_tokens)} multi-character tokens.")
    
    with open("svg_vocab.json", "w") as f:
        json.dump(new_tokens, f)

    # Test compression with Qwen + these tokens
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-0.5B', trust_remote_code=True)
    with open('dataset_raw.jsonl', 'r') as f:
        for line in f:
            sample_svg = json.loads(line)["svg"]
            break
            
    orig_tokens = tok.tokenize(sample_svg)
    print("Original tokens:", len(orig_tokens))
    
    from transformers import AddedToken
    added_tokens = [AddedToken(t, rstrip=False, lstrip=False, single_word=False, normalized=False) for t in new_tokens]
    tok.add_tokens(added_tokens)
    
    new_tokens_count = len(tok.tokenize(sample_svg))
    print("New tokens:", new_tokens_count)
    print("Compression ratio:", len(orig_tokens)/new_tokens_count)

