import json
import re

class SimpleWhisperTokenizer:
    # Initializes the SimpleWhisperTokenizer from tokenizer files.
    def __init__(self, tokenizer_path):
        vocab_file = f"{tokenizer_path}/vocab.json"
        special_tokens_map_file = f"{tokenizer_path}/special_tokens_map.json"
        normalizer_file = f"{tokenizer_path}/normalizer.json"
        merges_file = f"{tokenizer_path}/merges.txt"

        # Load vocabulary (token to ID mapping)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = json.load(f)

        # Create reverse vocabulary mapping
        self.reverse_vocab = {}
        for token, token_id in self.vocab.items():
            if isinstance(token, str) and isinstance(token_id, int):
                self.reverse_vocab[token_id] = token

        # Load special tokens map
        with open(special_tokens_map_file, 'r', encoding='utf-8') as f:
            special_tokens_map = json.load(f)
        self.special_tokens_ids = set()
        for token_type, token_value in special_tokens_map.items():
            if isinstance(token_value, str):
                token_id = self.vocab.get(token_value)
                if isinstance(token_id, int):
                    self.special_tokens_ids.add(token_id)
            elif isinstance(token_value, dict) and 'content' in token_value:
                content = token_value['content']
                if isinstance(content, str):
                    token_id = self.vocab.get(content)
                    if isinstance(token_id, int):
                        self.special_tokens_ids.add(token_id)

        # Load normalizer configuration
        try:
            with open(normalizer_file, 'r', encoding='utf-8') as f:
                self.normalizer_config = json.load(f)
        except FileNotFoundError:
            self.normalizer_config = None
            print(f"Warning: {normalizer_file} not found. Normalization will be basic.")

        # Load BPE merges and process them 
        try:
            with open(merges_file, 'r', encoding='utf-8') as f:
                merges_raw = f.readlines()
            # Preprocess merges to remove spaces and convert to tuples
            self.bpe_merges = [tuple(line.strip().split()) for line in merges_raw[1:] if line.strip()]

            # Create a rank mapping for efficient BPE reverse application
            self.bpe_ranks = {merge: i for i, merge in enumerate(self.bpe_merges)}
        except FileNotFoundError:
            print(f"Warning: {merges_file} not found. Assuming non-BPE tokenization.")
            self.bpe_merges = None
            self.bpe_ranks = {} # Initialize even if merges are not loaded

    # Detokenizes a list of token IDs back into text, handling BPE merges.
    def decode(self, token_ids, skip_special_tokens=True):
        detokenized_tokens = []
        for token_id in token_ids:
            if token_id in self.reverse_vocab:
                token = self.reverse_vocab[token_id]
                if skip_special_tokens and token_id in self.special_tokens_ids:
                    continue
                detokenized_tokens.append(token)
            else:
                detokenized_tokens.append(f"[UNK_ID:{token_id}]")

        # 1. Handle spaces - replace 'Ġ' with a space
        text = "".join(detokenized_tokens).replace("Ġ", " ").strip()


        # 2. Apply BPE reverse merges (if merges are loaded)
        if self.bpe_merges:
            def get_pairs(word):
                pairs = set()
                prev_char = word[0]
                for char in word[1:]:
                    pairs.add((prev_char, char))
                    prev_char = char
                return pairs

            def bpe_decode(token):
                word = tuple(token)
                while True:
                    pairs = get_pairs(word)
                    if not pairs:
                        break
                    # Find the pair with the lowest rank
                    min_rank = float('inf')
                    best_pair = None
                    for pair in pairs:
                        rank = self.bpe_ranks.get(pair, float('inf'))
                        if rank < min_rank:
                            min_rank = rank
                            best_pair = pair
                    if best_pair is None or min_rank == float('inf'):
                        break
                    # Apply the merge
                    first, second = best_pair
                    new_word = []
                    i = 0
                    while i < len(word):
                        try:
                            j = word.index(first, i) # Find the first occurrence of 'first' from index i
                            new_word.extend(word[i:j]) # Add parts before the pair
                            if j < len(word) - 1 and word[j+1] == second: # Check if the next symbol is 'second'
                                new_word.append(first + second) # Merge the pair
                                i = j + 2 # Skip both symbols of the pair
                            else:
                                new_word.append(first) # If not a pair, add 'first'
                                i = j + 1 # Move to the next symbol
                        except ValueError: # If 'first' is not found from index i
                            new_word.extend(word[i:]) # Add the rest of the word
                            break # Exit the loop
                    word = tuple(new_word) # Update word for the next iteration
                return "".join(word)


            # Split the text by spaces and apply BPE decode to each token
            decoded_words = [bpe_decode(word) for word in text.split(' ')]
            text = " ".join(decoded_words)


        # Basic cleaning - remove extra spaces and trim
        text = re.sub(r'\s+', ' ', text).strip()
        return text