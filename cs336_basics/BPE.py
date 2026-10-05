from pretokenization_example import find_chunk_boundaries
import regex as re
import multiprocessing

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]) -> dict[str, int]:
    """
    Train a BPE tokenizer on the given input file."""

    vocab: dict[int, bytes] = initialize_vocab(special_tokens)
    merges: list[tuple[bytes, bytes]] = []
    pretoken_counts: dict[str, int] = {}

    with open(input_path, "rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        # The following is a serial implementation, but you can parallelize this
        # by sending each start/end pair to a set of processes.
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            f.seek(start)
            chunk = f.read(end - start).decode("utf-8", errors="ignore")
            parts = chunk.split("<|endoftext|>")
            for part in parts:

                matches = re.finditer(PAT, part)

                local_counts = count_pretokens(matches)
                update_pretoken_counts(pretoken_counts, local_counts)

    pretoken_counts_encoded = encode_pretoken_counts(pretoken_counts)
    pair_counts = count_token_pairs(pretoken_counts_encoded)
    # The following is a placeholder for the BPE merge operations.

    #
    return vocab, merges

def initialize_vocab(special_tokens: list[str]):
    """Initialize the vocabulary with single-byte tokens and special tokens."""

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    vocab[256] = b"<|endoftext|>"
    return vocab

def count_pretokens(matches) -> dict[str, int]:
    """Count the occurrences of each pretoken in the matches."""
    counts: dict[str, int] = {}
    for match in matches:
        pretoken = match.group(0)
        counts[pretoken] = counts.get(pretoken, 0) + 1
    return counts

def update_pretoken_counts(pretoken_counts: dict[str, int], local_counts: dict[str, int]):
    """Update the global pretoken counts with the local counts from a chunk."""
    for key in local_counts.keys():
        pretoken_counts[key] = pretoken_counts.get(key, 0) + local_counts[key]

def encode_pretoken_counts(pretoken_counts: dict[str, int]):
    """Encode the pretoken counts dictionary to use bytes as keys."""

    pretoken_counts_encoded: dict[tuple[bytes, ...], int] = {}
    for key in pretoken_counts.keys():
        encoded_key = key.encode("utf-8")
        encoded_key_sequence = tuple(encoded_key[i: i + 1] for i in range(len(encoded_key)))
        cnt = pretoken_counts[key]
        pretoken_counts_encoded[encoded_key_sequence] = cnt
    return pretoken_counts_encoded

def count_token_pairs(pretoken_counts_encoded: dict[tuple[bytes, ...], int]):
    """Count the occurrences of each pair of tokens in the pretoken counts."""

    pair_counts: dict[tuple[bytes, bytes], int] = {}
    for pretoken, count in pretoken_counts_encoded.items():
        len = pretoken.__len__()
        for i in range(len - 1):
            pair = (pretoken[i : i + 1], pretoken[i + 1 : i + 2])
            pair_counts[pair] = pair_counts.get(pair, 0) + count

    print(pair_counts)
    return pair_counts

def main():
    file_path = "/Users/felix/project/CS336/assignments/assignment1-basics/data/test.txt"
    train_bpe(file_path, vocab_size=10000, special_tokens=["<|endoftext|>"])

if __name__ == "__main__":
    main()
