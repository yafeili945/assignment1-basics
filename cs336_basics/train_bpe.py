from .pretokenization_example import find_chunk_boundaries
import regex as re
import multiprocessing
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def initialize_vocab(special_tokens: list[str]):
    """Initialize the vocabulary with single-byte tokens and special tokens."""

    vocab: dict[int, bytes] = {}
    for i in range(256):
        vocab[i] = bytes([i])
    for token in special_tokens:
        vocab[len(vocab)] = token.encode("utf-8")
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

    pretoken_sequence_to_counts: dict[tuple[bytes, ...], int] = {}
    for key in pretoken_counts.keys():
        encoded_key = key.encode("utf-8")
        encoded_key_sequence = tuple(encoded_key[i: i + 1] for i in range(len(encoded_key)))
        cnt = pretoken_counts[key]
        pretoken_sequence_to_counts[encoded_key_sequence] = cnt
    return pretoken_sequence_to_counts

def count_token_pairs(pretoken_sequence_to_counts: dict[tuple[bytes, ...], int]):
    """Count the occurrences of each pair of tokens in the pretoken counts."""
    pair_counts: dict[tuple[bytes, bytes], int] = {}
    pair_to_sequence: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]] = {}

    for pretoken, count in pretoken_sequence_to_counts.items():
        len = pretoken.__len__()
        for i in range(len - 1):
            pair = (pretoken[i], pretoken[i + 1])
            pair_counts[pair] = pair_counts.get(pair, 0) + count
            pair_to_sequence.setdefault(pair, set()).add(pretoken)

    return pair_counts, pair_to_sequence
def merge_pretoken_pairs(pair_counts: dict[tuple[bytes, bytes], int], pair_to_sequence: dict[tuple[bytes, bytes], set[tuple[bytes, ...]]], pretoken_sequence_to_counts: dict[tuple[bytes, ...], int], vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_size: int):
    """Merge the specified pair of tokens in the pretoken counts."""
    
    pair_to_merge = vocab_size - vocab.__len__()
    while pair_to_merge > 0:
        if not pair_counts:
            break

        # Find the most frequent pair
        most_frequent_pair = max(pair_counts, key=lambda key: (pair_counts[key], key))

        idx = vocab.__len__()
        vocab[idx] = most_frequent_pair[0] + most_frequent_pair[1]
        merges.append(most_frequent_pair)

        # Update the pretoken counts by merging the most frequent pair
        tmp_sequence_to_counts = pretoken_sequence_to_counts.copy()
        for sequence in pair_to_sequence[most_frequent_pair]:
            sequence_count = pretoken_sequence_to_counts[sequence]
            new_sequence = []
            i = 0
            while i < len(sequence):
                if i < len(sequence) - 1 and (sequence[i], sequence[i + 1]) == most_frequent_pair:
                    new_sequence.append(most_frequent_pair[0] + most_frequent_pair[1])
                    i += 2
                else:
                    new_sequence.append(sequence[i])
                    i += 1
            tmp_sequence_to_counts[tuple(new_sequence)] = tmp_sequence_to_counts.get(tuple(new_sequence), 0) + sequence_count
            tmp_sequence_to_counts.pop(sequence, None)

        pretoken_sequence_to_counts = tmp_sequence_to_counts
        pair_counts, pair_to_sequence = count_token_pairs(pretoken_sequence_to_counts)
        
        pair_to_merge -= 1
def get_special_tokens_regex(special_tokens: list[str]) -> str:
    """Construct a regex pattern to match special tokens."""
    escaped_tokens = [re.escape(token) for token in special_tokens]
    return "|".join(escaped_tokens)

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
            special_tokens_pattern = get_special_tokens_regex(special_tokens)
            parts = re.split(special_tokens_pattern, chunk)
            for part in parts:

                matches = re.finditer(PAT, part)

                local_counts = count_pretokens(matches)
                update_pretoken_counts(pretoken_counts, local_counts)
    
    pretoken_sequence_to_counts = encode_pretoken_counts(pretoken_counts)
    pair_counts, pair_to_sequence = count_token_pairs(pretoken_sequence_to_counts)

    merge_pretoken_pairs(pair_counts, pair_to_sequence, pretoken_sequence_to_counts, vocab, merges, vocab_size)
    return vocab, merges
