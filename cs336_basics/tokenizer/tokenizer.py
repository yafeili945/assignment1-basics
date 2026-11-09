from collections.abc import Iterable, Iterator
import pickle
import regex as re

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_special_tokens_regex(special_tokens: list[str]) -> str:
    """Construct a regex pattern to match special tokens."""
    if not special_tokens:
        return None
    ordered_tokens = sorted(
        special_tokens,
        key=len,
        reverse=True,
    )
    escaped_tokens = [re.escape(token) for token in ordered_tokens]
    return "(" + "|".join(escaped_tokens) + ")"

class Tokenizer:
    def __init__(self, vocab, merges, special_tokens=None):
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []

        for special_token in self.special_tokens:
            special_bytes = special_token.encode("utf-8")
            if special_bytes not in vocab.values():
                vocab[len(vocab)] = special_bytes

        self.token_to_id = {
            token_byte : token_id
            for token_id, token_byte in vocab.items()
        }
        self.merges_to_rank = {
            merges : rank
            for rank, merges in enumerate(merges)
        }

        self.special_token_to_id = {
            special_token: self.token_to_id[special_token.encode("utf-8")]
            for special_token in self.special_tokens
        }

    @classmethod
    def from_files(cls, vocab_filepath, merges_filepath, special_tokens=None):
       
        with open(vocab_filepath, "rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with open(merges_filepath, "rb") as merges_file:
            merges = pickle.load(merges_file)

        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:

        res = []
        special_tokens_pattern = get_special_tokens_regex(self.special_tokens)

        if special_tokens_pattern is None:
            parts = [text]
        else:
            parts = re.split(special_tokens_pattern, text)

        for part in parts:
            if not part:
                continue
            if part in self.special_token_to_id:
                res.append(self.special_token_to_id[part])
                continue

            for match in re.finditer(PAT, part):
                res.extend(self.encode_one_word(match.group(0)))
            
        return res

    def encode_one_word(self, word: str) -> list[int]:

        wordByte = word.encode("utf-8")
        tokens = tuple(wordByte[i : i + 1] for i in range(len(wordByte)))

        while (len(tokens) >= 2):
            best_pair = None
            best_rank = float("inf")

            for i in range(len(tokens) - 1):
                cur_pair = (tokens[i], tokens[i + 1])
                cur_rank = self.merges_to_rank.get(cur_pair)

                if cur_rank is not None and cur_rank < best_rank:
                    best_rank = cur_rank
                    best_pair = cur_pair

            if best_pair is None:
                break

            new_tokens = []
            i = 0
            while i < len(tokens):
                if ((i + 1 < len(tokens)) and (tokens[i], tokens[i + 1]) == best_pair):
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1

            tokens = tuple(new_tokens)

        return [self.token_to_id[i] for i in tokens]
        
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text_chunk in iterable:
            yield from self.encode(text_chunk)

    def decode(self, ids: list[int]) -> str:

        res = b"".join(self.vocab[i] for i in ids)
        return res.decode("utf-8", errors="replace")