from cs336_basics.tokenizer.train_bpe import train_bpe
import modal
import time, resource
import pickle, os

app = modal.App("train-bpe-tinystories-and-owt")
volume = modal.Volume.from_name("cs336-data")
image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("regex")
    .add_local_python_source("cs336_basics")
)

output_directory = "/storage/data/results"

@app.function(image=image, cpu=(4, 4), memory=(16384), timeout=24 * 60 * 60, volumes={"/storage": volume})
def train_bpe_database(file_path, special_tokens, vocab_size, train_name):

    start = time.perf_counter()

    vocab, merges = train_bpe(file_path, vocab_size, special_tokens)

    elapsed = time.perf_counter() - start
    
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = usage.ru_maxrss

    # Longest token length in bytes 
    longest_token_id, longest_token = max(
        vocab.items(), 
        key=lambda item: len(item[1])
    )

    print(f"Training {train_name} BPE tokenizer took {elapsed:.2f} seconds")
    print(f"Peak memory usage for {train_name} BPE tokenizer: {peak_rss / 1024:.2f} MB")

    print("===============================================")

    print(f"Longest token in {train_name} BPE tokenizer: {longest_token.decode('utf-8', errors='ignore')} (ID: {longest_token_id}, Length: {len(longest_token)} bytes)")


    result_directory = output_directory + f"/{train_name}_results"
    os.makedirs(result_directory, exist_ok=True)

    with open(result_directory + "/vocab_" + train_name + ".pkl", "wb") as file:
        pickle.dump(vocab, file)

    with open(result_directory + "/merges_" + train_name + ".pkl", "wb") as file:
        pickle.dump(merges, file)

    volume.commit()
    return vocab, merges

@app.local_entrypoint()
def main():
    # Train BPE on TinyStories
    # train_bpe_database.spawn(
    #     file_path="/storage/data/TinyStoriesV2-GPT4-train.txt",
    #     special_tokens=["<|endoftext|>"],
    #     vocab_size=10000,
    #     train_name="TinyStories"
    # )

    # Train BPE on owt
    train_bpe_database.spawn(
        file_path="/storage/data/owt_train.txt",
        special_tokens=["<|endoftext|>"],
        vocab_size=32000,
        train_name="owt"
    )