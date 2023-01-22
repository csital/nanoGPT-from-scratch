import torch

from data import download_file_from_url

NAMES_DATASET_URL = (
    "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt"
)


def load_data() -> list[str]:
    """Loads the data."""
    data_filepath = download_file_from_url(NAMES_DATASET_URL)
    data = open(data_filepath, "r").read().splitlines()

    print(f"# Loaded names data set with {len(data):,.0f} entries")
    return data


def make_bigram_dict(data: list[str]) -> dict:
    """Extract bigrams for words in data set.

    Each word gives 3 types of information:
    - Which character likely follows another character
    - Which character is likely to start the word
    - Which character is likely to end the word
    """
    print("# Making bigram...")
    # placeholder for bigram counts
    b = {}

    for word in data:
        # prepend start char and append end char
        word_chars = ["<S>"] + list(word) + ["<E>"]

        for ch1, ch2 in zip(word_chars, word_chars[1:]):
            bigram = (ch1, ch2)

            # get bigram count from dict if it exists
            # otherwise add bigram and increment the counter
            b[bigram] = b.get(bigram, 0) + 1

    return b

def create_lut_char_to_int():
    # convert counts to array for easier manipulation
    unique_char_count = 28  # 26 letters in Alphabet + 2 for start and end chars

    N = torch.zeros((unique_char_count, unique_char_count), dtype=torch.int32)

    return N


if __name__ == "__main__":
    b = make_bigram_dict(load_data())
    print(f"Top 5 bigram counts: {sorted(b.items(), key = lambda kv: ~kv[1])[:5]}")
