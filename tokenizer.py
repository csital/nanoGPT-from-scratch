
from data import download_file_from_url


SHAKESPEARE_DATASET_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


class Tokenizer:
    """Tokenizer for datasets.

    Basic character level tokenizer.
    #TODO: switch to subword tokenizer for better performance.

    """

    def __init__(self, chars) -> None:
        self.chars = chars
        self.stoi = self.make_stoi()
        self.itos = self.make_itos()

    def make_stoi(self):
        # create str to int mapping
        return {ch: i for i, ch in enumerate(self.chars)}

    def make_itos(self):
        # create int to str mapping
        return {i: ch for i, ch in enumerate(self.chars)}

    def encode(self, s: str):
        """Takes a str and outputs a list of integers."""
        return [self.stoi[c] for c in s]

    def decode(self, l: list[int]):
        """Takes a list of integers and outputs a str."""
        return "".join([self.itos[i] for i in l])


if __name__ == "__main__":
    data_filepath = download_file_from_url(SHAKESPEARE_DATASET_URL)
    data = open(data_filepath, "r").read()
    
    t = Tokenizer(chars=data)
    print(t.encode("Hello"))
    print(t.decode(t.encode("Hello")))