from datasets import load_from_disk

PATH = "/home/tarran/compsci/tuts/encoder-decoder/dataset"

dataset = load_from_disk(PATH)

print(dataset["train"][0]["abstract"])
