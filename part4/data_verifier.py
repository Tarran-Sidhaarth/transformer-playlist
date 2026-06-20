from datasets import load_from_disk

PATH = "" # insert your desired path

dataset = load_from_disk(PATH)

print(dataset["train"][0]["abstract"])
