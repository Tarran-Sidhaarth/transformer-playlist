from datasets import load_dataset

PATH= ""
dataset = load_dataset("ccdv/pubmed-summarization")

dataset.save_to_disk(f"{PATH}")
print("SUCCESS")