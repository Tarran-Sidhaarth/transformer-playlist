class BytePairEncoder:
    def __init__(self, num_merges: int = 5):
        self.num_merges = num_merges
        self.merges = {} # (token_id,token_id) -> merged_token_id
        self.vocab = {i: bytes([i]) for i in range(256)}


    def train(self, text: str):
        sequence = list(text.encode("utf-8"))
        next_id = 256
        for _ in range(self.num_merges):
            counts = {}
            # create the pair sequences

            for i in range(len(sequence)-1):
                pair = (sequence[i],sequence[i+1])
                counts[pair] = counts.get(pair,0)+1
            
            if not counts:
                break

            best_pair = max(counts, key=counts.get) # type: ignore

            if counts[best_pair] < 2:
                break
        
            # record the merge
            self.merges[best_pair] = next_id
            self.vocab[next_id] = self.vocab[best_pair[0]]+self.vocab[best_pair[1]]

            #apply
            new_sequence = []
            i = 0
            while i< len(sequence):
                if i<len(sequence)-1 and (sequence[i],sequence[i+1]) == best_pair:
                    new_sequence.append(next_id)
                    i+=2
                else:
                    new_sequence.append(sequence[i])
                    i+=1
            
            sequence = new_sequence
            next_id +=1

        return

    def encode(self, text: str) -> list:

        sequence = list(text.encode("utf-8"))
        for pair, merged_id in self.merges.items():
            new_sequence = []
            i = 0
            while i< len(sequence):
                if i<len(sequence)-1 and (sequence[i],sequence[i+1]) == pair:
                    new_sequence.append(merged_id)
                    i+=2
                else:
                    new_sequence.append(sequence[i])
                    i+=1
            sequence = new_sequence

        return sequence

    def decode(self, token_ids: list) -> str:
        return b"".join(self.vocab[t] for t in token_ids).decode("utf-8")



# --- Example Usage ---
if __name__ == "__main__":
    text = "Betty bought some butter but the butter was bitter 🧈"

    encoder = BytePairEncoder(num_merges=5)
    encoder.train(text)

    encoded = encoder.encode(text)
    decoded = encoder.decode(encoded)

    print(f"Original  ({len(text.encode('utf-8'))} bytes): {text}")
    print(f"Encoded   ({len(encoded)} tokens): {encoded}")
    print(f"Decoded   : {decoded}")
    print(f"Roundtrip : {decoded == text}")
    print(f"Vocab size: {len(encoder.vocab)}  (256 base + {len(encoder.merges)} merges)")

    print("\nLearned Merges:")
    for (a, b), merged_id in encoder.merges.items():
        print(f"  {encoder.vocab[a]} + {encoder.vocab[b]} -> token {merged_id} ({encoder.vocab[merged_id]})")