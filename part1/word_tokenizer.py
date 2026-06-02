test_string = "The quick brown fox jumps over the lazy dog"

def get_vocab_list(string: str) -> list:
    """
    Returns word level vocabulary list.

    Returns:
        List of all unique word level vocabulary.
    """
    return list(dict.fromkeys(string.lower().split()))

def get_word_token_index_mapper(vocab_list: list)->dict:
    """
    Returns key-value pair where the key is the token and the value is the corresponding index.

    Args:
        vocab_list:     unique vocabulary list.

    Returns:
        Map of the token to index.
    """
    mapper = {}
    index = 0
    for element in vocab_list:
        if element not in mapper:
            mapper[element] = index
            index+=1

    return mapper

def get_index_to_token_mapper(token_mapper: dict) -> dict:
    """
    Returns key-value pair where the key is the token and the value is the corresponding index.

    Args:
        token_mapper:     the token to index mapper.

    Returns:
        Map of the index to token.
    """

    mapper = {}

    for key, value in token_mapper.items():
        mapper[value] = key
    return mapper

def encode(input_tokens: list, token_mapper: dict) -> list:
    """
    Returns list of encoded tokens

    Args:
        input_tokens:     the list of tokens to encode
        token_mapper:     the token to index mapper.

    Returns:
        Encoded tokens
    """
    encoded_list = []
    for element in input_tokens:
        encoded_list.append(token_mapper[element])

    return encoded_list

def decode(encoded_tokens: list, index_mapper: dict) -> list:
    """
    Returns list of decoded tokens

    Args:
        encoded_tokens:     the list of encoded token indices
        index_mapper:       the index to token mapper.

    Returns:
        Decoded tokens
    """
    return [index_mapper[element] for element in encoded_tokens]

vocab_list = get_vocab_list(test_string)
token_mapper = get_word_token_index_mapper(vocab_list)
index_mapper = get_index_to_token_mapper(token_mapper)

# print(f"Vocab list: {vocab_list}\nToken Mapper: {token_mapper}\nIndex Mapper: {index_mapper}")

print(f"Encoded: {encode(test_string.lower().split(),token_mapper)}")
print(f"Decoded: {decode(encode(test_string.lower().split(),token_mapper),index_mapper)}")