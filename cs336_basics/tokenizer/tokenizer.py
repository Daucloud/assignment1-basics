import ujson as json
from pydantic import BaseModel

class Tokenizer(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    token_to_id: dict[bytes, int] | None = None

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.token_to_id={token:id for id, token in self.vocab.items()}
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_file_path, special_tokens=None):

        def decode_space(token):
            return token.replace('\u0120', ' ')

        with open(vocab_filepath, 'r') as f:
            vocab_str=json.load(f)
            vocab={int(id):decode_space(token).encode() for token, id in vocab_str}
        merges=[]
        with open(merges_file_path, 'r') as f:
            for line in f:
                token1, token2=line.strip('\n').split(' ')
                merges.append((decode_space(token1).encode(),decode_space(token2).encode()))
        return cls(Tokenizer(vocab, merges, special_tokens))