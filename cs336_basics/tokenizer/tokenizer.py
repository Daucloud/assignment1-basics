from typing import Iterable, Iterator
import regex as re
import ujson as json
from pydantic import BaseModel
from ..utils import PRETOKENIZE_PAT, get_new_string

class Tokenizer(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    merges_rank: dict[tuple[bytes, bytes], int] | None = None
    special_tokens: list[str] | None = None
    token_to_id: dict[bytes, int] | None = None

    def __init__(self, vocab, merges, special_tokens=[], **kwargs):
        super().__init__(vocab=vocab, merges=merges, special_tokens=special_tokens, **kwargs)
        for i in range(2**8):
            vocab[i]=bytes([i])
        self.token_to_id={token:id for id, token in self.vocab.items()}
        self.merges_rank={merge: i for i, merge in enumerate(self.merges)}
        if self.special_tokens:
            self.special_tokens.sort(key=len, reverse=True)
    
    @classmethod
    def from_files(cls, vocab_filepath, merges_file_path, special_tokens=None):

        def decode_space(token):
            return token.replace('\u0120', ' ')

        with open(vocab_filepath, 'r') as f:
            vocab_str=json.load(f)
            vocab={int(id):decode_space(token).encode() for token, id in vocab_str.items()}
            for i in range(2**8):
                vocab[i]=bytes([i])
        merges=[]
        with open(merges_file_path, 'r') as f:
            for line in f:
                line=line.strip('\n')
                if not line:
                    continue
                token1, token2=line.split(' ')
                merges.append((decode_space(token1).encode(),decode_space(token2).encode()))
        return cls(vocab=vocab, merges=merges, special_tokens=special_tokens)
    
    @staticmethod
    def _pre_tokenize(text: str) -> list[tuple[bytes]]:
        return [tuple(bytes([a]) for a in match.group().encode()) for match in re.finditer(PRETOKENIZE_PAT, text)]

    def encode(self, text: str) -> list[int]:
        if self.special_tokens:
            SPECIAL_TOKEN_PAT="("+"|".join(re.escape(token) for token in self.special_tokens)+")"
            texts=re.split(SPECIAL_TOKEN_PAT,text)
        else:
            texts=[text]
        ret=[]
        for text in texts:
            if self.special_tokens and text in self.special_tokens:
                ret.append(self.token_to_id[text.encode()])
                continue
            pre_tokenized_text=self._pre_tokenize(text)
            for text_chunk in pre_tokenized_text:
                while len(text_chunk)>1:
                    pairs=set()
                    for merge in zip(text_chunk[:-1],text_chunk[1:]):
                        pairs.add(merge)
                    min_pair=min(
                        (pair for pair in pairs if pair in self.merges_rank),
                        key=lambda pair: self.merges_rank[pair],
                        default=None
                    )
                    if min_pair is None:
                        break
                    text_chunk=get_new_string(text_chunk, min_pair[0]+min_pair[1])
                ret.extend([self.token_to_id[token] for token in text_chunk])
        return ret

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for line in iterable:
            if self.special_tokens:
                SPECIAL_TOKEN_PAT="("+"|".join(re.escape(token) for token in self.special_tokens)+")"
                texts=re.split(SPECIAL_TOKEN_PAT,line)
            else:
                texts=[line]
            for text in texts:
                if self.special_tokens and text in self.special_tokens:
                    yield self.token_to_id[text.encode()]
                    continue
                pre_tokenized_text=self._pre_tokenize(text)
                for text_chunk in pre_tokenized_text:
                    while len(text_chunk)>1:
                        pairs=set()
                        for merge in zip(text_chunk[:-1],text_chunk[1:]):
                            pairs.add(merge)
                        min_pair=min(
                            (pair for pair in pairs if pair in self.merges_rank),
                            key=lambda pair: self.merges_rank[pair],
                            default=None
                        )
                        if min_pair is None:
                            break
                        text_chunk=get_new_string(text_chunk, min_pair[0]+min_pair[1])
                    for token in text_chunk:
                        yield self.token_to_id[token]
    

    def decode(self, ids: list[int])->str:
        return b''.join([self.vocab[id] for id in ids]).decode(errors='replace')