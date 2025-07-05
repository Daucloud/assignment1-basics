from typing import Optional
from pydantic import BaseModel

class Tokenizer(BaseModel):
    vocab: dict[int, bytes]
    merges: list[tuple[bytes, bytes]]
    special_tokens: list[str] | None = None
    token_to_id: dict[bytes, int]

    def __init__(self, **kwargs):
        super.__init__(**kwargs)
        self.token_to_id={token:id for id, token in self.vocab}
    
    def from_files