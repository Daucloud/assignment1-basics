import os
import regex as re
import multiprocessing
from collections import Counter

from .pretokenization_example import find_chunk_boundaries

def count_single_chunk(args):
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    input_path, special_tokens, start, end=args
    with open(input_path, 'rb') as f:
        f.seek(start)
        raw_data = f.read(end - start).decode("utf-8")

        chunks=re.split("|".join(re.escape(token) for token in special_tokens),raw_data)
        ret=Counter()
        for c in chunks:
            aa=False
            ret.update([tuple(bytes([a]) for a in match.group().encode()) for match in re.finditer(PAT, c)])
    return ret


def pre_tokenize(input_path,num_processes,special_tokens):
    with open(input_path,'rb') as f:
        chunk_boundaries=find_chunk_boundaries(f,num_processes,"<|endoftext|>".encode("utf-8"))
    tasks=[(input_path,special_tokens,i,j) for i,j in zip(chunk_boundaries[:-1],chunk_boundaries[1:])]
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        chunk_counters=pool.map(count_single_chunk, tasks)
    
    total_counter=Counter()
    for chunk_counter in chunk_counters:
        total_counter.update(chunk_counter)

    return total_counter

def run_train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    num_processes: int=4
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # initialization
    vocab={i:bytes([i]) for i in range(256)}
    for special_token in special_tokens:
        vocab[len(vocab)]=special_token.encode()
    merges=[]
    pre_tokenization_counter=pre_tokenize(input_path,num_processes,special_tokens)
    pair_counter=Counter()
    pair_to_string={}
    for k,v in pre_tokenization_counter.items():
        for pair in zip(k[:-1],k[1:]):
            pair_counter[pair]+=v
            if pair not in pair_to_string:
                pair_to_string[pair]=set()
            pair_to_string[pair].add(k)
    
    # merge loop
    while len(vocab)<vocab_size and len(pair_counter)>1:
        # find max pair
        max_pair,_=max(pair_counter.items(),key=lambda x: (x[1],x[0][0],x[0][1]))
        merged_pair=max_pair[0]+max_pair[1]
        vocab[len(vocab)]=merged_pair
        merges.append(max_pair)

        # update structures
        for old_string in list(pair_to_string[max_pair]):
            string_count=pre_tokenization_counter[old_string]
            old_pair_counter=Counter()
            for pair in zip(old_string[:-1],old_string[1:]):
                old_pair_counter.update({pair:string_count})
            new_string=[]
            i=0
            while i<len(old_string):
                if i<len(old_string)-1 and old_string[i]+old_string[i+1]==merged_pair:
                    new_string.append(merged_pair)
                    i+=2
                else:
                    new_string.append(old_string[i])
                    i+=1
            new_pair_counter=Counter()
            new_string=tuple(new_string)
            for pair in zip(new_string[:-1],new_string[1:]):
                new_pair_counter.update({pair:string_count})
            for pair, count in old_pair_counter.items():
                pair_counter[pair]-=count
            for pair, count in new_pair_counter.items():
                pair_counter[pair]+=count
            pre_tokenization_counter[new_string]=pre_tokenization_counter[old_string]
            del pre_tokenization_counter[old_string]
            pair_to_string[max_pair].remove(old_string)
            if not pair_to_string[max_pair]:
                del pair_to_string[max_pair]

            for pair in zip(new_string[:-1],new_string[1:]):
                if pair not in pair_to_string:
                    pair_to_string[pair]=set()
                pair_to_string[pair].add(new_string)
    
    return vocab,merges