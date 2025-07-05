import os
import pickle
import click
import regex as re
import multiprocessing
from pathlib import Path
from typing import BinaryIO
from collections import Counter

def find_chunk_boundaries(
    file: BinaryIO, 
    desired_num_chunks: int, 
    split_special_token: bytes
) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), (
        "Must represent special token as a bytestring"
    )

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

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

@click.command()
@click.argument('input_path')
@click.argument('vocab_size', type=int)
@click.option('--special_tokens',type=list[str], default=['<|endoftext|>'])
@click.option('--num_processes',type=int, default=4)
@click.option('--save_path',type=str,default='bpe_results')
def train_bpe(input_path, vocab_size, special_tokens, num_processes, save_path):
    vocab, merges=run_train_bpe(
        input_path=input_path,
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        num_processes=num_processes
    )
    Path(save_path).mkdir(exist_ok=True)
    with open(Path(save_path)/'vocab.pkl','wb') as f:
        pickle.dump(vocab, f)
    with open(Path(save_path)/'merges.pkl','wb') as f:
        pickle.dump(merges, f)

if __name__=='__main__':
    train_bpe()