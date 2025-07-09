PRETOKENIZE_PAT=r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def get_new_string(old_string:list[bytes], merged_pair:bytes)->list[bytes]:
    new_string=[]
    i=0
    while i<len(old_string):
        if i<len(old_string)-1 and old_string[i]+old_string[i+1]==merged_pair:
            new_string.append(merged_pair)
            i+=2
        else:
            new_string.append(old_string[i])
            i+=1
    return new_string