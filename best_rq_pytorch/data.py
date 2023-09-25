from pathlib import Path
from functools import wraps

from beartype import beartype
from beartype.typing import Optional, Tuple
from beartype.door import is_bearable

import torch
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader

import torchaudio

from einops import rearrange

def exists(val):
    return val is not None

class AudioDataset(Dataset):
    def __init__(
        self,
        folder,
        max_length_in_seconds: Optional[int] = None,
        pad_to_max_length = True
    ):
        super().__init__()
        path = Path(folder)
        assert path.exists(), 'folder does not exist'

        files = list(path.glob('**/*.flac'))
        assert len(files) > 0, 'no files found'

        self.files = files
        self.max_length = (max_length_in_seconds * 24_000) if exists(max_length_in_seconds) else None
        self.pad_to_max_length = pad_to_max_length
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        file = self.files[idx]
        
        # this assumes 24kHz single-channel audio

        wave, _ = torchaudio.load(file)
        wave = rearrange(wave, '1 n -> n')
        
        if exists(self.max_length):
            wave_length = wave.shape[0]
            
            if wave_length > self.max_length:
                # take a random crop to the max length
                start = torch.randint(0, wave_length - self.max_length, (1, )).item()
                wave = wave[start:start + self.max_length]
            elif self.pad_to_max_length:
                # work around variable length sequences recomputing the graph on mps
                wave = F.pad(wave, (0, self.max_length - wave_length))

        return wave
    
# data loader utilities

def collate_one_or_multiple_tensors(fn):
    @wraps(fn)
    def inner(data):
        is_one_data = not isinstance(data[0], tuple)

        if is_one_data:
            data = fn(data)
            return (data,)

        outputs = []
        for datum in zip(*data):
            if is_bearable(datum, Tuple[str, ...]):
                output = list(datum)
            else:
                output = fn(datum)

            outputs.append(output)

        return tuple(outputs)

    return inner

@collate_one_or_multiple_tensors
def curtail_to_shortest_collate(data):
    min_len = min(*[datum.shape[0] for datum in data])
    data = [datum[:min_len] for datum in data]
    return torch.stack(data)

@collate_one_or_multiple_tensors
def pad_to_longest_fn(data):
    return pad_sequence(data, batch_first = True)
    
def get_dataloader(ds, pad_to_longest = True, **kwargs):
    collate_fn = pad_to_longest_fn if pad_to_longest else curtail_to_shortest_collate
    return DataLoader(ds, collate_fn = collate_fn, **kwargs)
