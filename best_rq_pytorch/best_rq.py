from pathlib import Path
from typing import Optional

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchaudio import transforms as T

from vector_quantize_pytorch.random_projection_quantizer import RandomProjectionQuantizer

from best_rq_pytorch.conformer import ConformerWrapper

from einops import rearrange

# utilities

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def get_mask_subset_prob(mask, prob, min_mask = 0):
    batch, seq, device = *mask.shape, mask.device
    num_to_mask = (mask.sum(dim = -1, keepdim = True) * prob).clamp(min = min_mask)
    logits = torch.rand((batch, seq), device = device)
    logits = logits.masked_fill(~mask, -1)

    randperm = logits.argsort(dim = -1).float()

    num_padding = (~mask).sum(dim = -1, keepdim = True)
    randperm -= num_padding

    subset_mask = randperm < num_to_mask
    subset_mask.masked_fill_(~mask, False)
    return subset_mask

def set_eos_id(t: Tensor, eos_id: int, pad_id: int):
    eos_indices = ((t == pad_id).cumsum(dim = -1) == 0).sum(dim = -1, keepdim = True).long()

    batch_range = torch.arange(t.shape[0], device = t.device, dtype = torch.long)
    batch_range = rearrange(batch_range, '... -> ... 1')

    t = F.pad(t, (0, 1), value = pad_id)
    t[batch_range, eos_indices] = eos_id
    return t

# BEST-RQ model

class BestRQ(nn.Module):
    def __init__(
        self,
        codebook_size: int,
        codebook_dim: int,
        *,
        conformer: ConformerWrapper,
        sample_rate: int = 24_000,
        n_mels: int = 80,
        win_length: int = 960,
        hop_length: Optional[int] = None,
        n_fft: Optional[int] = None,
    ):
        super().__init__()

        self.rpq = nn.Sequential(
            nn.LayerNorm(n_mels, elementwise_affine = True),
            RandomProjectionQuantizer(
                dim = n_mels,
                codebook_size = codebook_size,
                codebook_dim = codebook_dim,
                norm = False
            )
        )
        self.rpq.requires_grad = False

        hop_length = default(hop_length, win_length // 4)

        self.feature_extractor = nn.Sequential(
            T.MelSpectrogram(
                sample_rate = sample_rate,
                n_mels = n_mels,
                win_length = win_length,
                hop_length = hop_length,
                n_fft = default(n_fft, win_length)
            ),
            T.AmplitudeToDB()
        )
        self.feature_extractor.requires_grad = False

        self.conformer = conformer

        self.pad_id = 0
        self.eos_id = codebook_size + 1

    def load(self, path, strict = True):
        path = Path(path)
        assert path.exists()
        pkg = torch.load(str(path), map_location = 'cpu')
        try:
            self.load_state_dict(pkg['model'], strict = strict)
        except Exception:
            self.load_state_dict(pkg['encoder'], strict = strict)
        return pkg

    @torch.no_grad()
    def extract_features(self, x):
        if x.device.type == 'mps':
            # work around ComplexFloat being unavailable with mps
            return self.feature_extractor.cpu()(x.cpu()).to(x.device)
        else:
            return self.feature_extractor(x)

    def forward(
        self,
        x = None,
        labels = None,
        mask = None,
        return_labels = False,
        return_emb = False,
        return_layer_output: Optional[int] = None
    ):
        assert exists(x) or exists(labels), "either input or labels must be provided"

        if exists(x) and not exists(labels):
            with torch.no_grad():
                features = self.extract_features(x)
                # print(f"features: {features.shape} {features[0]}")

                # project labels from features

                features = rearrange(features, "b c n -> b n c")
                labels = self.rpq(features)
                # print(f"labels: {labels.shape} {labels[0, 0:100]}")

                # offset for the pad id

                labels = torch.where(labels != self.pad_id, labels + 1, self.pad_id)

        if return_labels:
            return labels

        if not exists(mask):
            mask = torch.cat([rearrange(x != self.pad_id, "n -> 1 n") for x in labels.unbind(dim = 0)], dim = 0)

        outputs = self.conformer(
            labels,
            mask = mask,
            return_layer_output = return_layer_output,
            return_emb = return_emb
        )

        return outputs
    
# pretraining task wrapper

class BestRQPretrainWrapper(nn.Module):
    def __init__(
        self,
        model: BestRQ,
        mask_prob: float = 0.6
    ):
        super().__init__()

        self.model = model
        self.mask_prob = mask_prob
        self.pad_id = self.model.pad_id

    def forward(self, x):
        # determine labels from the random projection quantizer

        labels = self.model(x, return_labels = True)

        # mask a subset of the labels for pretraining
        # TODO: mask sequences of n milliseconds/frames

        seq_mask = torch.cat([rearrange(x != self.pad_id, "n -> 1 n") for x in labels.unbind(dim = 0)], dim = 0)
        mask = get_mask_subset_prob(seq_mask, self.mask_prob)

        # predict and compute ce loss

        logits = self.model(labels = labels, mask = mask)
        logits = rearrange(logits, "b n c -> b c n")

        masked_labels = labels.masked_fill(~mask, self.pad_id)

        loss = F.cross_entropy(logits, masked_labels, ignore_index = self.pad_id)

        return loss, logits
