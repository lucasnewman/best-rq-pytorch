import joblib
from pathlib import Path
from tqdm import tqdm

import torch

from einops import rearrange

from best_rq_pytorch.best_rq import BestRQ
from best_rq_pytorch.conformer import ConformerWrapper
from best_rq_pytorch.data import AudioDataset

from sklearn.cluster import MiniBatchKMeans

pretrained_checkpoint = "results/bestrq.XXXX.pt"
km_output_path = "results/bestrq_km1024.bin"
accelerator = "cuda"

# k-means utilities

def get_km_model(
    n_clusters,
    init,
    max_iter,
    batch_size,
    tol,
    max_no_improvement,
    n_init,
    reassignment_ratio,
):
    return MiniBatchKMeans(
        n_clusters = n_clusters,
        init = init,
        max_iter = max_iter,
        batch_size = batch_size,
        verbose = 1,
        compute_labels = False,
        tol = tol,
        max_no_improvement = max_no_improvement,
        init_size = None,
        n_init = n_init,
        reassignment_ratio = reassignment_ratio,
    )


# set up the encoder

brq = BestRQ(
    codebook_size = 1024,
    codebook_dim = 16,
    sample_rate = 24_000,
    n_mels = 80,
    win_length = 960,
    hop_length = 960 // 3,
    conformer = ConformerWrapper(
        num_tokens = 1024,
        conformer = dict(
            dim = 1024,
            depth = 24,
            heads = 16,
            conv_kernel_size = 5,
            ff_mult = 4,
            attn_dropout = 0.1,
            ff_dropout = 0.1,
            conv_dropout = 0.1,
            attn_flash = False
        )
    )
).to(accelerator)

brq.load(pretrained_checkpoint)

# load the dataset

dataset_folder = "..."
ds = AudioDataset(dataset_folder)

# collect activations for k-means

output_layer = 13

samples = 0
activations = []

for wave in tqdm(ds):
    with torch.no_grad():
        activation = brq(rearrange(wave, "n -> 1 n").to(accelerator), return_layer_output = output_layer)
        activations.append(activation.cpu())

activations = rearrange(torch.cat(activations, dim = 1), "1 n d -> n d")

# run k-means -- the centroid indicies will be the semantic token ids

n_clusters = 1024
max_iter = 100
batch_size = 10000

km_model = get_km_model(
    n_clusters,
    init = "k-means++",
    max_iter = max_iter,
    batch_size = batch_size,
    tol = 0.,
    max_no_improvement = 100,
    n_init = 20,
    reassignment_ratio = 0.,
)
km_model.fit(activations)

joblib.dump(km_model, km_output_path)

print(f"saved k-means quantizer to path: {km_output_path}")

# generate semantic tokens from the model + k-means quantizer

for file_index, wave in tqdm(enumerate(ds)):
    path = Path(ds.files[file_index])
    output_path = path.with_suffix(".semantic.pt")
    
    with torch.no_grad():
        activations = brq(rearrange(wave, "n -> 1 n"), return_layer_output = output_layer)
        activations = rearrange(activations, "1 n d -> n d")
        
        semantic_labels = km_model.predict(activations)
        torch.save(torch.tensor(semantic_labels, dtype = torch.long), output_path)
