from best_rq_pytorch.best_rq import BestRQ
from best_rq_pytorch.trainer import BestRQPretrainer
from best_rq_pytorch.conformer import ConformerWrapper
from best_rq_pytorch.data import AudioDataset

# load the dataset

dataset_folder = "..."

ds = AudioDataset(
    dataset_folder,
    max_length_in_seconds = 10
)

# set up the model

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
)

print(f"trainable parameters: {sum(p.numel() for p in brq.parameters() if p.requires_grad)}")

# train the model

trainer = BestRQPretrainer(
    model = brq,
    dataset = ds,
    num_train_steps = 100_000,
    lr = 5e-4,
    num_warmup_steps = 1000,
    initial_lr = 1e-5,
    batch_size = 8,
    grad_accum_every = 1,
    mask_prob = 0.6
)

trainer.train()
