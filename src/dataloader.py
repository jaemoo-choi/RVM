import os
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import Sampler
from torch.utils.data.distributed import DistributedSampler
import torch
import json
from PIL import Image


class TextPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}.txt")
        with open(self.file_path, "r") as f:
            self.prompts = [line.strip() for line in f.readlines()]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        # global_idx must be carried: evaluation.py keys resume shards, merged
        # rows and saved image filenames on metadata["global_idx"], falling back
        # to a RANK-LOCAL counter when it is absent. With DistributedSampler
        # (shuffle=False) every rank then emitted the same ids 0..N/world_size,
        # which silently (a) dropped all but one rank's rows in the merge, (b)
        # made ranks overwrite each other's images/{id}.jpg, and (c) made a
        # rerun resume-skip every batch. idx here is the global dataset index.
        return {"prompt": self.prompts[idx], "metadata": {"global_idx": idx}}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

class GenevalPromptDataset(Dataset):
    def __init__(self, dataset, split="train"):
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        return {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas

def _resize_center_crop(image, target_h, target_w):
    """Aspect-fill resize then center-crop a PIL image to exactly target_h x target_w.

    Used when reference images were generated at a different geometry than the
    training resolution (e.g. running SkyReels @ 400x640 on the Wan2.2-A14B
    832x480 ref images). The SkyReels pipeline VAE-encodes the ref image and
    concatenates the first-frame latents with the noise latents along the
    channel dim, so the ref image MUST match config.height x config.width or
    the concat fails with a shape mismatch.
    """
    w, h = image.size  # PIL reports (width, height)
    if (w, h) == (target_w, target_h):
        return image  # already the right size — no resampling
    scale = max(target_w / w, target_h / h)
    new_w = max(target_w, round(w * scale))
    new_h = max(target_h, round(h * scale))
    image = image.resize((new_w, new_h), Image.BICUBIC)
    left = (new_w - target_w) // 2
    top = (new_h - target_h) // 2
    return image.crop((left, top, left + target_w, top + target_h))


class ImagePromptDataset(Dataset):
    """Dataset that loads text prompts paired with pre-generated reference images.

    Used for SkyReels-I2V training with pre-generated Flux reference images.
    Expects a metadata.jsonl file mapping prompts to image paths, produced by
    scripts/dataset/generate_skyreels_ref_images.py.

    If target_height/target_width are given, each reference image is
    aspect-fill resized and center-cropped to that resolution so it matches
    the training geometry (config.height x config.width).
    """

    def __init__(self, dataset, split="train", target_height=None, target_width=None):
        ref_images_dir = os.path.join(dataset, f"ref_images_{split}")
        meta_path = os.path.join(ref_images_dir, "metadata.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No metadata.jsonl found at {meta_path}. "
                f"Run scripts/dataset/generate_skyreels_ref_images.py first."
            )
        self.ref_images_dir = ref_images_dir
        self.target_height = target_height
        self.target_width = target_width
        with open(meta_path, "r") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.prompts = [e["prompt"] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        img_path = os.path.join(self.ref_images_dir, entry["image_path"])
        image = Image.open(img_path).convert("RGB")
        if self.target_height is not None and self.target_width is not None:
            image = _resize_center_crop(image, self.target_height, self.target_width)
        return {
            "prompt": entry["prompt"],
            "metadata": {
                "ref_image": image,
                "global_idx": int(entry.get("index", idx)),
            },
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class EmbeddingPromptDataset(Dataset):
    """Dataset that loads pre-computed text embeddings instead of raw prompts.

    Used for memory-efficient training where text encoders are not loaded on GPU.
    Expects a directory with:
        prompt_embeds/{idx}.pt
        pooled_prompt_embeds/{idx}.pt
        prompt_attention_mask/{idx}.pt
        metadata.jsonl  (index -> caption mapping)
    Produced by scripts/dataset/precompute_hunyuan_embeddings.py.
    """

    def __init__(self, dataset, split="train"):
        embeddings_dir = os.path.join(dataset, f"embeddings_{split}")
        meta_path = os.path.join(embeddings_dir, "metadata.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No metadata.jsonl found at {meta_path}. "
                f"Run scripts/dataset/precompute_hunyuan_embeddings.py first."
            )
        self.embeddings_dir = embeddings_dir
        with open(meta_path, "r") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.prompts = [e["caption"] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_idx = entry["index"]
        prompt_embeds = torch.load(
            os.path.join(self.embeddings_dir, "prompt_embeds", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )
        # pooled_prompt_embeds is optional (not used by HunyuanVideo 1.5)
        pooled_path = os.path.join(self.embeddings_dir, "pooled_prompt_embeds", f"{file_idx}.pt")
        if os.path.exists(pooled_path):
            pooled_prompt_embeds = torch.load(pooled_path, map_location="cpu", weights_only=True)
        else:
            pooled_prompt_embeds = None
        prompt_attention_mask = torch.load(
            os.path.join(self.embeddings_dir, "prompt_attention_mask", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )
        metadata = {
            "prompt_embeds": prompt_embeds,
            "pooled_prompt_embeds": pooled_prompt_embeds,
            "prompt_attention_mask": prompt_attention_mask,
        }
        # Pass through extra fields
        for key in entry:
            if key not in ("index", "caption"):
                metadata[key] = entry[key]
        return {"prompt": entry["caption"], "metadata": metadata}

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class EmbeddingImagePromptDataset(Dataset):
    """Dataset that loads pre-computed text embeddings AND VAE-encoded ref image latents.

    Used for SkyReels-I2V memory-efficient training where text encoders,
    Flux, and VAE encoder are all skipped at training time.
    Expects (produced by scripts/dataset/precompute_skyreels_data.py):
        embeddings_{split}/prompt_embeds/{idx}.pt
        embeddings_{split}/pooled_prompt_embeds/{idx}.pt
        embeddings_{split}/prompt_attention_mask/{idx}.pt
        embeddings_{split}/first_frame_latents/{idx}.pt   # VAE-encoded + zero-padded
        embeddings_{split}/metadata.jsonl
    """

    def __init__(self, dataset, split="train"):
        embeddings_dir = os.path.join(dataset, f"embeddings_{split}")
        meta_path = os.path.join(embeddings_dir, "metadata.jsonl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"No metadata.jsonl found at {meta_path}. "
                f"Run scripts/dataset/precompute_skyreels_data.py first."
            )
        self.embeddings_dir = embeddings_dir
        with open(meta_path, "r") as f:
            self.entries = [json.loads(line) for line in f if line.strip()]
        self.prompts = [e["caption"] for e in self.entries]

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        entry = self.entries[idx]
        file_idx = entry["index"]
        prompt_embeds = torch.load(
            os.path.join(self.embeddings_dir, "prompt_embeds", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )
        pooled_prompt_embeds = torch.load(
            os.path.join(self.embeddings_dir, "pooled_prompt_embeds", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )
        prompt_attention_mask = torch.load(
            os.path.join(self.embeddings_dir, "prompt_attention_mask", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )
        first_frame_latents = torch.load(
            os.path.join(self.embeddings_dir, "first_frame_latents", f"{file_idx}.pt"),
            map_location="cpu", weights_only=True,
        )

        return {
            "prompt": entry["caption"],
            "metadata": {
                "prompt_embeds": prompt_embeds,
                "pooled_prompt_embeds": pooled_prompt_embeds,
                "prompt_attention_mask": prompt_attention_mask,
                "first_frame_latents": first_frame_latents,
            },
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        return prompts, metadatas


class DistributedKRepeatSampler(Sampler):
    def __init__(self, dataset, batch_size, k, num_replicas, rank, seed=0):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed

        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g)[: self.m].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]

            shuffled_indices = torch.randperm(len(repeated_indices), generator=g).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def get_testloader(config, world_size, rank, num_workers=0):
    if config.prompt_fn in ["general_ocr", "text"]:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "image_prompt":
        train_dataset = ImagePromptDataset(config.dataset, "train",
                                           target_height=config.height, target_width=config.width)
        test_dataset = ImagePromptDataset(config.dataset, "test",
                                          target_height=config.height, target_width=config.width)
    elif config.prompt_fn == "embedding":
        train_dataset = EmbeddingPromptDataset(config.dataset, "train")
        test_dataset = EmbeddingPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "embedding_image":
        train_dataset = EmbeddingImagePromptDataset(config.dataset, "train")
        test_dataset = EmbeddingImagePromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError(f"Prompt function not supported with dataset: {config.prompt_fn}")

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return test_dataloader



def get_dataloaders(config, world_size, rank, num_workers=0):
    """Build train/test dataloaders from config.

    Returns:
        tuple: (train_dataloader, test_dataloader)
    """
    if config.prompt_fn in ["general_ocr", "text"]:
        train_dataset = TextPromptDataset(config.dataset, "train")
        test_dataset = TextPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "geneval":
        train_dataset = GenevalPromptDataset(config.dataset, "train")
        test_dataset = GenevalPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "image_prompt":
        train_dataset = ImagePromptDataset(config.dataset, "train",
                                           target_height=config.height, target_width=config.width)
        test_dataset = ImagePromptDataset(config.dataset, "test",
                                          target_height=config.height, target_width=config.width)
    elif config.prompt_fn == "embedding":
        train_dataset = EmbeddingPromptDataset(config.dataset, "train")
        test_dataset = EmbeddingPromptDataset(config.dataset, "test")
    elif config.prompt_fn == "embedding_image":
        train_dataset = EmbeddingImagePromptDataset(config.dataset, "train")
        test_dataset = EmbeddingImagePromptDataset(config.dataset, "test")
    else:
        raise NotImplementedError(f"Prompt function not supported with dataset: {config.prompt_fn}")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(test_dataset, num_replicas=world_size, rank=rank, shuffle=False)
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,
        sampler=test_sampler,
        collate_fn=test_dataset.collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_dataloader, test_dataloader

