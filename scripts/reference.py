"""Independent float32 PyTorch oracle; validation only, never used by Rust.

Requires torch, numpy, librosa, safetensors (and soundfile for --audio).
Equations follow Hugging Face Transformers' Parakeet model and the published
thrush-ternary-v2 manifest. Default input is deterministic integer-generated PCM.
"""
import argparse
import json
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file


class Reference:
    def __init__(self, directory):
        self.weights = load_file(str(directory / "model.safetensors"))
        self.config = json.loads((directory / "config.json").read_text())

    def weight(self, name):
        return self.weights[name].float()

    def linear(self, x, name):
        if name + ".qweight" in self.weights:
            packed = self.weights[name + ".qweight"].long()
            scales = self.weight(name + ".scales")
            width = x.shape[-1]
            powers = torch.tensor([1, 3, 9, 27, 81])
            codes = (packed[..., None] // powers) % 3
            values = codes.reshape(packed.shape[0], -1)[:, :width].float() - 1
            weight = values * scales.repeat_interleave(self.config["ternary_group_size"], 1)[:, :width]
            return F.linear(x, weight)
        bias = self.weights.get(name + ".bias")
        return F.linear(x, self.weight(name + ".weight"), None if bias is None else bias.float())

    def norm(self, x, name):
        return F.layer_norm(x, (x.shape[-1],), self.weight(name + ".weight"), self.weight(name + ".bias"), 1e-5)

    def subsample(self, features):
        x = features[None, None]
        for index in [0, 2, 3, 5, 6]:
            name = f"encoder.subsampling.layers.{index}"
            depthwise = index in (2, 5)
            spatial = index in (0, 2, 5)
            x = F.conv2d(x, self.weight(name + ".weight"), self.weight(name + ".bias"),
                         stride=2 if spatial else 1, padding=1 if spatial else 0,
                         groups=x.shape[1] if depthwise else 1)
            if index in (0, 3, 6):
                x = x.relu()
        x = x.transpose(1, 2).reshape(x.shape[0], x.shape[2], -1)
        return self.linear(x, "encoder.subsampling.linear")[0]

    def attention(self, x, name, positions):
        frames, width = x.shape
        heads = self.config["encoder_config"]["num_attention_heads"]
        head_width = width // heads
        def heads_of(value):
            return value.reshape(-1, heads, head_width).transpose(0, 1)
        q, k, v = [heads_of(self.linear(x, name + suffix)) for suffix in (".q_proj", ".k_proj", ".v_proj")]
        relative = heads_of(self.linear(positions, name + ".relative_k_proj"))
        ac = (q + self.weight(name + ".bias_u")[:, None]) @ k.transpose(-1, -2)
        bd = (q + self.weight(name + ".bias_v")[:, None]) @ relative.transpose(-1, -2)
        # Use the upstream reshape-based shift, independent of Rust's indexing.
        bd = F.pad(bd, (1, 0)).reshape(heads, -1, frames)[:, 1:]
        bd = bd.reshape(heads, frames, 2 * frames - 1)[:, :, :frames]
        scores = ((ac + bd) / head_width**0.5).softmax(-1)
        attended = (scores @ v).transpose(0, 1).reshape(frames, width)
        return self.linear(attended, name + ".o_proj")

    def encode(self, x):
        frames, width = x.shape
        positions = torch.arange(frames - 1, -frames, -1).float()
        angles = positions[:, None] / 10000 ** (torch.arange(0, width, 2).float() / width)
        positions = torch.stack((angles.sin(), angles.cos()), -1).flatten(1)
        for index in range(self.config["encoder_config"]["num_hidden_layers"]):
            name = f"encoder.layers.{index}"
            def feedforward(x, number):
                prefix = name + f".feed_forward{number}"
                hidden = self.norm(x, name + f".norm_feed_forward{number}")
                return x + 0.5 * self.linear(F.silu(self.linear(hidden, prefix + ".linear1")), prefix + ".linear2")
            x = feedforward(x, 1)
            x = x + self.attention(self.norm(x, name + ".norm_self_att"), name + ".self_attn", positions)
            hidden = self.norm(x, name + ".norm_conv")
            hidden = F.glu(self.linear(hidden, name + ".conv.pointwise_conv1"), -1)
            hidden = F.conv1d(hidden.T[None], self.weight(name + ".conv.depthwise_conv.weight"),
                              padding=self.config["encoder_config"]["conv_kernel_size"] // 2, groups=width)
            prefix = name + ".conv.norm"
            hidden = F.batch_norm(hidden, self.weight(prefix + ".running_mean"), self.weight(prefix + ".running_var"),
                                  self.weight(prefix + ".weight"), self.weight(prefix + ".bias"), training=False, eps=1e-5)
            x = x + self.linear(F.silu(hidden[0].T), name + ".conv.pointwise_conv2")
            x = self.norm(feedforward(x, 2), name + ".norm_out")
        return x

    def vad(self, x):
        x = x.T[None]
        for name in ("proj", "ctx", "out"):
            prefix = "vad_head." + name
            x = F.conv1d(x, self.weight(prefix + ".weight"), self.weight(prefix + ".bias"), padding=2 if name == "ctx" else 0)
            if name != "out":
                x = F.silu(x)
        return x.sigmoid().flatten()

    def decode(self, x):
        width = self.config["decoder_hidden_size"]
        layers = self.config["num_decoder_layers"]
        lstm = torch.nn.LSTM(width, width, layers, batch_first=True)
        lstm.load_state_dict({k.removeprefix("decoder.lstm."): v.float() for k, v in self.weights.items() if k.startswith("decoder.lstm.")})
        state = None
        def predict(token):
            nonlocal state
            embedding = self.weight("decoder.embedding.weight")[token][None, None]
            output, state = lstm(embedding, state)
            return self.linear(output[0], "decoder.decoder_projector")
        blank = self.config["blank_token_id"]
        vocab = self.config["vocab_size"]
        prediction = predict(blank)
        projected = self.linear(x, "encoder_projector")
        frame = 0
        emissions = []
        for _ in range(len(x) * self.config["max_symbols_per_step"]):
            if frame >= len(x):
                return emissions
            logits = self.linear((projected[frame:frame+1] + prediction).relu(), "joint.head")[0]
            token = int(logits[:vocab].argmax())
            duration = self.config["durations"][int(logits[vocab:].argmax())]
            if token == blank:
                duration = max(duration, 1)
            else:
                emissions.append({"token_id": token, "frame": frame, "duration": duration})
                prediction = predict(token)
            frame += duration
        raise RuntimeError("reference decode budget exhausted")


def features(samples):
    x = torch.tensor(samples, dtype=torch.float32)
    emphasized = torch.cat((x[:1], x[1:] - 0.97 * x[:-1]))
    spectrum = torch.stft(emphasized, 512, hop_length=160, win_length=400,
                          window=torch.hann_window(400, periodic=False), pad_mode="constant", return_complex=True)
    filters = torch.from_numpy(librosa.filters.mel(sr=16000, n_fft=512, n_mels=128, norm="slaney"))
    values = (filters @ spectrum.abs().square() + 2**-24).log().T[:len(samples)//160]
    return (values - values.mean(0)) / (values.std(0) + 1e-5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=Path("models/parakeet-redux"))
    parser.add_argument("--audio", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    if args.audio:
        import soundfile as sf
        samples, rate = sf.read(args.audio, dtype="float32")
        assert rate == 16000 and samples.ndim == 1
    else:
        state = 42
        samples = []
        for _ in range(1600):
            state = (1664525 * state + 1013904223) & 0xffffffff
            samples.append(((state >> 16) - 32768) / 32768)
    model = Reference(args.model)
    mel = features(samples)
    hidden = model.subsample(mel)
    encoded = model.encode(hidden)
    emissions = model.decode(encoded)
    tokenizer = json.loads((args.model / "tokenizer.json").read_text())
    pieces = {value: key for key, value in tokenizer["model"]["vocab"].items()}
    text = "".join(pieces.get(item["token_id"], "") for item in emissions).replace("▁", " ").lstrip()
    result = {"samples": np.asarray(samples).tolist(), "features": mel.tolist(), "subsampled": hidden.tolist(),
              "encoded": encoded.tolist(), "vad": model.vad(hidden).tolist(), "emissions": emissions, "text": text}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result))
    print(text)
    print(f"Saved {len(samples)} samples, {len(mel)} features, {len(hidden)} encoder frames, {len(emissions)} tokens to {args.output}")


if __name__ == "__main__":
    main()
