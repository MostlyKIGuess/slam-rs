# Model Weights Directory

This directory stores pre-trained deep learning models for depth estimation.

## Quick Start

### Option 1: Download Pre-converted Models (Recommended)

If you have access to pre-converted TorchScript models:

```bash
cd weights/
# Download encoder and decoder
# Will update these soon
wget <url-to-encoder.pt>
wget <url-to-depth.pt>
```

Expected Hashes:
- TODO: Add this 

### Option 2: Convert from MonoDepth2 PyTorch Models

1. **Clone MonoDepth2 repository:**

```bash
git clone https://github.com/nianticlabs/monodepth2.git
cd monodepth2
```

2. **Download pretrained weights:**

```bash
# Download mono+stereo_640x192 (recommended)
wget https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono%2Bstereo_640x192.zip
unzip mono+stereo_640x192.zip

# OR download other models:
# mono_640x192.zip
# stereo_640x192.zip
# mono_1024x320.zip
# mono+stereo_1024x320.zip
```

3. **Convert to TorchScript:**

Create `convert_to_torchscript.py`:

```python
#!/usr/bin/env python3
"""
Convert MonoDepth2 .pth weights to TorchScript .pt format for use with tch-rs.

This script properly loads MonoDepth2 weights following the original loading logic
from test_simple.py to ensure correct model initialization.

Usage:
    1. Place this script in the same directory as encoder.pth and depth.pth
    2. Make sure the monodepth2 'networks' module is available
    3. Run: python convert.py
    4. Move the generated encoder.pt and depth.pt to the weights directory
"""

import os
import sys

import torch

sys.path.append(".")

from networks import DepthDecoder, ResnetEncoder

# Configuration - these must match the model variant you're using
WIDTH = 640
HEIGHT = 192

print("Converting MonoDepth2 model to TorchScript...")
print(f"Expected input size: {WIDTH}x{HEIGHT}")


print("\n[1/4] Loading encoder...")

encoder = ResnetEncoder(18, False)
encoder_path = "encoder.pth"

if not os.path.exists(encoder_path):
    print(f"ERROR: {encoder_path} not found!")
    print("Make sure encoder.pth is in the current directory.")
    sys.exit(1)

loaded_dict_enc = torch.load(encoder_path, map_location="cpu", weights_only=False)

# Check for metadata (height/width) that indicates the training resolution
if "height" in loaded_dict_enc:
    feed_height = loaded_dict_enc["height"]
    feed_width = loaded_dict_enc["width"]
    print(f"  Found model metadata: {feed_width}x{feed_height}")
    if feed_height != HEIGHT or feed_width != WIDTH:
        print(f"  WARNING: Model was trained at {feed_width}x{feed_height}")
        print(f"           but converting with {WIDTH}x{HEIGHT}")
else:
    print("  No height/width metadata found in encoder checkpoint")

# Filter encoder weights: only keep keys that exist in the model's state_dict
encoder_state_dict = encoder.state_dict()
filtered_dict_enc = {
    k: v for k, v in loaded_dict_enc.items() if k in encoder_state_dict
}

print(f"  Loaded {len(filtered_dict_enc)}/{len(encoder_state_dict)} encoder parameters")

# Load the filtered weights
encoder.load_state_dict(filtered_dict_enc, strict=False)
encoder.eval()

print("  Encoder loaded successfully!")


print("\n[2/4] Loading decoder...")

# The decoder uses scales range(4) = [0, 1, 2, 3] for multi-scale disparity output
decoder = DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))
decoder_path = "depth.pth"

if not os.path.exists(decoder_path):
    print(f"ERROR: {decoder_path} not found!")
    print("Make sure depth.pth is in the current directory.")
    sys.exit(1)

loaded_dict_dec = torch.load(decoder_path, map_location="cpu", weights_only=False)

decoder.load_state_dict(loaded_dict_dec, strict=True)
decoder.eval()

print("  Decoder loaded successfully!")


print("[3/4] Creating TorchScript-compatible wrapper...")


class DepthDecoderWrapper(torch.nn.Module):
    """
    Wrapper that converts the decoder's dict output to a list of tensors.

    The original decoder returns a dict with tuple keys like ("disp", 0), ("disp", 1), etc.
    TorchScript tracing has issues with tuple-keyed dicts, so we convert to an ordered list.

    The output list contains disparity maps at 4 scales:
        [0] = ("disp", 0) - Full resolution (1/1)
        [1] = ("disp", 1) - Half resolution (1/2)
        [2] = ("disp", 2) - Quarter resolution (1/4)
        [3] = ("disp", 3) - Eighth resolution (1/8)
    """

    def __init__(self, decoder):
        super().__init__()
        self.decoder = decoder

    def forward(self, input_features):
        output = self.decoder(input_features)
        # Convert dict with tuple keys to ordered list
        # Index 0 is the highest resolution disparity map
        return [output[("disp", i)] for i in range(4) if ("disp", i) in output]


wrapped_decoder = DepthDecoderWrapper(decoder)
wrapped_decoder.eval()

print("  Wrapper created!")


print("\n[4/4] Tracing and saving TorchScript models...")

with torch.no_grad():
    # Create dummy input matching expected dimensions
    dummy_input = torch.randn(1, 3, HEIGHT, WIDTH)
    print(f"  Using dummy input shape: {dummy_input.shape}")

    # Trace encoder
    print("  Tracing encoder...")
    encoder_traced = torch.jit.trace(encoder, dummy_input)

    # Verify encoder output
    features = encoder(dummy_input)
    print(f"  Encoder produces {len(features)} feature maps:")
    for i, f in enumerate(features):
        print(f"    [{i}] shape: {f.shape}")

    # Save encoder
    encoder_traced.save("encoder.pt")
    print("  Saved: encoder.pt")

    # Prepare features as list for decoder
    features_list = [features[i] for i in range(len(features))]

    # Trace the wrapped decoder
    print("  Tracing decoder...")
    decoder_traced = torch.jit.trace(wrapped_decoder, (features_list,), strict=False)

    # Verify decoder output
    outputs = wrapped_decoder(features_list)
    print(f"  Decoder produces {len(outputs)} disparity maps:")
    for i, o in enumerate(outputs):
        print(f"    [{i}] shape: {o.shape}, range: [{o.min():.4f}, {o.max():.4f}]")

    decoder_traced.save("depth.pt")
    print("  Saved: depth.pt")


print("\n" + "=" * 60)
print("Conversion complete!")
print("=" * 60)


print("\nTo use with slam-rs:")
print("  mv encoder.pt ../")
print("  mv depth.pt ../")
print("\nThen run:")
print("  cargo run --example depth_estimation --features depth \\")
print("    -- test_image.png --encoder weights/encoder.pt --decoder weights/depth.pt")
```

4. **Run conversion:**

```bash
cd monodepth2
python convert_to_torchscript.py
```

5. **Copy to weights directory:**

```bash
mv encoder.pt ../
mv depth.pt ../
```

## Available Models

MonoDepth2 provides several pretrained models:

| Model | Training Mode | Resolution | KITTI Error | File Size |
|-------|---------------|------------|-------------|-----------|
| `mono_640x192` | Monocular | 640×192 | 0.115 | ~97 MB |
| `stereo_640x192` | Stereo | 640×192 | 0.109 | ~97 MB |
| `mono+stereo_640x192` | Both | 640×192 | 0.106 | ~97 MB ⭐ |
| `mono_1024x320` | Monocular | 1024×320 | 0.115 | ~97 MB |
| `mono+stereo_1024x320` | Both | 1024×320 | 0.106 | ~97 MB |

**Recommended**: `mono+stereo_640x192` for best balance of speed and accuracy.

## Verify Installation

After downloading/converting models, verify they work:

```bash
cd slam-rs
cargo run --example depth_estimation --features depth -- \
    --encoder weights/encoder.pt \
    --decoder weights/depth.pt \
    ../test_image.jpg
```

## Alternative Models (Future)

Coming soon:
- MiDaS depth estimation
- UniDepth support
- Custom trained models

## License

MonoDepth2 models are released under the Niantic License. See:
- https://github.com/nianticlabs/monodepth2/blob/master/LICENSE

For commercial use, contact Niantic.

## References

- [MonoDepth2 Paper](https://arxiv.org/abs/1806.01260)
- [MonoDepth2 GitHub](https://github.com/nianticlabs/monodepth2)
- [LibTorch Download](https://pytorch.org/get-started/locally/)
- [tch-rs Documentation](https://github.com/LaurentMazare/tch-rs)
