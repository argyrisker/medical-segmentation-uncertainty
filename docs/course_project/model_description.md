# Model description: residual 3D U-Net

All shapes and parameter counts below were traced from the trained configuration
(`configs/experiments/baseline.yaml`, class `UNet3D` in `src/models.py`).

## Summary

The network is a five-level residual 3D U-Net (Ronneberger et al., 2015; Çiçek et
al., 2016) with 5,733,475 trainable parameters. It maps a single-channel,
z-score-normalised MRI volume, zero-padded to 48 × 64 × 48 voxels at 1 mm isotropic
resolution, to voxel-wise softmax probabilities over three classes (background,
anterior hippocampus, posterior hippocampus). The contracting path applies one
residual convolution block per level and 2 × 2 × 2 max pooling between levels,
doubling the channel count from 16 to 256 while halving each spatial dimension down
to a 3 × 4 × 3 bottleneck. The expanding path restores resolution with 2 × 2 × 2
transposed convolutions, concatenates the encoder feature map of the same resolution
through a skip connection, and fuses the result with a residual block. A 1 × 1 × 1
convolution produces the class logits. All convolutions are followed by instance
normalisation and LeakyReLU, and element-wise dropout (p = 0.1) after every
activation enables Monte Carlo dropout at test time.

## Residual convolution block

For input x with C_in channels and output with C_out channels:

```
h = D( σ( N1( W1 ∗ x ) ) )
y = D( σ( N2( W2 ∗ h ) + P x ) )
```

- W1, W2: 3 × 3 × 3 convolutions, stride 1, padding 1, no bias
- N1, N2: instance normalisation with learnable affine parameters
- σ: LeakyReLU, negative slope 0.01
- D: element-wise dropout, p = 0.1 (kept active at inference for MC dropout)
- P: 1 × 1 × 1 projection convolution (every block changes the channel count, so P is never the identity)

## Layer-by-layer specification

Feature maps are written as channels @ x × y × z (left–right × posterior–anterior × inferior–superior).

| Stage | Operation | Output | Parameters |
|---|---|---|---|
| Input | MRI volume, zero-padded | 1 @ 48 × 64 × 48 | |
| Encoder 1 | Residual block 1 → 16 | 16 @ 48 × 64 × 48 | 7,424 |
| | Max pool 2 × 2 × 2 | 16 @ 24 × 32 × 24 | |
| Encoder 2 | Residual block 16 → 32 | 32 @ 24 × 32 × 24 | 42,112 |
| | Max pool 2 × 2 × 2 | 32 @ 12 × 16 × 12 | |
| Encoder 3 | Residual block 32 → 64 | 64 @ 12 × 16 × 12 | 168,192 |
| | Max pool 2 × 2 × 2 | 64 @ 6 × 8 × 6 | |
| Encoder 4 | Residual block 64 → 128 | 128 @ 6 × 8 × 6 | 672,256 |
| | Max pool 2 × 2 × 2 | 128 @ 3 × 4 × 3 | |
| Bottleneck | Residual block 128 → 256 | 256 @ 3 × 4 × 3 | 2,688,000 |
| Decoder 4 | Transposed conv 2 × 2 × 2, 256 → 128 | 128 @ 6 × 8 × 6 | 262,272 |
| | Concatenate with Encoder 4 | 256 @ 6 × 8 × 6 | |
| | Residual block 256 → 128 | 128 @ 6 × 8 × 6 | 1,360,384 |
| Decoder 3 | Transposed conv, 128 → 64 | 64 @ 12 × 16 × 12 | 65,600 |
| | Concatenate with Encoder 3 | 128 @ 12 × 16 × 12 | |
| | Residual block 128 → 64 | 64 @ 12 × 16 × 12 | 340,224 |
| Decoder 2 | Transposed conv, 64 → 32 | 32 @ 24 × 32 × 24 | 16,416 |
| | Concatenate with Encoder 2 | 64 @ 24 × 32 × 24 | |
| | Residual block 64 → 32 | 32 @ 24 × 32 × 24 | 85,120 |
| Decoder 1 | Transposed conv, 32 → 16 | 16 @ 48 × 64 × 48 | 4,112 |
| | Concatenate with Encoder 1 | 32 @ 48 × 64 × 48 | |
| | Residual block 32 → 16 | 16 @ 48 × 64 × 48 | 21,312 |
| Head | 1 × 1 × 1 conv 16 → 3, softmax | 3 @ 48 × 64 × 48 | 51 |
| **Total** | | | **5,733,475** |

The bottleneck holds 47% of all parameters; the two finest levels together hold under 2%.

## Training and inference (for the caption)

- Loss: soft Dice over the two foreground classes plus voxel-wise cross-entropy.
- Optimiser: AdamW, learning rate 3e-4 with cosine decay over 200 epochs, weight decay 1e-5, batch of 4 volumes.
- Inference: one 48 × 64 × 48 window covers the whole scan; prediction is the arg-max of the softmax.
- Uncertainty: predictive entropy of the mean softmax over 20 MC-dropout passes, or over an ensemble of three independently seeded models.

## Figure specification

What to draw so the figure matches the model exactly:

- **Layout:** a "U". Encoder levels descend from top-left to the bottleneck at bottom-centre; decoder levels ascend to the top-right. Input volume at far left, output segmentation at far right.
- **Feature maps:** one box per block output. Box height proportional to spatial size, box width proportional to channel count. Label each box with its channels and size, for example "32 @ 24×32×24".
- **Arrows (with a legend):**
  - solid right arrow: residual conv block (inside a level)
  - downward arrow: 2×2×2 max pooling
  - upward arrow: 2×2×2 transposed convolution
  - dashed horizontal arrow: skip connection, copy and concatenate
  - final arrow: 1×1×1 convolution + softmax
- **Colours used in the slides:** encoder #1F4E8C (dark blue), decoder #7FA3CF (light blue), bottleneck #3A3F47 (dark grey), skip arrows #9AA3AE (grey, dashed).
- **Inset (optional):** the residual block as a small flow chart: Conv 3×3×3 → InstanceNorm → LeakyReLU → Dropout → Conv 3×3×3 → InstanceNorm → (+ 1×1×1 shortcut) → LeakyReLU → Dropout.
- **Ends:** an axial MRI slice as the input thumbnail; the same slice with a two-colour label (anterior, posterior) as the output thumbnail.

## Prompt for an image generator

Image generators often misspell numbers and labels. Check every label, or add the text afterwards in PowerPoint or draw.io.

```
Clean scientific architecture diagram of a 3D U-Net for medical image segmentation,
flat vector style, white background, publication quality, no shadows, no 3D rendering.
A U-shaped layout: on the left, an encoder of four levels descending diagonally,
boxes labelled "16 @ 48×64×48", "32 @ 24×32×24", "64 @ 12×16×12", "128 @ 6×8×6",
in dark blue (#1F4E8C); at the bottom centre a bottleneck box "256 @ 3×4×3" in dark
grey (#3A3F47); on the right, a decoder of four levels ascending with boxes
"128 @ 6×8×6", "64 @ 12×16×12", "32 @ 24×32×24", "16 @ 48×64×48" in light blue
(#7FA3CF). Box width grows with channel count. Downward arrows labelled "max pool
2×2×2", upward arrows labelled "transposed conv 2×2×2", dashed grey horizontal arrows
labelled "skip: concatenate" connecting each encoder level to the decoder level at the
same height. Far left: a small grayscale brain MRI slice labelled "Input 1 @ 48×64×48".
Far right: the same slice with a two-colour segmentation labelled "Output: 3 classes".
A small legend in the corner. A small inset showing one residual block:
Conv3×3×3, InstanceNorm, LeakyReLU, Dropout, Conv3×3×3, InstanceNorm, a 1×1×1
shortcut joined by a plus sign, LeakyReLU, Dropout. Sans-serif labels, thin lines.
```
