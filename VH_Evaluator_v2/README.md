# VH_Evaluator_v2

**Visual Hallucination Evaluator v2** - Enhanced version with image preprocessing for improved **grid pattern detection**.

## Overview

VH_Evaluator_v2 extends the original VH_Evaluator with **image preprocessing techniques** specifically designed to improve grid pattern detection by the MLLM evaluator.

### Key Difference from v1

**Preprocessing is ONLY used for grid pattern detection:**
- `grid_pattern_score`: Evaluated using BOTH original AND preprocessed images
- `detail_sharpness_score`: Evaluated using ORIGINAL image only
- `irrelevant_details_score`: Evaluated using ORIGINAL image only

This design ensures that preprocessing artifacts don't affect other evaluation metrics while maximizing grid pattern detection accuracy.

## Preprocessing Methods

VH_Evaluator_v2 uses three complementary preprocessing methods:

### 1. Laplacian Enhancement
- **Purpose**: Edge and fine detail enhancement
- **How it works**: Applies Laplacian operator to detect edges, then adds weighted result back to original
- **Best for**: Revealing grid patterns in smooth areas

### 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
- **Purpose**: Local contrast enhancement
- **How it works**: Divides image into tiles and applies histogram equalization locally
- **Best for**: Revealing patterns hidden in low-contrast regions

### 3. Bilateral Filter Enhancement
- **Purpose**: Edge-preserving detail enhancement
- **How it works**: Smooths flat areas while preserving edges, then amplifies the detail layer
- **Best for**: Enhancing texture patterns while maintaining edge sharpness

## Quick Start

### 1. Start the VLM Server

```bash
bash start_server.sh
```

### 2. Run Evaluation with Preprocessing

```bash
# Use all preprocessing methods (default)
bash eval.sh /path/to/images /path/to/output.jsonl

# Or specify methods explicitly
python vl_hallucination_client.py \
  --api_url http://localhost:8000 \
  --folder /path/to/images \
  --output_path ./results_v2.jsonl \
  --preprocessing laplacian clahe bilateral
```

### 3. Use Only Specific Methods

```bash
# Only use Laplacian and CLAHE
python vl_hallucination_client.py \
  --api_url http://localhost:8000 \
  --folder /path/to/images \
  --output_path ./results_v2.jsonl \
  --preprocessing laplacian clahe
```

## Standalone Texture Enhancement

You can also use the preprocessing module independently:

```bash
# Enhance a single image
python texture_enhancement.py input.jpg output_prefix

# Output files:
# - output_prefix_laplacian.jpg
# - output_prefix_clahe.jpg
# - output_prefix_bilateral.jpg
```

### Python API

```python
from texture_enhancement import TextureEnhancer

# Initialize with image path or numpy array
enhancer = TextureEnhancer('input.jpg')

# Apply individual methods
laplacian_result = enhancer.laplacian_enhancement(strength=2.5)
clahe_result = enhancer.clahe_enhancement(clip_limit=6.0, tile_size=6)
bilateral_result = enhancer.bilateral_filter_enhancement(d=7, sigma_color=50, strength=3.0)

# Or apply all methods at once
results = enhancer.apply_all_methods()
```

## Output Format

### Per-Image Results (JSONL)

```json
{
  "image_name": "example.jpg",
  "preprocessing_methods": ["laplacian", "clahe", "bilateral"],
  "evaluation": {
    "image": {
      "has_over_optimization": true,
      "over_optimization_level": 3,
      "detail_sharpness_score": 2,
      "irrelevant_details_score": 2,
      "grid_pattern_score": 4,
      "overall_hallucination_score": 2.67,
      "confidence": 0.9,
      "grid_pattern_detected_in": ["laplacian", "clahe"],
      "detailed_analysis": "Grid patterns visible in Laplacian and CLAHE enhanced versions..."
    }
  }
}
```

### New Field: `grid_pattern_detected_in`

This field indicates which preprocessing method(s) revealed grid patterns:
- `["original"]` - Grid visible in original image
- `["laplacian"]` - Grid revealed by Laplacian enhancement
- `["clahe"]` - Grid revealed by CLAHE enhancement
- `["bilateral"]` - Grid revealed by Bilateral enhancement

## Files

| File | Description |
|------|-------------|
| `vl_hallucination_client.py` | Main evaluation client with preprocessing |
| `texture_enhancement.py` | Image preprocessing module |
| `low_level_metric.py` | Traditional image quality metrics |
| `start_server.sh` | VLM server startup script |
| `eval.sh` | Evaluation convenience script |
| `example/` | Reference images for few-shot evaluation |

## Preprocessing Parameters

### Laplacian Enhancement
| Parameter | Default | Description |
|-----------|---------|-------------|
| `strength` | 2.5 | Enhancement multiplier (1.0-5.0) |

### CLAHE Enhancement
| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_limit` | 6.0 | Contrast limiting threshold (2.0-10.0) |
| `tile_size` | 6 | Grid tile size for local equalization (4-16) |

### Bilateral Enhancement
| Parameter | Default | Description |
|-----------|---------|-------------|
| `d` | 7 | Pixel neighborhood diameter (5-15) |
| `sigma_color` | 50 | Filter sigma in color space |
| `sigma_space` | 50 | Filter sigma in coordinate space |
| `strength` | 3.0 | Detail enhancement multiplier (1.0-5.0) |

## When to Use v2 vs v1

| Scenario | Recommended Version |
|----------|-------------------|
| General evaluation | v1 (faster, fewer API tokens) |
| **Grid pattern detection is critical** | **v2** |
| Research requiring accurate grid detection | **v2** |
| Large-scale batch processing | v1 (lower cost) |
| Subtle/hidden grid pattern analysis | **v2** |

> **Note:** v2 uses more API tokens because it sends multiple images (original + 3 preprocessed) per evaluation. Use v1 for general purposes and v2 when grid pattern detection accuracy is critical.

## Requirements

- Python 3.8+
- OpenCV (`cv2`)
- Pillow
- requests
- tqdm
- numpy

## Citation

If you use VH_Evaluator_v2 in your research, please cite our work.
