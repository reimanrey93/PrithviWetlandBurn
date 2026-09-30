# PrithviWetlandBurn

Burned-area segmentation and change-detection experiments using Prithvi EO V2 and Sentinel-2 imagery.

## Sardinia SURTR experiments

The Sardinia experiments use official burned-area polygons from Geoportale Sardegna and a bi-temporal Siamese architecture based on Prithvi.

The experimental protocol uses an event-wise split, ensuring that complete fire events are separated between training, validation, and test sets.

### Model comparison

| Model | Test IoU | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| POST-only | 0.8715 | 0.9313 | 0.9654 | 0.8996 |
| Siamese frozen | 0.8960 | 0.9452 | 0.9673 | 0.9240 |
| Siamese + LoRA | 0.9010 | 0.9479 | 0.9676 | 0.9290 |
| Siamese + Partial FT | 0.9155 | 0.9559 | 0.9689 | 0.9433 |

### Adaptation strategies

- Frozen Prithvi encoder
- Siamese bi-temporal change fusion
- LoRA adaptation on QKV and attention projections
- Partial fine-tuning of the last 4 Transformer blocks

## Repository structure

- `src/` — model and dataset implementations
- `scripts/` — dataset construction, training, evaluation and inference
- `results/` — compact experimental results
- `events/` — event metadata
- `preview_*` — example visualizations

Large satellite imagery, model checkpoints and generated raster products are not included in the repository.
