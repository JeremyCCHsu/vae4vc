# VAE for VC

Implementation of [Voice Conversion from Non-parallel Corpora Using Variational Auto-encoder](https://arxiv.org/abs/1610.04019).

We modify the variational auto-encoder (VAE) for our voice conversion (VC) task to deal with training from non-parallel corpora. There are 3 modules in the overall architecture:

1. feature extraction
2. conversion
3. speech synthesis

The 1st and the 3rd parts involve the STRAIGHT vocoder, so they are not included in this repository.

*Speaker dependent global variance was applied to the resulting spectra.

```python
python train.py --data_dir [path] --file_filter [regexp]
```
