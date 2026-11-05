# InsightTok provenance

InsightTok is the image codec of Lumina. `insight_tok.py`, `modules/diffusionmodules/model.py`, and `modules/vqvae/quantize.py` come from the `models/` directory of the InsightTok runtime; the empty `__init__.py` files make them a package. The only local changes are this repository's formatters, black, isort, and autoflake, so the logic is unchanged.

License: MIT, copyright LeapLab, see `LICENSE`.
