"""Optional reference implementations for the paper's representation pipelines.

The modules in this package intentionally import their heavyweight runtimes only
when needed. The base MAVIC assessment package therefore remains usable without
PyTorch, video decoders, or model weights.
"""
