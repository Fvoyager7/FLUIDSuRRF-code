# ed package: NASA Earthdata credentials live in edcreds.py (copy from edcreds.example.py).
try:
    import ed.edcreds  # noqa: F401
except ModuleNotFoundError:
    pass
