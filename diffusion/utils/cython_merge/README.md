# Optional Cython Acceleration

RAGenCO can run without this extension; TSP decoding falls back to a pure NumPy
merge implementation when the compiled module is unavailable.

To build the optional `cython_merge` module locally, run:

```bash
python setup.py build_ext --inplace
```
