"""Internal subprocess entry point. Inputs are generated locally, never pickled.

The numeric path reads ``X`` and ``labels``; the mixed-type adapter (``"adapter": "tabular"`` in
``config.json``) reads typed columns, numeric float arrays and 'U' string arrays. Both load with
``allow_pickle=False``.
"""
from pathlib import Path
import json
import sys
import numpy as np

from .core import CostAssumptions
from .training import _fit_one


def main():
    directory = Path(sys.argv[1])
    try:
        configuration = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        if configuration.get("costs") is not None:
            configuration["costs"] = CostAssumptions(**configuration["costs"])
        with np.load(directory / "input.npz", allow_pickle=False) as data:
            if configuration.get("adapter") == "tabular":
                from .mixed import fit_tabular_model
                result = fit_tabular_model({name: data[name] for name in data.files}, configuration)
            else:
                for name in ("groups", "times"):
                    if name in data:
                        configuration[name] = data[name]
                result = _fit_one(data["X"], data["labels"], **configuration)
        encoded = json.dumps(result, allow_nan=False)
    except Exception as exc:
        result = {"status": "error", "error_type": type(exc).__name__, "error": str(exc)}
        if isinstance(getattr(exc, "code", None), str):
            result["error_code"] = exc.code
        encoded = json.dumps(result, allow_nan=False)
    (directory / "result.json").write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
