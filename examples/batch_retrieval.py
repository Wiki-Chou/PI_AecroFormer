"""A reusable retriever for a dataframe or an (n,5) NumPy array."""
import pandas as pd
from pi_aecroformer import Retriever
from pi_aecroformer.api import INPUT_ORDER, default_root

root = default_root()
frame = pd.read_csv(root / "examples/nasa/smoke_input.csv")
runtime = Retriever(root)
out = runtime.retrieve(frame[list(INPUT_ORDER)].to_numpy(dtype=float))
frame["N"] = out["N"]
frame["N_std"] = out["uncertainty"]["N"]["std"]
frame["r_eff"] = out["r_eff"]
(root / "outputs").mkdir(exist_ok=True)
frame.to_csv(root / "outputs/batch_example.csv", index=False)

