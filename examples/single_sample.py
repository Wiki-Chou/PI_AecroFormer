"""Use the first curated NASA observation; do not invent optical units."""
import json
import pandas as pd
from pi_aecroformer import retrieve
from pi_aecroformer.api import INPUT_ORDER, default_root

row = pd.read_csv(default_root() / "examples/nasa/paper_input_113.csv").iloc[0]
out = retrieve(row[list(INPUT_ORDER)].to_numpy(dtype=float))
print(json.dumps({"N": out["N"], "N_uncertainty": out["uncertainty"]["N"], "r_eff": out["r_eff"], "SSA": out["SSA"]}, indent=2))

