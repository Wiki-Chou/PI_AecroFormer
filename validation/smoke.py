"""Run bounded NASA inference and one profile plot, with no training."""
from pi_aecroformer.cli import main

if __name__ == "__main__":
    import sys
    main(["smoke", *sys.argv[1:]])

