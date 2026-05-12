"""
Compatibility entrypoint.

The full objective trainer now lives in train_full_objective.py.
Running `python train.py ...` is kept working by forwarding to the new,
more explicit script.
"""

from train_full_objective import main


if __name__ == "__main__":
    main()
