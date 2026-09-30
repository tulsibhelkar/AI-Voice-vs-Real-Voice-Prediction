import random

import numpy as np
import torch

from training.config import RANDOM_SEED


def set_random_seed(seed: int = RANDOM_SEED) -> None:
    """
    Set random seeds for reproducible experiments.

    This controls randomness in:
    - Python
    - NumPy
    - PyTorch
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def main() -> None:

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("REPRODUCIBILITY TEST")
    print("=" * 60)

    print(
        f"\nSetting random seed to: "
        f"{RANDOM_SEED}"
    )

    # -----------------------------
    # First run
    # -----------------------------

    set_random_seed()

    python_value_1 = random.random()

    numpy_values_1 = np.random.rand(3)

    torch_values_1 = torch.rand(3)

    # -----------------------------
    # Reset the same seed
    # -----------------------------

    set_random_seed()

    python_value_2 = random.random()

    numpy_values_2 = np.random.rand(3)

    torch_values_2 = torch.rand(3)

    # -----------------------------
    # Verify reproducibility
    # -----------------------------

    python_match = (
        python_value_1 == python_value_2
    )

    numpy_match = np.allclose(
        numpy_values_1,
        numpy_values_2
    )

    torch_match = torch.allclose(
        torch_values_1,
        torch_values_2
    )

    print("\nPython reproducible:")
    print(python_match)

    print("\nNumPy reproducible:")
    print(numpy_match)

    print("\nPyTorch reproducible:")
    print(torch_match)

    if not (
        python_match
        and numpy_match
        and torch_match
    ):
        raise RuntimeError(
            "Reproducibility test failed."
        )

    print("\nAll random generators are reproducible.")

    print("\n" + "=" * 60)
    print("REPRODUCIBILITY TEST PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()