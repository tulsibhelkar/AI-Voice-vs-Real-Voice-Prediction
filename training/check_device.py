import torch


def main():

    print("=" * 60)
    print("AI VOICE VS REAL VOICE PREDICTION")
    print("PYTORCH DEVICE CHECK")
    print("=" * 60)

    print(f"\nPyTorch version: {torch.__version__}")

    cuda_available = torch.cuda.is_available()

    print(
        f"CUDA available: {cuda_available}"
    )

    if cuda_available:

        device = torch.device("cuda")

        print("\nTraining device: GPU")

        print(
            f"GPU name: "
            f"{torch.cuda.get_device_name(0)}"
        )

        print(
            f"Number of GPUs: "
            f"{torch.cuda.device_count()}"
        )

        gpu_memory = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / (1024 ** 3)
        )

        print(
            f"GPU memory: "
            f"{gpu_memory:.2f} GB"
        )

    else:

        device = torch.device("cpu")

        print("\nTraining device: CPU")

        print(
            "CUDA-compatible GPU is not "
            "available to PyTorch."
        )

    # Small tensor test
    print("\nTesting tensor operation...")

    x = torch.randn(
        3,
        3,
        device=device
    )

    y = torch.randn(
        3,
        3,
        device=device
    )

    result = torch.matmul(x, y)

    print(
        f"Tensor device: "
        f"{result.device}"
    )

    print(
        f"Tensor shape: "
        f"{result.shape}"
    )

    print("\nTensor operation successful.")

    print("\n" + "=" * 60)
    print("PYTORCH DEVICE CHECK PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()