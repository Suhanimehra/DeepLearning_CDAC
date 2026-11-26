import torch
import torch.nn as nn


def main() -> None:
    # Native Windows CPU execution (no CUDA or WSL2)
    device = torch.device("cpu")
    print("PyTorch version:", torch.__version__)
    print("Running on device:", device)

    # Explicit data: Monthly income (in ₹ thousands) -> Loan Approved (1) or Rejected (0)
    monthly_income_k = torch.tensor(
        [[15.0], [20.0], [25.0], [30.0], [35.0], [50.0], [60.0], [75.0], [90.0], [100.0]],
        dtype=torch.float32,
        device=device,
    )
    loan_approved = torch.tensor(
        [[0.0], [0.0], [0.0], [0.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]],
        dtype=torch.float32,
        device=device,
    )

    # TODO 1: Create a single linear layer (1 input feature -> 1 output logit) on `device`
    model = None

    # TODO 2: Choose binary classification loss (BCEWithLogitsLoss) and SGD optimizer (lr=0.05)
    criterion = None
    optimizer = None

    epochs = 4000
    for epoch in range(1, epochs + 1):
        # TODO 3: Complete the 5-step PyTorch training loop:
        #   1. Clear old gradients
        #   2. Forward pass on `monthly_income_k`
        #   3. Compute loss against `loan_approved`
        #   4. Backpropagation
        #   5. Update parameters
        pass

        if epoch % 1000 == 0:
            print(f"Epoch {epoch:4d}/{epochs} | Loss: {loss.item():.6f}")

    # Test decisions on new applicants
    applicants_k = torch.tensor([[18.0], [38.0], [42.0], [80.0]], dtype=torch.float32, device=device)

    model.eval()
    with torch.no_grad():
        # TODO 4: Compute predicted approval probabilities using torch.sigmoid(...)
        probs = None

    print("\nLoan Decision Results:")
    for income_k, prob in zip(applicants_k.squeeze(dim=1).tolist(), probs.tolist()):
        # TODO 5: Make a binary decision ("APPROVED" if probability >= 0.5 else "REJECTED")
        decision = "PENDING"
        print(f"  Income ₹{income_k:5.1f}k/month -> Prob: {prob:.4f} | Decision: {decision}")


if __name__ == "__main__":
    main()