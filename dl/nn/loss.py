import torch


class SoftmaxCrossEntropy:

    @staticmethod
    def forward(
        scores: torch.Tensor,
        y: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        num_train = y.shape[0]
        idx = torch.arange(num_train, device=scores.device)

        # Numerical stability
        shifted_scores = (
            scores
            - scores.max(dim=1, keepdim=True).values
        )

        # Softmax probabilities
        exp_scores = torch.exp(shifted_scores)
        probabilities = (
            exp_scores
            / exp_scores.sum(dim=1, keepdim=True)
        )

        # Log-probability of the correct class
        correct_log_probs = (
            shifted_scores[idx, y]
            - torch.log(exp_scores.sum(dim=1))
        )

        # Average cross-entropy loss
        loss = -correct_log_probs.mean()

        # Gradient with respect to scores
        dscores = probabilities.clone()
        dscores[idx, y] -= 1.0
        dscores /= num_train

        return loss, dscores
