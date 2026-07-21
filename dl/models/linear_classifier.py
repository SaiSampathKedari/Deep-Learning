from abc import ABC, abstractmethod
from typing import Callable, Optional, TypeAlias

import torch
import random

from dl.utils import sample_batch


LossOutput: TypeAlias = tuple[torch.Tensor, torch.Tensor]

LossFunction: TypeAlias = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    LossOutput,
]


class LinearClassifier(ABC):
    
    def __init__(self):
        random.seed(0)
        torch.manual_seed(0)
        self.W : Optional[torch.Tensor] = None
    
    def train(
        self,
        X_train         : torch.Tensor,
        y_train         : torch.Tensor,
        learning_rate   : float = 1e-3,
        reg             : float = 1e-5,
        num_iters       : int = 100,
        batch_size      : int = 200,
        verbose         : bool = False,
    ) -> list[float]:
        
        self.W, loss_history = train_linear_classifier(
            loss_func=self.loss,
            W=self.W,
            X=X_train,
            y=y_train,
            learning_rate=learning_rate,
            reg=reg,
            num_iters=num_iters,
            batch_size=batch_size,
            verbose=verbose,
        )
        return loss_history

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        if self.W is None:
            raise RuntimeError(
                "The classifier has not been trained or loaded."
            )
            
        return predict_linear_classifier(self.W, X)
    
    @abstractmethod
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ) -> LossOutput:
        """
        Compute the loss function and its derivative.
        Subclasses will override this.

        Inputs:
        - W: A PyTorch tensor of shape (D, C) containing (trained) weight of a model.
        - X_batch: A PyTorch tensor of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A PyTorch tensor of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an tensor of the same shape as W
        """
        raise NotImplementedError
    
    def save(self, path: str) -> None:
        torch.save({"W": self.W}, path)
        print("Saved in {}".format(path))
        
    def load(self, path: str) -> None:
        W_dict = torch.load(path, map_location="cpu")
        self.W = W_dict["W"]
        if self.W is None:
            raise Exception("Failed to load your checkpoint")
        # print("load checkpoint file: {}".format(path))


class LinearSVM(LinearClassifier):
    """A subclass that uses the Multiclass SVM loss function"""
    
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ) -> LossOutput:
        return svm_loss_vectorized(W, X_batch, y_batch, reg)
    
class Softmax(LinearClassifier):
    """A subclass that uses the Softmax + Cross-entropy loss function"""
    
    def loss(
        self,
        W: torch.Tensor,
        X_batch: torch.Tensor,
        y_batch: torch.Tensor,
        reg: float,
    ) -> LossOutput:
        return softmax_loss_vectorized(W, X_batch, y_batch, reg)
    

def svm_loss_naive(
    W   :   torch.Tensor, 
    X   :   torch.Tensor,
    y   :   torch.Tensor, 
    reg :   float
) -> LossOutput:
    """
    Compute the multiclass SVM loss and its gradient using explicit loops.

    Inputs:
    - W: Tensor of shape (D, C) containing the classifier weights.
    - X: Tensor of shape (N, D) containing a minibatch of N input examples.
    - y: Integer tensor of shape (N,) containing the correct class label
      for each example.
    - reg: Float giving the regularization strength.

    Returns:
    - loss: Scalar tensor containing the average SVM loss, including
      regularization.
    - dW: Tensor of shape (D, C) containing the gradient of the loss
      with respect to W.
    """
    
    # initialize the gradient and loss as zero
    dW = torch.zeros_like(W)
    loss = torch.zeros((), device=W.device, dtype=W.dtype) 
    
    num_train = X.shape[0]
    num_classes = W.shape[1]
    
    # compute the loss and the gradient
    for i in range(num_train):
        scores = W.t() @ X[i]
        correct_class_score = scores[y[i]]
        
        for j in range(num_classes):
            if j == y[i]:
                continue
            
            margin = scores[j] - correct_class_score + 1 # delta = 1
            
            if margin > 0:
                loss += margin
                dW[:, j] += X[i]
                dW[:, y[i]] -= X[i]
    
    # final data loss
    loss /= num_train
    dW /= num_train
    
    # Add regularization to the loss.
    loss += reg * torch.sum(W * W)
    
    # Add regularization gradient 
    dW += 2 * reg * W
    
    return loss, dW
    

def svm_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float,
) -> LossOutput:
    """
    Compute the multiclass SVM loss and gradient using vectorized operations.

    Inputs:
    - W: Weight matrix of shape (D, C).
    - X: Minibatch of N examples, with shape (N, D).
    - y: Integer class labels, with shape (N,).
    - reg: Regularization strength.

    Returns:
    - loss: Scalar tensor containing the average minibatch SVM loss,
      including regularization.
    - dW: Tensor of shape (D, C) containing the gradient of the
      minibatch objective with respect to W.
    """

    num_train = X.shape[0]
    idx = torch.arange(num_train, device=X.device)

    # Scores: (N, D) @ (D, C) = (N, C)
    scores = X @ W

    # Correct-class score for every example: (N,)
    correct_class_scores = scores[idx, y]

    # Margins: (N, C)
    margins = scores - correct_class_scores.view(-1, 1) + 1.0
    margins = torch.clamp(margins, min=0.0)

    # The correct class should not contribute to the loss
    margins[idx, y] = 0.0

    # Average minibatch loss plus regularization
    loss = margins.sum() / num_train
    loss += reg * torch.sum(W * W)

    # mask[i, j] = 1 when class j violates the margin
    mask = (margins > 0).to(dtype=W.dtype)

    # Number of violating classes for each example: (N,)
    num_violations = mask.sum(dim=1)

    # Correct class receives -x_i once per violating class
    mask[idx, y] = -num_violations

    # (D, N) @ (N, C) = (D, C)
    dW = X.t() @ mask
    dW /= num_train

    # Regularization gradient
    dW += 2.0 * reg * W

    return loss, dW

def softmax_loss_naive(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float,
) -> LossOutput:
    """
    Compute the average Softmax cross-entropy loss over a minibatch
    and its gradient using explicit loops.

    Inputs:
    - W: Tensor of shape (D, C) containing classifier weights.
    - X: Tensor of shape (N, D) containing a minibatch of N examples.
    - y: Integer tensor of shape (N,) containing the correct class
      label for each example.
    - reg: Float giving the regularization strength.

    Returns:
    - loss: Scalar tensor containing the average minibatch
      cross-entropy loss, including regularization.
    - dW: Tensor of shape (D, C) containing the gradient of the
      minibatch objective with respect to W.
    """

    dW = torch.zeros_like(W)
    loss = torch.zeros((), device=W.device, dtype=W.dtype)

    num_train = X.shape[0]

    for i in range(num_train):

        # Scores for one example: (C,)
        scores = W.t() @ X[i]

        # Numerical stability
        shifted_scores = scores - scores.max()

        # Softmax probabilities: (C,)
        exp_scores = torch.exp(shifted_scores)
        probabilities = exp_scores / exp_scores.sum()

        # Cross-entropy loss for example i
        loss += -shifted_scores[y[i]] + torch.log(exp_scores.sum())

        # Gradient with respect to scores:
        # probabilities - one_hot(y_i)
        dscores = probabilities.clone()
        dscores[y[i]] -= 1.0

        # Outer product:
        # (D,) outer (C,) = (D, C)
        dW += torch.outer(X[i], dscores)

    # Average minibatch loss and gradient
    loss /= num_train
    dW /= num_train

    # Regularization
    loss += reg * torch.sum(W * W)
    dW += 2.0 * reg * W

    return loss, dW

def softmax_loss_vectorized(
    W: torch.Tensor,
    X: torch.Tensor,
    y: torch.Tensor,
    reg: float,
) -> LossOutput:
    """
    Compute the average Softmax cross-entropy loss over a minibatch
    and its gradient using vectorized operations.

    Inputs:
    - W: Tensor of shape (D, C) containing classifier weights.
    - X: Tensor of shape (N, D) containing a minibatch of N examples.
    - y: Integer tensor of shape (N,) containing the correct class
      label for each example.
    - reg: Float giving the regularization strength.

    Returns:
    - loss: Scalar tensor containing the average minibatch
      cross-entropy loss, including regularization.
    - dW: Tensor of shape (D, C) containing the gradient of the
      minibatch objective with respect to W.
    """

    num_train = X.shape[0]
    idx = torch.arange(num_train, device=X.device)

    # Scores: (N, D) @ (D, C) = (N, C)
    scores = X @ W

    # Numerical stability: subtract each row's maximum
    shifted_scores = scores - scores.max(dim=1, keepdim=True).values

    # Softmax probabilities: (N, C)
    exp_scores = torch.exp(shifted_scores)
    probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)

    # Average cross-entropy loss
    correct_log_probs = (
        shifted_scores[idx, y]
        - torch.log(exp_scores.sum(dim=1))
    )

    loss = -correct_log_probs.mean()

    # Regularization loss
    loss += reg * torch.sum(W * W)

    # Gradient with respect to scores:
    # probabilities - one_hot(y)
    dscores = probabilities.clone()
    dscores[idx, y] -= 1.0

    # Average over the minibatch
    dscores /= num_train

    # (D, N) @ (N, C) = (D, C)
    dW = X.t() @ dscores

    # Regularization gradient
    dW += 2.0 * reg * W

    return loss, dW



def train_linear_classifier(
    loss_func: LossFunction,
    W: Optional[torch.Tensor],
    X: torch.Tensor,
    y: torch.Tensor,
    learning_rate: float = 1e-3,
    reg: float = 1e-5,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
) -> tuple[torch.Tensor, list[float]]:
    """
    Train a linear classifier using minibatch stochastic gradient descent.

    Inputs:
    - loss_func: Function that computes the minibatch loss and gradient.
    - W: Initial weight matrix of shape (D, C), or None.
    - X: Training data of shape (N, D).
    - y: Training labels of shape (N,).
    - learning_rate: SGD step size.
    - reg: Regularization strength.
    - num_iters: Number of SGD iterations.
    - batch_size: Number of examples sampled per iteration.
    - verbose: Whether to print training progress.

    Returns:
    - W: Trained weight matrix of shape (D, C).
    - loss_history: List containing the minibatch loss at each iteration.
    """

    # Validate inputs
    if X.ndim != 2:
        raise ValueError("X must have shape (N, D).")

    if y.ndim != 1:
        raise ValueError("y must have shape (N,).")

    if X.shape[0] != y.shape[0]:
        raise ValueError(
            "X and y must contain the same number of examples."
        )

    if X.shape[0] == 0:
        raise ValueError("Training data must not be empty.")

    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    if num_iters < 0:
        raise ValueError("num_iters cannot be negative.")
    
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")

    if reg < 0:
        raise ValueError("reg cannot be negative.")

    num_train, dim = X.shape

    if W is None:
        num_classes = int(torch.max(y).item()) + 1

        W = 1e-6 * torch.randn(
            dim,
            num_classes,
            device=X.device,
            dtype=X.dtype,
        )
    

    loss_history: list[float] = []

    for iteration in range(num_iters):

        # Random minibatch
        X_batch, y_batch = sample_batch(
            X,
            y,
            num_train,
            batch_size,
        )

        # Minibatch loss and stochastic gradient
        loss, grad = loss_func(
            W,
            X_batch,
            y_batch,
            reg,
        )

        loss_history.append(loss.item())

        # SGD update
        W -= learning_rate * grad

        if verbose and iteration % 100 == 0:
            print(
                "iteration %d / %d: loss %f"
                % (iteration, num_iters, loss.item())
            )

    return W, loss_history


def predict_linear_classifier(
    W   : torch.Tensor,
    X   : torch.Tensor
) -> torch.Tensor:
    scores = X @ W
    return scores.argmax(dim=1)

