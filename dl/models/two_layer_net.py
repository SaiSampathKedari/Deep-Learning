from typing import Optional, Dict

import torch

from dl.utils import sample_batch, accuracy, reset_seed


TwoLayerParams = Dict[str, torch.Tensor]


class TwoLayerNet:
  def __init__(
    self,
    input_size  : int,
    hidden_size : int,
    output_size : int,
    dtype       : torch.dtype = torch.float32,
    device      : str | torch.device = "cpu",
    std         : float = 1e-4
  )->None:
    reset_seed(0)
    
    self.params : TwoLayerParams = {
      "W1" : std * torch.randn(
        input_size,
        hidden_size,
        dtype=dtype,
        device=device,
      ),
      "b1" : torch.zeros(
        hidden_size,
        dtype=dtype,
        device=device,
      ),
      "W2" : std * torch.randn(
        hidden_size,
        output_size,
        dtype=dtype,
        device=device,
      ),
      "b2" : torch.zeros(
        output_size,
        dtype=dtype,
        device=device,
      ),
    }
  
  
  def loss(
    self,
    X     : torch.Tensor,
    y     : Optional[torch.Tensor] = None,
    reg   : float = 0.0,
  )-> torch.Tensor | tuple[torch.Tensor, TwoLayerParams]:
    return nn_forward_backward(self.params, X, y, reg)
  
  def train(
    self,
    X               : torch.Tensor,
    y               : torch.Tensor,
    X_val           : torch.Tensor,
    y_val           : torch.Tensor,
    learning_rate   : float = 1e-3,
    learning_rate_decay : float = 0.95,
    reg: float = 5e-6,
    num_iters: int = 100,
    batch_size: int = 200,
    verbose: bool = False,
  )-> Dict[str, list[float]]:
    
    return nn_train(
      self.params,
      X,
      y,
      X_val,       
      y_val,       
      learning_rate,
      learning_rate_decay,
      reg,          
      num_iters,     
      batch_size,    
      verbose,
    )
  
  def predict(
      self,
      X: torch.Tensor,
  ) -> torch.Tensor:
      return nn_predict(self.params, X)

  def save(self, path : str)->None:
    torch.save(self.params, path)
    print(f"Saved in {path}")

  def load(self, path: str) -> None:
      checkpoint = torch.load(path, map_location="cpu", weights_only=True)
      required_params = {"W1", "b1", "W2", "b2"}

      if not isinstance(checkpoint, dict):
          raise ValueError("Checkpoint must contain a parameter dictionary.")

      if set(checkpoint.keys()) != required_params:
          raise ValueError("Invalid TwoLayerNet checkpoint.")

      self.params = checkpoint
      
      
def nn_forward_pass(
  params    :   TwoLayerParams,
  X         :   torch.Tensor
)-> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the forward pass of a two-layer neural network.

    Architecture:
        affine -> ReLU -> affine

    Inputs:
    - params: Dictionary containing:
        W1: Tensor of shape (D, H)
        b1: Tensor of shape (H,)
        W2: Tensor of shape (H, C)
        b2: Tensor of shape (C,)
    - X: Input tensor of shape (N, D)

    Returns:
    - scores: Class scores of shape (N, C)
    - hidden: ReLU hidden activations of shape (N, H)
    """

    W1 = params["W1"]
    b1 = params["b1"]
    W2 = params["W2"]
    b2 = params["b2"]

    hidden_pre = X @ W1 + b1
    hidden = hidden_pre.clamp(min=0.0) #(N, H)

    scores = hidden @ W2 + b2 #(N, C)

    return scores, hidden



def nn_forward_backward(
    params: TwoLayerParams,
    X: torch.Tensor,
    y: Optional[torch.Tensor] = None,
    reg: float = 0.0,
)-> torch.Tensor | tuple[torch.Tensor, TwoLayerParams]:
    """
    Compute the Softmax loss and reverse-mode gradients for a two-layer network.

    Inputs:
    - params: Dictionary containing W1, b1, W2, and b2
    - X: Input tensor of shape (N, D)
    - y: Optional label tensor of shape (N,)
    - reg: L2 regularization strength

    Returns:
    - If y is None:
        scores: Tensor of shape (N, C)
    - Otherwise:
        loss: Scalar average minibatch loss including regularization
        grads: Dictionary containing gradients for W1, b1, W2, and b2
    """
    W1 = params["W1"]
    W2 = params["W2"]

    scores, hidden = nn_forward_pass(params, X)

    # Inference mode: no labels means no loss or backward pass.
    if y is None:
        return scores

    num_batch = X.shape[0]
    idx = torch.arange(num_batch, device=X.device)


    # Numerical stability: subtract each sample's largest score.
    shifted_scores = scores - scores.max(dim=1, keepdim=True).values

    # Softmax probabilities: (N, C)
    exp_scores = torch.exp(shifted_scores)
    probabilities = exp_scores / exp_scores.sum(dim=1, keepdim=True)

    # Log probability assigned to the correct class for every sample.
    correct_log_probs = (
        shifted_scores[idx, y]
        - torch.log(exp_scores.sum(dim=1))
    )

    # Average data loss over the current batch.
    data_loss = -correct_log_probs.mean()

    # L2 regularization loss. Biases are not regularized.
    reg_loss = reg * (
        W1.square().sum()
        + W2.square().sum()
    )

    loss = data_loss + reg_loss

    dscores = probabilities.clone()
    dscores[idx, y] -= 1.0
    dscores /= num_batch                         # (N, C)


    grads: TwoLayerParams = {}

    grads["W2"] = hidden.T @ dscores + 2.0 * reg * W2  # (H, C)
    grads["b2"] = dscores.sum(dim=0)                    # (C,)

    # Incoming gradient for the hidden activations.
    dhidden = dscores @ W2.T                            # (N, H)

    dhidden_pre = dhidden * (hidden > 0)                # (N, H)


    grads["W1"] = X.T @ dhidden_pre + 2.0 * reg * W1   # (D, H)
    grads["b1"] = dhidden_pre.sum(dim=0)                # (H,)

    return loss, grads
    

def nn_predict(
    params: TwoLayerParams,
    X: torch.Tensor,
) -> torch.Tensor:
    """Predict the highest-scoring class for each input."""
    scores, _ = nn_forward_pass(params, X)
    return scores.argmax(dim=1)
  
def nn_train(
  params        : TwoLayerParams,
  X             : torch.Tensor,
  y             : torch.Tensor,
  X_val         : torch.Tensor,
  y_val         : torch.Tensor,
  learning_rate : float = 1e-3,
  learning_rate_decay : float = 0.95,
  reg           : float = 5e-6,
  num_iters     : int = 100,
  batch_size    : int = 200,
  verbose       : bool=False
)-> Dict[str, list[float]]:
    """
    Train a two-layer neural network using minibatch SGD.

    Inputs:
    - params: Dictionary containing W1, b1, W2, and b2
    - X: Training data of shape (N, D)
    - y: Training labels of shape (N,)
    - X_val: Validation data of shape (N_val, D)
    - y_val: Validation labels of shape (N_val,)
    - learning_rate: SGD learning rate
    - learning_rate_decay: Learning-rate decay applied after each epoch
    - reg: L2 regularization strength
    - num_iters: Number of SGD iterations
    - batch_size: Minibatch size
    - verbose: Whether to print training progress

    Returns:
    - Dictionary containing loss, training-accuracy, and validation-accuracy
      histories.
    """
    
    num_train = X.shape[0]
    iterations_per_epoch = max(num_train // batch_size, 1)
    
    loss_history: list[float] = []
    train_acc_history: list[float] = []
    val_acc_history: list[float] = []
    
    for it in range(num_iters):
      X_batch, y_batch = sample_batch(X, y, num_train, batch_size)
      
      loss, grads = nn_forward_backward(params, X_batch, y_batch, reg)
      loss_history.append(loss.item())
      
      params["W1"] -= learning_rate * grads["W1"]
      params["b1"] -= learning_rate * grads["b1"]
      params["W2"] -= learning_rate * grads["W2"]
      params["b2"] -= learning_rate * grads["b2"]
      
      if verbose and it % 100 == 0:
        print(f"iteration {it} / {num_iters}: loss {loss.item():.6f}")
      
      # Every epoch, check train and val accuracy and decay learning rate.
      if (it + 1) % iterations_per_epoch == 0:
        # Check accuracy
        y_train_pred = nn_predict(params, X_batch)
        train_acc = accuracy(y_train_pred, y_batch)
        train_acc_history.append(train_acc)
        
        y_val_pred = nn_predict(params, X_val)
        val_acc = accuracy(y_val_pred, y_val)
        val_acc_history.append(val_acc)

        # Decay learning rate
        learning_rate *= learning_rate_decay
    
    return {
        "loss_history": loss_history,
        "train_acc_history": train_acc_history,
        "val_acc_history": val_acc_history,
    }
    
    
def nn_get_search_params(
) -> tuple[list[float], list[int], list[float], list[float]]:
    """
    Return a small hyperparameter grid for CPU training.

    Returns:
    - learning_rates: Candidate SGD learning rates
    - hidden_sizes: Candidate hidden-layer sizes
    - regularization_strengths: Candidate L2 strengths
    - learning_rate_decays: Candidate learning-rate decay factors
    """
    learning_rates = [0.5, 1.0]
    hidden_sizes = [64, 128]
    regularization_strengths = [1e-5, 1e-4]
    learning_rate_decays = [0.95]

    return (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    )
    
def find_best_net(
    data_dict: Dict[str, torch.Tensor],
) -> tuple[Optional[TwoLayerNet], Optional[Dict[str, list[float]]], float]:
    """
    Train models over a small hyperparameter grid and retain the model with
    the highest validation accuracy.

    Inputs:
    - data_dict: Dictionary containing X_train, y_train, X_val, and y_val

    Returns:
    - best_net: Best trained TwoLayerNet
    - best_stat: Training statistics for the best model
    - best_val_acc: Highest validation accuracy
    """
    best_net: Optional[TwoLayerNet] = None
    best_stat: Optional[Dict[str, list[float]]] = None
    best_val_acc = 0.0

    (
        learning_rates,
        hidden_sizes,
        regularization_strengths,
        learning_rate_decays,
    ) = nn_get_search_params()

    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]

    input_size = X_train.shape[1]
    output_size = int(y_train.max().item()) + 1
    device = X_train.device
    dtype = X_train.dtype

    for hidden_size in hidden_sizes:
        for learning_rate in learning_rates:
            for reg in regularization_strengths:
                for learning_rate_decay in learning_rate_decays:
                    net = TwoLayerNet(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        output_size=output_size,
                        dtype=dtype,
                        device=device,
                    )

                    stat = net.train(
                        X=X_train,
                        y=y_train,
                        X_val=X_val,
                        y_val=y_val,
                        learning_rate=learning_rate,
                        learning_rate_decay=learning_rate_decay,
                        reg=reg,
                        num_iters=500,
                        batch_size=256,
                        verbose=False,
                    )

                    val_acc = stat["val_acc_history"][-1]

                    print(
                        f"hidden={hidden_size}, "
                        f"lr={learning_rate}, "
                        f"reg={reg}, "
                        f"decay={learning_rate_decay}, "
                        f"val_acc={val_acc:.4f}"
                    )

                    if val_acc > best_val_acc:
                        best_val_acc = val_acc
                        best_net = net
                        best_stat = stat

    return best_net, best_stat, best_val_acc