"""
General utilities to help with implementation
"""
import random

import matplotlib.pyplot as plt
import torch


def reset_seed(number):
    """
    Reset random seed to the specific number

    Inputs:
    - number: A seed number to use
    """
    random.seed(number)
    torch.manual_seed(number)
    return


def tensor_to_image(tensor):
    """
    Convert a torch tensor into a numpy ndarray for visualization.

    Inputs:
    - tensor: A torch tensor of shape (3, H, W) with elements in the range [0, 1]

    Returns:
    - ndarr: A uint8 numpy array of shape (H, W, 3)
    """
    tensor = tensor.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0)
    ndarr = tensor.to("cpu", torch.uint8).numpy()
    return ndarr


def visualize_dataset(X_data, y_data, samples_per_class, class_list):
    """
    Make a grid-shape image to plot

    Inputs:
    - X_data: set of [batch, 3, width, height] data
    - y_data: paired label of X_data in [batch] shape
    - samples_per_class: number of samples want to present
    - class_list: list of class names (e.g.) ['plane', 'car', 'bird', 'cat',
      'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

    Outputs:
    - An grid-image that visualize samples_per_class number of samples per class
    """

    # Protected lazy import.
    from torchvision.utils import make_grid

    img_half_width = X_data.shape[2] // 2
    samples = []
    for y, cls in enumerate(class_list):
        plt.text(
            -4, (img_half_width * 2 + 2) * y + (img_half_width + 2), cls, ha="right"
        )
        idxs = (y_data == y).nonzero().view(-1)
        for i in range(samples_per_class):
            idx = idxs[random.randrange(idxs.shape[0])].item()
            samples.append(X_data[idx])

    img = make_grid(samples, nrow=samples_per_class)
    return tensor_to_image(img)


def sample_batch(
    X: torch.Tensor,
    y: torch.Tensor,
    num_train: int,
    batch_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Sample a random minibatch of training examples and corresponding labels.

    Inputs:
    - X: Tensor of shape (N, D) containing the training examples.
    - y: Tensor of shape (N,) containing the training labels.
    - num_train: Number of training examples available for sampling.
    - batch_size: Number of examples to sample.

    Returns:
    - X_batch: Tensor of shape (batch_size, D).
    - y_batch: Tensor of shape (batch_size,).
    """
    indices = torch.randint(
        low=0,
        high=num_train,
        size=(batch_size,),
        device=X.device,
    )

    X_batch = X[indices]
    y_batch = y[indices]

    return X_batch, y_batch


def accuracy(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
) -> float:
    """Return classification accuracy as a fraction in [0, 1]."""
    return (y_pred == y_true).double().mean().item()


def visualize_linear_weights(
    weights: torch.Tensor,
    class_names,
    image_shape=(3, 32, 32),
    title: str = None,
):
    """
    Plot each class's learned weight vector as an image.

    A linear classifier's weight column for a class is a template that class's
    images are matched against, so it should look faintly like the class.

    Inputs:
    - weights: Weight matrix of shape (D + 1, C); the last row is the bias
      contributed by the bias trick and is dropped before reshaping.
    - class_names: List of C class names, in label order.
    - image_shape: (channels, height, width) of a single input image.
    - title: Optional figure title.
    """
    channels, height, width = image_shape

    w = weights[:-1, :].reshape(channels, height, width, len(class_names))
    w = w.transpose(0, 2).transpose(1, 0)      # -> (height, width, channels, C)
    w_min, w_max = w.min(), w.max()

    plt.figure(figsize=(12, 5))
    for i, name in enumerate(class_names):
        plt.subplot(2, 5, i + 1)
        img = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(img.type(torch.uint8).cpu())
        plt.axis("off")
        plt.title(name)
    if title:
        plt.suptitle(title)
    plt.show()