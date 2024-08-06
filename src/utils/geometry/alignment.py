"""Methods for weighted rigid alignment of the ground truth onto the denoised structure before
the diffusion loss is applied."""
import torch
from src.utils.geometry.vector import Vec3Array
from src.utils.geometry.rotation_matrix import Rot3Array


def compute_covariance_matrix(P, Q):
    """Computes the covariance matrix between two sets of points P and Q.
    The covariance matrix H is calculated by H = P^T*Q. This is used to
    find the transformation from Q to P.
    Args:
        P: (bs, n_atoms, 3) tensor of points
        Q: (bs, n_atoms, 3) tensor of points
    Returns:
        (bs, 3, 3) tensor of covariance matrices.
    """
    return torch.matmul(P.transpose(-2, -1), Q).float()


def weighted_rigid_align(
        x: Vec3Array,  # (*, n_atoms)
        x_gt: Vec3Array,  # (*, n_atoms)
        weights: torch.Tensor,  # (*, n_atoms)
        mask: torch.Tensor = None  # (bs, n_atoms)
) -> Vec3Array:
    """Performs a weighted alignment of x to x_gt. Warning: ground truth here only refers to the structure
    not being moved, not to be confused with ground truth during training."""
    with torch.no_grad():
        # Set matmul precision to high
        original_precision = torch.get_float32_matmul_precision()
        torch.set_float32_matmul_precision("highest")

        # Mean-centre positions
        mu = (x * weights).mean(dim=-1, keepdim=True) / weights.mean(dim=-1, keepdim=True)
        mu_gt = (x_gt * weights).mean(dim=-1, keepdim=True) / weights.mean(dim=-1, keepdim=True)
        x -= mu  # Vec3Array of shape (*, n_atoms)
        x_gt -= mu_gt

        # Mask atoms before computing covariance matrix
        if mask is not None:
            x *= mask
            x_gt *= mask

        # Find optimal rotation from singular value decomposition in float32 precision
        U, S, Vh = torch.linalg.svd(
           compute_covariance_matrix(
               x_gt.to_tensor().float(),
               x.to_tensor().float()
           )
        )  # shapes: (bs, 3, 3)
        R = U @ Vh

        # Remove reflection
        d = torch.sign(torch.linalg.det(R.float()))  # (bs,)
        # TODO: make this work with arbitrary batch dimensions
        reflection_matrix = torch.eye(3, device=U.device, dtype=U.dtype).unsqueeze(0).expand_as(R)  # (bs, 3, 3)
        reflection_matrix = reflection_matrix.clone()  # avoid a memory issue
        reflection_matrix[:, 2, 2] = d  # replace the bottom right element with the sign of the determinant
        R = U @ reflection_matrix @ Vh  # (bs, 3, 3)

        R = Rot3Array.from_array(R).unsqueeze(-1)  # (bs, 1)

        # Apply alignment
        x_aligned = R.apply_to_point(x) + mu

        # Re-set the original precision
        torch.set_float32_matmul_precision(original_precision)
        return x_aligned

