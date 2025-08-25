# type: ignore

"""
Integrator implementing quadrature rule.
"""

from typing import Tuple
from typeguard import typechecked

from jaxtyping import Float, jaxtyped
import torch
from torch_nerf.src.renderer.integrators.integrator_base import IntegratorBase


class QuadratureIntegrator(IntegratorBase):
    """
    Numerical integrator which approximates integral using quadrature.
    """

    @jaxtyped
    @typechecked
    def integrate_along_rays(
        self,
        sigma: Float[torch.Tensor, "num_ray num_sample"],
        radiance: Float[torch.Tensor, "num_ray num_sample 3"],
        delta: Float[torch.Tensor, "num_ray num_sample"],
    ) -> Tuple[
        Float[torch.Tensor, "num_ray 3"], Float[torch.Tensor, "num_ray num_sample"]
    ]:
        """
        Computes quadrature rule to approximate integral involving in volume rendering.
        Pixel colors are computed as weighted sums of radiance values collected along rays.

        For details on the quadrature rule, refer to 'Optical models for
        direct volume rendering (IEEE Transactions on Visualization and Computer Graphics 1995)'.

        Args:
            sigma: Density values sampled along rays. [num_ray, num_sample]
            radiance: Radiance values sampled along rays. [num_ray, num_sample, 3]
            delta: Distance between adjacent samples along rays. [num_ray, num_sample]

        Returns:
            rgbs: Pixel colors computed by evaluating the volume rendering equation.
            weights: Weights used to determine the contribution of each sample to the final pixel color.
                A weight at a sample point is defined as a product of transmittance and opacity,
                where opacity (alpha) is defined as 1 - exp(-sigma * delta).
        """
        sigma_delta = torch.cumsum(sigma * delta, dim=-1)  # [num_ray, num_sample]

        zeros = torch.zeros_like(sigma_delta[:, :1])  # [num_ray, 1]

        sigma_delta_shifted = torch.cat([zeros, sigma_delta[:, :-1]], dim=-1)

        transmittance = torch.exp(-sigma_delta_shifted)  # [num_ray, num_sample]

        alpha = 1.0 - torch.exp(-sigma * delta)  # [num_ray, num_sample]

        weights = transmittance * alpha  # [num_ray, num_sample]

        rgbs = torch.sum(weights.unsqueeze(-1) * radiance, dim=-2)  # [num_ray, 3]

        return rgbs, weights
