"""LeJEPA/SIGReg losses - copied from lejepa_REPO_REFERENCE.

Statistical normality tests used as loss functions to push embeddings toward N(0,I).

This file consolidates ALL core LeJEPA components:

Utilities:
- log_norm_cdf_helper, norm_cdf, log_norm_cdf

Distributed utilities:
- is_dist_avail_and_initialized, all_reduce

Base classes:
- UnivariateTest
- MultivariatetTest

Univariate tests:
- EppsPulley
- AndersonDarling
- CramerVonMises
- Watson
- Entropy
- Moments
- VCReg
- ExtendedJarqueBera
- ShapiroWilk
- NLL

Multivariate tests:
- SlicingUnivariateTest
- BHEP
- BHEP_M
- COMB
- HV
- HZ
"""

import numpy as np
import torch
from torch import distributed as dist
import torch.distributed.nn
from scipy.stats import norm as scipy_norm
from typing import Union



def log_norm_cdf_helper(x):
    """Helper function for asymptotic approximation of log normal CDF in the tails."""
    return ((1 - 0.344) * x + 0.344 * x**2 + 5.334).sqrt()


def norm_cdf(x):
    """Compute the standard normal cumulative distribution function."""
    return (1 + torch.erf(x / np.sqrt(2))) / 2


def log_norm_cdf(x: torch.Tensor, thresh: float = 3.0) -> torch.Tensor:
    """Compute log of the standard normal CDF in a numerically stable way."""
    out = torch.empty_like(x)
    left = x < -thresh
    right = x > thresh
    middle = ~(left | right)

    if middle.any():
        out[middle] = norm_cdf(x[middle]).log()

    if left.any():
        x_left = x[left]
        out[left] = (
            -(x_left**2 + np.log(2 * np.pi)) / 2 - log_norm_cdf_helper(-x_left).log()
        )

    if right.any():
        x_right = x[right]
        out[right] = torch.log1p(
            -(-(x_right**2) / 2).exp()
            / np.sqrt(2 * np.pi)
            / log_norm_cdf_helper(x_right)
        )

    return out



def is_dist_avail_and_initialized():
    return dist.is_available() and dist.is_initialized()


def all_reduce(x, op="AVG"):
    if dist.is_available() and dist.is_initialized():
        from torch.distributed._functional_collectives import all_reduce as functional_all_reduce
        return functional_all_reduce(x, op.lower(), dist.group.WORLD)
    else:
        return x



class UnivariateTest(torch.nn.Module):
    def __init__(self, eps: float = 1e-5, sorted: bool = False):
        super().__init__()
        self.eps = eps
        self.sorted = sorted
        self.g = torch.distributions.normal.Normal(0, 1)

    def prepare_data(self, x):
        if self.sorted:
            s = x
        else:
            s = x.sort(descending=False, dim=-2)[0]
        return s

    def dist_mean(self, x):
        if is_dist_avail_and_initialized():
            torch.distributed.nn.functional.all_reduce(x, torch.distributed.ReduceOp.AVG)
        return x

    @property
    def world_size(self):
        if is_dist_avail_and_initialized():
            return dist.get_world_size()
        return 1



class MultivariatetTest(torch.nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def prepare_data(self, x):
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        if not isinstance(x, torch.Tensor):
            raise TypeError(f"Expected numpy array or torch tensor, got {type(x)}")
        if x.ndim != 2:
            raise ValueError(f"Expected 2D input (N, D), got shape {x.shape}")
        return x



class EppsPulley(UnivariateTest):
    """
    Fast Epps-Pulley two-sample test statistic for univariate distributions.

    This implementation uses numerical integration over the characteristic function
    to compute a goodness-of-fit test statistic.

    Args:
        t_max (float, optional): Maximum integration point. Default: 3.
        n_points (int, optional): Number of integration points. Must be odd. Default: 17.
        integration (str, optional): Integration method. Default: 'trapezoid'.
    """

    def __init__(self, t_max: float = 3, n_points: int = 17, integration: str = "trapezoid"):
        super().__init__()
        assert n_points % 2 == 1
        self.integration = integration
        self.n_points = n_points

        t = torch.linspace(0, t_max, n_points, dtype=torch.float32)
        self.register_buffer("t", t)
        dt = t_max / (n_points - 1)
        weights = torch.full((n_points,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        self.register_buffer("phi", self.t.square().mul_(0.5).neg_().exp_())
        self.register_buffer("weights", weights * self.phi)

    def forward(self, x):
        N = x.size(-2)
        x_t = x.unsqueeze(-1) * self.t
        cos_vals = torch.cos(x_t)
        sin_vals = torch.sin(x_t)

        cos_mean = cos_vals.mean(-3)
        sin_mean = sin_vals.mean(-3)

        cos_mean = all_reduce(cos_mean)
        sin_mean = all_reduce(sin_mean)

        err = (cos_mean - self.phi).square() + sin_mean.square()
        return (err @ self.weights) * N * self.world_size



class AndersonDarling(UnivariateTest):
    """
    Anderson-Darling goodness-of-fit test for normality.

    The test gives more weight to observations in the tails, making it
    particularly effective at detecting departures from normality in the extremes.
    """

    def forward(self, x):
        s = self.prepare_data(x)
        n = x.size(0)

        with torch.no_grad():
            k = (
                torch.arange(1, n + 1, device=x.device, dtype=torch.float)
                .mul_(2)
                .sub_(1)
            )

        A = log_norm_cdf(s) + log_norm_cdf(-s.flip(0))
        A_squared = -n - torch.tensordot(A, k, [[0], [0]]) / n

        return A_squared



class CramerVonMises(UnivariateTest):
    """
    Cramer-von Mises goodness-of-fit test for univariate distributions.

    Measures the discrepancy between the empirical distribution function
    and the cumulative distribution function of a reference distribution.
    """

    def forward(self, x):
        s = self.prepare_data(x)
        with torch.no_grad():
            n = x.size(0)
            k = (
                torch.arange(1, n + 1, device=x.device, dtype=x.dtype)
                .mul_(2)
                .sub_(1)
                .div_(2 * n)
            )
            k = k.view(n, *tuple([1] * (x.ndim - 1)))
        T = (k - self.g.cdf(s)).square().mean(0)
        return T



class Watson(CramerVonMises):
    """
    Watson's U^2 test for goodness-of-fit to standard normal N(0,1).

    Location-adjusted modification of the Cramer-von Mises test that reduces
    sensitivity to shifts in location while maintaining power to detect
    differences in scale and shape.
    """

    def forward(self, x):
        T = super().forward(x)
        m = self.g.cdf(x).mean(0)
        return T - (m - 0.5).square()



class Entropy(UnivariateTest):
    """
    Vasicek entropy-based test for normality.

    Reference: Vasicek, Oldrich (1976). "A Test for Normality Based on Sample Entropy".
    """

    def __init__(
        self,
        m: int = 1,
        eps: float = 1e-5,
        method: str = "centered",
        sorted: bool = False,
    ):
        super().__init__(eps=eps, sorted=sorted)
        self.m = m
        self.method = method

    def forward(self, x):
        s = self.prepare_data(x)
        cst = np.log(np.sqrt(2 * np.pi * np.exp(1)))
        if self.method == "right":
            stat = s[self.m :].sub(s[: -self.m]).clip(self.eps).log().sum(0) / x.size(0)
            return cst - stat - np.log(x.size(0)) + torch.log(self.m * x.std(0))
        diff = s[self.m * 2 :] - s[: -self.m * 2]
        stat = diff.clip(self.eps).log().sum(0) / x.size(0)
        for i in range(2 * self.m):
            delta = 1
            stat += (s[1 + i] - s[0]).mul(delta).clip(self.eps).log() / x.size(0)
            stat += (s[-1] - s[-2 - i]).mul(delta).clip(self.eps).log() / x.size(0)
        return (
            cst - stat - np.log(x.size(0)) + np.log(2 * self.m) + torch.log(x.std(0))
        ).exp()



class Moments(UnivariateTest):
    """Moments-based test for normality checking up to k_max moments."""

    def __init__(
        self,
        k_max: int = 4,
        sampler: torch.distributions.Distribution = torch.distributions.Normal(
            torch.tensor([0.0]), torch.tensor([1.0])
        ),
    ):
        super().__init__(sorted=True)
        self.k_max = k_max
        self.sampler = sampler
        moments = []
        for i in range(2, k_max + 1, 2):
            moment_val = scipy_norm(loc=0, scale=1).moment(i)
            moments.append(moment_val)
        self.register_buffer("moments", torch.Tensor(moments).unsqueeze(1))
        self.register_buffer("weights", torch.arange(2, self.k_max + 1).neg().exp())

    def forward(self, x):
        x = self.prepare_data(x)
        k = torch.arange(2, self.k_max + 1, device=x.device, dtype=x.dtype).view(
            -1, 1, 1
        )
        m1 = self.dist_mean(x.mean(0)).abs_()
        if self.k_max >= 2:
            xpow = self.dist_mean((x**k).mean(1))
            xpow[::2].sub_(self.moments)
            m2 = xpow.abs_().T.matmul(self.weights)
            return m1.add_(m2) / self.world_size
        return m1 / self.world_size



class VCReg(UnivariateTest):
    """
    VCReg test statistic testing mean=0 and var=1.
    """

    def forward(self, x):
        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        stat_mean = (mean**2) / (var / n)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))
        stat = stat_mean + stat_var
        return stat



class ExtendedJarqueBera(UnivariateTest):
    """
    Extended Jarque-Bera test for goodness-of-fit to standard normal N(0,1).

    Tests all four moments: mean=0, variance=1, skewness=0, kurtosis=3.
    Under H0: X ~ N(0,1), the total statistic ~ chi^2(4).
    """

    def forward(self, x):
        n = x.shape[0]
        mean = x.mean(dim=0)
        var = x.var(dim=0, unbiased=True)
        std = var.sqrt().clamp(min=1e-8)
        skewness = ((x - mean) / std).pow(3).mean(dim=0)
        kurtosis = ((x - mean) / std).pow(4).mean(dim=0)

        stat_mean = (mean**2) / (var / n)
        stat_var = ((var - 1) ** 2) / (2 / (n - 1))
        stat_skew_kurt = n / 6 * (skewness**2 + 0.25 * (kurtosis - 3) ** 2)

        stat = stat_mean + stat_var + stat_skew_kurt
        return stat



class ShapiroWilk(UnivariateTest):
    """
    Shapiro-Wilk test for standard normality N(0,1).

    A correlation-based goodness-of-fit test that measures how well ordered
    sample values match the expected order statistics from a standard normal.
    """

    def __init__(
        self,
        expectation_mode: str = "elfving",
        covariance_mode: str = "shapiro_francia",
        eps: float = 1e-5,
        sorted: bool = False,
    ):
        super().__init__(eps=eps, sorted=sorted)
        self.expectation_mode = expectation_mode
        self.covariance_mode = covariance_mode
        self._k = None

    def forward(self, x):
        s = self.prepare_data(x)
        if self._k is None or self._k.size(0) != x.size(0):
            with torch.no_grad():
                self._k = self.get_shapiro_weights(
                    x.size(0),
                    expectation_mode=self.expectation_mode,
                    covariance_mode=self.covariance_mode,
                    device=x.device,
                )
        extra_dims = tuple([1] * (x.ndim - 1))
        k = self._k.view(x.size(0), *extra_dims)
        return (
            1 - torch.nn.functional.cosine_similarity(k, s, dim=0, eps=self.eps).abs()
        )

    @staticmethod
    def get_shapiro_weights(
        N,
        expectation_mode="blom",
        covariance_mode="shapiro_francia",
        device="cpu",
    ):
        """Compute Shapiro-Wilk weights for correlation with order statistics."""
        g = torch.distributions.normal.Normal(0, 1)
        grid = torch.arange(1, N + 1, dtype=torch.float, device=device)

        if expectation_mode == "elfving":
            pi = grid.sub_(torch.pi / 8).div_(N + 1 / 4)
        elif expectation_mode == "blom":
            pi = grid.sub_(3 / 8).div_(N + 1 / 4)
        elif expectation_mode == "rahman":
            pi = grid.div_(N + 1)
        else:
            raise ValueError(f"Unknown expectation_mode: {expectation_mode}")

        m = g.icdf(pi)

        if covariance_mode == "shapiro_francia":
            a = m
        elif covariance_mode == "rahman":
            phi = g.log_prob(m).exp_()
            a = phi.square().mul_(m).mul_(2)
            cross = phi[1:] * phi[:-1]
            a[:-1] -= m[1:] * cross
            a[1:] -= m[:-1] * cross
        else:
            raise ValueError(f"Unknown covariance_mode: {covariance_mode}")

        return torch.nn.functional.normalize(a, p=2, dim=0)



class NLL(UnivariateTest):
    """Negative log-likelihood based test for normality."""

    def __init__(self, alpha: float = 0.5, k: int = None, N: int = None):
        super().__init__()
        assert 0 <= alpha <= 0.5
        self.alpha = alpha
        assert k is None or type(k) is int
        assert N is None or type(N) is int
        self.k = k
        self.N = N
        self.g = torch.distributions.Normal(0, 1)
        self._cached = (-1, -1)

    @torch.no_grad()
    def get_cutoffs(self, device, ndim):
        assert self.N is not None
        if self.alpha == 0.5:
            return
        elif self._cached == (self.k, self.N):
            return self.cutoffs

        original_alpha = self.alpha
        original_k = self.k
        self.alpha = 0.5

        samples = torch.linspace(-7, 7, 1000, device=device)
        if self.k is None:
            candidates = range(1, self.N + 1)
        else:
            assert type(self.k) is int and self.k >= 1 and self.k <= self.N
            candidates = [self.k]

        density = torch.empty((len(candidates), len(samples)), device=device)
        for i, k in enumerate(candidates):
            self.k = k
            density[i] = self.forward(samples)

        density.negative_().exp_()
        density = torch.cumsum(
            (density[:, 1:] + density[:, :-1]).mul_((samples[1] - samples[0]) / 2), 1
        )
        cutoff_low = (density > original_alpha).float().argmax(1)
        cutoff_high = (density > 1 - original_alpha).float().argmax(1)
        cutoffs = torch.stack([samples[cutoff_low], samples[cutoff_high]], 0)
        self.cutoffs = cutoffs.view((2, len(candidates)) + (1,) * (ndim - 1))

        self.k = original_k
        self.alpha = original_alpha
        self._cached = (self.k, self.N)
        return self.cutoffs

    @torch.no_grad()
    def get_constants(self, device, ndim: int):
        assert self.N is not None
        top = torch.arange(1, self.N + 1).log().sum()
        if type(self.k) is int:
            assert self.k >= 1 and self.k <= self.N
            k_factors = torch.full((1,), self.k - 1, device=device, dtype=torch.float)
            bottom_left = torch.arange(self.k).log()
            bottom_left[0] = 0
            bottom_left = bottom_left.sum()
            bottom_right = torch.arange(1, self.N - self.k + 1, device=device).log()
            bottom_right = bottom_right.sum()
            N_m_k_factors = torch.full(
                (1,), self.N - self.k, device=device, dtype=torch.float
            )
        else:
            k_factors = torch.arange(self.N, device=device, dtype=torch.float)
            bottom_left = k_factors.log()
            bottom_left[0] = 0
            torch.cumsum(bottom_left, dim=0, out=bottom_left)
            bottom_right = bottom_left.flip(0)
            N_m_k_factors = k_factors.flip(0)
        cst = top - bottom_left - bottom_right

        extra_dims = ndim - 1
        cst = cst.view([-1] + [1] * extra_dims)
        k_factors = k_factors.view([-1] + [1] * extra_dims)
        N_m_k_factors = N_m_k_factors.view([-1] + [1] * extra_dims)
        return k_factors, N_m_k_factors, cst

    def forward(self, x):
        if torch.isnan(x).any():
            raise ValueError("Given input to the loss contains NaN!")
        if x.ndim < 1:
            raise ValueError(f"input should have at least one dim, got {x.ndim}")
        if self.N is None:
            N_was_None = True
            self.N = x.size(0)
        else:
            N_was_None = False

        cutoffs = self.get_cutoffs(x.device, x.ndim)
        k_factors, N_m_k_factors, cst = self.get_constants(x.device, x.ndim)

        if self.k is None:
            s, indices = torch.sort(x, dim=0)
        else:
            s = x

        logcdf = log_norm_cdf(s)
        one_m_logcdf = log_norm_cdf(-s)

        sample_loss = -(
            cst
            + logcdf.mul(k_factors)
            + one_m_logcdf.mul(N_m_k_factors)
            + self.g.log_prob(s)
        )
        if self.alpha < 0.5:
            assert cutoffs is not None
            mask = s.gt(cutoffs[0]).logical_and_(s.lt(cutoffs[1]))
            sample_loss[mask] = torch.nan
        if N_was_None:
            self.N = None

        if self.k is None:
            return torch.gather(sample_loss, dim=0, index=indices)
        return sample_loss



class SlicingUnivariateTest(torch.nn.Module):
    """
    Multivariate distribution test using random slicing and univariate test statistics.

    Extends univariate statistical tests to multivariate data by projecting
    samples onto random 1D directions and aggregating univariate test statistics.

    Args:
        univariate_test (torch.nn.Module): A univariate test module (e.g., EppsPulley).
        num_slices (int): Number of random 1D projections to use.
        reduction (str, optional): How to aggregate statistics: 'mean', 'sum', or None.
        sampler (str, optional): Random sampling method: 'gaussian'. Default: 'gaussian'.
        clip_value (float, optional): Minimum threshold for test statistics. Default: None.
    """

    def __init__(
        self,
        univariate_test,
        num_slices: int,
        reduction: str = "mean",
        sampler: str = "gaussian",
        clip_value: float = None,
    ):
        super().__init__()
        self.reduction = reduction
        self.num_slices = num_slices
        self.sampler = sampler
        self.univariate_test = univariate_test
        self.clip_value = clip_value
        self.register_buffer("global_step", torch.zeros((), dtype=torch.long))
        self._generator = None
        self._generator_device = None

    def _get_generator(self, device, seed):
        if self._generator is None or self._generator_device != device:
            self._generator = torch.Generator(device=device)
            self._generator_device = device
        self._generator.manual_seed(seed)
        return self._generator

    def forward(self, x):
        with torch.no_grad():
            global_step_sync = all_reduce(self.global_step.clone(), op="MAX")
            seed = global_step_sync.item()
            g = self._get_generator(x.device, seed)
            proj_shape = (x.size(-1), self.num_slices)
            A = torch.randn(proj_shape, device=x.device, generator=g)
            A /= A.norm(p=2, dim=0)
            self.global_step.add_(1)

        stats = self.univariate_test(x @ A)
        if self.clip_value is not None:
            stats[stats < self.clip_value] = 0
        if self.reduction == "mean":
            return stats.mean()
        elif self.reduction == "sum":
            return stats.sum()
        elif self.reduction is None:
            return stats



class BHEP(MultivariatetTest):
    """
    Beta-Henze Energy-based Projection (BHEP) test statistic.

    Computes the BHEP test statistic for multivariate normality testing
    using a Gaussian kernel with bandwidth parameter beta.

    Args:
        beta: Bandwidth parameter for the Gaussian kernel (must be > 0)
    """

    def __init__(self, beta: float = 0.1):
        super().__init__()
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        self.beta = beta

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        x = self.prepare_data(x)
        N, D = x.shape

        if N == 0:
            raise ValueError("Input data cannot be empty")

        beta_squared = self.beta**2
        squared_norms = x.square().sum(dim=1)
        pairwise_distances = (
            -2 * x @ x.T + squared_norms.unsqueeze(1) + squared_norms.unsqueeze(0)
        )

        lhs = torch.exp(pairwise_distances * (-beta_squared / 2)).sum() / (N**2)
        scaling_factor = 2 / ((1 + beta_squared) ** (D / 2))
        exponent = -beta_squared / (2 + 2 * beta_squared)
        rhs = scaling_factor * torch.exp(squared_norms * exponent).sum() / N
        constant = 1 / ((1 + 2 * beta_squared) ** (D / 2))

        statistic = lhs - rhs + constant
        return statistic

    def __repr__(self) -> str:
        return f"BHEP(beta={self.beta})"



class BHEP_M(MultivariatetTest):
    """Modified BHEP test with different formulation."""

    def __init__(self, dim=None, beta=10):
        super().__init__(dim=dim)
        assert beta > 2
        self.beta = beta

    def forward(self, x):
        x = self.prepare_data(x)
        _, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = (
            (1 / self.beta ** (D / 2))
            * torch.exp(pair_sim.div(4 * self.beta)).sum()
            / x.size(0)
        )
        rhs = (2 / (self.beta - 0.5) ** (D / 2)) * torch.exp(
            norms / (4 * self.beta - 2)
        ).sum()
        return lhs - rhs



class COMB(MultivariatetTest):
    """
    Combination-based (COMB) test statistic for multivariate normality.

    Uses a combination of exponential and cosine kernels.

    Args:
        gamma: Bandwidth parameter for the kernel (must be > 0)
    """

    def __init__(self, gamma: float = 0.1):
        super().__init__()
        if gamma <= 0:
            raise ValueError(f"gamma must be positive, got {gamma}")
        self.gamma = gamma

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        x = self.prepare_data(x)
        N, D = x.shape

        if N == 0:
            raise ValueError("Input data cannot be empty")

        squared_norms = x.square().sum(dim=1)
        norm_diff_matrix = squared_norms.unsqueeze(1) - squared_norms.unsqueeze(0)
        exp_term = torch.exp(norm_diff_matrix / (4 * self.gamma))

        inner_products = x @ x.T
        cos_term = torch.cos(inner_products / (2 * self.gamma))

        kernel_matrix = exp_term * cos_term
        kernel_mean = kernel_matrix.mean()

        statistic = np.sqrt(N) * (kernel_mean - 1)
        return statistic

    def __repr__(self) -> str:
        return f"COMB(gamma={self.gamma})"



class HV(MultivariatetTest):
    """HV test statistic for multivariate normality."""

    def __init__(self, gamma=1):
        super().__init__()
        assert gamma > 0
        self.gamma = gamma

    def forward(self, x):
        x = self.prepare_data(x)
        N, D = x.shape
        norms = x.square().sum(1)
        pair_sim = 2 * x @ x.T + norms + norms.unsqueeze(1)
        lhs = torch.exp(pair_sim.div(4 * self.gamma))
        rhs = (
            x @ x.T
            - pair_sim / (2 * self.gamma)
            + D / (2 * self.gamma)
            + pair_sim / (4 * self.gamma**2)
        )
        return (lhs * rhs).sum() / N



class HZ(MultivariatetTest):
    """
    Henze-Zirkler (HZ) test for multivariate normality.

    Uses an adaptive bandwidth selection rule. Unlike the standard BHEP test
    which requires manual bandwidth tuning, the HZ test automatically computes
    an optimal bandwidth parameter based on sample size and dimensionality.
    """

    def __init__(self):
        super().__init__()
        self._bhep = BHEP(beta=1.0)

    def _compute_bhep_statistic(self, x: torch.Tensor, beta: float) -> torch.Tensor:
        """Compute BHEP statistic with given bandwidth."""
        self._bhep.beta = beta
        return self._bhep.forward(x)

    @staticmethod
    def compute_bandwidth(n_samples: int, n_dims: int) -> float:
        """
        Compute the Henze-Zirkler optimal bandwidth parameter.

        Formula: beta = (1/sqrt(2)) * [(2D + 1) * N / 4]^(1/(D+4))
        """
        if n_samples <= 0:
            raise ValueError(f"n_samples must be a positive integer, got {n_samples}")
        if n_dims <= 0:
            raise ValueError(f"n_dims must be a positive integer, got {n_dims}")

        if n_samples < 10:
            import warnings
            warnings.warn(
                f"Sample size {n_samples} is very small (< 10). "
                "The Henze-Zirkler test may not be reliable.",
                UserWarning,
                stacklevel=2,
            )

        if n_dims / n_samples > 0.1:
            import warnings
            warnings.warn(
                f"Dimensionality ({n_dims}) is high relative to sample size "
                f"({n_samples}). Test power may be reduced.",
                UserWarning,
                stacklevel=2,
            )

        numerator = (2 * n_dims + 1) * n_samples
        exponent = 1.0 / (n_dims + 4)
        base = numerator / 4.0
        power_term = base**exponent
        beta = power_term / np.sqrt(2.0)

        if beta <= 0 or not np.isfinite(beta):
            raise RuntimeError(
                f"Computed invalid bandwidth beta = {beta}. "
                f"n_samples={n_samples}, n_dims={n_dims}"
            )

        return float(beta)

    def forward(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        x = self.prepare_data(x)
        N, D = x.shape

        if N == 0:
            raise ValueError("Input data cannot be empty.")

        if torch.isnan(x).any():
            raise ValueError("Input data contains NaN values.")

        if torch.isinf(x).any():
            raise ValueError("Input data contains infinite values.")

        optimal_beta = self.compute_bandwidth(n_samples=N, n_dims=D)
        statistic = self._compute_bhep_statistic(x, beta=optimal_beta)

        return statistic

    def __call__(self, x: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        return self.forward(x)

    def __repr__(self) -> str:
        return "HZ()"

    def __str__(self) -> str:
        return "Henze-Zirkler test with adaptive bandwidth"
