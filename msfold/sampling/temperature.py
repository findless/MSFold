"""Temperature ladder generation and adaptive update."""

import torch


def non_uniform_log_space_levels(
    T_min: float,
    T_max: float,
    N: int,
    concentration: float = 1.0
):
    """Generate non-uniform log-spaced temperature levels.

    A concentration > 1.0 concentrates more levels in the low-temperature
    region, which is recommended for multi-state proteins.

    Args:
        T_min: Minimum temperature.
        T_max: Maximum temperature.
        N: Number of temperature levels.
        concentration: Density factor (>1 = dense at low T, <1 = dense at high T).

    Returns:
        levels: torch.Tensor shape [N], temperatures in descending order.
        intervals: torch.Tensor shape [N-1], adjacent temperature differences.
    """
    if concentration <= 0:
        raise ValueError("concentration must be positive.")

    normalized_space = torch.linspace(0, 1, steps=N)
    curved_space = normalized_space.pow(concentration)

    log_t_min = torch.log(torch.tensor(T_min, dtype=torch.float32))
    log_t_max = torch.log(torch.tensor(T_max, dtype=torch.float32))

    log_levels = log_t_min + curved_space * (log_t_max - log_t_min)
    levels = torch.exp(log_levels)
    levels = torch.flip(levels, dims=[0])
    intervals = levels[:-1] - levels[1:]

    return levels, intervals


def log_space_levels(T_min: float, T_max: float, N: int):
    """Generate uniform log-spaced temperature levels.

    Args:
        T_min: Minimum temperature.
        T_max: Maximum temperature.
        N: Number of temperature levels.

    Returns:
        levels: torch.Tensor shape [N], temperatures in descending order.
        intervals: torch.Tensor shape [N-1], adjacent temperature differences.
    """
    levels = torch.exp(torch.linspace(
        torch.log(torch.tensor(T_min, dtype=torch.float32)),
        torch.log(torch.tensor(T_max, dtype=torch.float32)),
        steps=N
    ))
    levels = torch.flip(levels, dims=[0])
    intervals = levels[:-1] - levels[1:]
    return levels, intervals


def scaling_norm(T: torch.Tensor, anchor_min: float, anchor_max: float) -> torch.Tensor:
    """Linearly rescale temperatures to [anchor_min, anchor_max].

    Args:
        T: Original temperature schedule (high-to-low), 1D tensor.
        anchor_min: Target minimum temperature.
        anchor_max: Target maximum temperature.

    Returns:
        Rescaled temperature schedule.
    """
    T_orig_max = T.max()
    T_orig_min = T.min()

    if T_orig_max == T_orig_min:
        return T

    scale = (anchor_max - anchor_min) / (T_orig_max - T_orig_min)
    return anchor_min + (T - T_orig_min) * scale


def exp_normalize_temperature_preserve_order(
    T: torch.Tensor, T_min: float, T_max: float
) -> torch.Tensor:
    """Exponential normalization of temperatures to [T_min, T_max].

    Args:
        T: Original temperature schedule (high-to-low), 1D tensor.
        T_min: Target minimum temperature.
        T_max: Target maximum temperature.

    Returns:
        Exponentially normalized temperature schedule.
    """
    T_orig_max = T.max()
    T_orig_min = T.min()
    scale = (T_orig_max - T_orig_min).clamp(min=1e-8)
    T_std = (T - T_orig_min) / scale
    ratio = T_max / T_min
    return T_min * (ratio ** T_std)


def sum_exponentials_after_inclusive(x: torch.Tensor) -> torch.Tensor:
    """Compute suffix cumulative sum of exp(x), with a trailing zero.

    Used to reconstruct temperature levels from log-intervals.

    Args:
        x: 1D tensor of log-intervals, shape [n].

    Returns:
        1D tensor shape [n+1], where result[i] = sum(exp(x[i:])).
    """
    exp_x = torch.exp(x)
    cumsum_rev_inclusive = torch.cumsum(exp_x.flip(0), dim=0).flip(0)
    result = torch.empty(x.shape[0] + 1, dtype=x.dtype, device=x.device)
    result[:-1] = cumsum_rev_inclusive
    result[-1] = 0
    return result
