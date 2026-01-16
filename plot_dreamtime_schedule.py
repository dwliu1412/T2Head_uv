import math
import matplotlib.pyplot as plt


def dreamtime_t(k: int, N: int, t_min: float, t_max: float, l: int = 100) -> float:
    """Compute t(k) following the formula:

        t(k) = t_max - (t_max - t_min) * log(1 + floor(k / l) * l / N)

    where
      - k: current training step index (1-based or 0-based is fine; we treat 0 as 0)
      - N: total training steps
      - t_min, t_max: lower/upper bounds of the time step range (actual time-step indices)
      - l: interval length used inside floor(k / l)

    This is a direct implementation for visualization purposes.
    """
    if N <= 0:
        raise ValueError("N must be positive")
    if l <= 0:
        raise ValueError("l must be positive")

    # clamp k to [0, N]
    k_clamped = max(0, min(k, N))
    block = (k_clamped // l) * l  # floor(k/l) * l
    inner = 1.0 + block / float(N)
    value = t_max - (t_max - t_min) * math.log2(inner)
    return value


def main():
    # Example parameters: you can modify these to match your experiment
    N = 10000          # total training steps
    # Use actual time-step indices on the vertical axis
    t_min = 50.0       # minimal diffusion time-step index
    t_max = 800.0      # maximal diffusion time-step index
    l = 1000            # block size in steps

    ks = list(range(0, N + 1))
    ts = [dreamtime_t(k, N, t_min, t_max, l) for k in ks]

    plt.figure(figsize=(8, 4))
    plt.plot(ks, ts, label=r"$t(k)$ schedule")
    plt.xlabel("Training step k")
    plt.ylabel("t (diffusion time-step index)")
    plt.title("DreamTime / DTC time-step curriculum (t in [50, 800])")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
