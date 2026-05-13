"""
Soft red-list watermark for LLMs.

Based on:
  Kirchenbauer et al. "A Watermark for Large Language Models."
  ICML 2023. https://arxiv.org/abs/2301.10226
  Code: https://github.com/jwkirchenbauer/lm-watermarking

Algorithm 2 (soft watermark) and the z-score detector (Eq. 3) are
implemented here verbatim. The entropy-aware extension described in
the CS5788 project proposal is a separate subclass.
"""

import math
import torch
from transformers import LogitsProcessor


# ---------------------------------------------------------------------------
# Soft watermark – generation side (Algorithm 2 from Kirchenbauer et al.)
# ---------------------------------------------------------------------------

class WatermarkLogitsProcessor(LogitsProcessor):
    """
    Adds a fixed bias δ to the logits of green-list tokens at every step.

    Green-list membership for token t is determined by hashing the previous
    token s^(t-1) with a fixed key, following the paper's single-token
    context window (h=1).

    Args:
        vocab_size: size of the model vocabulary.
        gamma: fraction of vocab placed on the green list (default 0.25).
        delta: logit boost applied to green-list tokens (default 2.0).
        seeding_scheme: only "lefthash" (hash prev token) is supported.
        hash_key: integer salt mixed into every hash for a private key.
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.25,
        delta: float = 2.0,
        adaptive: bool = False,
        alpha: float = 1.0,
        delta_min: float = 0.0,
        seeding_scheme: str = "lefthash",
        hash_key: int = 15485863,  # the millionth prime
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.delta = delta
        self.adaptive = adaptive
        self.alpha = alpha        # exponent for entropy scaling: δ = δ_max * H^alpha
        self.delta_min = delta_min  # floor to guarantee z-score accumulation
        self.seeding_scheme = seeding_scheme
        self.hash_key = hash_key
        self.rng = torch.Generator()

    def _seed_rng(self, prev_token: int) -> None:
        seed = self.hash_key * prev_token
        self.rng.manual_seed(seed % (2**31 - 1))

    def _get_green_list(self, prev_token: int) -> torch.Tensor:
        self._seed_rng(prev_token)
        green_list_size = int(self.vocab_size * self.gamma)
        vocab_perm = torch.randperm(self.vocab_size, generator=self.rng)
        return vocab_perm[:green_list_size]

    def _normalized_entropy(self, logits: torch.Tensor) -> float:
        """
        Compute Shannon entropy of the token distribution, normalized to [0, 1].

        logits: raw scores for one token step, shape (vocab_size,)
        Returns 0.0 at minimum entropy (one token has all mass),
                1.0 at maximum entropy (uniform distribution).
        """
        # cast to float32 — float16 softmax overflows on GPU with large logit magnitudes
        probs = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(probs * torch.log(probs.clamp(min=1e-10)))
        max_entropy = math.log(self.vocab_size)
        return float(entropy.item() / max_entropy)

    def _adaptive_delta(self, logits: torch.Tensor, delta_min: float = 0.0) -> float:
        """
        Scale delta with normalized entropy raised to alpha, with optional floor.
        alpha=1.0 → linear (default adaptive).
        alpha<1.0 → concave, more aggressive at low entropy.
        delta_min  → floor so z-score always accumulates (used for forced_adp).
        """
        return max(self.delta * (self._normalized_entropy(logits) ** self.alpha), delta_min)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor
    ) -> torch.FloatTensor:
        """Apply green-list bias; called by HuggingFace generate() at each step."""
        batch_size = input_ids.shape[0]
        for b in range(batch_size):
            prev_token = int(input_ids[b, -1].item())
            green_list = self._get_green_list(prev_token)
            # Depending on flag, add regular delta or adaptive delta
            bias = self._adaptive_delta(scores[b], self.delta_min) if self.adaptive else self.delta
            scores[b, green_list] += bias
        return scores


# ---------------------------------------------------------------------------
# Detector – z-score test (Equation 3 from Kirchenbauer et al.)
# ---------------------------------------------------------------------------

class WatermarkDetector:
    """
    Detects the soft watermark via a one-proportion z-test.

    Args:
        vocab_size: must match the processor used at generation.
        gamma: green-list fraction (must match generation).
        hash_key: salt (must match generation).
        z_threshold: reject H0 if z > threshold (default 4.0 → FPR ≈ 3e-5).
    """

    def __init__(
        self,
        vocab_size: int,
        gamma: float = 0.25,
        hash_key: int = 15485863,
        z_threshold: float = 4.0,
    ):
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.hash_key = hash_key
        self.z_threshold = z_threshold
        self.rng = torch.Generator()

    def _seed_rng(self, prev_token: int) -> None:
        seed = self.hash_key * prev_token
        self.rng.manual_seed(seed % (2**31 - 1))

    def _get_green_list(self, prev_token: int) -> set:
        self._seed_rng(prev_token)
        green_list_size = int(self.vocab_size * self.gamma)
        vocab_perm = torch.randperm(self.vocab_size, generator=self.rng)
        return set(vocab_perm[:green_list_size].tolist())

    def detect(self, token_ids: list[int]) -> dict:
        """
        Score a token sequence.

        Returns a dict with keys:
          num_tokens, green_token_count, z_score, is_watermarked
        """
        if len(token_ids) < 2:
            return {"num_tokens": 0, "green_token_count": 0, "z_score": 0.0, "is_watermarked": False}

        green_count = 0
        T = len(token_ids) - 1  # first token has no predecessor to hash
        for i in range(1, len(token_ids)):
            prev = token_ids[i - 1]
            tok = token_ids[i]
            if tok in self._get_green_list(prev):
                green_count += 1

        # Eq. 3 from the paper
        gamma = self.gamma
        z = (green_count - gamma * T) / (T * gamma * (1 - gamma)) ** 0.5

        return {
            "num_tokens": T,
            "green_token_count": green_count,
            "z_score": z,
            "is_watermarked": z > self.z_threshold,
        }
