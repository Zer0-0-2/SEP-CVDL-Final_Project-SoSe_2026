import torch
import torch.nn.functional as F


class OODGate:
    """Wraps a classifier and decides whether to accept or reject its prediction.

    Two modes (set via cfg.pipeline.ood_gate):
      - softmax_threshold: reject if max(softmax(logits)) < threshold
      - energy:            reject if -log(sum(exp(logits / T))) > threshold
                           (note: energy score has opposite polarity — lower is more in-distribution)
    """

    def __init__(self, cfg):
        self.mode = cfg.pipeline.ood_gate
        self.threshold = cfg.ood.threshold
        self.temperature = cfg.ood.temperature

    def __call__(self, logits: torch.Tensor) -> int:
        """Accept or reject a single prediction.

        Args:
            logits: raw model output, shape [num_classes] or [1, num_classes]

        Returns:
            Predicted class index (0-19), or -1 if rejected.
        """
        logits = logits.squeeze(0)  # ensure 1-D

        if self.mode == "softmax_threshold":
            probs = F.softmax(logits, dim=0)
            confidence = probs.max().item()
            if confidence < self.threshold:
                return -1
            return probs.argmax().item()

        elif self.mode in ("energy", "temperature_scaling"):
            energy = -self.temperature * torch.logsumexp(logits / self.temperature, dim=0).item()
            if energy > self.threshold:
                return -1
            return logits.argmax().item()

        else:
            raise ValueError(f"Unknown ood_gate mode: {self.mode!r}")
