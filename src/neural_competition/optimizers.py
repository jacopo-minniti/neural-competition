"""The small Muon pilot uses Adam on the one-dimensional parameters."""

import torch


class MuonAdam:
    """Muon for incoming weights, Adam for hidden biases/output weights.

    All parameters remain trainable. Both components have zero weight decay.
    Muon uses PyTorch's original aspect-ratio adjustment and five NS iterations.
    """

    def __init__(self, model, spec):
        if not hasattr(torch.optim, "Muon"):
            raise RuntimeError("The optional Muon experiment needs a PyTorch with optim.Muon")
        self.muon = torch.optim.Muon(
            [model.weight],
            lr=spec.lr,
            momentum=spec.momentum,
            weight_decay=0,
            nesterov=True,
            adjust_lr_fn="original",
        )
        self.adam = torch.optim.Adam(
            [model.bias, model.output], lr=spec.auxiliary_lr or 0.001, weight_decay=0
        )

    @property
    def state(self):
        return {**self.muon.state, **self.adam.state}

    def zero_grad(self, set_to_none=True):
        self.muon.zero_grad(set_to_none=set_to_none)
        self.adam.zero_grad(set_to_none=set_to_none)

    def step(self):
        self.muon.step()
        self.adam.step()

    def state_dict(self):
        return {"muon": self.muon.state_dict(), "adam": self.adam.state_dict()}
