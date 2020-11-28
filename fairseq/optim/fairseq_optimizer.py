# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import numpy
import torch
from fairseq import utils
from fairseq.dataclass.utils import gen_parser_from_dataclass
import matplotlib.pyplot as plt
import torch
import numpy as np
from matplotlib.lines import Line2D


class FairseqOptimizer(object):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.counter = 0

    @classmethod
    def add_args(cls, parser):
        """Add optimizer-specific arguments to the parser."""
        dc = getattr(cls, "__dataclass", None)
        if dc is not None:
            gen_parser_from_dataclass(parser, dc())

    @property
    def optimizer(self):
        """Return a torch.optim.optimizer.Optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """Reset optimizer instance."""
        if not hasattr(self, "_optimizer"):
            raise NotImplementedError
        if not isinstance(self._optimizer, torch.optim.Optimizer):
            raise ValueError("_optimizer must be an instance of torch.optim.Optimizer")
        self._optimizer = optimizer

    @property
    def optimizer_config(self):
        """
        Return a kwarg dictionary that will be used to override optimizer
        args stored in checkpoints. This allows us to load a checkpoint and
        resume training using a different set of optimizer args, e.g., with a
        different learning rate.
        """
        raise NotImplementedError

    @property
    def params(self):
        """Return an iterable of the parameters held by the optimizer."""
        for param_group in self.param_groups:
            for p in param_group["params"]:
                yield p

    @property
    def param_groups(self):
        return self.optimizer.param_groups

    def __getstate__(self):
        return self._optimizer.__getstate__()

    def get_lr(self):
        """Return the current learning rate."""
        return self.param_groups[0]["lr"]

    def set_lr(self, lr):
        """Set the learning rate."""
        for param_group in self.param_groups:
            param_group["lr"] = lr

    def state_dict(self):
        """Return the optimizer's state dict."""
        return self.optimizer.state_dict()

    def load_state_dict(self, state_dict, optimizer_overrides=None):
        """Load an optimizer state dict.

        In general we should prefer the configuration of the existing optimizer
        instance (e.g., learning rate) over that found in the state_dict. This
        allows us to resume training from a checkpoint using a new set of
        optimizer args.
        """
        self.optimizer.load_state_dict(state_dict)

        if optimizer_overrides is not None and len(optimizer_overrides) > 0:
            # override learning rate, momentum, etc. with latest values
            for group in self.param_groups:
                group.update(optimizer_overrides)

    def backward(self, loss):
        """Computes the sum of gradients of the given tensor w.r.t. graph leaves."""
        loss.backward()

    def plot_grad_flow(self, model):
        '''Plots the gradients flowing through different layers in the net during training.
        Can be used for checking for possible gradient vanishing / exploding problems.

        Usage: Plug this function in Trainer class after loss.backwards() as
        "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
        ave_grads = []
        max_grads = []
        shapes = []
        layers = []
        names = []
        for n, p in model.named_parameters():
            if "bias" not in n and "encoder" in n:
                if (p.requires_grad) and ("bias" not in n):
                    layers.append(p)
                    names.append(n)
                    shapes.append(p.shape)
                    ave_grads.append(p.grad.abs().mean())
                    max_grads.append(p.grad.abs().max())
                if "mask.loga" in n:
                    print("mask_grad:", p.grad)
                    print("p:",  p)
        fig = plt.figure()
        myplot = fig.add_subplot()
        myplot.bar(np.arange(len(max_grads)), max_grads, alpha=1, lw=1, color="c")
        myplot.bar(np.arange(len(max_grads)), ave_grads, alpha=1, lw=1, color="b")
        myplot.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
        #plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
        #myplot.set_xticks(range(0, len(ave_grads), 10))#rotation="vertical")
        myplot.set_xticks(range(0, len(ave_grads), 10), layers) #rotation="vertical")
        myplot.set_xlim(left=0, right=len(ave_grads))
        myplot.set_ylim(bottom=-0.01, top=0.25)  # zoom in on the lower gradient regions
        myplot.set_xlabel("Layers")
        myplot.set_ylabel("average gradient")
        myplot.set_title("Gradient flow")
        myplot.grid(True)
        myplot.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
        #plt.show()
        #name = "/mount/arbeitsdaten48/projekte/mt/foeldeni/pngs/grads/" + str(self.counter) + '.png'
        #fig.savefig(name)
        #plt.close(fig)

    def plot_weight_histogram(self, model):
        for n, p in model.named_parameters():
            param = p.cpu()
            param = param.detach().numpy()
            fig = plt.figure()
            myplot = fig.add_subplot()
            myplot.hist(param, bins=numpy.arange(-0.25, 0.25, 0.01))
            myplot.set_xlabel(n)
            #plt.show()
            name = "/mount/arbeitsdaten48/projekte/mt/foeldeni/pngs/weights/" + str(n) + str(self.counter) + '.png'
            fig.savefig(name)
            plt.close(fig)

    def plot_loss(self):
        fig = plt.figure()
        myplot = fig.add_subplot()
        myplot.plot(self.loss, linewidth=0.2)
        myplot.set_xlabel('loss')
        name = "/mount/arbeitsdaten48/projekte/mt/foeldeni/pngs/loss/" + str(self.counter) + '.png'
        fig.savefig(name)
        plt.close(fig)

    def all_reduce_grads(self, module):
        """Manually all-reduce gradients (if required)."""
        if hasattr(module, "all_reduce_grads"):
            module.all_reduce_grads()

    def multiply_grads(self, c):
        """Multiplies grads by a constant *c*."""
        for p in self.params:
            if p.grad is not None:
                p.grad.data.mul_(c)

    def clip_grad_norm(self, max_norm, aggregate_norm_fn=None):
        """Clips gradient norm."""
        return utils.clip_grad_norm_(self.params, max_norm, aggregate_norm_fn)

    def step(self, closure=None, scale=1.0):
        """Performs a single optimization step."""
        if self.supports_step_with_scale:
            self.optimizer.step(closure, scale=scale)
        else:
            if scale != 1.0:
                self.multiply_grads(1.0 / scale)
            self.optimizer.step(closure)

    def zero_grad(self):
        """Clears the gradients of all optimized parameters."""
        for p in self.params:
            p.grad = None
        self.optimizer.zero_grad()

    @property
    def supports_memory_efficient_fp16(self):
        if hasattr(self.optimizer, "supports_memory_efficient_fp16"):
            return self.optimizer.supports_memory_efficient_fp16
        return False

    @property
    def supports_step_with_scale(self):
        if hasattr(self.optimizer, "supports_step_with_scale"):
            return self.optimizer.supports_step_with_scale
        return False

    @property
    def supports_flat_params(self):
        """
        Whether the optimizer supports collapsing of the model
        parameters/gradients into a single contiguous Tensor.
        """
        if hasattr(self.optimizer, "supports_flat_params"):
            return self.optimizer.supports_flat_params
        return False

    def average_params(self):
        pass

    def broadcast_global_state_dict(self, state_dict):
        """
        Broadcasts a global state dict to all ranks.
        Useful for optimizers that shard state between ranks.
        """
        if hasattr(self.optimizer, "broadcast_global_state_dict"):
            return self.optimizer.broadcast_global_state_dict(state_dict)
        else:
            return state_dict


class LegacyFairseqOptimizer(FairseqOptimizer):
    def __init__(self, args):
        self.args = args
