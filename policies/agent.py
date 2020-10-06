from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from collections.abc import Iterable

import inspect

class Agent:

    @abstractmethod
    def sample_action(self, obs):
        '''
        Returns a selected action given an observation
        '''
        pass

class ApproximatingAgent(Agent, nn.Module):

    def __init__(self, neurons = 128, activations = 'ReLU', out_activation = None, n_actions=None, n_features=None):

        super().__init__()

        # If neurons is not itreable make it iterable

        if not isinstance(neurons, Iterable):
            neurons = [neurons]

        # Convert to list (for ease later on)

        neurons = list(neurons)

        # Append input and output sizes if specified via the use of n_actions and n_features (state size)

        if n_actions is not None and isinstance(n_actions, int):
            neurons = neurons + [n_actions]

        if n_features is not None and isinstance(n_features, int):
            neurons = [n_features] + neurons

        # If only one activation function is given, use the same one for all layers except output one

        if not isinstance(activations, Iterable) or isinstance(activations, str):
            activations = [activations]*(len(neurons) - 2)

        # Add output activation

        activations += [out_activation]

        if len(activations) != (len(neurons) - 1):
            raise Exception(f'Incompatible {len(activations)} activations with {len(neurons) - 1} layers')

        # Build MLP

        self._net = nn.Sequential()

        for i, (n_inputs, n_outputs, activation) in enumerate(zip(
                neurons[:-1],
                (neurons[1:] + neurons[:1])[:-1],
                activations,
        )):
            # Add linear layer
            self._net.add_module(str(i) + '_layer', nn.Linear(n_inputs, n_outputs))

            # Add activation if not None
            if activation is not None:
                a_fun = getattr(nn, activation)
                self._net.add_module(
                    str(i) + f'_{activation}_activation',
                    a_fun(**(
                        dict() if 'dim' not in inspect.getargs(a_fun.__init__.__code__).args else dict(dim=-1)
                    )),
                )


    def forward(self, obs):
        return self._net(obs)


