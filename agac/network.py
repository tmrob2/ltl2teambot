import numpy as np


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1.0 / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def initialize_hidden_layer(layer, b_init_value=0.1):
    fanin_init(layer.weight)
    layer.bias.data.fill_(b_init_value)


def initialize_last_layer(layer, init_w=1e-3):
    layer.weight.data.uniform_(-init_w, init_w)
    layer.bias.data.uniform_(-init_w, init_w)


class DiscreteActor(RLModel):
    def __init__(
        self,
        observation_dim: Tuple[int],
        action_dim: int,
        layers_dim: List[int],
        cnn_extractor: bool = False,
        layers_num_channels: List[int] = None,
    ):
        """
        Simple Discrete Actor with or without CNN extractor.
        observation_dim, Tuple: shape of environment's observation
        """
        super(DiscreteActor, self).__init__()

        if cnn_extractor:
            core_cnn = CNN(
                observation_dim, layers_num_channels, stride=2, kernel_size=3
            )
            flattened_dim = (
                observation_dim[1] * observation_dim[2] * layers_num_channels[-1]
            )
            core_mlp = MLP(flattened_dim, action_dim, layers_dim)
            layers = core_cnn.layers + core_mlp.layers
            self._core = nn.Sequential(core_cnn, nn.Flatten(), core_mlp)

        else:
            self._core = MLP(observation_dim[0], action_dim, layers_dim)
            layers = self._core.layers

        for layer in layers[:-1]:
            initialize_hidden_layer(layer)
        initialize_last_layer(layers[-1])

    def forward(
        self, x, deterministic: bool = False, return_log_prob: bool = False
    ) -> Tuple[Action, LogProb, Prob, LogProb]:
        m = self.compute_distribution(x)
        if deterministic:
            action = torch.argmax(m.probs, dim=-1)
        else:
            action = m.sample()

        if return_log_prob:
            log_prob = m.log_prob(action)
            probs = m.probs
            log_probs = m.logits
        else:
            log_probs, log_prob = None, None
        return action, log_prob, probs, log_probs

    def compute_distribution(self, observations: Observation) -> Categorical:
        logits = self._core(observations)
        probs = nn.Softmax(dim=-1)(logits)
        return Categorical(probs=probs)

    def compute_log_prob(self, observations: Observation, actions: Action) -> LogProb:
        # forward pass
        distribution = self.compute_distribution(observations)
        return distribution.log_prob(actions)

    def select_action(
        self, observation: Observation, deterministic: bool = True
    ) -> Action:
        action = (
            self(observation, deterministic=deterministic)[0]
            .cpu()
            .data.numpy()
            .flatten()
        )
        return int(action)