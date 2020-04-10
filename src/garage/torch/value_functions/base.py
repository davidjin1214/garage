"""Base class for all baselines."""
import abc


class ValueFunction(abc.ABC):
    """Base class for all baselines.

    Args:
        env_spec (garage.envs.env_spec.EnvSpec): Environment specification.
        name (str): Value function name, also the variable scope.

    """

    def __init__(self, env_spec, name):
        self._mdp_spec = env_spec
        self.name = name

    @abc.abstractmethod
    def get_param_values(self):
        """Get parameter values.

        Returns:
            List[np.ndarray]: A list of values of each parameter.

        """

    @abc.abstractmethod
    def set_param_values(self, flattened_params):
        """Set param values.

        Args:
            flattened_params (np.ndarray): A numpy array of parameter values.

        """

    @abc.abstractmethod
    def compute_loss(self, obs, returns):
        r"""Compute mean value of loss.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(N \dot [T], O*)`.
            returns (torch.Tensor): Acquired returns with shape :math:`(N, )`.

        Returns:
            torch.Tensor: Calculated negative mean scalar value of
                objective (float).

        """

    @abc.abstractmethod
    def predict(self, obs):
        r"""Predict value based on paths.

        Args:
            obs (torch.Tensor): Observation from the environment
                with shape :math:`(P, O*)`.

        Returns:
            torch.Tensor: Calculated baselines given observations with
                shape :math:`(P, O*)`.

        """

    def log_diagnostics(self, paths):
        """Log diagnostic information.

        Args:
            paths (list[dict]): A list of collected paths.

        """
