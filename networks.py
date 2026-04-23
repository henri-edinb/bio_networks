from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Sequence, Optional
import copy
from functools import partial
import numpy as np
import jax.numpy as jnp
import networkx as nx
from scipy.special import expit
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn import ensemble
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.util import view_as_blocks, view_as_windows
from scipy.stats import entropy


# lil-net
class LateralInhibitoryLayer:
    """
    A network that implements the original lateral inhibition algorithm to learn a factorized representation
    of some input by updating both weights using local plasticity rules.

    """

    @staticmethod
    def get_matrices():
        """
        Returns the list of matrices that are used in the network
        """
        return ["w", "m", "p"]

    @staticmethod
    def default_hparams(input_shape):
        hparams = dict()
        hparams["name"] = "lil-net"
        hparams["input_shape"] = input_shape
        hparams["num_units_y"] = 500
        hparams["num_blocks"] = 1
        hparams["step"] = 0.01
        hparams["num_steps"] = 1000
        hparams["multisteps"] = 1
        hparams["activation"] = "relu"
        hparams["threshold_dynamics"] = 0.0  # should be 0 for pure relu
        hparams["gradient"] = "voltage"  # voltage, rate
        hparams["spont_activity"] = 0.0
        hparams["bias"] = "zero_vector"
        hparams["threshold_learning"] = 0.0
        hparams["bcm_exprss"] = "y_mean"
        hparams["noisy_lr_std"] = 0.001
        hparams["dynamic_lr_exprss"] = "pehlevan_dynamic_lr"
        hparams["dynamic_lr_init"] = 1000
        hparams["foldiak_constant"] = 1.0
        hparams["zero_sequence"] = "zero"
        hparams["zero_epoch"] = "none"
        hparams["modulator"] = "self.y"
        hparams["memory_size"] = 32
        hparams["batch_size"] = 32
        hparams["w_init"] = "random"
        hparams["w_seed"] = np.random.randint(0, 100000)
        hparams["w_update"] = "bcm_abs"
        hparams["w_update_sleep"] = "bcm_abs"
        hparams["w_lr"] = 0.01
        hparams["w_noisy"] = False
        hparams["w_dynamic"] = False
        hparams["w_rectify"] = True
        hparams["w_kroned"] = "no"
        hparams["m_init"] = "random"
        hparams["m_seed"] = np.random.randint(0, 100000)
        hparams["m_update"] = "hebbian"
        hparams["m_update_sleep"] = "hebbian"
        hparams["m_lr"] = 0.01
        hparams["m_noisy"] = False
        hparams["m_dynamic"] = False
        hparams["m_rectify"] = True
        hparams["m_zero_dig"] = False
        hparams["m_bandwidth"] = (
            None  # None = full matrix; int = enforce banded structure
        )
        hparams["m_kroned"] = "no"
        hparams["p_init"] = "zero"
        hparams["p_seed"] = np.random.randint(0, 100000)
        hparams["p_update"] = "hebbian"
        hparams["p_update_sleep"] = "hebbian"
        hparams["p_lr"] = 0.01
        hparams["p_noisy"] = False
        hparams["p_dynamic"] = False
        hparams["p_rectify"] = True
        hparams["p_kroned"] = "no"
        hparams["p_zero_dig"] = False
        hparams["use_jax"] = "off"
        hparams["sleep_sparsity"] = 0.1
        return hparams

    def __init__(self, hparams):
        self.name = hparams["name"]

        self.y = np.zeros(hparams["num_units_y"])
        self.previous_y = np.zeros(hparams["num_units_y"])

        self.iteration = 1

        self.num_units_y = hparams["num_units_y"]
        self.num_assemblies = hparams["num_blocks"]
        assert (self.num_units_y % self.num_assemblies) == 0
        self.inp_shape = hparams["input_shape"]
        self.inp_size = np.prod(hparams["input_shape"])

        self.y_activity_track = np.zeros((hparams["num_steps"], self.num_units_y))

        self.activation = hparams["activation"]
        self.threshold = hparams["threshold_dynamics"]

        self.zero_sequence = hparams["zero_sequence"]
        self.zero_epoch = hparams["zero_epoch"]

        self.bias_expression = hparams["bias"]

        self.track_input = np.zeros(self.inp_size)
        self.track_y = np.zeros(self.num_units_y)
        self.track_y_squared = np.zeros(self.num_units_y)

        self.memory_size = hparams["memory_size"]

        self.wx_raw_window = np.zeros((hparams["num_units_y"], hparams["memory_size"]))
        self.py_raw_window = np.zeros((hparams["num_units_y"], hparams["memory_size"]))
        self.y_window = np.zeros((hparams["num_units_y"], hparams["memory_size"]))

        self.y_gradient_track = list()
        self.y_sparsity_track = list()
        self.y_entropy_track = list()

        self.init_w = hparams["w_init"]
        self.init_p = hparams["p_init"]
        self.init_m = hparams["m_init"]

        np.random.seed(hparams["w_seed"])
        if self.init_w == "random":
            self.w = np.random.rand(self.num_units_y, self.inp_size)
        elif self.init_w == "zero":
            self.w = np.zeros((self.num_units_y, self.inp_size))
        else:
            raise Exception("No valid initial value for W provided")

        np.random.seed(hparams["p_seed"])
        if self.init_p == "random":
            self.p = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_p == "zero":
            self.p = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception("No valid initial value for P provided")

        np.random.seed(hparams["m_seed"])
        if self.init_m == "random":
            self.m = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_m == "posdef":
            B = np.random.randn(self.num_units_y, self.num_units_y)
            self.m = B.T @ B
        elif self.init_m == "identity":
            self.m = np.identity(self.num_units_y)
        elif self.init_m == "zero":
            self.m = np.zeros((self.num_units_y, self.num_units_y))
        elif self.init_m == "WWT":
            self.m = self.w @ self.w.T
        else:
            raise Exception("No valid initial value for M provided")

        # DYNAMIC LEARNING RATES
        self.threshold_learning = hparams["threshold_learning"]
        self.dyn_lr_init = hparams["dynamic_lr_init"]
        self.pehvelan_lr = np.zeros(self.num_units_y)
        self.pehvelan_lr.fill(self.dyn_lr_init)
        self.dynamic_lr_increment_exprss = hparams["dynamic_lr_exprss"]
        self.modulate_stable_y_exprss = hparams["modulator"]

        # BCM
        self.bcm_trsh_exp = hparams["bcm_exprss"]

        #
        self.foldiak_constant = hparams["foldiak_constant"]

        self.num_steps = hparams["num_steps"]
        self.multisteps = hparams["multisteps"]
        self.step_size_y = hparams["step"]
        self.gradient = hparams["gradient"]

        self.spont_act = hparams["spont_activity"]

        self.lr_w = hparams["w_lr"]
        self.lr_p = hparams["p_lr"]
        self.lr_m = hparams["m_lr"]

        self.rule_w = hparams["w_update"]
        self.rule_p = hparams["p_update"]
        self.rule_m = hparams["m_update"]

        self.dynamic_w = hparams["w_dynamic"]
        self.dynamic_p = hparams["p_dynamic"]
        self.dynamic_m = hparams["m_dynamic"]

        self.noisy_w = hparams["w_noisy"]
        self.noisy_p = hparams["p_noisy"]
        self.noisy_m = hparams["m_noisy"]

        self.noise_std_w = hparams["noisy_lr_std"]
        self.noise_std_p = hparams["noisy_lr_std"]
        self.noise_std_m = hparams["noisy_lr_std"]

        self.rectify_w = hparams["w_rectify"]
        self.rectify_p = hparams["p_rectify"]
        self.rectify_m = hparams["m_rectify"]

        self.kroned_w = hparams["w_kroned"]
        self.kroned_p = hparams["p_kroned"]
        self.kroned_m = hparams["m_kroned"]

        self.zero_dig_p = hparams["p_zero_dig"]
        self.zero_dig_m = hparams["m_zero_dig"]
        self.bandwidth_m = hparams.get("m_bandwidth", None)
        if self.bandwidth_m is not None:
            k = int(self.bandwidth_m)
            i_idx, j_idx = np.indices((self.num_units_y, self.num_units_y))
            self._m_band_mask = np.abs(i_idx - j_idx) <= k  # True = inside band
        else:
            self._m_band_mask = None

        self.sleep_sparsity = hparams["sleep_sparsity"]

        self.batch_size = hparams["batch_size"]

        if "use_jax" in hparams:
            self.use_jax = hparams["use_jax"]
        else:
            self.use_jax = "off"

        self.matrices = ["w", "p", "m"]

    def set_learned_params(
        self,
        saved_w,
        saved_m,
        saved_p,
        saved_track_input,
        saved_track_y,
        saved_track_y_squared,
        saved_iterations,
    ):
        self.iteration = saved_iterations
        self.track_input = saved_track_input
        self.track_y = saved_track_y
        self.track_y_squared = saved_track_y_squared

        self.w = saved_w
        self.m = saved_m
        self.p = saved_p

    def gradient_step(self, add_feedback_y, bias_correction_y):
        gradient_y = (
            add_feedback_y
            - (self.m @ self.y)
            + (np.random.normal(loc=0.0, scale=self.spont_act, size=(self.num_units_y)))
            - bias_correction_y
            - self.threshold
        )
        self.y += self.step_size_y * gradient_y

        if self.activation == "relu":
            self.y = np.maximum(self.y, 0.0)
        elif self.activation == "sigmoid":
            self.y = 1 / (1 + np.exp(-self.y))
        elif self.activation == "linear":
            pass
        elif self.activation == "exp":
            self.y = np.exp(self.y)
        elif self.activation == "tanh":
            self.y = np.tanh(self.y)
        else:
            raise Exception("No valid activation function provided")

        return np.sum(np.abs(gradient_y))

    def gradient_step_rate(self, add_feedback_y, bias_correction_y):
        gradient_y = (
            add_feedback_y
            - (self.m @ self.y)
            + (np.random.normal(loc=0.0, scale=self.spont_act, size=(self.num_units_y)))
            - self.threshold
        )

        if self.activation == "relu":
            gradient_y = np.maximum(gradient_y, 0.0)
        elif self.activation == "sigmoid":
            gradient_y = 1 / (1 + np.exp(-gradient_y))
        elif self.activation == "linear":
            pass
        elif self.activation == "exp":
            gradient_y = np.exp(gradient_y)
        elif self.activation == "tanh":
            gradient_y = np.tanh(gradient_y)
        else:
            raise Exception("No valid activation function provided")

        gradient_y = gradient_y - bias_correction_y
        self.y += self.step_size_y * gradient_y

        return np.sum(np.abs(gradient_y))

    def projection(self):
        if self.kroned_w == "yin":
            original_squared_image_size = int(np.sqrt(self.inp_size))
            num_blocks_per_side = np.sqrt(self.num_assemblies)
            assert (
                num_blocks_per_side.is_integer()
            ), "Number of blocks per side is not an integer"
            num_blocks_per_side = int(num_blocks_per_side)
            block_size = int(original_squared_image_size / num_blocks_per_side)
            frame_block_single_vector = self.w.transpose() @ self.y
            original_blocks = np.reshape(
                frame_block_single_vector,
                (num_blocks_per_side, num_blocks_per_side, block_size, block_size),
            )
            list_of_rows = []
            for i in range(0, num_blocks_per_side):
                row = original_blocks[i][0]
                for j in range(1, num_blocks_per_side):
                    row = np.concatenate((row, original_blocks[i][j]), axis=1)
                list_of_rows.append(row)
            np.concatenate(list_of_rows, axis=0)
            image_rec = np.concatenate(list_of_rows, axis=0)
            return np.reshape(image_rec, self.inp_shape)
        elif self.kroned_w == "delayed":
            projection_flat = self.w.transpose() @ self.y
            return np.reshape(projection_flat[-self.frame_size :], self.inp_shape)
        else:
            projection_flat = self.w.transpose() @ self.y
            return np.reshape(projection_flat, self.inp_shape)

    def update_activations(self, current_input, train):
        if np.min(current_input) < 0:
            print("CAREFUL INPUT IS NEGATIVE!")

        zero_vector = np.zeros(self.num_units_y)

        y_mean = np.copy(self.track_y) / self.iteration
        y_squared_mean = np.copy(self.track_y_squared) / self.iteration

        if self.kroned_w == "yin":
            num_blocks_per_side = np.sqrt(self.num_assemblies)
            block_size = int(self.inp_shape[0] / num_blocks_per_side)
            original_image_view_as_blocks = view_as_blocks(
                current_input, (block_size, block_size)
            )
            input_vector = np.array(original_image_view_as_blocks, dtype=np.float64)
            self.current_input = np.reshape(
                input_vector,
                (self.num_assemblies * block_size * block_size * self.inp_shape[2]),
            )
        elif self.kroned_w == "delayed":
            self.current_input[-self.frame_size :] = np.copy(current_input.flatten())
        else:
            self.current_input = current_input.flatten()

        self.y_gradient_track = list()
        self.y_activity_track = list()
        self.y_sparsity_track = list()
        self.y_entropy_track = list()

        self.current_input_y = (self.w @ self.current_input) + (
            self.p @ self.previous_y
        )
        bias_correction_y = eval(self.bias_expression)
        if self.gradient == "voltage":
            for i in range(0, self.num_steps * self.multisteps):
                gmag_y = self.gradient_step(self.current_input_y, bias_correction_y)
                self.y_sparsity_track.append(
                    1 - (np.sum(self.y > 0.0) / self.num_units_y)
                )
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        elif self.gradient == "rate":
            for i in range(0, self.num_steps * self.multisteps):
                gmag_y = self.gradient_step_rate(
                    self.current_input_y, bias_correction_y
                )
                self.y_sparsity_track.append(
                    1 - (np.sum(self.y > 0.0) / self.num_units_y)
                )
                self.y_activity_track.append(np.copy(self.y))
                self.y_entropy_track.append(entropy(self.y))
                self.y_gradient_track.append(np.linalg.norm(gmag_y))
        else:
            raise Exception("No valid gradient function provided")

        self.y = eval(self.modulate_stable_y_exprss)

        return np.copy(self.y)

    def update_activations_blank(self):
        self.update_activations(np.zeros(self.inp_shape), False)

    # called at beginning of training/testing
    def reset_network(self):
        self.y = np.zeros(self.num_units_y)
        self.previous_y = np.zeros(self.num_units_y)

    # called at the beginning of each sequence
    def reset_activations(self):
        if self.zero_sequence == "zero":
            self.y.fill(0.0)
        elif self.zero_sequence == "randu":
            self.y = np.random.uniform(low=0, high=1, size=(self.num_units_y))
        elif self.zero_sequence == "randn":
            self.y = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y))
        else:
            pass
        self.previous_y = np.zeros(self.num_units_y)

    def new_epoch(self):
        self.previous_y = np.copy(self.y)
        if self.zero_epoch == "zero":
            self.y.fill(0.0)
        elif self.zero_epoch == "randu":
            self.y = np.random.uniform(low=0, high=1, size=(self.num_units_y))
        elif self.zero_epoch == "randn":
            self.y = np.random.normal(loc=0.0, scale=1.0, size=(self.num_units_y))
        else:
            pass

    def last_epoch(self, active_matrices):
        pass

    def update_counts(self):
        self.iteration += 1

        # Compute counts and means
        self.track_input += self.current_input
        self.track_y += self.y
        self.track_y_squared += self.y**2

        # compute new windows
        self.wx_raw_window = np.roll(self.wx_raw_window, 1, axis=1)
        self.wx_raw_window[:, 0] = self.w @ self.current_input
        self.py_raw_window = np.roll(self.py_raw_window, 1, axis=1)
        self.py_raw_window[:, 0] = self.p @ self.previous_y
        self.y_window = np.roll(self.y_window, 1, axis=1)
        self.y_window[:, 0] = np.copy(self.y)

    def update_parameters(self, active_matrices):

        # compute means
        input_mean = np.copy(self.track_input) / self.iteration
        y_mean = np.copy(self.track_y) / self.iteration
        y_raw_mean = np.mean(self.py_raw_window, axis=1) + np.mean(
            self.wx_raw_window, axis=1
        )
        y_squared_mean = np.copy(self.track_y_squared) / self.iteration

        y_window_mean = np.mean(self.y_window, axis=1)

        current_y = np.copy(self.y)

        # Compute BCM
        bcm_threshold = eval(self.bcm_trsh_exp)

        # Compute dynamic learning rates
        pehlevan_dynamic_lr = np.copy((self.track_y_squared * 0.01) + self.dyn_lr_init)
        self.dynamic_learning_rate = eval(self.dynamic_lr_increment_exprss)

        gradient_list = list()

        # Update W (x -> y)
        if "w" in active_matrices:
            # correlation rule (classic hebbian rule)
            if self.rule_w == "hebbian":  # delta wij = (y_i * x_j) - w_ij
                gradient_w = np.outer(self.y, self.current_input) - self.w
            # chklovskii rule (classic hebbian rule with mean normalization)
            # foldiak rule (classic hebbian rule with weighted forgetting factor)
            elif self.rule_w == "foldiak":  # delta wij = (y_i * x_j) - (w_ij * y_i)
                gradient_w = (
                    np.outer(self.y, self.current_input)
                    - self.y[:, np.newaxis] * self.w
                )
            # input reconstruction rule (classic hebbian rule with heavily weighted forgetting factor)
            elif (self.rule_w == "oja") or (
                self.rule_w == "pehlevan"
            ):  # delta wij = (y_i * x_j) - (w_ij * y_i^2)
                gradient_w = (
                    np.outer(self.y, self.current_input)
                    - (self.y**2)[:, np.newaxis] * self.w
                )

            elif (
                self.rule_w == "chklovskii"
            ):  # delta wij = (y_i - y_mean) * (x_j - x_mean) - w_ij
                gradient_w = (
                    np.outer(self.y - y_mean, self.current_input - input_mean) - self.w
                )

            # approximate stable state rule (objective derived rule to lead feedforward weigths to approximate dynamics)
            elif self.rule_w == "reconstruction":
                gradient_w = np.outer(
                    self.y, (self.current_input - (self.w.T @ self.y))
                )

            elif self.rule_w == "perceptron":  # delta wij =
                gradient_w = np.outer(
                    (self.y - (self.w @ self.current_input)), self.current_input
                )

            # bcm absolute with arbitrary threshold
            elif (
                self.rule_w == "bcm_abs"
            ):  # delta wij = (|y_i-y_i_mean| - y_i) * x_j) - w_ij
                gradient_w = (
                    np.outer(
                        (np.abs(self.y - bcm_threshold) - bcm_threshold),
                        self.current_input,
                    )
                    - self.w
                )

            # bcm linear with arbitrary threshold
            elif self.rule_w == "bcm_linear":
                gradient_w = (
                    np.outer((self.y - bcm_threshold), self.current_input) - self.w
                )

            # bcm rule square
            elif self.rule_w == "bcm_sqr":
                gradient_w = (
                    np.outer(
                        np.multiply(self.y, self.y)
                        - np.multiply(self.y, bcm_threshold),
                        self.current_input,
                    )
                    - self.w
                )
            # bcm rule threshold
            elif self.rule_w == "bcm_thrs":
                gradient_w = (
                    np.outer(np.abs(self.y - bcm_threshold), self.current_input)
                    - self.w
                )
            else:
                raise Exception("Rule W not implemented")

            gradient_list.append(gradient_w)

            if self.dynamic_w or (self.rule_w == "pehlevan"):
                self.w += (
                    1 / self.dynamic_learning_rate.reshape((self.num_units_y, 1))
                ) * gradient_w
            else:
                self.w += self.lr_w * gradient_w

            if self.noisy_w:
                self.w += (
                    norm.ppf(np.random.rand(self.num_units_y, self.inp_size))
                    * np.sqrt(self.lr_w)
                    * self.noise_std_w
                )

            if self.rectify_w:
                self.w[self.w < self.threshold_learning] = 0.0

        # Update P (prev_y -> y)
        if "p" in active_matrices:
            # correlation rule (classic hebbian rule)
            if self.rule_p == "hebbian":  # delta pij = (y_i * x_j) - p_ij
                gradient_p = np.outer(self.y, self.previous_y) - self.p
            # chklovskii rule (classic hebbian rule with mean normalization)
            elif (
                self.rule_p == "chklovskii"
            ):  # delta pij = (y_i - y_mean) * (previous_y_i - y_mean) - p_ij
                gradient_p = (
                    np.outer(self.y - y_mean, self.previous_y - y_mean) - self.p
                )
            # foldiak rule (classic hebbian rule with weighted forgetting factor)
            elif (
                self.rule_p == "foldiak"
            ):  # delta pij = (y_i * previous_y_j) - (p_ij * y_i)
                gradient_p = (
                    np.outer(self.y, self.previous_y) - self.y[:, np.newaxis] * self.p
                )
            # input reconstruction rule (classic hebbian rule with heavily weighted forgetting factor)
            elif (self.rule_p == "oja") or (
                self.rule_p == "pehlevan"
            ):  # delta pij = (y_i * previous_y_j) - (w_ij * y_i^2)
                gradient_p = (
                    np.outer(self.y, self.previous_y)
                    - (self.y**2)[:, np.newaxis] * self.p
                )
            elif self.rule_p == "reconstruction":
                gradient_p = np.outer(self.y, (self.previous_y - (self.p.T @ self.y)))
            # approximate stable state rule (objective derived rule to lead feedforward weigths to approximate dynamics)
            elif (
                self.rule_p == "perceptron"
            ):  # delta pij = (y_i * previous_y_j) - (w_ij * previous_y_j^2)
                # raw_y = (self.w @ self.current_input) + (self.p @ self.previous_y)
                gradient_p = np.outer(
                    (self.y - (self.p @ self.previous_y)), self.previous_y
                )
            # bcm rule absolute value
            elif self.rule_p == "bcm_abs":
                gradient_p = (
                    np.outer(
                        (np.abs(self.y - bcm_threshold) - bcm_threshold),
                        self.previous_y,
                    )
                    - self.p
                )
            elif self.rule_p == "bcm_linear":
                gradient_p = (
                    np.outer((self.y - bcm_threshold), self.previous_y) - self.p
                )
            # bcm rule square
            elif self.rule_p == "bcm_sqr":
                gradient_p = (
                    np.outer(
                        np.multiply(self.y, self.y)
                        - np.multiply(self.y, bcm_threshold),
                        self.previous_y,
                    )
                    - self.p
                )
            # bcm rule threshold
            elif self.rule_p == "bcm_thrs":
                gradient_p = (
                    np.outer(np.abs(self.y - bcm_threshold), self.previous_y) - self.p
                )
            # stdp rule from biology and neuroscience
            elif self.rule_p == "stdp":
                gradient_p = np.outer(self.y, self.previous_y) - np.outer(
                    self.previous_y, self.y
                )
            else:
                raise Exception("Rule P not implemented")

            gradient_list.append(gradient_p)

            if self.dynamic_p or (self.rule_p == "pehlevan"):
                self.p += (
                    1 / self.dynamic_learning_rate.reshape((self.num_units_y, 1))
                ) * gradient_w
            else:
                self.p += self.lr_p * gradient_p

            if self.noisy_p:
                self.p += (
                    norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))
                    * np.sqrt(self.lr_p)
                    * self.noise_std_p
                )

            if self.rectify_p:
                self.p[self.p < self.threshold_learning] = 0.0

            if self.zero_dig_p:
                np.fill_diagonal(self.p, 0.0)

        # Update M (z -> z)
        if "m" in active_matrices:
            if self.rule_m == "hebbian":  # delta mij = z_i * z_j - m_ij
                gradient_m = np.outer(self.y, self.y) - self.m
            elif (
                self.rule_m == "chklovskii"
            ):  # delta mij = (z_i - z_mean) * (z_j - z_mean) - m_ij
                gradient_m = np.outer(self.y - y_mean, self.y - y_mean) - self.m
            elif self.rule_m == "foldiak":  # delta mij = (z_i * z_j) - (m_ij * z_i)
                gradient_m = np.outer(self.y, self.y) - self.y[:, np.newaxis] * self.m
            elif (
                self.rule_m == "foldiak-constant"
            ):  # delta mij = z_i * z_j - constant^2
                gradient_m = np.outer(self.y, self.y) - self.foldiak_constant**2
            elif (self.rule_m == "oja") or (
                self.rule_m == "pehlevan"
            ):  # delta mij = z_i * (z_j - (m_ij * z_i)) = (z_i * z_j) - (m_ij * z_i^2))
                gradient_m = (
                    np.outer(self.y, self.y) - (self.y**2)[:, np.newaxis] * self.m
                )
            elif (
                self.rule_m == "perceptron"
            ):  # delta mij = (z_i - (m_ij * z_j)) * z_j   .... #does this makes sense???
                gradient_m = np.outer((self.y - (self.m @ self.y)), self.y)
            elif self.rule_m == "bcm_abs":
                gradient_m = (
                    np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.y)
                    - self.m
                )
            else:
                raise Exception("Rule M not implemented")

            gradient_list.append(gradient_m)

            if self.dynamic_m or (self.rule_m == "pehlevan"):
                self.m += (
                    1 / self.dynamic_learning_rate.reshape((self.num_units_y, 1))
                ) * gradient_m
            elif self.rule_m == "ff_outer":
                self.m = np.outer(self.w, self.w)
            else:
                self.m += self.lr_m * gradient_m

            if self.noisy_m:
                self.m += (
                    norm.ppf(np.random.rand(self.num_units_y, self.num_units_y))
                    * np.sqrt(self.lr_m)
                    * self.noise_std_m
                )

            if self.rectify_m:
                self.m[self.m < self.threshold_learning] = 0.0

            if self.zero_dig_m:
                np.fill_diagonal(self.m, 0.0)

            if self._m_band_mask is not None:
                self.m[~self._m_band_mask] = 0.0

        return gradient_list

    def get_name(self):
        return "Lateral Inhibitory Layer (" + str(self.num_units_y) + ")"

    def get_hyperparams_dict(self):
        hparams = dict()
        hparams["name"] = self.name
        hparams["input_shape"] = self.inp_shape
        hparams["num_units_y"] = self.num_units_y
        hparams["num_blocks"] = self.num_assemblies
        hparams["step"] = self.step_size_y
        hparams["num_steps"] = self.num_steps
        hparams["multisteps"] = self.multisteps
        hparams["activation"] = self.activation
        hparams["threshold_dynamics"] = self.threshold
        hparams["spont_activity"] = self.spont_act
        hparams["bias"] = self.bias_expression
        hparams["modulator"] = self.modulate_stable_y_exprss
        hparams["threshold_learning"] = self.threshold_learning
        hparams["bcm_exprss"] = self.bcm_trsh_exp
        hparams["dynamic_lr_init"] = self.dyn_lr_init
        hparams["dynamic_lr_exprss"] = self.dynamic_lr_increment_exprss
        hparams["foldiak_constant"] = self.foldiak_constant
        hparams["zero_epoch"] = self.zero_epoch
        hparams["w_init"] = self.init_w
        hparams["w_update"] = self.rule_w
        hparams["w_lr"] = self.lr_w
        hparams["w_noisy"] = self.noisy_w
        hparams["w_dynamic"] = self.dynamic_w
        hparams["w_rectify"] = self.rectify_w
        hparams["w_kroned"] = self.kroned_w
        hparams["p_init"] = self.init_p
        hparams["p_update"] = self.rule_p
        hparams["p_lr"] = self.lr_p
        hparams["p_noisy"] = self.noisy_p
        hparams["p_dynamic"] = self.dynamic_p
        hparams["p_rectify"] = self.rectify_p
        hparams["p_kroned"] = self.kroned_p
        hparams["p_zero_dig"] = self.zero_dig_p
        hparams["m_init"] = self.init_m
        hparams["m_update"] = self.rule_m
        hparams["m_lr"] = self.lr_m
        hparams["m_noisy"] = self.noisy_m
        hparams["m_dynamic"] = self.dynamic_m
        hparams["m_rectify"] = self.rectify_m
        hparams["m_kroned"] = self.kroned_m
        hparams["m_zero_dig"] = self.zero_dig_m
        hparams["m_bandwidth"] = self.bandwidth_m
        hparams["use_jax"] = self.use_jax
        hparams["sleep_sparsity"] = self.sleep_sparsity
        return hparams
