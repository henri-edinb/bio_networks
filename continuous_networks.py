
import numpy as np
import networkx as nx
from scipy.special import expit
from scipy.linalg import eigh
from scipy.stats import norm
from sklearn.neural_network import BernoulliRBM
from sklearn.decomposition import MiniBatchDictionaryLearning
from skimage.util import view_as_blocks, view_as_windows
import json


#async-rlil-network
class RateLateralInhibitoryLayer:
    """
    Hyperparams:
      - input_shape : Int
      - num_units_y : Int
      - step : Float
      - spont_activity : Float
      - threshold_dynamics : Float
      - activation : String
      - bias : String
      - threshold_learning : Float
      - voltage_threshold : String
      - voltage_threshold_growth_expression : String
      - calcium_threshold : String
      - calcium_growth_expression : String
      - w_init : String
      - w_seed : Int
      - w_update : String
      - w_lr : Float
      - w_rectify : Boolean
      - m_init : String
      - m_seed : Int
      - m_update : String
      - m_lr : Float
      - m_rectify : Boolean
      - m_zero_dig : Boolean
    """

    @staticmethod
    def default_hparams(input_shape):
        hparams = dict()
        hparams['input_shape'] = input_shape
        hparams['frequency'] = 10
        hparams['num_units_y'] = 500
        hparams['step'] = 0.01
        hparams['spont_activity'] = 0.0
        hparams['threshold_dynamics'] = 0.0 # should be 0 for relu
        hparams['activation'] = 'relu'
        hparams['bias'] = 'self.y'
        hparams['threshold_learning'] = 0.0
        hparams['voltage_threshold'] = '1.0'
        hparams['voltage_threshold_growth_expression'] = '0.0'
        hparams['calcium_threshold'] = '1.0'
        hparams['calcium_growth_expression'] = '0.001'
        hparams['w_init'] = 'random'
        hparams['w_seed'] = np.random.randint(0, 100000)
        hparams['w_update'] = 'hebbian'
        hparams['w_lr'] = 0.01
        hparams['w_rectify'] = True
        hparams['m_init'] = 'random'
        hparams['m_seed'] = np.random.randint(0, 100000)
        hparams['m_update'] = 'hebbian'
        hparams['m_lr'] = 0.01
        hparams['m_rectify'] = True
        hparams['m_zero_dig'] = False
        return hparams
    
    #initialize from scratch (hparams, initial_params)
    def __init__(self, hparams):
        self.name = 'async-rlil-network'

        self.inp_shape = hparams['input_shape']
        self.inp_size = self.inp_shape[0]*self.inp_shape[1]*self.inp_shape[2]
        self.num_units_y = hparams['num_units_y']

        #ACTIVITY
        self.y = np.zeros(hparams['num_units_y']) #rate

        self.latest_gradient_y = np.zeros(hparams['num_units_y'])

        self.activation = hparams['activation']

        self.frequency = hparams['frequency']

        


        self.track_y = np.ones(hparams['num_units_y'])
        self.track_y_squared = np.ones(hparams['num_units_y'])
        self.track_spikes_y = np.ones(hparams['num_units_y'])

        self.continuous_track_y = np.zeros(hparams['num_units_y'])
        



        #MATRICES
        self.init_w = hparams['w_init']
        self.init_m = hparams['m_init']


        np.random.seed(hparams['w_seed'])
        #feedforward matrix, x->y
        if self.init_w == 'random':
            self.w = np.random.rand(self.num_units_y, self.inp_size)
        elif self.init_w == 'zero':
            self.w = np.zeros((self.num_units_y, self.inp_size))
        else:
            raise Exception('No valid initial value for W provided')
        
        

        np.random.seed(hparams['m_seed'])
        #i to i matrix
        if self.init_m == 'random':
            self.m = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_m == 'random_normalized':
            self.m = np.random.rand(self.num_units_y, self.num_units_y)/(self.num_units_y/100)
        elif self.init_m == 'identity':
            self.m = np.identity(self.num_units_y)
        elif self.init_m == 'zero':
            self.m = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception('No valid initial value for M provided')
        

        #DYNAMICS HYPERPARAMS

        #for the voltage dynamics
        self.threshold = hparams['threshold_dynamics']
        self.activation = hparams['activation']
        self.bias_expression_y = hparams['bias']
        self.step_size = hparams['step']
        self.spont_act = hparams['spont_activity']

        #for the spiking dynamics
        self.calcium_threshold = hparams['calcium_threshold']
        self.calcium_growth_expression = hparams['calcium_growth_expression']

        self.voltage_thresholds_y = np.zeros(hparams['num_units_y']) #voltage threshold
        self.voltage_thresholds_y.fill(float(hparams['voltage_threshold'])) #voltage threshold
        self.vt_growth_rate = hparams['voltage_threshold_growth_expression']
        self.y_spiking_potential = np.ones(hparams['num_units_y']) #spiking potential
        self.spike_trace_y = list()
        

        #LEARNING RATES
        self.threshold_learning = hparams['threshold_learning']

        self.lr_w = hparams['w_lr']
        self.lr_m = hparams['m_lr']

        self.rule_w = hparams['w_update']
        self.rule_m = hparams['m_update']

        self.rectify_w = hparams['w_rectify']
        self.rectify_m = hparams['m_rectify']

        self.zero_dig_m = hparams['m_zero_dig']

        self.iteration = 1

        self.matrices = ['w', 'm']


    def set_learned_params(self, saved_w, saved_m, saved_track_y, saved_track_y_squared, saved_num_spikes_y):
        self.track_y = saved_track_y
        self.track_y_squared = saved_track_y_squared
        self.track_spikes_y = saved_num_spikes_y
        
        self.w = saved_w
        self.m = saved_m
    
    #dynamics
    def gradient_step(self, current_input):
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)
        zero_vector_y = np.zeros(self.num_units_y)

        self.current_input = np.reshape(current_input, self.inp_size)
        
        gradient_y = (self.w @ self.current_input) - (self.m @ self.y) + (self.spont_act*np.random.uniform(low=0,high=1,size=(self.num_units_y))) - eval(self.bias_expression_y)
        self.y += self.step_size * gradient_y
        if self.activation == 'relu':
            self.y = np.maximum(self.y, self.threshold)
        elif self.activation == 'sigmoid':
            self.y = 1 / (1 + np.exp(-self.y))
        elif self.activation == 'linear':
            pass
        elif self.activation == 'exp':
            self.y = np.exp(self.y)
        elif self.activation == 'tanh':
            self.y = np.tanh(self.y)
        else:
            raise Exception('No valid activation function provided')
        
        
        self.latest_gradient_y = np.copy(gradient_y)

        #compute spikes
        self.compute_spikes()

        self.iteration += 1
        self.continuous_track_y += self.y

        return np.sum(np.abs(gradient_y))
    

    def compute_spikes(self):

        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)

        #compute y spikes
        large_activity_y = (self.y > self.voltage_thresholds_y)
        large_calcium_y = (self.y_spiking_potential >= np.ones(self.num_units_y))
        spikes_y = np.logical_and(large_activity_y, large_calcium_y)

        self.spikes_y = spikes_y.astype(dtype=np.float64)
        self.y_spiking_potential[spikes_y] = 0.0
        self.y_spiking_potential += eval(self.calcium_growth_expression)
        self.y_spiking_potential = np.minimum(self.y_spiking_potential, 1.0)
        

    def projection(self):
        return np.reshape(self.w.transpose() @ self.y, self.inp_shape)

    # called at each beginning of sequence
    def reset_network(self):
        self.y = np.zeros(self.num_units_y)
        self.y_spiking_potential = np.ones(self.num_units_y)
        
    #plasticity
    def update_parameters(self):
        #compute means
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)

        y_continuous_mean = self.continuous_track_y / self.iteration
        
        current_y = np.copy(self.y)
        
        modulated_activity_y = np.multiply(self.y, self.spikes_y)
        modulated_mean_activity_y = np.multiply(y_mean, self.spikes_y)

        #time_constant_potential = 0.001
        self.voltage_thresholds_y += float(self.vt_growth_rate)*self.spikes_y
        #self.voltage_thresholds_z += time_constant_potential*(modulated_mean_activity_z - self.voltage_thresholds_z)

        self.track_y += modulated_activity_y
        self.track_y_squared += np.square(modulated_activity_y)

        gradients = list()

        #Update W (x -> y)
        if self.rule_w == 'hebbian':
            gradient_w = np.outer(self.y, self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs':
            bcm_threshold = y_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_continuous':
            bcm_threshold = y_continuous_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_foldiak':
            gradient_w = np.outer((np.abs(self.y - y_mean) - y_mean), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'perceptron':
            gradient_w = np.outer((self.y - (self.w @ self.current_input)), self.current_input)
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_perceptron':
            gradient_w = np.outer((np.abs(self.y - (self.w @ self.current_input)) - (self.w @ self.current_input)), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'oja':
            gradient_w = np.outer(self.spikes_y, self.current_input) - np.multiply(self.spikes_y, self.w.T).T
        else:
            gradient_w = np.zeros((self.num_units_y, self.inp_size))

        gradients.append(gradient_w)

        self.w += self.lr_w * gradient_w
        
        if self.rectify_w:
            self.w[self.w < 0.0] = 0.0
        
        
        #Update M (z -> z)
        if self.rule_m == 'hebbian':
            gradient_m = np.outer(self.y, self.y) - self.m
            gradient_m_T_modulated = np.multiply(gradient_m.T, self.spikes_y) # modulate so only incoming synapses are affected
            gradient_m = gradient_m_T_modulated.T
        elif self.rule_m == 'hebbian_inverse':
            gradient_m = np.outer(self.y, self.y) - self.m
            gradient_m = np.multiply(gradient_m, self.spikes_y) # modulate so only outgoing synapses are affected
        elif self.rule_m == 'hebbian_symmetric':
            gradient_m = np.outer(self.y, self.y) - self.m
            center_indices = np.where(self.spikes_y == 1)[0]
            modulated_matrix_cross = np.zeros((self.num_units_y, self.num_units_y))
            # Iterate through each center index and draw a cross
            for center_index in center_indices:
                # Horizontal line of the cross
                modulated_matrix_cross[center_index, :] = 1
                # Vertical line of the cross
                modulated_matrix_cross[:, center_index] = 1
            gradient_m = np.multiply(gradient_m, modulated_matrix_cross)
        elif self.rule_m == 'foldiak-constant':
            gradient_m = np.outer(self.y, self.y) - 0.01**2
            gradient_m_T_modulated = np.multiply(gradient_m.T, self.spikes_y) # modulate so only incoming synapses are affected
            gradient_m = gradient_m_T_modulated.T
        elif self.rule_m == 'oja':
            gradient_m = np.outer(self.spikes_y, self.y) - np.multiply(self.spikes_y, self.m.T).T
        else:
            gradient_m = np.zeros((self.num_units_y, self.num_units_y))
        
        gradients.append(gradient_m)

        self.m += self.lr_m * gradient_m
        
        if self.rectify_m:
            self.m[self.m < 0.0] = 0.0

        if self.zero_dig_m:
            np.fill_diagonal(self.m, 0.0)

        return gradients

    
    def get_name(self):
        return 'Rate LIL Network (' + str(self.num_units_y) + ')'


    def get_extensive_name(self):
        net_name = "Rate LIL Network (" + str(self.num_units_y) + ")  -  "
        net_name += "W: " + self.init_w + " -> " + self.rule_w + " ... "
        net_name += "M: " + self.init_m + " -> " + self.rule_m + " ... "

        return net_name

    
    def get_hyper_parameters_string(self):
        w_params = 'W init: ' + self.init_w + 'rule: ' + self.rule_w + ' lr: ' + str(self.lr_w) + ' rect: ' + str(self.rectify_w) + '  \n '
        m_params = 'M init: ' + self.init_m + 'rule: ' + self.rule_m + ' lr: ' + str(self.lr_m) + ' rect: ' + str(self.rectify_m) + ' zero_dig: ' + str(self.zero_dig_m) + '  \n '
         
        all_params = 'y_units: ' + str(self.num_units_y) + '   z_units: ' + str(self.num_units_z) + ' \n'
        all_params += ' step_y: ' + str(self.step_size) + ' \n'
        all_params += 'spontaneous__activity: ' + str(self.spont_act) + ' \n'
        all_params += 'global inhibit at each epoch: ' + str(self.zero_epoch) + ' \n'
        all_params += 'global inhibit at each sequence: ' + str(self.zero_begin_seq) + ' \n'
        all_params += 'bcm_update_y: ' + self.bcm_trsh_exp_y + ' \n '

        all_params += 'threshold: ' + str(self.threshold) + ' \n '
        all_params += 'bias_y: ' + str(self.bias_expression_y) + ' \n '
        all_params += w_params + m_params +' \n '

        return all_params


    def get_hyperparams_dict(self):
        hndict = dict()
        hndict['name'] = self.name
        hndict['input_shape'] = self.inp_shape
        hndict['frequency'] = self.frequency
        hndict['num_units_y'] = self.num_units_y
        hndict['step'] = self.step_size
        hndict['activation'] = self.activation
        hndict['threshold_dynamics'] = self.threshold
        hndict['spont_activity'] = self.spont_act
        hndict['bias'] = self.bias_expression_y
        hndict['threshold_learning'] = self.threshold_learning
        hndict['voltage_threshold'] = self.voltage_thresholds_y
        hndict['voltage_threshold_growth_expression'] = self.vt_growth_expression
        hndict['calcium_threshold'] = self.calcium_threshold
        hndict['calcium_growth_expression'] = self.calcium_growth_expression
        hndict['w_init'] = self.init_w
        hndict['w_update'] = self.rule_w
        hndict['w_lr'] = self.lr_w
        hndict['w_rectify'] = self.rectify_w
        hndict['m_init'] = self.init_m
        hndict['m_update'] = self.rule_m
        hndict['m_lr'] = self.lr_m
        hndict['m_rectify'] = self.rectify_m
        hndict['m_zero_dig'] = self.zero_dig_m
        return hndict


#probs-async-lil-network
class ProbabilisticLateralInhibitoryLayer:
    """
    Hyperparams:
      - input_shape : Int
      - num_units_y : Int
      - step : Float
      - spont_activity : Float
      - threshold_dynamics : Float
      - activation : String
      - bias : String
      - threshold_learning : Float
      - voltage_mean : String
      - voltage_std : String
      - voltage_growth_expression : String
      - calcium_mean : String
      - calcium_std : String
      - calcium_growth_expression : String
      - w_init : String
      - w_seed : Int
      - w_update : String
      - w_lr : Float
      - w_rectify : Boolean
      - m_init : String
      - m_seed : Int
      - m_update : String
      - m_lr : Float
      - m_rectify : Boolean
      - m_zero_dig : Boolean
    """

    @staticmethod
    def default_hparams(input_shape):
        hparams = dict()
        hparams['input_shape'] = input_shape
        hparams['frequency'] = 10
        hparams['num_units_y'] = 500
        hparams['step'] = 0.01
        hparams['spont_activity'] = 0.0
        hparams['threshold_dynamics'] = 0.0 # should be 0 for relu
        hparams['activation'] = 'relu'
        hparams['bias'] = 'self.y'
        hparams['threshold_learning'] = 0.0
        hparams['voltage_mean'] = '1.0'
        hparams['voltage_std'] = '0.3'
        hparams['voltage_growth_expression'] = '0.0'
        hparams['calcium_mean'] = '200'
        hparams['calcium_std'] = '50'
        hparams['calcium_growth_expression'] = '0.0'
        hparams['w_init'] = 'random'
        hparams['w_seed'] = np.random.randint(0, 100000)
        hparams['w_update'] = 'hebbian'
        hparams['w_lr'] = 0.01
        hparams['w_rectify'] = True
        hparams['m_init'] = 'random'
        hparams['m_seed'] = np.random.randint(0, 100000)
        hparams['m_update'] = 'hebbian'
        hparams['m_lr'] = 0.01
        hparams['m_rectify'] = True
        hparams['m_zero_dig'] = False
        return hparams
    
    #initialize from scratch (hparams, initial_params)
    def __init__(self, hparams):
        self.name = 'probs-async-lil-network'

        self.inp_shape = hparams['input_shape']
        self.inp_size = self.inp_shape[0]*self.inp_shape[1]*self.inp_shape[2]
        self.num_units_y = hparams['num_units_y']

        #ACTIVITY
        self.y = np.zeros(hparams['num_units_y']) #rate

        self.latest_gradient_y = np.zeros(hparams['num_units_y'])

        self.activation = hparams['activation']

        self.frequency = hparams['frequency']

        


        self.track_y = np.ones(hparams['num_units_y'])
        self.track_y_squared = np.ones(hparams['num_units_y'])
        self.track_spikes_y = np.ones(hparams['num_units_y'])

        self.continuous_track_y = np.zeros(hparams['num_units_y'])
        



        #MATRICES
        self.init_w = hparams['w_init']
        self.init_m = hparams['m_init']


        np.random.seed(hparams['w_seed'])
        #feedforward matrix, x->y
        if self.init_w == 'random':
            self.w = np.random.rand(self.num_units_y, self.inp_size)
        elif self.init_w == 'zero':
            self.w = np.zeros((self.num_units_y, self.inp_size))
        else:
            raise Exception('No valid initial value for W provided')
        
        

        np.random.seed(hparams['m_seed'])
        #i to i matrix
        if self.init_m == 'random':
            self.m = np.random.rand(self.num_units_y, self.num_units_y)
        elif self.init_m == 'random_normalized':
            self.m = np.random.rand(self.num_units_y, self.num_units_y)/(self.num_units_y/100)
        elif self.init_m == 'identity':
            self.m = np.identity(self.num_units_y)
        elif self.init_m == 'zero':
            self.m = np.zeros((self.num_units_y, self.num_units_y))
        else:
            raise Exception('No valid initial value for M provided')
        

        #DYNAMICS HYPERPARAMS

        #for the voltage dynamics
        self.threshold = hparams['threshold_dynamics']
        self.activation = hparams['activation']
        self.bias_expression_y = hparams['bias']
        self.step_size = hparams['step']
        self.spont_act = hparams['spont_activity']

        #for the spiking dynamics
        self.calcium_mean = np.zeros(hparams['num_units_y'])
        self.calcium_mean.fill(float(hparams['calcium_mean']))
        self.calcium_std = np.zeros(hparams['num_units_y'])
        self.calcium_std.fill(float(hparams['calcium_std']))
        self.calcium_growth_expression = hparams['calcium_growth_expression']

        self.voltage_mean = np.zeros(hparams['num_units_y']) #voltage threshold
        self.voltage_mean.fill(float(hparams['voltage_mean'])) #voltage threshold

        self.voltage_std = np.zeros(hparams['num_units_y'])
        self.voltage_std.fill(float(hparams['voltage_std']))
        self.vt_growth_rate = hparams['voltage_growth_expression']


        self.time_elapsed = np.zeros(hparams['num_units_y']) #spiking potential
        
        self.spike_trace_y = list()
        

        #LEARNING RATES
        self.threshold_learning = hparams['threshold_learning']

        self.lr_w = hparams['w_lr']
        self.lr_m = hparams['m_lr']

        self.rule_w = hparams['w_update']
        self.rule_m = hparams['m_update']

        self.rectify_w = hparams['w_rectify']
        self.rectify_m = hparams['m_rectify']

        self.zero_dig_m = hparams['m_zero_dig']

        self.iteration = 1

        self.matrices = ['w', 'm']


    def set_learned_params(self, saved_w, saved_m, saved_track_y, saved_track_y_squared, saved_num_spikes_y):
        self.track_y = saved_track_y
        self.track_y_squared = saved_track_y_squared
        self.track_spikes_y = saved_num_spikes_y
        
        self.w = saved_w
        self.m = saved_m
    
    #dynamics
    def gradient_step(self, current_input):
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)
        zero_vector_y = np.zeros(self.num_units_y)

        self.current_input = np.reshape(current_input, self.inp_size)
        
        gradient_y = (self.w @ self.current_input) - (self.m @ self.y) + (self.spont_act*np.random.uniform(low=0,high=1,size=(self.num_units_y))) - eval(self.bias_expression_y)
        self.y += self.step_size * gradient_y
        if self.activation == 'relu':
            self.y = np.maximum(self.y, self.threshold)
        elif self.activation == 'sigmoid':
            self.y = 1 / (1 + np.exp(-self.y))
        elif self.activation == 'linear':
            pass
        elif self.activation == 'exp':
            self.y = np.exp(self.y)
        elif self.activation == 'tanh':
            self.y = np.tanh(self.y)
        else:
            raise Exception('No valid activation function provided')
        
        
        self.latest_gradient_y = np.copy(gradient_y)

        #compute spikes
        self.compute_spikes()

        self.iteration += 1
        self.continuous_track_y += self.y

        return np.sum(np.abs(gradient_y))
    

    def compute_spikes(self):
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)
        y_continuous_mean = self.continuous_track_y / self.iteration

        # Standardize (vectorized)
        v_z = (self.y - self.voltage_mean) / self.voltage_std
        t_z = (self.time_elapsed - self.calcium_mean) / self.calcium_std

        # Joint CDF for independent normals = product of univariate CDFs
        probs = norm.cdf(v_z) * norm.cdf(t_z)   # shape (n,)

        # Bernoulli sampling via RNG (returns boolean mask, no ints needed)
        rng = np.random.default_rng()
        mask = rng.random(probs.shape) < probs   # True where we "draw 1"
        self.spikes_y = mask.astype(dtype=np.float64)
        self.track_spikes_y += self.spikes_y

        # Zero in-place where sampled==1
        self.time_elapsed[mask] = 0.0
        self.time_elapsed += 1
        

    def projection(self):
        return np.reshape(self.w.transpose() @ self.y, self.inp_shape)

    # called at each beginning of sequence
    def reset_network(self):
        self.y = np.zeros(self.num_units_y)
        self.time_elapsed = np.zeros(self.num_units_y)

    #plasticity
    def update_parameters(self):
        #compute means
        y_mean = np.divide(np.copy(self.track_y), self.track_spikes_y)
        y_squared_mean = np.divide(np.copy(self.track_y_squared), self.track_spikes_y)

        y_continuous_mean = self.continuous_track_y / self.iteration
        
        current_y = np.copy(self.y)
        
        modulated_activity_y = np.multiply(self.y, self.spikes_y)
        modulated_mean_activity_y = np.multiply(y_mean, self.spikes_y)

        self.voltage_mean += float(self.vt_growth_rate)*self.spikes_y

        #time_constant_potential = 0.001
        #self.voltage_thresholds_y += float(self.vt_growth_rate)*self.spikes_y
        #self.voltage_thresholds_z += time_constant_potential*(modulated_mean_activity_z - self.voltage_thresholds_z)

        self.track_y += modulated_activity_y
        self.track_y_squared += np.square(modulated_activity_y)

        gradients = list()

        #Update W (x -> y)
        if self.rule_w == 'hebbian':
            gradient_w = np.outer(self.y, self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs':
            bcm_threshold = y_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'pure_bcm_abs':
            bcm_threshold = y_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input)
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'pure_const_bcm_abs':
            bcm_threshold = y_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - 0.05**2
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_continuous':
            bcm_threshold = y_continuous_mean/2
            gradient_w = np.outer((np.abs(self.y - bcm_threshold) - bcm_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_foldiak':
            gradient_w = np.outer((np.abs(self.y - y_mean) - y_mean), self.current_input) - np.multiply(np.repeat(np.array([np.copy(self.y)]), self.inp_size, axis=0).T,self.w)
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'perceptron':
            percept_threshold = (self.w @ self.current_input)
            gradient_w = np.outer((self.y - percept_threshold), self.current_input)
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_perceptron_constant':
            percept_threshold = (self.w @ self.current_input)/2
            gradient_w = np.outer((np.abs(self.y - percept_threshold) - percept_threshold), self.current_input) - 0.05**2
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_perceptron':
            percept_threshold = (self.w @ self.current_input)/2
            gradient_w = np.outer((np.abs(self.y - percept_threshold) - percept_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'bcm_abs_constant':
            percept_threshold = self.voltage_mean
            gradient_w = np.outer((np.abs(self.y - percept_threshold) - percept_threshold), self.current_input) - self.w
            gradient_w_T_modulated = np.multiply(gradient_w.T, self.spikes_y)
            gradient_w = gradient_w_T_modulated.T
        elif self.rule_w == 'oja':
            gradient_w = np.outer(self.spikes_y, self.current_input) - np.multiply(self.spikes_y, self.w.T).T
        else:
            gradient_w = np.zeros((self.num_units_y, self.inp_size))

        gradients.append(gradient_w)

        self.w += self.lr_w * gradient_w
        
        if self.rectify_w:
            self.w[self.w < 0.0] = 0.0
        
        
        #Update M (z -> z)
        if self.rule_m == 'hebbian':
            gradient_m = np.outer(self.y, self.y) - self.m
            gradient_m_T_modulated = np.multiply(gradient_m.T, self.spikes_y) # modulate so only incoming synapses are affected
            gradient_m = gradient_m_T_modulated.T
        elif self.rule_m == 'hebbian_inverse':
            gradient_m = np.outer(self.y, self.y) - self.m
            gradient_m = np.multiply(gradient_m, self.spikes_y) # modulate so only outgoing synapses are affected
        elif self.rule_m == 'hebbian_symmetric':
            gradient_m = np.outer(self.y, self.y) - self.m
            center_indices = np.where(self.spikes_y == 1)[0]
            modulated_matrix_cross = np.zeros((self.num_units_y, self.num_units_y))
            # Iterate through each center index and draw a cross
            for center_index in center_indices:
                # Horizontal line of the cross
                modulated_matrix_cross[center_index, :] = 1
                # Vertical line of the cross
                modulated_matrix_cross[:, center_index] = 1
            gradient_m = np.multiply(gradient_m, modulated_matrix_cross)
        elif self.rule_m == 'foldiak-constant':
            gradient_m = np.outer(self.y, self.y) - 0.01**2
            gradient_m_T_modulated = np.multiply(gradient_m.T, self.spikes_y) # modulate so only incoming synapses are affected
            gradient_m = gradient_m_T_modulated.T
        elif self.rule_m == 'oja':
            gradient_m = np.outer(self.spikes_y, self.y) - np.multiply(self.spikes_y, self.m.T).T
        else:
            gradient_m = np.zeros((self.num_units_y, self.num_units_y))
        
        gradients.append(gradient_m)

        self.m += self.lr_m * gradient_m
        
        if self.rectify_m:
            self.m[self.m < 0.0] = 0.0

        if self.zero_dig_m:
            np.fill_diagonal(self.m, 0.0)

        return gradients

    
    def get_name(self):
        return 'Rate LIL Network (' + str(self.num_units_y) + ')'


    def get_extensive_name(self):
        net_name = "Rate LIL Network (" + str(self.num_units_y) + ")  -  "
        net_name += "W: " + self.init_w + " -> " + self.rule_w + " ... "
        net_name += "M: " + self.init_m + " -> " + self.rule_m + " ... "

        return net_name

    
    def get_hyper_parameters_string(self):
        w_params = 'W init: ' + self.init_w + 'rule: ' + self.rule_w + ' lr: ' + str(self.lr_w) + ' rect: ' + str(self.rectify_w) + '  \n '
        m_params = 'M init: ' + self.init_m + 'rule: ' + self.rule_m + ' lr: ' + str(self.lr_m) + ' rect: ' + str(self.rectify_m) + ' zero_dig: ' + str(self.zero_dig_m) + '  \n '
         
        all_params = 'y_units: ' + str(self.num_units_y) + '   z_units: ' + str(self.num_units_z) + ' \n'
        all_params += ' step_y: ' + str(self.step_size) + ' \n'
        all_params += 'spontaneous__activity: ' + str(self.spont_act) + ' \n'
        all_params += 'global inhibit at each epoch: ' + str(self.zero_epoch) + ' \n'
        all_params += 'global inhibit at each sequence: ' + str(self.zero_begin_seq) + ' \n'
        all_params += 'bcm_update_y: ' + self.bcm_trsh_exp_y + ' \n '

        all_params += 'threshold: ' + str(self.threshold) + ' \n '
        all_params += 'bias_y: ' + str(self.bias_expression_y) + ' \n '
        all_params += w_params + m_params +' \n '

        return all_params


    def get_hyperparams_dict(self):
        hndict = dict()
        hndict['name'] = self.name
        hndict['input_shape'] = self.inp_shape
        hndict['frequency'] = self.frequency
        hndict['num_units_y'] = self.num_units_y
        hndict['step'] = self.step_size
        hndict['activation'] = self.activation
        hndict['threshold_dynamics'] = self.threshold
        hndict['spont_activity'] = self.spont_act
        hndict['bias'] = self.bias_expression_y
        hndict['threshold_learning'] = self.threshold_learning
        hndict['voltage_threshold'] = self.voltage_thresholds_y
        hndict['voltage_threshold_growth_expression'] = self.vt_growth_expression
        hndict['calcium_threshold'] = self.calcium_threshold
        hndict['calcium_growth_expression'] = self.calcium_growth_expression
        hndict['w_init'] = self.init_w
        hndict['w_update'] = self.rule_w
        hndict['w_lr'] = self.lr_w
        hndict['w_rectify'] = self.rectify_w
        hndict['m_init'] = self.init_m
        hndict['m_update'] = self.rule_m
        hndict['m_lr'] = self.lr_m
        hndict['m_rectify'] = self.rectify_m
        hndict['m_zero_dig'] = self.zero_dig_m
        return hndict
