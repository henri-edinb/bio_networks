# Biologically plausible neural networks

Firing rate models of biological neural networks with recurrent inhibitory dynamics and local plasticity. Includes multiple network classes, multiple toy video generators and a general simulator.

## Networks

The networks are instantiated using a hyperparameter dictionary. The main network class, `LateralInhibitoryLayer`, relies on lateral inhibition to learn factorized representations of inputs via local plasticity rules.

### Creating a Network
To create a network, configure its parameters using the default hyperparameter template and instantiate the class:

```python
from networks import LateralInhibitoryLayer

# Get default hyperparameters for a given input shape (e.g., 12x12 image)
hparams = LateralInhibitoryLayer.default_hparams((12, 12, 1))

# Customize hyperparameters
hparams['num_units_y'] = 36       # Number of neurons in the layer
hparams['activation'] = 'relu'    # Activation function
hparams['gradient'] = 'voltage'   # Gradient calculation method
hparams['w_update'] = 'bcm_abs'   # Weight update rule
hparams['w_lr'] = 0.01            # Feedforward learning rate

# Instantiate the network
net = LateralInhibitoryLayer(hparams)
```

### Network Configuration Options

**Activation Functions** (`hparams['activation']`):
* `'relu'`: standard Rectified Linear Unit.
* `'sigmoid'`: standard logistic sigmoid.
* `'tanh'`: hyperbolic tangent.
* `'exp'`: exponential activation.
* `'linear'`: pass-through (linear) activation.

**Gradients** (`hparams['gradient']`):
* `'voltage'`: Updates based on voltage integration. 
* `'rate'`: Updates based solely on rate changes.

**Updates (`w_update`, `m_update`, `p_update`)**:
The network processes synaptic updates over three main connectivity matrices:
* **W (Feedforward)**: Connects inputs (x) to the layer (y). Updates usually utilize `'bcm_abs'` or `'hebbian'` variants.
* **M (Lateral/Recurrent)**: Internal recurrent connections between units (y to y) for lateral inhibition. Typically set to `'hebbian'`.
* **P (Temporal/Prediction)**: Connects previous states (y at t-1) to current states (y at t). Uses `'hebbian'` variants.

---

## Video Generators

Generators provide spatial and spatiotemporal sequences to test the networks. They rely on simplified structural environments to mimic moving obstacles, gratings, or random dots.

### Available Generators
* `MultipleSpeedCrossingBar` (Crossbars)
* `MovingDots`
* `DriftingGratings`
* `DotMotion`
* `DriftingSingleBar`

### Creating a Generator
Generators can be initialized with environmental parameters like screen size, speeds, and noise frequency. 

```python
from generators import MultipleSpeedCrossingBar

# Create a generator
gen = MultipleSpeedCrossingBar(
    req_screen_size=12,
    req_bar_size=1,
    req_max_bar_speed=1,
    req_noise_freq=0.05
)

# Fetching samples during a training loop
for step in range(100):
    # For i.i.d. samples:
    gen.set_random_position()
    
    # Or, to simulate continuous temporal motion:
    # gen.set_next_position() 

    # Retrieve the state
    frame = gen.get_current_frame()
```



