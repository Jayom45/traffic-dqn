# Traffic DQN - In-Depth System Analysis

## Table of Contents
1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Component Analysis](#component-analysis)
4. [Data Flow](#data-flow)
5. [Deep Q-Learning Implementation](#deep-q-learning-implementation)
6. [Training Process](#training-process)
7. [Testing Process](#testing-process)
8. [SUMO Integration](#sumo-integration)
9. [State and Action Spaces](#state-and-action-spaces)
10. [Reward System](#reward-system)

---

## Project Overview

This is a **Deep Q-Learning (DQN) Reinforcement Learning project** designed to optimize traffic light control at a 4-way intersection. The agent learns to select optimal traffic light phases to minimize vehicle waiting times and maximize traffic flow efficiency.

### Key Objectives:
- **Minimize cumulative waiting time** of vehicles at the intersection
- **Reduce queue lengths** in incoming lanes
- **Optimize traffic light phase selection** using Deep Q-Learning
- **Simulate realistic traffic patterns** using SUMO (Simulation of Urban MObility)

### Technology Stack:
- **Python 3.11** - Primary programming language
- **TensorFlow 2.16.1** - Deep learning framework
- **SUMO 1.2.0** - Traffic simulation environment
- **NumPy 1.19.5** - Numerical computations
- **Matplotlib 3.6.0** - Data visualization

---

## System Architecture

The system follows a modular architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                      MAIN ENTRY POINTS                       │
├──────────────────────┬──────────────────────────────────────┤
│  training_main.py    │       testing_main.py                │
└──────────┬───────────┴───────────────┬──────────────────────┘
           │                           │
           ▼                           ▼
┌──────────────────────┐    ┌──────────────────────┐
│ TrainingSimulation   │    │ TestingSimulation    │
└──────────┬───────────┘    └──────────┬───────────┘
           │                           │
           ├───────────────────────────┤
           │                           │
           ▼                           ▼
┌─────────────────────────────────────────────────┐
│            CORE COMPONENTS                       │
├──────────────┬──────────────┬────────────┬──────┤
│ Model        │ Memory       │ Generator  │ Utils│
│ (model.py)   │ (memory.py)  │(generator) │      │
└──────────────┴──────────────┴────────────┴──────┘
           │
           ▼
┌─────────────────────────────────────────────────┐
│         SUMO SIMULATION ENVIRONMENT              │
│         (TraCI Interface)                        │
└─────────────────────────────────────────────────┘
```

### Directory Structure:
```
traffic-dqn/
├── training_main.py          # Training entry point
├── testing_main.py           # Testing entry point
├── training_simulation.py    # Training simulation logic
├── testing_simulation.py     # Testing simulation logic
├── model.py                  # Neural network models (Train & Test)
├── memory.py                 # Experience replay memory
├── generator.py              # Traffic generation logic
├── utils.py                  # Configuration and utility functions
├── visualization.py          # Plotting and data visualization
├── gpucheck.py              # GPU availability checker
├── training_settings.ini     # Training configuration
├── testing_settings.ini      # Testing configuration
├── intersection/
│   ├── environment.net.xml   # SUMO network definition
│   ├── sumo_config.sumocfg   # SUMO configuration
│   └── episode_routes.rou.xml # Generated vehicle routes
└── models/
    ├── model_2/              # Trained model version 2
    ├── model_3/              # Trained model version 3
    └── model_5/              # Trained model version 5
```

---

## Component Analysis

### 1. **training_main.py** - Training Entry Point

**Purpose**: Orchestrates the entire training process

**Key Responsibilities**:
- Checks GPU availability and TensorFlow configuration
- Loads training configuration from `training_settings.ini`
- Initializes all core components (Model, Memory, TrafficGenerator, Simulation)
- Runs training loop for specified number of episodes
- Implements **epsilon-greedy exploration strategy**
- Saves trained model and visualization plots

**Training Loop**:
```python
while episode < config['total_episodes']:
    epsilon = 1.0 - (episode / config['total_episodes'])  # Decaying exploration
    simulation_time, training_time = Simulation.run(episode, epsilon)
    episode += 1
```

**Epsilon-Greedy Strategy**:
- Starts at ε=1.0 (100% exploration)
- Linearly decays to ε=0.0 (100% exploitation)
- Balances exploration of new actions vs exploitation of learned policy

---

### 2. **testing_main.py** - Testing Entry Point

**Purpose**: Evaluates trained model performance

**Key Responsibilities**:
- Loads testing configuration
- Loads pre-trained model from specified model path
- Runs single test episode with deterministic policy (no exploration)
- Saves test results and visualizations

**Key Difference from Training**:
- Uses `TestModel` class (loads pre-trained weights)
- No epsilon-greedy exploration (greedy action selection)
- No memory replay or model updates
- GUI can be enabled for visual inspection

---

### 3. **model.py** - Neural Network Architecture

Contains two classes: `TrainModel` and `TestModel`

#### **TrainModel Class**

**Neural Network Architecture**:
```python
Input Layer (80 neurons) → State representation
    ↓
Dense Layer (400 neurons, ReLU) → Initial feature extraction
    ↓
Dense Layer (400 neurons, ReLU) → Hidden layer 1
    ↓
Dense Layer (400 neurons, ReLU) → Hidden layer 2
    ↓
Dense Layer (400 neurons, ReLU) → Hidden layer 3
    ↓
Dense Layer (400 neurons, ReLU) → Hidden layer 4
    ↓
Output Layer (4 neurons, Linear) → Q-values for 4 actions
```

**Configuration**:
- **Input Dimensions**: 80 (state space size)
- **Output Dimensions**: 4 (action space size)
- **Hidden Layers**: 4 layers of 400 neurons each
- **Activation**: ReLU (Rectified Linear Unit) for hidden layers
- **Output Activation**: Linear (for Q-values)
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with learning rate 0.001

**Key Methods**:
- `predict_one(state)`: Predicts Q-values for a single state
- `predict_batch(states)`: Predicts Q-values for batch of states
- `train_batch(states, q_sa)`: Updates network weights using batch training
- `save_model(path)`: Saves model weights (.h5) and architecture diagram (.png)

#### **TestModel Class**

**Purpose**: Loads and uses pre-trained models for testing

**Key Methods**:
- `_load_my_model(model_folder_path)`: Loads saved .h5 model file
- `predict_one(state)`: Predicts Q-values using loaded model

---

### 4. **memory.py** - Experience Replay Buffer

**Purpose**: Stores and samples past experiences for training stability

**Memory Structure**:
```python
Sample = (state, action, reward, next_state)
```

**Key Features**:
- **Circular buffer**: When full, removes oldest samples (FIFO)
- **Random sampling**: Breaks correlation between consecutive experiences
- **Configurable size**: Min 600 samples, Max 50,000 samples

**Key Methods**:
- `add_sample(sample)`: Adds new experience to memory
- `get_samples(n)`: Randomly samples n experiences (if memory has enough)
- `_size_now()`: Returns current memory size

**Why Experience Replay?**
- Breaks temporal correlations in training data
- Enables efficient reuse of past experiences
- Stabilizes training by providing diverse samples
- Prevents catastrophic forgetting of previous states

---

### 5. **generator.py** - Traffic Generation

**Purpose**: Generates realistic vehicle routes for each episode

**Traffic Pattern**:
- **1000 cars per episode** by default
- **Time distribution**: Weibull distribution (simulates traffic waves)
- **Route selection**:
  - 75% straight routes (W→E, E→W, N→S, S→N)
  - 25% turning routes (all other combinations)

**Key Method**: `generate_routefile(seed)`

**Process**:
1. Uses Weibull distribution to determine car spawn times
2. Normalizes times to fit within episode duration (5400 steps)
3. For each car:
   - Randomly determines straight vs turn
   - Randomly selects specific route
   - Assigns random lane and initial speed (10 m/s)
4. Writes XML route file for SUMO

**Route Examples**:
```xml
<vehicle id="W_E_123" type="standard_car" route="W_E" depart="1234" 
         departLane="random" departSpeed="10" />
```

**Reproducibility**:
- Uses numpy seed for deterministic generation
- Same seed → Same traffic pattern
- Different episodes use different seeds

---

### 6. **utils.py** - Configuration and Utilities

**Key Functions**:

#### `import_train_configuration(config_file)`
Loads training parameters from `.ini` file:
- Simulation settings (GUI, episodes, steps, cars)
- Model architecture (layers, width, learning rate)
- Memory settings (min/max size)
- Agent parameters (states, actions, gamma)

#### `import_test_configuration(config_file)`
Loads testing parameters from `.ini` file

#### `set_sumo(gui, sumocfg_file_name, max_steps)`
Configures SUMO simulation:
- Determines binary: `sumo` (headless) or `sumo-gui` (visual)
- Sets command-line arguments
- Configures waiting time memory

#### `set_train_path(models_path_name)`
Creates versioned model directories:
- Checks existing model versions
- Increments version number
- Creates new directory (e.g., `models/model_4/`)

#### `set_test_path(models_path_name, model_n)`
Returns paths for model loading and test result saving

---

### 7. **visualization.py** - Data Visualization

**Purpose**: Creates plots and saves training/testing metrics

**Key Method**: `save_data_and_plot(data, filename, xlabel, ylabel)`

**Generates**:
- High-resolution PNG plots (20" × 11.25", 96 DPI)
- Text files with raw data
- Automatic y-axis scaling (±5% margins)

**Typical Plots**:
- **Reward plot**: Cumulative negative reward per episode
- **Delay plot**: Total waiting time per episode
- **Queue plot**: Average queue length per episode

---

### 8. **training_simulation.py** - Training Simulation Logic

**Most Complex Component** - Handles the core reinforcement learning loop

#### **Traffic Light Phases** (from SUMO network):
```python
PHASE_NS_GREEN = 0      # Action 0: North-South straight (green)
PHASE_NS_YELLOW = 1     # Transition yellow
PHASE_NSL_GREEN = 2     # Action 1: North-South + left turns
PHASE_NSL_YELLOW = 3    # Transition yellow
PHASE_EW_GREEN = 4      # Action 2: East-West straight
PHASE_EW_YELLOW = 5     # Transition yellow
PHASE_EWL_GREEN = 6     # Action 3: East-West + left turns
PHASE_EWL_YELLOW = 7    # Transition yellow
```

#### **Initialization** (`__init__`):
Stores references to:
- Model (neural network)
- Memory (experience replay)
- TrafficGenerator
- SUMO command
- Hyperparameters (gamma, durations, dimensions)
- Episode statistics storage

#### **Main Simulation Loop** (`run` method):

**Phase 1: Environment Setup**
```python
self._TrafficGen.generate_routefile(seed=episode)
traci.start(self._sumo_cmd)
```

**Phase 2: Episode Execution**
For each time step until max_steps:
1. **Get current state**: `_get_state()` → 80-dimensional binary vector
2. **Calculate reward**: Change in cumulative waiting time
3. **Store experience**: Add (state, action, reward, next_state) to memory
4. **Choose action**: ε-greedy policy
5. **Execute action**: 
   - Set yellow phase (if changing) for 4 seconds
   - Set green phase for 10 seconds
6. **Simulate**: Run SUMO for phase duration
7. **Update variables**: Store state, action, waiting time

**Phase 3: Training**
After episode completes:
```python
for _ in range(self._training_epochs):
    self._replay()  # Sample batch and update network
```

#### **Key Methods Explained**:

**`_get_state()`** - State Representation:
- Creates 80-dimensional binary state vector
- Represents vehicle positions in discretized cells
- 8 lane groups × 10 cells per group = 80 cells

**Lane Groups**:
```python
Lane 0: W2TL_0, W2TL_1, W2TL_2 (West straight/right)
Lane 1: W2TL_3 (West left-turn only)
Lane 2: N2TL_0, N2TL_1, N2TL_2 (North straight/right)
Lane 3: N2TL_3 (North left-turn only)
Lane 4: E2TL_0, E2TL_1, E2TL_2 (East straight/right)
Lane 5: E2TL_3 (East left-turn only)
Lane 6: S2TL_0, S2TL_1, S2TL_2 (South straight/right)
Lane 7: S2TL_3 (South left-turn only)
```

**Cell Discretization** (distance from traffic light):
```python
Cell 0: 0-7 meters
Cell 1: 7-14 meters
Cell 2: 14-21 meters
Cell 3: 21-28 meters
Cell 4: 28-40 meters
Cell 5: 40-60 meters
Cell 6: 60-100 meters
Cell 7: 100-160 meters
Cell 8: 160-400 meters
Cell 9: 400-750 meters
```

**State Encoding**:
- If cell has vehicle(s): state[cell_index] = 1
- If cell is empty: state[cell_index] = 0
- Example: Car at West lane, 50m away → state[5] = 1

**`_collect_waiting_times()`** - Reward Calculation:
- Tracks accumulated waiting time for each vehicle
- Only considers vehicles in incoming lanes
- Removes vehicles that cleared the intersection
- Returns total cumulative waiting time

**`_choose_action(state, epsilon)`** - Action Selection:
- With probability ε: Random action (exploration)
- With probability 1-ε: Best action from Q-network (exploitation)

**`_replay()`** - Experience Replay Training:
1. Sample random batch from memory (batch_size=100)
2. Predict Q-values for current states: Q(s,a)
3. Predict Q-values for next states: Q(s',a')
4. Update Q-values using Bellman equation:
   ```
   Q(s,a) = reward + γ × max(Q(s',a'))
   ```
5. Train neural network on updated Q-values

**`_simulate(steps_todo)`** - SUMO Step Execution:
- Advances simulation by specified steps
- Collects queue length statistics at each step
- Updates cumulative metrics

**`_get_queue_length()`**:
- Counts vehicles with speed=0 in all incoming lanes
- Returns total halted vehicles

---

### 9. **testing_simulation.py** - Testing Simulation Logic

**Similar to training but simplified**:
- No exploration (greedy action selection)
- No memory replay or training
- Records per-step metrics (reward, queue length)
- Used for evaluating trained model performance

**Key Differences**:
- Uses `_choose_action(state)` without epsilon
- No `_replay()` method
- Different statistics tracked (per-step vs per-episode)

---

## Data Flow

### Training Flow:
```
1. training_main.py loads configuration
        ↓
2. Initialize Model, Memory, TrafficGenerator, Simulation
        ↓
3. FOR each episode:
        ↓
4. Generate route file with TrafficGenerator
        ↓
5. Start SUMO simulation
        ↓
6. WHILE simulation running:
   a. Get current state from SUMO
   b. Calculate reward (waiting time change)
   c. Store experience in Memory
   d. Choose action (ε-greedy)
   e. Execute action (set traffic light)
   f. Simulate for phase duration
        ↓
7. Close SUMO
        ↓
8. Training phase:
   - Sample batches from Memory
   - Update Q-values using Bellman equation
   - Train neural network
        ↓
9. Save episode statistics
        ↓
10. Repeat for next episode
        ↓
11. Save final model and plots
```

### Testing Flow:
```
1. testing_main.py loads configuration
        ↓
2. Load pre-trained model
        ↓
3. Initialize TrafficGenerator, Simulation
        ↓
4. Generate route file with specified seed
        ↓
5. Start SUMO (optionally with GUI)
        ↓
6. WHILE simulation running:
   a. Get current state
   b. Calculate reward
   c. Choose best action (greedy)
   d. Execute action
   e. Simulate
   f. Record statistics
        ↓
7. Close SUMO
        ↓
8. Save test plots and results
```

---

## Deep Q-Learning Implementation

### Q-Learning Fundamentals

**Q-Function**: Q(s,a) represents expected future reward for taking action 'a' in state 's'

**Bellman Equation**:
```
Q(s,a) = reward + γ × max(Q(s',a'))
```
Where:
- `reward`: Immediate reward
- `γ` (gamma): Discount factor (0.75 in this project)
- `s'`: Next state
- `max(Q(s',a'))`: Maximum Q-value for next state

### Deep Q-Network (DQN)

**Why Deep Learning?**
- State space is too large for tabular Q-learning (2^80 possible states!)
- Neural network approximates Q-function
- Generalizes across similar states

**Key Components**:

1. **Function Approximation**:
   - Neural network maps states → Q-values
   - Input: 80-dimensional state vector
   - Output: 4 Q-values (one per action)

2. **Experience Replay**:
   - Stores past experiences in memory buffer
   - Samples random batches for training
   - Breaks temporal correlations
   - Improves sample efficiency

3. **Epsilon-Greedy Policy**:
   - Balances exploration vs exploitation
   - ε decays from 1.0 to 0.0 over training
   - Early episodes: More exploration
   - Later episodes: More exploitation

4. **Target Q-Value Computation**:
   ```python
   current_q = model.predict(state)
   target_q = current_q.copy()
   target_q[action] = reward + gamma * max(model.predict(next_state))
   model.train(state, target_q)
   ```

### Training Algorithm (Pseudo-code):
```
Initialize neural network Q with random weights
Initialize memory replay buffer M
FOR episode = 1 to 100:
    Generate traffic for episode
    Initialize state s
    epsilon = 1.0 - (episode / 100)  # Decay exploration
    
    FOR step = 1 to 5400:
        # Action selection
        IF random() < epsilon:
            action = random_action()
        ELSE:
            action = argmax(Q(s))
        
        # Environment interaction
        Execute action (set traffic light phase)
        Observe reward r and next state s'
        Store (s, a, r, s') in memory M
        
        # State transition
        s = s'
    
    # Training phase
    FOR training_epoch = 1 to 1:
        Sample random batch from M
        FOR each sample (s, a, r, s'):
            target = r + gamma * max(Q(s'))
            Update Q-network to minimize (Q(s,a) - target)^2
```

---

## Training Process

### Configuration (training_settings.ini):

```ini
[simulation]
gui = False                # Run without GUI (faster)
total_episodes = 100       # Number of training episodes
max_steps = 5400          # Steps per episode (~1.5 hours simulation)
n_cars_generated = 1000   # Cars per episode
green_duration = 10       # Green light duration (seconds)
yellow_duration = 4       # Yellow light transition (seconds)

[model]
num_layers = 4            # Hidden layers in neural network
width_layers = 400        # Neurons per hidden layer
batch_size = 100          # Training batch size
learning_rate = 0.001     # Adam optimizer learning rate
training_epochs = 1       # Training iterations per episode

[memory]
memory_size_min = 600     # Minimum samples before training
memory_size_max = 50000   # Maximum replay buffer size

[agent]
num_states = 80           # State space dimensionality
num_actions = 4           # Action space size
gamma = 0.75             # Discount factor

[dir]
models_path_name = models
sumocfg_file_name = sumo_config.sumocfg
```

### Training Workflow:

1. **Episode Start** (Episode 1-100):
   - Generate unique traffic pattern
   - Reset environment
   - Set epsilon based on episode number

2. **Action Loop** (~540 actions per episode):
   - Each action lasts 14 seconds (10s green + 4s yellow)
   - 5400 simulation steps / 14 seconds ≈ 385 actions
   - Agent observes state, selects action, receives reward

3. **Experience Storage**:
   - Each action generates one experience tuple
   - ~385 experiences per episode
   - ~38,500 total experiences over 100 episodes

4. **Training Phase** (After each episode):
   - Samples 100 random experiences from memory
   - Computes target Q-values
   - Updates neural network weights
   - One training epoch per episode

5. **Progress Tracking**:
   - Episode statistics logged
   - Running metrics displayed
   - GPU utilization monitored

6. **Model Persistence**:
   - Final model saved as `trained_model.h5`
   - Architecture diagram saved
   - Training plots generated (reward, delay, queue)

### Expected Training Time:
- **Per Episode**: ~60-120 seconds (simulation + training)
- **Total Training**: ~2-3 hours for 100 episodes
- GPU acceleration significantly reduces training time

---

## Testing Process

### Configuration (testing_settings.ini):

```ini
[simulation]
gui = True                 # Enable GUI for visualization
max_steps = 5400          # Same as training
n_cars_generated = 1000   
episode_seed = 10000      # Fixed seed for reproducibility
yellow_duration = 4
green_duration = 10

[agent]
num_states = 80
num_actions = 4

[dir]
models_path_name = models
sumocfg_file_name = sumo_config.sumocfg
model_to_test = 2         # Test model_2
```

### Testing Workflow:

1. **Model Loading**:
   - Loads specified trained model (e.g., `models/model_2/trained_model.h5`)
   - No training mode - frozen weights

2. **Deterministic Evaluation**:
   - No exploration (epsilon = 0)
   - Always selects best action
   - Fixed traffic seed for reproducibility

3. **Performance Metrics**:
   - Per-step rewards
   - Queue length evolution
   - Total simulation time
   - Visual inspection (if GUI enabled)

4. **Result Visualization**:
   - Plots saved in `models/model_X/test/`
   - Reward curve over simulation
   - Queue length dynamics

### Evaluation Criteria:
- **Lower cumulative waiting time** = Better performance
- **Shorter queue lengths** = Better flow
- **Smooth reward curve** = Stable policy
- **Comparison**: Compare against fixed-time control or other models

---

## SUMO Integration

### SUMO (Simulation of Urban MObility)

**What is SUMO?**
- Open-source traffic simulation suite
- Microscopic simulation (individual vehicles)
- Used for traffic engineering research
- Provides TraCI (Traffic Control Interface) for external control

### Network Definition (environment.net.xml):

**Intersection Layout**:
- **4-way intersection** with traffic light "TL"
- **4 arms**: North, South, East, West
- **4 lanes per arm**: 3 for straight/right, 1 for left-turn only
- **750 meters per road**: Allows realistic vehicle spacing
- **Speed limit**: 25 m/s (90 km/h)

**Traffic Light Phases** (8 total):
```xml
Phase 0 (NS_GREEN): North-South straight/right (green)
Phase 1 (NS_YELLOW): Transition (yellow)
Phase 2 (NSL_GREEN): North-South + left turns (green)
Phase 3 (NSL_YELLOW): Transition (yellow)
Phase 4 (EW_GREEN): East-West straight/right (green)
Phase 5 (EW_YELLOW): Transition (yellow)
Phase 6 (EWL_GREEN): East-West + left turns (green)
Phase 7 (EWL_YELLOW): Transition (yellow)
```

### TraCI Integration:

**Key TraCI Functions Used**:

1. **Vehicle Information**:
   - `traci.vehicle.getIDList()`: All vehicle IDs
   - `traci.vehicle.getLanePosition(id)`: Position on lane
   - `traci.vehicle.getLaneID(id)`: Current lane
   - `traci.vehicle.getRoadID(id)`: Current road
   - `traci.vehicle.getAccumulatedWaitingTime(id)`: Waiting time

2. **Traffic Light Control**:
   - `traci.trafficlight.setPhase("TL", phase)`: Set light phase

3. **Edge Information**:
   - `traci.edge.getLastStepHaltingNumber(edge)`: Stopped vehicles

4. **Simulation Control**:
   - `traci.start(cmd)`: Start simulation
   - `traci.simulationStep()`: Advance one time step
   - `traci.close()`: End simulation

### Configuration (sumo_config.sumocfg):
```xml
<input>
    <net-file value="environment.net.xml"/>  # Network topology
    <route-files value="episode_routes.rou.xml"/>  # Vehicle routes
</input>
<time>
    <begin value="0"/>  # Start at time 0
</time>
<processing>
    <time-to-teleport value="-1"/>  # Disable teleportation (no cheating!)
</processing>
```

---

## State and Action Spaces

### State Space (80 dimensions)

**Representation**: Binary occupancy grid
- **Dimension**: 80-element binary vector
- **Structure**: 8 lane groups × 10 cells = 80 cells
- **Encoding**: 1 if cell contains vehicle(s), 0 if empty

**Spatial Discretization**:
- Cells closer to intersection are smaller (more detail)
- Cells farther from intersection are larger
- Captures approaching traffic patterns

**Example State**:
```python
state = [
    1,0,0,0,1,1,0,0,0,0,  # West straight lanes
    0,0,0,0,0,1,0,0,0,0,  # West left lane
    1,1,0,0,0,0,0,0,0,0,  # North straight lanes
    # ... 50 more values
]
```

**State Space Size**: 2^80 ≈ 1.2 × 10^24 possible states (why we need deep learning!)

### Action Space (4 discrete actions)

**Actions**:
- **Action 0**: North-South straight/right green
- **Action 1**: North-South with left-turn green
- **Action 2**: East-West straight/right green
- **Action 3**: East-West with left-turn green

**Action Execution**:
1. If action differs from previous:
   - Activate yellow phase for 4 seconds
2. Activate selected green phase for 10 seconds
3. Total action duration: 14 seconds (with transition)

**Action Selection Strategy**:
- **Training**: ε-greedy (exploration vs exploitation)
- **Testing**: Greedy (always best action)

---

## Reward System

### Reward Function

**Definition**: 
```python
reward = old_total_wait - current_total_wait
```

**Interpretation**:
- **Positive reward**: Waiting time decreased (good!)
- **Negative reward**: Waiting time increased (bad!)
- **Zero reward**: No change in waiting time

**Cumulative Waiting Time**:
- Sum of all vehicles' accumulated waiting times
- Only counts vehicles in incoming lanes
- Tracked per vehicle, updated each step

### Why This Reward?

**Advantages**:
1. **Directly optimizes primary goal**: Minimize waiting time
2. **Sparse but informative**: Changes reflect action impact
3. **Bounded**: Can't grow unbounded (vehicles eventually leave)
4. **Continuous feedback**: Reward signal at each action

**Potential Issues**:
- **Negative bias**: Most rewards are negative (waiting accumulates)
- **Delayed credit assignment**: Action effects may take time

### Metric Tracking:

**During Training**:
- `_sum_neg_reward`: Cumulative negative reward per episode
- `_sum_waiting_time`: Total seconds waited by all vehicles
- `_avg_queue_length`: Average queued cars per step

**These metrics help evaluate**:
- Learning progress over episodes
- Policy effectiveness
- Traffic flow efficiency

---

## Advanced Topics

### Hyperparameter Choices

**Gamma (γ) = 0.75**:
- Relatively low discount factor
- Prioritizes immediate rewards
- Appropriate for traffic domain (quick response needed)

**Learning Rate = 0.001**:
- Standard Adam optimizer rate
- Balances convergence speed vs stability

**Batch Size = 100**:
- Sufficient for stable gradient estimates
- Memory efficient

**Memory Size = 50,000**:
- Covers ~130 episodes of experiences
- Ensures diverse sampling

**Network Architecture** (4 × 400):
- Deep enough for complex patterns
- Wide enough for rich representations
- Prevents underfitting

### Training Stability Techniques

1. **Experience Replay**: Breaks temporal correlations
2. **Epsilon Decay**: Smooth exploration-to-exploitation transition
3. **Fixed Episode Length**: Consistent training signal
4. **Reproducible Traffic**: Same seed → same scenario

### Potential Improvements

**Model Architecture**:
- **Dueling DQN**: Separate value and advantage streams
- **Double DQN**: Reduce overestimation bias
- **Target Network**: Stabilize Q-value targets
- **Prioritized Replay**: Focus on important experiences

**State Representation**:
- Include **vehicle speeds** (not just positions)
- Include **current traffic light phase**
- Include **time since last phase change**
- Add **temporal information** (time of day)

**Reward Shaping**:
- Penalize **long queues** directly
- Reward **throughput** (vehicles passed)
- Multi-objective reward (wait time + throughput + fairness)

**Action Space**:
- **Variable phase durations** (instead of fixed 10s)
- **Phase skipping** (stay in current phase)

**Training**:
- **Curriculum learning**: Start with simpler traffic
- **Multi-episode training**: Train on batches after multiple episodes
- **Transfer learning**: Pre-train on similar scenarios

---

## File-by-File Summary

### Python Files

1. **training_main.py** (94 lines):
   - Entry point for training
   - GPU detection
   - Component initialization
   - Training loop execution
   - Model saving and visualization

2. **testing_main.py** (56 lines):
   - Entry point for testing
   - Model loading
   - Single episode evaluation
   - Result visualization

3. **model.py** (117 lines):
   - `TrainModel`: DQN architecture and training
   - `TestModel`: Model loading for inference
   - Keras/TensorFlow integration

4. **training_simulation.py** (306 lines):
   - Core training simulation logic
   - SUMO interaction via TraCI
   - State extraction (80-dim)
   - Reward calculation
   - Experience replay training
   - Epsilon-greedy action selection

5. **testing_simulation.py** (242 lines):
   - Testing simulation logic
   - Similar to training but without training phase
   - Greedy action selection
   - Performance metric collection

6. **memory.py** (27 lines):
   - Simple replay buffer
   - FIFO circular buffer
   - Random sampling

7. **generator.py** (80 lines):
   - Traffic generation using Weibull distribution
   - Route file XML generation
   - 75% straight, 25% turns

8. **utils.py** (109 lines):
   - Configuration parsing
   - SUMO setup
   - Path management
   - Model versioning

9. **visualization.py** (32 lines):
   - Matplotlib plotting
   - Data persistence
   - High-quality figure generation

10. **gpucheck.py** (4 lines):
    - Simple GPU availability checker

### Configuration Files

1. **training_settings.ini**: Training hyperparameters
2. **testing_settings.ini**: Testing configuration

### SUMO Files

1. **environment.net.xml**: Intersection network topology
2. **sumo_config.sumocfg**: SUMO simulation configuration
3. **episode_routes.rou.xml**: Generated vehicle routes (regenerated each episode)

---

## Conclusion

This is a **well-structured, modular implementation** of Deep Q-Learning for traffic signal control. The code demonstrates:

### Strengths:
✅ Clear separation of concerns (Model, Memory, Simulation, etc.)
✅ Proper reinforcement learning practices (experience replay, epsilon-greedy)
✅ Integration with professional traffic simulator (SUMO)
✅ Reproducible experiments (seeded random generation)
✅ Comprehensive logging and visualization
✅ Configuration-driven design (easy to tune)

### Areas for Enhancement:
🔄 Could use more advanced DQN variants (Double DQN, Dueling DQN)
🔄 State representation could be richer (speeds, phases, etc.)
🔄 Reward function could be multi-objective
🔄 Could benefit from hyperparameter tuning
🔄 Could add comparison baselines (fixed-time, actuated control)

### Educational Value:
This project serves as an **excellent learning resource** for:
- Reinforcement learning concepts
- Deep Q-Learning implementation
- SUMO/TraCI integration
- Traffic optimization problems
- Neural network design for RL

### Practical Application:
The system demonstrates the **potential of RL for traffic management**:
- Adaptive to traffic patterns
- No pre-programmed rules needed
- Can handle complex intersections
- Optimizes for actual performance metrics
- Scalable to larger networks (with modifications)

---

**Total Lines of Code**: ~867 Python LOC (excluding comments/blanks)
**Complexity**: Medium-High (requires RL and traffic simulation knowledge)
**Maintainability**: Good (modular, documented, configurable)
**Performance**: GPU-accelerated training, efficient simulation

This project successfully demonstrates **autonomous traffic signal control using modern deep reinforcement learning techniques**.
