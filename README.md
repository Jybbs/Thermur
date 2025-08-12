# Thermur

## Introduction

I've spent countless hours watching starling murmurations since my childhood in Western Massachusetts. During winter breaks from school, I'd often walk through the snow-covered fields near our home at dusk, when thousands of starlings would gather before roosting. I still have a vivid memory from when I was about nine, and I watched a hawk cut through the flock directly overhead. What struck me wasn't just the synchronized beauty of their flight, but how instantly they transformed when threatened. The smooth, flowing patterns shattered into chaotic motion, with sharp directional changes and an even tighter ink-like density that painted the sky black wherever they went. Most remarkably, I could instinctively track the hawk's position by watching these ripples of disorder move through the flock. 

The invisible predator became visible through the birds' collective response.

On June 30, 2013, nineteen members of the Granite Mountain Hotshots were fatally overtaken by the Yarnell Hill Fire [^1]. The crew's lookout had lost visual contact, and by the time the fire's convective plume reappeared, its lethal speed and direction left no time for escape. That incident, my own visit to the entrapment site, and the ever-increasing incidence of destructive wildfires have profoundly shaped my perspective. They underscore an ethical obligation to orient my work in machine learning directly toward preventing this kind of tragedy. What's fundamentally challenging about firefighter entrapments like these [^2] is that the most critical information on a fireground—the instantaneous, localized flow of heat and air—is dangerously invisible. Anemometers provide numbers, but the human brain is wired for something far more primitive and powerful: the perception of coherent motion [^3]. Classic studies have shown that we can recognize complex activity from just a few moving points of light, a phenomenon known as "biological motion" perception [^4].

But, what if wildfire could be recast as the "predator" in this biological system? Could we create a swarm of robots that responds to thermal threats the way starlings react to a predator? Could we model their movement, measure their cohesion, and coordinate their alignment within the context of their flock [^5]? And most critically, could this system translate an invisible threat into a visual language that firefighters could intuitively understand?

**Thermur** (a portmanteau of "thermal" and "murmuration") was born from this biomimetic insight. With modern micro-robotics now capable of surviving brief excursions up to 475K [^6], we can finally ask:

*Can a swarm of robots, governed by the mathematical rules of a starling flock responding to a predator [^7], learn to move in such a way that it not only survives the thermal chaos of a wildfire but also translates that chaos into a dynamic visual display that is immediately legible to a human under pressure?*

### Hypothesis

In shot, the motivation behind Thermur is that by developing a **thermally-constrained flocking model** where fire acts as the "predator," we can teach a robot swarm to function as a **legible, ambient information display** [^8]. When the swarm encounters dangerous thermal gradients, it will naturally transition to alert, chaotic motion patterns that make the invisible threat instinctively perceptible, like a motion-based visual language that transforms complex vector fields into intuitive guidance.

---

## Starling Murmurations as a Model for Ambient Display

The collective motion of starlings offers a compelling biological blueprint for a dynamic, decentralized information system. Unlike the rigid, grid-based movement often seen in robotics, a murmuration is fluid, adaptive, and information-rich [^7]. It is a living example of a system that performs complex, global computations through simple, local rules [^10]. Key features of starling flight that inspire Thermur include:

-   **Decentralized Control with Global Cohesion**: Murmurations lack a leader. Each bird adjusts its flight based on the positions and velocities of a fixed number of its nearest neighbors, which is typically 6-7, not all birds in sight [^11]. This "topological" interaction, rather than metric, offers us a framework for a control strategy that is inherently scalable and computationally efficient for a large robot swarm [^12]. When a predator appears, this topological network maintains cohesion while enabling rapid information propagation about the threat's location and direction. This provides a perfect model for how our swarm should respond to thermal threats.

-   **Scale-Free Correlation and Information Propagation**: A murmuration exists in a "critical" state, much like a physical system at a phase transition [^13]. This allows information, such as a change in direction in response to a threat, to propagate across the entire flock almost instantaneously, regardless of the flock's size [^13]. For Thermur, this property is paramount. By treating fire as the "predator," the swarm will exhibit these same alert-state behaviors, reflecting large-scale wind shifts and turbulent gusts without a central controller, providing a unified, non-local picture of the fire environment from purely local interactions.

-   **Information Embodiment through Motion**: The flock's shape, density, and velocity are not random, but are a direct, emergent response to external stimuli and internal communication [^7]. When a predator approaches, this response becomes more pronounced, in that the flock's motion patterns change dramatically, with increased polarization, density fluctuations, and rapid directional shifts. What fascinated me in watching these displays was how precisely these motion changes communicated threat information. Thermur aims to co-opt this principle, forcing the swarm's motion to encode the local temperature and wind field with similar response characteristics, making the fire's behavior "readable" through the flock's collective motion [^8].

-   **Implicit Boundary Adherence**: Starlings naturally avoid collisions with the ground, obstacles, and especially predators. The flock's boundaries are emergent properties of its interaction rules [^5]. I've observed how they create a dynamic, adaptive "bubble" around an approaching threat without explicit communication. This natural avoidance behavior inspires a control framework where thermal safety limits, like a 475K isotherm, are not programmed as rigid "no-go zones," but are integrated into the flocking dynamics as repulsive forces, ensuring safety is an intrinsic property of the swarm's emergent behavior [^9].

These evolved strategies for high-speed, dense, and responsive group movement provide a rich foundation for designing a robotic system that must navigate a hazardous and dynamic environment while serving as a clear information display [^12].

---

## Project Goals

The central aim of Thermur is to develop and validate a swarm control framework that enables a team of micro-robots to safely map and visualize wildfire dynamics. This will be achieved through the following goals:

1.  **Develop a Thermally-Aware Flocking Model**:

    -   Create a multi-agent control policy, inspired by starling interaction rules during predator encounters [^11], that treats fire as the "predator" and incorporates local temperature and temperature gradients as repulsive forces.

    -   Implement this policy using a **Control Barrier Function (CBF)** [^9] to mathematically guarantee that no agent's estimated surface temperature exceeds a critical threshold (e.g., 475K) [^14].

2.  **Train for Environmental Legibility**:

    -   Use a combination of imitation learning (*as a core supervised task*) and reinforcement learning [^15] to train the swarm to arrange its velocity field to mirror that of the surrounding air. Simulate increased motion chaos in areas of higher temperature, thereby mimicking a starling flock's response to a predator.

    -   Develop a perceptually-accurate color-mapping function, where each robot's onboard LED hue directly and intuitively corresponds to the ambient temperature it is sensing, with a target accuracy of **≤ 3K**, using perceptually-uniform colormaps.

3.  **Ensure Robust Decentralized Operation**:

    -   Implement the control policy on a decentralized communication topology using a **Graph Neural Network (GNN)**, built with accessible Python libraries like PyTorch Geometric [^16], to ensure scalability and resilience to single-agent failures [^17].

    -   The swarm must maintain cohesion and informational integrity even with packet loss or the failure of several agents [^18].

4.  **Validate in High-Fidelity Simulation**:

    -   Build a simulation environment that couples a robotics physics engine (MuJoCo) [^19] with realistic wildfire data from models like the Weather Research and Forecasting (WRF) Model's fire module (WRF-Fire) [^20].

    -   Rigorously test the swarm's performance against key metrics: thermal safety violations, legibility (compared to ground-truth wind fields), and mission endurance.

---

## Data Acquisition & Preparation

To train and validate the Thermur system, we will rely on a combination of synthetic and real-world data, creating a robust pipeline to prepare it for our learning framework.

### Synthetic & Empirical Datasets

1.  **Synthetic Wildfire Environments**: Our primary training data will be generated from established atmospheric and fire-spread models. This allows us to train the swarm in a vast range of repeatable and perfectly-observable scenarios.

    -   **WRF-Fire / FARSITE**: We will use output from coupled weather-wildfire models like WRF-Fire [^20] and fire-spread simulators like FARSITE. These provide 3D gridded data of wind velocity, temperature, and heat flux at resolutions down to a few meters. The underlying physics, often based on semi-empirical models like the Rothermel spread equations [^21], provides a realistic basis for the environmental dynamics the swarm will face.

    -   **Large-Eddy Simulation (LES) Plume Data**: For fine-scale turbulence, we will leverage the WRF-SFIRE dataset from Moisseeva & Stull (2020) [^22], containing 147 high-resolution simulations that capture the Kelvin-Helmholtz instabilities and turbulent eddies critical for firefighter safety.

2.  **UAV Thermal Telemetry**: To ground our thermal models in physical reality, we will use open-source datasets from drone flights in heated environments.

    -   **FireDrone Kiln Tests**: Data from aerogel-protected drones tested in industrial kilns up to 475K provides crucial information on the thermal lag between ambient temperature and a drone's skin and core temperatures [^6].

    -   **USGS Thermal UAV Dataset**: Thermal‑infrared and photogrammetric data collected by a small multirotor over Oh‑be‑joyful Creek, Colorado, provide real‑world, geo‑referenced plume imagery for validating Thermur’s plume‑rise models and sensor‑fusion pipeline [^23].

### Pre-Processing Pipeline

Raw data must be transformed into a format suitable for training a multi-agent learning system.

-   **Eulerian-to-Lagrangian Transformation**: The gridded (Eulerian) data from WRF-Fire will be converted into agent-centric (Lagrangian) observations. For each agent, we will sample the surrounding 3D field to create an input tensor representing its local environment.

-   **Thermal RC Modeling**: We will use telemetry data [^6] to fit a simple resistor-capacitor (RC) thermal model. This allows us to estimate the drone's internal core temperature based on the history of its skin-temperature readings, which is critical for predicting battery health and survivability. The model takes the form $T_{\text{core}}(t) = T_{\text{skin}} - \tau \frac{dT_{\text{skin}}}{dt}$, where $`\tau`$ is the learned thermal time constant.

-   **Domain Randomization**: To ensure the learned policies are robust and generalize from simulation to the real world, we will inject noise and perturbations into the training data. This includes adding synthetic gusts sampled from turbulence spectra characteristic of wildfires [^24] and varying the thermal properties of the simulated agents.

---

## Mathematical & Algorithmic Framework

This section details the mathematical core of the Thermur system, which combines principles from collective motion, control theory, and machine learning. The primary focus for initial development is on a supervised learning approach, where a neural network policy is trained to imitate the behavior of a physics-based "expert" controller derived from first principles.

### State Definition and Multi-Agent System

The swarm is a multi-agent system of $`N`$ agents. The state of the entire system at time $`t`$ is the set of individual agent states, $`\mathbf{S}_t = \{\mathbf{s}_t^{(1)}, \ldots, \mathbf{s}_t^{(N)}\}`$. We model the communication topology as a dynamic, undirected graph $`G_t = (\mathcal{V}, \mathcal{E}_t)`$, where the vertices $`\mathcal{V}`$ are the agents and an edge $`(i, j) \in \mathcal{E}_t`$ exists if agent $`i`$ and $`j`$ are within communication range. The neighborhood of agent $`i`$ is thus defined as $`\mathcal{N}(i,t) = \{j \mid (i,j) \in \mathcal{E}_t\}`$.

Each agent $`i`$ operates under single-integrator dynamics, where its velocity is its control input: $`\dot{\mathbf{x}}_i = \mathbf{u}_i`$. The state vector $`\mathbf{s}_t^{(i)}`$ for agent $`i`$ is a concatenation of its own state and the relative states of its neighbors:

$`
\hspace{0.5cm} \displaystyle
\mathbf{s}_t^{(i)} = \Bigl[ \underbrace{\mathbf{x}_t^{(i)}, \mathbf{v}_t^{(i)}}_\text{Kinematics}, \underbrace{T_t^{(i)}, \nabla T_t^{(i)}}_\text{Thermal Sensing}, \underbrace{E_t^{(i)}}_\text{Energy}, \underbrace{\{\mathbf{x}_t^{(j)}, \mathbf{v}_t^{(j)}\}_{j \in \mathcal{N}(i,t)}}_\text{Neighbor States} \Bigr]
`$  
<br>

where:

- $`\mathbf{x}_t^{(i)} \in \mathbb{R}^3`$ and $`\mathbf{v}_t^{(i)} \in \mathbb{R}^3`$ are position and velocity.

- $`T_t^{(i)}`$ and $`\nabla T_t^{(i)}`$ are the locally sensed temperature and its gradient.

- $`E_t^{(i)}`$ is the estimated remaining battery life.

- $`\mathcal{N}(i,t)`$ is the set of neighboring agents within communication range of agent $`i`$.

### Thermal Safety via Control Barrier Functions (CBFs)

To guarantee that an agent never enters a region where its temperature would exceed a maximum safe value, $`T_{\max}`$ (e.g., 475K), we enforce safety via a **Control Barrier Function** [^9]. We define a safety set $`\mathcal{C}`$ based on a continuously differentiable function $`h(\mathbf{s}): \mathbb{R}^n \to \mathbb{R}`$:

$`
\hspace{0.5cm} \displaystyle
\mathcal{C} = \{\mathbf{s} \in \mathbb{R}^n \mid h(\mathbf{s}) \ge 0 \}, \quad \text{where} \quad h(\mathbf{s}) = T_{\max} - T(\mathbf{s})
`$  
<br>

For the state to remain in $`\mathcal{C}`$ for all time (a condition known as forward invariance), the time derivative of $`h`$ must satisfy $`\dot{h}(\mathbf{s}) \ge -\alpha(h(\mathbf{s}))`$ for some extended class $`\mathcal{K}`$ function $`\alpha`$ [^14]. Using the chain rule, we can express $`\dot{h}`$ in terms of the control input $`\mathbf{u}`$:

$`
\hspace{0.5cm} \displaystyle
\dot{h}(\mathbf{s}) = \frac{\partial h}{\partial \mathbf{s}}\frac{d\mathbf{s}}{dt} = \nabla_{\mathbf{s}}h \cdot \dot{\mathbf{s}}
`$  
<br>

Assuming the temperature $`T`$ depends on position $`\mathbf{x}`$, the constraint becomes $`\nabla_{\mathbf{x}}h \cdot \dot{\mathbf{x}} \ge -\alpha(h(\mathbf{x}))`$, which simplifies to a linear constraint on the control $`\mathbf{u}`$. A nominal (potentially unsafe) control policy $`\mathbf{u}_{\text{nom}}`$ is filtered by solving the following **Quadratic Program (QP)** in real-time, using an efficient Python-compatible solver like OSQP [^25]:

$`
\hspace{0.5cm} \displaystyle
\begin{aligned}
\mathbf{u}^* = \quad & \underset{\mathbf{u} \in \mathbb{R}^3}{\text{argmin}}
& & \frac{1}{2} \|\mathbf{u} - \mathbf{u}_{\text{nom}}\|^2 \\
& \text{subject to}
& & \nabla_{\mathbf{x}}h \cdot \mathbf{u} \ge -\alpha(h(\mathbf{x}))
\end{aligned}
`$  
<br>

This QP finds a safe control input $`\mathbf{u}^*`$ that is minimally invasive to the desired behavior $`\mathbf{u}_{\text{nom}}`$ [^14], providing a strong safety guarantee.

### Supervised Flocking Control via Imitation Learning

The primary goal of the supervised learning phase is to train a neural network policy, $`\pi_\theta(\mathbf{s}_t^{(i)})`$, that can replicate the behavior of a hand-crafted "expert" controller. This approach is known as **Behavioral Cloning** or Imitation Learning.

#### Expert Policy Definition

The expert control action, $`\mathbf{u}_{\text{nom}}^{(i)}`$, is derived from the negative gradient of a synthetic potential energy function $`U(\mathbf{S}_t)`$ that encodes the desired flocking behavior:

$`
\hspace{0.5cm} \displaystyle
\mathbf{u}_{\text{nom}}^{(i)} = -\nabla_{\mathbf{x}_i} U(\mathbf{S}_t)
`$  
<br>

The potential function $`U`$ is a weighted sum of several components based on the classic Reynolds rules [^5] and our environmental constraints:

$`
\hspace{0.5cm} \displaystyle
U(\mathbf{S}_t) = \sum_i \left( w_c U_{\text{coh}}^{(i)} + w_r U_{\text{sep}}^{(i)} + w_t U_{\text{therm}}^{(i)} \right) + \sum_{i,j} w_a U_{\text{align}}^{(i,j)}
`$  
<br>

-   **Cohesion Potential** (Attraction): $`U_{\text{coh}}^{(i)} = \frac{1}{2} \sum_{j \in \mathcal{N}(i)} \|\mathbf{x}_i - \mathbf{x}_j\|^2`$

-   **Separation Potential** (Repulsion): $`U_{\text{sep}}^{(i)} = \sum_{j \in \mathcal{N}(i)} \frac{1}{\|\mathbf{x}_i - \mathbf{x}_j\|}`$

-   **Alignment Potential**: $`U_{\text{align}}^{(i,j)} = \frac{1}{2} \|\mathbf{v}_i - \mathbf{v}_j\|^2`$

-   **Thermal Potential**: $`U_{\text{therm}}^{(i)}`$ is a function that increases sharply as $`T_i \to T_{\max}`$, e.g., $`U_{\text{therm}}^{(i)} = \frac{1}{T_{\max} - T_i}`$.

#### Self-Propulsion Dynamics

In addition to the gradient-based forces, each agent maintains a self-propulsion velocity following active matter theory [^31]. This ensures agents maintain forward motion characteristic of bird flight, with cruising speeds typically 10-20 m/s. The self-propulsion force follows:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{prop}}^{(i)} = \frac{\mathbf{v}_0 \hat{\mathbf{s}}_i - \mathbf{v}_i}{\tau} + \eta \boldsymbol{\xi}_i
`$  
<br>

where $`\mathbf{v}_0`$ is the target cruising speed, $`\hat{\mathbf{s}}_i`$ is the heading direction, $`\tau`$ is the velocity relaxation time (typically 0.5-2.0 seconds for smooth yet responsive motion), $`\eta`$ is the noise amplitude, and $`\boldsymbol{\xi}_i \sim \mathcal{N}(0, \mathbf{I})`$ represents Gaussian noise. This formulation ensures realistic flight dynamics while maintaining the critical state necessary for rapid information propagation.

#### Supervised Learning Objective

We generate a large dataset of state-action pairs, $`\mathcal{D} = \{(\mathbf{s}_k, \mathbf{u}_{\text{nom}, k})\}_{k=1}^M`$, by running the expert policy in simulation. The neural network policy $`\pi_\theta`$ is then trained to predict the expert action given a state. The objective is to minimize the **Mean Squared Error (MSE)** between the network's output and the expert's action:

$`
\hspace{0.5cm} \displaystyle
\mathcal{L}_{\text{imitation}}(\theta) = \frac{1}{M} \sum_{k=1}^{M} \left\| \pi_\theta(\mathbf{s}_k) - \mathbf{u}_{\text{nom}, k} \right\|^2_2
`$  
<br>

This loss is minimized using standard stochastic gradient descent methods.

### Graph Neural Network Policy Architecture

The policy $`\pi_\theta`$ is implemented as a **Graph Neural Network (GNN)** [^17], as GNNs are inherently suited to handle the dynamic, graph-structured data of a swarm. They are also **permutation-equivariant**, meaning the output for a given agent does not depend on the arbitrary ordering of its neighbors, a critical property for multi-agent systems. The GNN, implemented in a framework like PyTorch Geometric [^16], operates as follows:

1.  **Encoding**: Each agent's initial feature vector $`\mathbf{h}_i^{(0)}`$ is computed from its raw state $`\mathbf{s}_t^{(i)}`$ using an input MLP.

2.  **Message Passing (L layers)**: For each layer $`l \in \{0, \ldots, L-1\}`$:

    -   An aggregation function, $`\bigoplus`$, gathers information from the neighborhood:
        $`\mathbf{a}_i^{(l)} = \bigoplus_{j \in \mathcal{N}(i)} \text{MLP}_{\text{message}}^{(l)}(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)})`$

    -   An update function, often a gated recurrent unit (GRU) [^26] for temporal stability, combines the aggregated message with the node's previous state:
        $`\mathbf{h}_i^{(l+1)} = \text{GRU}(\mathbf{h}_i^{(l)}, \mathbf{a}_i^{(l)})`$

3.  **Decoding**: After $`L`$ layers, the final hidden state $`\mathbf{h}_i^{(L)}`$ for each agent is passed through a final MLP to produce the control action:
    $`\mathbf{u}_{\text{nom}}^{(i)} = \pi_\theta(\mathbf{s}_t^{(i)}) = \text{MLP}_{\text{decode}}(\mathbf{h}_i^{(L)})`$

### Supervised Learning for Perceptual Output (Color Mapping)

A separate, simpler supervised learning task is to create the temperature-to-color mapping. The goal is to learn a function $`f_\phi: \mathbb{R} \to \mathbb{R}^3`$ that maps a sensed temperature $`T_i`$ to a color vector $`\mathbf{c}_i`$ in a perceptually uniform color space like CIELAB. This ensures that changes in color appear uniform to the human eye.

-   **Model**: A small MLP, $`f_\phi`$, with parameters $`\phi`$.

-   **Dataset**: A set of temperature-color pairs $`\{(T_k, \mathbf{c}_k)\}`$ based on a perceptually uniform colormap.

-   **Loss Function**: The model is trained to minimize the MSE in the CIELAB space:

    $`
    \hspace{0.5cm} \displaystyle
    \mathcal{L}_{\text{color}}(\phi) = \frac{1}{M} \sum_{k=1}^{M} \|f_\phi(T_k) - \mathbf{c}_k\|^2
    `$  
    <br>

### Classification of Swarm Motion Motifs

To further enhance the swarm's expressive capability, we will train a model to recognize and classify emergent collective behaviors. A "pinwheeling" motion might indicate a vortex, while a "fanning out" could signal a gust front. We will use **contrastive learning** [^27] on unlabeled trajectory data to learn an embedding space where similar motion patterns are clustered together. A small number of these clusters can then be labeled by experts to create a dictionary of motion primitives, allowing the system to not only display the wind but also classify its behavior in real-time.

---

## Simulation & Evaluation

Rigorous validation will be conducted in a high-fidelity simulation environment before any physical deployment. This phase is critical for verifying the safety guarantees of the control framework, quantifying the legibility of the swarm's display, and tuning the parameters of the learning models in a repeatable and safe manner.

### Simulation Environment

-   **Platform**: We will use **MuJoCo** [^19] for rigid-body dynamics, coupled with interpolated data from **WRF-Fire** [^20] to simulate the thermal-fluid environment. The coupling will be achieved via a custom Python wrapper that loads the netCDF output from WRF-Fire and provides a queryable interface, `env.get_state(x,y,z)`, which returns the interpolated wind and temperature vectors at any given point in the simulation space. This allows for rapid prototyping and testing of control algorithms on hundreds of agents simultaneously.

-   **Execution Loop**: The simulation proceeds in discrete time steps (~20-50ms). In each step:

    1.  **Sensing**: Each agent queries the WRF-Fire data for local temperature and wind vectors and identifies its neighbors. To simulate real-world conditions, sensor noise sampled from a Gaussian distribution will be added to these readings.

    2.  **Policy Inference**: The GNN policy $`\pi_\theta`$ [^17], implemented in PyTorch Geometric [^16], computes a nominal control action $`\mathbf{u}_{\text{nom}}`$ for each agent based on its local graph-structured state.

    3.  **Safety Filtering**: The CBF-based QP [^14] solves for a safe control action $`\mathbf{u}_{\text{safe}}`$ using a fast solver like OSQP [^25]. This step acts as a high-priority safety shield, overriding the nominal policy if necessary to prevent thermal boundary violations.

    4.  **Actuation**: The safe velocity commands are sent to the MuJoCo drone models, which simulate the low-level flight dynamics.

    5.  **Visualization**: The state of the swarm and environment is rendered for analysis. This includes visualizing the drone models, plotting the ground-truth wind field using vector glyphs, and rendering the thermal field as a volumetric slice, using Python libraries like `matplotlib` and `VTK` for advanced 3D visualization.

### Test Scenarios

To ensure the system is robust, we will validate it across a suite of increasingly complex scenarios based on known fire behavior patterns:

-   **Gust Front Passage**: A scenario where the swarm must react to a sudden, coherent change in wind speed and direction, testing the system's response time and ability to maintain cohesion.

-   **Complex Terrain Navigation**: Simulation of a fire in hilly or mountainous terrain, where wind patterns are complex and unpredictable, testing the swarm's ability to map non-uniform flows.

-   **Canopy Interaction**: A scenario where the thermal field is highly stratified, with a cooler layer near the ground and extreme temperatures in the forest canopy, testing the CBF's ability to keep the swarm within a narrow safe corridor.

-   **Scalability Stress Test**: The number of agents will be incrementally increased from 20 to 200 to evaluate the computational performance of the decentralized GNN policy and ensure cohesion does not degrade with scale.

### Evaluation Metrics

The system's performance will be judged against baselines and project goals using the following metrics:

1.  **Thermal Safety**: The rate of safety violations, $`P(T_{\text{agent}} > T_{\max})`$, which must be functionally zero for the CBF-constrained controller [^9].

2.  **Legibility (SSIM)**: The primary metric for success. At each time step, the 3D positions and velocity vectors of the agents are projected onto a 2D plane. This sparse vector field is then rendered onto a grid using a kernel density estimator to create a smooth "swarm image." A corresponding "wind image" is generated from the ground-truth WRF-Fire data. The SSIM score [^28] is computed between these two images, with a target of **≥ 0.80**.

3.  **Cohesion (Graph Connectivity)**: The algebraic connectivity ($`\lambda_2`$) of the swarm's communication graph, averaged over time [^18]. A higher value indicates a more robustly connected flock that is less likely to split into subgroups.

4.  **Energy Consumption**: The simulated total energy consumed by the swarm. This will be estimated using a simplified quadrotor power model, where power $`P \propto \|\mathbf{u}_{\text{safe}} - \mathbf{g}\|^k`$, with $`\mathbf{g}`$ being the gravity vector and $`k`$ being an empirically-derived constant. The goal is to benchmark against a 15-minute target endurance on a standard LiFePO₄ battery pack [^29].

5.  **Color Accuracy**: The Mean Absolute Error (MAE) between the temperature sensed by an agent and the temperature decoded from its displayed RGB color, with a target of **≤ 3K**.

---

## Ethical Considerations & Risks

Deploying an autonomous swarm in a safety-critical environment requires careful consideration of potential risks [^30].

-   **False Sense of Security**: Firefighters might over-rely on the swarm's display. The system must have a built-in "visual uncertainty." For example, the color display could de-saturate or the flock's motion could become more disordered when sensor readings are noisy or conflicting, providing an intuitive cue of low confidence.

-   **Aerial Clutter and Environmental Impact**: The swarm adds objects to a complex airspace. We will implement robust geofencing and a "self-destruct" or "return-to-base" protocol if the swarm loses communication or strays from its operational area. The drones must be designed with minimal acoustic noise (< 65 dB at 10m) and use LED spectra chosen to minimize disturbance to wildlife, especially during nocturnal operations.

-   **System Failure**: The failure of the swarm must not create a greater hazard. The system is intended as an auxiliary information source, not a replacement for existing training, protocols, and situational awareness.

---

## References

[^1]: U.S. Fire Administration. 2013. “Yarnell Hill Fire, Arizona.” *Wildland Fire Fatality Reports*.
[^2]: Page, Wesley G., Patrick H. Freeborn, Bret W. Butler, and W. Matt Jolly. 2019. “A Review of US Wildland Firefighter Entrapments: Trends, Important Environmental Factors, and Research Needs.” *International Journal of Wildland Fire* 28 (8): 551–69. https://doi.org/10.1071/WF19022  
[^3]: Wolfe, Jeremy M. 2020. “Visual Search: How Do We Find What We Are Looking For?” *Annual Review of Vision Science* 6: 539–62. https://doi.org/10.1146/annurev-vision-091718-015048  
[^4]: Johansson, Gunnar. 1973. “Visual Perception of Biological Motion and a Model for Its Analysis.” *Perception & Psychophysics* 14 (2): 201–11. https://doi.org/10.3758/BF03212378  
[^5]: Reynolds, Craig W. 1987. “Flocks, Herds and Schools: A Distributed Behavioral Model.” *ACM SIGGRAPH Computer Graphics* 21 (4): 25–34. https://doi.org/10.1145/37402.37406  
[^6]: Häusermann, D., et al. 2023. “FireDrone: Multi-Environment Thermally Agnostic Aerial Robot.” *Advanced Intelligent Systems* 5 (23): 2300101. https://doi.org/10.1002/aisy.202300101  
[^7]: Bialek, William, Andrea Cavagna, Irene Giardina, Thierry Mora, Edmondo Silvestri, Massimiliano Viale, and Aleksandr M. Walczak. 2012. “Statistical Mechanics for Natural Flocks of Birds.” *Proceedings of the National Academy of Sciences* 109 (13): 4786–91. https://doi.org/10.1073/pnas.1118633109  
[^8]: Ishii, Hiroshi, and Brygg Ullmer. 1997. “Tangible Bits: Towards Seamless Interfaces between People, Bits and Atoms.” *CHI ’97 Proceedings*, 234–41. https://doi.org/10.1145/258549.258715  
[^9]: Wang, Li, Magnus Egerstedt, and Aaron D. Ames. 2017. “Safety Barrier Certificates for Collision-Free Multirobot Systems.” *IEEE Transactions on Robotics* 33 (3): 661–74. https://doi.org/10.1109/TRO.2017.2659727  
[^10]: Couzin, Iain D. 2009. “Collective Cognition in Animal Groups.” *Trends in Cognitive Sciences* 13 (1): 36–43. https://doi.org/10.1016/j.tics.2008.10.002  
[^11]: Ballerini, M. et al. 2008. “Interaction Ruling Animal Collective Behavior Depends on Topological Rather than Metric Distance.” *PNAS* 105 (4): 1232–37. https://doi.org/10.1073/pnas.0711437105  
[^12]: Brambilla, Manuele, Eliseo Ferrante, Mauro Birattari, and Marco Dorigo. 2013. “Swarm Robotics: A Review from the Swarm Engineering Perspective.” *Swarm Intelligence* 7 (1): 1–41. https://doi.org/10.1007/s11721-012-0075-2  
[^13]: Cavagna, Andrea, Alessio Cimarelli, Irene Giardina, Giorgio Parisi, Raffaele Santagati, Fabio Stefanini, and Massimiliano Viale. 2010. “Scale-Free Correlations in Starling Flocks.” *PNAS* 107 (26): 11865–70. https://doi.org/10.1073/pnas.1005766107  
[^14]: Ames, Aaron D., Xiangru Xu, Jessy W. Grizzle, and Paulo Tabuada. 2016. “Control Barrier Function-Based Quadratic Programs for Safety-Critical Systems.” *IEEE Transactions on Automatic Control* 62 (8): 3861–76. https://doi.org/10.1109/TAC.2016.2638961  
[^15]: Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.
[^16]: Fey, Matthias, and Jan E. Lenssen. 2019. “Fast Graph Representation Learning with PyTorch Geometric.” *arXiv* 1903.02428. https://doi.org/10.48550/arXiv.1903.02428  
[^17]: Gama, Fernando, Ekaterina Tolstaya, and Alejandro Ribeiro. 2021. “Graph Neural Networks for Decentralized Controllers.” *ICASSP 2021 — 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*: 5260–5264. https://doi.org/10.1109/ICASSP39728.2021.9414563  
[^18]: Olfati-Saber, Reza. 2007. “Consensus and Cooperation in Networked Multi-Agent Systems.” *Proceedings of the IEEE* 95 (1): 215–33. https://doi.org/10.1109/JPROC.2006.887293  
[^19]: Todorov, Emanuel, Tom Erez, and Yuval Tassa. 2012. “MuJoCo: A Physics Engine for Model-Based Control.” IROS 2012. https://github.com/deepmind/mujoco
[^20]: Coen, Janice L., et al. 2013. “WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research and Forecasting Model.” *Journal of Applied Meteorology and Climatology* 52 (1): 16–38. https://doi.org/10.1175/JAMC-D-12-023.1  
[^21]: Rothermel, Richard C. 1972. “A Mathematical Model for Predicting Fire Spread in Wildland Fuels.” *USDA Forest Service Research Paper INT-115*.  
[^22]: Moisseeva, Nadya, and Roland Stull. 2020. "WRF-SFIRE LES Synthetic Wildfire Plume Dataset." *Federated Research Data Repository*. https://doi.org/10.20383/102.0314  
[^23]: Dawson, Cian B., Christopher Holmquist‑Johnson, and Martin A. Briggs. 2018. “Thermal Infrared and Photogrammetric Data Collected by Small Unoccupied Aircraft System for Hydrogeologic Analysis of Oh‑be‑joyful Creek, Gunnison National Forest, Colorado, August 2017.” *U.S. Geological Survey Data Release*. https://doi.org/10.5066/1P931G95D  
[^24]: Heilman, Warren E. 2023. “Atmospheric Turbulence and Wildland Fires: A Review.” *International Journal of Wildland Fire* 32 (4): 476–495. https://doi.org/10.1071/WF22053  
[^25]: Stellato, Bartolomeo, Goran Banjac, Paul Goulart, Alberto Bemporad, and Stephen Boyd. 2020. “OSQP: An Operator Splitting Solver for Quadratic Programs.” *Mathematical Programming Computation* 12 (4): 637–672. https://doi.org/10.1007/s12532-020-00179-2  
[^26]: Cho, Kyunghyun, Bart Van Merriënboer, Caglar Gulcehre, Dzmitry Bahdanau, Fethi Bougares, Holger Schwenk, and Yoshua Bengio. 2014. “Learning Phrase Representations Using RNN Encoder-Decoder for Statistical Machine Translation.” *Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP 2014)*, 1724–1734. https://doi.org/10.3115/v1/D14-1179  
[^27]: Van den Oord, Aaron, Yazhe Li, and Oriol Vinyals. 2018. “Representation Learning with Contrastive Predictive Coding.” *arXiv* 1807.03748. https://doi.org/10.48550/arXiv.1807.03748  
[^28]: Wang, Zhou, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. “Image Quality Assessment: From Error Visibility to Structural Similarity.” *IEEE Transactions on Image Processing* 13 (4): 600–612. https://doi.org/10.1109/TIP.2003.819861  
[^29]: Lopez, Neil Stephen, Christian Roice Tayag, Joshua Ezekiel Rito, Jeun Rei Barlis, and Jose Bienvenido Manuel Biona. 2023. “Thermal Analysis of an EV Lithium Iron Phosphate Battery Pack for Improved Cooling.” *2023 IEEE Transportation Electrification Conference and Expo, Asia-Pacific (ITEC Asia-Pacific)*, 1-5. https://doi.org/10.1109/ITECAsia-Pacific59272.2023.10372365  
[^30]: Clarke, Roger. 2019. "Principles and Business Processes for Responsible AI." *Computer Law & Security Review* 35 (4): 410–422. https://doi.org/10.1016/j.clsr.2019.04.007
[^31]: Ginelli, Francesco. 2016. "The Physics of the Vicsek Model." *The European Physical Journal Special Topics* 225 (11): 2099–2117. https://doi.org/10.1140/epjst/e2016-60066-8
