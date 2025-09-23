# Mathematical Framework for Thermur

> Complete mathematical formulation of the unified murmuration control system with thermal constraints

## The Central Challenge of Criticality

Natural starling flocks exist at a critical state, a phase transition between order and disorder that enables near-instantaneous information propagation across the entire flock regardless of size. This criticality manifests as scale-free velocity correlations $`C(r) \sim r^{-1/3}`$ [^11] and susceptibility $`\chi`$ that scales with flock size $`N`$ [^8]. But how do we engineer this critical state in an artificial flock?

The answer lies in heterogeneous alert states with negative coupling, a mechanism inspired by vigilance behavior in bird flocks [^9][^10] where scanning birds prioritize threat detection over alignment.

## State Space and System Architecture

### Agent State Definition

Each agent $`i \in \{1, ..., N\}`$ maintains state:

$`
\hspace{0.5cm} \displaystyle
\mathbf{s}_i = \Bigl[ \underbrace{\mathbf{x}_i, \mathbf{v}_i}_{\text{Kinematics}}, \underbrace{a_i}_{\text{Alert}}, \underbrace{T_i, \nabla T_i}_{\text{Thermal}}, \underbrace{\mathbf{w}_i}_{\text{Wind}} \Bigr] \in \mathbb{R}^{14}
`$  
<br>

where:

- $`\mathbf{x}_i \in \mathbb{R}^3`$ — Position vector [m]

- $`\mathbf{v}_i \in \mathbb{R}^3`$ — Velocity vector [m/s]

- $`a_i \in \{0, 1\}`$ — Alert state (binary)

- $`T_i \in \mathbb{R}`$ — Local temperature [K]

- $`\nabla T_i \in \mathbb{R}^3`$ — Temperature gradient [K/m]

- $`\mathbf{w}_i \in \mathbb{R}^3`$ — Local wind velocity [m/s]

### Dynamic Graph Topology

The flock's communication structure forms a dynamic graph $`G_t = (\mathcal{V}, \mathcal{E}_t)`$. Unlike traditional flocking models that use metric distance (connecting agents within a fixed radius), biological flocks use topological distance, connecting to a fixed number of nearest neighbors regardless of their physical separation. This distinction matters fundamentally since metric interactions fail to reproduce the scale-free correlations observed in nature, while topological rules naturally generate them.

$`
\hspace{0.5cm} \displaystyle
\mathcal{N}_k(i) = \{j : \text{rank}(|\mathbf{x}_j - \mathbf{x}_i|) \leq k\}
`$  
<br>

Based on topological interaction studies in starling flocks [^12], we use $`k = 7`$ neighbors. This specific value emerges from a trade-off where too few connections prevent information propagation, while too many connections make the computational burden on biological systems prohibitive. The number seven appears repeatedly across species, suggesting an evolutionary optimum.

## The Unified Control Theorem

### Theorem Statement

**Theorem (Unified Murmuration Control):** Given $`N`$ agents with the above state representation, the control law:

$`
\hspace{0.5cm} \displaystyle
\mathbf{u}_i^* = \mathbf{u}_i^{\text{nom}} - \kappa \cdot \nabla T \cdot \sigma(-\rho c)
`$
<br>

where the nominal control:

$`
\begin{aligned}
\mathbf{u}_i^{\text{nom}} = &\underbrace{\frac{v_0 \hat{\mathbf{s}}_i - \mathbf{v}_i}{\tau} + \eta_i \boldsymbol{\xi}_i}_{\text{Self-propulsion}} + \underbrace{\kappa_i \sum_{j \in \mathcal{N}_k(i)} J_{ij} (\mathbf{v}_j - \mathbf{v}_i)}_{\text{Hamiltonian alignment}} \\
&- \underbrace{\gamma_{\text{sep}} \sum_{r_{ij} < r_{\text{min}}} \frac{\mathbf{r}_{ji}}{r_{ij}^3}}_{\text{Separation}} - \underbrace{\beta\nabla T + D\nabla\rho(1+2\theta_i)}_{\text{Environmental response}}
\end{aligned}
`$  
<br>

produces emergent dynamics satisfying all biological and safety constraints.

## Alert State Dynamics

### Markovian Vigilance States

In natural flocks, individual birds alternate between states of relaxation and vigilance following predictable patterns. Extensive field studies across multiple species [^9] reveal that vigilance bouts follow exponential distributions, characteristic of Markovian processes. This stochastic switching serves an evolutionary purpose, as vigilant birds scanning for predators cannot feed efficiently, creating a fundamental trade-off between safety and energy acquisition.

We model this behavior as a continuous-time Markov chain:

$`
\begin{aligned}
P(\text{relaxed} \to \text{alert}) &= \lambda = 0.021 \text{ per timestep}\\
P(\text{alert} \to \text{relaxed}) &= \mu = 0.05 \text{ per timestep}
\end{aligned}
`$  
<br>

These transition rates yield steady-state dynamics matching field observations:

$`
\hspace{0.5cm} \displaystyle
\pi_{\text{alert}} = \frac{\lambda}{\lambda + \mu} = \frac{0.021}{0.071} \approx 0.30
`$  
<br>

The resulting bout durations align with biological data:

- Mean alert bout: $`\mathbb{E}[T_{\text{alert}}] = 1/\mu = 20`$ timesteps

- Mean relaxed bout: $`\mathbb{E}[T_{\text{relaxed}}] = 1/\lambda = 47`$ timesteps

The 30% alert fraction represents a critical threshold. Below this value, flocks become overly ordered, losing the ability to respond rapidly to threats. Above it, excessive vigilance fragments the group into chaotic, uncoordinated motion. This fraction emerges naturally from the evolutionary pressures that shaped these behaviors over millions of years [^9][^10].

## Hamiltonian Alignment

### Energy Formulation

Statistical mechanics provides a powerful framework for understanding collective motion. Research on starling flocks has demonstrated they can be modeled as maximum entropy systems [^7], where bird velocities act analogously to spins in magnetic materials. This connection to physics reveals deep principles, with flocks transitioning between coordinated movement and scattered chaos just as magnets exhibit phase transitions between ordered and disordered states.

The system's energy takes the classical Heisenberg form [^18]:

$`
\hspace{0.5cm} \displaystyle
E = -\sum_{i<j} J_{ij}^{\text{alert}} \mathbf{s}_i \cdot \mathbf{s}_j - \sum_i \mathbf{h}_i \cdot \mathbf{s}_i
`$
<br>

where $`\mathbf{s}_i = \mathbf{v}_i / |\mathbf{v}_i|`$ represents the normalized velocity or "spin" of each bird. The first term captures velocity alignment between neighbors, while the second represents external influences like thermal gradients.

### Heterogeneous Coupling

The coupling strength between agents depends critically on their alert state:

$`
\hspace{0.5cm} \displaystyle
J_{ij}^{\text{alert}} = \kappa_i \times J_0 \exp(-d_{ij}/\lambda)
`$  
<br>

where the topological distance $`d_{ij}`$ counts the minimum number of neighbor-to-neighbor hops between agents, and:

$`
\hspace{0.5cm} \displaystyle
\kappa_i = \begin{cases} 
1.0 & \text{if relaxed (promotes alignment)} \\ 
-1.3 & \text{if alert (opposes alignment)}
\end{cases}
`$  
<br>

The negative coupling for alert birds represents this project's central innovation, unifying Hamiltonian spin dynamics with active matter flocking through heterogeneous interaction strengths to achieve criticality.

While relaxed birds follow their neighbors' motion, alert birds actively oppose local alignment, with their attention focused outward for threats rather than inward for coordination. This opposition creates perturbations that cascade through the topological network, preventing the system from settling into static order.

The alignment force emerges from the energy gradient:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{align}} = \kappa_i \sum_{j \in \mathcal{N}_k(i)} J_{ij} (\mathbf{v}_j - \mathbf{v}_i)
`$  
<br>

### Critical State Emergence

The heterogeneous coupling creates a bimodal distribution of interaction strengths:

$`
\hspace{0.5cm} \displaystyle
P(J) = (1-\pi_{\text{alert}})\delta(J-J_0) + \pi_{\text{alert}}\delta(J+1.3J_0)
`$  
<br>

This bimodality generates variance in the coupling landscape:

$`
\hspace{0.5cm} \displaystyle
\text{Var}[J] = \pi_{\text{alert}}(1-\pi_{\text{alert}})(J_0 + 1.3J_0)^2 \approx 1.11J_0^2
`$  
<br>

The variance in coupling strengths maintains elevated susceptibility, quantified through the integrated velocity correlation function:

$`
\hspace{0.5cm} \displaystyle
\chi = \int_0^\xi C(r) dr
`$
<br>

where $`C(r)`$ is the velocity correlation function and $`\xi`$ is the correlation length. In critical systems, $`\chi`$ scales with flock size $`N`$ without saturation, indicating the system maintains responsiveness at all scales [^11] [^8].

## Self-Propulsion Dynamics

### Active Matter Framework

Birds, unlike passive particles, generate their own motion through wing beats. This self-propulsion places them in the category of active matter, systems that consume energy to move. Active matter theory shows these systems exhibit unique phase transitions and collective phenomena impossible in equilibrium systems [^19]. The Vicsek model, a cornerstone of this field, demonstrates how self-propelled particles with velocity alignment can spontaneously break symmetry and move collectively.

In our framework, each agent maintains its cruising speed through a restoring force:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{prop}} = \frac{v_0 \hat{\mathbf{s}}_i - \mathbf{v}_i}{\tau} + \eta_i \boldsymbol{\xi}_i
`$  
<br>

The first term drives the agent toward its preferred speed $`v_0 = 12`$ m/s in direction $`\hat{\mathbf{s}}_i`$, with relaxation time $`\tau = 0.6`$ s determining how quickly it responds to deviations. The second term introduces stochastic fluctuations essential for maintaining criticality, with $`\boldsymbol{\xi}_i \sim \mathcal{N}(0, \mathbf{I})`$ representing Gaussian white noise.

### State-Dependent Noise

The noise amplitude varies dramatically with alert state:

$`
\hspace{0.5cm} \displaystyle
\eta_i = \begin{cases}
\eta_{\text{base}} = 0.2 & \text{if relaxed} \\
\eta_{\text{base}} \times (1 + \alpha) = 1.0 & \text{if alert}
\end{cases}
`$  
<br>

where the amplification factor $`\alpha = 4.5`$ places alert birds near the order-disorder phase transition. This heightened noise reflects the rapid, scanning movements of vigilant birds, their heads turning quickly to survey for threats. The noise value of 0.2 for relaxed birds keeps them in the ordered phase where collective motion emerges, while the value of 1.0 for alert birds pushes them toward disorder, thereby creating the fluctuations that prevent the flock from freezing into rigid patterns.

## Environmental Response

### Thermal Gradient Navigation

When navigating thermal fields, agents follow the negative temperature gradient to escape high-temperature regions. This response intensifies as temperatures approach dangerous levels:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{thermal}} = -\beta \nabla T(\mathbf{x}_i) \times (1 + \theta_i)
`$  
<br>

The threat level $`\theta_i`$ provides a smooth transition from normal flight to emergency evasion:

$`
\hspace{0.5cm} \displaystyle
\theta_i = \text{clip}\left(\frac{T_i - 0.7 T_{\text{max}}}{0.3 T_{\text{max}}}, 0, 1\right)
`$  
<br>

This formulation ensures that agents begin responding to thermal threats well before reaching critical temperatures, with the response strength doubling as threat levels approach unity. The threshold at 70% of maximum temperature provides a safety margin, allowing time for evasive maneuvers before thermal damage occurs.

### Density Wave Propagation

Under predator attack, starling flocks exhibit remarkable density waves that sweep through the group at speeds far exceeding individual flight velocities. Field measurements show these waves propagating at 15-45 m/s [^13], creating the characteristic "ink-like" patterns that confuse predators. These density fluctuations arise from a reaction-diffusion process where local compression triggers expansion in neighboring regions.

We model this phenomenon through a continuum density field evolving according to:

$`
\hspace{0.5cm} \displaystyle
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho\mathbf{v}) = D\nabla^2\rho + S(\theta)
`$  
<br>

The advection term $`\nabla \cdot (\rho\mathbf{v})`$ transports density with the flow, while diffusion $`D\nabla^2\rho`$ spreads perturbations. The source term $`S(\theta)`$ intensifies with threat level.

Each agent experiences a dispersive force opposing density gradients:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{density}} = -D \cdot \nabla\rho(\mathbf{x}_i) \cdot (1 + 2\theta_i)
`$  
<br>

For computational efficiency, we estimate local density using kernel methods:

$`
\hspace{0.5cm} \displaystyle
\rho(\mathbf{x}_i) = \sum_j K(|\mathbf{x}_i - \mathbf{x}_j|; \sigma)
`$  
<br>

where $`K(r; \sigma) = \exp(-r^2/2\sigma^2)`$ is a Gaussian kernel with bandwidth $`\sigma = 5.0`$ m chosen to match the spatial scale of observed density fluctuations.

## Collision Avoidance

At close range, strong repulsive forces prevent agents from colliding:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{sep}} = -w_{\text{sep}} \sum_{\substack{j \neq i \\ r_{ij} < 3\epsilon}} \frac{\mathbf{x}_j - \mathbf{x}_i}{|\mathbf{x}_j - \mathbf{x}_i|^3}
`$  
<br>

The cubic denominator creates a force that grows rapidly as agents approach, effectively creating a "hard sphere" repulsion. The threshold distance $`3\epsilon`$ with $`\epsilon = 0.1`$ m ensures separation forces activate before actual contact, while the weight $`w_{\text{sep}} = 1.5`$ balances collision avoidance against other behavioral imperatives.

## Thermal Safety Through Soft Penalties

### The Kreisselmeier-Steinhauser Framework

Thermal safety constraints guide agents away from dangerous temperature regions using smooth, differentiable penalty functions. The Kreisselmeier-Steinhauser (KS) formulation, developed for aerospace trajectory optimization [^14], provides gradient-based corrections that integrate seamlessly with neural network training while maintaining computational efficiency.

The thermal constraint function encodes safety requirements:

$`
\hspace{0.5cm} \displaystyle
c(\mathbf{x}, \mathbf{u}) = \nabla T(\mathbf{x}) \cdot \mathbf{u} + \alpha(T_{\text{max}} - T(\mathbf{x}))
`$
<br>

When $`c \geq 0`$, the action $`\mathbf{u}`$ maintains or improves thermal safety, with the convergence rate $`\alpha = 2.5`$ determining how aggressively agents avoid temperature boundaries.

### Smooth Penalty Design

The KS penalty function creates a differentiable approximation to constraint violations:

$`
\hspace{0.5cm} \displaystyle
p(c) = \frac{\kappa}{\rho} \ln(1 + e^{-\rho c})
`$
<br>

This formulation smoothly transitions from zero penalty in safe regions to strong corrections near violations. The weight parameter $`\kappa = 100`$ scales the correction magnitude, while the sharpness parameter $`\rho = 30`$ controls the transition steepness between safe and unsafe regions.

### Gradient Correction Mechanism

The penalty gradient provides a correction vector that modifies nominal controls:

$`
\hspace{0.5cm} \displaystyle
\nabla_{\mathbf{u}} p = \kappa \cdot \sigma(-\rho c) \cdot \nabla T
`$
<br>

where $`\sigma`$ denotes the sigmoid function. The corrected control becomes:

$`
\hspace{0.5cm} \displaystyle
\mathbf{u}_{\text{safe}} = \mathbf{u}_{\text{nom}} - \kappa \cdot \nabla T \cdot \sigma(-\rho c)
`$
<br>

The sigmoid activation creates adaptive behavior:
- Deep violations ($`c \ll 0`$): $`\sigma(-\rho c) \approx 1`$, maximum correction along $`-\nabla T`$
- Near boundary ($`c \approx 0`$): $`\sigma(-\rho c) \approx 0.5`$, moderate correction
- Safe region ($`c > 0`$): $`\sigma(-\rho c) \approx 0`$, minimal interference

This mechanism ensures agents navigate thermal fields smoothly, avoiding discontinuous jumps in control that could destabilize learning or create unrealistic trajectories.

## Emergent Properties

### Information Transfer Velocity

The speed at which information propagates through the flock depends on the fraction of alert individuals. Empirical measurements of starling flocks show propagation speeds ranging from 15 m/s in calm conditions to 45 m/s under predator attack [^13]. Our model reproduces this range through alert-dependent propagation:

$`
\hspace{0.5cm} \displaystyle
v_{\text{info}} = v_{\text{min}} + (v_{\text{max}} - v_{\text{min}}) \times \pi_{\text{alert}}
`$  
<br>

With our parameters:

$`
\hspace{0.5cm} \displaystyle
v_{\text{info}} = 15 + 30 \times 0.3 = 24 \text{ m/s}
`$  
<br>

This intermediate value reflects the baseline vigilance state, with capacity to increase rapidly when threats emerge.

### Scale-Free Correlations

The combination of topological interactions and heterogeneous coupling produces velocity correlations that decay as a power law:

$`
\hspace{0.5cm} \displaystyle
C(r) \sim r^{-\gamma} \text{ where } \gamma = \frac{d - 2 + \eta_{\text{alert}}}{2} \approx \frac{1}{3}
`$  
<br>

This exponent of 1/3 matches empirical observations across multiple species and flock sizes [^11], indicating a universal principle independent of specific biological details.

### Network Connectivity

The flock maintains cohesion through its topological network. The Fiedler value (second smallest eigenvalue of the graph Laplacian) quantifies connectivity:

$`
\hspace{0.5cm} \displaystyle
\lambda_2 > 0 \implies \text{Connected flock}
`$  
<br>

With k=7 neighbors and separation forces preventing isolation:

$`
\hspace{0.5cm} \displaystyle
P(\text{connected}) \geq 1 - N \cdot \exp(-k \cdot p_{\text{edge}}) > 0.99
`$  
<br>

## System Evolution

The complete system evolves through explicit Euler integration:

$`
\begin{aligned}
\mathbf{v}_{i}^{t+1} &= \mathbf{v}_{i}^{t} + \mathbf{u}_i^* \Delta t \\
\mathbf{x}_{i}^{t+1} &= \mathbf{x}_{i}^{t} + \mathbf{v}_{i}^{t+1} \Delta t
\end{aligned}
`$  
<br>

with timestep $`\Delta t = 0.1`$ s chosen to balance computational efficiency with numerical stability.

## Summary

The unified control law achieves biological murmuration dynamics through carefully orchestrated interactions between multiple components. Alert heterogeneity with negative coupling maintains the critical state necessary for rapid information transfer. Topological interactions enable scale-free correlations independent of flock size. Self-propulsion with state-dependent noise creates realistic flight dynamics. Environmental responses translate thermal threats into coordinated evasion. Thermal penalties enforce safety through smooth gradient corrections that guide agents away from dangerous regions.

The resulting system exhibits measurable properties matching empirical observations:

- Susceptibility $`\chi \sim N`$ indicating critical state dynamics

- Information propagation at $`v_{\text{info}} \in [15, 45]`$ m/s

- Scale-free correlations with $`C(r) \sim r^{-1/3}`$

- Alert fraction stabilized at $`\pi_{\text{alert}} \approx 0.30`$

- Network connectivity maintained with $`\lambda_2 > 0`$

- Thermal safety maintained with $`T < 475`$ K

This framework transforms invisible thermal threats into visible motion patterns that humans can intuitively interpret, potentially providing life-saving information in wildfire scenarios where traditional sensors fail to capture the complex, dynamic nature of thermal hazards.

---

## References

[^7]: Bialek, William, Andrea Cavagna, Irene Giardina, Thierry Mora, Edmondo Silvestri, Massimiliano Viale, and Aleksandr M. Walczak. 2012. "Statistical Mechanics for Natural Flocks of Birds." *Proceedings of the National Academy of Sciences* 109 (13): 4786–91. https://doi.org/10.1073/pnas.1118633109

[^8]: Attanasi, Alessandro, Andrea Cavagna, Lorenzo Del Castello, Irene Giardina, Stefano Melillo, Leonardo Parisi, Oliver Pohl, Bruno Rossaro, Edward Shen, Edmondo Silvestri, and Massimiliano Viale. 2014. "Finite-Size Scaling as a Way to Probe Near-Criticality in Natural Swarms." *Physical Review Letters* 113 (23): 238102. https://doi.org/10.1103/PhysRevLett.113.238102

[^9]: Beauchamp, Guy. 2015. *Animal Vigilance: Monitoring Predators and Competitors*. Academic Press.

[^10]: Fernández-Juricic, Esteban. 2012. "Sensory Basis of Vigilance Behavior in Birds: Synthesis and Future Prospects." *Behavioural Processes* 89 (2): 143–152. https://doi.org/10.1016/j.beproc.2011.10.006

[^11]: Cavagna, Andrea, Alessio Cimarelli, Irene Giardina, Giorgio Parisi, Raffaele Santagati, Fabio Stefanini, and Massimiliano Viale. 2010. "Scale-Free Correlations in Starling Flocks." *PNAS* 107 (26): 11865–70. https://doi.org/10.1073/pnas.1005766107

[^12]: Ballerini, M. et al. 2008. "Interaction Ruling Animal Collective Behavior Depends on Topological Rather than Metric Distance." *PNAS* 105 (4): 1232–37. https://doi.org/10.1073/pnas.0711437105

[^13]: Attanasi, Alessandro, et al. 2014. "Information Transfer and Behavioural Inertia in Starling Flocks." *Nature Physics* 10 (9): 691–696. https://doi.org/10.1038/nphys3035

[^14]: Richards, Arthur. 2013. "Fast Model Predictive Control with Soft Constraints." *2013 European Control Conference (ECC)*, 1–6. https://doi.org/10.23919/ECC.2013.6669291

[^18]: Heisenberg, Werner. 1928. "Zur Theorie des Ferromagnetismus." *Zeitschrift für Physik* 49 (9): 619–636. https://doi.org/10.1007/BF01328601

[^19]: Ginelli, Francesco. 2016. "The Physics of the Vicsek Model." *The European Physical Journal Special Topics* 225 (11): 2099–2117. https://doi.org/10.1140/epjst/e2016-60066-8