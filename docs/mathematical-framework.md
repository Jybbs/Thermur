# Mathematical Framework for Thermur

> Complete mathematical formulation of the unified murmuration control system with thermal constraints

## The Central Challenge of Criticality

Natural starling flocks exist at a critical state, a phase transition between order and disorder that enables near-instantaneous information propagation across the entire flock regardless of size. This criticality manifests as scale-free velocity correlations $`C(r) \sim r^{-1/3}`$ [^10] and susceptibility $`\chi`$ that scales with flock size $`N`$ [^9]. But how do we engineer this critical state in an artificial flock?

The answer lies in heterogeneous noise amplitudes, a mechanism inspired by behavioral diversity in bird flocks where individual agents exhibit a continuous spectrum of alignment behaviors. Rather than discrete behavioral states, each agent possesses a temporally varying noise amplitude $`\eta_i(t)`$ drawn from a normal distribution, creating the variance necessary for criticality [^8].

## State Space and System Architecture

### Agent State Definition

Each agent $`i \in \{1, ..., N\}`$ maintains state:

$`
\hspace{0.5cm} \displaystyle
\mathbf{s}_i = \Bigl[ \underbrace{\mathbf{x}_i, \mathbf{v}_i}_{\text{Kinematics}}, \underbrace{\eta_i}_{\text{Noise}}, \underbrace{T_i, \nabla T_i}_{\text{Thermal}}, \underbrace{\mathbf{w}_i}_{\text{Wind}} \Bigr] \in \mathbb{R}^{14}
`$
<br>

where:

- $`\mathbf{x}_i \in \mathbb{R}^3`$ — Position vector [m]

- $`\mathbf{v}_i \in \mathbb{R}^3`$ — Velocity vector [m/s]

- $`\eta_i \in \mathbb{R}^+`$ — Individual noise amplitude

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

Based on topological interaction studies in starling flocks [^11], we use $`k = 7`$ neighbors. This specific value emerges from a trade-off where too few connections prevent information propagation, while too many connections make the computational burden on biological systems prohibitive. The number seven appears repeatedly across species, suggesting an evolutionary optimum.

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
\mathbf{u}_i^{\text{nom}} = &\underbrace{-\frac{4\lambda}{v_0^6}(|\mathbf{v}_i|^2 - v_0^2)^3\mathbf{v}_i + \alpha_w \mathbf{w}_i + \eta_i(t) \boldsymbol{\xi}_i}_{\text{Marginal speed confinement}} + \underbrace{\sum_{j \in \mathcal{N}_k(i)} J_{ij} (\mathbf{v}_j - \mathbf{v}_i)}_{\text{Maximum entropy alignment}} \\
&- \underbrace{\gamma_{\text{sep}} \sum_{r_{ij} < r_{\text{min}}} \frac{\mathbf{r}_{ji}}{r_{ij}^3}}_{\text{Separation}} - \underbrace{\beta\nabla T + D\nabla\rho(1+2\theta_i)}_{\text{Environmental response}}
\end{aligned}
`$  
<br>

produces emergent dynamics satisfying all biological and safety constraints.

## Maximum Entropy Alignment

### Energy Formulation

Statistical inference provides a powerful framework for understanding collective motion. Research has demonstrated starling flocks can be modeled as maximum entropy systems [^7], where bird velocities act analogously to spins in magnetic materials. By inferring the simplest probability distribution consistent with observed correlations, this approach reveals an effective energy function without assuming underlying mechanics.

The inferred energy takes the Heisenberg form [^17]:

$`
\hspace{0.5cm} \displaystyle
E = -\sum_{i<j} J_{ij} \mathbf{s}_i \cdot \mathbf{s}_j - \sum_i \mathbf{h}_i \cdot \mathbf{s}_i
`$
<br>

where $`\mathbf{s}_i = \mathbf{v}_i / |\mathbf{v}_i|`$ represents the normalized velocity or "spin" of each bird. The first term captures velocity alignment between neighbors, while the second represents external influences like thermal gradients.

### Uniform Coupling

The coupling strength between agents follows a uniform model where all agents interact with the same base strength:

$`
\hspace{0.5cm} \displaystyle
J_{ij} = J_0 \exp(-d_{ij}/\lambda)
`$
<br>

where the topological distance $`d_{ij}`$ counts the minimum number of neighbor-to-neighbor hops between agents, and $`J_0 = 1.6`$ is the optimized uniform baseline coupling strength. This uniform coupling ensures that behavioral diversity emerges from heterogeneous noise amplitudes rather than coupling heterogeneity.

The alignment force emerges from the energy gradient:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{align}} = \sum_{j \in \mathcal{N}_k(i)} J_{ij} (\mathbf{v}_j - \mathbf{v}_i)
`$
<br>

### Critical State Through Noise Heterogeneity

Rather than heterogeneous coupling, criticality emerges through the distribution of individual noise amplitudes [^8]. This creates a continuous spectrum of behaviors, with agents having low $`\eta_i`$ strongly aligning with neighbors while those with high $`\eta_i`$ exhibiting more independent motion.

The variance in noise amplitudes maintains elevated susceptibility:

$`
\hspace{0.5cm} \displaystyle
\chi = \int_0^\xi C(r) dr
`$
<br>

where $`C(r)`$ is the velocity correlation function and $`\xi`$ is the correlation length. At criticality, susceptibility scales with flock size as $`\chi \sim L^{1.08}`$ [^9], indicating the system maintains responsiveness at all scales without saturation.

## Self-Propulsion Dynamics

### Active Matter Framework

Birds, unlike passive particles, generate their own motion through wing beats. This self-propulsion places them in the category of active matter, systems that consume energy to move. Active matter theory shows these systems exhibit unique phase transitions and collective phenomena impossible in equilibrium systems [^18]. The Vicsek model demonstrates how self-propelled particles with velocity alignment can spontaneously break symmetry and move collectively.

For self-propelled particles, the phase transition between disordered and ordered motion depends on the noise-to-speed ratio $`\eta/v_0`$ and density $`\rho`$. The order parameter (polarization) follows:

$`
\hspace{0.5cm} \displaystyle
\Phi = \frac{1}{N} \left| \sum_i \hat{\mathbf{v}}_i \right|
`$
<br>

where $`\Phi \approx 0`$ indicates disordered motion and $`\Phi \approx 1`$ represents collective alignment. At the critical point, fluctuations exhibit scaling behavior $`\delta\Phi^2 \sim N^{-\alpha}`$ with $`\alpha < 1`$, indicating long-range correlations.

### Marginal Speed Confinement

Natural flocks maintain stable cruising speeds while preserving the scale-free correlations necessary for collective response. The marginal speed confinement framework [^16] resolves this apparent conflict through a quartic potential that regulates individual speeds without damping fluctuations:

$`
\hspace{0.5cm} \displaystyle
V(\mathbf{v}_i) = \frac{\lambda}{v_0^6}(|\mathbf{v}_i|^2 - v_0^2)^4
`$
<br>

The potential is marginal at the reference speed, meaning its second derivative vanishes at $`v_0`$. This mathematical property eliminates quadratic restoring forces that would otherwise destroy scale-free correlations, since a harmonic potential would create exponentially decaying correlations rather than the observed power-law behavior.

The force on each agent emerges as the negative gradient:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{speed}} = -\frac{4\lambda}{v_0^6}(|\mathbf{v}_i|^2 - v_0^2)^3\mathbf{v}_i
`$
<br>

This force exhibits asymmetric behavior around $`v_0`$. Near the cruising speed where $`|\mathbf{v}_i| \approx v_0`$, the cubic term $(|\mathbf{v}_i|^2 - v_0^2)^3 \approx 0$ creates minimal resistance, allowing natural speed variations of ±2 m/s observed in starling flocks. For extreme deviations, the force grows as the seventh power of velocity, providing strong confinement within biomechanical limits.

### Complete Self-Propulsion Force

The full self-propulsion force combines speed regulation with environmental coupling and stochastic noise:

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{prop}} = -\frac{4\lambda}{v_0^6}(|\mathbf{v}_i|^2 - v_0^2)^3\mathbf{v}_i + \alpha_w \mathbf{w}_i + \eta_i(t) \boldsymbol{\xi}_i
`$
<br>

Each agent maintains a cruising speed of $`v_0 = 11.1`$ m/s [^21] through the quartic force, while incorporating environmental wind $`\mathbf{w}_i`$ with coupling $`\alpha_w = 0.3`$. In practice, numerical instability arises from the $`v_0^6 \approx 1.88 \times 10^6`$ term in the denominator, necessitating a dimensionless formulation for computational stability. By introducing the normalized speed $`s_i = |\mathbf{v}_i|/v_0`$, we can express the force as

$`
\hspace{0.5cm} \displaystyle
\mathbf{F}_{\text{speed}} = -4\lambda_\text{eff}(s_i^2 - 1)^3 s_i v_0 \hat{\mathbf{v}}_i
`$
<br>

wherein $`\lambda_\text{eff} = J_0 \cdot r_s`$ represents the effective confinement strength and $`\hat{\mathbf{v}}_i = \mathbf{v}_i/|\mathbf{v}_i|`$ denotes the velocity direction. This formulation preserves mathematical equivalence to the original while ensuring numerical stability, such that the speed regulation ratio $`r_s \in [0.01, 0.3]`$ balances individual regulation against collective alignment. When $`r_s < 0.1`$, collective forces dominate and speeds may drift, whereas values exceeding $`r_s > 0.2`$ strengthen individual speed control but reduce collective responsiveness. Stochastic fluctuations $`\eta_i(t) \boldsymbol{\xi}_i`$ with $`\boldsymbol{\xi}_i \sim \mathcal{N}(0, \mathbf{I})`$ maintain the behavioral diversity described in the heterogeneous noise model below.

### Heterogeneous Noise Model

Individual noise amplitudes vary temporally, drawn from:

$`
\hspace{0.5cm} \displaystyle
\eta_i(t) \sim \mathcal{N}(0.33, 0.20)
`$
<br>

This heterogeneous noise model creates a continuous spectrum of behavioral responses. The standard deviation $`\sigma = 0.20`$ represents a transition point where:

- For $`\sigma < 0.11`$: First-order transitions with band formation

- For $`\sigma \geq 0.20`$: Continuous phase transitions enabling flock formation

At this value, the system exhibits the observed critical exponents of $`\beta = 0.69`$, $`\gamma = 1.7`$, and $`\nu = 1.56`$ [^8].

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

Under predator attack, starling flocks exhibit density waves that sweep through the group at speeds far exceeding individual flight velocities. Field measurements show these waves propagating at 15-45 m/s [^12], creating characteristic "ink-like" patterns that confuse predators. These density fluctuations arise from a reaction-diffusion process where local compression triggers expansion in neighboring regions.

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

Thermal safety constraints guide agents away from dangerous temperature regions using smooth, differentiable penalty functions. The Kreisselmeier-Steinhauser (KS) formulation, developed for systematic control design [^13], provides gradient-based corrections that integrate seamlessly with neural network training while maintaining computational efficiency.

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

The speed at which information propagates through the flock depends on the distribution of noise amplitudes across the population. Empirical measurements show propagation speeds of 15-45 m/s [^12]. The heterogeneous noise distribution in our model creates similar variability.

### Scale-Free Correlations

The combination of topological interactions and heterogeneous noise amplitudes produces velocity correlations that decay as a power law:

$`
\hspace{0.5cm} \displaystyle
C(r) \sim r^{-1/3}
`$
<br>

This exponent matches empirical observations across multiple species and flock sizes [^10], indicating a universal principle independent of specific biological details.

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

with timeframe $`\Delta t = 0.1`$ s chosen to balance computational efficiency with numerical stability.

## Summary

The unified control law achieves biological murmuration dynamics through orchestrated interactions between multiple components. Heterogeneous noise amplitudes maintain criticality. Topological interactions enable scale-free correlations. Self-propulsion with wind coupling and individualized noise creates realistic flight dynamics. Environmental responses translate thermal threats into coordinated evasion. Thermal penalties enforce safety through smooth gradient corrections.

The resulting system exhibits properties matching empirical observations:

- Susceptibility scaling $`\chi \sim L^{1.08}`$ [^9]

- Information propagation at 15-45 m/s [^12]

- Scale-free correlations $`C(r) \sim r^{-1/3}`$ [^10]

- Critical exponents $`\beta = 0.69`$, $`\gamma = 1.7`$, $`\nu = 1.56`$ [^8]

- Network connectivity with $`\lambda_2 > 0`$

- Thermal safety $`T < 475`$ K

This framework transforms invisible thermal threats into visible motion patterns that humans can intuitively interpret, potentially providing life-saving information in wildfire scenarios where traditional sensors fail to capture the complex, dynamic nature of thermal hazards.

---

## References

[^7]: Bialek, William, Andrea Cavagna, Irene Giardina, Thierry Mora, Edmondo Silvestri, Massimiliano Viale, and Aleksandr M. Walczak. 2012. "Statistical Mechanics for Natural Flocks of Birds." *Proceedings of the National Academy of Sciences* 109 (13): 4786–91. https://doi.org/10.1073/pnas.1118633109

[^8]: Guisandez, Javier, Miguel Hoyuelos, and Horacio Sergio Wio. 2018. "Heterogeneous Agents Can Always Reach a Consensus: A Systematic Study." *Physical Review E* 98 (4): 042308. https://doi.org/10.1103/PhysRevE.98.042308

[^9]: Attanasi, Alessandro, Andrea Cavagna, Lorenzo Del Castello, Irene Giardina, Stefano Melillo, Leonardo Parisi, Oliver Pohl, Bruno Rossaro, Edward Shen, Edmondo Silvestri, and Massimiliano Viale. 2014. "Finite-Size Scaling as a Way to Probe Near-Criticality in Natural Swarms." *Physical Review Letters* 113 (23): 238102. https://doi.org/10.1103/PhysRevLett.113.238102

[^10]: Cavagna, Andrea, Alessio Cimarelli, Irene Giardina, Giorgio Parisi, Raffaele Santagati, Fabio Stefanini, and Massimiliano Viale. 2010. "Scale-Free Correlations in Starling Flocks." *PNAS* 107 (26): 11865–70. https://doi.org/10.1073/pnas.1005766107

[^11]: Ballerini, M. et al. 2008. "Interaction Ruling Animal Collective Behavior Depends on Topological Rather than Metric Distance." *PNAS* 105 (4): 1232–37. https://doi.org/10.1073/pnas.0711437105

[^12]: Attanasi, Alessandro, et al. 2014. "Information Transfer and Behavioural Inertia in Starling Flocks." *Nature Physics* 10 (9): 691–696. https://doi.org/10.1038/nphys3035

[^13]: Kreisselmeier, G., and R. Steinhauser. 1979. "Systematic Control Design by Optimizing a Vector Performance Index." *IFAC Proceedings Volumes* 12 (7): 113–17. https://doi.org/10.1016/S1474-6670(17)65584-8

[^16]: Cavagna, Andrea, Antonio Culla, Xiao Feng, Irene Giardina, Tomas S. Grigera, Willow Kion-Crosby, Stefania Melillo, Giulia Pisegna, Lorena Postiglione, and Pablo Villegas. 2022. "Marginal Speed Confinement Resolves the Conflict Between Correlation and Control in Collective Behaviour." *Nature Communications* 13 (1): 2315. https://doi.org/10.1038/s41467-022-29883-4

[^17]: Heisenberg, Werner. 1928. "Zur Theorie des Ferromagnetismus." *Zeitschrift für Physik* 49 (9): 619–636. https://doi.org/10.1007/BF01328601

[^18]: Ginelli, Francesco. 2016. "The Physics of the Vicsek Model." *The European Physical Journal Special Topics* 225 (11): 2099–2117. https://doi.org/10.1140/epjst/e2016-60066-8

[^21]: Ballerini, M., et al. 2008. "Empirical Investigation of Starling Flocks: A Benchmark Study in Collective Animal Behaviour." *Animal Behaviour* 76 (1): 201–215. https://doi.org/10.1016/j.anbehav.2008.02.004


