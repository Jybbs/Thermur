# Murmuration Dynamics for Thermur: A Supervised Learning Approach

## Abstract

This document outlines the replacement of basic geometric formations with biologically-inspired murmuration dynamics in the Thermur thermal drone swarm system. Building upon the existing imitation learning framework, we propose a new murmuration controller based on topological interaction rules and critical state dynamics observed in starling flocks (Ballerini et al., 2008; Cavagna et al., 2010; Bialek et al., 2012). The approach leverages supervised learning techniques to train a Graph Neural Network policy that exhibits scale-free correlations and rapid information propagation while maintaining compatibility with the existing Control Barrier Function safety layer. This implementation focuses exclusively on murmuration as the primary formation mode, removing the need for static cube or sphere formations.

## 1. Architectural Renaming Strategy

To embed murmuration as the core concept of Thermur, we propose the following comprehensive renaming:

### Core Component Renaming
- **`ExpertController` → `MurmurationController`** - The primary controller implementing murmuration dynamics

- **`ExpertModel` → `MurmurationModel`** - Unified configuration for all murmuration parameters

- **`expert.py` → `murmuration.py`** - File renaming to match the controller

- **Keep `FlockModel`** - Represents the overall swarm configuration with murmuration as its behavior

- **Use `mmm` abbreviation** - Delightful shorthand for MurmurationModel (e.g., `self.mmm`)

### Conceptual Shift
Rather than having murmuration as one formation option among many, it becomes THE fundamental behavior:
- Remove `initial_formation` field from `FlockModel`

- Add `murmuration: MurmurationDynamics` field directly to `FlockModel`

- All swarm dynamics are murmuration dynamics

This renaming makes murmuration the defining behavior of every flock, not an optional formation.

## 2. Mathematical Framework

### 2.1 Enhanced Hamiltonian Formulation

Following Bialek et al. (2012), we extend the current potential-based controller to incorporate an Ising-like Hamiltonian for velocity alignment:

$`
\hspace{0.5cm} \displaystyle
E = -\sum_{i,j} J_{ij} \mathbf{s}_i \cdot \mathbf{s}_j - \sum_i \mathbf{h}_i \cdot \mathbf{s}_i
`$  
<br>

where:
- $`\mathbf{s}_i = \mathbf{v}_i / |\mathbf{v}_i|`$ are normalized velocity vectors

- $`J_{ij}`$ are pairwise interaction strengths with topological decay

- $`\mathbf{h}_i`$ represents external fields (thermal gradients)

### 2.2 Topological Interaction Model

The interaction strength between agents follows topological distance $`d_{ij}`$ rather than metric distance:

$`
\hspace{0.5cm} \displaystyle
J_{ij} = J_0 \exp\left(-\frac{d_{ij}}{\lambda}\right) \cdot \mathbb{1}_{d_{ij} \leq k}
`$  
<br>

where:
- $`d_{ij}`$ is the number of nearest-neighbor hops

- $`k = 6-7`$ is the topological interaction range (Ballerini et al., 2008)

- $`\lambda`$ is the decay length scale

### 2.3 Mode-Dependent Dynamics

The control law switches between cruise and alert modes based on thermal threat level:

$`
\hspace{0.5cm} \displaystyle
\mathbf{u}_{\text{nom}}^{(i)} = \begin{cases}
    -\nabla_{\mathbf{x}_i} U_{\text{cruise}}(\mathbf{S}_t) & \text{if } h_{\text{threat}}(i) < \theta_{\text{alert}} \\
    -\nabla_{\mathbf{x}_i} U_{\text{alert}}(\mathbf{S}_t) & \text{otherwise}
\end{cases}
`$  
<br>

where the alert potential includes enhanced correlation terms:

$`
\hspace{0.5cm} \displaystyle
U_{\text{alert}} = U_{\text{cruise}} + \alpha_{\text{corr}} \sum_{i,j} w_{ij} |\mathbf{v}_i - \mathbf{v}_j|^2 + \beta_{\text{dense}} \sum_{i,j} \|\mathbf{x}_i - \mathbf{x}_j\|^2
`$  
<br>

The cruise mode uses standard Reynolds potentials:

$`
\hspace{0.5cm} \displaystyle
U_{\text{cruise}} = \omega_c U_{\text{coh}} + \omega_s U_{\text{sep}} + \omega_a U_{\text{align}} + \omega_t U_{\text{therm}}
`$  
<br>

For murmuration, the alignment weight becomes state-dependent:

$`
\hspace{0.5cm} \displaystyle
\omega_a = \omega_a^{(0)} \cdot (1 + \alpha_{\chi} \cdot \chi(\mathbf{S}_t))
`$  
<br>

where $`\chi(\mathbf{S}_t)`$ is the susceptibility measuring proximity to critical state.

### 2.4 Information Propagation Speed

The speed of information transfer through the flock follows:

$`
\hspace{0.5cm} \displaystyle
v_{\text{info}} = c_0 \sqrt{\frac{\chi}{m_{\text{eff}}}}
`$  
<br>

where $`\chi`$ is the susceptibility and $`m_{\text{eff}}`$ is the effective mass. Empirically, $`v_{\text{info}} \in [15, 45]`$ m/s.

### 2.5 Density Wave Dynamics

In alert mode, density waves propagate according to:

$`
\hspace{0.5cm} \displaystyle
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = D \nabla^2 \rho + S_{\text{threat}}
`$  
<br>

where $`S_{\text{threat}}`$ is a source term that increases density near thermal threats.

## 3. Supervised Learning Approach

### 3.1 Training Data Generation

Generate synthetic demonstrations using the enhanced Hamiltonian dynamics:

1. Initialize flock with small random velocities

2. Simulate dynamics with known parameters $`(J_0, \lambda, k)`$

3. Record state-action pairs: $`\mathcal{D} = \{(\mathbf{s}_t^{(i)}, \mathbf{u}_t^{(i)})\}_{t,i}`$

### 3.2 Learning Objectives

Train the GNN policy $`\pi_\theta`$ to minimize:

$`
\hspace{0.5cm} \displaystyle
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{imitation}} + \sum_{m} \alpha_m \mathcal{L}_m
`$  
<br>

where $`\mathcal{L}_m`$ are formation-specific metric losses defined in Section 3.

## 4. Murmuration-Specific Metrics

### 4.1 Scale-Free Correlation Metric

Measures deviation from theoretical power-law decay (Cavagna et al., 2010):

$`
\hspace{0.5cm} \displaystyle
\mathcal{M}_{\text{corr}} = \frac{1}{N_r} \sum_{r} \left( \log C(r) - \log C_0 + \gamma \log r \right)^2
`$  
<br>

where:
- $`C(r) = \langle \delta\mathbf{v}_i \cdot \delta\mathbf{v}_j \rangle_{|r_i-r_j|=r}`$ is the velocity correlation function

- $`\gamma = 1/3`$ is the theoretical exponent

- $`C_0`$ is a normalization constant

### 4.2 Topological Fidelity Metric

Quantifies concentration of interaction on k-nearest neighbors:

$`
\hspace{0.5cm} \displaystyle
\mathcal{M}_{\text{topo}} = \frac{1}{N} \sum_i \frac{\sum_{j \in \mathcal{N}_k(i)} w_{ij}}{\sum_{j \in \mathcal{N}(i)} w_{ij}}
`$  
<br>

where:
- $`\mathcal{N}_k(i)`$ are the k-nearest neighbors of agent $`i`$

- $`w_{ij} = |\mathbf{v}_i \cdot \mathbf{v}_j|`$ measures velocity alignment

Target: $`\mathcal{M}_{\text{topo}} \geq 0.85`$

### 4.3 Susceptibility Metric

Measures responsiveness to perturbations:

$`
\hspace{0.5cm} \displaystyle
\chi = N \cdot \text{Var}[\Phi], \quad \Phi = \frac{1}{N}\left|\sum_i \frac{\mathbf{v}_i}{|\mathbf{v}_i|}\right|
`$  
<br>

Target range: $`\chi \in [5, 20]`$ for critical state behavior

### 4.4 Information Propagation Speed

Measures velocity of perturbation propagation:

$`
\hspace{0.5cm} \displaystyle
v_{\text{info}} = \frac{\Delta r}{\Delta t}
`$  
<br>

where $`\Delta r`$ is the distance traveled by a velocity perturbation in time $`\Delta t`$.

Target: $`v_{\text{info}} \in [15, 45]`$ m/s (empirical range)

### 4.5 Dynamic Balance Score

Combines local disorder with global order:

$`
\hspace{0.5cm} \displaystyle
\mathcal{M}_{\text{balance}} = \exp\left(-\left|\Phi - \Phi_{\text{target}}\right|\right) \cdot \left(1 - \exp(-\sigma_v^2)\right)
`$  
<br>

where:
- $`\Phi_{\text{target}} = 0.7`$ represents moderate alignment

- $`\sigma_v^2`$ is the local velocity variance

## 5. Implementation Architecture

### 5.1 Configuration Schema Extensions

Add to `src/config/imitation/controller/schemas.py`:

```python
from pydantic import BaseModel, Field, NonNegativeFloat, PositiveFloat, PositiveInt


class MurmurationModel(BaseModel, extra="forbid"):
    """
    Unified configuration for murmuration dynamics and control weights.
    
    Combines topological interaction parameters, critical state dynamics,
    and Reynolds rule weights into a single model that fully specifies
    the murmuration behavior. The controller uses these parameters to
    implement biologically-inspired flocking with scale-free correlations
    and rapid information propagation.
    
    The murmuration exists at a critical state (phase transition) where
    susceptibility diverges, enabling near-instantaneous response to threats
    while maintaining cohesion through topological neighbor tracking.
    """
    alert_threshold: PositiveFloat = Field(
        default     = 0.3,
        description = (
            "Normalized threat level θ_alert ∈ [0,1] triggering transition from cruise "
            "to alert mode, where 0 represents ambient temperature and 1 represents T_max."
        )
    )
    correlation_exponent: PositiveFloat = Field(
        default     = 0.333,
        description = (
            "Target power-law exponent γ ≈ 1/3 for velocity correlation decay C(r) ∼ r^(-γ), "
            "matching empirical observations of scale-free correlations in starling flocks."
        )
    )
    correlation_strength: PositiveFloat = Field(
        default     = 1.5,
        description = (
            "Additional alignment weight α_corr applied in alert mode to enhance velocity "
            "correlation and create tighter, more responsive collective motion."
        )
    )
    coupling_decay: PositiveFloat = Field(
        default     = 0.5,
        description = (
            "Exponential decay rate λ for topological interaction strength J_ij = J_0 exp(-d_ij/λ), "
            "controlling how influence diminishes with topological distance."
        )
    )
    density_strength: PositiveFloat = Field(
        default     = 0.8,
        description = (
            "Additional cohesion weight β_dense applied in alert mode to increase flock "
            "density, creating the characteristic 'ink-like' appearance during evasion."
        )
    )
    epsilon: PositiveFloat = Field(
        default     = 1e-8,
        description = (
            "Numerical stability constant ε preventing division by zero in "
            "potential gradient calculations, particularly for separation forces."
        )
    )
    k_neighbors: PositiveInt = Field(
        default     = 7,
        description = (
            "Number of topological nearest neighbors each agent tracks, based on "
            "empirical observations of 6-7 neighbors in real starling flocks."
        )
    )
    min_distance: PositiveFloat = Field(
        default     = 0.1,
        description = (
            "Minimum inter-agent distance ε_dist in meters enforced during separation "
            "force computation to prevent singularities in U_sep = Σ 1/||𝐱_i - 𝐱_j||."
        )
    )
    susceptibility_amplification: PositiveFloat = Field(
        default     = 2.0,
        description = (
            "Amplification factor α_χ for alignment weight modulation based on susceptibility, "
            "creating stronger velocity correlation as the flock approaches critical state."
        )
    )
    susceptibility_target: PositiveFloat = Field(
        default     = 10.0,
        description = (
            "Target susceptibility χ_target for maintaining critical state dynamics, where "
            "χ = N·Var[Φ] measures the flock's responsiveness to perturbations."
        )
    )
    temperature_scaling: PositiveFloat = Field(
        default     = 1.0,
        description = (
            "Multiplicative factor λ_thermal adjusting thermal avoidance strength "
            "relative to Reynolds forces, balancing safety versus cohesive behavior."
        )
    )
    w_alignment: NonNegativeFloat = Field(
        default     = 1.0,
        description = (
            "Base weight ω_align for velocity alignment potential U_align = (1/2)Σ||𝐯_i - 𝐯_j||², "
            "modulated by susceptibility to achieve critical state dynamics."
        )
    )
    w_cohesion: NonNegativeFloat = Field(
        default     = 0.8,
        description = (
            "Weight ω_coh for cohesion potential U_coh = (1/2)Σ||𝐱_i - 𝐱_j||², slightly "
            "reduced from baseline to account for topological neighborhood effects."
        )
    )
    w_separation: NonNegativeFloat = Field(
        default     = 1.2,
        description = (
            "Weight ω_sep for separation potential U_sep = Σ 1/||𝐱_i - 𝐱_j||, maintained "
            "at standard level to ensure collision avoidance regardless of formation."
        )
    )
    w_thermal: NonNegativeFloat = Field(
        default     = 2.0,
        description = (
            "Weight ω_thermal for thermal potential U_thermal = 1/(T_max - T_i), creating "
            "exponentially stronger avoidance forces as temperature approaches T_max."
        )
    )


class FlockModel(BaseModel, extra="forbid"):
    """
    Unified configuration for the thermal drone flock.
    
    Maintains existing fields for agent count, communication range, and
    thermal properties. The initial_formation field is removed since
    murmuration is now the only movement pattern.
    """
    agent_count: PositiveInt = Field(
        default     = 30,
        gt          = 1,
        description = "Number of agents in the flock."
    )
    communication_range: PositiveFloat = Field(
        default     = 50.0,
        description = "Maximum distance for inter-agent communication."
    )
    thermal_time_constant: PositiveFloat = Field(
        default     = 5.0,
        description = "Time constant for thermal dynamics modeling."
    )
    # Remove: initial_formation field
    # Remove: formation_scale_factor (no longer needed without formations)
```

Add to `src/config/imitation/controller/builds.py`:

```python
from hydra_zen                    import builds, make_config
from thermur.imitation.controller import MurmurationController, SafetyFilter

from .schemas import *


CONTROLLER_USER_CONFIG = make_config(
    flock      = FlockModel(),
    mmm        = MurmurationModel(),
    safety     = SafetyModel(),
    thresholds = ThresholdsModel()
)

CONTROLLER_SYSTEM_BUILDS: dict[str, type[Builds[Any]]] = {
    "murmuration_controller": builds(
        MurmurationController,
        flock                   = "${controller.flock}",
        mmm                     = "${controller.mmm}",
        safety_filter           = "${_system.safety_filter}",
        thresholds              = "${controller.thresholds}",
        populate_full_signature = True,
        zen_partial             = True
    ),
    
    "safety_filter": builds(
        SafetyFilter,
        flock                   = "${controller.flock}",
        safety                  = "${controller.safety}",
        thresholds              = "${controller.thresholds}",
        populate_full_signature = True,
        zen_partial             = True
    )
}
```

### 5.2 Enhanced Murmuration Controller

Rename and modify `src/thermur/imitation/controller/expert.py` → `murmuration.py`:

```python
from tensordict import TensorDictBase
from torch      import Tensor

import torch as th


class MurmurationController:
    """
    Implements murmuration dynamics with topological interactions.
    
    This controller generates biologically-inspired flocking behavior based on
    starling murmurations, using topological neighborhoods (k-nearest neighbors)
    rather than metric distances. The flock maintains critical state dynamics
    for rapid information propagation and exhibits distinct cruise/alert modes.
    """
    
    def __init__(
        self,
        flock         : FlockModel,
        mmm           : MurmurationModel,
        thresholds    : ThresholdsModel,
        safety_filter : SafetyFilter | None = None
    ):
        """
        Initialize the murmuration controller.
        
        Args:
            flock         : Flock configuration with agent count and communication range
            mmm           : Murmuration model with dynamics and weight parameters  
            thresholds    : Safety thresholds including max temperature
            safety_filter : Optional CBF-based safety filter for control limiting
        """
        self.flock           = flock
        self.mmm             = mmm  # Delightful shorthand!
        self.thresholds      = thresholds
        self.safety_filter   = safety_filter
        self.last_positions  = None  # For topological neighbor computation
        self._reset_shared_state()
    
    def _compute_topological_neighbors(
        self, 
        positions : Tensor, 
        k         : int
    ) -> Tensor:
        """
        Compute k-nearest neighbors for each agent using topological distance.
        
        Args:
            positions : Tensor [N, 3] containing agent positions
            k         : Number of nearest neighbors to connect
            
        Returns:
            Tensor [2, E] edge index in COO format for PyG
        """
        distances       = th.cdist(positions, positions)
        _, indices      = distances.topk(k + 1, largest=False)
        
        # Convert to edge_index format, excluding self-connections
        edge_source = []
        edge_target = []
        
        for i in range(len(positions)):
            for j in indices[i, 1:]:  # Skip first (self)
                edge_source.append(i)
                edge_target.append(j.item())
                
        return th.tensor([edge_source, edge_target], dtype=th.long)
    
    def _compute_susceptibility(self, velocities: Tensor) -> Tensor:
        """
        Compute flock susceptibility χ = N · Var[Φ].
        
        Susceptibility measures the flock's responsiveness to perturbations.
        At critical state, χ diverges, enabling rapid information propagation.
        
        Args:
            velocities : Tensor [N, 3] containing agent velocities
            
        Returns:
            Scalar susceptibility value
        """
        normalized_vels = velocities / velocities.norm(dim=1, keepdim=True).clamp(min=1e-8)
        polarization    = normalized_vels.mean(dim=0).norm()
        variance        = ((normalized_vels.mean(dim=0).norm() - polarization) ** 2)
        
        return len(velocities) * variance
    
    def _compute_threat_level(self, temperature: Tensor) -> Tensor:
        """
        Compute normalized threat level for mode switching.
        
        Maps temperature to [0, 1] range where 0 is safe and 1 is critical.
        
        Args:
            temperature : Tensor [N] or [N, 1] containing agent temperatures
            
        Returns:
            Tensor [N] of normalized threat levels
        """
        temp_normalized = (temperature - self.thresholds.max_temperature * 0.7) / (
            self.thresholds.max_temperature * 0.3
        )
        
        return temp_normalized.clamp(0, 1)
    
    def _update_graph_state(
        self,
        edge_index : Tensor,
        num_agents : int
    ):
        """
        Update graph connectivity using topological neighborhoods.
        
        Overrides metric neighborhoods with k-nearest neighbor topology.
        
        Args:
            edge_index : Initial edge connectivity (ignored)
            num_agents : Total number of agents in flock
        """
        # Always use topological neighbors for murmuration
        if self.last_positions is not None:
            edge_index = self._compute_topological_neighbors(
                positions = self.last_positions,
                k         = self.mmm.k_neighbors
            )
            
        # Update shared graph state (inherited from base implementation)
        if edge_index.numel():
            self._edge_source, self._edge_target = edge_index
            self._neighbor_count = th.bincount(
                self._edge_source,
                minlength = num_agents
            )
        else:
            self._reset_shared_state(edge_index.device)
            self._neighbor_count = th.zeros(
                num_agents,
                device = edge_index.device,
                dtype  = th.long
            )
            
        self._safe_count = th.clamp(self._neighbor_count, min=1)
    
    def compute_nominal_action(self, flock: TensorDictBase) -> Tensor:
        """Enhanced compute with murmuration dynamics."""
        self.last_positions = flock["position"]
        self._update_graph_state(flock["edge_index"], flock["position"].size(0))
        
        # Base Reynolds forces
        cohesion = self._compute_cohesion(flock["position"])
        separation = self._compute_separation(flock["position"])
        alignment = self._compute_alignment(flock["velocity"])
        thermal = self._compute_thermal(
            gradient=flock["gradient"],
            temperature=flock["temperature"]
        )
        
        # Compute susceptibility for state-dependent weights
        susceptibility = self._compute_susceptibility(flock["velocity"])
        w_align_amplified = self.mmm.w_alignment * (
            1 + self.mmm.susceptibility_amplification * 
            torch.tanh(susceptibility / self.mmm.susceptibility_target)
        )
        
        # Check for alert mode
        threat_levels = self._compute_threat_level(flock["temperature"])
        in_alert_mode = threat_levels.max() > self.mmm.alert_threshold
        
        if in_alert_mode:
            # Enhanced correlation and density in alert mode
            u_nominal = (
                self.mmm.w_cohesion * cohesion +
                self.mmm.w_separation * separation +
                w_align_amplified * alignment +
                self.mmm.w_thermal * thermal +
                self.mmm.correlation_strength * alignment +  # Extra alignment
                self.mmm.density_strength * cohesion       # Extra cohesion
            )
        else:
            # Standard cruise mode with susceptibility-modulated alignment
            u_nominal = (
                self.mmm.w_cohesion * cohesion +
                self.mmm.w_separation * separation +
                w_align_amplified * alignment +
                self.mmm.w_thermal * thermal
            )
        
        if self.safety_filter is not None:
            return self.safety_filter.filter(flock, u_nominal)
        return u_nominal
```

### 5.3 Murmuration Metrics Integration

Add to `src/thermur/imitation/monitoring/metrics.py`:

```python
from torchmetrics import Metric, MetricCollection

import torch as th


class ScaleFreeCorrelationMetric(AveragingMetric):
    """
    Measures deviation from power-law velocity correlations.
    
    Computes the velocity correlation function C(r) and fits a power law
    to verify scale-free behavior characteristic of critical systems.
    """
    
    def __init__(self, target_exponent: float = 0.333):
        """
        Initialize with target power-law exponent.
        
        Args:
            target_exponent : Expected gamma value (1/3 for 3D murmurations)
        """
        super().__init__()
        self.target_exponent = target_exponent
    
    def update(
        self,
        positions  : Tensor,
        velocities : Tensor
    ):
        """
        Update metric with current flock state.
        
        Args:
            positions  : Tensor [N, 3] of agent positions
            velocities : Tensor [N, 3] of agent velocities
        """
        # Compute velocity fluctuations
        v_mean  = velocities.mean(dim=0)
        delta_v = velocities - v_mean
        
        # Compute pairwise distances and correlations
        distances = th.cdist(positions, positions)
        
        # Compute correlation for each distance bin
        # Implementation would bin distances and compute average correlation
        # per bin, then fit log C(r) = log C_0 - gamma log r
        
        # For now, compute simplified metric
        correlation_sum = 0.0
        count          = 0
        
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if distances[i, j] > 0:
                    correlation = th.dot(delta_v[i], delta_v[j])
                    correlation_sum += correlation / (distances[i, j] ** self.target_exponent)
                    count += 1
                    
        if count > 0:
            self.sum += correlation_sum / count
            self.count += 1


# In MetricsCollector._init_evaluation_metrics():
def _init_murmuration_metrics(self):
    """Initialize murmuration-specific evaluation metrics."""
    self.train_murmuration = MetricCollection({
        "correlation_mse"      : ScaleFreeCorrelationMetric(),
        "dynamic_balance"      : DynamicBalanceMetric(), 
        "info_speed"           : InformationPropagationMetric(),
        "susceptibility"       : SusceptibilityMetric(),
        "topological_fidelity" : TopologicalFidelityMetric()
    })
    self.val_murmuration = self.train_murmuration.clone(prefix="val_")
```

### 5.4 Training Pipeline Integration

The existing imitation learning pipeline requires minimal changes:

1. **Data Generation**: MurmurationController generates demonstrations
2. **Training**: GNN policy learns from these demonstrations (no changes needed)
3. **Evaluation**: Additional metrics computed during validation

## 6. Implementation Notes

### 6.1 Simple CBF Integration

For initial implementation, we use a simple linear Control Barrier Function:
- Define a vertical plane at x = x_barrier as the safety boundary

- The CBF constraint becomes: h(x) = x - x_barrier ≥ 0

- This creates a "wall" that the murmuration must avoid

- The safety filter modifies control actions to ensure the flock stays on the safe side

### 6.2 Training Approach

1. **Generate demonstrations**: Run MurmurationController to create state-action pairs

2. **Train GNN policy**: Use behavioral cloning to learn from demonstrations

3. **Evaluate metrics**: Monitor the five murmuration-specific metrics during validation

### 6.3 Supervised Learning Justification

This implementation uses three concrete supervised learning techniques from class:

**1. Non-linear Regression**
- **What**: The GNN policy learns to predict continuous control actions $`u \in \mathbb{R}^3`$ from input states

- **Training Data**: (state, action) pairs from MurmurationController demonstrations

- **Loss Function**: MSE between predicted and expert actions: $`L = ||u_{\text{predicted}} - u_{\text{expert}}||^2`$

- **Non-linearity**: Multiple GNN layers with ReLU activations capture complex state-action mappings

**2. K-Nearest Neighbors (KNN)**
- **What**: Each agent connects to exactly k=7 nearest neighbors in 3D space

- **Implementation**: `_compute_topological_neighbors()` finds k-NN for graph construction

- **Difference from Metric**: Traditional uses radius r, we use neighbor count k

- **Direct Application**: The topological distance metric is pure KNN - no learned parameters

**3. Loss Minimization with Multiple Objectives**
- **Primary Loss**: Behavioral cloning MSE (standard supervised learning)

- **Auxiliary Losses**: Regression targets for emergent properties:
  - Scale-free correlation coefficient (target: $`R^2 > 0.9`$)
  - Susceptibility value (target: $`\chi \in [5, 20]`$)
  - Information propagation speed (target: $`v \in [15, 45]`$ m/s)

- **Training**: Gradient descent on combined loss: $`L_{\text{total}} = L_{\text{BC}} + \sum_i \alpha_i L_{\text{metric}_i}`$

These are the tangible supervised ML components, and everything else emerges from these three core techniques.

## 7. Implementation Plan

### Phase 1: Foundation (Week 1)
1. **Rename Core Components**
   - Rename `expert.py` → `murmuration.py`
   - Update all imports and references
   - Commit: `refactor: rename expert controller to murmuration controller`

2. **Implement MurmurationModel Schema**
   - Add schema to `schemas.py` with all parameters
   - Ensure alphabetical ordering and proper field descriptions
   - Commit: `feat: add MurmurationModel configuration schema`

3. **Update Builds Configuration**
   - Modify `builds.py` to use MurmurationController
   - Add proper hydra-zen interpolation patterns
   - Remove formation-related builds
   - Commit: `feat: update hydra-zen builds for murmuration architecture`

### Phase 2: Core Dynamics (Week 2)
4. **Implement Topological Neighborhoods**
   - Add `_compute_topological_neighbors()` method
   - Override `_update_graph_state()` to use k-NN
   - Test with k=7 neighbors
   - Commit: `feat: implement topological neighbor computation`

5. **Add Susceptibility Computation**
   - Implement `_compute_susceptibility()` method
   - Add susceptibility-based weight modulation
   - Verify $`\chi \in [5, 20]`$ range
   - Commit: `feat: add susceptibility-based critical state dynamics`

6. **Implement Mode Switching**
   - Add threat level computation
   - Implement cruise/alert mode logic
   - Add enhanced correlation and density terms
   - Commit: `feat: implement adaptive cruise/alert mode switching`

### Phase 3: Metrics and Evaluation (Week 3)
7. **Implement Scale-Free Correlation Metric**
   - Add correlation function computation
   - Implement power-law fitting
   - Verify $`\gamma \approx 1/3`$ exponent
   - Commit: `feat: add scale-free correlation metric`

8. **Add Remaining Murmuration Metrics**
   - Topological fidelity metric
   - Information propagation speed
   - Dynamic balance score
   - Commit: `feat: complete murmuration evaluation metrics`

9. **Integrate Metrics into Training Pipeline**
   - Add to MetricsCollector
   - Update logging to track murmuration metrics
   - Commit: `feat: integrate murmuration metrics into training`

### Phase 4: Training and Validation (Week 4)
10. **Generate Murmuration Demonstrations**
    - Create dataset using MurmurationController
    - Verify emergent behaviors in demonstrations
    - Commit: `feat: generate murmuration training demonstrations`

11. **Train Initial GNN Policy**
    - Use existing imitation learning pipeline
    - Monitor all five murmuration metrics
    - Achieve baseline performance
    - Commit: `feat: train baseline murmuration GNN policy`

12. **Optimize and Fine-tune**
    - Adjust MurmurationModel parameters
    - Fine-tune for better metric scores
    - Validate emergent behaviors
    - Commit: `feat: optimize murmuration dynamics parameters`

### Validation Criteria
- Scale-free correlation: $`R^2 > 0.9`$ for power-law fit
- Topological fidelity: > 85% interaction concentration on k-NN
- Susceptibility: $`\chi \in [5, 20]`$ during cruise mode
- Information speed: $`v_{\text{info}} \in [15, 45]`$ m/s
- Dynamic balance: score > 0.7

### Risk Mitigation
- Start with reduced flock size (N=10) for faster iteration
- Use simple 2D scenarios before full 3D dynamics
- Implement comprehensive unit tests for each component
- Monitor computational performance (GNN + k-NN overhead)

## 8. References

- Ballerini, M. et al. (2008). "Interaction ruling animal collective behavior depends on topological rather than metric distance." PNAS.

- Bialek, W. et al. (2012). "Statistical mechanics for natural flocks of birds." PNAS.

- Cavagna, A. et al. (2010). "Scale-free correlations in starling flocks." PNAS.