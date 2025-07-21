# Data Procurement Guide

This guide provides instructions for obtaining the WRF-SFIRE wildfire simulation data used to train Thermur's thermally-aware flocking policies. The dataset contains Large Eddy Simulations *(LES)* of wildfire plumes with the thermal and fluid dynamics needed to train safe and legible flock behaviors.

## Dataset Overview

The **Moisseeva (2020) WRF-SFIRE LES Synthetic Wildfire Plume Dataset** contains **147 NetCDF files** totaling **5.33 TB**, with individual files ranging from 20-50 GB each. At **40m atmospheric resolution** with 4m fire mesh refinement, these simulations capture fine-scale turbulence patterns.

Each **20-30 minute simulation** traces a wildfire plume from ignition through fully-developed convection columns. The **coupled fire-atmosphere modeling** uses WRF-SFIRE's two-way feedback mechanism: fire heat release drives atmospheric motion, which in turn affects fire spread. This coupling produces the complex flow fields that characterize real wildfire environments.

Each simulation captures the complete 3D evolution of a wildfire plume. The data reveals **turbulent mixing processes and vortex dynamics** that create the chaotic motion patterns our flock must mirror for legibility. **Convective heat transport and radiative cooling** define the thermal boundaries that trigger our safety protocols, while **wind-fire interactions and plume-driven circulations** generate the complex flow fields through which our agents navigate. The simulations also track **smoke transport via passive tracer advection**, providing visual density cues that inform flock opacity adjustments.


### Why This Dataset?

The Moisseeva & Stull *(2020)* dataset was selected for factors that align with Thermur's biomimetic objectives:

1. **Biological Relevance**: The 40m resolution captures thermal gradients at the scale a bird or drone would experience while navigating around a fire. This matches the perceptual scale at which murmurations respond to predator threats.

2. **Turbulence Fidelity**: LES modeling explicitly resolves large eddies while parameterizing sub-grid turbulence, providing realistic chaotic motion patterns that our flock must learn to mirror. Just as starlings create visual chaos when evading a hawk, our drones must reflect fire-induced turbulence.

3. **Safety Training Data**: The dataset's temperature fields reach $`T > 600\text{K}`$ in plume cores, well beyond drone survival limits. This provides training data for Control Barrier Functions to learn hard thermal boundaries, analogous to how starlings maintain minimum separation distances.

4. **Temporal Dynamics**: The 15-second timesteps capture the rapid evolution of fire plumes, allowing our Graph Neural Networks to learn temporal patterns in thermal threat propagation, much like how information cascades through a murmuration.

5. **Validated Physics**: Published in peer-reviewed repositories with accompanying methodology papers, this dataset ensures we're training on scientifically accurate fire behavior, not approximations or empirical models.

## Data Acquisition Options

### Option 1: Sample Data (Quick Start)

For immediate testing and development, download our carefully curated 1.5GB sample:

```bash
# Download sample data from Hugging Face
thermur download --source sample

# Train with sample data
thermur train --sample

# Sample automatically used if no downloaded data exists
thermur train
```

**Why This Sample?**

The 1.5 GB sample *(`data/samples/wrf_sample.nc`)* was extracted from `wrfout_W5F7R4`, representing the Goldilocks zone of fire simulation data. This moderate intensity scenario strikes an optimal balance for algorithm development.

Unlike the gentle 3 m/s winds of low-wind scenarios, this 5 m/s breeze creates sufficient plume tilt and asymmetry to challenge formation-keeping algorithms. Yet, it avoids the opposite extreme of 12 m/s winds that would immediately sweep agents into dangerous thermal zones. The **Southern rough** fuel type *(F7)* creates a heterogeneous burning pattern with both smoldering ground fuels and active flaming fronts. In addition, minutes 3-4 of this particular simulation capture established fire spread with fully-developed convection columns reaching into the boundary layer.

This sample contains enough complexity to develop core algorithms while remaining computationally tractable for rapid iteration. Full specifications are in `data/samples/README.md`.

### Option 2: Direct Download via Thermur CLI *(2-4 hours per file)*

Download individual files using Globus Connect Personal:

```bash
# Interactive file selection
thermur download

# Extended timeout for slow connections (48 hours)
thermur download download.transfer_timeout=172800
```

**Setup Requirements:**
1. Install [Globus Connect Personal](https://www.globus.org/globus-connect-personal)
2. Run `thermur download` and follow authentication prompts
3. Select files from the interactive menu
4. Monitor transfer progress *(shows transfer rate in MB/s)*

**Recommended Starter Files:**

Each file represents a distinct fire behavior regime that trains different aspects of collective intelligence:

- **`wrfout_W3F1R0`** - *Gentle*: Light 3 m/s winds over short grass create predictable, laminar plume rise. Ideal for initial policy training where agents learn basic thermal avoidance and cohesion without chaotic disturbances.

- **`wrfout_W5F7R4`** - *Balanced*: Our sample scenario. Moderate winds through mixed fuels create realistic turbulence and thermal gradients. Trains robust policies that generalize well.

- **`wrfout_W8F13R6`** - *Stressful*: Strong winds through logging slash generate intense, tilted plumes with extreme turbulence. Tests whether learned policies maintain safety under severe conditions.

- **`wrfout_W12F4R8`** - *Extreme*: Near-catastrophic winds in chaparral with strong inversions. Pushes the theoretical limits of drone survival, essential for validating Control Barrier Functions.

### Option 3: HPC Cluster Transfer *(Recommended for Full Dataset)*

For institutional users with HPC access:

```bash
# On your HPC cluster
globus transfer \
  f163c1b3-9c88-42f6-a7bb-5839ed6c4063:/1/published/publication_309/submitted_data/ \
  YOUR_ENDPOINT_ID:/path/to/storage/ \
  --recursive
```

**Advantages:** The HPC approach reduces download times from days to hours. Endpoint-to-endpoint transfers bypass local network bottlenecks, achieving speeds 10-100x faster than Globus Connect Personal. This makes acquiring the entire 5.33 TB dataset feasible for research groups with access to high performance computing, like [**Explorer** at Northeastern Univeristy](https://rc-docs.northeastern.edu/en/explorer-main/).

## File Naming Convention

The systematic naming encodes the complete parameter space, allowing targeted selection for specific training objectives:

`wrfout_W*F*R*[L*][T][E]`

### Primary Parameters

- **W (Wind)**: Initial wind speed from 3-12 m/s
  - W3-W5: Light conditions for basic training
  - W6-W8: Moderate winds introducing complex plume dynamics  
  - W9-W12: Extreme conditions for robustness testing

- **F (Fuel)**: Anderson 13 fuel model categories
  - F1-F3: Grasses (fast-spreading, low intensity)
  - F4-F7: Shrubs and understory (moderate intensity, heterogeneous)
  - F8-F10: Timber litter (slow-spreading, long duration)
  - F11-F13: Slash and heavy fuels (extreme intensity, spotting)

- **R (Stability)**: Atmospheric profiles affecting plume rise
  - R0-R2: Unstable (strong mixing, tall plumes)
  - R3-R5: Neutral (typical afternoon conditions)
  - R6-R8: Stable (suppressed plumes, horizontal spread)

### Optional Modifiers

- **L**: Fireline length (1-4 km) - affects total heat release

- **T**: Tall domain (5000m) - for deep convective plumes

- **E**: Extended runtime (30 min) - captures longer-term dynamics

## Storage Requirements

Plan your storage accordingly:

- Single file: 20-50 GB

- Recommended starter set (4 files): ~150 GB

- Full dataset: 5.33 TB

Default locations:
- Full dataset: `data/wrf-sfire/`
- Sample data: `data/samples/`

## Network Considerations

**Download Speed Estimates:**

- 10 Mbps: ~12 hours per file

- 100 Mbps: ~1.5 hours per file

- 1 Gbps: ~10 minutes per file

The Thermur CLI shows real-time transfer rates to help estimate completion times while running the `thermur download` command.

## Using Your Own Globus Endpoint

If you have access to the dataset through a different Globus endpoint (e.g., your institution's copy), you can configure Thermur to use it by modifying the following Pydantic fields in `src/configs/cli/schemas/interaction.py`:

```python
class DownloadModel(BaseModel):
    # Update these fields for your endpoint:
    globus_endpoint_id: str = Field(
        default     = "YOUR_ENDPOINT_UUID",
        description = "UUID of your Globus endpoint"
    )
    globus_dataset_path: str = Field(
        default     = "/path/to/dataset/",
        description = "Path to dataset within your endpoint"
    )
    
    # Sample data configuration (if hosting your own):
    sample_data_url: str = Field(
        default     = "YOUR_HUGGINGFACE_URL",
        description = "Direct download URL for sample data (e.g., from Hugging Face)"
    )
    
    # These typically remain the same:
    globus_client_id: str = Field(
        default     = "ac349f52-8197-4a41-8d6d-5ae1c879273f",
        description = "Native app client ID for OAuth2"
    )
    globus_scopes: str = Field(
        default     = "urn:globus:auth:scope:transfer.api.globus.org:all",
        description = "OAuth2 scopes for transfer operations"
    )
```

To find your endpoint UUID:

1. Log into [Globus Web App](https://app.globus.org)

2. Navigate to your endpoint

3. Copy the UUID from the endpoint details

**Authentication Note**: The `thermur download` command handles OAuth2 authentication automatically. On first run, it opens your browser for Globus login and securely stores refresh tokens in `~/.config/thermur/secrets/`. These tokens persist across sessions, so you only need to authenticate once per machine.

## Understanding the Physics

Training thermally-aware flocks requires understanding the physical processes captured in these simulations. The WRF-SFIRE data provides high-resolution views of wildfire dynamics that directly inform our safety constraints and learning objectives.

### Heat Transfer Mechanisms

Wildfire environments present three distinct thermal challenges for aerial systems. **Convection** dominates the vertical dimension, with hot gases rising at velocities of $`w = 10-20`$ m/s. Since most micro-drones only achieve 3-5 m/s descent rates, these powerful thermals can trap them in uncontrollable ascent.

**Radiation** creates a different threat near the fire front, where heat flux can exceed $`\dot{q}_r > 100`$ kW/m². At these intensities, plastic drone components begin melting within seconds. The danger extends well beyond visible flames, so drones must maintain significant separation even from areas that appear safe.

**Turbulent mixing** throughout the plume creates the chaotic velocity fields that characterize wildfire smoke. These patterns span from meter-scale eddies to large vortices hundreds of meters across. Our flock learns to match this turbulence, transforming invisible airflow into visible coordinated motion that firefighters can interpret.

### Safety Thresholds

Converting physical measurements to operational limits requires understanding drone hardware constraints. We enforce a **temperature threshold** of $`T_{\max} = 475\text{K}`$ *(202°C)* based on lithium battery chemistry and aerogel protection. Above this temperature, batteries enter thermal runaway and fail catastrophically. This represents a hard limit rather than gradual degradation.

**Vertical velocity** limits of $`|w| = 10`$ m/s reflect the aerodynamic capabilities explained above. The **heat flux limit** of $`\dot{q}_{\max} = 10`$ kW/m² protects electronic components from cumulative thermal damage, particularly in motors and speed controllers.

These constraints suggest maintaining altitude above $`h > 50\text{m}`$ AGL. This height provides separation from the most intense heating while preserving enough margin for emergency maneuvers when conditions change rapidly.

### Training Implications

These physical constraints directly shape our learning approach. Control Barrier Functions enforce temperature and velocity limits as absolute boundaries that no learned behavior can violate. This ensures safety regardless of reward structures or training objectives.

The flock's behavior naturally reflects environmental conditions through this safety-first design. Formations remain tight in calm air but loosen as turbulence increases, since matching chaotic flow patterns while maintaining safety requires more spacing. This visible degradation communicates atmospheric stability to ground observers.

Color mapping follows the same physics-informed principle. Blues indicate safe temperatures, progressing through yellows to reds as conditions approach operational limits. Formation geometry also adapts to thermal boundaries, with the flock deforming to trace temperature gradients and reveal the shape of invisible dangers.

## Citation

When using this dataset, please cite both the dataset and the methodology paper:

```bibtex
@dataset{moisseeva2020wrfsfire,
  author    = {Moisseeva, Nadya and Stull, Roland},
  title     = {WRF-SFIRE LES Synthetic Wildfire Plume Dataset},
  year      = {2020},
  publisher = {Federated Research Data Repository},
  doi       = {10.20383/102.0314}
}

@article{moisseeva2021acp,
  author    = {Moisseeva, Nadya and Stull, Roland},
  title     = {Capturing plume rise and dispersion with a coupled 
               large-eddy simulation: case study of a prescribed burn},
  journal   = {Atmospheric Chemistry and Physics},
  volume    = {21},
  pages     = {12359--12374},
  year      = {2021},
  doi       = {10.5194/acp-21-12359-2021}
}
```

## Next Steps

After acquiring data:

1. **Explore the physics**: Use tools like `xarray` and `matplotlib` to visualize temperature and velocity fields
2. **Verify thermal safety**: Check that maximum temperatures align with hardware constraints
3. **Start training**: Begin with `thermur train --sample` to test your setup
4. **Monitor convergence**: Use `thermur monitor` to track safety violations and legibility metrics
5. **Iterate on full data**: Graduate to downloaded files for robust policy learning

Remember: This data represents real fire physics. The chaotic, dangerous air patterns in these simulations are exactly what firefighters face, and what our bio-inspired flock aims to make visible.