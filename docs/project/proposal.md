# Thermur: A Supervised Learning Approach to Thermally-Constrained Flocking for Wildfire Visualization

James Parkington  
June 21, 2025  
DS 5220: Supervised Machine Learning and Learning Theory

## Problem Description

### Starling Flocks: A Biomimetic Model for Safety Visualization

I've spent countless hours watching starling murmurations since my childhood in Western Massachusetts. During winter breaks from school, I'd often walk through the snow-covered fields near our home at dusk, when thousands of starlings would gather before roosting. I still have a vivid memory from when I was about nine, watching a hawk cut through the flock directly overhead. What captivated me wasn't just the synchronized beauty of their flight, but how instantly they transformed when threatened. The smooth, flowing patterns shattered into chaotic motion, with sharp directional changes and an even tighter ink-like density in the sky.

Even more compelling was that I felt I could instinctively track the hawk's position by watching these ripples of disorder move through the flock. The invisible predator became **visible** through the birds' collective response.

This childhood fascination gained profound significance years later, while touring the Southwestern United States. On June 30, 2013, nineteen members of the Granite Mountain Hotshots were fatally overtaken by the Yarnell Hill Fire[^1]. The crew's lookout had lost visual contact with the fire, and by the time its convective plume reappeared, the lethal speed and direction left no time for escape. My friends and I visited the entrapment after stopping in Prescott, Arizona, only a few years after the tragedy. What happened at Yarnell Hill underscores a fundamental challenge in wildfire management, which is that the most critical information on a fireground—the instantaneous, localized flow of heat and air—is **dangerously invisible**.

### Information Asymmetry in Wildfire Management

Current approaches to wildfire visualization rely primarily on numerical data from sparse sensor networks or satellite imagery with limited temporal resolution. While anything is valuable in an active fire-fight, these methods fail to leverage the human brain's exceptional capacity for processing visual motion patterns[^3]. Cognitive science research has established that humans possess specialized neural pathways for perceiving biological motion—the ability to extract meaningful information from even minimal representations of moving entities[^4].

Herein lies my hypothesis, and the motivation for the project. Could we create a swarm of robots that responds to thermal threats the way starlings react to a predator? **Thermur** *(a portmanteau of "thermal" and "murmuration")* emerged from this biomimetic insight.

### Mathematical Formulation

From a computational perspective, Thermur addresses the following problem:

**Input**: A dynamic, three-dimensional temperature and wind velocity field ($`T(x,y,z,t)`$, $`v(x,y,z,t)`$) representing wildfire conditions.

**Output**: A coordinated motion pattern and color display across a swarm of $N$ robots that:

1. **Safely navigates** the thermal environment (*never exceeding critical temperature thresholds*)

2. **Renders the invisible** thermal and wind patterns visually legible to human observers for safer positioning and better anticipation of risk areas

3. **Communicates urgency** and danger through biologically-inspired motion characteristics. 

<br>

More formally, given a thermal field $`T(x,y,z,t)`$ and $N$ agents with states $`S = \{s^1, s^2, ..., s^N\}`$, find a policy $`\pi: S \rightarrow A`$ that maps agent states to actions while:

- Maintaining thermal safety: $`T(x^i) < T_{max}`$ for all agents $i$

- Maximizing visual information transfer *(measured by structural similarity between the swarm's emergent motion pattern and the underlying wind field)*

- Preserving swarm cohesion through topological neighbor relationships inspired by starling murmurations

In my opinion, the novelty inherent in this approach is its focus on creating an intuitive visual language through motion and color, rather than symbolic or numerical representations. By leveraging our innate perceptual abilities, Thermur tries to reduce cognitive load in high-stress situations and provide firefighters with immediate, intuitive understanding of their environment. It transforms invisible threats into visible patterns, just as starlings reveal the position of a hidden predator.

## Project Participants

I am undertaking this practical project as a solo participant, diverging from the recommended team size of two. My rationale for this approach stems from my intention to develop Thermur as an extended, multi-course project that will serve as a foundation for my work in upcoming classes ("Unsupervised Machine Learning", "Building Scalable Distributed Systems", "Large-Scale Parallel Data Processing") and ultimately culminate in my **capstone**.

From my childhood memories of watching murmurations to visiting various sites of fire tragedies, I've developed a deep connection to both the biological inspiration and the life-saving potential of this work. I anticipate investing substantially more than the suggested 40-60 hours during this class interval, driven by genuine enthusiasm and belief in its potential impact. Working alone allows me to maintain this accelerated pace without imposing it on teammates.

For this specific class, I will confine my presentation and paper to the supervised machine learning components that I can reasonably accomplish within the 40-60 hour scope while demonstrating meaningful progress in the broader vision. To be abundantly clear, my intent is to specifically develop and present on these components:

1. **Imitation learning** for flocking behavior using Graph Neural Networks

2. **Supervised regression** for temperature-to-color mapping

3. **Evaluation methodologies** for measuring both model performance and visual legibility

## Literature Review

The Thermur project sits at the intersection of several research domains, including swarm robotics, biomimetic control, thermal sensing, and information visualization. While each of these areas has a rich research history, their combination in the context of wildfire visualization represents a **novel approach**, in my opinion.

The foundational work on computational models of flocking begins with Reynolds' 1987 paper introducing the "boids" model[^5], which decomposed flocking behavior into three simple rules: separation, alignment, and cohesion. While revolutionary, this model assumed metric interactions *(fixed radius)* rather than the topological relationships discovered later.

Ballerini et al. (2008) found that starlings interact with a fixed number of neighbors (typically 6-7) regardless of density, not all birds within a certain distance[^11]. This topological interaction model is important for Thermur's approach, as it allows for a control strategy that is inherently scalable and computationally efficient for large robot swarms.

Further work by Bialek et al. (2012) and Cavagna et al. (2010) revealed that starling flocks exist in a state near criticality, allowing information to propagate rapidly across the entire flock[^7][^13]. I think this property is valuable for Thermur, as it suggests a swarm can reflect large-scale wind shifts without central control, and proposed a reasonable analogue to backpropoagation in a computationally tractable deep-learning model.

For the thermal safety aspect, **Control Barrier Functions (CBFs)** have emerged as an effective tool. Wang et al. (2017) demonstrated how CBFs can provide mathematical guarantees of safety in multi-robot systems while minimally interfering with nominal control objectives[^9]. Ames et al. (2016) extended this work to formulate safety constraints as quadratic programs solvable in real-time[^14].

Recent work by Häusermann et al. (2023) on thermally protected drones provides important empirical data on the survivability of small UAVs in high-temperature environments, demonstrating that modern micro-robotics can now withstand brief excursions up to 400°F[^6].

The concept of using motion as an information medium builds on Ishii and Ullmer's (1997) pioneering work on "tangible bits" and ambient displays[^8]. However, existing work typically focuses on static or screen-based displays rather than embodied, dynamic systems.

In summary, while previous research has developed components of what Thermur requires—swarm control, thermal sensing, safety guarantees, no existing system integrates these elements to create an embodied information display for wildfire dynamics.

## Algorithms

For the scope of this course, I will develop and simulate two core supervised learning components of the Thermur system:

### Imitation Learning for Flocking Behavior

The primary algorithm will be **Behavioral Cloning**, a form of supervised learning where a neural network policy is trained to mimic an expert controller. This approach is particularly appropriate for Thermur because:

- It allows us to incorporate domain knowledge from the physics-based "expert" controller while gaining the computational efficiency of a neural network
- It provides a foundation that can later be fine-tuned with reinforcement learning
- It's well-suited to the high-dimensional, continuous action space of the swarm control problem

The learning process will involve generating demonstrations using a custom expert controller based on physical principles derived from mathematics by Reynolds et al.[^5], training a **Graph Neural Network (GNN)** to predict the expert's actions given state observations, and evaluating the trained policy on its ability to maintain safe, cohesive flocking while responding appropriately to thermal gradients.

Similar approaches have been successfully applied in other multi-agent control problems, such as Gama et al. (2021)[^17], who demonstrated that GNNs can effectively learn decentralized controllers for swarm systems.

### Supervised Learning for Temperature-to-Color Mapping

The second algorithm will be a supervised regression task to learn a perceptually-accurate color mapping function. This will involve:

- Training a small **Multi-Layer Perceptron (MLP)** to map temperature values to colors in a perceptually uniform color space (CIELAB)
- Optimizing for minimal perceptual difference between the temperature values and their color representations

This component ensures that the visual information displayed by the swarm is immediately legible to humans, particularly in high-stress situations where rapid perception is critical.

In both cases, I'll use **Mean Squared Error (MSE)** as the primary loss function, with standard stochastic gradient descent methods for optimization. For the GNN architecture, I'll implement the **Encoder-Processor-Decoder** paradigm, which follows state-of-the-art practices in graph representation learning.[^16]

### Safety Guarantees via Control Barrier Functions

To formally guarantee thermal safety, we will leverage Control Barrier Functions (CBFs). A CBF, $`h(x)`$, is a function of the system's state $`x`$ that defines a safe set of conditions where $`h(x) \geq 0`$. For Thermur, a simple CBF would be $`h(x) = T_{\text{max}} - T(x)`$, where $`T(x)`$ is the temperature at the agent's position.

Following the work of Ames et al. [14], we can enforce this safety constraint in real-time by solving the following Quadratic Program (QP) to find the final control action $`\mathbf{u}`$:

$`
\hspace{0.5cm} \displaystyle
\begin{aligned}
\mathbf{u}^*(x) = \arg\min_{\mathbf{u} \in \mathbb{R}^m} & \quad \frac{1}{2} \|\mathbf{u} - \mathbf{u}_{\text{imitation}}(x)\|^2 \\
\text{s.t.} & \quad L_f h(x) + L_g h(x) \mathbf{u} \ge -\alpha(h(x))
\end{aligned}
`$  
<br>

Here, $`\mathbf{u}_{\text{imitation}}`$ is the nominal control input from the primary imitation learning policy. The constraint ensures that the final action $`\mathbf{u}^*`$ will not cause the agent to violate the safety boundary (i.e., it keeps $`h(x)`$ from becoming negative). This QP finds an action that is as close as possible to the desired flocking behavior while rigorously satisfying the thermal safety limits.

### Evaluation Methodology

In order to ensure the performance of these models is validated and understood with rigor, I will use the following evaluation methodology:

1. **K-Fold Cross-Validation**: To ensure robust performance estimates, I'll use 5-fold cross-validation when evaluating model performance, reporting mean and standard deviation across folds.

2. **Quantitative Metrics**:
   - For the imitation learning task: MSE between predicted and expert actions, as well as higher-level metrics like swarm cohesion[^18] and thermal safety violation rate
   - For the color mapping task: Mean Absolute Error (MAE) in temperature prediction from color, and perceptual color difference (ΔE) in CIELAB space

3. **Ablation Studies**: To understand the contribution of different components, I'll conduct ablation studies by removing or modifying key aspects of the models *(e.g., varying GNN layers, message passing iterations)*.

4. **Visual Validation**: Beyond numerical metrics, I'll develop visualization tools to qualitatively assess the swarm's behavior, examining whether it exhibits the desired emergent properties.

5. **Out-of-Distribution Testing**: To evaluate generalization, I'll test the models on scenarios significantly different from the training distribution *(e.g., different fire intensities, novel terrain assuming that data is readily accessible)*.

This comprehensive evaluation approach will provide both quantitative performance measures and qualitative insights into the models' capabilities and limitations.

## Data Sets

### Training Data Sources

For training the supervised learning models, I will use a combination of synthetic and empirical data:

1. **Synthetic Wildfire Environments**: The primary training data will come from established atmospheric and fire-spread models, which allow for training in a vast range of repeatable scenarios:

   - **WRF-Fire / FARSITE Output**: I'll use 3D gridded data of wind velocity, temperature, and heat flux from these coupled weather-wildfire models[^20]. These provide realistic environmental dynamics based on semi-empirical models like the Rothermel spread equations[^21].

   - **Large-Eddy Simulation (LES) Plume Data**: For fine-scale turbulence, I'll leverage public datasets like the library from Moisseeva (2020)[^22], which captures important turbulent eddies.

2. **UAV Thermal Telemetry**: To ground the thermal models in physical reality:

   - **FireDrone Kiln Tests**: Data from aerogel-protected drones tested in industrial kilns provides important information on thermal lag between ambient temperature and a drone's surface and core temperatures[^6].

   - **USGS Thermal UAV Dataset**: Thermal imaging data collected by drones over natural environments provides real-world validation points[^23].

3. **Starling Flight Recordings**: While no comprehensive public dataset of starling murmurations exists, I'll synthesize demonstrations based on mathematical principles derived from key research papers mentioned above:
   
   - Implementing the **topological interaction model** from Ballerini et al. [11], where each agent interacts with a fixed number (6-7) of nearest neighbors rather than all agents within a metric radius, capturing the scale-invariant property of real starling flocks.
   
   - Incorporating the **scale-free correlation structure** identified by Cavagna et al. [13], where velocity fluctuations exhibit power-law correlations ($`C(r) \sim r^{-\gamma}`$) rather than exponential decay, enabling global information transfer.
   
   - Applying the **maximum entropy model** developed by Bialek et al. [7], which characterizes the pairwise interactions between birds through an effective energy function. This function takes a form analogous to the Ising model in physics:
   
     $`
     \hspace{0.5cm} \displaystyle
     E({\bf s}) = -\sum_{i<j} J_{ij} {\bf s}_i \cdot {\bf s}_j - \sum_i {\bf h}_i \cdot {\bf s}_i
     `$  
     <br>
     
     In this model, $`{\bf s}_i`$ is the normalized velocity vector (or "spin") of agent $`i`$, such that $`{\bf s}_i = {\bf v}_i / |{\bf v}_i|`$. The term $`J_{ij}`$ represents the alignment interaction strength between agents $`i`$ and $`j`$, while $`{\bf h}_i`$ is an external field that can represent environmental forces like wind or a directional goal.


### Data Preprocessing

The raw data proposed above requires significant preprocessing to be suitable for training:

1. **Eulerian-to-Lagrangian Transformation**: Converting the gridded (Eulerian) data from WRF-Fire into agent-centric (Lagrangian) observations. For each agent, I'll sample the surrounding 3D field to create an input tensor representing its local environment.

2. **Graph Construction**: For each time step, I'll construct a dynamic graph where nodes represent agents and edges connect topological neighbors (the 6-7 nearest agents).

3. **Thermal RC Modeling**: Using telemetry data to fit a simple resistor-capacitor thermal model that estimates drone core temperature based on skin-temperature readings.

4. **Domain Randomization**: Injecting noise and perturbations into the training data to ensure robustness, including synthetic gusts sampled from turbulence spectra characteristic of wildfires[^24].

5. **Data Augmentation**: Generating variations through rotations, reflections, and density changes to improve generalization.

This preprocessing pipeline will be implemented using PyTorch and NumPy, with dedicated functions for each transformation step to ensure reproducibility.

## Libraries and Tools

The MLOps infrastructure for Thermur has been one of my favorite aspects of the work so far. In other classes, I've found myself frustrated when the training code is a mess of hardcoded paths and magic numbers, making it nearly impossible to reproduce results. For Thermur, I'm building the foundation right from the start with proper configuration management, type validation, and dependency tracking. I've already derived a lot of satisfaction from learning more about the MLOps side of preparing Supervised Machine Learning models based on the `Pipeline` exercises we've done in class, and so I'm using this project as an opportunity to extend further into some of the modern ways researching teams orchestrate ML (e.g. `hydra-zen`, `poetry`, `Pydantic`).

### Core ML Framework
- **PyTorch**: For building and training neural network models
- **PyTorch Geometric**[^16]: For implementing Graph Neural Networks (GNNs)
- **TorchRL**: For the data collection and replay buffer components of imitation learning

### MLOps and Reproducibility
- **Hydra-zen**: For configuration management and experiment tracking
- **Pydantic**: For type validation and configuration schema definition
- **Poetry**: For dependency management and packaging
- **Weights & Biases**: As a stretch goal, I'd like to integrate W&B for experiment tracking and visualization

### Simulation and Visualization
- **MuJoCo**[^19]: For physics-based simulation of the drone swarm
- **matplotlib and VTK**: For visualization of the swarm's behavior and the environmental conditions

### Mathematics and Optimization
- **OSQP**[^25]: For solving the quadratic programs required by Control Barrier Functions
- **NumPy and SciPy**: For numerical operations and scientific computing

For this course, I'll need to learn and integrate several additional components:

1. **NetCDF and xarray**: For working with the gridded climate and fire model outputs
2. **Structural Similarity Index (SSIM)**[^28]: For quantitatively evaluating the legibility of the swarm's visual display
3. **ColorLab**: For working with perceptually uniform color spaces and color transformations

I've already made significant progress in setting up the ML infrastructure, having built some CLI architecture and model definitions in the codebase. For example, I have the skeleton of a `train_imitation_learning` function and a GNN policy module, which will support the supervised learning tasks I'll be focusing on.

## Results

### Ideal Outcome

The ideal outcome for this phase of the Thermur project is a trained swarm control system that demonstrates the following:

1. **Safe Navigation**: The swarm consistently maintains thermal safety constraints, never exceeding critical temperature thresholds *(e.g., 400°F)*.

2. **Legible Motion**: The swarm's collective motion accurately reflects the underlying wind patterns, with a Structural Similarity Index (SSIM)[^28] ≥ 0.80 when compared to ground-truth wind fields.

3. **Intuitive Temperature Display**: Each agent's LED color accurately represents its local temperature *(with MAE ≤ 5°F)*, providing an intuitive thermal map.

4. **Biologically-Inspired Behavior**: The swarm exhibits emergent behaviors reminiscent of starling murmurations, with increased motion complexity in response to thermal "predators" *(high-temperature regions)*.

5. The system maintains **robust** performance across various simulated wildfire scenarios, including complex terrain and rapidly changing conditions.

6. **Quantitative metrics** showing that the learned policy maintains swarm cohesion *(measured by graph connectivity)* while responding appropriately to thermal gradients. 

Rather than repeating expected results that can be inferred from the above, I'll focus on potential challenges and mitigation strategies in the next section.

### Risks and Mitigation Strategies

Several risks could impact the achievement of these results:

1. **Data Quality**: Synthetic data may not capture the full complexity of real-world wildfires.
   *Mitigation*: Use domain randomization and incorporate available real-world data to improve robustness.

2. **Imitation Learning Limitations**: The learned policy might not generalize beyond the distribution of expert demonstrations.
   *Mitigation*: Generate a diverse set of demonstrations across varying conditions; prepare for potential future reinforcement learning[^15] fine-tuning.

3. **Computational Constraints**: GNN training can be computationally intensive, especially with large graphs.
   *Mitigation*: Implement efficient batching strategies; use smaller swarm sizes for initial development.

4. **Metric Definition**: Quantifying "legibility" is inherently subjective.
   *Mitigation*: Combine multiple metrics to triangulate effectiveness.

If I encounter significant obstacles, I will prioritize the core supervised learning components while potentially simplifying the environmental complexity to ensure meaningful progress within the course timeframe.

## References

[1] U.S. Fire Administration. 2013. "Yarnell Hill Fire, Arizona." *Wildland Fire Fatality Reports*.

[3] Wolfe, Jeremy M. 2020. "Visual Search: How Do We Find What We Are Looking For?" *Annual Review of Vision Science* 6: 539–62. https://doi.org/10.1146/annurev-vision-091718-015048  

[4] Johansson, Gunnar. 1973. "Visual Perception of Biological Motion and a Model for Its Analysis." *Perception & Psychophysics* 14 (2): 201–11. https://doi.org/10.3758/BF03212378  

[5] Reynolds, Craig W. 1987. "Flocks, Herds and Schools: A Distributed Behavioral Model." *ACM SIGGRAPH Computer Graphics* 21 (4): 25–34. https://doi.org/10.1145/37402.37406

[6] Häusermann, D., et al. 2023. "FireDrone: Multi-Environment Thermally Agnostic Aerial Robot." *Advanced Intelligent Systems* 5 (23): 2300101. https://doi.org/10.1002/aisy.202300101

[7] Bialek, William, Andrea Cavagna, Irene Giardina, Thierry Mora, Edmondo Silvestri, Massimiliano Viale, and Aleksandr M. Walczak. 2012. "Statistical Mechanics for Natural Flocks of Birds." *Proceedings of the National Academy of Sciences* 109 (13): 4786–91. https://doi.org/10.1073/pnas.1118633109

[8] Ishii, Hiroshi, and Brygg Ullmer. 1997. "Tangible Bits: Towards Seamless Interfaces between People, Bits and Atoms." *CHI '97 Proceedings*, 234–41. https://doi.org/10.1145/258549.258715

[9] Wang, Li, Magnus Egerstedt, and Aaron D. Ames. 2017. "Safety Barrier Certificates for Collision-Free Multirobot Systems." *IEEE Transactions on Robotics* 33 (3): 661–74. https://doi.org/10.1109/TRO.2017.2659727

[11] Ballerini, M. et al. 2008. "Interaction Ruling Animal Collective Behavior Depends on Topological Rather than Metric Distance." *PNAS* 105 (4): 1232–37. https://doi.org/10.1073/pnas.0711437105

[13] Cavagna, Andrea, Alessio Cimarelli, Irene Giardina, Giorgio Parisi, Raffaele Santagati, Fabio Stefanini, and Massimiliano Viale. 2010. "Scale-Free Correlations in Starling Flocks." *PNAS* 107 (26): 11865–70. https://doi.org/10.1073/pnas.1005766107

[14] Ames, Aaron D., Xiangru Xu, Jessy W. Grizzle, and Paulo Tabuada. 2016. "Control Barrier Function-Based Quadratic Programs for Safety-Critical Systems." *IEEE Transactions on Automatic Control* 62 (8): 3861–76. https://doi.org/10.1109/TAC.2016.2638961

[15] Sutton, Richard S., and Andrew G. Barto. 2018. *Reinforcement Learning: An Introduction*. 2nd ed. MIT Press.

[16] Fey, Matthias, and Jan E. Lenssen. 2019. "Fast Graph Representation Learning with PyTorch Geometric." *arXiv* 1903.02428. https://doi.org/10.48550/arXiv.1903.02428

[17] Gama, Fernando, Ekaterina Tolstaya, and Alejandro Ribeiro. 2021. "Graph Neural Networks for Decentralized Controllers." *ICASSP 2021 — 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*: 5260–5264. https://doi.org/10.1109/ICASSP39728.2021.9414563

[18] Olfati-Saber, Reza. 2007. "Consensus and Cooperation in Networked Multi-Agent Systems." *Proceedings of the IEEE* 95 (1): 215–33. https://doi.org/10.1109/JPROC.2006.887293

[19] Todorov, Emanuel, Tom Erez, and Yuval Tassa. 2012. "MuJoCo: A Physics Engine for Model-Based Control." IROS 2012. https://github.com/deepmind/mujoco

[20] Coen, Janice L., et al. 2013. "WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research and Forecasting Model." *Journal of Applied Meteorology and Climatology* 52 (1): 16–38. https://doi.org/10.1175/JAMC-D-12-023.1

[21] Rothermel, Richard C. 1972. "A Mathematical Model for Predicting Fire Spread in Wildland Fuels." *USDA Forest Service Research Paper INT-115*.

[22] Moisseeva, Nadejda. 2020. *A Numerical Perspective on Wildfire Plume-Rise Dynamics*. (T). University of British Columbia. https://doi.org/10.14288/1.0395299

[23] Dawson, Cian B., Christopher Holmquist‑Johnson, and Martin A. Briggs. 2018. "Thermal Infrared and Photogrammetric Data Collected by Small Unoccupied Aircraft System for Hydrogeologic Analysis of Oh‑be‑joyful Creek, Gunnison National Forest, Colorado, August 2017." *U.S. Geological Survey Data Release*. https://doi.org/10.5066/1P931G95D

[24] Heilman, Warren E. 2023. "Atmospheric Turbulence and Wildland Fires: A Review." *International Journal of Wildland Fire* 32 (4): 476–495. https://doi.org/10.1071/WF22053

[25] Stellato, Bartolomeo, Goran Banjac, Paul Goulart, Alberto Bemporad, and Stephen Boyd. 2020. "OSQP: An Operator Splitting Solver for Quadratic Programs." *Mathematical Programming Computation* 12 (4): 637–672. https://doi.org/10.1007/s12532-020-00179-2

[28] Wang, Zhou, Alan C. Bovik, Hamid R. Sheikh, and Eero P. Simoncelli. 2004. "Image Quality Assessment: From Error Visibility to Structural Similarity." *IEEE Transactions on Image Processing* 13 (4): 600–612. https://doi.org/10.1109/TIP.2003.819861
