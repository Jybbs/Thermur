# Thermur: A Supervised Learning Approach to Thermally-Constrained Flocking for Wildfire Visualization

## Project Participants

I am undertaking this practical project as a solo participant, diverging from the recommended team size of two. My rationale for this approach stems from my intention to develop Thermur as an extended, multi-course project that will serve as a foundation for my work in upcoming classes ("Unsupervised Machine Learning", "Building Scalable Distributed Systems", "Large-Scale Parallel Data Processing") and ultimately culminate in my capstone.

This project represents more than just a course requirement for me—it's a passionate pursuit of something I believe can make a meaningful difference in wildfire safety. I anticipate investing substantially more than the suggested 40-60 hours during this class interval, driven by genuine enthusiasm and belief in its potential impact. Working alone allows me to maintain this accelerated pace without imposing it on teammates.

For this specific class, I will confine my presentation and paper to the supervised machine learning components that I can reasonably accomplish within the 40-60 hour scope while demonstrating meaningful progress in the broader vision.

## Problem Description

### The Invisible Killer

On June 30, 2013, nineteen members of the Granite Mountain Hotshots were fatally overtaken by the Yarnell Hill Fire. The crew's lookout had lost visual contact with the fire, and by the time its convective plume reappeared, the lethal speed and direction left no time for escape. This tragedy underscores a fundamental challenge in wildfire management: the most critical information on a fireground—the instantaneous, localized flow of heat and air—is dangerously invisible.

While anemometers provide numerical data, the human brain is wired for something far more primitive and powerful: the perception of coherent motion. Research has shown we can recognize complex activity from just a few moving points of light, a phenomenon known as "biological motion" perception. This insight forms the core of Thermur's approach.

### Computational Formulation

From a computational perspective, Thermur addresses the following problem:

**Input**: A dynamic, three-dimensional temperature and wind velocity field (T(x,y,z,t), v(x,y,z,t)) representing wildfire conditions.

**Output**: A coordinated motion pattern and color display across a swarm of N robots that:
1. Safely navigates the thermal environment (never exceeding critical temperature thresholds)
2. Renders the invisible thermal and wind patterns visually legible to human observers
3. Communicates urgency and danger through biologically-inspired motion characteristics

**Formal Definition**: Given a thermal field T(x,y,z,t) and N agents with states S = {s¹, s², ..., sᴺ}, find a policy π: S → A that maps agent states to actions while:
- Maintaining thermal safety: T(xⁱ) < Tₘₐₓ for all agents i
- Maximizing visual information transfer (measured by structural similarity between the swarm's emergent motion pattern and the underlying wind field)
- Preserving swarm cohesion through topological neighbor relationships inspired by starling murmurations

What makes this problem particularly interesting is the fusion of critical safety applications with biomimetic inspiration. Starling murmurations represent one of nature's most spectacular displays of coordinated motion—a living, breathing entity that communicates information through its fluid, mesmerizing form. By recasting the fire itself as the "predator" in this biological model, we can leverage evolved strategies for collective motion to create an intuitive, life-saving visual language.

## Literature Review

The Thermur project sits at the intersection of several research domains: swarm robotics, biomimetic control, thermal sensing, and information visualization. While each of these areas has a rich research history, their combination in the context of wildfire visualization represents a novel approach.

### Biological Collective Motion

The foundational work on computational models of flocking begins with Reynolds' 1987 paper introducing the "boids" model [5], which decomposed flocking behavior into three simple rules: separation, alignment, and cohesion. While revolutionary, this model assumed metric interactions (fixed radius) rather than the topological relationships discovered later.

Ballerini et al. (2008) [11] made the crucial discovery that starlings interact with a fixed number of neighbors (typically 6-7) regardless of density, not all birds within a certain distance. This topological interaction model is fundamental to Thermur's approach, as it allows for a control strategy that is inherently scalable and computationally efficient for large robot swarms.

Further work by Bialek et al. (2012) [7] and Cavagna et al. (2010) [13] revealed that starling flocks exist in a critical state, allowing information to propagate almost instantaneously across the entire flock. This property is paramount for Thermur, as it suggests a swarm can reflect large-scale wind shifts without central control.

### Safety-Critical Robotics

For the thermal safety aspect, Control Barrier Functions (CBFs) have emerged as a powerful tool. Wang et al. (2017) [9] demonstrated how CBFs can provide mathematical guarantees of safety in multi-robot systems while minimally interfering with nominal control objectives. Ames et al. (2016) [14] extended this work to formulate safety constraints as quadratic programs solvable in real-time.

Recent work by Häusermann et al. (2023) [6] on thermally protected drones provides crucial empirical data on the survivability of small UAVs in high-temperature environments, demonstrating that modern micro-robotics can now withstand brief excursions up to 400°F.

### Ambient Information Display

The concept of using motion as an information medium builds on Ishii and Ullmer's (1997) [8] pioneering work on "tangible bits" and ambient displays. However, existing work typically focuses on static or screen-based displays rather than embodied, dynamic systems.

### Novelty of Thermur

While previous research has developed components of what Thermur requires—swarm control, thermal sensing, safety guarantees—no existing system integrates these elements to create an embodied information display for wildfire dynamics. The key innovations of Thermur include:

1. Reframing wildfire as a "predator" in the biological flocking model, causing the swarm to exhibit more alert-like, chaotic motion in areas of higher temperature
2. Using supervised learning to train swarm behaviors that are both safe and visually informative
3. Developing a perceptually-uniform temperature-to-color mapping that intuitively communicates thermal conditions
4. Creating a scalable, decentralized control architecture that maintains global cohesion through local interactions

## Algorithms

For this Supervised Machine Learning course, I will focus on two core supervised learning components of the Thermur system:

### 1. Imitation Learning for Flocking Behavior

The primary algorithm will be **Behavioral Cloning**, a form of supervised learning where a neural network policy is trained to mimic an expert controller. This approach is particularly appropriate for Thermur because:

- It allows us to incorporate domain knowledge from the physics-based "expert" controller while gaining the computational efficiency of a neural network
- It provides a foundation that can later be fine-tuned with reinforcement learning
- It's well-suited to the high-dimensional, continuous action space of the swarm control problem

The learning process will involve:

1. Generating demonstrations using a hand-crafted expert controller based on physical principles
2. Training a Graph Neural Network (GNN) to predict the expert's actions given state observations
3. Evaluating the trained policy on its ability to maintain safe, cohesive flocking while responding appropriately to thermal gradients

Similar approaches have been successfully applied in other multi-agent control problems, such as Gama et al. (2021) [17], who demonstrated that GNNs can effectively learn decentralized controllers for swarm systems.

### 2. Supervised Learning for Temperature-to-Color Mapping

The second algorithm will be a supervised regression task to learn a perceptually-accurate color mapping function. This will involve:

1. Training a small Multi-Layer Perceptron (MLP) to map temperature values to colors in a perceptually uniform color space (CIELAB)
2. Optimizing for minimal perceptual difference between the temperature values and their color representations
3. Validating the mapping for intuitive human interpretation

This component ensures that the visual information displayed by the swarm is immediately legible to humans, particularly in high-stress situations where rapid perception is critical.

In both cases, I'll use Mean Squared Error (MSE) as the primary loss function, with standard stochastic gradient descent methods for optimization. For the GNN architecture, I'll implement the Encoder-Processor-Decoder paradigm shown in my current `GNNPolicy` class, which follows state-of-the-art practices in graph representation learning.

## Data Sets

### Training Data Sources

For training the supervised learning models, I will use a combination of synthetic and empirical data:

1. **Synthetic Wildfire Environments**: The primary training data will come from established atmospheric and fire-spread models, which allow for training in a vast range of repeatable scenarios:

   - **WRF-Fire / FARSITE Output**: I'll use 3D gridded data of wind velocity, temperature, and heat flux from these coupled weather-wildfire models [20]. These provide realistic environmental dynamics based on semi-empirical models like the Rothermel spread equations [21].

   - **Large-Eddy Simulation (LES) Plume Data**: For fine-scale turbulence, I'll leverage public datasets like the library from Moisseeva (2020) [22], which captures critical turbulent eddies.

2. **UAV Thermal Telemetry**: To ground the thermal models in physical reality:

   - **FireDrone Kiln Tests**: Data from aerogel-protected drones tested in industrial kilns [6] provides crucial information on thermal lag between ambient temperature and a drone's surface and core temperatures.

   - **USGS Thermal UAV Dataset**: Thermal imaging data collected by drones over natural environments [23] provides real-world validation points.

3. **Starling Flight Recordings**: While no comprehensive public dataset of starling murmurations exists, I'll use the mathematical principles derived from research papers [7, 11, 13] to generate synthetic demonstrations that capture the essential topological interaction properties.

### Data Preprocessing

The raw data requires significant preprocessing to be suitable for training:

1. **Eulerian-to-Lagrangian Transformation**: Converting the gridded (Eulerian) data from WRF-Fire into agent-centric (Lagrangian) observations. For each agent, I'll sample the surrounding 3D field to create an input tensor representing its local environment.

2. **Graph Construction**: For each time step, I'll construct a dynamic graph where nodes represent agents and edges connect topological neighbors (the 6-7 nearest agents).

3. **Thermal RC Modeling**: Using telemetry data to fit a simple resistor-capacitor thermal model that estimates drone core temperature based on skin-temperature readings.

4. **Domain Randomization**: Injecting noise and perturbations into the training data to ensure robustness, including synthetic gusts sampled from turbulence spectra characteristic of wildfires.

5. **Data Augmentation**: Generating variations through rotations, reflections, and density changes to improve generalization.

This preprocessing pipeline will be implemented using PyTorch and NumPy, with dedicated functions for each transformation step to ensure reproducibility.

## Libraries and Tools

The Thermur project leverages a modern ML engineering stack that I've already begun implementing:

### Core ML Framework
- **PyTorch**: For building and training neural network models
- **PyTorch Geometric**: For implementing Graph Neural Networks (GNNs)
- **TorchRL**: For the data collection and replay buffer components of imitation learning

### MLOps and Reproducibility
- **Hydra-zen**: For configuration management and experiment tracking
- **Pydantic**: For type validation and configuration schema definition
- **Weights & Biases**: For experiment tracking and visualization
- **Poetry**: For dependency management and packaging

### Simulation and Visualization
- **MuJoCo**: For physics-based simulation of the drone swarm
- **matplotlib and VTK**: For visualization of the swarm's behavior and the environmental conditions

### Mathematics and Optimization
- **OSQP**: For solving the quadratic programs required by Control Barrier Functions
- **NumPy and SciPy**: For numerical operations and scientific computing

For this course, I'll need to learn and integrate several additional components:

1. **NetCDF and xarray**: For working with the gridded climate and fire model outputs
2. **Structural Similarity Index (SSIM)**: For quantitatively evaluating the legibility of the swarm's visual display
3. **ColorLab**: For working with perceptually uniform color spaces and color transformations

I've already made significant progress in setting up the ML infrastructure, as evidenced by the CLI architecture and model definitions in the codebase. The `train_imitation_learning` function and GNN policy module are already structured to support the supervised learning tasks I'll be focusing on.

## Results

### Ideal Outcome

The ideal outcome for this phase of the Thermur project is a trained swarm control system that demonstrates:

1. **Safe Navigation**: The swarm consistently maintains thermal safety constraints, never exceeding critical temperature thresholds (e.g., 500°F).

2. **Legible Motion**: The swarm's collective motion accurately reflects the underlying wind patterns, with a Structural Similarity Index (SSIM) ≥ 0.80 when compared to ground-truth wind fields.

3. **Intuitive Temperature Display**: Each agent's LED color accurately represents its local temperature (with MAE ≤ 5°F), providing an intuitive thermal map.

4. **Biologically-Inspired Behavior**: The swarm exhibits emergent behaviors reminiscent of starling murmurations, with increased motion complexity in response to thermal "predators" (high-temperature regions).

5. **Robustness**: The system maintains performance across various simulated wildfire scenarios, including complex terrain and rapidly changing conditions.

### Expected Results and Comparisons

For this course project, I expect to achieve:

1. A trained GNN policy that successfully imitates the expert controller, with validation loss within 10% of the training loss, demonstrating good generalization.

2. Quantitative metrics showing that the learned policy maintains swarm cohesion (measured by graph connectivity) while responding appropriately to thermal gradients.

3. A perceptually-accurate temperature-to-color mapping with MAE ≤ 8°F (slightly higher than the ideal target but still functionally useful).

4. Comparative analysis between the GNN-based policy and alternative approaches (e.g., centralized MLP, reactive controllers) across key metrics:
   - Computational efficiency (inference time)
   - Scalability with increasing swarm size
   - Robustness to agent failures
   - Legibility scores

5. Visualization of the swarm's behavior in at least three distinct wildfire scenarios, demonstrating its adaptability.

### Risks and Mitigation Strategies

Several risks could impact the achievement of these results:

1. **Data Quality and Quantity**: Synthetic data may not capture the full complexity of real-world wildfires.
   *Mitigation*: Use domain randomization and incorporate available real-world data to improve robustness.

2. **Imitation Learning Limitations**: The learned policy might not generalize beyond the distribution of expert demonstrations.
   *Mitigation*: Generate a diverse set of demonstrations across varying conditions; prepare for potential future reinforcement learning fine-tuning.

3. **Computational Constraints**: GNN training can be computationally intensive, especially with large graphs.
   *Mitigation*: Implement efficient batching strategies; use smaller swarm sizes for initial development.

4. **Metric Definition Challenges**: Quantifying "legibility" is inherently subjective.
   *Mitigation*: Combine multiple metrics (SSIM, user studies in future phases) to triangulate effectiveness.

If I encounter significant obstacles, I will prioritize the core supervised learning components (imitation learning and color mapping) while potentially simplifying the environmental complexity or reducing the swarm size to ensure meaningful progress within the course timeframe.

## Relationship to My Learning Journey

This project represents an exciting opportunity to apply and extend my supervised machine learning knowledge in a domain that genuinely matters. I've already derived significant satisfaction from learning more about the MLOps side of preparing supervised machine learning models through the Pipeline exercises we've done in class. I'm using Thermur as an opportunity to extend further into modern ML engineering practices (hydra-zen, poetry, Pydantic) that enhance reproducibility and maintainability.

Throughout my research for this project, I've used tools like Google Gemini to help navigate academic papers and find relevant references. NotebookLM has been invaluable for dictating these sources in a podcast-like format, allowing me to digest material during commutes and exercise. These tools have helped me organize information contextually, making it easier to connect related concepts across different papers.

While the full vision of Thermur spans multiple courses and techniques, the supervised learning foundation we're establishing now is critical to its success. This initial phase will not only deliver tangible results in its own right but will also provide the groundwork for more advanced approaches in future classes.

## References

[5] Reynolds, Craig W. 1987. "Flocks, Herds and Schools: A Distributed Behavioral Model." *ACM SIGGRAPH Computer Graphics* 21 (4): 25–34. https://doi.org/10.1145/37402.37406

[6] Häusermann, D., et al. 2023. "FireDrone: Multi-Environment Thermally Agnostic Aerial Robot." *Advanced Intelligent Systems* 5 (23): 2300101. https://doi.org/10.1002/aisy.202300101

[7] Bialek, William, Andrea Cavagna, Irene Giardina, Thierry Mora, Edmondo Silvestri, Massimiliano Viale, and Aleksandr M. Walczak. 2012. "Statistical Mechanics for Natural Flocks of Birds." *Proceedings of the National Academy of Sciences* 109 (13): 4786–91. https://doi.org/10.1073/pnas.1118633109

[8] Ishii, Hiroshi, and Brygg Ullmer. 1997. "Tangible Bits: Towards Seamless Interfaces between People, Bits and Atoms." *CHI '97 Proceedings*, 234–41. https://doi.org/10.1145/258549.258715

[9] Wang, Li, Magnus Egerstedt, and Aaron D. Ames. 2017. "Safety Barrier Certificates for Collision-Free Multirobot Systems." *IEEE Transactions on Robotics* 33 (3): 661–74. https://doi.org/10.1109/TRO.2017.2659727

[11] Ballerini, M. et al. 2008. "Interaction Ruling Animal Collective Behavior Depends on Topological Rather than Metric Distance." *PNAS* 105 (4): 1232–37. https://doi.org/10.1073/pnas.0711437105

[13] Cavagna, Andrea, Alessio Cimarelli, Irene Giardina, Giorgio Parisi, Raffaele Santagati, Fabio Stefanini, and Massimiliano Viale. 2010. "Scale-Free Correlations in Starling Flocks." *PNAS* 107 (26): 11865–70. https://doi.org/10.1073/pnas.1005766107

[14] Ames, Aaron D., Xiangru Xu, Jessy W. Grizzle, and Paulo Tabuada. 2016. "Control Barrier Function-Based Quadratic Programs for Safety-Critical Systems." *IEEE Transactions on Automatic Control* 62 (8): 3861–76. https://doi.org/10.1109/TAC.2016.2638961

[17] Gama, Fernando, Ekaterina Tolstaya, and Alejandro Ribeiro. 2021. "Graph Neural Networks for Decentralized Controllers." *ICASSP 2021 — 2021 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*: 5260–5264. https://doi.org/10.1109/ICASSP39728.2021.9414563

[20] Coen, Janice L., et al. 2013. "WRF-Fire: Coupled Weather–Wildland Fire Modeling with the Weather Research and Forecasting Model." *Journal of Applied Meteorology and Climatology* 52 (1): 16–38. https://doi.org/10.1175/JAMC-D-12-023.1

[21] Rothermel, Richard C. 1972. "A Mathematical Model for Predicting Fire Spread in Wildland Fuels." *USDA Forest Service Research Paper INT-115*.

[22] Moisseeva, Nadejda. 2020. *A Numerical Perspective on Wildfire Plume-Rise Dynamics*. (T). University of British Columbia. https://doi.org/10.14288/1.0395299

[23] Dawson, Cian B., Christopher Holmquist‑Johnson, and Martin A. Briggs. 2018. "Thermal Infrared and Photogrammetric Data Collected by Small Unoccupied Aircraft System for Hydrogeologic Analysis of Oh‑be‑joyful Creek, Gunnison National Forest, Colorado, August 2017." *U.S. Geological Survey Data Release*. https://doi.org/10.5066/1P931G95D
