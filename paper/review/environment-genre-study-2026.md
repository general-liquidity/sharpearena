# The Agent Environment / Simulator Paper as a Genre: Structural Study and Mapping to SharpeArena

Date: 2026-08-25
Scope: read-only study. Corpus of 27 sources fetched and read in full text this session.

Method note: `WebSearch` was used for two discovery queries only. All structural claims below come
from `WebFetch` against full-text renderings (`ar5iv.labs.arxiv.org/html/`, `arxiv.org/html/`,
`jasss.org`, `minari.farama.org`). arXiv `/abs/` landing pages return only abstracts and were
discarded; arXiv `/pdf/` returns undecodable binary through this tool and was also discarded.
Where a full-text render was unavailable the row is marked accordingly.

---

## Part 1. The corpus

### 1.1 Classic RL environments

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| Gymnasium: A Standard Interface for RL Environments | Farama / arXiv 2024 | https://arxiv.org/html/2407.17032v1 | 1 Introduction, 2 Design Decisions, 3 Environment Specification, 4 Built-in Environments, 5 Related Work, 6 Conclusion, References | ~4 pp, no appendix. Extremely short. The whole paper is API plus design rationale. |
| Machado et al., Revisiting the Arcade Learning Environment | JAIR 2018 | https://ar5iv.labs.arxiv.org/html/1709.06009 | 1 Introduction, 2 Background (2.1 Setting, 2.2 Control in the ALE), 3 Divergent Evaluation Methodologies in the ALE, 4 Summarizing Learning Performance (4.1 Common Performance Measures, 4.2 Proposal: Performance During Training), 5 Determinism and Stochasticity in the ALE (5.1 The Brute, 5.2 Sticky Actions, 5.3 Alternative Forms of Stochasticity), 6 Benchmark Results in the ALE (6.1 Experimental Method, 6.2 Benchmark Results), 7 Open Problems and the Current State-of-the-Art, 8 Conclusion | Journal length. App. A The Brute (A.1 Determinism and starting configurations), B Experimental Setup, C Complete Benchmark Results, D Number of Game Modes and Difficulties. |
| Procgen Benchmark (Cobbe et al.) | ICML 2020 | https://ar5iv.labs.arxiv.org/html/1912.01588 | 1 Introduction, 2 Procgen Benchmark (2.1 Environment Desiderata, 2.2 Experimental Protocols, 2.3 Hyperparameter Selection), 3 Generalization Experiments (3.1 Level Requirements, 3.2 An Ablation with Deterministic Levels, 3.3 500 Level Generalization), 4 Scaling Model Size, 5 Comparing Algorithms, 6 Related Work, 7 Conclusion | ~27 pp with appendices. App. A Environment Descriptions (A.1-A.16), B Core Capabilities in RL, C Normalization Constants, D Hyperparameters, E Test Performance for All Training Sets, F ALE Performance, G Training Curves by Architecture, H Frame Stack vs LSTM, I Easy Difficulty Baseline Results. |
| Crafter (Hafner) | ICLR 2022 | https://ar5iv.labs.arxiv.org/html/2109.06780 | 1 Introduction, 2 Related Work, 3 Crafter Benchmark (3.1 Game Mechanics, 3.2 Environment Interface, 3.3 Evaluation Protocol, 3.4 Research Challenges), 4 Experiments (4.1 Benchmark with Rewards, 4.2 Unsupervised Benchmark, 4.3 Emergent Behaviors, 4.4 Human Experts Dataset), 5 Discussion | ~15 pp. App. A-L incl. Success Rates with/without Rewards, Human Expert Rates, Textures, Action Space, Achievement Curves. |
| NetHack Learning Environment (Kuttler et al.) | NeurIPS 2020 | https://ar5iv.labs.arxiv.org/html/2006.13760 | 1 Introduction, 2 NetHack: a Frontier for RL Research (2.1 NetHack, 2.2 The NetHack Learning Environment, 2.3 Tasks, 2.4 Evaluation Protocol, 2.5 Baseline Models), 3 Experiments and Results, 4 Related Work, 5 Conclusion and Future Work, 6 Broader Impact | ~25 pp. App. A Further Details on NetHack, **B Observation Space**, **C Action Space**, **D Environment Speed Comparison**, E Task Details, F Baseline CNN Details, G RND Details, H Dashboard, I NetHack Bots, J Viewing Agent Videos. |

### 1.2 Multi-agent and social

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| PettingZoo (Terry et al.) | NeurIPS 2021 | https://ar5iv.labs.arxiv.org/html/2009.14471 | 1 Introduction, 2 Background and Related Works (2.1 POSGs and RLlib, 2.2 OpenSpiel and Extensive Form Games), 3 PettingZoo Design Goals (3.1 Be like Gym, 3.2 Be a Universal API), 4 Case Studies of Problems With The POSG Model in MARL (4.1, 4.2), 5 The Agent Environment Cycle Games Model, **6 API Design (6.1 Basic API, 6.2 agent_iter, 6.3 last, 6.4 Additional API Features, 6.5 Environment Creation and the Parallel API)**, 7 Default Environments, **8 Adoption**, 9 Conclusion | ~20 pp. App. A Additional Case Study Information, B Default Environments, C Formal Definitions, D Omitted Proofs. |
| Melting Pot (Leibo et al.) | ICML 2021 | https://ar5iv.labs.arxiv.org/html/2107.06857 | 1 Introduction, 2 What does Melting Pot evaluate?, 3 Related work, **4 The Melting Pot protocol (4.1 Definitions, 4.2 Testing, 4.3 Training, 4.4 Secondary evaluation metrics)**, 5 Description of the substrates, **6 Extending Melting Pot**, 7 Experiments, 8 Conclusion | 33 pp per arXiv comments. App. A Secondary evaluation metrics, B Substrate details, C Agent architecture details, D Training setup, E Scenario details, F Raw performance scores. |

### 1.3 Web, OS and tool sandboxes

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| WebArena (Zhou et al.) | ICLR 2024 | https://ar5iv.labs.arxiv.org/html/2307.13854 | 1 Introduction, **2 WebArena: Websites as an Environment for Autonomous Agents (2.1 Controlling Agents through High-level Natural Language, 2.2 Website Selection, 2.3 Observation Space, 2.4 Action Space)**, 3 Benchmark Suite of Web-based Tasks (3.1 Intent Collection, 3.2 Evaluation Annotation), 4 Baseline Web Agents, 5 Results (5.1 Analysis), 6 Related Work, 7 Conclusion | ~16 pp. App. A.1 Website Implementation, **A.2 Environment Delivery and Reset**, A.3 User Roles Simulation, A.4 Intent Distribution, **A.5 Human Performance**, A.6 Experiment Configurations, A.7-A.8 fuzzy_match, A.9 Prompts, A.10 Additional Error Analysis. No standalone Limitations section. |
| OSWorld (Xie et al.) | NeurIPS 2024 D&B | https://arxiv.org/html/2404.07972v2 | 1 Introduction, **2 OSWorld Environment (2.1 Task Definition, 2.2 Real Computer Environment Infrastructure incl. 2.2.1 Overview, 2.2.2 Initial Task Environment Setup, 2.2.3 Execution-Based Evaluation, 2.3 Observation Space, 2.4 Action Space)**, 3 OSWorld Benchmark (3.1 OS and Software Environments, 3.2 Tasks, 3.3 Data Statistics, **3.4 Human Performance**), 4 Benchmarking LLM and VLM Agent Baselines (4.1, 4.2 Results), 5 Analysis (5.1-5.4), 6 Related Work, 7 Conclusion and Future Work | App. A Details of OSWorld Environment, B Details of OSWorld Benchmark, C Details of Baseline Methods, D Examples of Qualitative Analysis. |
| tau-bench (Yao et al.) | 2024 | https://ar5iv.labs.arxiv.org/html/2406.12045 | 1 Introduction, 2 Related Work, **3 tau-bench: A benchmark for Tool-Agent-User Interaction (Databases and APIs, Domain policy, User simulation, Task instances, Reward, Pass^k metric)**, 4 Benchmark Construction (4.1 Domains, 4.2 Key Characteristics), 5 Experiments (5.1 Main results, 5.2 Research challenge analysis), 6 Discussion | ~15-20 pp. App. A Additional Results, B Benchmark Construction, C Retail Examples, D Airline Examples. |
| SWE-Gym (Pan et al.) | 2024 | https://arxiv.org/html/2412.21139v2 | 1 Introduction, 2 Related Work, **3 SWE-Gym Environment (3.1 Dataset Construction, 3.2 SWE-Gym Lite, 3.3 Dataset Statistics)**, 4 Training LMs as Agents with SWE-Gym (4.1 Setting, 4.2, 4.3), 5 Scaling Agent Performance with SWE-Gym (5.1 Inference-Time Scaling with Verifiers, 5.2 Training-Time Scaling with Data), **6 Conclusions, Limitations, and Future Work** | ~16 pp. |

### 1.4 Embodied and robotics simulators

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| Habitat (Savva et al.) | ICCV 2019 | https://ar5iv.labs.arxiv.org/html/1904.01201 | 1 Introduction, 2 Related Work, **3 Habitat Platform (Design requirements, Design overview, Generic 3D dataset API using scene graphs, Rendering engine, Performance, Efficient GPU throughput)**, 4 PointGoal Navigation at Scale (Task definition, Agent embodiment and action space, Collision dynamics, Goal specification, Sensory input, Episode specification, Evaluation, Episode dataset preparation, Baselines, Training procedure), 5 Results and Findings (Learning vs SLAM, Generalization across datasets), **6 Habitat Challenge (Challenge infrastructure)**, 7 Future Work | ~20 pp. App. A Habitat Platform Details, B Additional Dataset Statistics, C Additional Experimental Results, D Gibson Dataset Curation, **E Reproducing Experimental Results**, F Example Navigation Episodes. |
| ALFWorld (Shridhar et al.) | ICLR 2021 | https://ar5iv.labs.arxiv.org/html/2010.03768 | 1 Introduction, **2 Aligning ALFRED and TextWorld**, 3 Introducing BUTLER (3.1 Brain, 3.2 Vision, 3.3 Body), 4 Experiments (4.1 Importance of Interactive Language, 4.2 Transferring to Embodied Tasks, 4.3 Generalizing to Human-Annotated Goals, 4.4 To Pretrain or not to Pretrain), **5 Ablations (5.1 Generalization within TextWorld, 5.2 Unimodal Baselines, 5.3 Model Ablations)**, 6 Related Work, 7 Conclusion | ~20 pp plus App. A-K. |
| Isaac Gym (Makoviychuk et al.) | NeurIPS 2021 D&B | https://ar5iv.labs.arxiv.org/html/2108.10470 | 1 Introduction, 2 Background (incl. **2.3 Tensor API: 2.3.1 Python Interface, 2.3.2 Physics State Tensors, 2.3.3 Physics Control Tensors**), 3 Physics Simulation, 4 Environments, **5 Characterising Simulation Performance**, **6 Characterising Environment Performance**, 7 Summary, 8 Acknowledgements, References, Appendix A | ~40 pp. Two full sections devoted to throughput and cost. |
| CARLA (Dosovitskiy et al.) | CoRL 2017 | https://ar5iv.labs.arxiv.org/html/1711.03938 | 1 Introduction, **2 Simulation Engine**, 3 Autonomous Driving, 4 Experiments, 5 Results, 6 Conclusion, 7 Acknowledgements, References, Supplementary Material | ~9 pp main. Compact: environment plus protocol plus three-approach comparison. |

### 1.5 Market and economic simulators, and simulator-validation methodology

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| ABIDES (Byrd, Hybinette, Balch) | SIGSIM-PADS 2020 | https://ar5iv.labs.arxiv.org/html/1904.12066 | 1 Background, **2 Important Questions Simulation Can Help Us Address**, 3 ABIDES Architecture (3.1 Kernel functions, 3.2 Kernel Lifecycle Phases, 3.3 Agent Class, 3.4 Exchange Agent Class, 3.5 Order Book, 3.6 Trading Agent Class), 4 ABIDES Implementation (4.1 Example: A Momentum Trading Agent), **5 Case Study: Background Agents**, **6 Case Study: Market Impact**, 7 Conclusion and Future Challenges, **8 Open Source Access and License**, 9 Acknowledgements | ~10 pp, no appendix. The two case studies are validation studies, structurally separated from architecture. |
| ABIDES-Gym (Amrouni et al.) | ICAIF 2021 | https://ar5iv.labs.arxiv.org/html/2110.14771 | 1 Introduction, 2 ABIDES (2.1 ABIDES-Core, 2.2 ABIDES-Markets), 3 ABIDES-Gym (3.1 Motivation, 3.2 Approach, 3.3 Key idea: interruptible simulation kernel, 3.4 Gym-Core environment, 3.5 sub-environments, 3.6 More details), **4 ABIDES-Gym application to finance: introducing two market environments (4.1 Daily Investor Environment, 4.2 Algorithmic Execution Environment)**, 5 Experimental Example: Training an RL agent, 6 Related Work, 7 Acknowledgments | ~11 pp. No appendix, no Limitations section. |
| FinRL-Meta (Liu et al.) | NeurIPS 2022 D&B | https://ar5iv.labs.arxiv.org/html/2211.03107 | 1 Introduction, 2 Related Works, 3 Financial RL and FinRL-Meta Framework (3.1 Challenges, 3.2 Overview), **4 Financial Big Data and DataOps for Dynamic Datasets (4.1 Data Layer, 4.2 Environment Layer, 4.3 Advantages)**, 5 Tutorials and Benchmarks (5.1 Metrics and Baselines, 5.2 Tutorials and Demos, 5.3 Reproducing Prior Papers as Benchmarks), 6 Conclusion | ~30 pp. App. A Terminology, B DataOps Paradigm, C FinRL, **D Dataset Documentation and Usages**, E Open Source and DAO/DeFi/Web3, F Data Privacy, **G Accessibility, Accountability, Maintenance and Rights**. Appendices D and G are a datasheet in all but name. |
| mbt_gym (Jerome et al.) | ICAIF 2022 | https://ar5iv.labs.arxiv.org/html/2209.07823 | Abstract, Keywords, 1 Introduction, 2 Design of the Module, remainder truncated in render | Short workshop-scale paper. Render partial; structure read only through Section 2. |
| Platt and Gebbie, Can Agent-Based Models Probe Market Microstructure? | Physica A 2017 | https://ar5iv.labs.arxiv.org/html/1611.08510 | 1 Introduction, 2 The Preis et al. Model, **3 Calibration Experiment Design**, 4 Calibration Results, 5 Realistic Order Matching Procedures and Price Dynamics, **6 Relevant Caveats**, 7 Conclusion, 8 Acknowledgements | ~12-13 pp. The single most load-bearing methodology source in this corpus for SharpeArena. |
| Limit Order Book Simulations: A Review | arXiv 2024 | https://arxiv.org/html/2402.17359v1 | 1 Limit Order Books, **2 Stylized Facts**, 3 Point Processes Models, 4 Agent Based Models, 5 Deep Learning Based Models, 6 SDE Based Models, **7 Responsiveness to Trades: Market Impact**, **8 Comparative Study**, 9 Conclusion and Future Work, **10 Disclaimer** | ~40 pp. Table 1 catalogues 40+ studies by tested facts and methodology. |
| TRADES (diffusion LOB simulation) | 2025 | https://arxiv.org/html/2502.07071v1 | 1 Introduction, 2 Background, 3 Related Works, 4 Transformer-based Denoising Diffusion Engine for LOB Simulations, 5 TRADES for Market Simulation, **6 DeepMarket framework with synthetic dataset**, 7 Experiments, 8 Conclusion, App. A-F | ~16 pp main. |
| Deep Learning Meets Queue-Reactive (Bodor and Carlier) | 2025 | https://arxiv.org/html/2501.08822v1 | 1 Introduction and Motivation, 2 Related Work, 3 The Queue-Reactive Model and Its Extensions, 4 Multidimensional Deep Queue-Reactive Model (incl. 4.5 Market Impact, 4.6 mid-price prediction, 4.7 Other Stylized Facts), 5 Comparison with Recent Deep Learning Approaches, 6 Conclusion | ~15-20 pp. A negative exemplar: validation and downstream findings are interleaved at the same structural level. |
| High-Frequency Financial Market Simulation and Flash Crash Scenarios | JASSS 27(2) 2024 | https://www.jasss.org/27/2/8.html | 1 Introduction, 2 Background and Related Work, 3 Model Structure, **4 Model Calibration and Validation**, 5 2010 Flash Crash Scenarios, 6 Mini Flash Crash Scenarios | Journal length. Section 4 is a dedicated validation section, structurally before the findings sections 5 and 6. |

### 1.6 Offline data and reproducibility infrastructure

| Paper | Venue / Year | URL | Top-level sections, in order | Length / appendix |
|---|---|---|---|---|
| D4RL (Fu et al.) | 2020 | https://ar5iv.labs.arxiv.org/html/2004.07219 | 1 Introduction, 2 Related Work, 3 Background, **4 Task Design Factors**, 5 Tasks and Datasets (Maze2D, AntMaze, Gym-MuJoCo, Adroit, FrankaKitchen, Flow, Offline CARLA, **Evaluation protocol**), 6 Benchmarking Prior Methods, 7 Discussion | ~16 pp main. App. A Task Properties, B Results by Domain, C Task and Datasets, **D Training and Evaluation Task Split**, E Experiment Details, **F Assessing the Feasibility of Hard Tasks**, G Maze Domain Trajectories. |
| Minari dataset standard | Farama docs 2024 | https://minari.farama.org/main/content/dataset_standards/ | Documentation, not a paper. Structure: Directory Structure, Required Metadata, Episode Metadata, Supported Spaces, Episode Data Structure. | Required attributes: `dataset_id`, `total_episodes`, `total_steps`. Optional: author, algorithm name, **environment specifications, code repository links, Minari version**. Naming: `(namespace/)(env_name/)dataset_name(-v(version))`. |
| Datasheets for Datasets (Gebru et al.) | CACM 2021 | https://ar5iv.labs.arxiv.org/html/1803.09010 | 1 Introduction, 2 Development Process, **3 Questions and Workflow**, 4 Impact and Challenges. Template categories in order: Motivation, Composition, Collection Process, Preprocessing/cleaning/labeling, **Uses**, Distribution, Maintenance | The canonical environment-card analogue. |

Corpus total: 27 sources, 24 of them papers with a recoverable full section list.

---

## Part 2. The genre grammar

### 2.1 The modal section sequence

Across the 24 papers with recoverable structure, one sequence dominates. I write it with the
canonical name and the count of corpus papers that have a recognizable instance.

1. **Introduction** (24/24). Always states the gap, always states contributions as an enumerated list in the newer papers.
2. **Related work** (21/24). Placed second in the ML-venue papers (Procgen, OSWorld, tau-bench, Crafter, NLE, D4RL, ALFWorld, SWE-Gym, FinRL-Meta, PettingZoo, Habitat). Placed late, before the conclusion, in only a minority. Note: placing related work early is the modal choice, and SharpeArena places it seventh of nine.
3. **The environment / platform** (24/24). Universally the structural centerpiece. Almost always subdivided into: design requirements or desiderata; architecture; **observation space**; **action space**; task definition; and, for sandbox papers, environment setup and reset.
4. **The task suite / benchmark** (20/24). A separate section from the environment in WebArena (2 vs 3), OSWorld (2 vs 3), Procgen (env desiderata 2.1 vs experimental protocols 2.2), D4RL (4 Task Design Factors vs 5 Tasks and Datasets), Melting Pot (4 protocol vs 5 substrates). The split is the norm: the machine and the content it runs are different objects.
5. **Evaluation protocol** (18/24). Sometimes a subsection (Crafter 3.3, NLE 2.4, D4RL "Evaluation protocol" inside 5, Procgen 2.2), sometimes a whole section (Melting Pot 4 "The Melting Pot protocol"). In every case it is stated before any result is reported and is written prescriptively for future users, not descriptively about what the authors did.
6. **Validation of the platform** (12/24, and this is the interesting split, see 2.3).
7. **Baselines and results** (24/24).
8. **Analysis** (9/24, a distinct section in OSWorld 5, WebArena 5.1, Melting Pot 7).
9. **Limitations / Discussion** (15/24 as a titled unit; SWE-Gym folds it into the conclusion title "Conclusions, Limitations, and Future Work"; WebArena and ABIDES-Gym have none).
10. **Conclusion**, then **appendices**.

**Variants.**

- *The thin-API variant* (Gymnasium, PettingZoo, mbt_gym, Minari docs): 4 to 20 pages, no validation section, because the contribution is the interface itself. Gymnasium is four pages and spends section 3 entirely on Environment Specification. PettingZoo devotes section 6 to API Design with five subsections and adds a section 8 titled simply **Adoption**.
- *The methodology-revisit variant* (Machado et al.): the environment already exists; the paper's contribution is the protocol. Sections 3 through 5 are entirely about how the environment has been evaluated wrongly and how to evaluate it correctly. This is the closest genre-neighbor to SharpeArena's actual argument.
- *The simulator-with-validation variant* (ABIDES, JASSS flash-crash, CARLA, Isaac Gym): a dedicated validation or characterisation block sits between the architecture and the findings.
- *The sandbox-benchmark variant* (WebArena, OSWorld, tau-bench, SWE-Gym): environment section, benchmark section, baselines, analysis. Realism is argued by provenance rather than by statistics.

### 2.2 Where each genre-standard element lives

**API and interface specification.** Always a numbered, named unit, and almost always early.
Gymnasium: "3 Environment Specification". PettingZoo: "6 API Design" with 5 subsections. WebArena:
"2.3 Observation Space" and "2.4 Action Space" as sibling subsections. OSWorld: identical pairing at
"2.3 Observation Space", "2.4 Action Space". NLE puts them in **Appendix B (Observation Space)** and
**Appendix C (Action Space)** while keeping the interface prose in 2.2. Isaac Gym: "2.3 Tensor API"
with three subsections plus two numbered tables of state and control tensors. The genre rule is:
the observation and action space get their own headed unit with a name containing the words
"observation space" and "action space". Twelve of the fourteen corpus papers with a programmatic
interface do this. Gymnasium's framing of the minimal boundary:

> "The only requirement is that the environment subclass's gym.Env and two core functions (Env.reset and Env.step) are implemented."

**Determinism and reproducibility claims.** Three distinct placements.

(a) *A design property stated inside the API section*. Gymnasium: "Gymnasium automatically handles
seeding the random number generator and maintaining its state behind the scenes. The user can simply
specify the seed through env.reset(seed=seed)", plus "Environment versioning - Creating an environment
requires the specification of the version created, e.g., gym.make("CarRacing-v2")."

(b) *A property of the delivery mechanism*. WebArena: "We achieve reproducibility by making the
environment standalone, without relying on live websites" and "The Docker images are fully
self-contained with all the code of the website, database, as well as any other software
dependencies." OSWorld's parallel: "We provide convenient APIs to configure initial conditions and
world settings."

(c) *A hazard to be designed against*. Machado et al. This is the unique and important case. ALE is
deterministic, and the paper treats that as a threat:

> "The deterministic ALE does not effectively distinguish between agents that learn robust, closed-loop policies from brittle memorization-based agents."

> "The agent always starts at the same initial state, and a given sequence of actions always leads to the same outcome... Such an approach is not likely to be successful beyond the ALE."

Their remedy is sticky actions, and their recommendation is prescriptive: "Our recommendation is to
use sticky actions, implemented in the latest version of the ALE." Procgen makes the same move
structurally by dedicating **3.2 "An Ablation with Deterministic Levels"** to it.

**Task or scenario generation protocol.** Procgen is the reference implementation of the genre move,
and it is a protocol stated to future users:

> "When evaluating generalization, we train on a finite set of levels and test on full distribution... we recommend training on 500 levels from each environment and testing on held out levels... At test time, we measure agents' zero-shot performance averaged over unseen levels."

D4RL states the split as a table plus a rule: "Hyperparameters should be tuned on tasks listed...in
'Training' column, evaluated without tuning on 'Evaluation' tasks." Melting Pot separates 4.2 Testing
from 4.3 Training and says "Melting Pot focuses only on test-time evaluation, and is agnostic to the
method of training."

**Baselines.** Universal. Two flavors: learned baselines (Procgen, Crafter, NLE, Habitat, D4RL,
OSWorld, WebArena) and analytical or oracle baselines. mbt_gym is explicit about the second: "We
provide optimal baseline agents so that one can benchmark the performance of RL algorithms."
ALFWorld introduces "BUTLER-Oracle... oracle variant of BUTLER which uses perfect state-estimation,
object-detection, and navigation" precisely to separate environment misalignment from component
failure. A third flavor, present in six papers, is the **human baseline**: OSWorld 3.4 Human
Performance, WebArena Appendix A.5 Human Performance, Crafter's human expert dataset (4.4), which the
paper uses as its central difficulty calibration (human experts score 50.5% against top agents at 10%).

**Validation-of-realism argument.** See 2.3, the load-bearing subsection.

**Benchmarking protocol for future users.** The strongest papers write a section that a reader who
will use the environment can follow without reading the results. Melting Pot section 4 is titled
"The Melting Pot protocol". Machado section 4.2 is titled "Proposal: Performance During Training" and
reads as instruction: "At the end of training (and ideally at other points as well) report the average
performance of the last k episodes. This protocol does not use the explicit evaluation phase, thus
requiring an agent to perform well while it is learning." Crafter 3.3 defines the score in one
sentence: "The Crafter score...is computed as the geometric mean across achievements of their success
rates." D4RL normalizes: "Normalized score = 100 x (score - random score) / (expert score - random
score)."

**Limitations.** The genre has settled on a small set of standard admissions: coverage of the domain
(Melting Pot, "The largest category of extant research that we left unrepresented is
communication/language"), unmodeled real-world properties (D4RL, "There are several important
properties exhibited in some real-world applications of RL that are not explored in-depth in our
benchmark...stochasticity...large action spaces"), diversity of the instance set (SWE-Gym, "SWE-Gym,
while effective, is limited by its environment diversity"), and metric validity (NLE, "using
NetHack's in-game score is likely insufficient for solving the game"). Only a minority put
Limitations in a numbered top-level section; more often it is a Discussion subsection or a run-on
paragraph at the end.

### 2.3 How simulator papers argue for validity when the world is synthetic

This is the question SharpeArena most needs answered, so I separate the accepted moves and then the
accepted ways of scoping a failure.

**The five accepted validity moves, ranked by how much weight the genre lets them carry.**

1. **Sim-to-real transfer.** The strongest available move, and the only one that terminates the
argument. Habitat: "We aim to support a complementary research program: training embodied agents in
rich realistic simulators and then transferring the learned skills to reality." Isaac Gym reports a
number and does not round it up: "We also transfer results from simulation to the real world and note
that our mean success rate in the real world is 55%." Isaac Gym has **no explicit physics
validation section at all**; validation occurs entirely through successful transfer. When transfer is
unavailable, as it is for a financial environment with no live venue, the genre does not accept a
substitute of equal strength, and the honest papers say so.

2. **Provenance of the artifact.** The dominant move in the sandbox family, and it is a construction
argument rather than a statistical one. WebArena: "We achieve realism by using open-source libraries
that underlie many in-use sites from several popular categories and importing data to our environment
from their real-world counterparts." OSWorld: real OS, real apps, execution-based reward. This move is
unavailable to a purely procedural generator, which is exactly SharpeArena's position.

3. **Calibration to stylized facts.** The dominant move in market simulation, and the corpus is
unusually clear that the field has downgraded it. ABIDES is careful and conditional: "Ideally, we will
see a price history that closely resembles the day in history, with similar statistical properties."
The LOB review catalogues 40+ studies by which facts they test and how, and observes: "The technique
for testing against this stylized fact is usually a qualitative test where the two distributions
(empirical data and simulations) are plotted against each other." The JASSS flash-crash paper is the
most rigorous instance: it fixes a moment set (Hill tail index, volatility, return autocorrelation,
squared-return autocorrelation), defines a moment-specific p-value against bootstrapped empirical
values, and a moment coverage ratio against 95% confidence intervals, and states a pass rule:
"the moment-specific p-values are greater than 0.05".

   And the field's own critique of this move is explicit and citable. Platt and Gebbie:

   > "A key concern is the fact that a very large number of models with very different design philosophies are able to replicate the stylized facts of return time series equally well"

   > "Traditional stylized fact-centric validation seems unable to detect these potential problems, suggesting that such methods of validation are simply not sufficient"

   and their positive recommendation:

   > "calibration and stylized-fact replication should both be considered when developing such ABMs, in order to ensure a replication of reasonable model behavior"

   This is the single most useful citation available to SharpeArena, because it converts a failed
   stylized-facts gate from an embarrassment into a demonstration that the gate has teeth. The
   standard failure mode of the market-simulator genre is a model that passes the facts and is still
   wrong; SharpeArena has the mirror image, a model that fails a stated gate and says so.

4. **Responsiveness and impact validation, distinct from unconditional statistics.** The LOB review
gives this its own top-level section, **7 Responsiveness to Trades: Market Impact**, separate from
**2 Stylized Facts**. Bodor and Carlier likewise validate impact (4.5) separately from other
stylized facts (4.7). ABIDES devotes an entire numbered section, **6 Case Study: Market Impact**, to
it and reports a correlation of r = -0.31. The genre lesson is that unconditional distributional
realism and conditional response realism are two different validation axes and get two different
headings. SharpeArena already has the second axis (F5 manipulation boundary, F6 markout, the concave
ablation) but does not name it as a validation axis.

5. **Ablation of model components, sensitivity, and positive controls.** Procgen 3.2 ablates
deterministic levels. ALFWorld isolates domain misalignment with an oracle. TRADES reports a
predeclared predictive score. The LOB review recommends: "robustness of the model's calibrated
parameters should be checked against initial conditions, market volatility and other quantities."
SharpeArena's positive control at beta = 0.5, and its normalized-flow concave ablation, are already
textbook instances of this move; they are simply not labelled as validation.

**How the genre scopes a failed or partial validation.** Four moves, all present in the corpus.

- **State the failure as the result and keep the instrument.** CARLA does this in the open: "None of
the methods performs perfectly even on the simplest task of driving straight on an empty street in the
training conditions", and elsewhere reports generalization dropping over 50% on unseen towns. The
paper does not weaken the benchmark to make the numbers nicer; the failure is the finding.
- **Report the specific unmatched fact, name the mechanism, do not remediate.** TRADES: on one
instrument the volume-volatility correlation was "slightly negative" in real data while the model
generated positive, "because TRADES observes a positive correlation during training". One sentence,
one mechanism, no defensiveness.
- **Downgrade the claim's domain rather than the evidence.** TRADES again: "we want to be clear that
it technically enables evaluating trading strategies, but it does not assure any profitability in a
real market scenario." Melting Pot: "The largest category of extant research that we left
unrepresented is communication/language."
- **Reframe the failure as evidence about the diagnostic.** Machado's whole paper is this move: the
ALE's determinism is a finding about the evaluation methodology, and the contribution is the
corrected protocol. Platt and Gebbie is the same shape for market ABMs, and their section 6 is
literally titled **"Relevant Caveats"** as a top-level numbered section.

The move the genre does **not** accept: burying a failed check inside a results table and continuing
to make claims conditioned on the simulator being realistic. Bodor and Carlier is the cautionary
example in this corpus, where no formal separation exists between validation and downstream
findings, so a reader cannot tell which results survive which check.

### 2.4 Research instrument, not product, and the "who should use this" question

The genre has a settled register for this. Three markers.

- **Explicit refusal to claim superiority.** Habitat: "We compare representative instances of these
families to gain insight into questions of scaling and generalization, and do not make any claims about
intrinsic superiority of one or the other." This sentence is the genre's signature move.
- **A stated user and use.** ABIDES: "ABIDES is intended to be a curated, collaborative open-source
project that provides researchers with tools that support the rapid prototyping and evaluation of
complex market agents", followed by an enumerated list of target uses (co-location benefits, market
impact models, AI cost-benefit analysis, explainability). Gymnasium: "Gymnasium keeps its focus
entirely on the environment side of RL research, abstracting away the aspect of agent design and
implementation."
- **Extension and governance as a section.** Melting Pot has a numbered section **6 Extending Melting
Pot** and argues the suite improves as the field does: "since the effectiveness of the bots in test
scenarios is itself advanced by improvements in the performance of learning systems, Melting Pot will
likewise improve over time." PettingZoo has section **8 Adoption**. ABIDES has section **8 Open Source
Access and License**. FinRL-Meta has appendices D (Dataset Documentation and Usages) and G
(Accessibility, Accountability, Maintenance and Rights), which is a datasheet in the Gebru sense.

The datasheet template's own wording is the checklist SharpeArena should answer, in order:
**Motivation, Composition, Collection Process, Preprocessing, Uses, Distribution, Maintenance**, with
the pivotal question being "Are there tasks for which the dataset should not be used? If so, please
provide a description."

### 2.5 Multi-surface projects and cross-runtime guarantees

Weakly served by the genre, which is an opportunity rather than a constraint.

- Habitat separates Habitat-Sim from Habitat-API and argues modularity: "Habitat-API allows for
alternative simulator backends to be used, beyond the Habitat-Sim module that we implemented. This
modularity has the advantage of allowing incorporation of existing simulator backends." It does not
claim numerical identity across backends.
- Isaac Gym's Tensor API section documents a C++/CUDA core surfaced as PyTorch tensors "with no
copying overhead", which is a performance guarantee, not an identity guarantee.
- Gymnasium's cross-version discipline is environment versioning in the ID: `gym.make("CarRacing-v2")`.
- Minari's standard is the closest analogue to a cross-surface provenance contract, and it is
metadata-based: required `dataset_id`, `total_episodes`, `total_steps`; optional but recommended
environment specifications, code repository links, and Minari version; and a naming scheme
`(namespace/)(env_name/)dataset_name(-v(version))` that carries the version in the identifier.
- SWE-Gym's answer is containerization at scale: "pre-built Docker images for each instance, totaling
6 TB".

**No paper in this corpus claims byte-identical outputs across independently compiled runtimes and
pins it with committed hashes.** That is SharpeArena's genuinely novel property, and the genre gives
it no template, which means it must be stated in a place a reader will find it.

### 2.6 Leaderboard and evaluation-protocol sections

Present as a named unit in 6 of 24: Habitat **6 Habitat Challenge** (with a "Challenge
infrastructure" subsection), Melting Pot **4 The Melting Pot protocol**, Crafter **3.3 Evaluation
Protocol**, NLE **2.4 Evaluation Protocol**, D4RL's "Evaluation protocol" plus **Appendix D Training
and Evaluation Task Split**, tau-bench's "Pass^k metric" subsection. Procgen's **2.2 Experimental
Protocols** is functionally the same. The consistent property is that the protocol section is written
in the imperative or recommending voice and is separable from the results.

---

## Part 3. Mapping the genre onto SharpeArena

SharpeArena as it stands: `main.tex` inputs 00-abstract, 01-introduction, 02-principles,
03-environment, 04-contract, 05-experiments, 06-related, 07-limitations, 08-reproducibility, then
appendix A-commands. Experiments is 307 lines and inputs eight fragment files
(arena-witness, calm-tails, concave, endogenous-adverse, manipulation-sweeps, positive-control,
predictability, sealed-seeds), producing fifteen subsections carrying ten findings.

### 3.1 Two-column mapping

| Genre-standard element (exemplar) | SharpeArena today | Verdict |
|---|---|---|
| Introduction with enumerated contributions (all) | Sec. 1, six enumerated contributions plus a "Scope and contribution boundary" paragraph | Present and strong. The scope-boundary paragraph is above genre norm. |
| Related work placed early (Procgen, OSWorld, D4RL, tau-bench, Habitat) | Sec. 7 of 9, after experiments | **Misplaced relative to the mode.** The four-axis competitive framing in 06-related is the paper's clearest statement of what is new, and it arrives after 36 KB of experiments. |
| Design desiderata / requirements (Procgen 2.1, Habitat "Design requirements", Crafter design goals) | Sec. 2 "Design principles", seven principles | Present, well done, unusually explicit. Genre-conformant and an asset. |
| Environment / platform architecture (all) | Sec. 3 "The environment", 7 subsections | Present. |
| **Observation space as a named unit** (WebArena 2.3, OSWorld 2.3, NLE App. B, Isaac Gym 2.3.2) | **Absent as a heading.** `MarketObservation` is described in prose inside 3.1 (leak-freedom) and 4 (contract) | **Missing.** The single most consistent genre convention in the corpus. |
| **Action space as a named unit** (WebArena 2.4, OSWorld 2.4, NLE App. C, ABIDES-Gym 4.1.x) | **Absent as a heading.** `Decision` appears in 4 as a JSON contract, with no enumeration of action variants, bounds, or weight semantics | **Missing.** |
| Task suite as a section separate from the machine (WebArena 3, OSWorld 3, D4RL 5, Melting Pot 5) | Sec. 3.5 "The task suite and the closed-form reference policy", a subsection | Present but **subordinated**. Five distinct task families (position trading, simplex allocation, VWAP/TWAP execution, endogenous shared-book, deterministic LOB) are compressed into one subsection that also carries the Avellaneda-Stoikov reference-policy exposition. |
| **Task taxonomy / instance statistics** (OSWorld 3.3, D4RL App. A, Procgen App. A, Melting Pot App. B) | **Absent.** No table enumerating tasks by family, mandate, tier, episode length, symbol count, action semantics | **Missing.** |
| Scenario generation protocol (Procgen 2.2, D4RL App. D, Melting Pot 4.2/4.3) | Sec. 3.4, canonical bands train [0,256), gap 10k, test [10256,10512), plus the 10^6 namespace | Present and genre-conformant. |
| **Benchmarking protocol for future users** (Melting Pot 4, Machado 4.2, Crafter 3.3, D4RL normalized score) | **Diffused.** The protocol exists in fragments across 3.4, 3.5, 4 and 8, and is written descriptively ("the fixed-policy F3 and witness producers instead document their own 16-seed subsets") rather than prescriptively | **Missing as a unit.** A reader who wants to submit an agent must reconstruct the protocol from five places. |
| Determinism as a design property (Gymnasium seeding, WebArena Docker) | Sec. 3.2, plus the second principle in 2, plus 8 | Present and far stronger than any corpus exemplar. |
| Determinism as a **hazard** (Machado 5, Procgen 3.2) | Handled, but in the predictability fragment inside Experiments and in Limitations | **Buried.** The Machado argument that determinism invites memorization is exactly SharpeArena's `sec:predictability` finding (16/16 seed recovery). It is genre-central and currently reads as a late probe. |
| API / interface specification (Gymnasium 3, PettingZoo 6, Isaac Gym 2.3) | Sec. 4 "The agent contract and its governance" | Present and above genre norm on governance. Below norm on shape: it specifies versioning and evolution rules but not the observation and action spaces. |
| Baselines, analytical (mbt_gym optimal agents, ALFWorld oracle) | Avellaneda-Stoikov closed-form reference in 3.5, oracle witness in the witness fragment | Present and strong. |
| Human baseline (OSWorld 3.4, WebArena A.5, Crafter 4.4) | Absent | Not applicable to this domain. Not a gap. |
| **Validation section distinct from findings** (JASSS 4, ABIDES 5 and 6, LOB review 2 vs 7) | **Absent.** F4 realism, F5 manipulation boundary, F6 markout, the concave ablation and the positive control are validation of the simulator, and they sit interleaved with F1 pipeline, F3 generalization, F7 failure taxonomy and F8 ecology, which are findings produced by the instrument | **Missing, and this is the highest-value structural change.** |
| Ablation of model components (Procgen 3.2, ALFWorld 5) | concave-fragment, positive-control-fragment, manipulation-sweeps-fragment | Present, unlabelled as ablation. |
| Sensitivity analysis (LOB review recommendation) | Boundary and size-response sweeps in F5, six-value lambda sweep in F6 | Present. |
| **Compute cost and runtime section** (Isaac Gym 5 and 6, NLE App. D, Habitat "Performance" and "Efficient GPU throughput", SWE-Gym 10,000 CPU core hours) | **Absent.** No steps-per-second, no episode wall time, no total experiment cost, despite "compute as the stated constraint" being invoked in Limitations | **Missing.** Four corpus papers give this a top-level section or appendix; SharpeArena gives it zero sentences while relying on it rhetorically. |
| Environment card / datasheet analogue (FinRL-Meta App. D and G, Gebru template) | Partially in 8 Reproducibility and in the Broader impact paragraph of 7 | **Partial.** Uses, distribution and maintenance are covered obliquely. The "tasks for which it should not be used" answer exists (retail performance marketing) but is one sentence inside Limitations. |
| Leaderboard / challenge infrastructure (Habitat 6) | Delegated to SharpeBench; "no third party has yet submitted an agent" | Correctly scoped. Not a gap, but see R12. |
| Limitations (D4RL 7, Melting Pot 8, SWE-Gym 6, Platt and Gebbie 6 "Relevant Caveats") | Sec. 7, nine substantial paragraphs | Present and **best-in-corpus**. No paper in this corpus limits itself this carefully. |
| Reproducibility statement | Sec. 8 | Present, above genre norm. |
| Appendix of commands / exact reproduction (Habitat App. E, D4RL App. E, Procgen App. D) | App. A with a provenance manifest at revision fa494389 | Present, above genre norm. |

### 3.2 What SharpeArena has that the genre does not

| SharpeArena feature | Genre precedent | Asset or oddity |
|---|---|---|
| Cross-runtime byte-identity pinned by committed FNV-1a goldens asserted by native, WASM and Python surfaces | **None in corpus.** Closest is Gymnasium's version-in-the-ID and Minari's version metadata | **Asset, and under-sold.** The paper's most defensible novelty, and currently one subsection (3.2) plus scattered mentions. |
| Recompute-from-decisions tamper evidence with an adversarial doctored-trajectory test | None. D4RL, Minari and RL Unplugged standardize storage, not verifiability against tampering | **Asset.** Genuinely new to the genre. |
| Structural leak-freedom as an interface-shape property, with a deny-list guard behind it | Closest is OSWorld's execution-based reward and tau-bench's hidden `s_db`, "hidden from the agent and the user, and can only be read from or written to using API actions" | **Asset.** The "shape not rule" framing is sharper than anything in the corpus. |
| A governed wire contract with written additive-only deprecation policy, JSON Schemas and conformance fixtures | PettingZoo 6 and Gymnasium 3 specify APIs; **neither governs them in writing** | **Asset.** Section 4 has no genre peer. |
| A "Design principles" section stating seven principles including "Distrust the simulator too" | Procgen 2.1 and Habitat "Design requirements" are closest; neither includes a self-skepticism principle | **Asset.** Keep. |
| Delegating all scoring to a separate companion system | Melting Pot separates training from testing; no corpus paper separates producing from scoring across two papers | **Asset with a cost.** It forces a reader to hold two papers, and the abstract spends a clause on it. |
| Sealed-seed commit-reveal derivation against seed recovery | None. Machado's sticky actions is the functional analogue (defeat memorization), but by injected noise rather than by cryptographic custody | **Asset.** Novel, and the better answer to Machado's problem for a verifiability-first environment. |
| Ten findings across fifteen subsections with eight `\input` fragments | No corpus paper exceeds about six result subsections in the main body | **Oddity, and a liability.** See 3.3. |
| An abstract that reports its own gate failing 23 of 24 panels | CARLA reports failure in results; TRADES reports one unmatched fact; **no corpus paper puts its own validation failure in the abstract** | **Asset if reframed, liability if left as-is.** See R2. |

### 3.3 Should the fifteen-subsection Experiments be split?

**Yes.** The genre answer is unambiguous and there are four direct precedents.

The JASSS flash-crash paper puts **4 Model Calibration and Validation** before **5 2010 Flash Crash
Scenarios** and **6 Mini Flash Crash Scenarios**. ABIDES puts architecture in 3 and 4 and then two
numbered validation case studies in 5 and 6. The LOB review separates **2 Stylized Facts** from
**7 Responsiveness to Trades: Market Impact** and **8 Comparative Study**. Melting Pot separates
**4 The Melting Pot protocol** from **7 Experiments**. And Bodor and Carlier is the negative control:
because no formal separation exists, a reader cannot tell which downstream result is conditioned on
which check.

SharpeArena's fifteen subsections partition cleanly along exactly this line.

**Instrument validation (does the simulator behave like a market, and does the probe layer work).**
- F4 stylized-facts certification (`sec:f4`) plus the Calm calibration follow-up (`sec:calm-tails`)
- F5 manipulation boundary (`sec:f5`), the concave ablation (`sec:f5-concave`), the asymmetric
  positive control (`sec:f5-positive-control`), the extended sweeps (`sec:f5-extended-sweeps`)
- F6 adverse-selection markout (`sec:f6`) and the endogenous arm (`sec:f6-endogenous`)
- The predictability probe (`sec:predictability`) and the sealed-seed mitigation (`sec:sealed-seeds`)

**Instrument calibration and findings (what the environment yields when you run policies through it).**
- F1 pipeline validation under the corrected kernel (`sec:f1`) and the eligibility witness (`sec:witness`)
- F2 regret against the closed-form reference (`sec:f2`)
- F3 generalization gap and cross-regime transfer (`sec:f3`)
- F7 failure-mode distributions (`sec:f7`)
- F8 ecology under shocks, replicated (`sec:f8`)

That is nine subsections in validation and six in findings, within genre range for both.

The split also fixes a rhetorical problem the paper currently has. F4's failure and F5's manipulation
null are evidence that the probe layer works, and placing them beside F2's regret curve makes them
read as disappointing results rather than as the environment auditing itself, which is what principle
seven ("Distrust the simulator too") promises.

One caution. The paper's existing placement rule, "every subsection sits directly after the finding
it extends", is well-reasoned and should survive the split; it applies within each of the two new
sections without modification, since every fragment stays adjacent to its parent.

---

## Part 4. Recommendations

Prioritized. Each names the exemplar it derives from, a target length, a placement, and whether it
needs new experiments or is pure restructuring.

### MUST-DO

**R1. Split Experiments into "Instrument validation" and "Calibration findings".**

Exemplars: JASSS 27(2)/8 sections 4 vs 5-6; ABIDES sections 5-6 as case studies; LOB review 2 vs 7-8;
negative exemplar Bodor and Carlier, where no formal separation exists.

Placement: replace `\section{Experiments}` with two sections, validation first. Keep the placement
rule verbatim, restated once per section.

Length: no net change; the 307-line file becomes two files of roughly 130 and 180 lines. Move the
existing roadmap paragraph into the first of the two and add a two-sentence bridge to the second.

Restructuring only. **No new experiments.**

Why it is first: largest effect per unit of effort, costs nothing empirically, and every other
recommendation is easier once it lands.

**R2. Reframe the 23-of-24 realism failure as a diagnostic-with-teeth result, and cite the field's own critique of stylized-fact validation.**

Exemplars: Platt and Gebbie, "A key concern is the fact that a very large number of models with very
different design philosophies are able to replicate the stylized facts of return time series equally
well" and "Traditional stylized fact-centric validation seems unable to detect these potential
problems, suggesting that such methods of validation are simply not sufficient"; CARLA's unretouched
"None of the methods performs perfectly even on the simplest task"; TRADES's one-sentence mechanism
for its single unmatched fact.

The argument to make, in one added paragraph at the head of the new validation section: the standard
failure mode of this genre is a simulator that passes a stylized-facts battery and is nonetheless
degenerate, because the battery does not discriminate between models. A gate that a public,
committed, deterministic generator can fail, on 23 of 24 seeded panels, against a null fixed in source
and not tuned to any market panel, is a gate with discriminating power. SharpeArena's contribution
here is a calibrated, falsifiable, committed realism gate that reports against its own generator,
which is rarer in the literature than a generator that passes an uncalibrated one.

Placement: new opening paragraph of the validation section, plus one sentence in the abstract and one
in the Limitations "One synthetic generator family" paragraph.

Length: 150 words in the validation section, 25 words in the abstract, 20 in Limitations.

Needs: two new bibliography entries (Platt and Gebbie 2017; the LOB review). **No new experiments.**

**R3. Add a named "Observation and action spaces" subsection.**

Exemplars: WebArena 2.3 and 2.4; OSWorld 2.3 and 2.4; NLE Appendices B and C; Isaac Gym 2.3.2 and
2.3.3 with numbered tensor tables.

Content: the exact fields of `MarketObservation` (trailing window shape, per-symbol snapshot,
fundamentals and news channels, mandate, clock), the exact variants and bounds of `Decision`
(including the signed-weight shorting semantics the conformance fixtures already cover), and one
table of each. This material exists across 3.1 and 4 but is never enumerated, and it is what an
implementer reads first.

Placement: new `\subsection` in Section 3, immediately after 3.1. Alternatively as Appendix B if
space binds, following NLE's precedent exactly.

Length: three quarters of a page plus two small tables.

Restructuring plus writing. **No new experiments.**

**R4. Add a "Benchmarking protocol" subsection written for future users.**

Exemplars: Melting Pot section 4 "The Melting Pot protocol"; Machado 4.2 "Proposal: Performance During
Training", whose voice is the model ("At the end of training... report the average performance of the
last k episodes"); Procgen 2.2, "we recommend training on 500 levels... and testing on held out
levels"; D4RL's normalized-score formula and training/evaluation split table.

Content: one prescriptive page. Which bands to train on and which to evaluate on; that the canonical
held-out namespace is [10256,10512) and that the 16-seed subsets used in F3 and the witness are not
it; how many seeds to run and why; that sealed derivation is the recommended mode for adversarial
evaluation and that any public bounded-band evaluation should be treated as compromised (this
sentence already exists in Limitations and belongs in the protocol); that scores are computed by the
SharpeBench kernel; and how to run the replay verifier on a submitted trajectory.

Placement: new Section 5, between the contract (4) and validation. This gives the paper a clean
front-half arc: principles, environment, contract, protocol, then evidence.

Length: 1 page.

Restructuring plus writing, drawing entirely on text already scattered in 3.4, 3.5, 4, 7 and 8.
**No new experiments.**

**R5. Move Related work forward, to immediately after the Introduction.**

Exemplars: the newer benchmark and environment papers in this corpus overwhelmingly place Related
Work or Background second (OSWorld, tau-bench, D4RL, Crafter, NLE, ALFWorld, SWE-Gym, PettingZoo,
FinRL-Meta, Habitat).

Rationale specific to this paper: 06-related contains the four-axis competitive frame (leak-freedom
mechanism, determinism guarantee, trajectory verifiability, external-agent contract, plus
generalization protocol) against FinRL, FinRL-Meta, ABIDES-Gym, TradeMaster and mbt_gym. That frame
is the paper's argument for novelty, and a reader currently meets it on page 20.

Placement: Section 2, pushing principles to 3.

Length: unchanged. Optionally compress the financial-RL paragraph by 15% since the protocol section
(R4) will absorb some of its comparative material.

Restructuring only. **No new experiments.**

### SHOULD-DO

**R6. Add a compute cost and runtime paragraph or subsection.**

Exemplars: Isaac Gym devotes **two** top-level sections (5 Characterising Simulation Performance,
6 Characterising Environment Performance) and quotes throughput of 700K steps per second for Ant; NLE
Appendix D "Environment Speed Comparison" reports 14.4K steps per second against ALE's 0.90K and makes
a democratization argument from it; Habitat quotes "several thousand frames per second running
single-threaded, and can reach over 10,000 fps multi-process on a single GPU"; SWE-Gym reports "200
human annotation hours and 10,000 CPU core hours".

Rationale: the Limitations section invokes compute as the reason no learner is trained, "with compute
as the stated constraint", without ever quantifying it, which is the weakest sentence in an otherwise
scrupulous section. Reporting steps per second for the Rust core, WASM and Python surfaces would also
convert the three-surface architecture from an engineering claim into a measured one, and would
directly support the future-work PPO baseline by letting a reader estimate its cost.

Placement: a subsection at the end of Section 3, or a paragraph in Reproducibility.

Length: one third of a page plus a three-row table (native, WASM, Python) with steps per second and
episode wall time.

**Needs new measurement, not new experiments.** A benchmarking harness run, not a research result.
This is the only recommendation on the list that requires touching a machine.

**R7. Name the validation axes explicitly, following the LOB review's split.**

Exemplar: LOB review separates **2 Stylized Facts** (unconditional distributional realism) from
**7 Responsiveness to Trades: Market Impact** (conditional response realism); Bodor and Carlier
separates 4.5 Market Impact from 4.7 Other Stylized Facts; ABIDES gives impact its own numbered
section 6.

Content: inside the new validation section (R1), group as: (a) unconditional realism, F4 and Calm;
(b) response realism, F5 and its three fragments; (c) mechanism realism, F6 and the endogenous arm;
(d) adversarial realism, predictability and sealed seeds. Two sentences introducing the four axes.
This costs almost nothing and makes the section legible at a glance, and it lets the paper say
truthfully that it fails axis (a) while passing structural checks on (b), (c) and (d) rather than
appearing to fail "realism" wholesale.

Placement: opening of the validation section. Length: 120 words. Restructuring only.
**No new experiments.**

**R8. Add a task taxonomy table.**

Exemplars: OSWorld 3.3 Data Statistics; D4RL Appendix A Task Properties; Procgen Appendix A with a
per-environment description; Melting Pot Appendix B Substrate details.

Content: one table, rows equal to task families (position trading, simplex allocation, VWAP/TWAP
execution, endogenous shared-book, deterministic LOB, market making, cointegrated pairs, regime
shift), columns equal to clearing mechanism, tiers available, mandate support, action semantics,
reference policy available, whether covered by the golden-hash guarantee, whether covered by F4.

The last two columns are valuable beyond taxonomy: they make the Limitations claim that "the Python
probe layer is outside the golden-hash guarantee" and that "the realism diagnostic does not certify
every experiment's process" visible in one place instead of stated in prose two sections apart.

Placement: Section 3, replacing part of the prose in 3.5. Length: one table, half a page.
Restructuring only. **No new experiments.**

**R9. Add an environment card appendix.**

Exemplars: Gebru et al.'s template categories in order (Motivation, Composition, Collection Process,
Preprocessing, **Uses**, Distribution, Maintenance), with the pivotal question "Are there tasks for
which the dataset should not be used? If so, please provide a description."; FinRL-Meta's Appendix D
(Dataset Documentation and Usages) and Appendix G (Accessibility, Accountability, Maintenance and
Rights) as the in-genre instance; ABIDES section 8 (Open Source Access and License); Minari's required
and recommended metadata fields.

Content: a one-page appendix answering the seven categories for the environment rather than a
dataset. Most answers already exist somewhere in the paper: motivation (Section 1), composition
(Section 3), preprocessing (not applicable, procedural), uses and non-uses (the Broader impact
paragraph already answers this: "presenting them to retail audiences as evidence of real-market skill
would be a misuse the project's documentation states as out of bounds"), distribution (crates.io,
npm, PyPI, MIT and Apache-2.0), maintenance (the contract governance document and deprecation window).

Placement: Appendix B, after the commands appendix. Length: 1 page. Restructuring plus assembly.
**No new experiments.**

**R10. Promote the determinism-as-hazard argument out of the experiments and into the environment section.**

Exemplar: Machado et al. spend section 5 of 8 on exactly this, and the paper's central prescription
follows from it; Procgen gives it 3.2 "An Ablation with Deterministic Levels".

Rationale: SharpeArena's `sec:predictability` finding (a public 2^16 band inverted 16/16 from one bar
in about a second) is the genre's canonical determinism hazard, and its sealed-seed derivation is a
better answer than sticky actions for a verifiability-first environment. Presently a reader meets the
hazard as the fourteenth subsection of Experiments. A forward reference of three sentences in 3.2,
naming the tension between "determinism is required for verifiability" and "determinism invites
memorization", and citing Machado, would let the later probe land as the resolution of a stated
tension rather than as a surprise. The Related work section already cites machado2018revisiting for
exactly this point; the environment section should too.

Placement: three sentences added to 3.2, ending in a `\Cref` to the probe. Length: 60 words.
Restructuring only. **No new experiments.**

### OPTIONAL

**R11. Add a "Who should use this and for what" paragraph.**

Exemplars: ABIDES, "ABIDES is intended to be a curated, collaborative open-source project that
provides researchers with tools that support the rapid prototyping and evaluation of complex market
agents", with an enumerated list of target uses; Gymnasium, "Gymnasium keeps its focus entirely on
the environment side of RL research"; Habitat's refusal sentence, "we... do not make any claims about
intrinsic superiority of one or the other".

Placement: end of Section 1 or start of Section 3. Length: 100 words. Restructuring only.
Note: R9's environment card partially subsumes this; do R11 only if R9 is skipped.

**R12. Add an "Extending SharpeArena" subsection.**

Exemplar: Melting Pot section 6 "Extending Melting Pot", and its argument that "since the
effectiveness of the bots in test scenarios is itself advanced by improvements in the performance of
learning systems, Melting Pot will likewise improve over time"; PettingZoo section 8 "Adoption".

Rationale: the paper is honest that "no third party has yet submitted an agent", which is the right
call. A short section on how a third party would add a scenario family or a probe would convert
that admission from a bare absence into a stated invitation, without claiming adoption.

Placement: end of Section 3 or a short section before Limitations. Length: one third of a page.
Restructuring plus writing. **No new experiments.**

**R13. Consider promoting cross-runtime byte-identity to a first-class claim in its own subsection with a table.**

Exemplar: none in corpus, which is the point. The closest precedents for presenting a cross-surface
guarantee are Minari's metadata standard and Gymnasium's version-in-the-ID, neither of which claims
identity.

Content: a three-row table (native Rust, WASM, Python binding) by three columns (what is pinned, by
which committed golden, asserted by which test), which would make the exact coverage boundary that
Section 4.4 and the Limitations paragraph both describe in prose immediately legible, and would
foreclose the reviewer question "identical in what sense, and on what surface".

Placement: Section 3.2. Length: half a page including the table. Restructuring only. Overlaps with
R8's last two columns; do one or the other, not both.

---

## Appendix: sources fetched, with retrieval method

All rows retrieved by `WebFetch` unless noted. Two `WebSearch` calls were used for discovery only
(ABM validation methodology; 2025-2026 LOB simulator papers) and no claim in this report rests on a
search snippet alone.

Full text recovered: Gymnasium (arxiv/html), Machado ALE (ar5iv), Procgen (ar5iv), Crafter (ar5iv),
NLE (ar5iv), PettingZoo (ar5iv), Melting Pot (ar5iv), WebArena (ar5iv), OSWorld (arxiv/html v2),
tau-bench (ar5iv), SWE-Gym (arxiv/html v2), Habitat (ar5iv), ALFWorld (ar5iv), Isaac Gym (ar5iv),
CARLA (ar5iv), ABIDES (ar5iv), ABIDES-Gym (ar5iv), FinRL-Meta (ar5iv), Platt and Gebbie (ar5iv),
LOB Simulations Review (arxiv/html), TRADES (arxiv/html), Bodor and Carlier (arxiv/html),
JASSS flash-crash (jasss.org), D4RL (ar5iv), Datasheets for Datasets (ar5iv), Minari standards
(minari.farama.org).

Partial: mbt_gym (ar5iv render truncated after Section 2; structure recorded only as far as
recovered, and no structural claim in Part 2 depends on it).

Failed and worked around: arXiv `/abs/` pages return abstract only (six attempts, all discarded);
arXiv `/pdf/` returns undecodable compressed streams (two attempts, both rerouted to ar5iv);
`arxiv.org/html/2404.07972v3` returns 404 (rerouted to v2); `openreview.net/forum` returns a browser
verification interstitial (rerouted to arXiv).
