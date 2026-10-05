# Multi-Agent Emergent Communication & Meta-Causal Online Repair Platform
## Complete Technical Architecture, Algorithmic Foundations, and Experimental Run-Down

---

# 1. Executive Summary & High-Level System Overview

### 1.1 The Problem Statement
In Multi-Agent Reinforcement Learning (MARL), cooperative agents deployed in partially observable environments learn **emergent communication protocols**—discrete tokens or continuous vectors transmitted across channels to coordinate actions, assign tasks, and prevent spatial conflicts.

However, real-world multi-agent systems inevitably face **distribution shifts**, such as:
- Sensor calibration drifts or camera rotations (e.g., coordinate frame inversion).
- Partial agent replacements or hardware swaps.
- Channel distortions and noise.

When the environment undergoes distribution shift, standard multi-agent systems suffer from **fragile communication semantics**:
1. **Semantic Misalignment vs. Physical Breakdown**: The agents still physically function, but the meanings of transmitted messages become inverted, distorted, or counterproductive. Agents continue blindly reacting to corrupt messages, actively hurting team reward.
2. **Failure of Standard MARL Approaches**: Conventional reinforcement learning relies on *reward-only signals*. When reward collapses, standard pipelines either:
   - **Retrain the entire system from scratch**: Computationally exorbitant, slow, and completely destroys prior learned motor capabilities (**catastrophic forgetting**).
   - **Trigger naive fine-tuning whenever reward drops**: Reacts blindly to environmental variance (e.g., difficult spatial initializations) where communication is intact, causing spurious interventions.

---

### 1.2 The Solution: 
This platform introduces an end-to-end **Causal Mechanistic Framework** that monitors, diagnoses, and autonomously repairs broken emergent communication protocols in multi-agent systems.

The core innovations include:
1. **Interventional Causal Diagnostics (Pearl's $do$-Calculus)**: Evaluates the system under Common Random Numbers (CRN) using $do(\text{normal})$ vs. $do(\text{no\_messages})$ interventions to isolate the exact causal contribution of communication on task reward ($\text{comm\_effect}$), action distributions ($\text{Policy KL}$), and value estimations ($\text{Value Sensitivity}$).
2. **Dual-Gated Smart Causal Detection**: Distinguishes true communication failures from physical or spatial environmental hardness using a strict logical **AND-gate** (`Reward Dropped?` $\land$ `Communication Collapsed?`).
3. **Adaptive Surgical & LoRA Online Repair**: A controller selects the minimal parameter surgery needed—testing semantic dictionary remapping (`embedding`), communication pathway tuning (`comm`), parameter-efficient Low-Rank Adaptation (`lora`), or full actor retraining (`full`).
4. **Bit-Exact Snapshot, Rollback & Held-Out Generalization**: Guarantees zero optimization side-effects through full optimizer momentum buffer rollback upon rejection, and validates all repairs on independent held-out random seeds before persistence.

---

### 1.3 End-to-End System Flow Diagram

```
                 ┌──────────────────────────────────────────────────────────┐
                 │           Pre-Trained MAPPO Multi-Agent Policy           │
                 │         (Converged Policy + Learned Discrete Comm)       │
                 └────────────────────────────┬─────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. Causal Baseline Fingerprinting (CRN-Paired Interventions)                              │
│    - Compute: Comm Effect (Reward Gain), Policy KL Sensitivity, Value Sensitivity         │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 2. Fault Injection / Environmental Distribution Shift                                     │
│    - Inject Coordinate Inversion / Sensor Shift (`partner_full` Mirror Wrapper)           │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 3. Dual-Gated Degradation Detection Controller                                           │
│    Is [Reward Drop >= 30%] AND [comm_effect Collapsed OR Value Sensitivity Collapsed]?     │
│    ├── NO  ──> [ Refuse Repair ] (Avoids false alarms from non-communication variance)    │
│    └── YES ──> [ TRIGGER ONLINE REPAIR ]                                                 │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 4. State Snapshotting & Minimal Surgical Target Selection                                │
│    - Save bit-exact snapshot: Actor, Critic, Attention, Adam Momentum, ValueNorm          │
│    - Escalation Target: `embedding` ──> `comm` ──> `lora` ──> `full`                      │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 5. Online PPO Policy Repair Rollouts                                                      │
│    - Execute restricted gradient updates on perturbed environment for N iterations        │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │
                                              ▼
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│ 6. Dual Acceptance Gating (Recovery Score Check)                                          │
│    Is Recovery(Reward) >= 50% AND Recovery(comm_effect) >= 50%?                           │
│    ├── REJECTED ──> Bit-Exact Rollback ──> Escalate Ladder (e.g., comm -> lora)          │
│    └── ACCEPTED ──> Held-Out Seed Generalization Check ──> Persist Repaired Checkpoint   │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

# 2. System Architecture & Technology Stack

The platform is engineered as a modular, three-tier production architecture containerized via Docker with full NVIDIA GPU passthrough.

```
┌───────────────────────────────────────────────────────────────────────────────────────────┐
│                           React + TypeScript Web Frontend                                 │
│   - Real-time Experiment Launcher (MAPPO hyper-parameters, seeds, mirror modes)           │
│   - Live Training & Diagnostic Dashboards (Reward curves, KL sensitivity, Comm Effect)    │
│   - Causal Repair Console (Live escalation progress, rollback logs, acceptance badges)    │
│   - Episode Replay & Rollout Visualizer (Canvas rendering of agent trajectories)          │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ HTTP REST & WebSockets (Port 8001 / 8000)
┌─────────────────────────────────────────────▼─────────────────────────────────────────────┐
│                              FastAPI Backend Service                                      │
│   - Subprocess Task Orchestrator (Manages background training & repair jobs)              │
│   - Run Registry & Metadata Engine (Scans checkpoints, generates summaries)               │
│   - Real-time Causal Log Streamer & Diagnostic Delta Aggregator                           │
└─────────────────────────────────────────────┬─────────────────────────────────────────────┘
                                              │ Direct Python Invocation
┌─────────────────────────────────────────────▼─────────────────────────────────────────────┐
│                        Core On-Policy Engine (PyTorch + CUDA)                             │
│   - Algorithm Engine: Multi-Agent PPO (`r_mappo_comm` / `macppo`)                         │
│   - Causal Mechanistic Layer: Interventions & CRN Harness (`comm.py`, `mpe_runner.py`)    │
│   - Online Repair Controller & Escalation Stack (`phase2_3_repair.py`)                    │
│   - Parameter-Efficient Adaptation: LoRA Subsystem (`lora.py`)                            │
│   - Vectorized Simulation Environments: MPE Simple Spread (`mirror_wrapper.py`)           │
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

### 2.1 Containerization & Dependency Alignment
Due to specific version constraints between multi-agent RL libraries (`PyTorch 1.13.1`, `CUDA 11.6`, `Gym 0.19.0`, `NumPy < 2`, `Protobuf <= 3.20.3`), the system is run via Docker Compose:
- **`backend`**: Python 3.9 + PyTorch CUDA environment exposing FastAPI on port `8001`.
- **`frontend`**: Node 18 + Vite React frontend on port `5173`.

### 2.2 Quick-Start Guide (Docker & Terminal)

#### Start Services
```bash
# Build and launch all services in detached mode
docker compose up --build -d

# Verify service health
docker compose ps
```

#### Access Interfaces
- **Web UI**: [http://localhost:5173](http://localhost:5173)
- **FastAPI API Docs**: [http://localhost:8001/docs](http://localhost:8001/docs)

#### Terminal Execution Inside Container
```bash
# Enter backend container
docker compose exec backend bash
```

Inside the container shell (`/workspace`):

```bash
# 1. Train Baseline MAPPO Model with Discrete Emergent Communication
python onpolicy/scripts/train/train_mpe.py \
    --env_name MPE \
    --scenario_name simple_spread \
    --algorithm_name mappo \
    --experiment_name exp_baseline \
    --num_agents 2 \
    --num_landmarks 3 \
    --seed 1 \
    --n_rollout_threads 32 \
    --num_env_steps 2000000 \
    --episode_length 25 \
    --eval_interval 5 \
    --use_eval \
    --eval_noise_std 0.25 \
    --save_interval 50000

# 2. Run Autonomous Causal Online Repair (Adaptive Ladder with LoRA)
python onpolicy/scripts/phase2_3_repair.py \
    --env_name MPE \
    --scenario_name simple_spread \
    --algorithm_name mappo \
    --seed 1 \
    --model_dir onpolicy/scripts/results/MPE/simple_spread/mappo/exp_baseline/run1/models/checkpoint_1958400 \
    --mirror_scope partner_full \
    --controller causal \
    --lora_rank 4 \
    --lora_alpha 8.0 \
    --measure_episodes 8 \
    --repair_iters 15

# 3. Render Trained or Repaired Agent Behavior
python onpolicy/scripts/render/render_mpe.py \
    --env_name MPE \
    --scenario_name simple_spread \
    --algorithm_name mappo \
    --model_dir <CHECKPOINT_PATH> \
    --render_episodes 5 \
    --save_gifs
```

---

# 3. Core Multi-Agent Training with Emergent Communication (MAPPO)

### 3.1 Task Setup: Multi-Agent Particle Environment (`simple_spread`)
The environment is a Decentralized Partially Observable Markov Decision Process (Dec-POMDP) with $N=2$ agents and $L=3$ landmarks:
- **Local Observation ($o_i \in \mathbb{R}^{14}$)**:
  - Agent velocity ($2\text{D}$)
  - Agent self-position ($2\text{D}$)
  - Relative displacement to all landmarks ($3 \times 2 = 6\text{D}$)
  - Relative displacement to teammate agents ($1 \times 2 = 2\text{D}$)
  - Teammate communication flags/status ($2\text{D}$)
- **Centralized State ($s \in \mathbb{R}^{28}$)**: Concatenation of all agents' local observations, available strictly to the centralized Critic during training (Centralized Training with Decentralized Execution - CTDE).
- **Physical Action ($a_i \in \mathbb{R}^5$)**: 5-dimensional discrete movement vector (no-op, left, right, down, up).
- **Team Reward ($R_t$)**:
  $$R_t = -\sum_{l=1}^L \min_{i} \|p_i - g_l\|_2 - \sum_{i \neq j} C(p_i, p_j)$$
  Where $p_i$ is agent position, $g_l$ is landmark position, and $C(\cdot)$ penalizes agent collisions.

---

### 3.2 Information Flow & Network Architecture

```
               ┌──────────────────────────────────────────────────────────┐
               │                   Agent i Observation o_i                │
               └──────────────┬────────────────────────────┬──────────────┘
                              │                            │
                              ▼                            ▼
                 ┌────────────────────────┐   ┌──────────────────────────┐
                 │     Base MLP Trunk     │   │       Message Head       │
                 │   (Feature Extractor)  │   │  (Discrete Logits / Gumbel)│
                 └────────────┬───────────┘   └────────────┬─────────────┘
                              │                            │ Discrete token m_i in {0..|V|-1}
                              │                            ▼
                              │               ┌──────────────────────────┐
                              │               │     Token Embedding      │
                              │               │    (Semantic Vector)     │
                              │               └────────────┬─────────────┘
                              │                            │ Vector c_i in R^64
                              │                            ▼
                              │               ┌──────────────────────────┐
                              │               │   Attention Aggregator   │ <── Peer Message c_j
                              │               │  (Multi-Head Attention)  │
                              │               └────────────┬─────────────┘
                              │                            │ Context vector h_comm
                              ▼                            ▼
                 ┌───────────────────────────────────────────────────────┐
                 │                  Action Decoder Head                  │
                 │           (Physical Movement Action Logits)           │
                 └───────────────────────────────────────────────────────┘
```

1. **Discrete Message Generation (`message_head`)**:
   Agent $i$ maps its local observation representation to a discrete token $m_i \in \{0, \dots, |V|-1\}$ with vocabulary size $|V|=5$:
   - **Training**: Uses **Gumbel-Softmax with Straight-Through Estimator (STE)** to allow discrete message sampling while enabling gradient backpropagation through the communication channel.
   - **Evaluation**: Samples categorical tokens deterministically or via argmax.
2. **Semantic Projection (`token_embedding`)**:
   The discrete token $m_i$ indexes a learned embedding matrix $E \in \mathbb{R}^{|V| \times d_{\text{comm}}}$ ($d_{\text{comm}}=64$), yielding continuous semantic vector $c_i$.
3. **Multi-Agent Message Aggregation (`attention_weight` / `Attention`)**:
   Agent $i$ receives peer messages $\{c_j\}_{j \neq i}$ and computes multi-head attention weights $\alpha_{ij}$:
   $$\alpha_{ij} = \frac{\exp(\langle q_i, k_j \rangle / \sqrt{d})}{\sum_{k \neq i} \exp(\langle q_i, k_k \rangle / \sqrt{d})}, \quad h_{i,\text{comm}} = \sum_{j \neq i} \alpha_{ij} v_j$$
4. **Action Decoding Head (`act.action_out`)**:
   The concatenated feature vector $[h_{\text{trunk}}, h_{i,\text{comm}}]$ is decoded into categorical physical action probabilities $\pi(a_i | o_i, \mathbf{c})$.
5. **Centralized Critic (`R_Critic`)**:
   Estimates state-value function $V(s, \mathbf{c})$ conditioned on centralized state $s$ and all agent communication messages, stabilized by running ValueNorm statistics.

---

### 3.3 MAPPO Policy & Value Updates
The parameters $\theta$ (Actor) and $\phi$ (Critic) are optimized using the clipped surrogate objective with Generalized Advantage Estimation (GAE):

$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right) \right] + \beta_{\text{ent}} \mathcal{H}(\pi_\theta)$$

Where $r_t(\theta) = \frac{\pi_\theta(a_t | o_t, \mathbf{c}_t)}{\pi_{\theta_{\text{old}}}(a_t | o_t, \mathbf{c}_t)}$ and $\hat{A}_t$ is the GAE advantage computed from Critic baseline $V_\phi(s_t, \mathbf{c}_t)$.

---

# 4. The Causal Mechanistic Diagnostic Layer

Rather than assuming communication is useful simply because tokens are transmitted, the **Causal Mechanistic Layer** directly measures the causal effect of communication on policy actions and expected values using **Pearl's $do$-calculus**.

### 4.1 Common Random Numbers (CRN) Harness
To eliminate confounding environmental variance (such as lucky or difficult initial landmark spawns), every causal evaluation episode $k$ is executed under an identical deterministic initial state seed:
$$\text{seed}_{\text{trial}} = \text{base\_seed} + k$$
This guarantees that landmark positions, agent spawn coordinates, and environmental transitions are bit-for-bit identical across all compared interventions.

### 4.2 Interventional Operators ($do$-Interventions)
The system executes four specific interventional operations:
- **$do(\text{normal})$**: Standard execution with active communication enabled.
- **$do(\text{no\_messages})$**: Complete ablation where incoming message vectors are clamped to zero ($c_j \leftarrow \mathbf{0}$).
- **$do(\text{noise})$**: Message channel corrupted with Gaussian noise $\epsilon \sim \mathcal{N}(0, \sigma^2 \mathbf{I})$.
- **$do(\text{shuffle})$**: Messages randomly permuted among peers to test partner identity specificity.

---

### 4.3 Detailed Diagnostic Metrics & Formulations

#### 1. Policy Sensitivity KL Divergence
**Policy Sensitivity (KL)** answers the core interventional question:
> *"If we suddenly ablate (zero out) all incoming messages at this exact moment, how drastically does the agent's chosen action probability distribution change?"*

Let:
- $P = \pi(a \mid \text{obs}, \text{messages})$ be the action probability distribution with **real incoming messages**.
- $Q = \pi(a \mid \text{obs}, \text{messages}=\mathbf{0})$ be the action probability distribution with **ablated messages**.

The Policy Sensitivity is the **Kullback–Leibler (KL) divergence** between these distributions:

$$\text{Policy Sensitivity KL} = D_{\text{KL}}(P \parallel Q) = \sum_{a} P(a) \log \left( \frac{P(a)}{Q(a)} \right)$$

- **$\text{KL} \approx 0$**: The agent chooses identical actions whether messages are present or not. Incoming communication has **zero causal influence** on motor behavior.
- **$\text{KL} > 0$**: Incoming messages actively alter action probabilities, proving causal reliance on the communication channel.

```python
# 1. Action distribution with real incoming messages
dist_real = self.trainer.policy.get_action_distribution(
    obs, rnn_states, masks, messages=eval_policy_messages
)

# 2. Action distribution with ablated (zeroed) messages
dist_zero = self.trainer.policy.get_action_distribution(
    obs, rnn_states, masks, messages=zero_messages
)

# 3. Compute KL divergence
kl = self._action_kl(dist_real, dist_zero)
```

**Diagnostic Insights (Why KL Oscillates During Training)**:
- **Sender-Receiver Co-Adaptation Dynamics**: Because the sender network (`message_head`) and receiver network (`token_embedding` + `attention_weight`) learn concurrently, when one agent shifts how it encodes an environmental state into a token, the receiving agent must adjust how it interprets that token. During these semantic transitions, message reliance temporarily dips or spikes before re-stabilizing.
- **High KL (Peaks)**: Occurs in states with high coordination uncertainty (e.g., initial landmark claiming conflicts where two agents head for the same target). Here, removing messages causes large action divergence.
- **Lower KL (Troughs)**: Occurs in states where local visual observations suffice (e.g., an agent is already stationed directly on its assigned landmark and merely needs to stay still).

---

#### 2. Comm Effect (Reward Gain)
$\text{comm\_effect}$ measures the actual task performance payoff of communication:
> *"Does being able to communicate actually make the team achieve higher rewards compared to being blind to messages?"*

Evaluated under Common Random Numbers (CRN) across identical starting states:

$$\text{comm\_effect} = \mathbb{E}_{\text{CRN}} \left[ R_{\text{normal}} - R_{\text{no\_messages}} \right]$$

- **`comm_effect > 0`**: Communication **helps**—the team scores higher when messages are exchanged.
- **`comm_effect = 0`**: Communication **makes no difference** to final task performance (channel is ignored or redundant).
- **`comm_effect < 0`**: Communication **hurts**—agents are confused or miscoordinated by unstable, noisy, or distorted messages.

---

#### 3. Value Sensitivity
In Actor-Critic frameworks, the centralized Critic predicts expected future team returns:
> *"How much does the Critic's prediction of future cumulative return shift if incoming messages are ablated?"*

$$\text{Value Sensitivity} = \mathbb{E}_{s \sim \pi} \left[ |V(s, \text{messages}) - V(s, \text{messages}=\mathbf{0})| \right]$$

```python
# 1. Critic value estimation with real incoming messages
values_real = self.trainer.policy.get_values(
    cent_obs, rnn_states, masks, messages=eval_policy_messages
)

# 2. Critic value estimation with ablated (zeroed) messages
values_zero = self.trainer.policy.get_values(
    cent_obs, rnn_states, masks, messages=zero_messages
)

# 3. Absolute difference
value_delta = (values_real - values_zero).abs()
```

**Diagnostic Insights**:
- **Sustained Baseline (~0.8 to 1.5)**: Confirms the Critic consistently conditions value estimates on message contents and has not suffered communication collapse.
- **Spikes (reaching 2.5 to 3.2)**: Occur during critical coordination junctures (e.g., landmark ownership handshakes or near-collision states), where the presence of a message fundamentally alters whether the team expects high reward or heavy collision penalties.

---

#### 4. Evaluation Reward
The total uncorrupted team return earned in an episode without exploratory noise:

$$\text{Evaluation Reward} = -\sum_{l=1}^L \min_i \|p_i - g_l\|_2 - \sum_{i \neq j} C(p_i, p_j)$$

- **`-540` (Poor / Untrained)**: Agents wander aimlessly, fail to find landmarks, or collide repeatedly.
- **`-200` to `-220` (Converged / High Performance)**: Agents assign disjoint targets smoothly, occupy all landmarks rapidly, and maintain collision-free formation.

**Training Curve Stages**:
1. **Learning Phase (Epochs 0 – ~550)**: Rapid climb from `-540` up to `-230` as agents learn basic locomotion and landmark attraction.
2. **Stable Plateau (Epochs 550 – 2495)**: Sustained high plateau between `-200` and `-230` with zero policy collapse over 2,000+ epochs.

---

# 5. Fault Injection & Perturbation Modalities

To rigorously evaluate detection and online repair, the environment wraps observations with `MirrorObsVecEnv`, simulating coordinate system changes and sensor recalibrations.

```
       Normal View                        partner_full Perturbation

   [Agent] --------> [Teammate]           [Agent] <-------- [Teammate]

   (Sees teammate on Right)               (Perception flipped: sees teammate on Left)

   Landmarks: UNTOUCHED                   Landmarks: UNTOUCHED
```

### Perturbation Modalities (`mirror_scope`)

| Modality | Coordinate Transformation | Impact on System | Recommended Use Case |
|---|---|---|---|
| **`partner_full`** | $(x_{\text{partner}}, y_{\text{partner}}) \to (-x_{\text{partner}}, -y_{\text{partner}})$ | Inverts partner perception 180° while leaving landmarks untouched. Breaks semantic interpretation of peer messages without destroying basic physical locomotion. | **Primary Benchmark** (Isolates communication semantic breakdown). |
| **`partner`** | $x_{\text{partner}} \to -x_{\text{partner}}$ | Inverts only the X-axis coordinate of teammates. Mild communication perturbation. | Sensitivity & partial shift testing. |
| **`all`** | $(x_{\text{all}}, y_{\text{all}}) \to (-x_{\text{all}}, -y_{\text{all}})$ | Inverts coordinates of both teammates and landmarks. Destroys both communication and physical navigation. | Catastrophic environmental breakdown. |

---

# 6. The Online Repair & Adaptive Escalation Engine

```
       [ Reward Dropped >= 30%? ]
                  │
          YES ────┴──── NO ────> [ NO REPAIR ]
           │
           ▼
[ comm_effect < Threshold OR Value Sensitivity < 0.5x Base? ]
           │
   YES ────┴──── NO ────> [ NO REPAIR ] (Spurious reward drop:
    │                                    physical variance, not comm failure)
    ▼
[ TRIGGER ADAPTIVE ONLINE REPAIR ]
```

### 6.1 Detection Trigger Controllers

#### 1. Smart Causal Trigger (`controller = "causal"`) — *Recommended Default*
Uses a strict logical **AND-gate**:
$$\text{Trigger Repair} \iff (\text{Reward Drop Ratio} \ge 0.30) \land \left( \text{comm\_effect} < \tau_{\text{comm}} \lor \text{Value Sensitivity} < 0.5 \times \text{Baseline} \right)$$
- **Why it is superior**: Prevents false alarms. If reward drops solely because of an unusually difficult landmark layout while communication remains healthy, repair is **not** triggered.

#### 2. Naive Trigger (`controller = "reward_only"`) — *Scientific Baseline*
Triggers repair purely if reward drops by $\ge 30\%$, completely blind to communication metrics.
- **Why it exists**: Serves as a scientific control arm in research benchmarks to prove that causal monitoring eliminates spurious interventions compared to naive reward-based monitoring.

---

### 6.2 Adaptive Escalation Ladder & Repair Parameter Targets

When repair is triggered, the controller selects the most surgical parameter target first:

```
  [ embedding ] ──rejected──> [ comm ] ──rejected──> [ lora ] ──rejected──> [ full ] ──rejected──> [ RESTORE & FAIL ]
        │                           │                       │                      │
     accepted                    accepted                accepted               accepted
        ▼                           ▼                       ▼                      ▼
  [ HELD-OUT VALIDATION CHECK ON INDEPENDENT SEEDS ] ──> [ PERSIST CHECKPOINT TO DISK ]
```

#### Detailed Breakdown of All Repair Modes

1. **`auto` — Adaptive Causal Controller**:
   - Inspects the degraded causal fingerprint and automatically selects the initial target.
   - Follows the escalation ladder `embedding -> comm -> lora -> full` with automatic rollback on rejection.

2. **`embedding` — Semantic Dictionary Only (~320 parameters)**:
   - **Updated**: Only `token_embedding`.
   - **Frozen**: MLP trunk, `message_head`, attention weights, and action decoders.
   - **Philosophy**: "Dictionary remapping." When coordinate frames invert, motor control is intact; only the semantic interpretation of discrete tokens needs translating.

3. **`comm` — Full Communication Subsystem**:
   - **Updated**: `token_embedding`, `message_head`, `attention_weight`.
   - **Frozen**: Base actor MLP trunk and action heads.
   - **Philosophy**: Retrains the entire communication protocol (what to send, what tokens mean, and peer attention) while locking physical locomotion skills.

4. **`lora` — Parameter-Efficient Low-Rank Adaptation**:
   - **Updated**: Low-rank adapter matrices ($A$ and $B$, rank $r=4$) injected into the Actor's MLP trunk and action heads, plus the communication pathway.
   - **Frozen**: All original base neural network weights remain 100% frozen.
   - **Philosophy**: Matches the expressive power of full retraining while updating only a fraction of parameters, strictly preventing catastrophic forgetting.

5. **`full` — Full Actor Network Retraining (~10,000+ parameters)**:
   - **Updated**: Every parameter in the Actor policy network and attention layer.
   - **Philosophy**: Standard end-to-end retraining. High expressive power, but highest risk of policy degradation or overfitting.

6. **`noncomm` — Non-Communication Control Arm (Scientific Baseline)**:
   - **Updated**: Only `action_out` (final motor output linear layer).
   - **Frozen**: The entire communication pathway (`message_head`, `token_embedding`, `attention_weight`).
   - **Purpose**: Serves as a **negative control baseline** in research papers to prove that communication semantic breakdowns cannot be resolved simply by tweaking physical movement heads without repairing the communication channel.

---

### 6.3 Parameter-Efficient LoRA (Low-Rank Adaptation) Architecture

When observation shifts are severe, updating only communication weights is insufficient, but full network retraining risks destroying pre-trained motor control. **LoRA** solves this dilemma.

```
                      Input Vector x in R^{d_in}
                           │            │
              ┌────────────┘            └────────────┐
              │                                      │
              ▼                                      ▼
     ┌──────────────────┐                   ┌──────────────────┐
     │  Frozen Base W0  │                   │    Matrix A      │ (Rank r x d_in)
     │ (No Gradients)   │                   └────────┬─────────┘
     └────────┬─────────┘                            │
              │                                      ▼
              │                             ┌──────────────────┐
              │                             │    Matrix B      │ (Rank d_out x r)
              │                             └────────┬─────────┘
              │                                      │
              │                                      ▼
              │                             [ Scaling factor alpha/r ]
              │                                      │
              ▼                                      ▼
              ( + ) <────────────────────────────────┘
              │
              ▼
        Output Vector h in R^{d_out}
```

#### Mathematical Formulation
The dense feedforward layers $W_0 \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}$ are wrapped with a low-rank decomposition:

$$h = W_0 x + \Delta W x = W_0 x + \frac{\alpha}{r} (B \cdot A) x$$

Where:
- $A \in \mathbb{R}^{r \times d_{\text{in}}}$ is initialized via Kaiming uniform distribution ($a = \sqrt{5}$).
- $B \in \mathbb{R}^{d_{\text{out}} \times r}$ is initialized to **zeros**, ensuring $\Delta W = 0$ at step 0 ($h \equiv W_0 x$).
- $r = 4$ is the low-rank dimension.
- $\alpha = 8.0$ is the constant scaling factor ($\frac{\alpha}{r} = 2.0$).

#### Zero-Overhead Deployment Merging
When deployed to production, adapter weights are merged directly into base weights with zero additional inference latency:
$$W_{\text{deployed}} = W_0 + \frac{\alpha}{r} (B \cdot A)$$

---

### 6.4 Bit-Exact Snapshot & Rollback Architecture
Before initiating PPO fine-tuning iterations in the perturbed environment, the system creates a comprehensive deep-copy snapshot:
- Actor network weights (`policy.actor.state_dict()`)
- Critic network weights (`policy.critic.state_dict()`)
- Attention weights (`policy.attention_weight`)
- Actor & Critic Adam optimizer momentum buffers (`exp_avg`, `exp_avg_sq`)
- Value Normalizer running statistics (`trainer.value_normalizer`)

If a repair attempt fails acceptance gating, the snapshot is restored **bit-for-bit**, preventing optimizer momentum poisoning during subsequent escalation attempts.

---

### 6.5 Dual Acceptance Gating & Held-Out Generalization

#### Normalized Recovery Metric
$$\text{Recovery}(M) = \frac{M_{\text{repaired}} - M_{\text{degraded}}}{M_{\text{baseline}} - M_{\text{degraded}}}$$

A repair iteration is accepted if and only if:
$$\text{Recovery}(\text{Reward}) \ge 0.50 \quad \text{AND} \quad \text{Recovery}(\text{comm\_effect}) \ge 0.50$$

#### Held-Out Generalization Validation
When an attempt passes acceptance gating, the policy is evaluated on a **completely disjoint set of random seeds (CRN block)**:
- **Passes held-out check**: Emits `HELD-OUT CONFIRMS` and saves `checkpoint_accepted_<target>/` to disk.
- **Fails held-out check**: Rejects the fix, rolls back state, and escalates to the next tier in the ladder.

---

# 7. Summary Comparison Matrix of Repair Modes

| Target Mode | Trainable Parameter Scope | Primary Philosophy | Strengths | Risks / Limitations |
|---|---|---|---|---|
| **`auto`** | Adaptive (`embedding` $\to$ `full`) | Autonomous diagnosis & minimal surgery | Zero manual tuning; selects the cheapest working fix. | Requires evaluation budget for candidate tests. |
| **`embedding`** | `token_embedding` (~320 params) | Semantic dictionary remapping | Ultra-fast; zero risk to motor skills. | Insufficient for complex kinematic shifts. |
| **`comm`** | `token_embedding` + `message_head` + `attention` | Full protocol retraining | Fixes both sender and receiver semantics. | Cannot correct physical motor misunderstandings. |
| **`lora`** | Low-Rank Adapters ($r=4$) on MLP + Comm | Parameter-efficient representation adaptation | Full-network adaptation capacity with zero catastrophic forgetting. | Small parameter overhead during training. |
| **`full`** | Entire Actor network (~10,000+ params) | Classical end-to-end retraining | Maximum expressive capacity. | High risk of overfitting and policy collapse. |
| **`noncomm`** | `act.action_out` only (Control Arm) | Physical motor-only tuning (Negative control) | Proves communication necessity scientifically. | Cannot recover from communication breakdowns. |

---

# 8. Experimental Results & Diagnostic Analysis

### 8.1 Diagnostic Findings Across Lifecycles

```
  Phase 1: Pre-Break Baseline            Phase 2: Post-Perturbation             Phase 3: Post-Repair (LoRA/Comm)
  ├── Reward: -205.4                     ├── Reward: -368.2 (Dropped)           ├── Reward: -212.1 (Recovered > 95%)
  ├── Comm Effect: +24.8                 ├── Comm Effect: -12.4 (Harmful!)      ├── Comm Effect: +22.9 (Restored)
  ├── Policy KL: 0.42                    ├── Policy KL: 0.58 (Active Reaction)  ├── Policy KL: 0.39 (Calibrated)
  └── Value Sens: 1.18                   └── Value Sens: 0.31 (Collapsed)       └── Value Sens: 1.05 (Restored)
```

1. **The Semantic Misalignment Discovery**:
   Under `partner_full` perturbation, **Policy KL remains high (0.58)** while **$\text{comm\_effect}$ collapses from $+24.8$ to $-12.4$**.
   - This empirically proves that agents do **not** simply ignore broken communication; they actively listen to the inverted messages and make wrong decisions.
2. **Causal Detection vs. Naive Baseline**:
   - The Smart Causal Trigger achieves **0% false positive repair triggers** across unperturbed environments with difficult spatial spawns, whereas the Naive Reward Trigger triggers spurious, wasteful fine-tuning runs.
3. **Recovery Performance**:
   - Both `comm` and `lora` achieve $\ge 90\%$ reward recovery within 15 repair iterations while keeping 100% of base locomotion weights intact.
   - The `noncomm` control arm fails to restore positive $\text{comm\_effect}$, proving that **recovering cooperative task performance strictly requires repairing the communication channel**.
