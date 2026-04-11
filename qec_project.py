import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error, pauli_error
import cirq
from scipy import stats
import io          
import base64      



def build_qiskit_noise_model(noise_type="bitflip", p=0.05):
    noise_model = NoiseModel()

    if noise_type == "bitflip":
        err_1q = pauli_error([("X", p), ("I", 1 - p)])
        err_2q = pauli_error([
            ("II", (1 - p)**2),
            ("IX", p * (1 - p)),
            ("XI", p * (1 - p)),
            ("XX", p**2)
        ])
    elif noise_type == "depolarizing":
        err_1q = depolarizing_error(p, 1)
        err_2q = depolarizing_error(p, 2)
    else:
        raise ValueError("noise_type must be 'bitflip' or 'depolarizing'")

    for gate in ["id", "h", "x", "y", "z", "s", "sdg", "t", "tdg"]:
        noise_model.add_all_qubit_quantum_error(err_1q, gate)

    noise_model.add_all_qubit_quantum_error(err_2q, "cx")
    return noise_model


def qiskit_5qubit_circuit():
    qc = QuantumCircuit(5, 5)
    qc.h(0)
    qc.cx(0, 1)
    qc.cx(0, 2)
    qc.cx(0, 3)
    qc.cx(0, 4)
    qc.measure(range(5), range(5))
    return qc


def run_qiskit_multi_p(noise_type="bitflip", p_values=None, shots=2000, seed=42):
    if p_values is None:
        p_values = [0.01, 0.03, 0.05, 0.08, 0.10]

    results = {}
    for p in p_values:
        qc = qiskit_5qubit_circuit()
        noise_model = build_qiskit_noise_model(noise_type=noise_type, p=p)

        sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
        job = sim.run(qc, shots=shots)
        counts = job.result().get_counts()
        results[p] = counts

    return results

def apply_stabilizer_measurement(qc, data_qubits, ancilla, stabilizer):
    qc.reset(ancilla)

    for i, op in enumerate(stabilizer):
        if op == "X":
            qc.h(data_qubits[i])

    for i, op in enumerate(stabilizer):
        if op == "I":
            continue
        elif op == "Z":
            qc.h(ancilla)
            qc.cx(data_qubits[i], ancilla)
            qc.h(ancilla)
        elif op == "X":
            qc.cx(data_qubits[i], ancilla)
        else:
            raise ValueError("Only I/X/Z supported")

    for i, op in enumerate(stabilizer):
        if op == "X":
            qc.h(data_qubits[i])


def qiskit_syndrome_measurement_circuit():
    data = list(range(5))
    anc = list(range(5, 9))
    cl = list(range(4))

    qc = QuantumCircuit(9, 4)
    stabilizers = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

    for k, stab in enumerate(stabilizers):
        apply_stabilizer_measurement(qc, data_qubits=data, ancilla=anc[k], stabilizer=stab)
        qc.measure(anc[k], cl[k])

    return qc


def run_syndrome_measurement_qiskit(noise_type="bitflip", p=0.05, shots=2000, seed=42):
    qc = qiskit_syndrome_measurement_circuit()
    noise_model = build_qiskit_noise_model(noise_type=noise_type, p=p)

    sim = AerSimulator(noise_model=noise_model, seed_simulator=seed)
    job = sim.run(qc, shots=shots)
    syndrome_counts = job.result().get_counts()
    return syndrome_counts


def cirq_5qubit_circuit(noise_type="bitflip", p=0.05):
    qubits = cirq.LineQubit.range(5)
    circuit = cirq.Circuit()

    circuit.append(cirq.H(qubits[0]))
    for i in range(1, 5):
        circuit.append(cirq.CNOT(qubits[0], qubits[i]))

    if noise_type == "bitflip":
        noise_gate = cirq.bit_flip(p)
    elif noise_type == "depolarizing":
        noise_gate = cirq.depolarize(p)
    else:
        raise ValueError("noise_type must be 'bitflip' or 'depolarizing'")

    circuit.append(noise_gate.on_each(*qubits))
    circuit.append(cirq.measure(*qubits, key="m"))
    return circuit


def run_cirq_multi_p(noise_type="bitflip", p_values=None, reps=2000, seed=42):
    if p_values is None:
        p_values = [0.01, 0.03, 0.05, 0.08, 0.10]

    results = {}
    sim = cirq.Simulator(seed=seed)

    for p in p_values:
        circuit = cirq_5qubit_circuit(noise_type=noise_type, p=p)
        result = sim.run(circuit, repetitions=reps)
        hist = result.histogram(key="m")
        results[p] = hist

    return results


def qiskit_counts_to_df(qiskit_results, noise_type):
    rows = []
    for p, counts in qiskit_results.items():
        for outcome, count in counts.items():
            rows.append({
                "backend": "qiskit",
                "noise": noise_type,
                "p": float(p),
                "outcome": outcome,
                "count": int(count)
            })
    return pd.DataFrame(rows)


def cirq_hist_to_df(cirq_results, noise_type, n_qubits=5):
    rows = []
    for p, hist in cirq_results.items():
        for outcome_int, count in hist.items():
            outcome = format(outcome_int, f"0{n_qubits}b")
            rows.append({
                "backend": "cirq",
                "noise": noise_type,
                "p": float(p),
                "outcome": outcome,
                "count": int(count)
            })
    return pd.DataFrame(rows)


def run_simulation_both_noises(p_values=None, shots=2000, reps=2000, seed=42):
    if p_values is None:
        p_values = [0.01, 0.03, 0.05, 0.08, 0.10]

    results = {"qiskit": {}, "cirq": {}}
    for noise_type in ["bitflip", "depolarizing"]:
        results["qiskit"][noise_type] = run_qiskit_multi_p(noise_type, p_values, shots, seed)
        results["cirq"][noise_type] = run_cirq_multi_p(noise_type, p_values, reps, seed)
    return results


def save_simulation_results_both_noises(results, merged_all_csv="simulation_results_ALL.csv"):
    df_q_bf = qiskit_counts_to_df(results["qiskit"]["bitflip"], "bitflip")
    df_q_dp = qiskit_counts_to_df(results["qiskit"]["depolarizing"], "depolarizing")
    df_c_bf = cirq_hist_to_df(results["cirq"]["bitflip"], "bitflip")
    df_c_dp = cirq_hist_to_df(results["cirq"]["depolarizing"], "depolarizing")

    df_q_bf.to_csv("qiskit_bitflip_results.csv", index=False)
    df_q_dp.to_csv("qiskit_depolarizing_results.csv", index=False)
    df_c_bf.to_csv("cirq_bitflip_results.csv", index=False)
    df_c_dp.to_csv("cirq_depolarizing_results.csv", index=False)

    merged_all = pd.concat([df_q_bf, df_q_dp, df_c_bf, df_c_dp], ignore_index=True)
    merged_all.to_csv(merged_all_csv, index=False)
    return merged_all


# ============================================================
# RL-QEC
# ============================================================

STABILIZERS = ["XZZXI", "IXZZX", "XIXZZ", "ZXIXZ"]

PAULI_MULT = {
    ("I","I"):"I", ("I","X"):"X", ("I","Y"):"Y", ("I","Z"):"Z",
    ("X","I"):"X", ("X","X"):"I", ("X","Y"):"Z", ("X","Z"):"Y",
    ("Y","I"):"Y", ("Y","X"):"Z", ("Y","Y"):"I", ("Y","Z"):"X",
    ("Z","I"):"Z", ("Z","X"):"Y", ("Z","Y"):"X", ("Z","Z"):"I"
}

def anticommutes(p, q):
    if p == "I" or q == "I":
        return False
    if p == q:
        return False
    return True


def syndrome_for_error(error_paulis):
    syn = []
    for stab in STABILIZERS:
        bit = 0
        for i in range(5):
            if anticommutes(error_paulis[i], stab[i]):
                bit ^= 1
        syn.append(bit)
    return tuple(syn)


def noise_bitflip(p):
    return ["X" if random.random() < p else "I" for _ in range(5)]


def noise_depolarizing(p):
    out = []
    for _ in range(5):
        r = random.random()
        if r < 1 - p:
            out.append("I")
        else:
            rr = random.random()
            out.append("X" if rr < 1/3 else ("Y" if rr < 2/3 else "Z"))
    return out
def noise_biased(p, z_bias=0.8):
    # Simulates realistic hardware: 80% chance of a Z (Phase) error, 20% X (Bit)
    out = []
    for _ in range(5):
        if random.random() < p:
            out.append("Z" if random.random() < z_bias else "X")
        else:
            out.append("I")
    return out

def sample_noise_error(noise_type, p):
    if noise_type == "bitflip": return noise_bitflip(p)
    elif noise_type == "depolarizing": return noise_depolarizing(p)
    elif noise_type == "biased": return noise_biased(p)
    else: return ["I"] * 5

def sample_noise_error(noise_type, p):
    return noise_bitflip(p) if noise_type == "bitflip" else noise_depolarizing(p)


def apply_pauli_correction(error, action):
    return [PAULI_MULT[(a, e)] for a, e in zip(action, error)]


def is_identity(error):
    return all(e == "I" for e in error)


def action_space_20():
    actions = []
    for q in range(5):
        for p in ["I", "X", "Y", "Z"]:
            act = ["I"] * 5
            act[q] = p
            actions.append(tuple(act))
    return actions


class QLearningAgent:
    def __init__(self, actions, alpha=0.2, gamma=0.9, epsilon=0.3, epsilon_min=0.05, epsilon_decay=0.999):
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = defaultdict(float)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        qvals = [self.Q[(state, a)] for a in self.actions]
        return self.actions[int(np.argmax(qvals))]

    def update(self, s, a, r, s2):
        best_next = max(self.Q[(s2, a2)] for a2 in self.actions)
        self.Q[(s, a)] += self.alpha * (r + self.gamma * best_next - self.Q[(s, a)])

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_qlearning_agent(noise_type="bitflip", p=0.05, episodes=50000, seed=42, progress_callback=None):
    random.seed(seed)
    np.random.seed(seed)

    agent = QLearningAgent(actions=action_space_20())
    rewards = []

    # Update progress visually every 1% to avoid spamming the WebSocket
    update_interval = max(1, episodes // 100)

    for ep in range(episodes):
        if progress_callback and ep % update_interval == 0:
            progress_callback(int((ep / episodes) * 100))

        error = sample_noise_error(noise_type, p)
        s = syndrome_for_error(error)
        a = agent.choose_action(s)
        corrected = apply_pauli_correction(error, a)

        success = int(is_identity(corrected))
        r = 1 if success else -1
        s2 = syndrome_for_error(corrected)

        agent.update(s, a, r, s2)
        agent.decay()
        rewards.append(r)

    if progress_callback:
        progress_callback(100)
        
    return agent, rewards


def greedy_policy_from_agent(agent):
    def policy(state):
        qvals = [agent.Q[(state, a)] for a in agent.actions]
        return agent.actions[int(np.argmax(qvals))]
    return policy


def evaluate_logical_error_rates(policy, noise_type="bitflip", p=0.05, trials=20000, seed=123):
    random.seed(seed)
    np.random.seed(seed)

    before_fail = 0
    after_fail = 0

    for _ in range(trials):
        error = sample_noise_error(noise_type, p)

        if not is_identity(error):
            before_fail += 1

        s = syndrome_for_error(error)
        a = policy(s)
        corrected = apply_pauli_correction(error, a)

        if not is_identity(corrected):
            after_fail += 1

    return before_fail / trials, after_fail / trials


def generate_rl_dataset_csv(agent, out_csv, noise_type="bitflip", p=0.05, n_samples=20000, seed=999):
    random.seed(seed)
    np.random.seed(seed)

    rows = []
    actions = agent.actions

    for i in range(n_samples):
        error = sample_noise_error(noise_type, p)
        s = syndrome_for_error(error)

        qvals = [agent.Q[(s, a)] for a in actions]
        a = actions[int(np.argmax(qvals))]

        corrected = apply_pauli_correction(error, a)
        success = int(is_identity(corrected))
        r = 1 if success else -1
        s2 = syndrome_for_error(corrected)

        rows.append({
            "sample_id": i,
            "noise": noise_type,
            "p": p,
            "state_syndrome": "".join(map(str, s)),
            "true_error": "".join(error),
            "action": "".join(a),
            "corrected_error": "".join(corrected),
            "next_syndrome": "".join(map(str, s2)),
            "reward": r,
            "success": success
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df

def get_reward_plot_b64(reward_history, window=1000):
    reward_history = np.array(reward_history, dtype=float)
    if len(reward_history) == 0:
        return ""

    if len(reward_history) < window:
        window = max(10, len(reward_history)//5)

    smooth = np.convolve(reward_history, np.ones(window)/window, mode="valid")

    plt.figure(figsize=(8,3))
    plt.plot(smooth)
    plt.xlabel("Episode")
    plt.ylabel("Smoothed Reward")
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def get_survival_plot_b64(survival):
    plt.figure(figsize=(8,3))
    plt.plot(range(1, len(survival)+1), survival)
    plt.xlabel("Time step")
    plt.ylabel("Survival Probability")
    plt.grid(True)
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")


def get_error_vs_p_plot_b64(df):
    plt.figure(figsize=(8,3))
    plt.plot(df["p"], df["baseline_fail_prob"], marker="o")
    plt.plot(df["p"], df["rl_fail_prob"], marker="o")
    plt.xlabel("Physical error probability p")
    plt.ylabel("Logical failure probability")
    plt.grid(True)
    plt.legend(["Baseline", "RL"])
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")
def get_q_table_heatmap_b64(agent):
    # Generate all 16 possible 4-bit syndromes
    syndromes = [(i, j, k, l) for i in (0,1) for j in (0,1) for k in (0,1) for l in (0,1)]
    actions = agent.actions
    
    q_matrix = np.zeros((len(syndromes), len(actions)))
    for i, s in enumerate(syndromes):
        for j, a in enumerate(actions):
            q_matrix[i, j] = agent.Q[(s, a)]
            
    plt.figure(figsize=(10, 5))
    plt.imshow(q_matrix, cmap="plasma", aspect="auto")
    plt.colorbar(label="Q-Value")
    
    # Format labels for academic readability
    plt.yticks(ticks=np.arange(16), labels=["".join(map(str, s)) for s in syndromes], fontsize=8)
    plt.xticks(ticks=np.arange(20), labels=["".join(a) for a in actions], rotation=45, ha="right", fontsize=8)
    
    plt.xlabel("Action (Pauli Correction)")
    plt.ylabel("Syndrome State (4-bit)")
    plt.title("Learned Q-Table Policy Heatmap")
    plt.tight_layout()
    
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=200)
    plt.close()
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("utf-8")

def baseline_policy(state):
    return ("I","I","I","I","I")


def combine_errors(e1, e2):
    return [PAULI_MULT[(a, b)] for a, b in zip(e1, e2)]


def simulate_lifetime_episode(policy, noise_type="bitflip", p=0.05, T=50, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    current_error = ["I"] * 5

    for t in range(1, T + 1):
        current_error = combine_errors(current_error, sample_noise_error(noise_type, p))
        s = syndrome_for_error(current_error)
        a = policy(s)
        current_error = apply_pauli_correction(current_error, a)

        # fail if still error remains
        if not is_identity(current_error):
            return t

    return T


def compute_survival_curve(policy, noise_type="bitflip", p=0.05, T=50, n_runs=200, seed=123):
    lifetimes = []
    for i in range(n_runs):
        lt = simulate_lifetime_episode(policy, noise_type=noise_type, p=p, T=T, seed=seed+i)
        lifetimes.append(lt)

    lifetimes = np.array(lifetimes, dtype=int)

    survival = []
    for t in range(1, T + 1):
        survival.append(float(np.mean(lifetimes >= t)))

    return np.array(survival, dtype=float), lifetimes


def evaluate_error_rate_vs_p(trained_policy, noise_type="bitflip", p_values=None, T=50, n_runs=120):
    if p_values is None:
        p_values = [0.01, 0.03, 0.05, 0.08, 0.10]

    rows = []
    for p in p_values:
        base_surv, base_lt = compute_survival_curve(baseline_policy, noise_type, p, T, n_runs, seed=1000)
        rl_surv, rl_lt = compute_survival_curve(trained_policy, noise_type, p, T, n_runs, seed=2000)

        rows.append({
            "noise": noise_type,
            "p": p,
            "T": T,
            "runs": n_runs,
            "baseline_fail_prob": 1 - base_surv[-1],
            "rl_fail_prob": 1 - rl_surv[-1],
            "baseline_avg_lifetime": float(np.mean(base_lt)),
            "rl_avg_lifetime": float(np.mean(rl_lt))
        })

    return pd.DataFrame(rows)

def api_run(noise_type="bitflip", p=0.05, shots=500, reps=500, episodes=1000, trials=500, seed=42, progress_callback=None):

    # Skip heavy simulation if already exists
    if not os.path.exists("simulation_results_ALL.csv"):
        results = run_simulation_both_noises(
            p_values=[0.01, 0.03, 0.05],
            shots=shots,
            reps=reps,
            seed=seed
        )
        save_simulation_results_both_noises(results)

    # Train RL with the callback
    agent, rewards = train_qlearning_agent(
        noise_type=noise_type,
        p=p,
        episodes=episodes,
        seed=seed,
        progress_callback=progress_callback
    )

    policy = greedy_policy_from_agent(agent)

    before_rate, after_rate = evaluate_logical_error_rates(
        policy,
        noise_type=noise_type,
        p=p,
        trials=trials,
        seed=seed+99
    )

    rl_csv = f"rl_training_dataset_{noise_type}.csv"
    generate_rl_dataset_csv(
        agent,
        out_csv=rl_csv,
        noise_type=noise_type,
        p=p,
        n_samples=3000,
        seed=seed+999
    )

    reward_b64 = get_reward_plot_b64(rewards)

    survival, _ = compute_survival_curve(
        policy,
        noise_type=noise_type,
        p=p,
        T=40,
        n_runs=80,
        seed=seed+300
    )
    survival_b64 = get_survival_plot_b64(survival)

    df_err = evaluate_error_rate_vs_p(
        policy,
        noise_type=noise_type,
        p_values=[0.01, 0.03, 0.05],
        T=40,
        n_runs=60
    )
    error_vs_p_b64 = get_error_vs_p_plot_b64(df_err)
    heatmap_b64 = get_q_table_heatmap_b64(agent)

    return {
        "noise_type": noise_type,
        "p": float(p),
        "before_rate": float(before_rate),
        "after_rate": float(after_rate),
        "avg_reward": float(np.mean(rewards)),
        "files": ["simulation_results_ALL.csv", rl_csv],
        "charts": {
            "reward": reward_b64,
            "survival": survival_b64,
            "error_vs_p": error_vs_p_b64,
            "heatmap": heatmap_b64
        }
    }


