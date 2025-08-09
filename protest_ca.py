"""
protest_ca.py
Versão: 1.0
Tema: Propagação de protestos / movimentos sociais com agentes carismáticos.


Dependências:
  pip install numpy matplotlib networkx imageio


Descrição resumida:
- Grade 2D toroidal (h,w).
- Ligações de longo alcance tipo small-world (rewiring) implementadas com networkx
  para criar arestas extra entre células (ligações de influência à distância).
- Estados (inteiros):
    0: Neutro (não-engajado)
    1: Ativista (participando do protesto)
    2: Simpatizante (apoia, pode ser convencido a participar)
    3: Repressor (polícia/controle; diminui participação local)
    4: Carismático (líder/influencer: maior poder de ativação)
- Memória curta: cada agente armazena um contador de últimos k passos de exposição.
- Mobilidade: agentes podem trocar de posição com vizinhos (probabilidade p_move).
- Atualização: assíncrona (ordem aleatória por passo) ou síncrona opcional.
- Métricas: fração de ativistas ao longo do tempo, tempo até pico, alcance espacial.
"""

from typing import Tuple, Optional, Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import imageio
import os
import random

# -------------------------
# Utilitários de vizinhança
# -------------------------
def toroidal_index(i, n):
    return i % n

def von_neumann_coords(h, w, i, j):
    return [((i-1)%h,j), ((i+1)%h,j), (i,(j-1)%w), (i,(j+1)%w)]

def moore_coords(h, w, i, j):
    coords = []
    for di in (-1,0,1):
        for dj in (-1,0,1):
            if di==0 and dj==0: continue
            coords.append(((i+di)%h, (j+dj)%w))
    return coords

# -------------------------
# Classe principal
# -------------------------
class ProtestCA:
    def __init__(
        self,
        shape: Tuple[int,int] = (80,80),
        p_sympathetic: float = 0.2,
        n_charismatic: int = 8,
        n_repressors: int = 20,
        rng: Optional[np.random.Generator] = None,
        long_range_k: int = 2,
        rewiring_prob: float = 0.05
    ):
        self.h, self.w = shape
        self.N = self.h * self.w
        self.rng = rng if rng is not None else np.random.default_rng()
        # state grid
        self.state = np.zeros((self.h,self.w), dtype=int)
        # memory: exposure counts in last k steps (sliding window counter)
        self.k_memory = 3
        self.memory = np.zeros((self.h,self.w), dtype=int)
        # delays/hesitation
        self.delay = np.zeros((self.h,self.w), dtype=int)
        # create long-range influence graph (nodes 0..N-1)
        self.G = self._build_small_world(long_range_k, rewiring_prob)
        # init populations
        # some sympathetic (2)
        mask = self.rng.random((self.h,self.w)) < p_sympathetic
        self.state[mask] = 2
        # place repressors (3)
        reps = self.rng.choice(self.N, size=n_repressors, replace=False)
        for idx in reps:
            i,j = divmod(idx, self.w)
            self.state[i,j] = 3
        # place charismatic leaders (4)
        choices = [x for x in range(self.N) if self.state.flat[x]==0 or self.state.flat[x]==2]
        if n_charismatic > len(choices):
            n_charismatic = len(choices)
        chars = self.rng.choice(choices, size=n_charismatic, replace=False)
        for idx in chars:
            i,j = divmod(idx, self.w)
            self.state[i,j] = 4
        # initial seeds: few activistas
        n_seed = max(1, self.N//400)
        seeds = self.rng.choice(self.N, size=n_seed, replace=False)
        for idx in seeds:
            i,j = divmod(idx, self.w)
            # do not overwrite repressors
            if self.state[i,j] != 3:
                self.state[i,j] = 1
        self.time = 0
        # record history
        self.history = []
        # mobility parameters
        self.p_move = 0.0

    def _build_small_world(self, k, p):
        # start from ring lattice over grid flattened order: we'll add random long-range edges
        G = nx.Graph()
        G.add_nodes_from(range(self.N))
        # add k random extra edges per node (approx)
        for node in range(self.N):
            for _ in range(k):
                other = self.rng.integers(0,self.N)
                if other != node:
                    if self.rng.random() < p:
                        G.add_edge(node, other)
        return G

    def coords_to_idx(self, i, j):
        return i * self.w + j

    def idx_to_coords(self, idx):
        return divmod(idx, self.w)

    def neighbors(self, i, j, mode='von_neumann'):
        if mode == 'moore':
            return moore_coords(self.h, self.w, i, j)
        else:
            return von_neumann_coords(self.h, self.w, i, j)

    def long_range_neighbors(self, i, j):
        idx = self.coords_to_idx(i,j)
        return [self.idx_to_coords(nb) for nb in self.G.neighbors(idx)]

    def step(self,
             params: Optional[Dict[str,Any]] = None,
             asynchronous: bool = True,
             neighborhood: str = 'von_neumann'):
        """
        Uma iteração do modelo. Regras principais:
        - Um neutro/simpatizante vira ativista se soma ponderada de vizinhos ativistas/carismáticos
          exceder limiar. Carismáticos têm peso maior.
        - Repressors (3) reduzem chance local de adoção.
        - Memória: agentes que foram expostos em últimos k passos tem maior propensão.
        - Mobilidade: com prob p_move, agente troca de posição com vizinho aleatório.
        """
        if params is None: params = {}
        base_threshold = params.get('threshold', 2)
        charismatic_weight = params.get('char_weight', 3)
        lr_weight = params.get('long_range_weight', 2)
        repressor_penalty = params.get('repressor_penalty', 0.5)
        base_prob = params.get('base_prob', 0.6)
        sympath_prob_bonus = params.get('sympath_prob_bonus', 0.2)
        memory_bonus = params.get('memory_bonus', 0.15)

        coords = [(i,j) for i in range(self.h) for j in range(self.w)]
        if asynchronous:
            random.shuffle(coords)

        new_state = self.state.copy()
        # mobility phase (swap with random neighbor with p_move)
        if self.p_move > 0:
            for i,j in coords:
                if self.rng.random() < self.p_move:
                    neighs = self.neighbors(i,j,neighborhood)
                    ni,nj = random.choice(neighs)
                    # swap states and memories/delays
                    new_state[i,j], new_state[ni,nj] = new_state[ni,nj], new_state[i,j]
                    self.memory[i,j], self.memory[ni,nj] = self.memory[ni,nj], self.memory[i,j]
        # update phase
        for i,j in coords:
            s = new_state[i,j]
            if s == 3:
                # repressor: maybe convert nearby activists to neutral with prob
                for ni,nj in self.neighbors(i,j,neighborhood):
                    if new_state[ni,nj] == 1:
                        if self.rng.random() < 0.02:  # small chance to suppress
                            new_state[ni,nj] = 0
                continue
            # compute local influence
            ln = self.neighbors(i,j,neighborhood) + self.long_range_neighbors(i,j)
            influence = 0.0
            repressor_local = False
            for (ni,nj) in ln:
                st = self.state[ni,nj]  # use previous synchronous state for long-range consistency
                if st == 1:
                    influence += 1.0
                elif st == 4:
                    influence += charismatic_weight
                elif st == 3:
                    repressor_local = True
                # long-range edges counted with lr_weight if from G
                # (we assume long_range_neighbors are extra; for simplicity, already added)
            # apply repressor penalty
            if repressor_local:
                influence *= (1 - repressor_penalty)
            # memory effect
            mem = self.memory[i,j]
            prob = base_prob + (memory_bonus * mem)
            # sympathetic bonus
            if self.state[i,j] == 2:
                prob += sympath_prob_bonus
            # decide transition
            if s in (0,2):  # neutral or sympath
                if influence >= base_threshold and self.rng.random() < min(1.0, prob):
                    new_state[i,j] = 1
                    # increase memory for neighbors (they observed protest)
                    for ni,nj in self.neighbors(i,j,neighborhood):
                        self.memory[ni,nj] = min(self.k_memory, self.memory[ni,nj] + 1)
            elif s == 4:
                # charismatic can become activist directly if exposed or remain
                if influence >= 1 and self.rng.random() < 0.9:
                    new_state[i,j] = 1
            elif s == 1:
                # activist can become neutral over time (burnout)
                if self.rng.random() < 0.01:
                    new_state[i,j] = 0
        # update memory sliding window: decay by 1 each step
        self.memory = np.maximum(0, self.memory - 1)
        self.state = new_state
        self.time += 1
        # record metrics
        frac_activists = np.sum(self.state==1) / self.N
        self.history.append(frac_activists)
        return self.state

    def run(self, steps=200, params=None, asynchronous=True, neighborhood='von_neumann', record_dir:Optional[str]=None):
        frames = []
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)
        for t in range(steps):
            self.step(params=params, asynchronous=asynchronous, neighborhood=neighborhood)
            if record_dir:
                fname = os.path.join(record_dir, f"step_{t:04d}.png")
                self.plot_state(save=fname, title=f"t={self.time}")
                frames.append(fname)
        return frames

    def plot_state(self, save:Optional[str]=None, title:Optional[str]=None):
        cmap = plt.get_cmap('tab10')
        plt.figure(figsize=(6,6))
        plt.imshow(self.state, interpolation='nearest', cmap=cmap, vmin=0, vmax=5)
        plt.xticks([]); plt.yticks([])
        if title: plt.title(title)
        if save:
            plt.savefig(save, bbox_inches='tight', dpi=150)
            plt.close()
        else:
            plt.show()

    def save_gif(self, frames:List[str], out_path='protest_demo.gif', fps=6):
        images = [imageio.imread(f) for f in frames]
        imageio.mimsave(out_path, images, fps=fps)
        return out_path

    def reset_history(self):
        self.history = []

# -------------------------
# Exemplo de uso
# -------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(42)
    ca = ProtestCA(shape=(80,80), p_sympathetic=0.25, n_charismatic=10, n_repressors=30, rng=rng,
                   long_range_k=3, rewiring_prob=0.08)
    ca.p_move = 0.02  # pequena mobilidade
    params = {
        'threshold': 2,
        'char_weight': 4,
        'long_range_weight': 2,
        'repressor_penalty': 0.6,
        'base_prob': 0.5,
        'sympath_prob_bonus': 0.25,
        'memory_bonus': 0.12
    }
    frames = ca.run(steps=150, params=params, asynchronous=True, record_dir="frames_protest")
    ca.save_gif(frames, out_path="protest_sim.gif", fps=8)
    # Plot time series
    import matplotlib.pyplot as plt
    plt.plot(ca.history)
    plt.xlabel("Passos"); plt.ylabel("Fracao de ativistas")
    plt.title("Evolução temporal da participação")
    plt.savefig("participation_timeseries.png", bbox_inches='tight', dpi=150)
    print("Simulação completa. GIF e gráfico salvos.")
