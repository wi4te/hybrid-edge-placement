"""
hybrid_placement.py
===================
Advanced Hybrid Service Placement for Edge Computing

Core Features:
- Multi-Agent RL (MARL) with one agent per edge server region
- Graph Attention Network (GAT) for server scoring
- Dynamic environments: server failures, flash crowds, time-of-day demand
- Curriculum learning with latency-aware reward shaping
- Pretraining-then-finetuning + online learning paradigm
- CSV logging of all routing decisions
- 6-plot Matplotlib dashboard + research-paper-style sensitivity plots

Additional Features:
- EnergyAwareScheduler: green-computing placement mode (minimize Watts)
- SLAMonitor: per-app-type latency SLA violation tracking & reporting
- AdaptiveWeightTuner: feedback-driven objective weight adjustment
- FailoverManager: automatic service migration on server failure
- BenchmarkRunner: random + first-fit greedy baselines for rigorous comparison

Based on:
Paper 1: "Dynamic Service Placement in Edge Computing" (Kazmi et al., 2025)
Paper 2: "GAL-VNE" (Geng et al., 2023)

Requirements: pip install numpy torch matplotlib
Usage: python hybridplacement.py
"""

import numpy as np
import random
import time
import csv
import os
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from enum import Enum
from collections import deque, Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# REPRODUCIBILITY
# ================================================================
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# Check if PyTorch is available
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
    torch.manual_seed(SEED)
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARNING] PyTorch not found. RL agent will use greedy heuristic fallback.")
    print("         Install with: pip install torch")


# ================================================================
# SECTION 1: DATA MODELS
# ================================================================

class AppType(Enum):
    """Application types from Paper 1 (Kazmi et al.)"""
    MMA = "Mobile Medical Analytics"
    LTS = "Language Translation Service"
    PWD = "People With Disabilities AR"
    GENERIC = "Generic Application"


@dataclass
class Service:
    """A single microservice in an application DAG."""
    service_id: int
    cpu_required: float
    ram_required: float
    max_latency_ms: float


@dataclass
class Application:
    """A microservices-based application represented as a chain DAG."""
    app_id: int
    app_type: AppType
    services: List[Service]
    placement_deadline_ms: float
    user_x: float = 0.0
    user_y: float = 0.0

    @property
    def num_services(self):
        return len(self.services)


@dataclass
class EdgeServer:
    """An edge server with location and resources."""
    server_id: int
    x: float
    y: float
    total_cpu: float
    total_ram: float
    allocated_cpu: float = 0.0
    allocated_ram: float = 0.0
    is_failed: bool = False
    region_id: int = 0

    @property
    def available_cpu(self):
        if self.is_failed:
            return 0.0
        return self.total_cpu - self.allocated_cpu

    @property
    def available_ram(self):
        if self.is_failed:
            return 0.0
        return self.total_ram - self.allocated_ram

    @property
    def cpu_utilization(self):
        if self.is_failed or self.total_cpu == 0:
            return 0.0
        return self.allocated_cpu / self.total_cpu

    @property
    def ram_utilization(self):
        if self.is_failed or self.total_ram == 0:
            return 0.0
        return self.allocated_ram / self.total_ram

    @property
    def energy_consumption(self):
        """Linear energy model: idle + proportional to utilization."""
        if self.is_failed:
            return 0.0
        idle_power = 50.0  # Watts
        max_power = 150.0  # Watts
        util = (self.cpu_utilization + self.ram_utilization) / 2.0
        return idle_power + (max_power - idle_power) * util

    def can_host(self, service: Service) -> bool:
        if self.is_failed:
            return False
        return (self.available_cpu >= service.cpu_required and
                self.available_ram >= service.ram_required)


@dataclass
class PlacementSolution:
    """Maps each service to a server with evaluation metrics."""
    mapping: Dict[int, int]
    is_feasible: bool = True
    latency: float = 0.0
    avg_distance: float = 0.0
    num_servers: int = 0
    resource_std: float = 0.0
    fitness: float = float('inf')
    decision_time_ms: float = 0.0


# ================================================================
# SECTION 2: DYNAMIC EDGE NETWORK ENVIRONMENT
# ================================================================

class EdgeNetwork:
    """
    Simulates a grid of edge servers with dynamic events.
    Supports server failures, flash crowds, and time-of-day patterns.
    """

    def __init__(self, grid_rows: int = 4, grid_cols: int = 5,
                 area_km: float = 4.0, cpu_range=(20, 50),
                 ram_range=(16, 64), num_regions: int = 4):
        self.num_servers = grid_rows * grid_cols
        self.area_km = area_km
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.num_regions = num_regions
        self.servers: List[EdgeServer] = []
        self.current_time_step = 0

        # Dynamic event tracking
        self.event_log: List[Dict] = []
        self.utilization_history: List[List[float]] = []
        self.energy_history: List[float] = []

        spacing_x = area_km / grid_cols
        spacing_y = area_km / grid_rows

        for i in range(grid_rows):
            for j in range(grid_cols):
                # Assign regions based on quadrant
                region = 0
                if num_regions == 4:
                    region = (0 if i < grid_rows // 2 else 2) + \
                             (0 if j < grid_cols // 2 else 1)
                elif num_regions == 2:
                    region = 0 if i < grid_rows // 2 else 1
                else:
                    region = (i * grid_cols + j) % num_regions

                server = EdgeServer(
                    server_id=i * grid_cols + j,
                    x=j * spacing_x + spacing_x / 2,
                    y=i * spacing_y + spacing_y / 2,
                    total_cpu=np.random.uniform(*cpu_range),
                    total_ram=np.random.uniform(*ram_range),
                    region_id=region,
                )
                self.servers.append(server)

        self._compute_matrices()

    def _compute_matrices(self):
        """Precompute pairwise distance and latency."""
        n = self.num_servers
        self.distance_matrix = np.zeros((n, n))
        self.latency_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                dx = self.servers[i].x - self.servers[j].x
                dy = self.servers[i].y - self.servers[j].y
                dist = np.sqrt(dx ** 2 + dy ** 2)
                self.distance_matrix[i][j] = dist
                self.latency_matrix[i][j] = 5.0 + 10.0 * dist

    def get_adjacency_matrix(self) -> np.ndarray:
        """Build adjacency matrix for GNN/GAT (Paper 2 style)."""
        threshold = self.area_km / 2.0
        adj = (self.distance_matrix < threshold).astype(np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    def get_latency(self, server_a: int, server_b: int) -> float:
        return self.latency_matrix[server_a][server_b]

    def get_distance(self, server_a: int, server_b: int) -> float:
        return self.distance_matrix[server_a][server_b]

    def distance_to_user(self, server_id: int, user_x: float,
                         user_y: float) -> float:
        s = self.servers[server_id]
        return np.sqrt((s.x - user_x) ** 2 + (s.y - user_y) ** 2)

    def allocate(self, server_id: int, service: Service) -> bool:
        server = self.servers[server_id]
        if not server.can_host(service):
            return False
        server.allocated_cpu += service.cpu_required
        server.allocated_ram += service.ram_required
        return True

    def release(self, server_id: int, service: Service):
        server = self.servers[server_id]
        server.allocated_cpu = max(0, server.allocated_cpu - service.cpu_required)
        server.allocated_ram = max(0, server.allocated_ram - service.ram_required)

    def get_utilization_vector(self) -> np.ndarray:
        return np.array([s.cpu_utilization for s in self.servers])

    def get_resource_std(self) -> float:
        cpu_utils = [s.cpu_utilization for s in self.servers if not s.is_failed]
        ram_utils = [s.ram_utilization for s in self.servers if not s.is_failed]
        if not cpu_utils:
            return 0.0
        return (np.std(cpu_utils) + np.std(ram_utils)) / 2.0

    def get_total_energy(self) -> float:
        return sum(s.energy_consumption for s in self.servers)

    def get_state_vector(self) -> np.ndarray:
        state = []
        for s in self.servers:
            state.append(s.available_cpu / s.total_cpu if s.total_cpu > 0 and not s.is_failed else 0)
            state.append(s.available_ram / s.total_ram if s.total_ram > 0 and not s.is_failed else 0)
        return np.array(state, dtype=np.float32)

    def get_node_features(self) -> np.ndarray:
        """Get per-server feature matrix for GAT [N, feat_dim]."""
        features = []
        for s in self.servers:
            features.append([
                s.available_cpu / max(s.total_cpu, 1),
                s.available_ram / max(s.total_ram, 1),
                s.cpu_utilization,
                s.ram_utilization,
                0.0 if s.is_failed else 1.0,
                s.x / self.area_km,
                s.y / self.area_km,
            ])
        return np.array(features, dtype=np.float32)

    def get_feasible_servers(self, service: Service) -> List[int]:
        return [s.server_id for s in self.servers if s.can_host(service)]

    def get_region_servers(self, region_id: int) -> List[int]:
        return [s.server_id for s in self.servers if s.region_id == region_id]

    def reset(self):
        for s in self.servers:
            s.allocated_cpu = 0.0
            s.allocated_ram = 0.0
            s.is_failed = False
        self.current_time_step = 0

    def snapshot(self) -> List[Tuple[float, float, bool]]:
        return [(s.allocated_cpu, s.allocated_ram, s.is_failed) for s in self.servers]

    def restore(self, snap: List[Tuple[float, float, bool]]):
        for s, (cpu, ram, failed) in zip(self.servers, snap):
            s.allocated_cpu = cpu
            s.allocated_ram = ram
            s.is_failed = failed

    def record_state(self):
        """Record current utilization and energy for dashboard."""
        utils = [s.cpu_utilization for s in self.servers]
        self.utilization_history.append(utils)
        self.energy_history.append(self.get_total_energy())

    # --- Dynamic Environment Events ---

    def advance_time(self, time_step: int, failure_prob: float = 0.03,
                     recovery_prob: float = 0.15,
                     flash_crowd_prob: float = 0.05):
        """Advance simulation time with dynamic events."""
        self.current_time_step = time_step

        # Server failures
        for s in self.servers:
            if not s.is_failed and random.random() < failure_prob:
                s.is_failed = True
                s.allocated_cpu = 0.0
                s.allocated_ram = 0.0
                self.event_log.append({
                    'time': time_step, 'type': 'FAILURE',
                    'server_id': s.server_id
                })
            elif s.is_failed and random.random() < recovery_prob:
                s.is_failed = False
                self.event_log.append({
                    'time': time_step, 'type': 'RECOVERY',
                    'server_id': s.server_id
                })

        # Flash crowd in random region
        if random.random() < flash_crowd_prob:
            target_region = random.randint(0, self.num_regions - 1)
            self.event_log.append({
                'time': time_step, 'type': 'FLASH_CROWD',
                'region': target_region
            })

    def get_time_of_day_multiplier(self, time_step: int) -> float:
        """Sinusoidal demand pattern — peak at midday, low at night."""
        hour = (time_step % 24)
        return 0.5 + 0.5 * np.sin(np.pi * hour / 12.0 - np.pi / 2.0)


# ================================================================
# SECTION 3: OBJECTIVE / FITNESS FUNCTIONS
# ================================================================

class ObjectiveCalculator:
    """
    Computes the 4 placement objectives from Paper 1:
    1. Latency       - sum of inter-service latencies
    2. Distance      - avg distance between used servers
    3. Num servers   - unique servers used (privacy proxy)
    4. Resource std  - std dev of utilization (load balance)
    """

    MAX_LATENCY_MS = 500.0
    MAX_DISTANCE_KM = 6.0
    INFEASIBILITY_PENALTY = 10.0

    @staticmethod
    def compute_latency(mapping: Dict[int, int], app: Application,
                        network: EdgeNetwork) -> float:
        if app.num_services <= 1:
            return 0.0
        total_latency = 0.0
        service_ids = sorted(mapping.keys())
        for i in range(len(service_ids) - 1):
            server_a = mapping[service_ids[i]]
            server_b = mapping[service_ids[i + 1]]
            total_latency += network.get_latency(server_a, server_b)
        return total_latency

    @staticmethod
    def compute_avg_distance(mapping: Dict[int, int],
                             network: EdgeNetwork) -> float:
        servers_used = list(set(mapping.values()))
        if len(servers_used) <= 1:
            return 0.0
        total_dist = 0.0
        count = 0
        for i in range(len(servers_used)):
            for j in range(i + 1, len(servers_used)):
                total_dist += network.get_distance(servers_used[i],
                                                    servers_used[j])
                count += 1
        return total_dist / count if count > 0 else 0.0

    @staticmethod
    def compute_num_servers(mapping: Dict[int, int]) -> int:
        return len(set(mapping.values()))

    @staticmethod
    def compute_resource_std(network: EdgeNetwork) -> float:
        return network.get_resource_std()

    @staticmethod
    def check_feasibility(mapping: Dict[int, int], app: Application,
                          network: EdgeNetwork) -> bool:
        server_load = {}
        for svc in app.services:
            sid = mapping[svc.service_id]
            if sid not in server_load:
                server_load[sid] = {'cpu': 0.0, 'ram': 0.0}
            server_load[sid]['cpu'] += svc.cpu_required
            server_load[sid]['ram'] += svc.ram_required

        for sid, load in server_load.items():
            server = network.servers[sid]
            if (server.available_cpu < load['cpu'] or
                    server.available_ram < load['ram']):
                return False
        return True

    @classmethod
    def compute_fitness(cls, mapping: Dict[int, int], app: Application,
                        network: EdgeNetwork, weights: Dict[str, float],
                        check_feasible: bool = True) -> Tuple[float, dict]:
        feasible = cls.check_feasibility(mapping, app, network)
        if check_feasible and not feasible:
            return cls.INFEASIBILITY_PENALTY, {
                'feasible': False, 'latency': 0, 'distance': 0,
                'num_servers': 0, 'resource_std': 0, 'fitness': cls.INFEASIBILITY_PENALTY
            }

        latency = cls.compute_latency(mapping, app, network)
        distance = cls.compute_avg_distance(mapping, network)
        num_servers = cls.compute_num_servers(mapping)
        resource_std = cls.compute_resource_std(network)

        norm_latency = min(latency / cls.MAX_LATENCY_MS, 1.0)
        norm_distance = min(distance / cls.MAX_DISTANCE_KM, 1.0)
        norm_servers = num_servers / max(app.num_services, 1)
        norm_resource = min(resource_std / 0.5, 1.0)

        fitness = (weights.get('latency', 0.25) * norm_latency +
                   weights.get('distance', 0.25) * norm_distance +
                   weights.get('num_servers', 0.25) * norm_servers +
                   weights.get('resource_std', 0.25) * norm_resource)

        metrics = {
            'feasible': feasible,
            'latency': latency,
            'distance': distance,
            'num_servers': num_servers,
            'resource_std': resource_std,
            'fitness': fitness
        }
        return fitness, metrics


# ================================================================
# SECTION 4: GENETIC ALGORITHM — Privacy-Optimized
# ================================================================

class GeneticAlgorithm:
    """
    GA for service placement, optimized for PRIVACY.
    Chromosome: [server_id_for_svc_0, server_id_for_svc_1, ...]
    """

    WEIGHTS = {
        'latency': 0.15,
        'distance': 0.35,
        'num_servers': 0.35,
        'resource_std': 0.15
    }

    def __init__(self, population_size: int = 60, generations: int = 40,
                 crossover_rate: float = 0.85, mutation_rate: float = 0.05,
                 tournament_size: int = 5):
        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size

    def _initialize_population(self, app: Application,
                                network: EdgeNetwork) -> List[List[int]]:
        population = []
        num_svc = app.num_services
        num_srv = network.num_servers

        for i in range(self.population_size):
            if i < self.population_size // 3:
                distances = [network.distance_to_user(s, app.user_x, app.user_y)
                             for s in range(num_srv)]
                probs = 1.0 / (np.array(distances) + 0.1)
                probs /= probs.sum()
                chromosome = list(np.random.choice(num_srv, size=num_svc, p=probs))
            elif i < 2 * self.population_size // 3:
                num_servers_to_use = np.random.randint(1, min(4, num_srv + 1))
                chosen_servers = random.sample(range(num_srv), num_servers_to_use)
                chromosome = [random.choice(chosen_servers) for _ in range(num_svc)]
            else:
                chromosome = [random.randint(0, num_srv - 1) for _ in range(num_svc)]
            population.append(chromosome)
        return population

    def _evaluate(self, chromosome: List[int], app: Application,
                  network: EdgeNetwork) -> float:
        mapping = {app.services[i].service_id: chromosome[i]
                   for i in range(app.num_services)}
        fitness, _ = ObjectiveCalculator.compute_fitness(
            mapping, app, network, self.WEIGHTS
        )
        return fitness

    def _tournament_select(self, population: List[List[int]],
                           fitnesses: List[float]) -> List[int]:
        indices = random.sample(range(len(population)),
                                min(self.tournament_size, len(population)))
        best_idx = min(indices, key=lambda i: fitnesses[i])
        return population[best_idx].copy()

    def _crossover(self, parent1: List[int],
                   parent2: List[int]) -> Tuple[List[int], List[int]]:
        if random.random() > self.crossover_rate or len(parent1) <= 1:
            return parent1.copy(), parent2.copy()
        point = random.randint(1, len(parent1) - 1)
        child1 = parent1[:point] + parent2[point:]
        child2 = parent2[:point] + parent1[point:]
        return child1, child2

    def _mutate(self, chromosome: List[int], num_servers: int) -> List[int]:
        mutated = chromosome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mutation_rate:
                if random.random() < 0.6:
                    mutated[i] = random.choice(mutated)
                else:
                    mutated[i] = random.randint(0, num_servers - 1)
        return mutated

    def solve(self, app: Application,
              network: EdgeNetwork) -> PlacementSolution:
        start_time = time.time()

        population = self._initialize_population(app, network)
        fitnesses = [self._evaluate(ch, app, network) for ch in population]

        best_idx = int(np.argmin(fitnesses))
        best_chromosome = population[best_idx].copy()
        best_fitness = fitnesses[best_idx]

        for gen in range(self.generations):
            new_population = []
            sorted_indices = list(np.argsort(fitnesses))
            new_population.append(population[sorted_indices[0]].copy())
            new_population.append(population[sorted_indices[1]].copy())

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitnesses)
                parent2 = self._tournament_select(population, fitnesses)
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1, network.num_servers)
                child2 = self._mutate(child2, network.num_servers)
                new_population.extend([child1, child2])

            population = new_population[:self.population_size]
            fitnesses = [self._evaluate(ch, app, network) for ch in population]

            gen_best_idx = int(np.argmin(fitnesses))
            if fitnesses[gen_best_idx] < best_fitness:
                best_fitness = fitnesses[gen_best_idx]
                best_chromosome = population[gen_best_idx].copy()

        mapping = {app.services[i].service_id: best_chromosome[i]
                   for i in range(app.num_services)}
        _, metrics = ObjectiveCalculator.compute_fitness(
            mapping, app, network, self.WEIGHTS
        )
        decision_time = (time.time() - start_time) * 1000

        return PlacementSolution(
            mapping=mapping,
            is_feasible=metrics['feasible'],
            latency=metrics['latency'],
            avg_distance=metrics['distance'],
            num_servers=metrics['num_servers'],
            resource_std=metrics['resource_std'],
            fitness=best_fitness,
            decision_time_ms=decision_time
        )


# ================================================================
# SECTION 5: PSO — Latency-Optimized
# ================================================================

class ParticleSwarmOptimization:
    """
    PSO for service placement, optimized for LATENCY.
    Continuous relaxation with discretization for server IDs.
    """

    WEIGHTS = {
        'latency': 0.50,
        'distance': 0.15,
        'num_servers': 0.10,
        'resource_std': 0.25
    }

    def __init__(self, num_particles: int = 40, max_iterations: int = 40,
                 w: float = 0.729, c1: float = 1.49, c2: float = 1.49):
        self.num_particles = num_particles
        self.max_iterations = max_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2

    def _discretize(self, position: np.ndarray,
                    num_servers: int) -> List[int]:
        discrete = np.floor(np.abs(position) % num_servers).astype(int)
        return discrete.tolist()

    def _evaluate(self, position: np.ndarray, app: Application,
                  network: EdgeNetwork) -> float:
        chromosome = self._discretize(position, network.num_servers)
        mapping = {app.services[i].service_id: chromosome[i]
                   for i in range(app.num_services)}
        fitness, _ = ObjectiveCalculator.compute_fitness(
            mapping, app, network, self.WEIGHTS
        )
        return fitness

    def solve(self, app: Application,
              network: EdgeNetwork) -> PlacementSolution:
        start_time = time.time()
        dim = app.num_services
        num_srv = network.num_servers

        positions = np.random.uniform(0, num_srv,
                                       (self.num_particles, dim))
        velocities = np.random.uniform(-num_srv / 4, num_srv / 4,
                                        (self.num_particles, dim))

        p_best_positions = positions.copy()
        p_best_fitnesses = np.array([
            self._evaluate(p, app, network) for p in positions
        ])

        g_best_idx = int(np.argmin(p_best_fitnesses))
        g_best_position = p_best_positions[g_best_idx].copy()
        g_best_fitness = p_best_fitnesses[g_best_idx]

        for iteration in range(self.max_iterations):
            for i in range(self.num_particles):
                r1 = np.random.random(dim)
                r2 = np.random.random(dim)

                cognitive = self.c1 * r1 * (p_best_positions[i] - positions[i])
                social = self.c2 * r2 * (g_best_position - positions[i])
                velocities[i] = self.w * velocities[i] + cognitive + social

                max_vel = num_srv / 2
                velocities[i] = np.clip(velocities[i], -max_vel, max_vel)
                positions[i] += velocities[i]

                fitness = self._evaluate(positions[i], app, network)

                if fitness < p_best_fitnesses[i]:
                    p_best_fitnesses[i] = fitness
                    p_best_positions[i] = positions[i].copy()

                if fitness < g_best_fitness:
                    g_best_fitness = fitness
                    g_best_position = positions[i].copy()

        best_chromosome = self._discretize(g_best_position, num_srv)
        mapping = {app.services[i].service_id: best_chromosome[i]
                   for i in range(app.num_services)}
        _, metrics = ObjectiveCalculator.compute_fitness(
            mapping, app, network, self.WEIGHTS
        )
        decision_time = (time.time() - start_time) * 1000

        return PlacementSolution(
            mapping=mapping,
            is_feasible=metrics['feasible'],
            latency=metrics['latency'],
            avg_distance=metrics['distance'],
            num_servers=metrics['num_servers'],
            resource_std=metrics['resource_std'],
            fitness=g_best_fitness,
            decision_time_ms=decision_time
        )

# ================================================================
# SECTION 6: GAT-BASED Q-NETWORK (Paper 2 — GAL-VNE inspired)
# ================================================================

if TORCH_AVAILABLE:
    class GATLayer(nn.Module):
        """
        Graph Attention Network layer (Velickovic et al., 2018).
        Replaces GraphSAGE from Paper 2 for better feature aggregation.
        """
        def __init__(self, in_features: int, out_features: int,
                     num_heads: int = 4, concat: bool = True,
                     dropout: float = 0.1):
            super().__init__()
            self.num_heads = num_heads
            self.concat = concat
            self.out_per_head = out_features // num_heads if concat else out_features

            self.W = nn.Linear(in_features, self.out_per_head * num_heads, bias=False)
            self.a_src = nn.Parameter(torch.zeros(num_heads, self.out_per_head))
            self.a_dst = nn.Parameter(torch.zeros(num_heads, self.out_per_head))
            nn.init.xavier_uniform_(self.a_src.data.unsqueeze(0))
            nn.init.xavier_uniform_(self.a_dst.data.unsqueeze(0))
            self.leaky_relu = nn.LeakyReLU(0.2)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, adj):
            """
            x:   [N, in_features]
            adj: [N, N] adjacency matrix
            """
            N = x.size(0)
            # h: [N, num_heads, out_per_head]
            h = self.W(x).view(N, self.num_heads, self.out_per_head)

            # Attention scores per head: each is [N, num_heads]
            attn_src = (h * self.a_src.unsqueeze(0)).sum(dim=-1)  # [N, num_heads]
            attn_dst = (h * self.a_dst.unsqueeze(0)).sum(dim=-1)  # [N, num_heads]

            # Reshape to [num_heads, N, 1] and [num_heads, 1, N] for proper broadcast
            attn_src = attn_src.permute(1, 0).unsqueeze(2)   # [num_heads, N, 1]
            attn_dst = attn_dst.permute(1, 0).unsqueeze(1)   # [num_heads, 1, N]
            attn = attn_src + attn_dst                        # [num_heads, N, N]
            attn = self.leaky_relu(attn)

            # Mask: expand adj [N,N] -> [num_heads, N, N]
            mask = adj.unsqueeze(0).expand(self.num_heads, -1, -1)  # [num_heads, N, N]
            attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = F.softmax(attn, dim=2)   # softmax over source dimension
            attn = self.dropout(attn)

            # Aggregate: h_permuted [num_heads, N, out_per_head]
            h_permuted = h.permute(1, 0, 2)                  # [num_heads, N, out_per_head]
            out = torch.bmm(attn, h_permuted)                # [num_heads, N, out_per_head]
            out = out.permute(1, 0, 2)                        # [N, num_heads, out_per_head]

            if self.concat:
                return out.reshape(N, -1)                     # [N, num_heads * out_per_head]
            else:
                return out.mean(dim=1)                        # [N, out_per_head]


    class GATQNetwork(nn.Module):
        """
        GAT-based Q-Network for service placement.
        Uses graph attention to aggregate neighbor server features
        before computing Q-values (inspired by Paper 2's GNN approach).
        """
        def __init__(self, node_feat_dim: int = 7, num_nodes: int = 20,
                     service_feat_dim: int = 3, hidden_dim: int = 64,
                     num_heads: int = 4):
            super().__init__()
            self.num_nodes = num_nodes
            gat_out = hidden_dim

            self.gat1 = GATLayer(node_feat_dim, gat_out, num_heads=num_heads, concat=True)
            self.gat2 = GATLayer(gat_out, gat_out, num_heads=num_heads, concat=False)

            self.service_encoder = nn.Sequential(
                nn.Linear(service_feat_dim, 32),
                nn.ReLU(),
            )

            self.q_head = nn.Sequential(
                nn.Linear(gat_out + 32, 64),
                nn.ReLU(),
                nn.Linear(64, 1)
            )

        def forward(self, node_features, adj, service_features):
            """
            node_features:    [N, node_feat_dim]
            adj:              [N, N]
            service_features: [service_feat_dim]
            Returns: Q-values [N] (one per server)
            """
            gat_out = F.elu(self.gat1(node_features, adj))
            gat_out = F.elu(self.gat2(gat_out, adj))

            svc_enc = self.service_encoder(service_features)
            svc_expanded = svc_enc.unsqueeze(0).expand(self.num_nodes, -1)

            combined = torch.cat([gat_out, svc_expanded], dim=-1)
            q_values = self.q_head(combined).squeeze(-1)
            return q_values


# ================================================================
# SECTION 7: REPLAY BUFFER
# ================================================================

class ReplayBuffer:
    """Experience replay buffer for DQN training."""
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        states, actions, rewards, next_states, dones = zip(*batch)
        return (list(states), list(actions), list(rewards),
                list(next_states), list(dones))

    def __len__(self):
        return len(self.buffer)


# ================================================================
# SECTION 8: MULTI-AGENT RL WITH GAT + CURRICULUM LEARNING
# ================================================================

class MARLPlacementAgent:
    """
    Multi-Agent RL with GAT-based Q-Networks.

    Architecture (based on Paper 2's GNN + Paper 1's RL approach):
    - One DQN agent per edge server region (cooperative MARL)
    - GAT-based state encoder replaces flat MLP
    - Curriculum learning: start with 1-service apps, increase complexity
    - Latency-aware reward shaping
    - Pretraining-then-finetuning paradigm with online learning
    """

    WEIGHTS = {
        'latency': 0.20,
        'distance': 0.15,
        'num_servers': 0.15,
        'resource_std': 0.50
    }

    def __init__(self, network: EdgeNetwork, learning_rate: float = 1e-3,
                 gamma: float = 0.95, epsilon_start: float = 1.0,
                 epsilon_end: float = 0.05, epsilon_decay: float = 0.995,
                 batch_size: int = 32):
        self.network = network
        self.num_servers = network.num_servers
        self.num_regions = network.num_regions
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.trained = False
        self.online_learning = False

        self.node_feat_dim = 7
        self.service_feat_dim = 3

        # Per-region agents (MARL)
        self.agents = {}
        self.target_agents = {}
        self.optimizers = {}
        self.replay_buffers = {}

        if TORCH_AVAILABLE:
            for region_id in range(self.num_regions):
                q_net = GATQNetwork(
                    node_feat_dim=self.node_feat_dim,
                    num_nodes=self.num_servers,
                    service_feat_dim=self.service_feat_dim,
                    hidden_dim=64, num_heads=4
                )
                target_net = GATQNetwork(
                    node_feat_dim=self.node_feat_dim,
                    num_nodes=self.num_servers,
                    service_feat_dim=self.service_feat_dim,
                    hidden_dim=64, num_heads=4
                )
                target_net.load_state_dict(q_net.state_dict())

                self.agents[region_id] = q_net
                self.target_agents[region_id] = target_net
                self.optimizers[region_id] = optim.Adam(q_net.parameters(), lr=learning_rate)
                self.replay_buffers[region_id] = ReplayBuffer(capacity=5000)

        self.training_steps = 0
        self.adj_matrix = None

    def _get_adj_tensor(self, network: EdgeNetwork):
        if self.adj_matrix is None:
            adj = network.get_adjacency_matrix()
            self.adj_matrix = torch.FloatTensor(adj) if TORCH_AVAILABLE else adj
        return self.adj_matrix

    def _get_service_features(self, service: Service, remaining: int,
                               total: int) -> np.ndarray:
        return np.array([
            service.cpu_required / 50.0,
            service.ram_required / 64.0,
            remaining / max(total, 1)
        ], dtype=np.float32)

    def _select_action_greedy(self, network: EdgeNetwork,
                               service: Service) -> int:
        """Greedy heuristic fallback: pick server closest to mean utilization."""
        feasible = network.get_feasible_servers(service)
        if not feasible:
            non_failed = [s.server_id for s in network.servers if not s.is_failed]
            return random.choice(non_failed) if non_failed else 0

        utils = network.get_utilization_vector()
        mean_util = np.mean(utils)
        best_server = feasible[0]
        best_score = float('inf')
        for sid in feasible:
            new_util = ((network.servers[sid].allocated_cpu + service.cpu_required)
                        / network.servers[sid].total_cpu)
            score = abs(new_util - mean_util)
            if score < best_score:
                best_score = score
                best_server = sid
        return best_server

    def _get_region_for_service(self, network: EdgeNetwork,
                                 service: Service) -> int:
        """Determine which regional agent handles this service."""
        feasible = network.get_feasible_servers(service)
        if not feasible:
            return 0
        region_counts = Counter(network.servers[s].region_id for s in feasible)
        return region_counts.most_common(1)[0][0]

    def select_action(self, network: EdgeNetwork, service: Service,
                      remaining: int, total: int,
                      training: bool = False) -> int:
        feasible = network.get_feasible_servers(service)
        if not feasible:
            return self._select_action_greedy(network, service)

        if not TORCH_AVAILABLE or (not self.trained and not training):
            return self._select_action_greedy(network, service)

        if training and random.random() < self.epsilon:
            return random.choice(feasible)

        region_id = self._get_region_for_service(network, service)
        q_net = self.agents[region_id]

        with torch.no_grad():
            node_feats = torch.FloatTensor(network.get_node_features())
            adj = self._get_adj_tensor(network)
            svc_feats = torch.FloatTensor(
                self._get_service_features(service, remaining, total)
            )
            q_values = q_net(node_feats, adj, svc_feats)

            mask = torch.full((self.num_servers,), float('-inf'))
            for s in feasible:
                mask[s] = 0.0
            masked_q = q_values + mask
            return masked_q.argmax().item()

    def _compute_shaped_reward(self, network: EdgeNetwork, service: Service,
                                action: int, success: bool,
                                initial_std: float,
                                app: Application, mapping: Dict) -> float:
        """
        Latency-aware reward shaping (addresses Paper 1's RL limitations).
        """
        if not success:
            return -2.0

        # Load-balancing reward
        new_std = network.get_resource_std()
        balance_reward = (initial_std - new_std) * 5.0

        # Latency penalty
        latency_penalty = 0.0
        placed_svc_ids = sorted(mapping.keys())
        if len(placed_svc_ids) >= 2:
            prev_svc = placed_svc_ids[-2]
            prev_server = mapping[prev_svc]
            lat = network.get_latency(prev_server, action)
            latency_penalty = -0.01 * lat

        # Distance to user reward
        user_dist = network.distance_to_user(action, app.user_x, app.user_y)
        dist_reward = -0.05 * user_dist

        # Energy efficiency reward
        energy_penalty = -0.001 * network.servers[action].energy_consumption

        return 0.2 + balance_reward + latency_penalty + dist_reward + energy_penalty

    def train_step(self, region_id: int):
        """Train a single regional agent."""
        if not TORCH_AVAILABLE:
            return
        buf = self.replay_buffers[region_id]
        if len(buf) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = buf.sample(self.batch_size)

        q_net = self.agents[region_id]
        target_net = self.target_agents[region_id]
        optimizer = self.optimizers[region_id]

        # Compute current Q-values
        current_qs = []
        target_qs = []
        for i in range(len(states)):
            s = states[i]
            ns = next_states[i]

            node_f = torch.FloatTensor(s['node_features'])
            adj = torch.FloatTensor(s['adj'])
            svc_f = torch.FloatTensor(s['service_features'])

            q_vals = q_net(node_f, adj, svc_f)
            current_qs.append(q_vals[actions[i]])

            with torch.no_grad():
                n_node_f = torch.FloatTensor(ns['node_features'])
                n_adj = torch.FloatTensor(ns['adj'])
                n_svc_f = torch.FloatTensor(ns['service_features'])

                next_q = target_net(n_node_f, n_adj, n_svc_f)
                best_next = next_q.max()
                target = rewards[i] + self.gamma * best_next * (1 - dones[i])
                target_qs.append(target)

        current_q_tensor = torch.stack(current_qs)
        target_q_tensor = torch.stack(target_qs)

        loss = F.mse_loss(current_q_tensor, target_q_tensor)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(q_net.parameters(), 1.0)
        optimizer.step()

        self.training_steps += 1
        if self.training_steps % 50 == 0:
            target_net.load_state_dict(q_net.state_dict())

    def solve(self, app: Application, network: EdgeNetwork,
              training: bool = False) -> PlacementSolution:
        start_time = time.time()
        mapping = {}
        initial_std = network.get_resource_std()
        adj_np = network.get_adjacency_matrix()

        for idx, service in enumerate(app.services):
            remaining = app.num_services - idx

            action = self.select_action(network, service, remaining,
                                         app.num_services, training)
            success = network.allocate(action, service)
            mapping[service.service_id] = action

            reward = self._compute_shaped_reward(
                network, service, action, success, initial_std, app, mapping
            )
            initial_std = network.get_resource_std()

            region_id = self._get_region_for_service(network, service)

            state_dict = {
                'node_features': network.get_node_features().copy(),
                'adj': adj_np.copy(),
                'service_features': self._get_service_features(
                    service, remaining, app.num_services
                ).copy()
            }

            if idx < app.num_services - 1:
                next_svc = app.services[idx + 1]
                next_state_dict = {
                    'node_features': network.get_node_features().copy(),
                    'adj': adj_np.copy(),
                    'service_features': self._get_service_features(
                        next_svc, remaining - 1, app.num_services
                    ).copy()
                }
                done = 0
            else:
                next_state_dict = state_dict
                done = 1

            if training or self.online_learning:
                if TORCH_AVAILABLE:
                    self.replay_buffers[region_id].push(
                        state_dict, action, reward, next_state_dict, done
                    )
                    self.train_step(region_id)

        # Release resources
        for service in app.services:
            network.release(mapping[service.service_id], service)

        _, metrics = ObjectiveCalculator.compute_fitness(
            mapping, app, network, self.WEIGHTS
        )
        decision_time = (time.time() - start_time) * 1000

        return PlacementSolution(
            mapping=mapping,
            is_feasible=metrics['feasible'],
            latency=metrics['latency'],
            avg_distance=metrics['distance'],
            num_servers=metrics['num_servers'],
            resource_std=metrics['resource_std'],
            fitness=metrics['fitness'],
            decision_time_ms=decision_time
        )

    def pretrain(self, network: EdgeNetwork, num_episodes: int = 300,
                 apps_per_episode: int = 30):
        """
        Pretraining phase with curriculum learning.
        Stage 1 (episodes 0-99):   1-2 service apps only
        Stage 2 (episodes 100-199): 1-4 service apps
        Stage 3 (episodes 200+):    full complexity (1-8 services)
        """
        print(f"    Pretraining MARL ({self.num_regions} regional agents) "
              f"with curriculum learning...")

        for episode in range(num_episodes):
            network.reset()

            # Curriculum: gradually increase service complexity
            if episode < num_episodes // 3:
                svc_range = (1, 3)
                stage = 1
            elif episode < 2 * num_episodes // 3:
                svc_range = (1, 5)
                stage = 2
            else:
                svc_range = (1, 9)
                stage = 3

            for _ in range(apps_per_episode):
                app = WorkloadGenerator.generate_single_app(
                    app_id=0, app_type=AppType.GENERIC,
                    area_km=network.area_km,
                    services_override=svc_range
                )
                solution = self.solve(app, network, training=True)
                for svc in app.services:
                    network.allocate(solution.mapping[svc.service_id], svc)

            self.epsilon = max(self.epsilon_end,
                               self.epsilon * self.epsilon_decay)

            if (episode + 1) % 50 == 0:
                print(f"      Episode {episode + 1}/{num_episodes} "
                      f"[Stage {stage}], Epsilon: {self.epsilon:.3f}, "
                      f"Resource Std: {network.get_resource_std():.4f}")

        self.trained = True
        print("    MARL pretraining complete!")

    def finetune(self, network: EdgeNetwork, num_episodes: int = 100,
                 apps_per_episode: int = 20):
        """
        Finetuning phase: train on dynamic environment with failures.
        """
        print(f"    Finetuning MARL on dynamic environment...")
        self.epsilon = max(0.3, self.epsilon)

        for episode in range(num_episodes):
            network.reset()
            for t in range(apps_per_episode):
                network.advance_time(t, failure_prob=0.05,
                                      recovery_prob=0.2)
                app = WorkloadGenerator.generate_single_app(
                    app_id=0, app_type=AppType.GENERIC,
                    area_km=network.area_km
                )
                solution = self.solve(app, network, training=True)
                for svc in app.services:
                    network.allocate(solution.mapping[svc.service_id], svc)

            self.epsilon = max(self.epsilon_end,
                               self.epsilon * self.epsilon_decay)

            if (episode + 1) % 25 == 0:
                print(f"      Finetune episode {episode + 1}/{num_episodes}, "
                      f"Epsilon: {self.epsilon:.3f}")

        print("    MARL finetuning complete!")

    def enable_online_learning(self):
        """Enable online learning during deployment."""
        self.online_learning = True
        self.epsilon = max(0.1, self.epsilon_end)
        print("    Online learning ENABLED for MARL agent.")


# ================================================================
# SECTION 9: WORKLOAD GENERATOR
# ================================================================

class WorkloadGenerator:
    """Generate realistic application workloads based on Paper 1."""

    PROFILES = {
        AppType.MMA: {
            'services_range': (3, 6),
            'cpu_range': (1, 4),
            'ram_range': (1, 4),
            'latency_range': (20, 50),
            'deadline_ms': 200,
        },
        AppType.LTS: {
            'services_range': (3, 7),
            'cpu_range': (2, 6),
            'ram_range': (2, 6),
            'latency_range': (30, 80),
            'deadline_ms': 500,
        },
        AppType.PWD: {
            'services_range': (4, 8),
            'cpu_range': (4, 10),
            'ram_range': (4, 10),
            'latency_range': (40, 100),
            'deadline_ms': 400,
        },
        AppType.GENERIC: {
            'services_range': (1, 8),
            'cpu_range': (1, 8),
            'ram_range': (1, 8),
            'latency_range': (30, 100),
            'deadline_ms': 600,
        },
    }

    @classmethod
    def generate_single_app(cls, app_id: int, app_type: AppType,
                             area_km: float,
                             services_override: Tuple[int, int] = None) -> Application:
        profile = cls.PROFILES[app_type]
        
        svc_range = services_override if services_override else profile['services_range']
        num_services = np.random.randint(*svc_range)

        services = []
        for i in range(num_services):
            services.append(Service(
                service_id=i,
                cpu_required=np.random.uniform(*profile['cpu_range']),
                ram_required=np.random.uniform(*profile['ram_range']),
                max_latency_ms=np.random.uniform(*profile['latency_range']),
            ))

        return Application(
            app_id=app_id,
            app_type=app_type,
            services=services,
            placement_deadline_ms=profile['deadline_ms'],
            user_x=np.random.uniform(0, area_km),
            user_y=np.random.uniform(0, area_km),
        )

    @classmethod
    def generate_workload(cls, num_requests: int,
                          area_km: float) -> List[Application]:
        """
        Generate realistic workload mix from Paper 1:
        50% Generic, 30% LTS, 10% PWD, 10% MMA
        """
        workload = []
        type_distribution = [
            (AppType.GENERIC, 0.50),
            (AppType.LTS, 0.30),
            (AppType.MMA, 0.10),
            (AppType.PWD, 0.10),
        ]

        for i in range(num_requests):
            r = random.random()
            cumulative = 0
            for app_type, prob in type_distribution:
                cumulative += prob
                if r <= cumulative:
                    app = cls.generate_single_app(i, app_type, area_km)
                    workload.append(app)
                    break

        return workload


# ================================================================
# SECTION 10: CSV DECISION LOGGER
# ================================================================

class DecisionLogger:
    """Log every routing decision to CSV for offline analysis."""

    def __init__(self, filepath: str = "routing_decisions.csv"):
        self.filepath = filepath
        self.rows = []
        self.fieldnames = [
            'timestamp', 'time_step', 'app_id', 'app_type',
            'num_services', 'algorithm_used', 'is_feasible',
            'latency_ms', 'avg_distance_km', 'num_servers_used',
            'resource_std', 'fitness', 'decision_time_ms',
            'dynamic_event'
        ]

    def log(self, time_step: int, app: Application, algorithm: str,
            solution: PlacementSolution, event: str = ""):
        self.rows.append({
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'time_step': time_step,
            'app_id': app.app_id,
            'app_type': app.app_type.name,
            'num_services': app.num_services,
            'algorithm_used': algorithm,
            'is_feasible': solution.is_feasible,
            'latency_ms': round(solution.latency, 2),
            'avg_distance_km': round(solution.avg_distance, 4),
            'num_servers_used': solution.num_servers,
            'resource_std': round(solution.resource_std, 4),
            'fitness': round(solution.fitness, 4),
            'decision_time_ms': round(solution.decision_time_ms, 2),
            'dynamic_event': event,
        })

    def save(self):
        with open(self.filepath, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"    CSV log saved to: {self.filepath} ({len(self.rows)} entries)")


# ================================================================
# SECTION 11: MATPLOTLIB DASHBOARD VISUALIZER
# ================================================================

class DashboardVisualizer:
    """Generate a 6-plot analytics dashboard."""

    @staticmethod
    def generate_dashboard(all_results: List[Dict], network: EdgeNetwork,
                           event_log: List[Dict],
                           save_path: str = "dashboard.png"):
        plt.style.use('seaborn-v0_8-darkgrid' if 'seaborn-v0_8-darkgrid'
                       in plt.style.available else 'ggplot')

        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle('Advanced Hybrid Edge Placement — Analytics Dashboard',
                     fontsize=16, fontweight='bold', y=0.98)

        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6']

        # --- Plot 1: Server Utilization Over Time ---
        ax1 = axes[0, 0]
        if network.utilization_history:
            history = np.array(network.utilization_history)
            avg_util = history.mean(axis=1)
            max_util = history.max(axis=1)
            min_util = history.min(axis=1)
            steps = range(len(avg_util))

            ax1.fill_between(steps, min_util, max_util,
                             alpha=0.2, color=colors[2])
            ax1.plot(steps, avg_util, color=colors[2], linewidth=2,
                     label='Mean Utilization')
            ax1.plot(steps, max_util, color=colors[1], linewidth=1,
                     linestyle='--', alpha=0.7, label='Max')
            ax1.plot(steps, min_util, color=colors[0], linewidth=1,
                     linestyle='--', alpha=0.7, label='Min')
            ax1.set_title('Server CPU Utilization Over Time')
            ax1.set_xlabel('Time Step')
            ax1.set_ylabel('Utilization')
            ax1.legend(fontsize=8)
        else:
            ax1.text(0.5, 0.5, 'No utilization data', ha='center',
                     va='center', transform=ax1.transAxes)
            ax1.set_title('Server CPU Utilization Over Time')

        # --- Plot 2: Energy Consumption ---
        ax2 = axes[0, 1]
        if network.energy_history:
            steps = range(len(network.energy_history))
            ax2.fill_between(steps, 0, network.energy_history,
                             alpha=0.4, color=colors[3])
            ax2.plot(steps, network.energy_history, color=colors[3],
                     linewidth=2)
            ax2.set_title('Total Energy Consumption Over Time')
            ax2.set_xlabel('Time Step')
            ax2.set_ylabel('Energy (Watts)')
        else:
            ax2.text(0.5, 0.5, 'No energy data', ha='center',
                     va='center', transform=ax2.transAxes)
            ax2.set_title('Energy Consumption')

        # --- Plot 3: Algorithm Comparison Bar Chart ---
        ax3 = axes[0, 2]
        algo_names = [r['algorithm'] for r in all_results]
        success_rates = [r['success_rate'] * 100 for r in all_results]
        bar_colors = colors[:len(algo_names)]
        bars = ax3.bar(algo_names, success_rates, color=bar_colors,
                       edgecolor='white', linewidth=1.5)
        ax3.set_title('Placement Success Rate (%)')
        ax3.set_ylabel('Success Rate (%)')
        ax3.set_ylim(0, 110)
        for bar, val in zip(bars, success_rates):
            ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                     f'{val:.1f}%', ha='center', va='bottom', fontsize=9)

        # --- Plot 4: Latency Box Plot ---
        ax4 = axes[1, 0]
        latency_data = [r['latencies'] for r in all_results]
        bp = ax4.boxplot(latency_data, tick_labels=algo_names, patch_artist=True)
        for patch, color in zip(bp['boxes'], bar_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax4.set_title('Latency Distribution (ms)')
        ax4.set_ylabel('Latency (ms)')

        # --- Plot 5: Dynamic Event Timeline ---
        ax5 = axes[1, 1]
        if event_log:
            failure_times = [e['time'] for e in event_log if e['type'] == 'FAILURE']
            recovery_times = [e['time'] for e in event_log if e['type'] == 'RECOVERY']
            flash_times = [e['time'] for e in event_log if e['type'] == 'FLASH_CROWD']

            if failure_times:
                ax5.scatter(failure_times, [1] * len(failure_times),
                           marker='x', s=60, color=colors[1], label='Failure',
                           zorder=5)
            if recovery_times:
                ax5.scatter(recovery_times, [2] * len(recovery_times),
                           marker='^', s=60, color=colors[0], label='Recovery',
                           zorder=5)
            if flash_times:
                ax5.scatter(flash_times, [3] * len(flash_times),
                           marker='s', s=60, color=colors[3],
                           label='Flash Crowd', zorder=5)
            ax5.set_yticks([1, 2, 3])
            ax5.set_yticklabels(['Failure', 'Recovery', 'Flash Crowd'])
            ax5.set_title('Dynamic Event Timeline')
            ax5.set_xlabel('Time Step')
            ax5.legend(fontsize=8, loc='upper right')
        else:
            ax5.text(0.5, 0.5, 'No dynamic events', ha='center',
                     va='center', transform=ax5.transAxes)
            ax5.set_title('Dynamic Event Timeline')

        # --- Plot 6: Routing Decision Distribution (Pie) ---
        ax6 = axes[1, 2]
        for r in all_results:
            if 'routing_choices' in r and r['routing_choices']:
                counts = Counter(r['routing_choices'])
                labels = list(counts.keys())
                sizes = list(counts.values())
                pie_colors = colors[:len(labels)]
                wedges, texts, autotexts = ax6.pie(
                    sizes, labels=labels, colors=pie_colors,
                    autopct='%1.1f%%', startangle=90,
                    textprops={'fontsize': 9}
                )
                ax6.set_title('Hybrid Routing Distribution')
                break
        else:
            ax6.text(0.5, 0.5, 'No routing data', ha='center',
                     va='center', transform=ax6.transAxes)
            ax6.set_title('Routing Distribution')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"    Dashboard saved to: {save_path}")


# ================================================================
# SECTION 11b: RESEARCH-PAPER-STYLE PLOTS (GAL-VNE / KDD'23 style)
# ================================================================

class SensitivityAnalyzer:
    """Run sensitivity analysis experiments varying key parameters."""

    @staticmethod
    def run_over_time(algorithms: Dict, network: EdgeNetwork,
                      num_requests: int = 200) -> Dict[str, Dict]:
        """Run each algorithm over time, tracking cumulative metrics."""
        area_km = network.area_km
        results = {}

        for algo_name, algo in algorithms.items():
            network.reset()
            workload = WorkloadGenerator.generate_workload(num_requests, area_km)

            cumulative_success = []
            cumulative_rc_ratio = []
            cumulative_revenue = []
            total_success = 0
            total_resource_cost = 0.0
            total_revenue = 0.0

            for i, app in enumerate(workload):
                solution = algo.solve(app, network)
                if solution.is_feasible:
                    for svc in app.services:
                        network.allocate(solution.mapping[svc.service_id], svc)
                    total_success += 1
                    # Revenue = sum of service resources placed
                    rev = sum(s.cpu_required + s.ram_required for s in app.services)
                    total_revenue += rev
                    # Cost = resource consumed relative to capacity
                    cost = sum(s.cpu_required / 50.0 + s.ram_required / 64.0
                               for s in app.services)
                    total_resource_cost += cost

                accept_ratio = total_success / (i + 1)
                rc_ratio = total_revenue / max(total_resource_cost, 0.01)
                avg_revenue = total_revenue / (i + 1)

                cumulative_success.append(accept_ratio)
                cumulative_rc_ratio.append(rc_ratio)
                cumulative_revenue.append(avg_revenue)

            results[algo_name] = {
                'accept_ratios': cumulative_success,
                'rc_ratios': cumulative_rc_ratio,
                'avg_revenues': cumulative_revenue,
            }

        return results

    @staticmethod
    def vary_network_size(algorithms_factory, sizes: List[int],
                          num_requests: int = 50) -> Dict[str, List[float]]:
        """Measure solving time as network size varies."""
        solve_times = {name: [] for name in ['GA', 'PSO', 'MARL', 'HYBRID']}

        for size in sizes:
            rows = max(2, int(np.sqrt(size)))
            cols = max(2, size // rows)
            actual_size = rows * cols

            net = EdgeNetwork(grid_rows=rows, grid_cols=cols, area_km=4.0,
                               num_regions=min(4, actual_size))
            workload = WorkloadGenerator.generate_workload(num_requests, 4.0)

            ga = GeneticAlgorithm(population_size=30, generations=20)
            pso = ParticleSwarmOptimization(num_particles=20, max_iterations=20)
            marl = MARLPlacementAgent(network=net)

            for algo_name, algo in [('GA', ga), ('PSO', pso), ('MARL', marl)]:
                net.reset()
                total_time = 0.0
                for app in workload:
                    sol = algo.solve(app, net)
                    total_time += sol.decision_time_ms
                    if sol.is_feasible:
                        for svc in app.services:
                            net.allocate(sol.mapping[svc.service_id], svc)
                solve_times[algo_name].append(total_time / 1000.0)

            # Hybrid
            net.reset()
            engine = HybridPlacementEngine(network=net, train_rl=False)
            engine.marl_agent = marl
            total_time = 0.0
            for app in workload:
                sol, _ = engine.place(app)
                total_time += sol.decision_time_ms
            solve_times['HYBRID'].append(total_time / 1000.0)

            print(f"      Network size {actual_size}: done")

        return solve_times

    @staticmethod
    def vary_arrival_rate(network: EdgeNetwork, marl_agent,
                          rates: List[float],
                          base_requests: int = 80) -> Dict[str, List[float]]:
        """Measure acceptance ratio at different arrival rates (workload intensity)."""
        accept_rates = {name: [] for name in ['GA', 'PSO', 'MARL', 'HYBRID']}

        ga = GeneticAlgorithm(population_size=40, generations=25)
        pso = ParticleSwarmOptimization(num_particles=30, max_iterations=25)

        for rate in rates:
            num_req = int(base_requests * rate / 0.1)

            for algo_name, algo in [('GA', ga), ('PSO', pso), ('MARL', marl_agent)]:
                network.reset()
                workload = WorkloadGenerator.generate_workload(num_req, network.area_km)
                success = 0
                for app in workload:
                    sol = algo.solve(app, network)
                    if sol.is_feasible:
                        for svc in app.services:
                            network.allocate(sol.mapping[svc.service_id], svc)
                        success += 1
                accept_rates[algo_name].append(success / max(num_req, 1))

            # Hybrid
            network.reset()
            workload = WorkloadGenerator.generate_workload(num_req, network.area_km)
            engine = HybridPlacementEngine(network=network, train_rl=False)
            engine.marl_agent = marl_agent
            success = 0
            for app in workload:
                sol, _ = engine.place(app)
                if sol.is_feasible:
                    success += 1
            accept_rates['HYBRID'].append(success / max(num_req, 1))

            print(f"      Arrival rate {rate:.2f}: done")

        return accept_rates

    @staticmethod
    def vary_service_count(network: EdgeNetwork, marl_agent,
                           svc_ranges: List[Tuple[int, int]],
                           num_requests: int = 60) -> Dict[str, List[float]]:
        """Measure acceptance ratio at different service complexities."""
        accept_rates = {name: [] for name in ['GA', 'PSO', 'MARL', 'HYBRID']}

        ga = GeneticAlgorithm(population_size=40, generations=25)
        pso = ParticleSwarmOptimization(num_particles=30, max_iterations=25)

        for svc_range in svc_ranges:
            workload = []
            for i in range(num_requests):
                app = WorkloadGenerator.generate_single_app(
                    i, AppType.GENERIC, network.area_km,
                    services_override=svc_range
                )
                workload.append(app)

            for algo_name, algo in [('GA', ga), ('PSO', pso), ('MARL', marl_agent)]:
                network.reset()
                success = 0
                for app in workload:
                    sol = algo.solve(app, network)
                    if sol.is_feasible:
                        for svc in app.services:
                            network.allocate(sol.mapping[svc.service_id], svc)
                        success += 1
                accept_rates[algo_name].append(success / num_requests)

            # Hybrid
            network.reset()
            engine = HybridPlacementEngine(network=network, train_rl=False)
            engine.marl_agent = marl_agent
            success = 0
            for app in workload:
                sol, _ = engine.place(app)
                if sol.is_feasible:
                    success += 1
            accept_rates['HYBRID'].append(success / num_requests)

            print(f"      Services {svc_range}: done")

        return accept_rates


class ResearchPaperPlots:
    """Generate publication-quality plots matching GAL-VNE (KDD'23) style."""

    COLORS = {
        'GA': '#1f77b4',       # blue
        'PSO': '#ff7f0e',      # orange
        'MARL': '#d62728',     # red
        'HYBRID': '#9467bd',   # purple
    }
    MARKERS = {'GA': 'o', 'PSO': 's', 'MARL': '^', 'HYBRID': 'D'}

    @classmethod
    def generate_paper_plots(cls, time_results: Dict, solve_times: Dict,
                              arrival_results: Dict, svc_results: Dict,
                              network_sizes: List[int],
                              arrival_rates: List[float],
                              svc_labels: List[str],
                              save_path: str = "research_plots.png"):
        """Generate 2x3 grid of research-paper-style plots."""

        plt.rcParams.update({
            'font.size': 11,
            'axes.labelsize': 12,
            'axes.titlesize': 13,
            'legend.fontsize': 9,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'lines.linewidth': 1.8,
            'lines.markersize': 5,
        })

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Performance w.r.t. Simulation Parameters (Paper-Style Analysis)',
                     fontsize=15, fontweight='bold', y=0.98)

        algos = ['GA', 'PSO', 'MARL', 'HYBRID']

        # ── TOP ROW: Performance over simulation time ──

        # (a) Total Acceptance Ratio
        ax = axes[0, 0]
        for algo in algos:
            if algo in time_results:
                data = time_results[algo]['accept_ratios']
                x = range(1, len(data) + 1)
                ax.plot(x, data, color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo], markevery=len(data) // 8)
        ax.set_xlabel('Simulation Time (requests)')
        ax.set_ylabel('Acceptance Ratio')
        ax.set_title('(a) Total Acceptance Ratio')
        ax.legend(loc='lower right')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

        # (b) Long-term Total R/C Ratio
        ax = axes[0, 1]
        for algo in algos:
            if algo in time_results:
                data = time_results[algo]['rc_ratios']
                x = range(1, len(data) + 1)
                ax.plot(x, data, color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo], markevery=len(data) // 8)
        ax.set_xlabel('Simulation Time (requests)')
        ax.set_ylabel('Long-term R/C Ratio')
        ax.set_title('(b) Long-term Total R/C Ratio')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # (c) Long-term Average Revenue
        ax = axes[0, 2]
        for algo in algos:
            if algo in time_results:
                data = time_results[algo]['avg_revenues']
                x = range(1, len(data) + 1)
                ax.plot(x, data, color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo], markevery=len(data) // 8)
        ax.set_xlabel('Simulation Time (requests)')
        ax.set_ylabel('Long-term Average Revenue')
        ax.set_title('(c) Long-term Average Revenue')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        # ── BOTTOM ROW: Sensitivity analysis ──

        # (d) Solving Time vs Network Size
        ax = axes[1, 0]
        for algo in algos:
            if algo in solve_times:
                ax.plot(network_sizes, solve_times[algo],
                        color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo])
        ax.set_xlabel('Substrate Network Size (servers)')
        ax.set_ylabel('Solving Time (seconds)')
        ax.set_title('(d) Solving Time vs. Network Size')
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)

        # (e) Accept Ratio vs Arrival Rate
        ax = axes[1, 1]
        for algo in algos:
            if algo in arrival_results:
                ax.plot(arrival_rates, arrival_results[algo],
                        color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo])
        ax.set_xlabel('Arrival Rate (multiplier)')
        ax.set_ylabel('Acceptance Ratio')
        ax.set_title('(e) Accept Ratio vs. Arrival Rate')
        ax.legend(loc='lower left')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

        # (f) Accept Ratio vs Service Complexity
        ax = axes[1, 2]
        for algo in algos:
            if algo in svc_results:
                ax.plot(svc_labels, svc_results[algo],
                        color=cls.COLORS[algo], label=algo,
                        marker=cls.MARKERS[algo])
        ax.set_xlabel('Services per Application')
        ax.set_ylabel('Acceptance Ratio')
        ax.set_title('(f) Accept Ratio vs. Service Complexity')
        ax.legend(loc='lower left')
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"    Research paper plots saved to: {save_path}")


# ================================================================
# SECTION 12: HYBRID PLACEMENT ENGINE
# ================================================================

class HybridPlacementEngine:
    """
    Dynamically routes placement requests to the best-suited algorithm.
    Uses MARL with GAT instead of single-agent RL.
    """

    def __init__(self, network: EdgeNetwork, train_rl: bool = True,
                 rl_pretrain_episodes: int = 150,
                 rl_finetune_episodes: int = 50):
        self.network = network
        self.ga = GeneticAlgorithm(population_size=60, generations=40,
                                    crossover_rate=0.85, mutation_rate=0.05)
        self.pso = ParticleSwarmOptimization(num_particles=40, max_iterations=40)
        self.marl_agent = MARLPlacementAgent(network=network)

        self.routing_table = {
            AppType.LTS: 'GA',
            AppType.MMA: 'PSO',
            AppType.PWD: 'MARL',
            AppType.GENERIC: 'MARL',
        }
        self.performance_history: Dict[str, List[float]] = {
            'GA': [], 'PSO': [], 'MARL': []
        }

        if train_rl:
            self.marl_agent.pretrain(network, num_episodes=rl_pretrain_episodes,
                                     apps_per_episode=30)
            network.reset()
            self.marl_agent.finetune(network, num_episodes=rl_finetune_episodes,
                                      apps_per_episode=20)
            network.reset()
            self.marl_agent.enable_online_learning()

    def classify(self, app: Application) -> str:
        algorithm = self.routing_table[app.app_type]
        avg_utilization = np.mean(self.network.get_utilization_vector())
        if avg_utilization > 0.75:
            algorithm = 'MARL'
        if app.num_services <= 2:
            algorithm = 'PSO'
        if app.placement_deadline_ms < 100:
            algorithm = 'PSO'
        return algorithm

    def place(self, app: Application) -> Tuple[PlacementSolution, str]:
        algorithm_name = self.classify(app)
        snapshot = self.network.snapshot()

        if algorithm_name == 'GA':
            solution = self.ga.solve(app, self.network)
        elif algorithm_name == 'PSO':
            solution = self.pso.solve(app, self.network)
        elif algorithm_name == 'MARL':
            solution = self.marl_agent.solve(app, self.network)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

        if not solution.is_feasible:
            self.network.restore(snapshot)
            fallback = {'GA': 'PSO', 'PSO': 'MARL', 'MARL': 'GA'}.get(algorithm_name)
            if fallback:
                if fallback == 'GA':
                    solution = self.ga.solve(app, self.network)
                elif fallback == 'PSO':
                    solution = self.pso.solve(app, self.network)
                elif fallback == 'MARL':
                    solution = self.marl_agent.solve(app, self.network)
                algorithm_name = f"{algorithm_name}->{fallback}"

        if solution.is_feasible:
            for svc in app.services:
                self.network.allocate(solution.mapping[svc.service_id], svc)

        base_algo = algorithm_name.split('->')[0]
        self.performance_history[base_algo].append(solution.fitness)
        return solution, algorithm_name

    def get_stats(self) -> Dict:
        stats = {}
        for algo, fitnesses in self.performance_history.items():
            if fitnesses:
                stats[algo] = {'count': len(fitnesses),
                               'avg_fitness': np.mean(fitnesses),
                               'std_fitness': np.std(fitnesses)}
        return stats


# ================================================================
# SECTION 13: EVALUATION FRAMEWORK
# ================================================================

class Evaluator:
    """Compare hybrid vs. individual algorithms."""

    @staticmethod
    def run_single_algorithm(algo_name, algorithm, workload, network):
        network.reset()
        results = {'algorithm': algo_name, 'latencies': [], 'distances': [],
                   'num_servers_list': [], 'resource_stds': [],
                   'decision_times': [], 'success_count': 0,
                   'total_count': len(workload)}
        for app in workload:
            solution = algorithm.solve(app, network)
            if solution.is_feasible:
                for svc in app.services:
                    network.allocate(solution.mapping[svc.service_id], svc)
                results['success_count'] += 1
            results['latencies'].append(solution.latency)
            results['distances'].append(solution.avg_distance)
            results['num_servers_list'].append(solution.num_servers)
            results['resource_stds'].append(solution.resource_std)
            results['decision_times'].append(solution.decision_time_ms)
        results['success_rate'] = results['success_count'] / max(results['total_count'], 1)
        results['avg_latency'] = np.mean(results['latencies'])
        results['avg_distance'] = np.mean(results['distances'])
        results['avg_num_servers'] = np.mean(results['num_servers_list'])
        results['avg_resource_std'] = np.mean(results['resource_stds'])
        results['avg_decision_time'] = np.mean(results['decision_times'])
        results['total_decision_time'] = sum(results['decision_times'])
        return results

    @staticmethod
    def run_hybrid(engine, workload, network, logger=None):
        network.reset()
        results = {'algorithm': 'HYBRID', 'latencies': [], 'distances': [],
                   'num_servers_list': [], 'resource_stds': [],
                   'decision_times': [], 'success_count': 0,
                   'total_count': len(workload), 'routing_choices': []}
        for idx, app in enumerate(workload):
            network.advance_time(idx, failure_prob=0.03,
                                  recovery_prob=0.15, flash_crowd_prob=0.04)
            recent = [e for e in network.event_log if e['time'] == idx]
            event_str = ", ".join(e['type'] for e in recent) if recent else ""
            solution, algo_used = engine.place(app)
            if solution.is_feasible:
                results['success_count'] += 1
            results['latencies'].append(solution.latency)
            results['distances'].append(solution.avg_distance)
            results['num_servers_list'].append(solution.num_servers)
            results['resource_stds'].append(solution.resource_std)
            results['decision_times'].append(solution.decision_time_ms)
            results['routing_choices'].append(algo_used)
            network.record_state()
            if logger:
                logger.log(idx, app, algo_used, solution, event_str)
        results['success_rate'] = results['success_count'] / max(results['total_count'], 1)
        results['avg_latency'] = np.mean(results['latencies'])
        results['avg_distance'] = np.mean(results['distances'])
        results['avg_num_servers'] = np.mean(results['num_servers_list'])
        results['avg_resource_std'] = np.mean(results['resource_stds'])
        results['avg_decision_time'] = np.mean(results['decision_times'])
        results['total_decision_time'] = sum(results['decision_times'])
        return results

    @staticmethod
    def print_comparison(all_results):
        print("\n" + "=" * 95)
        print(f"{'ALGORITHM':<12} | {'Success%':>9} | {'Avg Lat(ms)':>12} | "
              f"{'Avg Dist':>10} | {'Avg #Srv':>9} | {'Res StdDev':>11} | "
              f"{'Avg Time(ms)':>13}")
        print("-" * 95)
        for r in all_results:
            print(f"{r['algorithm']:<12} | {r['success_rate']*100:>8.1f}% | "
                  f"{r['avg_latency']:>12.2f} | {r['avg_distance']:>10.3f} | "
                  f"{r['avg_num_servers']:>9.2f} | {r['avg_resource_std']:>11.4f} | "
                  f"{r['avg_decision_time']:>13.2f}")
        print("=" * 95)
        metrics = [('avg_latency', 'Best Latency'), ('avg_distance', 'Best Distance'),
                   ('avg_num_servers', 'Fewest Servers'), ('avg_resource_std', 'Best Load Balance'),
                   ('avg_decision_time', 'Fastest Decision')]
        print("\n  Per-metric winners (lower = better):")
        for mk, mn in metrics:
            best = min(all_results, key=lambda r: r[mk])
            print(f"    {mn:<20} -> {best['algorithm']:<12} ({best[mk]:.4f})")
        for r in all_results:
            if 'routing_choices' in r:
                counts = Counter(r['routing_choices'])
                print(f"\n  Hybrid routing distribution:")
                for algo, count in sorted(counts.items()):
                    pct = count / len(r['routing_choices']) * 100
                    bar = '#' * int(pct / 2)
                    print(f"    {algo:<15}: {count:>4} ({pct:>5.1f}%) {bar}")


# ================================================================
# SECTION 14: ENERGY-AWARE SCHEDULER
# ================================================================

class EnergyAwareScheduler:
    """
    Green-computing placement mode.
    Ranks feasible servers by a combined score:
        score = fitness * (1 - energy_weight) + norm_energy * energy_weight
    Wraps any base solver (GA, PSO, MARL) with an energy penalty.
    """

    def __init__(self, base_solver, energy_weight: float = 0.3):
        """
        Args:
            base_solver: Any object with a .solve(app, network) method.
            energy_weight: 0 = ignore energy, 1 = optimize energy only.
        """
        self.base_solver = base_solver
        self.energy_weight = energy_weight
        self.MAX_ENERGY_W = 150.0 * 20  # 20 servers at max draw

    def solve(self, app: 'Application',
              network: 'EdgeNetwork') -> 'PlacementSolution':
        """Solve and re-rank if a greener option exists with similar fitness."""
        solution = self.base_solver.solve(app, network)
        if not solution.is_feasible:
            return solution

        current_energy = network.get_total_energy()
        norm_energy = min(current_energy / self.MAX_ENERGY_W, 1.0)

        # Try to find a lower-energy feasible mapping
        best_fitness = solution.fitness
        best_mapping = solution.mapping
        best_energy = current_energy

        for _ in range(5):   # 5 greedy energy-aware candidates
            snap = network.snapshot()
            candidate_mapping = {}
            feasible_overall = True

            for svc in app.services:
                feasible = network.get_feasible_servers(svc)
                if not feasible:
                    feasible_overall = False
                    break
                # Sort by energy consumption (ascending)
                feasible.sort(key=lambda s: network.servers[s].energy_consumption)
                chosen = feasible[0]
                candidate_mapping[svc.service_id] = chosen
                network.allocate(chosen, svc)

            if feasible_overall:
                cand_energy = network.get_total_energy()
                cand_fitness, _ = ObjectiveCalculator.compute_fitness(
                    candidate_mapping, app, network,
                    {'latency':0.25,'distance':0.25,'num_servers':0.25,'resource_std':0.25}
                )
                combined_base = best_fitness * (1 - self.energy_weight) + \
                                min(best_energy/self.MAX_ENERGY_W, 1.0) * self.energy_weight
                combined_cand = cand_fitness * (1 - self.energy_weight) + \
                                min(cand_energy/self.MAX_ENERGY_W, 1.0) * self.energy_weight
                if combined_cand < combined_base:
                    best_fitness  = cand_fitness
                    best_mapping  = candidate_mapping.copy()
                    best_energy   = cand_energy

            network.restore(snap)

        # Apply best mapping
        for svc in app.services:
            network.allocate(best_mapping[svc.service_id], svc)
        _, metrics = ObjectiveCalculator.compute_fitness(
            best_mapping, app, network,
            {'latency':0.25,'distance':0.25,'num_servers':0.25,'resource_std':0.25}
        )
        return PlacementSolution(
            mapping=best_mapping,
            is_feasible=metrics['feasible'],
            latency=metrics['latency'],
            avg_distance=metrics['distance'],
            num_servers=metrics['num_servers'],
            resource_std=metrics['resource_std'],
            fitness=best_fitness,
        )


# ================================================================
# SECTION 15: SLA MONITOR
# ================================================================

class SLAMonitor:
    """
    Tracks Service Level Agreement (SLA) violations per application type.
    A violation occurs when placed latency exceeds the app's deadline.
    Generates a summary report after simulation.
    """

    def __init__(self):
        self.records: List[Dict] = []
        self.violation_counts: Dict[str, int] = {}
        self.total_counts: Dict[str, int] = {}

    def record(self, app: 'Application',
               solution: 'PlacementSolution') -> bool:
        """
        Record placement result. Returns True if SLA was violated.
        """
        violated = (
            solution.is_feasible and
            solution.latency > app.placement_deadline_ms
        )
        app_name = app.app_type.name
        self.total_counts[app_name] = self.total_counts.get(app_name, 0) + 1
        if violated:
            self.violation_counts[app_name] = \
                self.violation_counts.get(app_name, 0) + 1
        self.records.append({
            'app_type': app_name,
            'latency': solution.latency,
            'deadline': app.placement_deadline_ms,
            'violated': violated,
            'feasible': solution.is_feasible,
        })
        return violated

    def get_violation_rate(self, app_type: str) -> float:
        total = self.total_counts.get(app_type, 0)
        if total == 0:
            return 0.0
        return self.violation_counts.get(app_type, 0) / total

    def print_report(self):
        print("\n  SLA Violation Report:")
        print("  " + "-" * 55)
        total_v, total_all = 0, 0
        for atype, total in sorted(self.total_counts.items()):
            viols = self.violation_counts.get(atype, 0)
            rate  = viols / total * 100
            bar   = '█' * int(rate / 5) + '░' * (20 - int(rate / 5))
            print(f"    {atype:<10}: {viols:>3}/{total:<3} violated  "
                  f"({rate:>5.1f}%) {bar}")
            total_v += viols; total_all += total
        if total_all:
            overall = total_v / total_all * 100
            print(f"    {'OVERALL':<10}: {total_v:>3}/{total_all:<3} violated  "
                  f"({overall:>5.1f}%)")
        print("  " + "-" * 55)


# ================================================================
# SECTION 16: ADAPTIVE WEIGHT TUNER
# ================================================================

class AdaptiveWeightTuner:
    """
    Adjusts objective weights in real time based on recent SLA violations.
    If latency SLAs are being violated frequently, increases the latency weight.
    Uses exponential moving average to avoid thrashing.
    """

    BASE_WEIGHTS = {
        'latency': 0.25,
        'distance': 0.25,
        'num_servers': 0.25,
        'resource_std': 0.25,
    }

    def __init__(self, sla_monitor: SLAMonitor,
                 alpha: float = 0.1,
                 update_interval: int = 20):
        """
        Args:
            sla_monitor: SLAMonitor instance to query.
            alpha: EMA smoothing factor (0=no adaptation, 1=instant).
            update_interval: How often (requests) to re-tune weights.
        """
        self.sla_monitor    = sla_monitor
        self.alpha          = alpha
        self.update_interval = update_interval
        self.weights        = dict(self.BASE_WEIGHTS)
        self.request_count  = 0
        self.history: List[Dict] = []

    def step(self) -> Dict[str, float]:
        """Call after each placement. Returns current weights."""
        self.request_count += 1
        if self.request_count % self.update_interval == 0:
            self._update()
        return dict(self.weights)

    def _update(self):
        """Compute new weights based on SLA violations."""
        # Violation rates per type (treat LTS latency SLA as proxy for latency weight)
        lts_vr  = self.sla_monitor.get_violation_rate('LTS')
        mma_vr  = self.sla_monitor.get_violation_rate('MMA')
        pwd_vr  = self.sla_monitor.get_violation_rate('PWD')
        avg_vr  = (lts_vr + mma_vr + pwd_vr) / 3.0

        # Target: if violation rate > 0.2, shift weight toward latency
        latency_boost  = max(0.0, (avg_vr - 0.1) * 0.5)
        target_latency = min(0.60, self.BASE_WEIGHTS['latency'] + latency_boost)
        remainder      = 1.0 - target_latency

        target = {
            'latency':      target_latency,
            'distance':     remainder * 0.33,
            'num_servers':  remainder * 0.33,
            'resource_std': remainder * 0.34,
        }

        # EMA blend
        for k in self.weights:
            self.weights[k] = (
                self.alpha * target[k] +
                (1 - self.alpha) * self.weights[k]
            )
        self.history.append(dict(self.weights))

    def print_history(self):
        if not self.history:
            return
        print("\n  Adaptive Weight History (every",
              self.update_interval, "requests):")
        print(f"    {'Step':>4}  {'Latency':>8}  {'Distance':>9}  "
              f"{'Privacy':>8}  {'LoadStd':>8}")
        for i, w in enumerate(self.history):
            print(f"    {(i+1)*self.update_interval:>4}  "
                  f"{w['latency']:>8.3f}  {w['distance']:>9.3f}  "
                  f"{w['num_servers']:>8.3f}  {w['resource_std']:>8.3f}")


# ================================================================
# SECTION 17: FAILOVER MANAGER
# ================================================================

class FailoverManager:
    """
    Detects newly failed servers and migrates their services
    to the next-best available server.

    Usage:
        fm = FailoverManager(network)
        fm.register(placement_solution, app)   # after each placement
        migrations = fm.check_and_migrate()     # call each timestep
    """

    def __init__(self, network: 'EdgeNetwork'):
        self.network  = network
        # Maps server_id -> list of (service, app) tuples hosted on it
        self.hosted: Dict[int, List[Tuple['Service', 'Application']]] = {}
        self.migration_log: List[Dict] = []

    def register(self, solution: 'PlacementSolution',
                 app: 'Application'):
        """Record which services are placed on which servers."""
        if not solution.is_feasible:
            return
        for svc in app.services:
            sid = solution.mapping[svc.service_id]
            if sid not in self.hosted:
                self.hosted[sid] = []
            self.hosted[sid].append((svc, app))

    def check_and_migrate(self) -> List[Dict]:
        """
        Check for newly failed servers and attempt to migrate
        each of their services to the best available alternative.
        Returns list of migration records.
        """
        migrations = []
        failed_servers = [
            s for s in self.network.servers
            if s.is_failed and s.server_id in self.hosted
        ]
        for server in failed_servers:
            services = self.hosted.pop(server.server_id, [])
            for svc, app in services:
                # Find nearest feasible replacement
                candidates = self.network.get_feasible_servers(svc)
                if not candidates:
                    self.migration_log.append({
                        'svc_id': svc.service_id,
                        'from': server.server_id,
                        'to': None,
                        'status': 'FAILED_NO_HOST'
                    })
                    continue

                # Pick candidate with lowest utilisation
                candidates.sort(
                    key=lambda s: self.network.servers[s].cpu_utilization
                )
                dest = candidates[0]
                self.network.allocate(dest, svc)

                if dest not in self.hosted:
                    self.hosted[dest] = []
                self.hosted[dest].append((svc, app))

                record = {
                    'svc_id': svc.service_id,
                    'from': server.server_id,
                    'to': dest,
                    'status': 'MIGRATED'
                }
                self.migration_log.append(record)
                migrations.append(record)

        return migrations

    def print_migration_log(self):
        if not self.migration_log:
            return
        print("\n  Failover Migration Log:")
        print(f"    {'SvcID':>5}  {'From':>5}  {'To':>5}  {'Status'}")
        for rec in self.migration_log:
            to_str = str(rec['to']) if rec['to'] is not None else 'N/A'
            print(f"    {rec['svc_id']:>5}  {rec['from']:>5}  "
                  f"{to_str:>5}  {rec['status']}")


# ================================================================
# SECTION 18: BENCHMARK RUNNER (Greedy + Random Baselines)
# ================================================================

class FirstFitGreedy:
    """
    First-Fit Greedy baseline.
    Assigns each service to the first server (by ID order)
    that has enough resources. O(services × servers).
    """

    def solve(self, app: 'Application',
              network: 'EdgeNetwork') -> 'PlacementSolution':
        start = time.time()
        mapping = {}
        for svc in app.services:
            placed = False
            for server in network.servers:
                if server.can_host(svc):
                    mapping[svc.service_id] = server.server_id
                    network.allocate(server.server_id, svc)
                    placed = True
                    break
            if not placed:
                mapping[svc.service_id] = 0  # infeasible marker

        feasible = ObjectiveCalculator.check_feasibility(
            mapping, app, network
        )
        latency = ObjectiveCalculator.compute_latency(mapping, app, network)
        distance = ObjectiveCalculator.compute_avg_distance(mapping, network)
        num_servers = ObjectiveCalculator.compute_num_servers(mapping)
        resource_std = ObjectiveCalculator.compute_resource_std(network)
        decision_time = (time.time() - start) * 1000

        return PlacementSolution(
            mapping=mapping,
            is_feasible=feasible,
            latency=latency,
            avg_distance=distance,
            num_servers=num_servers,
            resource_std=resource_std,
            fitness=latency / 500 + distance / 6,
            decision_time_ms=decision_time
        )


class RandomPlacement:
    """
    Random baseline.
    Assigns each service to a uniformly random server (may be infeasible).
    """

    def solve(self, app: 'Application',
              network: 'EdgeNetwork') -> 'PlacementSolution':
        start = time.time()
        num_srv = network.num_servers
        mapping = {
            svc.service_id: random.randint(0, num_srv - 1)
            for svc in app.services
        }
        feasible = ObjectiveCalculator.check_feasibility(
            mapping, app, network
        )
        latency = ObjectiveCalculator.compute_latency(mapping, app, network)
        distance = ObjectiveCalculator.compute_avg_distance(mapping, network)
        num_servers = ObjectiveCalculator.compute_num_servers(mapping)
        resource_std = ObjectiveCalculator.compute_resource_std(network)
        decision_time = (time.time() - start) * 1000

        return PlacementSolution(
            mapping=mapping,
            is_feasible=feasible,
            latency=latency,
            avg_distance=distance,
            num_servers=num_servers,
            resource_std=resource_std,
            fitness=10.0 if not feasible else latency / 500,
            decision_time_ms=decision_time
        )


class BenchmarkRunner:
    """
    Runs all algorithms — including greedy and random baselines —
    over the same workload and prints a unified comparison table.
    This ensures academically rigorous evaluation.
    """

    @staticmethod
    def run_all(
        workload: List['Application'],
        network: 'EdgeNetwork',
        ga, pso, marl_agent,
        hybrid_engine
    ) -> List[Dict]:
        algorithms = [
            ('RANDOM',    RandomPlacement()),
            ('FIRST-FIT', FirstFitGreedy()),
            ('GA',        ga),
            ('PSO',       pso),
            ('MARL',      marl_agent),
        ]
        all_results = []

        for name, algo in algorithms:
            all_results.append(
                Evaluator.run_single_algorithm(name, algo, workload, network)
            )

        # Hybrid run
        hybrid_results = Evaluator.run_hybrid(hybrid_engine, workload, network)
        all_results.append(hybrid_results)

        return all_results

    @staticmethod
    def print_extended_comparison(all_results: List[Dict]):
        """Print extended comparison table including baseline algorithms."""
        print("\n" + "=" * 110)
        print(f"{'ALGORITHM':<12} | {'Success%':>9} | {'Avg Lat(ms)':>12} | "
              f"{'Avg Dist':>10} | {'Avg #Srv':>9} | {'Res StdDev':>11} | "
              f"{'Avg Time(ms)':>13} | {'vs Greedy':>10}")
        print("-" * 110)

        greedy_success = next(
            (r['success_rate'] for r in all_results
             if r['algorithm'] == 'FIRST-FIT'), None
        )

        for r in all_results:
            vs_greedy = ''
            if greedy_success and r['algorithm'] not in ('RANDOM', 'FIRST-FIT'):
                delta = (r['success_rate'] - greedy_success) * 100
                vs_greedy = f"+{delta:.1f}%" if delta >= 0 else f"{delta:.1f}%"

            print(
                f"{r['algorithm']:<12} | {r['success_rate']*100:>8.1f}% | "
                f"{r['avg_latency']:>12.2f} | {r['avg_distance']:>10.3f} | "
                f"{r['avg_num_servers']:>9.2f} | {r['avg_resource_std']:>11.4f} | "
                f"{r['avg_decision_time']:>13.2f} | {vs_greedy:>10}"
            )
        print("=" * 110)


# ================================================================
# SECTION 19: MAIN
# ================================================================

def main():
    print()
    print("=" * 70)
    print("   ADVANCED HYBRID SERVICE PLACEMENT FOR EDGE COMPUTING")
    print("   GA + PSO + Multi-Agent RL (GAT) + Dynamic Environments")
    print("=" * 70)

    NUM_REQUESTS = 100
    GRID_ROWS, GRID_COLS = 4, 5
    AREA_KM = 4.0
    PRETRAIN_EP = 150
    FINETUNE_EP = 50
    NUM_REGIONS = 4

    # Step 1: Create network
    print(f"\n[1/8] Creating edge network ({NUM_REGIONS} regions)...")
    network = EdgeNetwork(grid_rows=GRID_ROWS, grid_cols=GRID_COLS,
                           area_km=AREA_KM, num_regions=NUM_REGIONS)
    print(f"      {network.num_servers} servers in {AREA_KM}x{AREA_KM} km")
    for r in range(NUM_REGIONS):
        srvs = network.get_region_servers(r)
        print(f"      Region {r}: {len(srvs)} servers")

    # Step 2: Generate workload
    print(f"\n[2/8] Generating {NUM_REQUESTS} requests...")
    workload = WorkloadGenerator.generate_workload(NUM_REQUESTS, AREA_KM)
    tc = {}
    for app in workload:
        tc[app.app_type.name] = tc.get(app.app_type.name, 0) + 1
    for k, v in sorted(tc.items()):
        print(f"      {k:>10}: {v}")

    # Step 3: Init GA, PSO
    print(f"\n[3/8] Initializing GA and PSO...")
    ga = GeneticAlgorithm(population_size=60, generations=40)
    pso = ParticleSwarmOptimization(num_particles=40, max_iterations=40)

    # Step 4: MARL pretrain with curriculum
    print(f"\n[4/8] MARL pretraining (curriculum learning)...")
    marl_agent = MARLPlacementAgent(network=network)
    marl_agent.pretrain(network, num_episodes=PRETRAIN_EP, apps_per_episode=30)
    network.reset()

    # Step 5: MARL finetune on dynamic env
    print(f"\n[5/8] MARL finetuning (dynamic environment)...")
    marl_agent.finetune(network, num_episodes=FINETUNE_EP, apps_per_episode=20)
    network.reset()
    marl_agent.enable_online_learning()

    # Step 6: Run individual algorithms
    print(f"\n[6/8] Running individual algorithms...")
    print("      Running GA...")
    ga_results = Evaluator.run_single_algorithm('GA', ga, workload, network)
    print(f"      GA: {ga_results['success_rate']*100:.1f}% success")
    print("      Running PSO...")
    pso_results = Evaluator.run_single_algorithm('PSO', pso, workload, network)
    print(f"      PSO: {pso_results['success_rate']*100:.1f}% success")
    print("      Running MARL...")
    marl_results = Evaluator.run_single_algorithm('MARL', marl_agent, workload, network)
    print(f"      MARL: {marl_results['success_rate']*100:.1f}% success")

    # Step 7: Run hybrid with dynamic events
    print(f"\n[7/8] Running HYBRID (dynamic environment + online learning)...")
    hybrid_engine = HybridPlacementEngine(network=network, train_rl=False)
    hybrid_engine.marl_agent = marl_agent
    logger = DecisionLogger(filepath="routing_decisions.csv")
    hybrid_results = Evaluator.run_hybrid(hybrid_engine, workload, network, logger)
    print(f"      HYBRID: {hybrid_results['success_rate']*100:.1f}% success")
    logger.save()

    # Step 8: Results
    print(f"\n[8/8] RESULTS")
    all_results = [ga_results, pso_results, marl_results, hybrid_results]
    Evaluator.print_comparison(all_results)

    # Per-type breakdown
    print("\n  Per-application-type breakdown (Hybrid):")
    print("  " + "-" * 70)
    for app_type in AppType:
        type_apps = [a for a in workload if a.app_type == app_type]
        if not type_apps:
            continue
        network.reset()
        te = HybridPlacementEngine(network=network, train_rl=False)
        te.marl_agent = marl_agent
        tlat, tdist, tsrv, tsuc, talgos = [], [], [], 0, []
        for app in type_apps:
            sol, alg = te.place(app)
            if sol.is_feasible:
                tsuc += 1
            tlat.append(sol.latency)
            tdist.append(sol.avg_distance)
            tsrv.append(sol.num_servers)
            talgos.append(alg)
        ac = Counter(talgos)
        astr = ", ".join(f"{a}:{c}" for a, c in ac.most_common())
        print(f"    {app_type.name:>10}: Success={tsuc}/{len(type_apps):>2}, "
              f"AvgLat={np.mean(tlat):>6.1f}ms, AvgDist={np.mean(tdist):>5.3f}km, "
              f"AvgSrv={np.mean(tsrv):>3.1f}, Algos=[{astr}]")

    # Dynamic events summary
    ec = Counter(e['type'] for e in network.event_log)
    print(f"\n  Dynamic Events: {dict(ec)}")

    # Generate dashboard
    print(f"\n  Generating 6-plot analytics dashboard...")
    DashboardVisualizer.generate_dashboard(all_results, network,
                                            network.event_log,
                                            save_path="dashboard.png")

    # --- Generate research paper plots ---
    print(f"\n  ═══ Running sensitivity analysis for paper-style plots ═══")

    # (a-c) Performance over simulation time
    print("    Running over-time analysis (200 requests per algorithm)...")
    time_algos = {'GA': ga, 'PSO': pso, 'MARL': marl_agent}
    time_results = SensitivityAnalyzer.run_over_time(time_algos, network,
                                                      num_requests=200)
    # Add HYBRID over time
    network.reset()
    hybrid_wl = WorkloadGenerator.generate_workload(200, AREA_KM)
    h_engine = HybridPlacementEngine(network=network, train_rl=False)
    h_engine.marl_agent = marl_agent
    h_accept, h_rc, h_rev = [], [], []
    h_suc, h_cost, h_revenue = 0, 0.0, 0.0
    for i, app in enumerate(hybrid_wl):
        sol, _ = h_engine.place(app)
        if sol.is_feasible:
            h_suc += 1
            rev = sum(s.cpu_required + s.ram_required for s in app.services)
            h_revenue += rev
            cost = sum(s.cpu_required / 50.0 + s.ram_required / 64.0
                       for s in app.services)
            h_cost += cost
        h_accept.append(h_suc / (i + 1))
        h_rc.append(h_revenue / max(h_cost, 0.01))
        h_rev.append(h_revenue / (i + 1))
    time_results['HYBRID'] = {
        'accept_ratios': h_accept,
        'rc_ratios': h_rc,
        'avg_revenues': h_rev,
    }
    print("    Over-time analysis complete.")

    # (d) Solving time vs network size
    print("    Running network size sensitivity...")
    net_sizes = [4, 9, 16, 25, 36]
    solve_times_data = SensitivityAnalyzer.vary_network_size(
        None, net_sizes, num_requests=30
    )

    # (e) Accept ratio vs arrival rate
    print("    Running arrival rate sensitivity...")
    arr_rates = [0.05, 0.10, 0.20, 0.30, 0.50]
    arrival_data = SensitivityAnalyzer.vary_arrival_rate(
        network, marl_agent, arr_rates, base_requests=40
    )





    

    # (f) Accept ratio vs service count
    print("    Running service complexity sensitivity...")
    svc_ranges = [(1, 2), (2, 4), (3, 6), (5, 8), (7, 10)]
    svc_labels = ['1-2', '2-4', '3-6', '5-8', '7-10']
    svc_data = SensitivityAnalyzer.vary_service_count(
        network, marl_agent, svc_ranges, num_requests=40
    )

    ResearchPaperPlots.generate_paper_plots(
        time_results=time_results,
        solve_times=solve_times_data,
        arrival_results=arrival_data,
        svc_results=svc_data,
        network_sizes=net_sizes,
        arrival_rates=arr_rates,
        svc_labels=svc_labels,
        save_path="research_plots.png"
    )

    # ── Extended Benchmark (with greedy + random baselines) ──
    print(f"\n  Running extended benchmark (Greedy + Random baselines)...")
    workload_bench = WorkloadGenerator.generate_workload(60, AREA_KM)
    bench_engine = HybridPlacementEngine(network=network, train_rl=False)
    bench_engine.marl_agent = marl_agent
    bench_results = BenchmarkRunner.run_all(
        workload_bench, network,
        GeneticAlgorithm(population_size=40, generations=20),
        ParticleSwarmOptimization(num_particles=30, max_iterations=20),
        marl_agent, bench_engine
    )
    BenchmarkRunner.print_extended_comparison(bench_results)

    # ── SLA Report ──
    sla_monitor = SLAMonitor()
    network.reset()
    eval_engine = HybridPlacementEngine(network=network, train_rl=False)
    eval_engine.marl_agent = marl_agent
    for app in workload:
        sol, _ = eval_engine.place(app)
        sla_monitor.record(app, sol)
    sla_monitor.print_report()

    # ── Adaptive Weight History ──
    weight_tuner = AdaptiveWeightTuner(sla_monitor, alpha=0.1, update_interval=10)
    for _ in range(len(workload)):
        weight_tuner.step()
    weight_tuner.print_history()

    print("\n" + "=" * 70)
    print("   EXPERIMENT COMPLETE")
    print("   Outputs: routing_decisions.csv, dashboard.png, research_plots.png")
    print("=" * 70)


if __name__ == "__main__":
    main()
