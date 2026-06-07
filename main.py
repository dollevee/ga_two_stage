import os
import random
import time
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import pandas as pd
import pulp

@dataclass
class InputData:
    consumers: List[str]
    stage1: List[str]
    stage2: List[str]

    demand: Dict[str, float]
    capacity_stage1: Dict[str, float]
    capacity_stage2: Dict[str, float]

    open_cost_stage1: Dict[str, float]
    open_cost_stage2: Dict[str, float]

    cost_1_2: Dict[Tuple[str, str], float]
    cost_2_c: Dict[Tuple[str, str], float]

@dataclass
class Solution:
    objective_value: Optional[float]
    solve_time: float

    opened_stage1: List[str]
    opened_stage2: List[str]

    flow_1_2: Dict[Tuple[str, str], float]
    flow_2_c: Dict[Tuple[str, str], float]

def read_input_data(data_dir: str) -> InputData:
    consumers_df   = pd.read_csv(os.path.join(data_dir, "consumers.csv"))
    stage1_df      = pd.read_csv(os.path.join(data_dir, "plants_stage1.csv"))
    stage2_df      = pd.read_csv(os.path.join(data_dir, "plants_stage2.csv"))
    cost_1_2_df    = pd.read_csv(os.path.join(data_dir, "cost_1_2.csv"))
    cost_2_c_df    = pd.read_csv(os.path.join(data_dir, "cost_2_C.csv"))

    consumers = consumers_df["id"].astype(str).tolist()
    stage1    = stage1_df["id"].astype(str).tolist()
    stage2    = stage2_df["id"].astype(str).tolist()

    demand            = dict(zip(consumers_df["id"].astype(str), consumers_df["demand"].astype(float)))
    capacity_stage1   = dict(zip(stage1_df["id"].astype(str), stage1_df["capacity"].astype(float)))
    capacity_stage2   = dict(zip(stage2_df["id"].astype(str), stage2_df["capacity"].astype(float)))
    open_cost_stage1  = dict(zip(stage1_df["id"].astype(str), stage1_df["open_cost"].astype(float)))
    open_cost_stage2  = dict(zip(stage2_df["id"].astype(str), stage2_df["open_cost"].astype(float)))

    cost_1_2 = {
        (str(row["i_id"]), str(row["j_id"])): float(row["cost"])
        for _, row in cost_1_2_df.iterrows()
    }
    cost_2_c = {
        (str(row["j_id"]), str(row["r_id"])): float(row["cost"])
        for _, row in cost_2_c_df.iterrows()
    }

    return InputData(
        consumers=consumers,
        stage1=stage1,
        stage2=stage2,
        demand=demand,
        capacity_stage1=capacity_stage1,
        capacity_stage2=capacity_stage2,
        open_cost_stage1=open_cost_stage1,
        open_cost_stage2=open_cost_stage2,
        cost_1_2=cost_1_2,
        cost_2_c=cost_2_c,
    )

@dataclass
class Chromosome:
    genes_stage1: List[int]  
    genes_stage2: List[int]   

    fitness: Optional[float] = field(default=None, compare=False)

def random_chromosome(data: InputData) -> Chromosome:
    genes1 = [random.randint(0, 1) for _ in data.stage1]
    genes2 = [random.randint(0, 1) for _ in data.stage2]

    if sum(genes1) == 0:
        genes1[random.randrange(len(genes1))] = 1
    if sum(genes2) == 0:
        genes2[random.randrange(len(genes2))] = 1

    return Chromosome(genes_stage1=genes1, genes_stage2=genes2)

def opened_stage1(chrom: Chromosome, data: InputData) -> List[str]:
    return [data.stage1[i] for i, g in enumerate(chrom.genes_stage1) if g == 1]

def opened_stage2(chrom: Chromosome, data: InputData) -> List[str]:
    return [data.stage2[j] for j, g in enumerate(chrom.genes_stage2) if g == 1]

PENALTY = 1_000_000   

def compute_flows(
    data: InputData,
    active1: List[str],
    active2: List[str],
) -> Tuple[float, Dict[Tuple[str, str], float], Dict[Tuple[str, str], float]]:
    
    if not active1 or not active2:
        return float("inf"), {}, {}

    lp = pulp.LpProblem("flows", pulp.LpMinimize)

    f12 = pulp.LpVariable.dicts(
        "f12",
        [(i, j) for i in active1 for j in active2],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    f2c = pulp.LpVariable.dicts(
        "f2c",
        [(j, r) for j in active2 for r in data.consumers],
        lowBound=0,
        cat=pulp.LpContinuous,
    )
    unmet = pulp.LpVariable.dicts(
        "unmet",
        data.consumers,
        lowBound=0,
        cat=pulp.LpContinuous,
    )

    lp += (
        pulp.lpSum(data.cost_1_2[(i, j)] * f12[(i, j)] for i in active1 for j in active2)
        + pulp.lpSum(data.cost_2_c[(j, r)] * f2c[(j, r)] for j in active2 for r in data.consumers)
        + PENALTY * pulp.lpSum(unmet[r] for r in data.consumers)
    )

    for r in data.consumers:
        lp += (
            pulp.lpSum(f2c[(j, r)] for j in active2) + unmet[r] >= data.demand[r],
            f"demand_{r}",
        )

    for i in active1:
        lp += (
            pulp.lpSum(f12[(i, j)] for j in active2) <= data.capacity_stage1[i],
            f"cap1_{i}",
        )

    for j in active2:
        lp += (
            pulp.lpSum(f12[(i, j)] for i in active1) <= data.capacity_stage2[j],
            f"cap2_{j}",
        )

    for j in active2:
        lp += (
            pulp.lpSum(f2c[(j, r)] for r in data.consumers)
            <= pulp.lpSum(f12[(i, j)] for i in active1),
            f"balance_{j}",
        )

    solver = pulp.PULP_CBC_CMD(msg=False)
    lp.solve(solver)

    if pulp.LpStatus[lp.status] not in {"Optimal", "Feasible"}:
        return float("inf"), {}, {}

    transport_cost = pulp.value(lp.objective)

    result_f12 = {
        (i, j): pulp.value(f12[(i, j)])
        for i in active1 for j in active2
        if pulp.value(f12[(i, j)]) is not None and pulp.value(f12[(i, j)]) > 1e-7
    }
    result_f2c = {
        (j, r): pulp.value(f2c[(j, r)])
        for j in active2 for r in data.consumers
        if pulp.value(f2c[(j, r)]) is not None and pulp.value(f2c[(j, r)]) > 1e-7
    }

    return transport_cost, result_f12, result_f2c

def evaluate(chrom, data):
    active1 = opened_stage1(chrom, data)
    active2 = opened_stage2(chrom, data)
    
    open_cost = sum(data.open_cost_stage1[i] for i in active1) + \
                sum(data.open_cost_stage2[j] for j in active2)
    
    cap1 = {i: data.capacity_stage1[i] for i in active1}
    cap2 = {j: data.capacity_stage2[j] for j in active2}
    
    transport_cost = 0
    unmet_demand_penalty = 0

    for r in sorted(data.consumers, key=lambda x: data.demand[x], reverse=True):
        demand_left = data.demand[r]
        
        sorted_stage2 = sorted(active2, key=lambda j: data.cost_2_c[(j, r)])
        
        for j in sorted_stage2:
            if demand_left <= 0:
                break
            
            flow = min(demand_left, cap2[j])
            if flow > 0:
                transport_cost += flow * data.cost_2_c[(j, r)]
                cap2[j] -= flow
                demand_left -= flow
                
                sorted_stage1 = sorted(active1, key=lambda i: data.cost_1_2[(i, j)])
                flow_1_to_2 = flow
                
                for i in sorted_stage1:
                    if flow_1_to_2 <= 0:
                        break
                    flow_1 = min(flow_1_to_2, cap1[i])
                    if flow_1 > 0:
                        transport_cost += flow_1 * data.cost_1_2[(i, j)]
                        cap1[i] -= flow_1
                        flow_1_to_2 -= flow_1
                        
                if flow_1_to_2 > 0:
                    unmet_demand_penalty += flow_1_to_2 * PENALTY
                    
        if demand_left > 0:
            unmet_demand_penalty += demand_left * PENALTY

    chrom.fitness = open_cost + transport_cost + unmet_demand_penalty
    return chrom.fitness

def tournament_selection(
    population: List[Chromosome],
    tournament_size: int = 3,
) -> Chromosome:
    contestants = random.sample(population, min(tournament_size, len(population)))
    return min(contestants, key=lambda c: c.fitness)

def one_point_crossover(
    parent1: Chromosome,
    parent2: Chromosome,
) -> Tuple[Chromosome, Chromosome]:

    def cross(g1: List[int], g2: List[int]) -> Tuple[List[int], List[int]]:
        if len(g1) <= 1:
            return g1[:], g2[:]
        point = random.randint(1, len(g1) - 1)
        child1 = g1[:point] + g2[point:]
        child2 = g2[:point] + g1[point:]
        return child1, child2

    c1_genes1, c2_genes1 = cross(parent1.genes_stage1, parent2.genes_stage1)
    c1_genes2, c2_genes2 = cross(parent1.genes_stage2, parent2.genes_stage2)

    child1 = Chromosome(genes_stage1=c1_genes1, genes_stage2=c1_genes2)
    child2 = Chromosome(genes_stage1=c2_genes1, genes_stage2=c2_genes2)

    _ensure_at_least_one(child1)
    _ensure_at_least_one(child2)

    return child1, child2

def mutate(chrom: Chromosome, mutation_rate: float) -> Chromosome:
    mutated = deepcopy(chrom)

    for idx in range(len(mutated.genes_stage1)):
        if random.random() < mutation_rate:
            mutated.genes_stage1[idx] = 1 - mutated.genes_stage1[idx]

    for idx in range(len(mutated.genes_stage2)):
        if random.random() < mutation_rate:
            mutated.genes_stage2[idx] = 1 - mutated.genes_stage2[idx]

    _ensure_at_least_one(mutated)
    mutated.fitness = None  
    return mutated

def _ensure_at_least_one(chrom: Chromosome) -> None:
    if sum(chrom.genes_stage1) == 0:
        chrom.genes_stage1[random.randrange(len(chrom.genes_stage1))] = 1
    if sum(chrom.genes_stage2) == 0:
        chrom.genes_stage2[random.randrange(len(chrom.genes_stage2))] = 1

@dataclass
class GAParams:
    population_size: int  = 50      
    max_generations: int  = 100     
    crossover_rate: float = 0.8     
    mutation_rate: float  = 0.05    
    tournament_size: int  = 3       
    elitism: int          = 2       
    no_improve_limit: int = 20      
    seed: Optional[int]   = None    

def solve_genetic(
    data: InputData,
    params: GAParams = GAParams(),
) -> Solution:
  
    if params.seed is not None:
        random.seed(params.seed)

    start_time = time.time()

    population: List[Chromosome] = [
        random_chromosome(data) for _ in range(params.population_size)
    ]

    for chrom in population:
        evaluate(chrom, data)

    best: Chromosome = min(population, key=lambda c: c.fitness)
    best = deepcopy(best)

    no_improve_count = 0
    history: List[float] = [best.fitness]

    for generation in range(1, params.max_generations + 1):

        population.sort(key=lambda c: c.fitness)
        new_population: List[Chromosome] = [
            deepcopy(population[k]) for k in range(params.elitism)
        ]

        while len(new_population) < params.population_size:

            parent1 = tournament_selection(population, params.tournament_size)
            parent2 = tournament_selection(population, params.tournament_size)

            if random.random() < params.crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                child1 = deepcopy(parent1)
                child2 = deepcopy(parent2)

            child1 = mutate(child1, params.mutation_rate)
            child2 = mutate(child2, params.mutation_rate)

            evaluate(child1, data)
            evaluate(child2, data)

            new_population.append(child1)
            if len(new_population) < params.population_size:
                new_population.append(child2)

        population = new_population

        generation_best = min(population, key=lambda c: c.fitness)
        if generation_best.fitness < best.fitness:
            best = deepcopy(generation_best)
            no_improve_count = 0
        else:
            no_improve_count += 1

        history.append(best.fitness)

        if no_improve_count >= params.no_improve_limit:
            print(f"Stop on generation {generation}")
            break

    solve_time = time.time() - start_time


    active1 = opened_stage1(best, data)
    active2 = opened_stage2(best, data)

    _, flow_1_2, flow_2_c = compute_flows(data, active1, active2)

    open_cost = (
        sum(data.open_cost_stage1[i] for i in active1)
        + sum(data.open_cost_stage2[j] for j in active2)
    )
    transport_cost = sum(
        data.cost_1_2[(i, j)] * v for (i, j), v in flow_1_2.items()
    ) + sum(
        data.cost_2_c[(j, r)] * v for (j, r), v in flow_2_c.items()
    )

    return Solution(
        objective_value=open_cost + transport_cost,
        solve_time=solve_time,
        opened_stage1=active1,
        opened_stage2=active2,
        flow_1_2=flow_1_2,
        flow_2_c=flow_2_c,
    )


def print_solution(solution: Solution) -> None:
    print(f"Total cost: {solution.objective_value:.2f}")
    print(f"Time: {solution.solve_time:.4f} seconds")

    print("\nOpened stage 1 plants:")
    for i in solution.opened_stage1:
        print(f"  {i}")

    print("\nOpened stage 2 plants:")
    for j in solution.opened_stage2:
        print(f"  {j}")

    print("\nFlows from Stage 1 to Stage 2:")
    for (i, j), v in sorted(solution.flow_1_2.items()):
        print(f"  {i} -> {j}: {v:.2f}")

    print("\nFlows from Stage 2 to consumers:")
    for (j, r), v in sorted(solution.flow_2_c.items()):
        print(f"  {j} -> {r}: {v:.2f}")


def save_solution(solution: Solution, output_dir: str, prefix: str = "ga") -> None:
    os.makedirs(output_dir, exist_ok=True)

    pd.DataFrame([{
        "objective_value": solution.objective_value,
        "solve_time_seconds": solution.solve_time,
        "opened_stage1_count": len(solution.opened_stage1),
        "opened_stage2_count": len(solution.opened_stage2),
    }]).to_csv(os.path.join(output_dir, f"{prefix}_summary.csv"), index=False)

    pd.DataFrame({"id": solution.opened_stage1}).to_csv(
        os.path.join(output_dir, f"{prefix}_opened_stage1.csv"), index=False
    )
    pd.DataFrame({"id": solution.opened_stage2}).to_csv(
        os.path.join(output_dir, f"{prefix}_opened_stage2.csv"), index=False
    )
    pd.DataFrame(
        [{"i_id": i, "j_id": j, "flow": v} for (i, j), v in solution.flow_1_2.items()]
    ).to_csv(os.path.join(output_dir, f"{prefix}_flow_1_2.csv"), index=False)

    pd.DataFrame(
        [{"j_id": j, "r_id": r, "flow": v} for (j, r), v in solution.flow_2_c.items()]
    ).to_csv(os.path.join(output_dir, f"{prefix}_flow_2_C.csv"), index=False)


def main() -> None:
    data_dir = "data"
    output_dir = "results"

    datasets_to_process = ["small", "medium", "large"]

    params = GAParams(
        population_size=50,
        max_generations=100,
        crossover_rate=0.8,
        mutation_rate=0.05,
        tournament_size=3,
        elitism=2,
        no_improve_limit=20,
        seed=42,
    )

    for dataset_name in datasets_to_process:
        full_dataset_path = os.path.join(data_dir, dataset_name)

        print("=" * 50)
        print(f"Solution for dataset: {dataset_name}")
        print("=" * 50)
        data = read_input_data(full_dataset_path)

        solution = solve_genetic(data, params)

        print_solution(solution)

        save_solution(solution, output_dir, prefix=f"ga_{dataset_name}")

if __name__ == "__main__":
    main()