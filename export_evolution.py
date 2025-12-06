"""
SPEA2 Biomarker Optimization - Export Evolution Data for Web Demo
VERSION WITH INDIVIDUAL PARETO SOLUTIONS for DNA animation
"""

import json
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
import random
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# DATA LOADING
# =============================================================================

def load_golub_data():
    """Load simulated Golub-like Leukemia dataset with REALISTIC difficulty."""
    np.random.seed(42)
    n_samples = 72
    n_genes = 200
    
    X = np.random.randn(n_samples, n_genes) * 2.0
    y = np.array([0] * 47 + [1] * 25)
    
    strong_markers = list(range(0, 10))
    weak_markers = list(range(10, 25))
    
    for g in strong_markers:
        effect = np.random.uniform(0.8, 1.5)
        X[:47, g] += effect
        X[47:, g] -= effect
        X[:, g] += np.random.randn(n_samples) * 1.2
    
    for g in weak_markers:
        effect = np.random.uniform(0.2, 0.5)
        X[:47, g] += effect
        X[47:, g] -= effect
        X[:, g] += np.random.randn(n_samples) * 1.5
    
    gene_names = [
        "M63138_at", "U82311_at", "X13238_at", "X66401_cds1_at", "Y07604_at",
        "X16546_at", "M23197_at", "U05259_rna1_at", "M27891_at", "M84526_at",
        "L09209_s_at", "U50136_rna1_at", "M31523_at", "M96326_rna1_s_at",
        "HG1612-HT1612_at", "U46499_at", "M92287_at", "M55150_at", "D88270_at",
        "M83667_rna1_s_at"
    ]
    gene_names = gene_names + [f"Gene_{i:04d}_at" for i in range(20, n_genes)]
    
    return X, y, gene_names

# =============================================================================
# SPEA2 ALGORITHM
# =============================================================================

class Individual:
    def __init__(self, genes):
        self.genes = genes
        self.accuracy = 0.0
        self.gene_count = 0
        self.is_pareto = False

def evaluate_individual(ind, X, y, scaler, k_min, k_max):
    selected = np.where(ind.genes)[0]
    ind.gene_count = len(selected)
    
    if ind.gene_count < k_min or ind.gene_count > k_max:
        ind.accuracy = 0.0
        return
    
    X_subset = X[:, selected]
    X_scaled = scaler.fit_transform(X_subset)
    
    clf = KNeighborsClassifier(n_neighbors=3)
    try:
        scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
        ind.accuracy = scores.mean()
    except:
        ind.accuracy = 0.0

def dominates(ind1, ind2):
    better_acc = ind1.accuracy >= ind2.accuracy
    fewer_genes = ind1.gene_count <= ind2.gene_count
    strictly_better = ind1.accuracy > ind2.accuracy or ind1.gene_count < ind2.gene_count
    return better_acc and fewer_genes and strictly_better

def update_pareto_status(population):
    for ind in population:
        ind.is_pareto = True
        for other in population:
            if ind is not other and dominates(other, ind):
                ind.is_pareto = False
                break

def tournament_selection(population, tournament_size=3):
    tournament = random.sample(population, min(tournament_size, len(population)))
    return max(tournament, key=lambda x: x.accuracy)

def crossover(parent1, parent2):
    child_genes = np.array([
        p1 if random.random() < 0.5 else p2 
        for p1, p2 in zip(parent1.genes, parent2.genes)
    ])
    return Individual(child_genes)

def mutate(ind, mut_prob=0.02):
    for i in range(len(ind.genes)):
        if random.random() < mut_prob:
            ind.genes[i] = not ind.genes[i]

def run_spea2(X, y, config, gene_names):
    """Run SPEA2 and record individual Pareto solutions."""
    n_genes = X.shape[1]
    scaler = StandardScaler()
    
    pop_size = config.get('pop_size', 50)
    max_gen = config.get('max_gen', 50)
    k_min = config.get('k_min', 5)
    k_max = config.get('k_max', 25)
    crossover_prob = config.get('crossover_prob', 0.8)
    mutation_prob = config.get('mutation_prob', 0.1)
    
    target_k = (k_min + k_max) / 2
    prob_init = max(0.05, min(0.3, target_k / n_genes))
    
    population = []
    for _ in range(pop_size):
        genes = np.random.random(n_genes) < prob_init
        population.append(Individual(genes))
    
    for ind in population:
        evaluate_individual(ind, X, y, scaler, k_min, k_max)
    
    history = []
    all_pareto_fronts = []
    all_pareto_solutions = []  # NEW: Store individual solutions with their genes
    
    print(f"Running SPEA2 for {max_gen} generations...")
    
    for gen in range(max_gen):
        update_pareto_status(population)
        
        valid = [ind for ind in population if ind.accuracy > 0]
        if valid:
            best_acc = max(ind.accuracy for ind in valid)
            avg_acc = np.mean([ind.accuracy for ind in valid])
            best_ind = max(valid, key=lambda x: x.accuracy)
            best_genes = best_ind.gene_count
            avg_genes = np.mean([ind.gene_count for ind in valid])
            
            pareto = [ind for ind in valid if ind.is_pareto]
            dominated = [ind for ind in valid if not ind.is_pareto][:10]
            
            # Pareto points for chart
            pareto_points = [
                {"x": ind.gene_count, "y": round(ind.accuracy, 4), "isPareto": True}
                for ind in pareto
            ] + [
                {"x": ind.gene_count, "y": round(ind.accuracy, 4), "isPareto": False}
                for ind in dominated
            ]
            
            # NEW: Individual Pareto solutions with their selected genes
            pareto_solutions = []
            for ind in pareto:
                selected_indices = np.where(ind.genes)[0].tolist()
                selected_names = [gene_names[i] if i < len(gene_names) else f"Gene_{i}" 
                                  for i in selected_indices[:20]]  # Limit to 20 genes
                pareto_solutions.append({
                    "accuracy": round(ind.accuracy, 4),
                    "geneCount": ind.gene_count,
                    "selectedGeneIndices": selected_indices[:50],  # Limit for JSON size
                    "selectedGeneNames": selected_names
                })
        else:
            best_acc = avg_acc = 0.5
            best_genes = k_min
            avg_genes = target_k
            pareto_points = []
            pareto_solutions = []
        
        history.append({
            "generation": gen + 1,
            "bestAccuracy": round(best_acc, 4),
            "avgAccuracy": round(avg_acc, 4),
            "bestGenes": int(best_genes),
            "avgGenes": round(avg_genes, 2)
        })
        
        all_pareto_fronts.append(pareto_points)
        all_pareto_solutions.append(pareto_solutions)  # NEW
        
        if (gen + 1) % 10 == 0:
            print(f"  Gen {gen + 1}: Best Acc = {best_acc:.4f}, Best Genes = {best_genes}, Pareto size = {len(pareto_solutions)}")
        
        offspring = []
        while len(offspring) < pop_size:
            parent1 = tournament_selection(population)
            parent2 = tournament_selection(population)
            
            if random.random() < crossover_prob:
                child = crossover(parent1, parent2)
            else:
                child = Individual(parent1.genes.copy())
            
            if random.random() < mutation_prob:
                mutate(child)
            
            evaluate_individual(child, X, y, scaler, k_min, k_max)
            offspring.append(child)
        
        combined = population + offspring
        update_pareto_status(combined)
        combined.sort(key=lambda x: (-x.accuracy, x.gene_count))
        population = combined[:pop_size]
    
    valid_final = [ind for ind in population if ind.accuracy > 0]
    final_pareto = [ind for ind in valid_final if ind.is_pareto]
    best = max(population, key=lambda x: x.accuracy) if population else None
    
    gene_frequencies = np.zeros(n_genes)
    if final_pareto:
        for ind in final_pareto:
            gene_frequencies += ind.genes.astype(float)
        gene_frequencies /= len(final_pareto)
    
    return {
        "history": history,
        "paretoFronts": all_pareto_fronts,
        "paretoSolutions": all_pareto_solutions,  # NEW
        "finalPareto": [
            {"x": ind.gene_count, "y": round(ind.accuracy, 4), "isPareto": True}
            for ind in final_pareto
        ],
        "geneFrequencies": gene_frequencies.tolist(),
        "bestSolution": {
            "accuracy": round(best.accuracy, 4) if best else 0,
            "geneCount": best.gene_count if best else 0,
            "selectedGenes": np.where(best.genes)[0].tolist() if best else []
        } if best else None
    }

def calculate_baseline(X, y):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = KNeighborsClassifier(n_neighbors=3)
    scores = cross_val_score(clf, X_scaled, y, cv=5, scoring='accuracy')
    return {
        "accuracy": round(scores.mean(), 4),
        "geneCount": X.shape[1]
    }

def main():
    print("=" * 60)
    print("SPEA2 Biomarker Optimization - WITH INDIVIDUAL SOLUTIONS")
    print("=" * 60)
    
    print("\n1. Loading simulated Golub-like data...")
    X, y, gene_names = load_golub_data()
    print(f"   Samples: {X.shape[0]}, Genes: {X.shape[1]}")
    
    print("\n2. Calculating baseline...")
    baseline = calculate_baseline(X, y)
    print(f"   Baseline: {baseline['accuracy']*100:.1f}% with {baseline['geneCount']} genes")
    
    print("\n3. Running SPEA2...")
    config = {
        'pop_size': 50,
        'max_gen': 50,
        'k_min': 5,
        'k_max': 25,
        'crossover_prob': 0.8,
        'mutation_prob': 0.1
    }
    
    results = run_spea2(X, y, config, gene_names)
    
    output = {
        "baseline": baseline,
        "config": config,
        "totalGenes": len(gene_names),
        "geneNames": gene_names[:50],
        "generations": results["history"],
        "paretoFronts": results["paretoFronts"],
        "paretoSolutions": results["paretoSolutions"],  # NEW
        "finalPareto": results["finalPareto"],
        "geneFrequencies": results["geneFrequencies"][:50],
        "bestSolution": results["bestSolution"]
    }
    
    output_path = "biomarker-demo/public/evolution_data.json"
    print(f"\n4. Saving to {output_path}...")
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print("\n" + "=" * 60)
    print("DONE! Now includes individual Pareto solutions with genes!")
    print("=" * 60)

if __name__ == "__main__":
    main()
