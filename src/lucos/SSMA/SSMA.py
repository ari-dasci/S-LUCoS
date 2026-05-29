


import os
import logging
import copy
import numpy as np
from tabpfn import TabPFNClassifier
from tabpfn.constants import ModelVersion
from lucos.SSMA.Chromosome import Chromosome
from lucos.SSMA.ssma_plots import plot_ssma_fit_histories
from lucos.SSMA.ssma_utils import load_json_file, save_json_atomic
import lucos.SSMA.ssma_paths as ssma_paths

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)


# ==========================================
# SSMA CLASS (MAIN ALGORITHM)
# Equivalent to SSMA.java
# https://github.com/SCI2SUGR/KEEL/blob/master/src/keel/Algorithms/Instance_Selection/SSMA/SSMA.java
# ==========================================

# TIMES:
# - electricity, fold0, 350 steps: 14 hours
class SSMA_TabPFN:
    def __init__(self, 
                 dataset_name,
                 pop_size=20, 
                 p_cross=0.5, 
                 p_mut=0.01,
                 local_search_attempts=10,
                 patience=50,
                 sizes_we_want_to_reduce_to=[],
                 alphas=[0.5],
                 val_size_split=0.2,
                 steps_per_alpha=500,
                 reset_steps_without_improvement=False,
                 load_until_step=None):
        
        self.steps_per_chromosomes_log = 20
        self.dataset_name = dataset_name
        self.pop_size = pop_size
        self.p_cross = p_cross
        self.p_mut = p_mut
        self.local_search_attempts = local_search_attempts 
        self.patience = patience       
        self.sizes_we_want_to_reduce_to = sizes_we_want_to_reduce_to
        self.alphas = alphas
        self.steps_per_alpha = steps_per_alpha
        self.reset_steps_without_improvement = reset_steps_without_improvement
        self.step = 0
        self.min_size_reached_in_population = np.inf
        self.best_chromosomes_curve = {}  # dict {size: Chromosome}
        self.fit_histories = {}  # dict {initial_selection_size: {'step': int, 'steps_without_improvement': int, 'population': [list of populations per step]}}
        self.fit_histories_path = ssma_paths.ssma_fit_histories_folder / f'{self.dataset_name}_fit_histories.json'
        self.best_chromosomes_curve_path = ssma_paths.ssma_best_chromosomes_curve_folder / f'{self.dataset_name}_best_chromosomes_curve.json'
        self.classifier = TabPFNClassifier.create_default_for_version(ModelVersion.V2_5, device='cuda', n_estimators=1, ignore_pretraining_limits=True) # 1 estimator to make it faster
        self.val_size_split = val_size_split

        self.load_checkpoint(load_until_step)
        
        
    def check_new_chromosomes_improve_curve(self, chromosomes_to_add):
        """
        Check if any of the new chromosomes improve the best_chromosomes curve.
        We keep a new chromosome if:

        chromosomes_to_add: list of Chromosome
        """
        new_best_chromosome_found = False

        for new_chrom in chromosomes_to_add:

            biggerorequal_and_worse_chromosomes = {s: c for s, c in self.best_chromosomes_curve.items() if s >= new_chrom.count_active() and c.val_auc < new_chrom.val_auc}
            smallerorequal_and_better_chromosomes = {s: c for s, c in self.best_chromosomes_curve.items() if s <= new_chrom.count_active() and c.val_auc > new_chrom.val_auc}

            if (biggerorequal_and_worse_chromosomes or len(smallerorequal_and_better_chromosomes) == 0) and new_chrom.val_auc > 0.0:
                # REMOVE bigger and worse chromosomes
                for s, c in biggerorequal_and_worse_chromosomes.items():
                    del self.best_chromosomes_curve[s]
                # ADD new chromosome and sort by size (decreasing)
                self.best_chromosomes_curve[new_chrom.count_active()] = copy.deepcopy(new_chrom)
                self.best_chromosomes_curve = dict(sorted(self.best_chromosomes_curve.items(), reverse=True))
                new_best_chromosome_found = True
                self.log(f"New best chrom in curve:  fitness={new_chrom.fitness:.3f} | val_auc={new_chrom.val_auc:.3f} | test_auc={new_chrom.test_auc:.3f} | reduction={new_chrom.reduction:.3f} | size={new_chrom.count_active()} | ir={new_chrom.ir:.3f}")

        return new_best_chromosome_found


    def append_step_to_history(self, population, steps_without_improvement):
        self.fit_histories[self.current_initial_selection_size]['step'] = self.step
        self.fit_histories[self.current_initial_selection_size]['steps_without_improvement'] = steps_without_improvement
        self.fit_histories[self.current_initial_selection_size]['population'].append(copy.deepcopy(population))


    def fit(self, X_train, y_train, X_val, y_val, X_test, y_test):
        """
        return fit_histories
        """
        logger.debug(f"Starting SSMA fit for dataset: {self.dataset_name}")
        logger.debug(f"    Train data - X: {X_train.shape}, y: {y_train.shape}")
        logger.debug(f"    Val data   - X: {X_val.shape}, y: {y_val.shape}")
        logger.debug(f"    Test data  - X: {X_test.shape}, y: {y_test.shape}")
        logger.debug(f"    Number of classes: {len(np.unique(y_train))}")

        sizes_left = self.sizes_we_want_to_reduce_to.copy()
        logger.debug(f"sizes_left: {sizes_left}")
        population = None
        while sizes_left:
            size = sizes_left.pop(0)
            self.current_initial_selection_size = size
            for i, alpha in enumerate(self.alphas):
                steps = self.steps_per_alpha * (i + 1)
                logger.debug(f"Starting SSMA fit with size={size}, alpha={alpha}, steps={steps}")
                population = self._fit(X_train, y_train, X_val, y_val, X_test, y_test, steps, alpha, start_population=population)
            sizes_left = [s for s in sizes_left if s < self.min_size_reached_in_population]
            logger.debug(f"Sizes left to reach: {sizes_left}")

            # force reduce population to size_left[0] if any left
            if sizes_left:
                logger.debug(f"Reducing population to next size: {sizes_left[0]}")
                for chrom in population:
                    # If the chromosome is larger than the next size, randomly reduce genes
                    chrom.random_reduce_to_size(sizes_left[0], y_train)

        logger.debug(f"SSMA fitting completed. All initial sizes processed. Saving checkpoint")
        logger.debug(f"Final best_chromosomes_curve contains {len(self.best_chromosomes_curve)} chromosomes")

        return self.fit_histories


    def _resume_from_checkpoint(self, steps, initial_selection_size):
        """
        Try to resume training from checkpoint.
        NOTE: now that there is an alpha vector, execution will start with the first alpha in the vector even if training already ran with that alpha
        
        Returns:
            tuple: (population, start_step, steps_without_improvement)
                   or None if cannot resume or training already complete
        """
        has_checkpoint = initial_selection_size in self.fit_histories and self.fit_histories[initial_selection_size]['step'] > 0

        if not has_checkpoint:
            logger.debug(f"No checkpoint found for initial_selection_size={initial_selection_size}")
            return None

        saved_step = self.fit_histories[initial_selection_size]['step']
        saved_steps_without_improvement = self.fit_histories[initial_selection_size]['steps_without_improvement']
        saved_population = self.fit_histories[initial_selection_size]['population'][-1]
        start_step = saved_step + 1
        self.step = saved_step
        
        logger.debug(f"Resuming from checkpoint at step {start_step} with {len(saved_population)} chromosomes")
        
        return saved_population, start_step, saved_steps_without_improvement

    def log(self, message):
        logger.debug(f"{self.dataset_name} | Init Size {self.current_initial_selection_size} | Step {self.step} | {message}")

    def _fit(self, X_train, y_train, X_val, y_val, X_test, y_test, steps, alpha=0.5, start_population=None):
        n_genes = X_train.shape[0]
        self.log(f"Starting _fit with {n_genes} genes, {self.pop_size} population size, steps={steps}, initial_size={self.current_initial_selection_size}, alpha={alpha}")
        
        # Check if there's a checkpoint to resume from
        resume_result = self._resume_from_checkpoint(steps, self.current_initial_selection_size)
        if resume_result is not None:
            # Resume from checkpoint
            population, start_step, steps_without_improvement = resume_result
            if self.reset_steps_without_improvement:
                steps_without_improvement = 0
                self.log(f"Resetting steps_without_improvement to 0 as requested")
            if start_step >= steps:
                self.log(f"Training already completed up to requested steps ({steps}). Skipping _fit.")
                return population
        else:
            # Start fresh
            self.log(f"No checkpoint found, starting fresh")
            self.log(f"Initializing population of {self.pop_size} chromosomes")
            population = [Chromosome(n_genes, self.current_initial_selection_size, y_train=y_train, val_size_split=self.val_size_split) for _ in range(self.pop_size)]
            start_step = 0
            self.step = 0
            steps_without_improvement = 0
            self.fit_histories[self.current_initial_selection_size] = {
                'step': 0,
                'steps_without_improvement': 0,
                'population': []
            }

        if start_population is not None:
            # Use the provided start_population to initialize
            self.log(f"Using provided start_population to initialize")
            population = copy.deepcopy(start_population)

        # Initial evaluation
        self.log(f"Performing initial evaluation of population")
        for chrom in population:
            chrom.evaluate(self.classifier, X_train, y_train, X_val, y_val, X_test, y_test, alpha)
        population.sort(key=lambda x: x.fitness, reverse=True)
        self.check_new_chromosomes_improve_curve(population)
        self.log_population_and_curve(population)

        current_best_fitness = population[0].fitness
        self.min_size_reached_in_population = min(c.count_active() for c in population)

        # Main loop (Steady-State)
        self.log(f"Starting main SSMA loop with {steps} steps (from step {start_step})")
        while self.step < steps and steps_without_improvement < self.patience:
            self.step += 1
            # --- SELECTION (Binary Tournament) ---
            # Select parent 1
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            parent1 = population[idx1] if population[idx1].fitness > population[idx2].fitness else population[idx2]
            
            # Select parent 2
            idx1, idx2 = np.random.choice(self.pop_size, 2, replace=False)
            parent2 = population[idx1] if population[idx1].fitness > population[idx2].fitness else population[idx2]
            
            self.log(f"Parent 1 selected:        fitness={parent1.fitness:.3f} | val_auc={parent1.val_auc:.3f} | test_auc={parent1.test_auc:.3f} | reduction={parent1.reduction:.3f} | size={parent1.count_active()} | ir={parent1.ir:.3f}")
            self.log(f"Parent 2 selected:        fitness={parent2.fitness:.3f} | val_auc={parent2.val_auc:.3f} | test_auc={parent2.test_auc:.3f} | reduction={parent2.reduction:.3f} | size={parent2.count_active()} | ir={parent2.ir:.3f}")

            # --- CROSSOVER (Uniform or Single Point) ---
            child1_genes = np.copy(parent1.genes)
            child2_genes = np.copy(parent2.genes)
            
            crossover_applied = False
            if np.random.rand() < self.p_cross:
                # Uniform crossover
                # NOTE: crossover does not guarantee that there is one instance of each class; mutation handles that
                mask = np.random.rand(n_genes) < 0.5
                child1_genes[mask] = parent2.genes[mask]
                child2_genes[mask] = parent1.genes[mask]
                crossover_applied = True
                
            child1 = Chromosome(n_genes, initial_selection_size=self.current_initial_selection_size, genes=child1_genes, y_train=y_train, val_size_split=self.val_size_split)
            child2 = Chromosome(n_genes, initial_selection_size=self.current_initial_selection_size, genes=child2_genes, y_train=y_train, val_size_split=self.val_size_split)
            
            if crossover_applied:
                self.log(f"Crossover applied | New Children sizes: C1={child1.count_active()}, C2={child2.count_active()}")
            else:
                self.log("Crossover skipped")
                
            # --- MUTATION ---
            adds_c1, removes_c1 = child1.mutate(self.p_mut, y_train)
            adds_c2, removes_c2 = child2.mutate(self.p_mut, y_train)
            
            # --- BASIC CHILD EVALUATION ---
            c1_old_fitness = child1.fitness
            c2_old_fitness = child2.fitness
            child1.evaluate(self.classifier, X_train, y_train, X_val, y_val, X_test, y_test, alpha)
            child2.evaluate(self.classifier, X_train, y_train, X_val, y_val, X_test, y_test, alpha)
            self.log(f"C1 mut (f{child1.fitness - c1_old_fitness:.3f}) ({adds_c1}+, {removes_c1}-): fitness={child1.fitness:.3f} | val_auc={child1.val_auc:.3f} | test_auc={child1.test_auc:.3f} | reduction={child1.reduction:.3f} | size={child1.count_active()} | ir={child1.ir:.3f}")
            self.log(f"C2 mut (f{child2.fitness - c2_old_fitness:.3f}) ({adds_c2}+, {removes_c2}-): fitness={child2.fitness:.3f} | val_auc={child2.val_auc:.3f} | test_auc={child2.test_auc:.3f} | reduction={child2.reduction:.3f} | size={child2.count_active()} | ir={child2.ir:.3f}")
            
            # --- LOCAL SEARCH (MEMETIC) ---
            # Apply only if the child looks promising (heuristic to save time). If it is better than the worst in the population
            # or with a small probability (as in the Java "Random < 0.0625")
            # Local search is one of the most expensive parts (10 TabPFN evaluations)
            # Note: In strict SSMA, each local search step counts as an evaluation.
            if child1.fitness > population[-1].fitness or np.random.rand() < 0.0625 or child1.count_active() < 50:
                # TODO: after many steps, children often no longer improve the population and rarely enter here.
                (adds_c1, removes_c1), delta_c1 = child1.local_search(X_train, y_train, X_val, y_val, X_test, y_test, self.classifier, alpha, attempts=self.local_search_attempts)
                self.log(f"C1 LS (f+{delta_c1:.3f}) ({adds_c1}+, {removes_c1}-): fitness={child1.fitness:.3f} | val_auc={child1.val_auc:.3f} | test_auc={child1.test_auc:.3f} | reduction={child1.reduction:.3f} | size={child1.count_active()} | ir={child1.ir:.3f}")
            else:
                self.log(f"C1 LS skipped (fitness {child1.fitness:.3f} <= worst population fitness {population[-1].fitness:.3f}) or not small enough size")

            if child2.fitness > population[-1].fitness or np.random.rand() < 0.0625 or child2.count_active() < 50: 
                (adds_c2, removes_c2), delta_c2 = child2.local_search(X_train, y_train, X_val, y_val, X_test, y_test, self.classifier, alpha, attempts=self.local_search_attempts)
                self.log(f"C2 LS (f+{delta_c2:.3f}) ({adds_c2}+, {removes_c2}-): fitness={child2.fitness:.3f} | val_auc={child2.val_auc:.3f} | test_auc={child2.test_auc:.3f} | reduction={child2.reduction:.3f} | size={child2.count_active()} | ir={child2.ir:.3f}")
            else:
                self.log(f"C2 LS skipped (fitness {child2.fitness:.3f} <= worst population fitness {population[-1].fitness:.3f}) or not small enough size")


            # --- REPLACEMENT (Steady State) ---
            # Children replace the worst individuals in the population if they are better
            # Keep the best N individuals (where N is the population size)
            population += [child1, child2]
            population.sort(key=lambda x: x.fitness, reverse=True)
            population = population[:self.pop_size]

            # Check whether the new children are part of the best chromosome curve
            self.check_new_chromosomes_improve_curve([child1, child2])

            self.min_size_reached_in_population = min(self.min_size_reached_in_population, child1.count_active(), child2.count_active())

            if population[0].fitness > current_best_fitness:
                self.log(f"New best population fitness: {population[0].fitness:.4f} (+{(population[0].fitness - current_best_fitness):.4f})")
                current_best_fitness = population[0].fitness
                steps_without_improvement = 0
            else:
                steps_without_improvement += 1

            self.append_step_to_history(population, steps_without_improvement)

            # Save checkpoint
            self.save_checkpoint()

            if self.step % self.steps_per_chromosomes_log == 0:
                self.log(f"No improvement for {steps_without_improvement} steps")
                self.log_population_and_curve(population)
                plot_ssma_fit_histories(self.dataset_name, self.fit_histories, X_train.shape[0])

            if steps_without_improvement >= self.patience:
                self.log(f"Early stopping: No improvement in {self.patience} steps.")
                self.log_population_and_curve(population)
                break # the break is not necessary; this is also in the while condition

            logger.debug("")

        # End of loop
        self.log(f"SSMA run completed!!!")
        self.log(f"min_size_reached_in_population: {self.min_size_reached_in_population}")
        self.log_population_and_curve(population)
        self.save_checkpoint()
        logger.debug("\n\n\n")
        return population
        

    def log_population_and_curve(self, population):
        
        self.log(f"Best Chromosome Curve:")
        for size, chrom in self.best_chromosomes_curve.items():
            logger.debug(f"  size={size} | fitness={chrom.fitness:.3f} | val_auc={chrom.val_auc:.3f} | test_auc={chrom.test_auc:.3f} | reduction={chrom.reduction:.3f} | ir={chrom.ir:.3f}")
        
        self.log(f"Current Population:")
        for i, chrom in enumerate(population):
            logger.debug(f"  Population Chromosome {i}: fitness={chrom.fitness:.3f} | val_auc={chrom.val_auc:.3f} | test_auc={chrom.test_auc:.3f} | reduction={chrom.reduction:.3f} | size={chrom.count_active()} | ir={chrom.ir:.3f}")
        

    def save_checkpoint(self):
        save_json_atomic(self.fit_histories_path, self.fit_histories)
        save_json_atomic(self.best_chromosomes_curve_path, self.best_chromosomes_curve)


    def load_checkpoint(self, load_history_until_step=None):
            
        if os.path.exists(self.fit_histories_path) and os.path.exists(self.best_chromosomes_curve_path):
            logger.debug(f"Loading checkpoint from {self.fit_histories_path} and {self.best_chromosomes_curve_path}")

            self.fit_histories = load_json_file(self.fit_histories_path)
            self.best_chromosomes_curve = load_json_file(self.best_chromosomes_curve_path)

            if load_history_until_step is not None:
                # Truncate fit histories to only include up to from_step
                for size, history in self.fit_histories.items():
                    if history['step'] > load_history_until_step:
                        history['step'] = load_history_until_step
                        last_step_with_improvement = history['step'] - history['steps_without_improvement']
                        history['steps_without_improvement'] = load_history_until_step - last_step_with_improvement
                        history['population'] = history['population'][:load_history_until_step + 1]
                    else:
                        logger.debug(f"Size {size} fit history step {history['step']} is before load_until_step {load_history_until_step}, no truncation needed.")

            logger.debug(f"Loaded {len(self.best_chromosomes_curve)} best chromosomes in curve from checkpoint")
        else:
            logger.debug(f"No checkpoint found. Starting fresh SSMA instance.")
