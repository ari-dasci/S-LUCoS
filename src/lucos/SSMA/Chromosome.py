
import logging
import numpy as np
from sklearn.metrics import accuracy_score
from lucos.utils.evaluation import auc_score, imbalance_ratio
from sklearn.model_selection import train_test_split    

TABPFN_MAX_SAMPLES = 20000  # Approximate TabPFN limit for number of samples

logger = logging.getLogger("lucos")
logger.setLevel(logging.DEBUG)

# ==========================================
# CHROMOSOME CLASS
# Equivalent to Cromosoma.java
# ==========================================
class Chromosome():
    def __init__(self, n_genes, initial_selection_size, genes=None, y_train=None, val_size_split=0.2):
        self.n_genes = n_genes
        self.fitness = -1.0
        self.val_auc = 0.0
        self.val_acc = 0.0
        self.test_auc = 0.0
        self.test_acc = 0.0
        self.reduction = 0.0
        self.log_reduction = 0.0
        self.linear_reduction = 0.0
        self.alpha = 0.0
        self.ir = 0.0
        self.n_classes = len(np.unique(y_train)) if y_train is not None else 0
        self.initial_selection_size = initial_selection_size # this has no value if we make a copy (a child)
        self.max_selection_size = 500 # Hardcoded. This is the size over which reduction is calculated
        self.val_size_split = val_size_split

        # Random initialization or copy
        if genes is None:
            # Initialize while ensuring it does not exceed the TabPFN limit
            if initial_selection_size > TABPFN_MAX_SAMPLES:
                logger.warning(f"Initial selection size {initial_selection_size} exceeds TabPFN max samples {TABPFN_MAX_SAMPLES}. Adjusting initial gene probability.")
                initial_selection_size = TABPFN_MAX_SAMPLES
                self.genes = np.ones(n_genes, dtype=int)
            else:
                # Ensure at least one instance per class is selected
                self.genes = np.zeros(n_genes, dtype=int)
                selected_indices = set()
                
                if y_train is not None:
                    # First, select at least one instance per class
                    for class_label in np.unique(y_train):
                        class_indices = np.where(y_train == class_label)[0]
                        selected_idx = np.random.choice(class_indices)
                        selected_indices.add(selected_idx)
                
                # Then, randomly select remaining instances to reach initial_selection_size
                n_remaining = initial_selection_size - len(selected_indices)
                if n_remaining > 0:
                    available_indices = [i for i in range(n_genes) if i not in selected_indices]
                    if len(available_indices) > 0:
                        remaining_indices = np.random.choice(available_indices, 
                                                            size=min(n_remaining, len(available_indices)), 
                                                            replace=False)
                        selected_indices.update(remaining_indices)
                
                self.genes[list(selected_indices)] = 1
        else:
            self.genes = np.array(genes, copy=True)

    def __setstate__(self, state):
        # Function to allow backward compatibility when loading saved chromosomes
        # variables that we did not save at the time, so we save them now
        if 'alpha' not in state:
            state['alpha'] = 0.5
        if 'n_classes' not in state:
            state['n_classes'] = 0
        if 'linear_reduction' not in state:
            if 'old_reduction' in state:
                state['linear_reduction'] = state['old_reduction']
            else:
                state['linear_reduction'] = 0.0
        self.__dict__.update(state)

    def to_dict(self):
        return {
            'n_genes': int(self.n_genes),
            'initial_selection_size': int(self.initial_selection_size),
            'genes': self.genes.tolist() if isinstance(self.genes, np.ndarray) else list(self.genes),
            'fitness': float(self.fitness),
            'val_auc': float(self.val_auc),
            'val_acc': float(self.val_acc),
            'test_auc': float(self.test_auc),
            'test_acc': float(self.test_acc),
            'reduction': float(self.reduction),
            'log_reduction': float(self.log_reduction),
            'linear_reduction': float(self.linear_reduction),
            'alpha': float(self.alpha),
            'ir': float(self.ir),
            'n_classes': int(self.n_classes),
            'max_selection_size': int(self.max_selection_size),
        }

    @classmethod
    def from_dict(cls, data):
        state = dict(data)

        genes = np.array(state['genes'], dtype=int)
        n_genes = int(state['n_genes'])
        initial_selection_size = int(state['initial_selection_size'])

        chromosome = cls(
            n_genes=n_genes,
            initial_selection_size=initial_selection_size,
            genes=genes,
            y_train=None,
        )

        int_fields = {'n_genes', 'initial_selection_size', 'n_classes', 'max_selection_size'}
        float_fields = {'fitness', 'val_auc', 'val_acc', 'test_auc', 'test_acc', 'reduction', 'log_reduction', 'linear_reduction', 'alpha', 'ir'}

        for key, value in state.items():
            if key == 'genes':
                value = np.array(value, dtype=int)
            elif key in int_fields:
                value = int(value)
            elif key in float_fields:
                value = float(value)
            setattr(chromosome, key, value)

        return chromosome



    def count_active(self):
        return np.sum(self.genes).item()

    def evaluate(self, classifier, X_train, y_train, X_val=None, y_val=None, X_test=None, y_test=None, alpha=0.5, trained=False):
        """
        Evaluates the chromosome using TabPFN.
        Equivalent to evaluacionCompleta() in Java.
        alpha: weighting of AUC vs reduction
        """
        selected_indices = np.where(self.genes == 1)[0]
        n_selected = len(selected_indices)

        if n_selected > TABPFN_MAX_SAMPLES or n_selected < 2:
            logger.debug(f"Chromosome with {n_selected} selected samples cannot be evaluated (outside TabPFN limits).")
            self.reset_clf_metrics()
            return 0

        X_sub = X_train[selected_indices]
        y_sub = y_train[selected_indices]

        if len(np.unique(y_sub)) < len(np.unique(y_train)):
            logger.warning(f"Chromosome with {n_selected} selected samples cannot be evaluated (not all classes present).")
            self.reset_clf_metrics()
            return 0
        
        try:
            if not trained:
                self.reset_clf_metrics()
                classifier.fit(X_sub, y_sub) 

            if X_val is not None:
                val_logits = classifier.predict_proba(X_val)
                val_predictions = np.argmax(val_logits, axis=1)
                self.val_acc = accuracy_score(y_val, val_predictions)
                self.val_auc = auc_score(y_val, val_logits)                
            
            if X_test is not None:
                test_logits = classifier.predict_proba(X_test)
                test_predictions = np.argmax(test_logits, axis=1)
                self.test_acc = accuracy_score(y_test, test_predictions)
                self.test_auc = auc_score(y_test, test_logits)

        except Exception as e:
            # Catch errors if TabPFN fails
            # If y_test has classes not seen in y_sub, this fails and we return fitness=0
            logger.exception(f"Error en TabPFN prediction")
            self.reset_clf_metrics()

        self.ir = imbalance_ratio(y_sub)
        self.linear_reduction = 1.0 - (n_selected / self.max_selection_size)
        self.log_reduction = self.log_reduction_metric() # LET'S SEE HOW THIS GOES
        self.reduction = 0.5 * self.linear_reduction + 0.5 * self.log_reduction # 50% linear and 50% logarithmic combination
        self.alpha = alpha # store which alpha was used to evaluate this chromosome
        self.fitness = (alpha * self.val_auc) + ((1.0 - alpha) * self.reduction)
        return self.fitness

    def log_reduction_metric(self):
        
        size_log = np.log2(self.count_active())

        size_min = np.log2(self.n_classes) # the minimum size is the number of classes
        size_max = np.log2(self.max_selection_size)  # the maximum size is the total training set size
        new_reduction_metric = 1 - (size_log - size_min) / (size_max - size_min) # normalize it to [0, 1]

        # move it to [0.5, 1].
        # moving the scale does not matter, but multiplying by 0.5 reduces the importance of the reduction, because otherwise it is too dominant
        new_reduction_metric = 0.5 + 0.5 * new_reduction_metric
        return new_reduction_metric

    def reset_clf_metrics(self):
        self.fitness = 0.0
        self.val_acc = 0.0
        self.val_auc = 0.0
        self.test_acc = 0.0
        self.test_auc = 0.0

    def compute_ir(self, y_train):
        selected_indices = np.where(self.genes == 1)[0]
        y_sub = y_train[selected_indices]
        self.ir = imbalance_ratio(y_sub)
        return self.ir

    def get_protected_instances(self, y_train):
        """
        Returns the indices of instances that must be protected to ensure
        at least one instance per class is selected.
        Only protects instances that are the ONLY instance of their class.
        This allows free removal of instances as long as the class still has others.
        """
        protected_indices = set()
        selected_indices = np.where(self.genes == 1)[0]
        
        for class_label in np.unique(y_train):
            # Find selected instances of this class
            class_selected = selected_indices[y_train[selected_indices] == class_label]
            
            # Only protect if this is the only instance of the class
            if len(class_selected) == 1:
                protected_indices.add(class_selected[0])
            elif len(class_selected) == 0:
                # If no instance of this class is selected, log a warning
                logger.debug(f"Class {class_label} has no selected instances!")
        
        return protected_indices

    def mutate(self, p_mut, y_train):
        """Mutation operator: Bit Flip
        
        If y_train is provided, ensures at least one instance per class is selected.
        This prevents mutations from removing the only instance of a class.
        Protected instances are recalculated after each removal to reflect changes.
        """

        # BALANCE THE NUMBER OF ADDS AND REMOVES: 60% REMOVES, 40% ADDS

        n_mutations = np.sum(np.random.rand(self.n_genes) < p_mut)
        # Limit the maximum number of mutations to: 10 or half the current size
        n_mutations = min(n_mutations, 10, int(self.count_active()*0.5))
        # Modify the proportion of adds and removes: 60% - 40%
        p_mut_remove = 0.6 
        n_adds = np.sum(np.random.rand(n_mutations) > p_mut_remove)
        n_removes = np.sum(np.random.rand(n_mutations) <= p_mut_remove)
        
        if n_adds > 0:
            inactive_genes = np.where(self.genes == 0)[0]
            add_indices = np.random.choice(inactive_genes, size=n_adds, replace=False)
            self.genes[add_indices] = 1

        # If it does not have at least one instance per class, add it randomly
        classes_in_chromosome = np.unique(y_train[np.where(self.genes == 1)[0]])
        if len(classes_in_chromosome) < len(np.unique(y_train)):
            logger.debug(f"Chromosome missing classes after Crossover. Adding instances in mutation to ensure all classes are represented.")
            # select a random instance from each class that is not selected
            inactive_genes = np.where(self.genes == 0)[0]
            for class_label in np.unique(y_train):
                if class_label not in classes_in_chromosome:
                    class_indices = np.where(y_train == class_label)[0]
                    inactive_class_indices = np.intersect1d(class_indices, inactive_genes)
                    if len(inactive_class_indices) > 0:
                        idx_to_add = np.random.choice(inactive_class_indices, size=1)[0]
                        self.genes[idx_to_add] = 1
                        n_adds += 1

        n_real_removes = 0
        for _ in range(n_removes):
            protected_indices = self.get_protected_instances(y_train)
            active_genes = np.where(self.genes == 1)[0]
            # Filter out protected instances
            removable_genes = np.array([g for g in active_genes if g not in protected_indices])
            # Only remove if we have removable genes
            if len(removable_genes) > 0:
                idx_to_remove = np.random.choice(removable_genes, size=1)[0]
                self.genes[idx_to_remove] = 0
                n_real_removes += 1
            else:
                # No more removable genes, stop attempting removals
                logger.debug(f"No more removable genes available during mutation. active_genes={self.count_active()} | num_classes={len(np.unique(y_train))}")
                break
        
        return n_adds, n_real_removes

    def local_search(self, X_train, y_train, X_val, y_val, X_test, y_test, classifier, alpha, attempts=10):
        """
        Adaptation of optimizacionLocal().
        Instead of trying ALL genes (too slow for TabPFN),
        try 'attempts' random changes to improve.
        Ensures at least one instance per class is selected at all times.
        NOTE: this could be more powerful if we tried many random modifications in the same batch and kept the best one
        """
        initial_fitness = self.fitness
        # Save current metrics as a dict for easy restoration
        current_metrics = {
            'fitness': self.fitness,
            'val_acc': self.val_acc,
            'val_auc': self.val_auc,
            'test_acc': self.test_acc,
            'test_auc': self.test_auc,
            'reduction': self.reduction,
            'ir': self.ir
        } 
        
        actions_with_improvement = []
        for attempt in range(attempts):
            # Recalculate protected instances before attempting removal to reflect current state
            protected_indices = self.get_protected_instances(y_train)
            active_indices = np.where(self.genes == 1)[0]
            removable_active_indices = np.array([idx for idx in active_indices if idx not in protected_indices])
            inactive_indices = np.where(self.genes == 0)[0]
            
            # Randomly decide whether to remove or add
            # SSMA prioritizes removal (reduction), so give a 70% chance to remove
            # But only attempt removals if we have removable instances
            if len(removable_active_indices) > 0 and np.random.rand() < 0.7:
                idx_to_flip = np.random.choice(removable_active_indices)
                action = "remove"
            elif len(inactive_indices) > 0:
                idx_to_flip = np.random.choice(inactive_indices)
                action = "add"
            else:
                continue
            
            # Temporary flip
            self.genes[idx_to_flip] = 1 - self.genes[idx_to_flip]
            
            # Evaluate
            self.evaluate(classifier, X_train, y_train, X_val, y_val, None, None, alpha)
            
            if self.fitness > current_metrics['fitness']:
                # Keep the change (Memetic improvement)
                # if we improved, evaluate test without calling cls.fit()
                self.evaluate(classifier, X_train, y_train, None, None, X_test, y_test, alpha, trained=True)

                # Update saved metrics
                for key in current_metrics:
                    current_metrics[key] = getattr(self, key)
                self.compute_ir(y_train)

                actions_with_improvement.append(action)
                # No need to manually update index lists - they will be recalculated in next iteration
                
            else:
                # Revert change
                self.genes[idx_to_flip] = 1 - self.genes[idx_to_flip]

                # Restore metrics from dict
                for key, value in current_metrics.items():
                    setattr(self, key, value)
                self.compute_ir(y_train)

        adds = actions_with_improvement.count("add")
        removes = actions_with_improvement.count("remove")
        delta = current_metrics['fitness'] - initial_fitness
        return [adds, removes], delta
    
    def random_reduce_to_size(self, target_size, y_train):
        """
        Reduce the chromosome to the target size by randomly removing genes.
        Ensures at least one instance per class is selected.
        """
        current_size = self.count_active()
        if target_size >= current_size:
            return  # No reduction needed

        protected_indices = self.get_protected_instances(y_train)
        active_indices = np.where(self.genes == 1)[0]
        removable_active_indices = np.array([idx for idx in active_indices if idx not in protected_indices])

        n_to_remove = current_size - target_size
        if n_to_remove > len(removable_active_indices):
            n_to_remove = len(removable_active_indices)
            logger.debug(f"Cannot reduce to target size {target_size} without removing protected instances. Reducing as much as possible.")

        indices_to_remove = np.random.choice(removable_active_indices, size=n_to_remove, replace=False)
        self.genes[indices_to_remove] = 0

    def get_genes_mask_with_validation_set(self, full_size):
        X_train_indices, X_val_indices = train_test_split(np.arange(full_size), test_size=self.val_size_split, random_state=42, shuffle=True)
        mask = np.zeros(full_size, dtype=bool)
        mask[X_train_indices] = self.genes
        return mask   
    
    def get_genes_indexes_with_validation_set(self, full_size):
        mask = self.get_genes_mask_with_validation_set(full_size)
        return np.where(mask)[0]
