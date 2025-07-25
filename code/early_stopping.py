import torch

class EarlyStopping:
    def __init__(self,
                 patience: int = 10,
                 min_delta: float = 0.0,
                 mode: str = 'max',
                 verbose: bool = False):

        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose

        self.counter = 0
        self.early_stop = False

        if mode not in ('min', 'max'):
            raise ValueError("mode must be 'min' or 'max'")
        # init best_score
        self.best_score = float('inf') if mode == 'min' else float('-inf')

    def __call__(self, current_score: float, model: torch.nn.Module, save_path: str):
        if self.mode == 'min':
            improvement = self.best_score - current_score
        else:  # 'max'
            improvement = current_score - self.best_score

        if improvement > self.min_delta:
            self.best_score = current_score
            torch.save(model.state_dict(), save_path)
            self.counter = 0
            if self.verbose:
                print(f"[EarlyStopping] Improved by {improvement:.6f}, reset counter.")
        else:
            self.counter += 1
            if self.verbose:
                print(f"[EarlyStopping] {self.counter}/{self.patience} no real improvement "
                      f"(∆={improvement:.6f} ≤ {self.min_delta})")
            if self.counter >= self.patience:
                self.early_stop = True