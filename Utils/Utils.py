import torch

class Utils:

    def __init__(self):
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"
        self.is_multi_gpu = torch.cuda.is_available() and torch.cuda.device_count() > 1
        
    def wrap_model_for_multi_gpu(self, model):
        """Wrap model with DataParallel if multiple GPUs are available"""
        if self.is_multi_gpu:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel")
            return torch.nn.DataParallel(model)
        return model

