import torch
import torch.nn as nn

class RankCorrelationLoss(nn.Module):
    def __init__(self):
        super(RankCorrelationLoss, self).__init__()

    def forward(self, x, y):

        x_sorted, x_indices = torch.sort(x)
        y_sorted, y_indices = torch.sort(y)

        n = x.size(0)
        x_rank = torch.argsort(x_indices)
        y_rank = torch.argsort(y_indices)
        mean_rank = (n + 1) / 2.0
        
        cov_xy = torch.mean((x_rank - mean_rank) * (y_rank - mean_rank))

        std_x = torch.sqrt(torch.mean((x_rank - mean_rank)**2))
        std_y = torch.sqrt(torch.mean((y_rank - mean_rank)**2))
        correlation = cov_xy / (std_x * std_y)

        return 1 - correlation
    
class CosineLoss(nn.Module):
    def __init__(self):
        super(CosineLoss, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=0)

    def forward(self, output, target):
        # Calculate cosine similarity
        cosine_sim = self.cosine_similarity(output, target)
        # Convert to a loss (1 - similarity)
        return 1 + cosine_sim.mean()  # Take mean to ensure scalar output
    
class ICBasedLoss(nn.Module):
    def __init__(self):
        super(ICBasedLoss, self).__init__()

    def forward(self, predictions, targets):
        covariance = torch.mean((predictions - torch.mean(predictions)) * (targets - torch.mean(targets)))
        std_predictions = torch.std(predictions)
        std_targets = torch.std(targets)
        ic = covariance / (std_predictions * std_targets + 1e-6)
        loss = 1/(ic + 1 + 1e-6) - 0.5
        return loss
    
class multiIC(nn.Module):
    def __init__(self, lamb):
        super(multiIC, self).__init__()
        self.IC = ICBasedLoss()
        self.lamb = lamb
    def forward(self , x , y , target):
        loss = -self.IC(x , target) - self.IC(y , target) + self.lamb * self.IC(x , y)
        return loss
