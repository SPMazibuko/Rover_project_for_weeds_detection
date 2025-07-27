from .tooth_landmark_loss import ToothLandmarkLoss, DentalAnatomyLoss
from .adversarial_loss import AdversarialLoss
from .combined_loss import CombinedDentalLoss

__all__ = [
    'ToothLandmarkLoss', 'DentalAnatomyLoss', 
    'AdversarialLoss', 'CombinedDentalLoss'
]
