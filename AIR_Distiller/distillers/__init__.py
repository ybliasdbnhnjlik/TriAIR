from ._base import Vanilla
from .KD import VanillaKD
from .FitNet import FitNet
from .CC import CC
from .RKD import RKD
from .PKT import PKT
from .CSD import CSD
from .ROP import ROP
from .RAML import RAML
from .D3 import D3
from .UGD import UGD
from .TriAIR import TriAIR
from .TriKD import TriKD

distiller_dict = {
    "NONE": Vanilla,
    "VanillaKD": VanillaKD,
    "FitNet": FitNet,
    "CC": CC,
    "RKD": RKD,
    "PKT": PKT,
    "CSD": CSD,
    "ROP": ROP,
    "RAML": RAML,
    "D3": D3,
    "UGD": UGD,
    "TriAIR": TriAIR,
    "TriKD": TriKD,  # Compatibility with saved configs from the original release.
}
