from .trainer import BaseTrainer, KDTrainer

trainer_dict = {
    "vanilla": BaseTrainer,
    "kd": KDTrainer,
}
