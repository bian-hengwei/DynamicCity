from omegaconf import OmegaConf

config = OmegaConf.load('conf/others/constants.yaml')
for key, value in config.items():
    globals()[key] = value
