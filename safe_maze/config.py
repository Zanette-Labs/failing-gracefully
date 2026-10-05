import yaml
from pathlib import Path

_cfg = yaml.safe_load((Path(__file__).parent / 'config.yaml').read_text())
H: int = _cfg['grid_height']
W: int = _cfg['grid_width']
n_blocks: int = _cfg.get('n_blocks', 4)
