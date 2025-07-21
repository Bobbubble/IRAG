from src.dataset.expla_graphs import ExplaGraphsDataset
from src.dataset.expla_graphs_v2 import EdgePairDataset
from src.dataset.webqsp import WebQSPDataset
# from src.dataset.webqsp_v2 import EdgePairDataset
from src.dataset.scene_graphs import SceneGraphsDataset
# from src.dataset.scene_graphs_v2 import EdgePairDataset

load_dataset = {
    # 'expla_graphs': ExplaGraphsDataset,
    'expla_graphs': EdgePairDataset,
    # 'webqsp': WebQSPDataset,
    'webqsp': EdgePairDataset,
    # 'scene_graphs': SceneGraphsDataset,
    'scene_graphs': EdgePairDataset
}
