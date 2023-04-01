from realtime_panoptic.config import cfg
from realtime_panoptic.models.rt_pano_net import RTPanoNet

from datasets.cityscapes import cityscapes_colormap, cityscapes_instance_label_name, cityscapes_base_instance_threshold

class DatasetSelection():
    def __init__(self, config):
        if(config.model.name == 'Cityscape_realtime_panoptic'):
            self.__model__ = RTPanoNet(
                backbone=config.model.backbone, 
                num_classes=config.model.panoptic.num_classes,
                things_num_classes=config.model.panoptic.num_thing_classes,
                pre_nms_thresh=config.model.panoptic.pre_nms_thresh,
                pre_nms_top_n=config.model.panoptic.pre_nms_top_n,
                nms_thresh=config.model.panoptic.nms_thresh,
                fpn_post_nms_top_n=config.model.panoptic.fpn_post_nms_top_n,
                instance_id_range=config.model.panoptic.instance_id_range)
            self.label_map = cityscapes_instance_label_name
            self.score_thr = cityscapes_base_instance_threshold
        else:
            raise RuntimeError("Model Invalid")
        self.__config_name__ = config.model.name

    def getNetwork(self):
        return self.__model__
    



