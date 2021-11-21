import yaml
from detectron2.config import get_cfg
from detectron2 import model_zoo


if __name__ == '__main__':
    cfg_source = 'COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml'
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(cfg_source))
    cfg.MODEL.ROI_HEADS.SCORE_THRES_TEST = 0.5
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(cfg_source)

    # save cfg object as yaml
    with open('config.yaml', 'w') as f:
        f.write(cfg.dump())
    
    # check cfg object
    with open('config.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    print(yaml.dump(cfg))
