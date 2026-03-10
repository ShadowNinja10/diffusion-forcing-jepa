from datasets.video import (
    MinecraftAdvancedVideoDataset,
    Kinetics600AdvancedVideoDataset,
    RealEstate10KAdvancedVideoDataset,
    RealEstate10KMiniAdvancedVideoDataset,
    RealEstate10KOODAdvancedVideoDataset,
)
from algorithms.dfot import (
    DFoTVideo,
    DFoTVideoPose,
    DFoTVideoJEPA,
    DFoTVideoJEPASpatial,
    DFoTVideoJEPASpatialSigREG,
    DFoTVideoIJEPASigREG,
    DFoTVideoDecoupled,
    JEPATraining,
    JEPASpatialTraining,
    JEPASpatialSmallP,
)
from .base_exp import BaseLightningExperiment
from .data_modules.utils import _data_module_cls


class VideoGenerationExperiment(BaseLightningExperiment):
    """
    A video generation experiment
    """

    compatible_algorithms = dict(
        dfot_video=DFoTVideo,
        dfot_video_pose=DFoTVideoPose,
        dfot_video_jepa=DFoTVideoJEPA,
        dfot_video_jepa_spatial=DFoTVideoJEPASpatial,
        dfot_video_jepa_spatial_sigreg=DFoTVideoJEPASpatialSigREG,
        dfot_video_ijepa_sigreg=DFoTVideoIJEPASigREG,
        dfot_video_decoupled=DFoTVideoDecoupled,
        dfot_video_ema_finetune=DFoTVideoJEPA,
        jepa_training=JEPATraining,
        jepa_spatial_training=JEPASpatialTraining,
        jepa_spatial_small_p=JEPASpatialSmallP,
        sd_video=DFoTVideo,
        sd_video_3d=DFoTVideoPose,
    )

    compatible_datasets = dict(
        # video datasets
        minecraft=MinecraftAdvancedVideoDataset,
        realestate10k=RealEstate10KAdvancedVideoDataset,
        realestate10k_ood=RealEstate10KOODAdvancedVideoDataset,
        realestate10k_mini=RealEstate10KMiniAdvancedVideoDataset,
        kinetics_600=Kinetics600AdvancedVideoDataset,
    )

    data_module_cls = _data_module_cls
