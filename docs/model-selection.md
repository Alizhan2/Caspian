# Model selection record

## Runtime target

The live pipeline expects a two-channel Sentinel-1 VV/VH semantic-segmentation checkpoint. Every result records the checkpoint filename and SHA-256 digest.

## Sources reviewed

- [SkyTruth Cerulean Cloud](https://github.com/SkyTruth/cerulean-cloud): strong open-source architecture and human-review reference, but no verified drop-in Caspian U-Net/ResNet34 checkpoint was identified.
- [SatlasPretrain models](https://github.com/allenai/satlaspretrain_models): useful Sentinel-1 pretrained backbone; its segmentation head is randomly initialized and must be fine-tuned on labelled oil/look-alike data.
- [TheArchitect416 oil-spill model](https://huggingface.co/TheArchitect416/oil-spill-segmentation-model): advertises U-Net/ResNet34, but the model card is empty and does not document SAR bands, dataset split, licence, metrics, threshold or regional generalization.

## Decision

No public checkpoint is installed automatically. A model may be placed at `models/oil_unet.pt` only after verifying:

1. Sentinel-1 VV/VH input compatibility;
2. checkpoint licence and provenance;
3. date-separated and region-separated evaluation;
4. precision, recall, IoU and false alarms per scene;
5. threshold calibration on Caspian data;
6. explicit treatment of look-alikes.

Until this gate is passed, the API remains fail-closed and cannot issue an AI screening result.
