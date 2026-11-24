from torch.nn import functional as F
from models.base import SWKDStudent, init_weight


class DeepLabV3Student(SWKDStudent):
    feature_channels = 1280

    def __init__(self):
        super().__init__()
        import segmentation_models_pytorch as smp

        self.Model = smp.DeepLabV3(encoder_name="mobilenet_v2", encoder_weights=None, decoder_channels=256, in_channels=3, classes=3, activation=None, upsampling=8, )
        self.apply(init_weight)

    def encoder_modules(self):
        return (self.Model.encoder,)

    def decoder_modules(self):
        return self.Model.decoder, self.Model.segmentation_head

    def encode(self, image):
        features = self.Model.encoder(image)
        return {"features": features, "final_feature": features[-1]}

    def decode(self, feature_pack, output_size):
        decoder_feature = self.Model.decoder(*feature_pack["features"])
        logits = self.Model.segmentation_head(decoder_feature)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
