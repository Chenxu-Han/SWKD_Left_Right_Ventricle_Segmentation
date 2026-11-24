from torch import nn
from torch.nn import functional as F
from models.base import SWKDStudent, init_weight


class MobileUNetStudent(SWKDStudent):
    feature_channels = 1280

    def __init__(self):
        super().__init__()
        from models.base_models.Mobile_UNet.model import MobileUNet

        self.Model = MobileUNet()
        self.Model.conv1x1_decode = nn.Conv2d(16, 3, 1)
        self.apply(init_weight)

    def encoder_modules(self):
        return (self.Model.conv3x3, self.Model.irb_bottleneck1, self.Model.irb_bottleneck2, self.Model.irb_bottleneck3, self.Model.irb_bottleneck4, self.Model.irb_bottleneck5, self.Model.irb_bottleneck6, self.Model.irb_bottleneck7,
                self.Model.conv1x1_encode,)

    def decoder_modules(self):
        return (self.Model.D_irb1, self.Model.D_irb2, self.Model.D_irb3, self.Model.D_irb4, self.Model.DConv4x4, self.Model.conv1x1_decode,)

    def encode(self, image):
        feature_1 = self.Model.conv3x3(image)
        feature_2 = self.Model.irb_bottleneck1(feature_1)
        feature_3 = self.Model.irb_bottleneck2(feature_2)
        feature_4 = self.Model.irb_bottleneck3(feature_3)
        feature_5 = self.Model.irb_bottleneck4(feature_4)
        feature_6 = self.Model.irb_bottleneck5(feature_5)
        feature_7 = self.Model.irb_bottleneck6(feature_6)
        feature_8 = self.Model.irb_bottleneck7(feature_7)
        final_feature = self.Model.conv1x1_encode(feature_8)
        return {"feature_2": feature_2, "feature_3": feature_3, "feature_4": feature_4, "feature_6": feature_6, "final_feature": final_feature, }

    def decode(self, feature_pack, output_size):
        feature = self.Model.D_irb1(feature_pack["final_feature"]) + feature_pack["feature_6"]
        feature = self.Model.D_irb2(feature) + feature_pack["feature_4"]
        feature = self.Model.D_irb3(feature) + feature_pack["feature_3"]
        feature = self.Model.D_irb4(feature) + feature_pack["feature_2"]
        feature = self.Model.DConv4x4(feature)
        logits = self.Model.conv1x1_decode(feature)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
