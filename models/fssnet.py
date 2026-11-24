from torch.nn import functional as F
from models.base import SWKDStudent, init_weight


class FSSNetStudent(SWKDStudent):
    feature_channels = 128

    def __init__(self):
        super().__init__()
        from models.base_models.FSSNet.model.FSSNet import FSSNet

        self.Model = FSSNet(classes=3)
        self.apply(init_weight)

    def encoder_modules(self):
        return (
            self.Model.initial_block, self.Model.downsample1_0, self.Model.factorized1_1, self.Model.factorized1_2, self.Model.factorized1_3, self.Model.factorized1_4, self.Model.downsample2_0, self.Model.dilated2_1, self.Model.dilated2_2,
            self.Model.dilated2_3, self.Model.dilated2_4, self.Model.dilated2_5, self.Model.dilated2_6,)

    def decoder_modules(self):
        return (self.Model.upsample4_0, self.Model.bottleneck4_1, self.Model.bottleneck4_2, self.Model.upsample5_0, self.Model.bottleneck5_1, self.Model.bottleneck5_2, self.Model.transposed_conv,)

    def encode(self, image):
        initial = self.Model.initial_block(image)
        skip_1 = self.Model.downsample1_0(initial)
        feature = self.Model.factorized1_1(skip_1)
        feature = self.Model.factorized1_2(feature)
        feature = self.Model.factorized1_3(feature)
        feature = self.Model.factorized1_4(feature)
        skip_2 = self.Model.downsample2_0(feature)
        feature = self.Model.dilated2_1(skip_2)
        feature = self.Model.dilated2_2(feature)
        feature = self.Model.dilated2_3(feature)
        feature = self.Model.dilated2_4(feature)
        feature = self.Model.dilated2_5(feature)
        feature = self.Model.dilated2_6(feature)
        return {"skip_1": skip_1, "skip_2": skip_2, "final_feature": feature}

    def decode(self, feature_pack, output_size):
        feature = self.Model.upsample4_0(feature_pack["final_feature"], feature_pack["skip_2"])
        feature = self.Model.bottleneck4_1(feature)
        feature = self.Model.bottleneck4_2(feature)
        feature = self.Model.upsample5_0(feature, feature_pack["skip_1"])
        feature = self.Model.bottleneck5_1(feature)
        feature = self.Model.bottleneck5_2(feature)
        logits = self.Model.transposed_conv(feature)
        if logits.shape[-2:] != output_size:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return logits
