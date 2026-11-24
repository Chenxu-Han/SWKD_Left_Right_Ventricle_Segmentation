import torch
from torch import nn
from torch.nn import functional as F


def init_weight(module):
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm, nn.LayerNorm)):
        if module.weight is not None:
            nn.init.ones_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


class SWKDStudent(nn.Module):
    feature_channels = 0

    def prepare_input(self, image):
        return image.repeat(1, 3, 1, 1) if image.shape[1] == 1 else image

    def set_stage_1(self, is_train):
        self.train(is_train)
        for module in self.encoder_modules():
            module.train(is_train)
            for parameter in module.parameters():
                parameter.requires_grad = is_train
        for module in self.decoder_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False

    def set_stage_2(self, is_train):
        self.train(is_train)
        for module in self.encoder_modules():
            module.eval()
            for parameter in module.parameters():
                parameter.requires_grad = False
        for module in self.decoder_modules():
            module.train(is_train)
            for parameter in module.parameters():
                parameter.requires_grad = is_train

    def encoder_parameters(self):
        parameters = []
        for module in self.encoder_modules():
            parameters.extend(module.parameters())
        return parameters

    def decoder_parameters(self):
        parameters = []
        for module in self.decoder_modules():
            parameters.extend(module.parameters())
        return parameters

    def forward_all(self, image, temperature=2.0):
        input = self.prepare_input(image)
        feature_pack = self.encode(input)
        logits = self.decode(feature_pack, image.shape[-2:])
        probability = torch.softmax(logits, dim=1)
        return {"logits": logits, "probability": probability, "log_probability": F.log_softmax(logits, dim=1), "log_soft_probability": F.log_softmax(logits / temperature, dim=1), "feature": feature_pack["final_feature"], }

    def forward(self, image):
        feature_pack = self.encode(self.prepare_input(image))
        logits = self.decode(feature_pack, image.shape[-2:])
        return logits.softmax(dim=1)
