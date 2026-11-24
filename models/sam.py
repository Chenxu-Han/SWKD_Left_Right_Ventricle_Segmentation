import torch
from torch import nn
from torch.nn import functional as F
from segment_anything import sam_model_registry


class SAMTeacher(nn.Module):
    def __init__(self, checkpoint):
        super().__init__()
        self.Model = sam_model_registry["vit_b"](checkpoint=None)
        if checkpoint:
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            self.Model.load_state_dict(state.get("model_state_dict", state.get("Model", state)), strict=True)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        return super().train(False)

    def encode(self, images):
        # Official SAM preprocessing; grayscale inputs are in [0, 1].
        images = F.interpolate(images, size=(1024, 1024), mode="bilinear", align_corners=False)
        images = self.Model.preprocess(images.repeat(1, 3, 1, 1) * 255.0)
        return self.Model.image_encoder(images)

    def decode(self, embedding, masks):
        logits = embedding.new_zeros((len(embedding), 3, *masks.shape[-2:]))
        for index in range(len(embedding)):
            for organ in range(2):
                coordinates = torch.nonzero(masks[index, organ], as_tuple=False)
                if len(coordinates) == 0:
                    continue
                pick = torch.randint(len(coordinates), (1,), device=coordinates.device)
                point = coordinates[pick][:, [1, 0]].float().reshape(1, 1, 2)
                point = point * point.new_tensor([1024 / masks.shape[-1], 1024 / masks.shape[-2]])
                label = torch.ones((1, 1), device=point.device, dtype=torch.long)
                sparse, dense = self.Model.prompt_encoder(points=(point, label), boxes=None, masks=None)
                mask, _ = self.Model.mask_decoder(image_embeddings=embedding[index: index + 1], image_pe=self.Model.prompt_encoder.get_dense_pe(), sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False, )
                logits[index, organ + 1] = F.interpolate(mask, size=masks.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
        return logits

    def forward(self, images, masks):
        logits = self.decode(self.encode(images), masks)
        probability = logits.softmax(dim=1)
        return {"logits": logits, "probability": probability}
