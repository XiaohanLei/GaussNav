import torch
import torch.nn as nn
from segmentation_models_pytorch.encoders import get_encoder
from habitat_baselines.rl.ddppo.policy import resnet


def init_distributed(model, device, find_unused_params: bool = True):
    if torch.cuda.is_available():
        ddp = nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device],
            output_device=device,
            find_unused_parameters=find_unused_params,
        )
    else:
        ddp = nn.parallel.DistributedDataParallel(
            model,
            find_unused_parameters=find_unused_params,
        )

    return ddp


class DropoutPart(nn.Module):
    def __init__(self, p, embedding_size):
        super().__init__()
        self.dropout = nn.Dropout(p, inplace=True)
        self.embedding_size = embedding_size

    def forward(self, x):
        self.dropout(x[:, self.embedding_size:])
        return x


class VONetBase(nn.Module):
    def __init__(self, encoder, fc, compression=None, action_embedding=None, collision_embedding=None):
        super().__init__()
        self.action_embedding = action_embedding
        self.collision_embedding = collision_embedding
        self.encoder = encoder
        self.compression = compression
        self.flatten = nn.Flatten()
        self.fc = fc

    def forward(self, x, action=None, collision=None):
        x = self.encoder_forward(x)
        x = self.compression_forward(x)
        x = self.fc_forward(self.flatten(x), action, collision)

        return x

    def encoder_forward(self, x):
        return self.encoder(x)

    def compression_forward(self, x):
        return self.compression(x)

    def fc_forward(self, x, action, collision):
        if self.action_embedding is not None:
            x = torch.cat([self.action_embedding(action).detach(), x], 1)
        if self.collision_embedding is not None:
            x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
        return self.fc(x)

    @staticmethod
    def get_encoder(name, encoder_params):
        raise NotImplemented()

    @staticmethod
    def get_compression(encoder, encoder_params):
        # no compression
        return nn.Identity()

    @staticmethod
    def get_fc(
            encoder_output_size: int,
            embedding_size: int,
            hidden_size: list,
            output_size: int,
            p_dropout: float = 0.0
    ):
        hidden_size.insert(0, encoder_output_size + embedding_size)

        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Sequential(
                DropoutPart(p_dropout, embedding_size) if i == 0 and embedding_size > 0 else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i], hidden_size[i + 1]),
                nn.ReLU(True),
            ))
        layers.append(nn.Sequential(
            nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1], output_size)
        ))

        return nn.Sequential(*layers)

    @staticmethod
    def compute_encoder_output_size(encoder, compression, encoder_params):
        input_size = (1, encoder_params.in_channels, encoder_params.in_height, encoder_params.in_width)

        encoder_input = torch.randn(*input_size)
        with torch.no_grad():
            output = encoder(encoder_input)
            output = compression(output)

        return output[-1].view(-1).size(0)

    @classmethod
    def from_config(cls, model_config):
        model_params = model_config.params
        encoder_params = model_params.encoder.params
        fc_params = model_params.fc.params

        action_embedding = nn.Embedding(
            num_embeddings=model_params.n_action_values,
            embedding_dim=model_params.action_embedding_size
        ) if model_params.action_embedding_size > 0 else None

        collision_embedding = nn.Embedding(
            num_embeddings=model_params.n_collision_values,
            embedding_dim=model_params.collision_embedding_size
        ) if model_params.collision_embedding_size > 0 else None

        encoder = cls.get_encoder(
            name=model_params.encoder.type,
            encoder_params=encoder_params
        )

        compression = cls.get_compression(encoder, encoder_params)

        fc = cls.get_fc(
            encoder_output_size=cls.compute_encoder_output_size(encoder, compression, encoder_params),
            embedding_size=model_params.collision_embedding_size + model_params.action_embedding_size,
            hidden_size=fc_params.hidden_size,
            output_size=fc_params.output_size,
            p_dropout=fc_params.p_dropout
        )

        return cls(encoder, fc, compression, action_embedding, collision_embedding)


# VONet
class VONetSMP(VONetBase):
    """
    VO model with segmentation_models.pytorch-based (ResNet) encoder and no compression,
    embedding vector is concatenated to only THE FIRST fc layer. DropOut is not applied
    to neurons corresponding to embedding vector (see DropoutPart).
    """
    def encoder_forward(self, x):
        return self.encoder(x)[-1]  # get last stage output

    @staticmethod
    def get_encoder(name, encoder_params):
        return get_encoder(
            name=name,
            in_channels=encoder_params.in_channels,
            depth=encoder_params.depth,
            weights=encoder_params.weights
        )

    @staticmethod
    def compute_encoder_output_size(encoder, compression, encoder_params):
        input_size = (1, encoder_params.in_channels, encoder_params.in_height, encoder_params.in_width)

        encoder_input = torch.randn(*input_size)
        with torch.no_grad():
            output = encoder(encoder_input)[-1]  # get last stage output
            output = compression(output)

        return output[-1].view(-1).size(0)


# VONetV2
class VONetHabitat(VONetBase):
    """
    VO model with habitat-lab-based half-width ResNet encoder and conv compression,
    embedding vector is concatenated to only THE FIRST fc layer. DropOut is not applied
    to neurons corresponding to embedding vector (see DropoutPart).
    """
    @staticmethod
    def get_encoder(name, encoder_params):
        return getattr(resnet, name)(
            encoder_params.in_channels,
            encoder_params.base_planes,
            encoder_params.ngroups
        )

    @staticmethod
    def get_compression(encoder, encoder_params):
        return nn.Sequential(
            nn.Conv2d(
                encoder.final_channels,
                encoder_params.num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, encoder_params.num_compression_channels),
            nn.ReLU(True),
        )


# VONetV3
class VONetHabitatV2(VONetHabitat):
    """
    VO model with habitat-lab-based half-width ResNet encoder and conv compression,
    embedding vector is concatenated to EVERY fc layer. DropOut is not applied
    to neurons corresponding to embedding vector (see DropoutPart).
    """
    def fc_forward(self, x, action, collision):
        for fc_layer in self.fc:
            if self.action_embedding is not None:
                x = torch.cat([self.action_embedding(action).detach(), x], 1)
            if self.collision_embedding is not None:
                x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
            x = fc_layer(x)
        return x

    @staticmethod
    def get_fc(
            encoder_output_size: int,
            embedding_size: int,
            hidden_size: list,
            output_size: int,
            p_dropout: float = 0.0
    ):
        hidden_size.insert(0, encoder_output_size)

        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Sequential(
                DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i] + embedding_size, hidden_size[i + 1]),
                nn.ReLU(True),
            ))
        layers.append(nn.Sequential(
            DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1] + embedding_size, output_size)
        ))

        return nn.Sequential(*layers)


# VONetV4
class VONetSMPV2(VONetSMP):
    """
    VO model with segmentation_models.pytorch-based (RegNetY) encoder and conv compression,
    embedding vector is concatenated to EVERY fc layer. DropOut is not applied
    to neurons corresponding to embedding vector (see DropoutPart).
    """
    def fc_forward(self, x, action, collision):
        for fc_layer in self.fc:
            if self.action_embedding is not None:
                x = torch.cat([self.action_embedding(action).detach(), x], 1)
            if self.collision_embedding is not None:
                x = torch.cat([self.collision_embedding(collision).detach(), x], 1)
            x = fc_layer(x)
        return x

    @staticmethod
    def get_compression(encoder, encoder_params):
        return nn.Sequential(
            nn.Conv2d(
                encoder.num_features,
                encoder_params.num_compression_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.GroupNorm(1, encoder_params.num_compression_channels),
            nn.ReLU(True),
        )

    @staticmethod
    def get_fc(
            encoder_output_size: int,
            embedding_size: int,
            hidden_size: list,
            output_size: int,
            p_dropout: float = 0.0
    ):
        hidden_size.insert(0, encoder_output_size)

        layers = []
        for i in range(len(hidden_size) - 1):
            layers.append(nn.Sequential(
                DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
                nn.Linear(hidden_size[i] + embedding_size, hidden_size[i + 1]),
                nn.ReLU(True),
            ))
        layers.append(nn.Sequential(
            DropoutPart(p_dropout, embedding_size) if embedding_size > 0 else nn.Dropout(p_dropout),
            nn.Linear(hidden_size[-1] + embedding_size, output_size)
        ))

        return nn.Sequential(*layers)


# For backward compatibility:
VONet = VONetSMP
VONetV2 = VONetHabitat
VONetV3 = VONetHabitatV2
VONetV4 = VONetSMPV2
