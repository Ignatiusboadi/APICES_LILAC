import torch.nn as nn
import torch
from torchvision.models import resnet18, resnet50
import copy
import torch._dynamo as dynamo

# a set of scaling variations estimated using the training set
# needs to be updated accordingly
scaling_dict = {"stdev": {"age_decimal": 2.88, "CDRSB": 2.03},
                "non_zero_stdev": {"age_decimal": 2.88, "CDRSB": 2.53},
                "max_interval": {"age_decimal": 19.33, "CDRSB": 16}}


@dynamo.disable
def contrastive_index(features, index_pairs, meta=None):
    try:
        f_i = features[index_pairs[0]]
        f_j = features[index_pairs[1]]
        f = f_i - f_j
    except IndexError:
        print('index pairs after index error', index_pairs)
        print('features after index error', features)
        raise

    if meta is not None:
        meta_i = meta[index_pairs[0]]
        meta_i = torch.as_tensor(meta_i, dtype=f.dtype, device=f.device)
        f = torch.cat((f, meta_i), dim=1)

    return f


class MaxAvgPool3D(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0):
        super(MaxAvgPool3D, self).__init__()
        self.max_pool = nn.MaxPool3d(kernel_size, stride, padding)
        self.avg_pool = nn.AvgPool3d(kernel_size, stride, padding)

    def forward(self, x):
        x_max = self.max_pool(x)
        x_avg = self.avg_pool(x)
        return torch.cat((x_max, x_avg), dim=1)


class EncoderBlock3D(nn.Module):
    def __init__(self, in_num_ch, out_num_ch, kernel_size=3, conv_act='leaky_relu', dropout=0, pooling=nn.AvgPool3d,
                 layer_norm="batchnorm"):
        super(EncoderBlock3D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)
        if layer_norm == "instance":
            norm = nn.GroupNorm(num_groups=out_num_ch, num_channels=out_num_ch, affine=True)
        elif layer_norm == "layer":
            intermediate_size = []
            norm = nn.LayerNorm(intermediate_size, elementwise_affine=False)
        else:
            norm = nn.BatchNorm3d(out_num_ch)
        self.conv = nn.Sequential(
            nn.Conv3d(in_num_ch, out_num_ch, kernel_size=kernel_size, padding=1),
            norm,
            conv_act_layer,
            nn.Dropout3d(dropout),
            pooling(2))
        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        return self.conv(x)


class Encoder3D(nn.Module):
    def __init__(self, in_num_ch=1, num_block=4, inter_num_ch=16, kernel_size=3, conv_act='leaky_relu',
                 pooling=nn.AvgPool3d, layer_norm="batchnorm"):
        super(Encoder3D, self).__init__()
        if pooling is MaxAvgPool3D:
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        conv_blocks = []
        for i in range(num_block):
            if i == 0:
                conv_blocks.append(EncoderBlock3D(in_num_ch, inter_num_ch, kernel_size=kernel_size, conv_act=conv_act,
                                                  dropout=0, pooling=pooling, layer_norm=layer_norm))
            elif i == (num_block - 1):  # last block
                conv_blocks.append(
                    EncoderBlock3D(num_channel_modifier * inter_num_ch * (2 ** (i - 1)), inter_num_ch,
                                   kernel_size=kernel_size, conv_act=conv_act, dropout=0, pooling=pooling,
                                   layer_norm=layer_norm))
            else:
                conv_blocks.append(
                    EncoderBlock3D(num_channel_modifier * inter_num_ch * (2 ** (i - 1)), inter_num_ch * (2 ** i),
                                   kernel_size=kernel_size, conv_act=conv_act, dropout=0, pooling=pooling,
                                   layer_norm=layer_norm))

        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x


class ResEncoderBlock3D(nn.Module):
    def __init__(self, in_num_ch, intermediate_size, kernel_size=3, conv_act='leaky_relu', dropout=0,
                 pooling=nn.AvgPool3d, layer_norm="batchnorm"):
        super(ResEncoderBlock3D, self).__init__()
        if conv_act == 'relu':
            conv_act_layer = nn.ReLU(inplace=True)
        elif conv_act == 'leaky_relu':
            conv_act_layer = nn.LeakyReLU(0.2, inplace=True)
        else:
            raise ValueError('No implementation of ', conv_act)

        if layer_norm == "instance":
            norm = nn.InstanceNorm3d(intermediate_size[0], affine=True, track_running_stats=False)
        elif layer_norm == "layer":
            norm = nn.LayerNorm(intermediate_size, elementwise_affine=False)
        else:
            norm = nn.BatchNorm3d(intermediate_size[0])

        self.conv = nn.Sequential(
            nn.Conv3d(in_num_ch, intermediate_size[0], kernel_size=kernel_size, padding=1),
            norm,
            conv_act_layer,
            nn.Dropout3d(dropout, inplace=True),
            nn.Conv3d(intermediate_size[0], intermediate_size[0], kernel_size=1),
            copy.deepcopy(norm))

        self.identity_path = nn.Sequential(
            nn.Conv3d(in_num_ch, intermediate_size[0], kernel_size=1, stride=1, bias=False), copy.deepcopy(norm))
        self.output_layer = nn.Sequential(conv_act_layer,
                                          nn.Dropout3d(dropout, inplace=True),
                                          pooling(2))

        self.init_model()

    def init_model(self):
        for layer in self.conv.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)
        for layer in self.identity_path.children():
            if isinstance(layer, nn.Conv3d):
                for name, weight in layer.named_parameters():
                    if 'weight' in name:
                        nn.init.kaiming_normal_(weight)
                    if 'bias' in name:
                        nn.init.constant_(weight, 0.0)

    def forward(self, x):
        out = self.conv(x) + self.identity_path(x)
        return self.output_layer(out)


class ResEncoder3D(nn.Module):
    def __init__(self, input_size=(1, 128, 128, 128), num_block=4, inter_num_ch=16, kernel_size=3,
                 conv_act='leaky_relu', pooling=nn.AvgPool3d, layer_norm="batchnorm"):
        super(ResEncoder3D, self).__init__()
        intermediate_size = input_size
        if pooling is MaxAvgPool3D:
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        conv_blocks = []
        for i in range(num_block):
            if i == 0:
                C, H, W, L = intermediate_size
                C = inter_num_ch
                intermediate_size = [C, H, W, L]

                conv_blocks.append(
                    ResEncoderBlock3D(input_size[0], intermediate_size, kernel_size=kernel_size, conv_act=conv_act,
                                      dropout=0, pooling=pooling, layer_norm=layer_norm))
            elif i == (num_block - 1):  # last block

                C, H, W, L = intermediate_size
                C = inter_num_ch
                intermediate_size = [C, H // 2, W // 2, L // 2]
                conv_blocks.append(
                    ResEncoderBlock3D(num_channel_modifier * inter_num_ch * (2 ** (i - 1)), intermediate_size,
                                      kernel_size=kernel_size, conv_act=conv_act, dropout=0, pooling=pooling,
                                      layer_norm=layer_norm))
            else:
                C, H, W, L = intermediate_size
                C = inter_num_ch * (2 ** (i))
                intermediate_size = [C, H // 2, W // 2, L // 2]
                conv_blocks.append(
                    ResEncoderBlock3D(num_channel_modifier * inter_num_ch * (2 ** (i - 1)), intermediate_size,
                                      kernel_size=kernel_size, conv_act=conv_act, dropout=0, pooling=pooling,
                                      layer_norm=layer_norm))

        self.conv_blocks = nn.Sequential(*conv_blocks)

    def forward(self, x):

        for cb in self.conv_blocks:
            x = cb(x)

        return x


class CNNbasic3D(nn.Module):
    def __init__(self, inputsize=(128, 128, 128), channels=1, n_of_blocks=4, initial_channel=16, pooling=nn.AvgPool3d,
                 additional_feature=0):
        super(CNNbasic3D, self).__init__()

        self.feature_image = (torch.tensor(inputsize) / (2 ** n_of_blocks))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,
                                 pooling=pooling)
        self.linear = nn.Linear(
            (self.feature_channel * (self.feature_image.prod()).type(torch.int).item()) + additional_feature, 1,
            bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


class CNN_3D_Compressed(nn.Module):
    def __init__(self,
                 inputsize=(128, 128, 128),
                 channels=1,
                 n_of_blocks=5,
                 initial_channel=16,
                 pooling=nn.AvgPool3d,
                 additional_feature=0,
                 compression_factor=4,
                 layer_norm="batchnorm"):
        super(CNN_3D_Compressed, self).__init__()

        num_channel_modifier = 2 if pooling is MaxAvgPool3D else 1

        self.feature_image = (torch.tensor(inputsize) // (2 ** n_of_blocks))
        self.feature_channel = initial_channel

        self.encoder = Encoder3D(
            in_num_ch=channels,
            num_block=n_of_blocks,
            inter_num_ch=initial_channel,
            pooling=pooling,
            layer_norm=layer_norm
        )

        compressed_channels = int((num_channel_modifier * self.feature_channel) / compression_factor)

        self.compression = nn.Conv3d(
            num_channel_modifier * self.feature_channel,
            compressed_channels,
            kernel_size=3,
            padding=1
        )

        flattened_size = compressed_channels * int(self.feature_image.prod().item())
        self.linear = nn.Linear(flattened_size + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.compression(x)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        x = self.compression(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


class CNN_medium_3D(nn.Module):
    def __init__(self, inputsize=(128, 128, 128), channels=1, n_of_blocks=5, initial_channel=16, pooling=nn.AvgPool3d,
                 additional_feature=0, layer_norm="batchnorm", output_shape=None):
        super(CNN_medium_3D, self).__init__()
        if pooling is MaxAvgPool3D:
            print("MaxAvgPooling used")
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        self.output_shape = output_shape
        self.feature_image = (torch.tensor(inputsize) / (2 ** n_of_blocks))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,
                                 pooling=pooling, layer_norm=layer_norm)
        self.linear = nn.Linear(
            (num_channel_modifier * self.feature_channel * (self.feature_image.prod()).type(torch.int).item()
             ) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        if not self.output_shape:
            self.output_shape = x.shape
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


class CNN_medium_3D_residual(nn.Module):
    def __init__(self, inputsize=(128, 128, 128), channels=1, n_of_blocks=5, initial_channel=16, pooling=nn.AvgPool3d,
                 additional_feature=0, layer_norm="batchnorm"):
        super(CNN_medium_3D_residual, self).__init__()
        if pooling is MaxAvgPool3D:
            print("MaxAvgPooling used")
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        self.feature_image = (torch.tensor(inputsize) / (2 ** n_of_blocks))
        self.feature_channel = initial_channel
        input_size = [channels] + inputsize
        self.encoder = ResEncoder3D(input_size, num_block=n_of_blocks, inter_num_ch=initial_channel, pooling=pooling,
                                    layer_norm=layer_norm)
        self.linear = nn.Linear((num_channel_modifier * self.feature_channel * (self.feature_image.prod()).type(
            torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


class CNN_medium_3D_compressed(nn.Module):
    def __init__(self, inputsize=(128, 128, 128), channels=1, n_of_blocks=5, initial_channel=16, pooling=nn.AvgPool3d,
                 additional_feature=0, compression_factor=4, layer_norm="batchnorm", output_shape=None):
        super(CNN_medium_3D_compressed, self).__init__()
        if pooling is MaxAvgPool3D:
            print("MaxAvgPooling used")
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        self.output_shape = output_shape

        self.feature_image = (torch.tensor(inputsize) / (2 ** n_of_blocks))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,
                                 pooling=pooling, layer_norm=layer_norm)
        self.final_compression_layer = nn.Conv3d(num_channel_modifier * self.feature_channel,
                                                 int(self.feature_channel / compression_factor), kernel_size=3,
                                                 padding=1)
        self.linear = nn.Linear((int(self.feature_channel / compression_factor) * (self.feature_image.prod()).type(
            torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = self.final_compression_layer(x)
        if not self.output_shape:
            self.output_shape = x.shape
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        x = self.final_compression_layer(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


class CNN_large_3D(nn.Module):
    def __init__(self, inputsize=(128, 128, 128), channels=1, n_of_blocks=6, initial_channel=16, pooling=MaxAvgPool3D,
                 additional_feature=0):
        super(CNN_large_3D, self).__init__()
        if pooling is MaxAvgPool3D:
            print("MaxAvgPooling used")
            num_channel_modifier = 2
        else:
            num_channel_modifier = 1
        self.feature_image = (torch.tensor(inputsize) / (2 ** n_of_blocks))
        self.feature_channel = initial_channel
        self.encoder = Encoder3D(in_num_ch=channels, num_block=n_of_blocks, inter_num_ch=initial_channel,
                                 pooling=pooling)
        self.linear = nn.Linear((num_channel_modifier * self.feature_channel * (self.feature_image.prod()).type(
            torch.int).item()) + additional_feature, 1, bias=False)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.shape[0], -1)
        y = self.linear(x)
        return y

    def extract(self, x, reshape=True):
        x = self.encoder(x)
        if reshape:
            x = x.view(x.shape[0], -1)
        return x


def get_backbone(args=None):
    assert args is not None, 'arguments are required for network configurations'
    # TODO args.optional_meta type should be list
    # quick and dirty fix for the moment, can probably be removed permanently
    n_of_meta = len(args.optional_meta)

    backbone_name = args.backbone_name
    if backbone_name == 'cnn_3D':
        backbone = CNNbasic3D(inputsize=args.image_size, channels=args.image_channel, additional_feature=n_of_meta)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'cnn_3D_medium':
        backbone = CNN_medium_3D(inputsize=args.image_size, channels=args.image_channel, additional_feature=n_of_meta,
                                 layer_norm=args.layer_norm)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'cnn_3D_medium_compressed':
        backbone = CNN_medium_3D_compressed(inputsize=args.image_size, channels=args.image_channel,
                                            additional_feature=n_of_meta, layer_norm=args.layer_norm)
        linear = backbone.linear
        backbone.linear = nn.Identity()
    elif backbone_name == 'cnn_3D_medium_compressed_maxavg':
        backbone = CNN_medium_3D_compressed(inputsize=args.image_size, channels=args.image_channel,
                                            additional_feature=n_of_meta, layer_norm=args.layer_norm,
                                            pooling=MaxAvgPool3D)
        linear = backbone.linear
        backbone.linear = nn.Identity()

    else:
        raise NotImplementedError(f"{args.backbone_name} not implemented yet")

    return backbone, linear


# class LILAC(nn.Module):
#     def __init__(self, args):
#         super().__init__()
#         self.backbone, self.linear = get_backbone(args)
#         self.optional_meta = len(args.optional_meta) > 0
#
#     def forward(self, x1, x2, meta=None):
#         f = self.backbone(x1) - self.backbone(x2)
#         if not self.optional_meta:
#             return self.linear(f)
#         else:
#             m1, m2 = meta
#             m = m1
#             f = torch.concat((f, m), 1)
#             return self.linear(f)
#
#     def get_extractor(self):
#         return self.backbone


# class multi_LILAC(nn.Module):
#     def __init__(self, args, num_heads):
#         super().__init__()
#         self.backbone, self.linear = get_backbone(args)
#         self.optional_meta = len(args.optional_meta) > 0
#         self.network_heads = []
#         for head in range(num_heads):
#             self.network_heads.append(copy.deepcopy(self.linear))
#         self.network_heads = nn.ModuleList(self.network_heads)
#
#     def forward(self, x1, x2, meta=None):
#         f = self.backbone(x1) - self.backbone(x2)
#         result_list = []
#         if not meta:
#             for head in self.network_heads:
#                 result = head(f)
#                 result_list.append(result)
#         else:
#             f = torch.concat((f, meta), 1)
#             for head in self.network_heads:
#                 result = head(f)
#                 result_list.append(result)
#         return torch.transpose(torch.stack(result_list, axis=0), 0, 1)
#
#     def get_extractor(self):
#         return self.backbone


# efficient version that does not compute pairwise, but instead gets a batch of images and corresponding index combinations
# and computes outputs based on the indices dynamically
class efficient_multi_LILAC(nn.Module):
    def __init__(self, args, num_heads, output_scaling="non"):
        super().__init__()
        self.backbone, self.linear = get_backbone(args)
        self.optional_meta = len(args.optional_meta) > 0
        self.network_heads = []
        for head in range(num_heads):
            self.network_heads.append(copy.deepcopy(self.linear))
            # TODO remove original linear layer here?
        self.network_heads = nn.ModuleList(self.network_heads)
        self.linear = None

        self.scaling_list = []
        for attribute in args.attribute_list:
            if output_scaling in scaling_dict.keys():
                if attribute in scaling_dict[output_scaling].keys():
                    self.scaling_list.append(scaling_dict[output_scaling][attribute])
                else:
                    self.scaling_list.append(1.0)

            else:
                self.scaling_list.append(1.0)

    # def forward(self, x, index_pairs, meta=None):
    #     features = self.backbone(x)
    #     f = contrastive_index(features, index_pairs, meta)
    #     return f

    def forward(self, x, index_pairs, meta=None):
        # print('printing index pairs:')
        # print(index_pairs)
        features = self.backbone(x)

        f = contrastive_index(features, index_pairs, meta)
        # print('features shape', features.shape)
        # print('feature pairs', features[index_pairs[0]].shape, features[index_pairs[1]].shape)
        # f = features[index_pairs[0]] - features[index_pairs[1]]
        #
        # if meta is not None:
        #     print('f', f.shape, 'meta', meta[index_pairs[0]].shape)
        #     meta = torch.as_tensor(meta[index_pairs[0]], dtype=f.dtype, device=f.device)
        #     f = torch.cat((f, meta[index_pairs[0]]), dim=1)

        result_list = []
        for i, head in enumerate(self.network_heads):
            # print('head product -->', f.shape, len(self.scaling_list))
            result = head(f) * self.scaling_list[i]
            result_list.append(result)

        return torch.transpose(torch.stack(result_list, dim=0), 0, 1)

    def get_extractor(self):
        return self.backbone

# important args:
# attribute_list
# csv_file_train/val/test
# batchsize
# max_class_size = args.max_class_size
# image_size
# image_directory
# rescale_intensity
# output_fullname

