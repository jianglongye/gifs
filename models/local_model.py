import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D conv usage:
# batch_size (N) = #3D objects , channels = features, signal_lengt (L) (convolution dimension) = #point samples
# kernel_size = 1 i.e. every convolution over only all features of one point sample


class GIFS(nn.Module):
    def __init__(self, hidden_dim=256):
        super(GIFS, self).__init__()

        self.conv_in = nn.Conv3d(1, 16, 3, padding=1)  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv3d(16, 32, 3, padding=1)  # out: 128
        self.conv_0_1 = nn.Conv3d(32, 32, 3, padding=1)  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv3d(32, 64, 3, padding=1)  # out: 64
        self.conv_1_1 = nn.Conv3d(64, 64, 3, padding=1)  # out: 64 -> mp 32
        self.conv_2 = nn.Conv3d(64, 128, 3, padding=1)  # out: 32
        self.conv_2_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 32 -> mp 16
        self.conv_3 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16
        self.conv_3_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 16 -> mp 8
        self.conv_4 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8
        self.conv_4_1 = nn.Conv3d(128, 128, 3, padding=1)  # out: 8

        displacments_num = 1
        feature_size = (1 + 16 + 32 + 64 + 128 + 128 + 128) * displacments_num + 3
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 1, 1)
        self.actvn = nn.ReLU()

        self.gifs_fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.gifs_fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.gifs_fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.gifs_fc_3 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.gifs_fc_out = nn.Conv1d(hidden_dim, 1, 1)

        self.maxpool = nn.MaxPool3d(2)

        self.conv_in_bn = nn.BatchNorm3d(16)
        self.conv0_1_bn = nn.BatchNorm3d(32)
        self.conv1_1_bn = nn.BatchNorm3d(64)
        self.conv2_1_bn = nn.BatchNorm3d(128)
        self.conv3_1_bn = nn.BatchNorm3d(128)
        self.conv4_1_bn = nn.BatchNorm3d(128)

        displacment = 0.0722
        displacments = []
        displacments.append([0, 0, 0])
        # for x in range(3):
        #     for y in [-1, 1]:
        #         input = [0, 0, 0]
        #         input[x] = y * displacment
        #         displacments.append(input)
        assert len(displacments) == displacments_num

        self.displacments = torch.Tensor(displacments).cuda()

    def encoder(self, x):
        x = x.unsqueeze(1)
        f_0 = x

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        f_1 = net
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        f_2 = net
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        f_3 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        f_4 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        f_5 = net
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        f_6 = net

        return f_0, f_1, f_2, f_3, f_4, f_5, f_6

    def embedding(self, p, f_0, f_1, f_2, f_3, f_4, f_5, f_6):
        # pe_embd = self.harmonic_embedding(p).transpose(1, -1)

        p_features = p.transpose(1, -1)

        p = p.unsqueeze(1).unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=2)

        # feature extraction
        feature_0 = F.grid_sample(f_0, p, align_corners=True)
        feature_1 = F.grid_sample(f_1.float(), p, align_corners=True)
        feature_2 = F.grid_sample(f_2.float(), p, align_corners=True)
        feature_3 = F.grid_sample(f_3.float(), p, align_corners=True)
        feature_4 = F.grid_sample(f_4.float(), p, align_corners=True)
        feature_5 = F.grid_sample(f_5.float(), p, align_corners=True)
        feature_6 = F.grid_sample(f_6.float(), p, align_corners=True)

        # here every channel corresponds to one feature.

        features = torch.cat(
            (feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6), dim=1
        )  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(
            features, (shape[0], shape[1] * shape[3], shape[4])
        )  # (B, featues_per_sample, samples_num)
        # features = torch.cat((features, p_features, pe_embd), dim=1)  # (B, featue_size, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num)

        return features

    def udf_mlp(self, feat):
        net = self.actvn(self.fc_0(feat))
        net = self.actvn(self.fc_1(net))
        # net = self.actvn(self.fc_2(net))
        net = self.actvn(self.fc_2(net)) + net
        net = self.actvn(self.fc_3(net)) + net
        net = self.actvn(self.fc_out(net))
        out = net.squeeze(1)

        return out

    def gifs_mlp(self, feat):
        net = self.actvn(self.gifs_fc_0(feat))
        net = self.actvn(self.gifs_fc_1(net))
        # net = self.actvn(self.gifs_fc_2(net))
        net = self.actvn(self.gifs_fc_2(net)) + net
        net = self.actvn(self.gifs_fc_3(net)) + net
        out = self.actvn(self.gifs_fc_out(net))
        out = out.squeeze(1)

        return out

    def query(self, p, fs):
        feat0 = self.embedding(p[:, :, :3], *fs)
        feat1 = self.embedding(p[:, :, 3:], *fs)

        comb_feat = torch.cat([feat0[None], feat1[None]], dim=0)
        comb_feat, _ = torch.max(comb_feat, dim=0)

        udf0 = self.udf_mlp(feat0)
        udf1 = self.udf_mlp(feat1)

        udf = torch.cat([udf0[:, :, None], udf1[:, :, None]], dim=-1)

        binary_flag = self.gifs_mlp(comb_feat)

        return {"udf": udf, "pred": binary_flag, "feat0": feat0, "feat1": feat1}

    def query_udf(self, p, fs):
        feat = self.embedding(p, *fs)

        udf = self.udf_mlp(feat)

        return {"udf": udf}

    def get_udf(self, p, x):
        feat = self.embedding(p, *self.encoder(x))
        udf = self.udf_mlp(feat)

        return {"udf": udf}

    def forward(self, p, x):
        # out = self.decoder(p, *self.encoder(x))
        fs = self.encoder(x)
        feat0 = self.embedding(p[:, :, :3], *fs)
        feat1 = self.embedding(p[:, :, 3:], *fs)

        comb_feat = torch.cat([feat0[None], feat1[None]], dim=0)
        comb_feat, _ = torch.max(comb_feat, dim=0)

        udf0 = self.udf_mlp(feat0)
        udf1 = self.udf_mlp(feat1)

        udf = torch.cat([udf0[:, :, None], udf1[:, :, None]], dim=-1)

        binary_flag = self.gifs_mlp(comb_feat)

        return {"udf": udf, "pred": binary_flag, "feat0": feat0, "feat1": feat1}
