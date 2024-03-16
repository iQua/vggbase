"""
This is the implementation of the basic structure of the
visual and query (vaq) transformer.

This is a generalized structure as the query can be replaced with
    1.- text for the visual grounding
    2.- detection token for the object detection

This transformer receivers the vision and query as the input.

The visual input is referred to as the patches.
The query input is referred to as the q_tokens.
The additional input is referred to as the a_tokens.


This implementation of vaq transformer provides a consistent structure for subsequence research.

It is designed to be:
    - Block
    - Layer
    - Main

Block: the main operation level
    is the major component of the vaq. It performs the attention mechanism with key, query,
    value elements, the MLP, and the residual connection.

    input   -> norm -> ---> multi-head attn ->  + ->  norm -> MLP ->    output
                 |                              |       |                 |
                 |______________________________|       |_________________|
    Thus, Blocks does maintain
        - #channels of the input.

        Therefore, the input and output of the one blocks have the same channels


Layer: the intermediate level
    is the container and packaging component of the vaq. It includes multiple blocks and necessary
    operations required to process the learned representation to be forward to the next layer.

    input --> layer_ipt_op -> block1->block2->...->blockn -> layer_opt_op --> output

"""

import math
from typing import Optional, List

import torch
import torch.nn as nn

from vggbase.models.regions import AdaptivePatchMapper, PatchMapper
from vggbase.models.position_encoding import absolute_position_encoding

from vggbase.models.transformers.basic_vaq_layer import BasicVaQLayer


# pylint: disable=invalid-name
class VaQEmbedder(nn.Module):
    """The basic embedding encoding operation.

    Basically, it includes:
        - position encoding
        - token type encoding

    The output embedding is the sum of these two.


    Inputs:
        embedding_features (int): the features of the embeded
            tensor.
        token_types_count (int): the number of types of the tokens
            for this visual and query structure, there are two types
            of tokens. Thus, Default: 2.
        is_custom_position_ids (Boolean): whether the input tensor
            of the position encoder contains the position ids.
            If True,
                the input tensor should contain the position ids that
                will be used for position encoding.
            If False,
                the values containing in the input tensor can be randomly
                as only the shape of the tensor will be used to generate
                position ids for further encoding
    """

    def __init__(
        self,
        embedding_features,
        token_types_count=2,
        is_custom_position_ids=False,
        drop_prob=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()

        self.visual_features = embedding_features
        self.query_features = embedding_features
        self.token_type_features = embedding_features
        self.token_types_count = token_types_count
        self.is_custom_position_ids = is_custom_position_ids

        self.visual_position_encoder = None
        self.query_position_encoder = None
        self.token_type_encoder = None

        self.token_type_encoder = None
        self.token_type_mapper = {}

        # define the position encoder for visual and the query,
        self.init_visual_position_encoder()
        self.init_query_position_encoder()

        # define the embedding for two types of token,
        self.init_token_type_encoder()
        self.init_token_types_mapper()

        self.LayerNorm = norm_layer(embedding_features, eps=1e-12)
        self.dropout = nn.Dropout(drop_prob)

    def init_visual_position_encoder(self):
        """Customize the position encoder for the visual part."""
        # define the absolute position encoder
        self.visual_position_encoder = (
            absolute_position_encoding.SinusoidalPosition2DEncoder(
                enc_features=self.visual_features,
                temperature=10000.0,
                scale=2 * math.pi,
                is_custom_position_ids=self.is_custom_position_ids,
                is_apply_scale=True,
                device=None,
            )
        )

    def init_query_position_encoder(self):
        """Customize the position encoder for the query part."""
        # define the absolute position encoder
        self.query_position_encoder = (
            absolute_position_encoding.SinusoidalPosition1DEncoder(
                enc_features=self.query_features,
                temperature=10000.0,
                scale=2 * math.pi,
                is_custom_position_ids=self.is_custom_position_ids,
            )
        )

    def init_token_type_encoder(self):
        """Customize the encoder for different input token types."""
        # query should be 0
        # visual should be 1
        self.token_type_encoder = nn.Embedding(
            self.token_types_count, self.token_type_features
        )

    def init_token_types_mapper(self):
        """Customize the type to fun mapper."""
        self.token_type_mapper = {"visual": torch.ones, "tquery": torch.zeros}

    def generate_token_types_id(self, desired_shape, token_type, device):
        """Generate the ids for the tokens based on the shape and the type."""

        return self.token_type_mapper[token_type](
            desired_shape, device=device, dtype=torch.long
        )

    def forward(self, input_tv, input_tq, tv_positions, tq_positions):
        """Forward the input to obtain the embeddings.

        Args:
            input_tv (torch.tensor): a tensor with shape
                [B, C, H, W]
            input_tq (torch.tensor): a tensor with shape
                [B, N, D]
            tv_positions (torch.tensor): a tensor with shape
                [1, H, W] or [1, H, W, 2]
                [B, H, W] or [B, H, W, 2]
            tq_positions (torch.tensor): a tensor with shape
                [1, N] or [1, N, 1]
                [B, N] or [B, N, 1]
                where 2 for tv and 1 for tq are the position
                id when the user defines this ids.
        Note:
            the meaning of values in the input tensor depends
            on the 'is_custom_position_ids'
        """

        # obtain the position encodings
        # tv_pos_encodings, [1, H, W, visual_features]
        # tq_pos_encodings, [1, N, query_features]
        tv_pos_encodings = self.visual_position_encoder(tv_positions)
        tq_pos_encodings = self.query_position_encoder(tq_positions)

        # define the token type
        # visual_token_type_ids, [1, H, W]
        # tquery_token_type_ids, [1, N]
        main_device = input_tv.get_device() if torch.cuda.is_available() else None
        visual_token_type_ids = self.generate_token_types_id(
            desired_shape=tv_positions.shape[0:3],
            token_type="visual",
            device=main_device,
        )
        tquery_token_type_ids = self.generate_token_types_id(
            desired_shape=tq_positions.shape[0:2],
            token_type="tquery",
            device=main_device,
        )
        #
        # obtain the type encodings
        # tv_type_encoding, [1, H, W, token_type_features]
        # tq_type_encoding, [1, N, token_type_features]
        tv_type_encoding = self.token_type_encoder(visual_token_type_ids)
        tq_type_encoding = self.token_type_encoder(tquery_token_type_ids)

        # convert the shape of encodings to be consitent with the inputs
        # tv_pos_encodings, [1, visual_features, H, W]
        # tv_type_encoding, [1, token_type_features, H, W]
        tv_pos_encodings = tv_pos_encodings.permute(0, 3, 1, 2)
        tv_type_encoding = tv_type_encoding.permute(0, 3, 1, 2)

        # add three terms
        # input_tv, [B, visual_features, H, W]
        # input_tq, [B, query_features, N]
        input_tv = input_tv + tv_pos_encodings + tv_type_encoding
        input_tq = input_tq + tq_pos_encodings + tq_type_encoding

        # convert to the shape for normalization
        # convert to
        # input_tv, [B, H * W, C]
        [B, C, H, W] = input_tv.shape
        input_tv = input_tv.view(B, C, -1).permute(0, 2, 1)

        input_tv = self.LayerNorm(input_tv)
        input_tv = self.dropout(input_tv)

        input_tq = self.LayerNorm(input_tq)
        input_tq = self.dropout(input_tq)

        # convert back to the input shape order
        # input_tv, [B, C, H, W]
        input_tv = input_tv.permute(0, 2, 1).view(B, C, H, W)

        return input_tv, input_tq

    def freeze_embedder(self):
        """Freeze compoenets of the embedder."""
        for param in self.visual_position_encoder.parameters():
            param.requires_grad = False
        for param in self.query_position_encoder.parameters():
            param.requires_grad = False
        for param in self.token_type_encoder.parameters():
            param.requires_grad = False

        self.dropout.eval()


# pylint: disable=invalid-name
class VaQTransformer(nn.Module):
    """The basic vision and query (VaQ) transformer for visual grounding.

        The PyTorch impl of : vision transformer for the visual grounding

        This framework directly inherits from the vanilla vision transformer (ViT)
        with a simple modification to process the visual and text (VaT) data for the
        the visual grounding task.
        Therefore, this framework is referred to as VaTTransformer.

    Parameters:
        pretrain_img_size (int): Input image size for training the pretrained model,
            used in absolute postion embedding. Default 224.
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_n_channels (int): Number of input image channels. Default: 3.
        embed_n_features (int): Number of linear projection output features. Default: 96.
                            The embed_n_features is utilized as the basic features of
                            layers of the transformer.
                            For instance, the 96 is the input features of the first layer
                            Then the second layer.
                            The visual and text should be mapped to this features.
        query_position_n_features (int): Number of position encoding features for the query token.
                                Default: ,
        query_encoded_n_features (int): Encoded input query features. Default: 768 for the Bert features.
        depths (tuple[int]): Depths of the VaT transformer. It presents the #Blocks
                            in each layer. i-layer contains #depths[i] Blocks.
        n_heads (tuple[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden features to embedding features. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_features ** -0.5 if set.
        final_pool (str): Which pooling mechanism used to process the output of the transformer
            to prepare for the downstream task. Default: None.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        cross_indices (Sequence[int]): Where to perform the cross attention.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        pretrain_img_size=224,
        patch_size=4,
        in_n_channels=3,
        embed_n_features=96,
        query_position_n_features=96,
        query_encoded_n_features=768,
        depths=[2, 2, 6, 2],
        n_heads=[3, 6, 12, 24],
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        query_embed_drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        norm_layer=nn.LayerNorm,
        ape=False,
        patch_norm=True,
        cross_indices=[0, 1, 2, 3],
        out_indices=[1, 2, 3],
        frozen_stages=-1,
        use_checkpoint=False,
    ) -> None:
        super().__init__()

        # set the necessary parameters
        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.depths = depths
        self.embed_n_features = embed_n_features
        self.query_position_n_features = query_position_n_features
        self.query_encoded_n_features = query_encoded_n_features
        self.ape = ape
        self.patch_norm = patch_norm
        self.cross_indices = cross_indices
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.patch_size = patch_size

        # the model structure parameters
        self.layers = nn.ModuleList()
        # feature channels of each layer
        self.layers_channels = []

        # the necessary components
        # 1. the resolution in terms of pixel
        #  for the input rgb image
        self.original_pixel_resolution = []
        # 2. the resolution of patches obtained
        #  when instantiate the patch mapper.
        self.original_patch_resolution = []
        # 3. the patch_resolution obtained by
        #  performing the patch mapper during
        #  the inference
        # Note, the reason why we need two
        #  variables of path resolution is:
        #  There are two types of patch mapper
        #  one is for adaptive mapping
        #  one is for normal mapping
        #  The normal patch mapper requires to
        #  set the image size when instantiation
        #  thus, the resolution can be computed
        #  directly.
        #  The adaptive one computes the resolution
        #  when processing the rgb image. In this case,
        #  as different rgb batches can have different
        #  H, W, making the resolution is computed after the
        #  padding.
        self.patch_resolution = []

        # number of queries is the maximum queries in one batch
        # as there are multiple samples in one batch and each sample
        # contains different number of queries. After padding, the
        # number of queries for each sample in the batch is the same.
        # we denote this as the self.num_queries.
        # Then, before padding, this number is denoted as
        # self.num_real_queries
        self.num_queries = None
        self.num_real_queries = None
        self.num_patches = None

        # set the droppath rate for blocks
        blocks_dp_rate = self.init_blocks_droppath_prob(drop_path_rate, depths=depths)

        # get the patch mapper to convert 2d rgb image to the patches
        self.patch_mapper, self.original_patch_resolution = self.init_patch_mapper(
            patch_size,
            in_n_channels,
            embed_n_features,
            norm_layer,
            image_size=pretrain_img_size,
        )

        # get the query mapper
        self.query_mapper, self.query_mapped_dropout = self.init_tquery_mapper(
            query_input_features=query_encoded_n_features,
            mapped_n_features=embed_n_features,
            query_embed_drop_rate=query_embed_drop_rate,
        )

        # get the visual and tquery embedder
        self.vaq_embedder = VaQEmbedder(
            embedding_features=self.query_position_n_features,
            is_custom_position_ids=False,
            drop_prob=drop_rate,
            norm_layer=norm_layer,
        )

        # build the layers
        self.init_layers(
            n_heads,
            mlp_ratio,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            blocks_dp_rate=blocks_dp_rate,
            norm_layer=norm_layer,
            use_checkpoint=use_checkpoint,
        )

        # prepare the model structure
        self.prepare_model_structure()

    def to(self, device, **kwargs):
        """Move the sub-module to the device."""
        super().to(device, **kwargs)

        self.vaq_embedder.to(device, **kwargs)

    def prepare_model_structure(self):
        """Prepare the model based on the input parameters."""

        # prepare the channels size for each layer,
        # thus the model structure is same as the one computed during the
        # forward process
        # by default, the channels used in the each layer is the same
        self.layers_channels = [
            int(self.embed_n_features) for _ in range(self.num_layers)
        ]

        structure_config = {}
        # projection matrix for query position encoding in each VaQ layer
        for layer in self.layers:
            layer.prepare_layer_structure(structure_config)

        self._freeze_stages()

    def _freeze_stages(self):
        """Freeze the specific stages based on the requirement."""
        if self.frozen_stages >= 0:
            self.patch_mapper.eval()
            self.patch_mapper.freeze_mapper()

        if self.frozen_stages >= 1 and self.ape:
            self.vaq_embedder.freeze_embedder()

        if self.frozen_stages >= 2:
            for i in range(0, self.frozen_stages - 1):
                layer_i = self.layers[i]
                layer_i.freeze_layer()

    def forward_visual_mapping(self, x_rgb):
        """Forward the mapping for the visual and the query to the
        common space."""

        # For the input x_rgb with shape (B, C, pix_H, pix_W)
        # where the pix_H means the length of H measured by pixels
        # 1. obtain the visual patches from the 2D image.
        #    the output x_rgb of patch_mapper is (B, embed_n_features, Ph, Pw)
        #    where Ph = int(pix_H/patch_size)
        #          Pw = int(pix_W/patch_size)
        #    The Ph and Pw here can be smaller as 'int' is performed.
        #    making the pixels of patch embedded x_rgb is less than
        #    the original one. Thus, padding is required afterwards
        #    if the input masking is utilized to operate with the x_rgb.
        _, _, pix_H, pix_W = x_rgb.shape
        self.original_pixel_resolution = [pix_H, pix_W]
        x_rgb = self.patch_mapper(x_rgb)
        patch_resolution = self.patch_mapper.patches_resolution

        return x_rgb, patch_resolution

    def forward_tquery_mapping(self, x_tquery):
        """Forward the mapping for the query to the common space
        with deired channels."""
        x_tquery = self.query_mapper(x_tquery)
        x_tquery = self.query_mapped_dropout(x_tquery)

        return x_tquery

    def preprocess_masks(self, rgb_mask, tquery_mask):
        """Preprocess the masks."""
        # prepare the rgb mask
        # convert to Boolean type, thus contains elements: True, False
        # rgb_mask,[B, pix_H, pix_W] with element: True, False
        # True for masked while False for unmasked
        rgb_mask = rgb_mask.to(torch.bool)

        # convert to Boolean type, thus contains elements: True, False
        # tquery_mask,[B, N] with element: True, False
        # B, N
        tquery_mask = tquery_mask.to(torch.bool)

        return rgb_mask, tquery_mask

    def get_query_prediction_scores(self, batch_size, num_predictions, device):
        """Obtain the query prediction scores.

        In the visual grounding task, each prediction should correspond to
        one specific query asthe prediction can be regarded as the response
        of this query.
        Thus, the scores should be a one-hot tensor.

        :return a tensor containg one hot values,
            shape, [batch_size, num_predictions, num_queries]

        """
        return torch.arange(num_predictions, device=device).repeat(batch_size, 1)

    def forward_layers(
        self, x_rgb, rgb_mask, x_tquery, tquery_mask, patch_resolution, **kwargs
    ):
        """Forward the built layers of the transformer.

        Args:
            x_rgb [Torch.tensor]: The visual tensor with shape
                B, C, Ph, Pw
            rgb_mask [Torch.tensor]: The visual mask with shape
                B, ori_Ph, ori_Pw
                containing True: masked, False: unmasked
                where ori_Ph and ori_Pw are the #patches after
                performing patch mapping for the original input
                rgb.

                rgb_mask -> layer1 --> layer2 --> layerK
                    ↓          ↑          ↑          ↑
                    ↓__________↑__________↑__________↑

            x_tquery [Torch.tensor]: The query tensor with shape
                B, N, D
            tquery_mask [Torch.tensor]: The query tensor with shape
                B, N
                containing True: masked, False: unmasked
            patch_resolution [Torch.tensor]: The visual tensor with shape
                Ph, Pw
        """
        outputs = None

        # convert x_rgb
        # from B, C, Ph, Pw
        # to B, Ph * Pw, C
        x_rgb = x_rgb.flatten(2).transpose(1, 2)

        for stage in range(self.num_layers):
            layer = self.layers[stage]

            # convert the mask to be the one
            # rgb_mask, [B, Ph, Pw]
            rgb_mask = layer.convert_rgb_mask(rgb_mask, patch_resolution)

            # num_queries_token
            # set the number of query token for the current layer
            # the query length can vary from batch to batch.
            layer.set_n_queries_token(self.num_queries)

            # concat input for efficiency
            # input
            #   - x_rgb: B, Ph * Pw, layer_channels
            #   - x_tquery: B, number_of_queries, layer_channels
            #   if the first layer,
            #    - x_rgb, layer_channels = embed_n_features
            #    - x_tquery, layer_channels = embed_n_features
            # after cat, the x_combined shape:
            #   B, Ph * Pw + number_of_queries, layer_channels
            x_combined = torch.cat([x_rgb, x_tquery], dim=1)

            # forward the combined visual-text to the layer
            outputs = layer(
                x_combined,
                patch_resolution,
                # additional input for pro_trans
                rgb_mask=rgb_mask,
                tquery_mask=tquery_mask,
            )

            # decode the combined output
            [Ph, Pw] = outputs["patch_resolution"]
            x_combined = outputs["output_tvq"]
            self.patch_resolution = [Ph, Pw]
            x_rgb, x_tquery = x_combined[:, : Ph * Pw, :], x_combined[:, Ph * Pw :, :]

        return outputs

    def forward(self, x_rgb, rgb_mask, x_tquery, tquery_mask):
        """Forward function.

        Parameters:
            x_rgb (torch tensor): input rgb images with shape B,C,pix_H,pix_W
            rgb_mask (torch tensor): input padding masks with shape B,pix_H,pix_W
                for the x_rgb [0: rgb values, 1: padded values]
            x_tquery (torch tensor): input query emebdding with shape B, N, D
            tquery_mask (torch tensor): input text padding mask with shape B,N
                where N is the max number of phrases in one batch
                    or N is the max number of words in queries of one batch.
                [0: vaild parts, 1: invaild parts]

            Note, for mask part, 1 always means masked while 0 means unmasked.

        Returns:
            patch_outs: multi-scale [PATCH] tokens (four scales are used)
                these tokens are the first input of the neck decoder
            det_tgt: final [DET] tokens obtained at the last stage
                this tokens are the second input of the neck decoder
                shape:
                    batch_size, channels, num_windows * num_queries

            det_pos: the learnable pos encoding for [DET] tokens.
                these encodings are used to generate reference points in deformable attention
        """
        ## 1. forward the visual mapping
        # visual part -> obtain the visual patches, i.e. visual tokens
        # maintain the consistent name 'x_rgb' for brevity and simplicity
        # convert the shape of x_rgb
        # from B, C, pix_H, pix_W
        # to (B, mapped_features, Ph, Pw)
        x_rgb, self.patch_resolution = self.forward_visual_mapping(x_rgb)

        Ph, Pw = self.patch_resolution[0], self.patch_resolution[1]
        self.num_patches = Ph * Pw

        # query part -> obtain the query tokens
        # maintain the consistent name 'x_tquery' for brevity and simplicity
        # conver the shape of x_tquery
        # from B, N, D
        # to B, num_queries, embed_n_features
        self.num_queries = x_tquery.shape[1]
        # num_real_queries, a 1d tensor  with length[B]
        self.num_real_queries = torch.count_nonzero(tquery_mask, dim=1)

        x_tquery = self.forward_tquery_mapping(x_tquery)

        ## 2. forward the visual and tquery embedding
        # to obtain
        # x_rgb, [B, C, Ph, Pw]
        # x_tquery, [B, N, D]
        main_device = x_tquery.get_device() if torch.cuda.is_available() else None
        rgb_position_tensor = torch.zeros(size=(1, Ph, Pw), device=main_device)
        query_position_tensor = torch.zeros(
            size=(1, self.num_queries), device=main_device
        )
        x_rgb, x_tquery = self.vaq_embedder(
            x_rgb, x_tquery, rgb_position_tensor, query_position_tensor
        )

        ## 3. process the masks to be Boolean type for the subsequent learning
        # - rgb_mask, [B, Ph, Pw] containing True: masked, False: unmasked
        # - tquery_mask: [B, N] containing True: masked, False: unmasked
        rgb_mask, tquery_mask = self.preprocess_masks(rgb_mask, tquery_mask)

        ## 3. forward the built layers
        outputs = self.forward_layers(
            x_rgb,
            rgb_mask,
            x_tquery,
            tquery_mask,
            patch_resolution=self.patch_resolution,
        )

        # decode the combined output
        [Ph, Pw] = outputs["patch_resolution"]
        x_combined = outputs["output_tvq"]
        self.patch_resolution = [Ph, Pw]
        x_rgb, x_tquery = x_combined[:, : Ph * Pw, :], x_combined[:, Ph * Pw :, :]

        B, num_predictions, _ = x_tquery.shape
        aligned_query_predictions = self.get_query_prediction_scores(
            B, num_predictions, device=main_device
        )

        return {
            "visual_output": x_rgb,
            "query_output": x_tquery,
            "aligned_query_predictions": aligned_query_predictions,
            "patch_resolution": [Ph, Pw],
        }

    def init_blocks_droppath_prob(self, max_drop_path_rate, depths):
        """Customize the prob of the drop path for the residual_connect in each block."""
        # by default, the stochastic depth decay rule is utilized
        # stochastic depth decay rule for the residual connection in
        # each block
        blocks_dp_rate = [
            x.item() for x in torch.linspace(0, max_drop_path_rate, sum(depths))
        ]
        return blocks_dp_rate

    def init_tquery_mapper(
        self, query_input_features, mapped_n_features, query_embed_drop_rate
    ):
        """The customize text mapper to map the input text to the embedding space."""

        tquery_mapper = nn.Linear(query_input_features, mapped_n_features, bias=True)
        tquery_map_dropout = nn.Dropout(query_embed_drop_rate)

        return tquery_mapper, tquery_map_dropout

    def init_patch_mapper(
        self,
        patch_size,
        in_n_channels,
        mapped_n_features,
        norm_layer,
        image_size: Optional[List[int]] = None,
    ):
        """The customized patch embedder."""

        if image_size is [None, None]:
            # split image into non-overlapping patches adaptively
            patch_mapper = AdaptivePatchMapper(
                patch_size=patch_size,
                in_n_channels=in_n_channels,
                embed_n_channels=mapped_n_features,
                norm_layer=norm_layer if self.patch_norm else None,
            )

            patches_resolution = None
        else:
            # split image into non-overlapping patches based on the image size
            patch_mapper = PatchMapper(
                img_size=image_size,
                patch_size=patch_size,
                in_n_channels=in_n_channels,
                embed_n_channels=mapped_n_features,
                norm_layer=norm_layer if self.patch_norm else None,
            )

            patches_resolution = patch_mapper.patches_resolution

        return patch_mapper, patches_resolution

    def init_layers(
        self,
        n_heads,
        mlp_ratio,
        qkv_bias,
        qk_scale,
        drop_rate,
        attn_drop_rate,
        blocks_dp_rate,
        norm_layer,
        use_checkpoint,
    ):
        """Customize the layers for the vat transformer."""
        # build layers
        depths = self.depths
        for i_layer in range(self.num_layers):
            layer = BasicVaQLayer(
                n_channels=self.embed_n_features,
                depth=depths[i_layer],
                n_heads=n_heads[i_layer],
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop=drop_rate,
                attn_drop=attn_drop_rate,
                drop_path=blocks_dp_rate[
                    sum(depths[:i_layer]) : sum(depths[: i_layer + 1])
                ],
                norm_layer=norm_layer,
                use_checkpoint=use_checkpoint,
            )
            self.layers.append(layer)


def build_model(model_config: dict, device: Optional[torch.device] = None):
    """Build the grounding model."""
    # an example of configs
    # - pretrain_img_size=[224, 224],
    # - embed_n_features=96,
    # - depths=[2, 2, 6, 2],
    # - n_heads=[3, 6, 12, 24],
    # - drop_path_rate=0.2
    grounding_model = VaQTransformer(
        pretrain_img_size=model_config.grounding_model.pretrain_img_size,
        embed_n_features=model_config.grounding_model.embed_n_features,
        query_position_n_features=model_config.query_position_n_features,
        query_encoded_n_features=model_config.query_encoded_n_features,
        depths=model_config.grounding_model.depths,
        n_heads=model_config.grounding_model.n_heads,
        drop_path_rate=model_config.grounding_model.drop_path_rate,
    )

    return grounding_model
