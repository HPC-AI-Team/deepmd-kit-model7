import logging
import warnings
from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.common import (
    add_data_requirement,
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    TF_VERSION,
    global_cvt_2_tf_float,
    tf,
)
from deepmd.backbone.backbone import (
    Backbone,
)
from deepmd.infer import (
    DeepPotential,
)
from deepmd.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.utils.network import one_layer
from deepmd.utils.network import (
    one_layer_rand_seed_shift,
)
from deepmd.utils.type_embed import (
    embed_atom_type,
)

log = logging.getLogger(__name__)


class EvoformerBackbone (Backbone):
    def __init__(self,
                 descrpt: tf.Tensor,
                 evo_layer=6,
                 attn_head=8,
                 feature_dim=1024,
                 ffn_dim=2048,
                 layer_norm: str = 'pre',
                 final_layer_norm: bool = True,
                 final_head_layer_norm: bool = True,
                 emb_layer_norm: bool = True,
                 activation_function: str = "gelu",
                 trainable: List[bool] = None,
                 seed: int = None,
                 precision: str = 'default',
                 uniform_seed: bool = False
                 ) -> None:
        self.ntypes = descrpt.get_ntypes()
        # self.dim_embd = descrpt.get_dim_rot_mat_1()
        self.kernel_num = descrpt.get_kernel_num()
        self.evo_layer = evo_layer
        self.attn_head = attn_head
        self.feature_dim = feature_dim
        self.head_dim = feature_dim // attn_head
        assert self.head_dim * self.attn_head == self.feature_dim, "feature_dim must be divisible by attn_head!"
        self.ffn_dim = ffn_dim
        self.backbone_precision = get_precision(precision)
        self.backbone_variables = None
        self.activation_function = get_activation_func(activation_function)
        self.beta = np.zeros([self.evo_layer * 2 + 3, self.feature_dim]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.gamma = np.ones([self.evo_layer * 2 + 3, self.feature_dim]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.beta_head = np.zeros([1, self.attn_head]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.gamma_head = np.ones([1, self.attn_head]).astype(
            GLOBAL_NP_FLOAT_PRECISION
        )
        self.layer_norm = True
        if layer_norm == 'pre':
            self.pre_layer_norm = True
        elif layer_norm == 'post':
            self.pre_layer_norm = False
        elif layer_norm == 'none':
            self.layer_norm = False
        else:
            RuntimeError(f"Unknown layer norm mode {layer_norm}! Supported options are ['pre', 'post', 'none'].")
        self.final_layer_norm = final_layer_norm
        self.final_head_layer_norm = final_head_layer_norm
        self.emb_layer_norm = emb_layer_norm
        self.trainable = trainable
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.seed_shift = one_layer_rand_seed_shift()

    def build(self,
              inputs: tf.Tensor,
              environ_G: tf.Tensor,
              natoms: tf.Tensor,
              input_dict: dict = None,
              reuse: bool = None,
              suffix: str = '',
              ):
        type_embedding = input_dict.get('type_embedding', None)  # ntypes x 8
        tebd_size = type_embedding.shape[-1]
        atype = input_dict.get('atype', None)
        assert type_embedding is not None and atype is not None, "type_embedding and atype must be transfered to this backbone!"
        atype_nall = tf.reshape(atype, [-1, natoms[1]])
        self.atype_nloc = tf.reshape(tf.slice(atype_nall, [0, 0], [-1, natoms[0]]), [-1])  ## lammps will make error
        nframes = input_dict['nframes']
        nloc = natoms[0]
        nnei = input_dict['nnei']
        nlist = input_dict['nlist']
        nlist = tf.where(tf.equal(nlist, -1), tf.ones_like(nlist, dtype=tf.int32) * nloc, nlist)
        # nframes x nloc x nnei
        nmask = input_dict['nmask']
        # (nframes x nloc) x nnei
        self.namsk_nnei = tf.cast(tf.reshape(nmask, [nframes * nloc, nnei]), dtype=self.backbone_precision)
        # (nframes x nloc) x 1
        self.masked_nnei = tf.reduce_sum(self.namsk_nnei, axis=-1, keepdims=True)
        self.rij = tf.reshape(input_dict['rij'], [nframes * nloc, nnei, 3])
        inputs = tf.reshape(inputs, [nframes * nloc, tebd_size])  # (nframes x nloc) x tebd_size
        environ_G = tf.reshape(environ_G, [nframes * nloc * nnei, self.kernel_num])  # (nframes x nloc x nnei) x K
        self.nmask = tf.reshape(
            tf.tile(
                tf.reshape(
                    nmask,
                    [nframes, 1, nloc, nnei],
                ),
                [1, self.attn_head, 1, 1],
            ),
            [nframes * self.attn_head * nloc, 1, nnei],
        )  # (nframes x h x nloc) x 1 x nnei

        self.scaling = tf.cast(self.head_dim, dtype=self.backbone_precision) ** -0.5
        extra_layer_norm = 0
        extra_head_layer_norm = 0

        # # Global branch
        # atype_embed = tf.nn.embedding_lookup(type_embedding, self.atype_nloc)  # (nframes x nloc) x 8
        # inputs = tf.concat(
        #     [inputs, atype_embed],
        #     axis=1
        # )  # (nframes x nloc) x (M1 x M2 + 8)
        name = 'backbone_layer' + suffix
        with tf.variable_scope(name, reuse=reuse):
            atomic_rep = one_layer(
                        inputs,
                        self.feature_dim,
                        name='atomic_trans',
                        scope=name + '/',
                        reuse=reuse,
                        seed=self.seed,
                        activation_fn=None,
                        precision=self.backbone_precision,
                        trainable=self.trainable,
                        uniform_seed=self.uniform_seed,
                        initial_variables=self.backbone_variables)  # (nframes x nloc) x d
            if self.emb_layer_norm:
                atomic_rep = tf.keras.layers.LayerNormalization(
                    beta_initializer=tf.constant_initializer(self.beta[self.evo_layer * 2 + extra_layer_norm]),
                    gamma_initializer=tf.constant_initializer(self.gamma[self.evo_layer * 2 + extra_layer_norm]),
                )(atomic_rep)
                extra_layer_norm += 1

            # atomic_rep = tf.nn.dropout(atomic_rep, p=self.emb_dropout, training=self.training)

        # Local branch
        with tf.variable_scope(name, reuse=reuse):
            pair_rep = one_layer(
                        environ_G,
                        self.attn_head,
                        name='pair_trans',
                        scope=name + '/',
                        reuse=reuse,
                        seed=self.seed,
                        activation_fn=None,
                        precision=self.backbone_precision,
                        trainable=self.trainable,
                        uniform_seed=self.uniform_seed,
                        initial_variables=self.backbone_variables)  # (nframes x nloc x nnei) x h

        # Communication
        # (nframes x h x nloc) x 1 x nnei
        pair_rep = tf.reshape(
            tf.transpose(
                tf.reshape(
                    pair_rep,
                    [nframes, nloc, nnei, self.attn_head],
                ),
                (0, 3, 1, 2),
            ),
            [nframes * self.attn_head * nloc, 1, nnei],
        )
        input_pair_rep = pair_rep
        # input_atomic_rep = atomic_rep
        # pair_rep = pair_rep + self.negative_mask
        pair_rep = self.mask_fill(pair_rep, self.nmask, -(2 << 32))
        for i in range(self.evo_layer):
            name_layer = 'evo_layer_{}{}'.format(i, suffix)
            with tf.variable_scope(name_layer, reuse=reuse):
                atomic_rep, pair_rep, _ = self.evoformer_encoder(
                    atomic_rep,
                    nlist,
                    nframes,
                    nloc,
                    nnei,
                    layer=i,
                    attn_bias=pair_rep,
                    return_attn=True,
                    scope=name_layer,
                    reuse=reuse,
                )

        # atomic_rep : (nframes x nloc) x d
        # pair_rep : (nframes x h x nloc) x 1 x nnei
        # TODO norm loss of atomic_rep, no need of mask
        norm_x = tf.reduce_mean(self.norm_loss(atomic_rep))
        if self.final_layer_norm:
            atomic_rep = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta[self.evo_layer * 2 + extra_layer_norm]),
                gamma_initializer=tf.constant_initializer(self.gamma[self.evo_layer * 2 + extra_layer_norm]),
            )(atomic_rep)
            extra_layer_norm += 1

        # 1 : use delta_pair_rep
        # (nframes x h x nloc) x 1 x nnei
        delta_pair_rep = pair_rep - input_pair_rep
        delta_pair_rep = self.mask_fill(delta_pair_rep, self.nmask, 0.0)
        # (nframes x nloc x nnei) x h
        delta_pair_rep = tf.reshape(
            tf.transpose(
                tf.reshape(
                    delta_pair_rep,
                    [nframes, self.attn_head, nloc, nnei],
                ),
                (0, 2, 3, 1),
            ),
            [nframes * nloc * nnei, self.attn_head],
        )

        # TODO norm loss of delta_pair_rep, need mask
        norm_delta_pair_rep = self.norm_loss(delta_pair_rep)
        norm_delta_pair_rep = tf.reshape(norm_delta_pair_rep, [nframes * nloc, nnei])
        norm_delta_pair_rep = self.masked_mean(self.namsk_nnei, norm_delta_pair_rep)

        if self.final_head_layer_norm:
            delta_pair_rep = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta_head[extra_head_layer_norm]),
                gamma_initializer=tf.constant_initializer(self.gamma_head[extra_head_layer_norm]),
            )(delta_pair_rep)
            extra_head_layer_norm += 1

        # TODO 2 : use delta_G

        # encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0 ?

        # HEAD1: coord head
        # (nframes x nloc x nnei) x h
        attn_probs = one_layer(
            delta_pair_rep,
            self.attn_head,
            name="pair2coord_1",
            reuse=reuse,
            seed=self.seed,
            activation_fn=self.activation_function,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        # (nframes x nloc x nnei) x 1
        attn_probs = one_layer(
            attn_probs,
            1,
            name="pair2coord_2",
            reuse=reuse,
            seed=self.seed,
            activation_fn=None,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        # (nframes x nloc) x 1 x nnei
        attn_probs = tf.reshape(attn_probs, [nframes * nloc, 1, nnei])
        # (nframes x nloc) x 1 x 3
        coord_update = tf.matmul(attn_probs, self.rij)
        # (nframes x nloc) x 3
        coord_update = tf.reshape(coord_update, [nframes * nloc, 3]) / self.masked_nnei

        # HEAD2: element head
        # type_predict = one_layer(
        #     atomic_rep,
        #     self.ntypes,
        #     name="atom2type",
        #     reuse=reuse,
        #     seed=self.seed,
        #     activation_fn=self.activation_function,
        #     precision=self.backbone_precision,
        #     trainable=self.trainable,
        #     uniform_seed=self.uniform_seed,
        #     initial_variables=self.backbone_variables,
        # )
        type_predict = one_layer(
            atomic_rep,
            self.feature_dim,
            name="atom2type_1",
            reuse=reuse,
            seed=self.seed,
            activation_fn=self.activation_function,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        type_predict = tf.keras.layers.LayerNormalization(
            beta_initializer=tf.constant_initializer(self.beta[self.evo_layer * 2 + extra_layer_norm]),
            gamma_initializer=tf.constant_initializer(self.gamma[self.evo_layer * 2 + extra_layer_norm]),
        )(type_predict)
        extra_layer_norm += 1
        logits = one_layer(
            type_predict,
            self.ntypes - 1,
            name="atom2type_2",
            reuse=reuse,
            seed=self.seed,
            activation_fn=None,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        return atomic_rep, pair_rep, coord_update, logits, norm_x, norm_delta_pair_rep

    def evoformer_encoder(self, x, nlist, nframes, nloc, nnei, layer=0, attn_bias=None, return_attn=False, scope="", reuse=None):
        residual = x
        if self.layer_norm and self.pre_layer_norm:
            x = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta[layer * 2]),
                gamma_initializer=tf.constant_initializer(self.gamma[layer * 2]),
            )(x)
        x = self.LocalSelfMultiheadAttention(
            query=x,
            nlist=nlist,
            nframes=nframes,
            nloc=nloc,
            nnei=nnei,
            attn_bias=attn_bias,
            return_attn=return_attn,
            scope=scope,
            reuse=reuse,
        )
        if return_attn:
            x, attn_weights, attn_probs = x
        # tf.nn.dropout
        x = residual + x
        if self.layer_norm and not self.pre_layer_norm:
            x = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta[layer * 2]),
                gamma_initializer=tf.constant_initializer(self.gamma[layer * 2]),
            )(x)

        residual = x
        if self.layer_norm and self.pre_layer_norm:
            x = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta[layer * 2 + 1]),
                gamma_initializer=tf.constant_initializer(self.gamma[layer * 2 + 1]),
            )(x)
        x = self._feedforward(x, d_in=self.feature_dim, d_mid=self.ffn_dim, scope=scope, reuse=reuse)
        # tf.nn.dropout
        x = residual + x
        if self.layer_norm and not self.pre_layer_norm:
            x = tf.keras.layers.LayerNormalization(
                beta_initializer=tf.constant_initializer(self.beta[layer * 2 + 1]),
                gamma_initializer=tf.constant_initializer(self.gamma[layer * 2 + 1]),
            )(x)
        if not return_attn:
            return x
        else:
            return x, attn_weights, attn_probs

    def LocalSelfMultiheadAttention(self, query, nlist, nframes, nloc, nnei, attn_bias, return_attn, scope="", reuse=None):
        # query : (nframes x nloc) x d
        # nlist : nframes x (nloc*nnei)
        # attn_bias : (nframes x nloc x nnei) x h

        # global
        # q k v : (nframes x nloc) x d
        # q x k : (nframes x nloc) x nloc
        # softmax(q x k) * v : (nframes x nloc) x d

        # local
        # q k v : (nframes x nloc) x d
        # q_b : (nframes x nloc) x 1 x d
        # k_b : (nframes x nloc) x nnei x d
        # v_b : (nframes x nloc) x nnei x d
        # q_b x k_b : (nframes x nloc) x nnei x 1
        # softmax(q_b x k_b) * v_b : (nframes x nloc) x d

        # multi head local
        # q k v : nframes x nloc x h x d1
        # tranpose nframes x h x nloc x d1
        # q_b : nframes x h x nloc x 1 x d1
        # k_b : nframes x h x nloc x nnei x d1
        # v_b : nframes x h x nloc x nnei x d1
        # q_b x k_b : nframes x h x nloc x 1 x nnei
        # mask : nframes x [h] x nloc x 1 x nnei
        # softmax(q_b x k_b) * v_b : nframes x h x nloc x d1
        # tranpose : nframes x nloc x h x d1

        [q, k, v] = tf.split(
            one_layer(
                query,
                self.feature_dim * 3,
                name="in_attn_layer",
                reuse=reuse,
                scope=scope + '/',
                seed=self.seed,
                activation_fn=None,
                precision=self.backbone_precision,
                trainable=self.trainable,
                uniform_seed=self.uniform_seed,
                initial_variables=self.backbone_variables,
            ),
            3,
            axis=1,
        )  # (nframes x nloc) x d
        # (nframes x h x nloc) x 1 x d1
        q = tf.reshape(
            tf.transpose(
                tf.reshape(q,
                           [nframes, nloc, self.attn_head, 1, self.head_dim],
                           ),
                (0, 2, 1, 3, 4),
            ),
            [nframes * self.attn_head * nloc, 1, self.head_dim],
        ) * self.scaling
        # padding for blank neighbors
        # nframes x 1 x d
        padding = tf.cast(tf.zeros([nframes, 1, self.feature_dim]), self.backbone_precision)
        # nframes x (nloc + 1) x d
        k = tf.concat([tf.reshape(k, [nframes, nloc, self.feature_dim]), padding], axis=1)
        # nframes x (nloc + 1) x d
        v = tf.concat([tf.reshape(v, [nframes, nloc, self.feature_dim]), padding], axis=1)
        # nframes x (nloc x nnei)
        nlist = tf.reshape(nlist, [nframes, nloc * nnei])
        # (nframes x h x nloc) x nnei x d1
        k = tf.reshape(
            tf.transpose(
                tf.reshape(
                    tf.batch_gather(
                        k,
                        nlist,
                    ),
                    [nframes, nloc, nnei, self.attn_head, self.head_dim],
                ),
                (0, 3, 1, 2, 4),
            ),
            [nframes * self.attn_head * nloc, nnei, self.head_dim],
        )
        # (nframes x h x nloc) x nnei x d1
        v = tf.reshape(
            tf.transpose(
                tf.reshape(
                    tf.batch_gather(
                        v,
                        nlist,
                    ),
                    [nframes, nloc, nnei, self.attn_head, self.head_dim],
                ),
                (0, 3, 1, 2, 4),
            ),
            [nframes * self.attn_head * nloc, nnei, self.head_dim],
        )
        # (nframes x h x nloc) x 1 x nnei
        attn_weights = tf.matmul(q, k, transpose_b=True)
        # attn_weights += self.negative_mask
        attn_weights = self.mask_fill(attn_weights, self.nmask, -(2 << 32))

        if return_attn:
            attn_weights += attn_bias

        # softmax
        # (nframes x h x nloc) x 1 x nnei
        attn = tf.nn.softmax(attn_weights, axis=-1)
        # (nframes x h x nloc) x 1 x d1
        o = tf.matmul(attn, v)
        # (nframes x nloc) x (h x d1)
        o = tf.reshape(
            tf.transpose(
                tf.reshape(
                    o,
                    [nframes, self.attn_head, nloc, self.head_dim],
                ),
                (0, 2, 1, 3),
            ),
            [nframes * nloc, self.attn_head * self.head_dim],
        )
        if not return_attn:
            return o
        else:
            return o, attn_weights, attn

    def _feedforward(self, x, d_in, d_mid, scope="", reuse=None):
        x = one_layer(
            x,
            d_mid,
            name="c_ffn1",
            reuse=reuse,
            scope=scope + '/',
            seed=self.seed,
            activation_fn=self.activation_function,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        # tf.nn.dropout
        x = one_layer(
            x,
            d_in,
            name="c_ffn2",
            reuse=reuse,
            scope=scope + '/',
            seed=self.seed,
            activation_fn=None,
            precision=self.backbone_precision,
            trainable=self.trainable,
            uniform_seed=self.uniform_seed,
            initial_variables=self.backbone_variables,
        )
        # tf.nn.dropout
        return x

    # Fill fill_val in False place in mask
    def mask_fill(self, x, mask, fill_val):
        return tf.where(mask, x, fill_val * tf.ones_like(x, dtype=self.backbone_precision))

    def norm_loss(self, x, eps=1e-10, tolerance=1.0):
        max_norm = tf.sqrt(tf.cast(x.shape[-1], dtype=self.backbone_precision))
        norm = tf.sqrt(tf.reduce_sum(x ** 2, axis=-1) + eps)
        error = tf.nn.relu(tf.abs(norm - max_norm) - tolerance)
        return error

    @staticmethod
    def masked_mean(mask, value, axis=-1, eps=1e-10):
        return tf.reduce_mean(
            tf.reduce_sum(mask * value, axis=axis) / (eps + tf.reduce_sum(mask, axis=axis))
        )








