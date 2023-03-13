import math
from abc import ABC
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
from packaging.version import (
    Version,
)

from deepmd.common import (
    cast_precision,
    get_activation_func,
    get_precision,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    TF_VERSION,
    default_tf_session_config,
    op_module,
    tf,
)
from deepmd.utils.errors import (
    GraphWithoutTensorError,
)
from deepmd.utils.graph import (
    get_attention_layer_variables_from_graph_def,
    get_tensor_by_name,
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.utils.network import (
    embedding_net,
    embedding_net_rand_seed_shift,
    one_layer,
)
from deepmd.utils.sess import (
    run_sess,
)
from deepmd.utils.tabulate import (
    DPTabulate,
)
from deepmd.utils.type_embed import (
    embed_atom_type,
)

from .descriptor import (
    Descriptor,
)

from IPython import embed

@Descriptor.register("gaussian")
class DescrptGaussian(Descriptor, ABC):
    """
    Parameters
    ----------
    rcut
            The cut-off radius :math:`r_c`
    rcut_smth
            From where the environment matrix should be smoothed :math:`r_s`
    sel : list[str]
            sel[i] specifies the maxmum number of type i atoms in the cut-off radius
    trainable
            If the weights of embedding net are trainable.
    seed
            Random seed for initializing the network parameters.
    precision
            The precision of the embedding net parameters. Supported options are |PRECISION|
    uniform_seed
            Only for the purpose of backward compatibility, retrieves the old behavior of using the random seed
    return_G
            Return G or not.
    """

    def __init__(
        self,
        rcut: float,
        rcut_smth: float,
        sel: int,
        ntypes: int,
        kernel_num: int = 128,
        tebd_size: int = 8,
        trainable: bool = True,
        seed: Optional[int] = None,
        precision: str = "default",
        uniform_seed: bool = False,
        return_G: bool = False,
    ) -> None:
        """
        Constructor
        """
        assert Version(TF_VERSION) > Version(
            "2"
        ), "gaussian descriptor only supports tensorflow version 2.0 or higher."
        self.sel_a = [sel]
        self.rcut_r = rcut
        self.rcut_r_smth = rcut_smth
        self.seed = seed
        self.uniform_seed = uniform_seed
        self.trainable = trainable
        self.filter_precision = get_precision(precision)
        self.kernel_num = kernel_num
        self.tebd_size = tebd_size

        # descrpt config
        self.sel_r = [0]
        self.rcut_a = -1
        # numb of neighbors and numb of descrptors
        self.nnei_a = np.cumsum(self.sel_a)[-1]
        self.nnei_r = np.cumsum(self.sel_r)[-1]
        self.nnei = self.nnei_a + self.nnei_r
        self.ndescrpt_a = self.nnei_a * 4
        self.ndescrpt_r = self.nnei_r * 1
        self.ndescrpt = self.ndescrpt_a + self.ndescrpt_r
        self.dstd = None
        self.davg = None
        self.embedding_net_variables = None
        self.mixed_prec = None
        self.place_holders = {}
        self.original_sel = None
        self.ntypes = ntypes
        self.return_G = return_G
        # affine transformation
        self.mul = np.ones([self.ntypes * (self.ntypes + 1), 1])
        self.bias = np.zeros([self.ntypes * (self.ntypes + 1), 1])
        # gaussian kernels
        self.means = None
        self.stds = None

        # descrpt config
        self.sel_all_a = [sel]
        self.sel_all_r = [0]

    def compute_input_stats(
        self,
        data_coord: list,
        data_box: list,
        data_atype: list,
        natoms_vec: list,
        mesh: list,
        input_dict: dict,
        mixed_type: bool = False,
        real_natoms_vec: Optional[list] = None,
    ) -> None:
        pass

    def build(
        self,
        coord_: tf.Tensor,
        atype_: tf.Tensor,
        natoms: tf.Tensor,
        box_: tf.Tensor,
        mesh: tf.Tensor,
        input_dict: dict,
        reuse: Optional[bool] = None,
        suffix: str = "",
    ) -> Union[tf.Tensor, Tuple[tf.Tensor, tf.Tensor]]:
        """
        Build the computational graph for the descriptor

        Parameters
        ----------
        coord_
            The coordinate of atoms
        atype_
            The type of atoms
        natoms
            The number of atoms. This tensor has the length of Ntypes + 2
            natoms[0]: number of local atoms
            natoms[1]: total number of atoms held by this processor
            natoms[i]: 2 <= i < Ntypes+2, number of type i atoms
        mesh
            For historical reasons, only the length of the Tensor matters.
            if size of mesh == 6, pbc is assumed.
            if size of mesh == 0, no-pbc is assumed.
        input_dict
            Dictionary for additional inputs
        reuse
            The weights in the networks should be reused when get the variable.
        suffix
            Name suffix to identify this descriptor

        Returns
        -------
        descriptor
            The output descriptor
        """
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            davg = np.zeros([self.ntypes, self.ndescrpt])
            dstd = np.ones([self.ntypes, self.ndescrpt])
            t_rcut = tf.constant(
                np.max([self.rcut_r, self.rcut_a]),
                name="rcut",
                dtype=GLOBAL_TF_FLOAT_PRECISION,
            )
            t_ntypes = tf.constant(self.ntypes, name="ntypes", dtype=tf.int32)
            t_ndescrpt = tf.constant(self.ndescrpt, name="ndescrpt", dtype=tf.int32)
            t_sel = tf.constant(self.sel_a, name="sel", dtype=tf.int32)
            t_original_sel = tf.constant(
                self.original_sel if self.original_sel is not None else self.sel_a,
                name="original_sel",
                dtype=tf.int32,
            )
            self.t_avg = tf.get_variable(
                "t_avg",
                davg.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(davg),
            )
            self.t_std = tf.get_variable(
                "t_std",
                dstd.shape,
                dtype=GLOBAL_TF_FLOAT_PRECISION,
                trainable=False,
                initializer=tf.constant_initializer(dstd),
            )

        with tf.control_dependencies([t_sel, t_original_sel]):
            coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        box = tf.reshape(box_, [-1, 9])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        # self.debug_clean_coord = input_dict['clean_coord']
        # self.debug_clean_atype = input_dict['clean_type']
        # self.debug_noise_mask = input_dict['noise_mask']
        # self.debug_token_mask = input_dict['token_mask']
        # self.debug_preop_coord = coord
        # self.debug_preop_atype = atype
        # self.debug_preop_natoms = natoms
        # self.debug_preop_box = box
        (
            self.descrpt,
            self.descrpt_deriv,
            self.rij,
            self.nlist,
            self.nei_type_vec,
            self.nmask,
        ) = op_module.prod_env_mat_a_mix(
            coord,
            atype,
            natoms,
            box,
            mesh,
            self.t_avg,
            self.t_std,
            rcut_a=self.rcut_a,
            rcut_r=self.rcut_r,
            rcut_r_smth=self.rcut_r_smth,
            sel_a=self.sel_all_a,
            sel_r=self.sel_all_r,
        )
        input_dict['rij'] = self.rij
        input_dict['nlist'] = self.nlist
        input_dict['descrpt'] = self.descrpt
        input_dict['nmask'] = self.nmask

        # (nframes x nloc x nnei) x 3
        self.rij = tf.reshape(self.rij, [-1, 3])
        # nframes x nloc x nnei x 1
        self.dist = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.rij), axis=-1)), [-1, natoms[1], self.nnei, 1])

        # nframes x nloc x nnei
        # val : [0, ntypes + 1) , ntypes is the padding, ntypes - 1 is the mask
        self.nei_type_vec = tf.reshape(self.nei_type_vec, [-1, natoms[1], self.nnei])
        # nframes x nloc x 1
        # val : [0, ntypes), ntypes - 1 is the mask
        atype = tf.reshape(atype, [-1, natoms[1], 1])
        # nframes x nloc x nnei
        # val : [0, ntypes * (ntypes + 1) )
        self.edge_type = atype * (self.ntypes + 1) + self.nei_type_vec

        with tf.variable_scope('gaussian', reuse=reuse):
            # init affine transformation
            mul_initializer = tf.constant_initializer(self.mul)
            self.t_mul = tf.get_variable(
                        "mul",
                        [self.ntypes * (self.ntypes + 1), 1],
                        self.filter_precision,
                        mul_initializer,
                        trainable=self.trainable,
            )
            bias_initializer = tf.constant_initializer(self.bias)
            self.t_bias = tf.get_variable(
                        "bias",
                        [self.ntypes * (self.ntypes + 1), 1],
                        self.filter_precision,
                        bias_initializer,
                        trainable=self.trainable,
            )

            # init gaussian kernels
            if self.means is None:
                means_initializer = tf.random_uniform_initializer(
                    minval=0.0, maxval=3.0,
                    seed=self.seed if (self.seed is None or self.uniform_seed) else self.seed + 0,
                )
            else:
                means_initializer = tf.constant_initializer(self.means)
            self.t_means = tf.get_variable(
                        "means",
                        [1, 1, 1, self.kernel_num],
                        self.filter_precision,
                        means_initializer,
                        trainable=self.trainable,
            )
            if self.stds is None:
                stds_initializer = tf.random_uniform_initializer(
                    minval=0.0, maxval=3.0,
                    seed=self.seed if (self.seed is None or self.uniform_seed) else self.seed + 1,
                )
            else:
                stds_initializer = tf.constant_initializer(self.stds)
            self.t_stds = tf.get_variable(
                        "stds",
                        [1, 1, 1, self.kernel_num],
                        self.filter_precision,
                        stds_initializer,
                        trainable=self.trainable,
            )
        # nframes x nloc x nnei x 1
        self.edge_mul = tf.nn.embedding_lookup(self.t_mul, self.edge_type)
        self.edge_bias = tf.nn.embedding_lookup(self.t_bias, self.edge_type)

        # nframes x nloc x nnei x K
        self.x = tf.tile(self.edge_mul * self.dist + self.edge_bias, [1, 1, 1, self.kernel_num])

        def gaussian(x, mean, std):
            pi = 3.14159
            a = (2 * pi) ** 0.5
            return tf.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

        # self.x = gaussian(self.x, self.t_means, tf.abs(self.t_stds) + 1e-5)
        self.x = gaussian(self.x, 0.0, 1.0)

        assert (
                input_dict is not None
                and input_dict.get("type_embedding", None) is not None
        ), "se_atten desctiptor must use type_embedding"
        type_embedding = input_dict.get("type_embedding", None)

        # nframes x nloc
        atype = tf.reshape(atype, [-1, natoms[1]])
        # nframes x nloc x type_size
        self.atm_embed = tf.nn.embedding_lookup(type_embedding, atype)

        if not self.return_G:
            return self.atm_embed
        else:
            return self.atm_embed, self.x

    def get_dim_out(self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        return self.tebd_size

    def get_ntypes(self) -> int:
        """
        Returns the number of atom types
        """
        return self.ntypes

    def get_rcut(self) -> float:
        """
        Returns the cut-off radius
        """
        return self.rcut_r

    def get_kernel_num(self) -> int:
        """
        Returns the cut-off radius
        """
        return self.kernel_num

    def prod_force_virial(
        self, atom_ener: tf.Tensor, natoms: tf.Tensor
    ):
        pass

    def init_variables(
        self,
        graph: tf.Graph,
        graph_def: tf.GraphDef,
        suffix: str = "",
    ) -> None:
        """
        Init the embedding net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str, optional
            The suffix of the scope
        """
        super().init_variables(graph=graph, graph_def=graph_def, suffix=suffix)
        self.attention_layer_variables = get_attention_layer_variables_from_graph_def(
            graph_def, suffix=suffix
        )
        if self.attn_layer > 0:
            self.beta[0] = self.attention_layer_variables[
                "attention_layer_0{}/layer_normalization/beta".format(suffix)
            ]
            self.gamma[0] = self.attention_layer_variables[
                "attention_layer_0{}/layer_normalization/gamma".format(suffix)
            ]
            for i in range(1, self.attn_layer):
                self.beta[i] = self.attention_layer_variables[
                    "attention_layer_{}{}/layer_normalization_{}/beta".format(
                        i, suffix, i
                    )
                ]
                self.gamma[i] = self.attention_layer_variables[
                    "attention_layer_{}{}/layer_normalization_{}/gamma".format(
                        i, suffix, i
                    )
                ]
