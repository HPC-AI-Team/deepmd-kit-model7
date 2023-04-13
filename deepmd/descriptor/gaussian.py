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
    get_gaussian_variables_from_graph_def,
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

from .se_atten import (
    DescrptSeAtten
)

from IPython import embed

@Descriptor.register("gaussian")
class DescrptGaussian(DescrptSeAtten):
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
        neuron: List[int] = [24, 48, 96],
        axis_neuron: int = 8,
        resnet_dt: bool = False,
        type_one_side: bool = True,
        exclude_types: List[List[int]] = [],
        set_davg_zero: bool = False,
        activation_function: str = "tanh",
        attn: int = 128,
        attn_layer: int = 2,
        attn_dotr: bool = True,
        attn_mask: bool = False,
        kernel_num: int = 128,
        tebd_size: int = 8,
        trainable: bool = True,
        seed: Optional[int] = None,
        precision: str = "default",
        uniform_seed: bool = False,
        use_D: bool = True,
        use_G: bool = True,
        return_G: bool = False,
    ) -> None:
        """
        Constructor
        """
        assert Version(TF_VERSION) > Version(
            "2"
        ), "gaussian descriptor only supports tensorflow version 2.0 or higher."
        DescrptSeAtten.__init__(
                self,
                rcut=rcut,
                rcut_smth=rcut_smth,
                sel=sel,
                ntypes=ntypes,
                neuron=neuron,
                axis_neuron=axis_neuron,
                resnet_dt=resnet_dt,
                trainable=trainable,
                seed=seed,
                type_one_side=type_one_side,
                exclude_types=exclude_types,
                set_davg_zero=set_davg_zero,
                activation_function=activation_function,
                precision=precision,
                uniform_seed=uniform_seed,
                attn=attn,
                attn_layer=attn_layer,
                attn_dotr=attn_dotr,
                attn_mask=attn_mask,
                return_G=return_G)
        self.use_D = use_D
        self.use_G = use_G
        self.tebd_size = tebd_size
        self.G_dim = self.get_dim_rot_mat_1()
        self.D_dim = self.get_dim_out()
        self.kernel_num = kernel_num
        self.start_mean = 0.0
        self.stop_mean = 9.0
        self.std_width = 1.0
        if use_G:
            self.kernel_num = self.G_dim
        # affine transformation
        self.mul = np.ones([self.ntypes * (self.ntypes + 1), 1])
        self.bias = np.zeros([self.ntypes * (self.ntypes + 1), 1])
        self.gaussian_variables = None
        # gaussian kernels
        self.means = None
        self.stds = None

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
        if self.use_D or self.use_G:
            super().compute_input_stats(
                data_coord=data_coord,
                data_box=data_box,
                data_atype=data_atype,
                natoms_vec=natoms_vec,
                mesh=mesh,
                input_dict=input_dict,
                mixed_type=mixed_type,
                real_natoms_vec=real_natoms_vec,
            )

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
        davg = self.davg
        dstd = self.dstd
        self.debug_dict = {}
        with tf.variable_scope("descrpt_attr" + suffix, reuse=reuse):
            if davg is None:
                davg = np.zeros([self.ntypes, self.ndescrpt])
            if dstd is None:
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
        self.attn_weight = [None for i in range(self.attn_layer)]
        self.angular_weight = [None for i in range(self.attn_layer)]
        self.attn_weight_final = [None for i in range(self.attn_layer)]
        self.debug_dict["clean_coord"] = input_dict['clean_coord']
        self.debug_dict["clean_atype"] = input_dict['clean_type']
        self.debug_dict["noise_mask"] = input_dict['noise_mask']
        self.debug_dict["token_mask"] = input_dict['token_mask']
        self.debug_dict["preop_coord"] = coord
        self.debug_dict["preop_atype"] = atype
        self.debug_dict["preop_natoms"] = natoms
        self.debug_dict["preop_box"] = box
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
        self.debug_dict["rij"] = self.rij
        self.debug_dict["nlist"] = self.nlist
        self.debug_dict["descrpt"] = self.descrpt
        self.debug_dict["nmask"] = self.nmask

        if self.use_D or self.use_G:
            self.nei_type_vec = tf.reshape(self.nei_type_vec, [-1])
            self.nmask = tf.cast(
                tf.reshape(self.nmask, [-1, 1, self.sel_all_a[0]]),
                self.filter_precision,
            )
            self.negative_mask = -(2 << 32) * (1.0 - self.nmask)
            # only used when tensorboard was set as true
            tf.summary.histogram("descrpt", self.descrpt)
            tf.summary.histogram("rij", self.rij)
            tf.summary.histogram("nlist", self.nlist)

            self.descrpt_reshape = tf.reshape(self.descrpt, [-1, self.ndescrpt])
            self.atype_nloc = tf.reshape(
                tf.slice(atype, [0, 0], [-1, natoms[0]]), [-1]
            )  ## lammps will have error without this
            self._identity_tensors(suffix=suffix)

            self.dout, self.qmat = self._pass_filter(
                self.descrpt_reshape,
                self.atype_nloc,
                natoms,
                input_dict,
                suffix=suffix,
                reuse=reuse,
                trainable=self.trainable,
            )
            self.pair_rep = self.xyz_scatter_att
            self.atomic_rep = self.dout

        if not self.use_G:
            # (nframes x nloc x nnei) x 3
            self.rij = tf.reshape(self.rij, [-1, 3])
            # nframes x nloc x nnei x 1
            self.dist = tf.reshape(tf.sqrt(tf.reduce_sum(tf.square(self.rij), axis=-1)), [-1, natoms[1], self.nnei, 1])
            self.debug_dict["dist"] = self.dist

            # nframes x nloc x nnei
            # val : [0, ntypes + 1) , ntypes is the padding, ntypes - 1 is the mask
            self.nei_type_vec = tf.reshape(self.nei_type_vec, [-1, natoms[1], self.nnei])
            # nframes x nloc x 1
            # val : [0, ntypes), ntypes - 1 is the mask
            atype = tf.reshape(atype, [-1, natoms[1], 1])
            # nframes x nloc x nnei
            # val : [0, ntypes * (ntypes + 1) )
            self.edge_type = atype * (self.ntypes + 1) + self.nei_type_vec

            with tf.variable_scope('gaussian' + suffix, reuse=reuse):
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
                    self.t_means = tf.cast(tf.linspace(self.start_mean, self.stop_mean, self.kernel_num), self.filter_precision)
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
                    self.t_stds = tf.cast(self.std_width * (self.stop_mean - self.start_mean) / (self.kernel_num - 1), self.filter_precision)
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

            self.x = gaussian(self.x, self.t_means, tf.abs(self.t_stds) + 1e-5)
            # self.x = gaussian(self.x, 0.0, 1.0)
            self.pair_rep = self.x

        if not self.use_D:
            assert (
                    input_dict is not None
                    and input_dict.get("type_embedding", None) is not None
            ), "se_atten desctiptor must use type_embedding"
            type_embedding = input_dict.get("type_embedding", None)

            # nframes x nloc
            atype = tf.reshape(atype, [-1, natoms[1]])
            # nframes x nloc x type_size
            self.atm_embed = tf.nn.embedding_lookup(type_embedding, atype)
            self.atomic_rep = self.atm_embed

        if not self.return_G:
            return self.atomic_rep
        else:
            return self.atomic_rep, self.pair_rep

    def get_dim_out(self) -> int:
        """
        Returns the output dimension of this descriptor
        """
        if self.use_D:
            return self.filter_neuron[-1] * self.n_axis_neuron
        else:
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
        self.gaussian_variables = get_gaussian_variables_from_graph_def(
            graph_def, suffix=suffix
        )
        self.mul = self.gaussian_variables['gaussian{}/mul'.format(suffix)]
        self.bias = self.gaussian_variables['gaussian{}/bias'.format(suffix)]
        # if self.attn_layer > 0:
        #     self.beta[0] = self.attention_layer_variables[
        #         "attention_layer_0{}/layer_normalization/beta".format(suffix)
        #     ]
        #     self.gamma[0] = self.attention_layer_variables[
        #         "attention_layer_0{}/layer_normalization/gamma".format(suffix)
        #     ]
        #     for i in range(1, self.attn_layer):
        #         self.beta[i] = self.attention_layer_variables[
        #             "attention_layer_{}{}/layer_normalization_{}/beta".format(
        #                 i, suffix, i
        #             )
        #         ]
        #         self.gamma[i] = self.attention_layer_variables[
        #             "attention_layer_{}{}/layer_normalization_{}/gamma".format(
        #                 i, suffix, i
        #             )
        #         ]
