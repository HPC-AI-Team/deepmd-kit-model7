from typing import (
    List,
    Optional,
    Tuple,
)

import numpy as np

from deepmd.env import (
    GLOBAL_TF_FLOAT_PRECISION,
    MODEL_VERSION,
    global_cvt_2_ener_float,
    op_module,
    tf,
)
from deepmd.utils.pair_tab import (
    PairTab,
)

from .model import (
    Model,
)
from .model_stat import (
    make_stat_input,
    merge_sys_stat,
)


class EvoformerModel(Model):
    """Evoformer model.

    Parameters
    ----------
    descrpt
            Descriptor
    backbone
            Backbone net
    type_map
            Mapping atom type to the name (str) of the type.
            For example `type_map[1]` gives the name of the type 1.
    data_stat_nbatch
            Number of frames used for data statistic
    data_stat_protect
            Protect parameter for atomic energy regression
    use_srtab
            The table for the short-range pairwise interaction added on top of DP. The table is a text data file with (N_t + 1) * N_t / 2 + 1 columes. The first colume is the distance between atoms. The second to the last columes are energies for pairs of certain types. For example we have two atom types, 0 and 1. The columes from 2nd to 4th are for 0-0, 0-1 and 1-1 correspondingly.
    smin_alpha
            The short-range tabulated interaction will be swithed according to the distance of the nearest neighbor. This distance is calculated by softmin. This parameter is the decaying parameter in the softmin. It is only required when `use_srtab` is provided.
    sw_rmin
            The lower boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    sw_rmin
            The upper boundary of the interpolation between short-range tabulated interaction and DP. It is only required when `use_srtab` is provided.
    """
    model_type = 'evoformer'

    def __init__(
            self,
            descrpt,
            backbone,
            typeebd,
            type_map: List[str] = None,
            data_stat_nbatch: int = 10,
            data_stat_protect: float = 1e-2,
            noise: float = 1.00,
            noise_type: str = 'uniform',
    ) -> None:
        """
        Constructor
        """
        # descriptor
        self.descrpt = descrpt
        assert self.descrpt.return_G, "Descrpt in evoformer must return G!"
        self.rcut = self.descrpt.get_rcut()
        self.ntypes = self.descrpt.get_ntypes()
        # backbone
        self.backbone = backbone
        # type embedding
        self.typeebd = typeebd
        # other inputs
        if type_map is None:
            self.type_map = []
        else:
            self.type_map = type_map
        self.data_stat_nbatch = data_stat_nbatch
        self.data_stat_protect = data_stat_protect
        self.noise = float(noise)
        self.noise_type = noise_type

    def get_rcut(self):
        return self.rcut

    def get_ntypes(self):
        return self.ntypes

    def get_type_map(self):
        return self.type_map

    def data_stat(self, data):
        all_stat = make_stat_input(data, self.data_stat_nbatch, merge_sys=False)
        m_all_stat = merge_sys_stat(all_stat)
        self._compute_input_stat(m_all_stat, mixed_type=data.mixed_type)
        # self._compute_output_stat(all_stat, mixed_type=data.mixed_type)
        # self.bias_atom_e = data.compute_energy_shift(self.rcond)

    def _compute_input_stat(self, all_stat, mixed_type=False):
        if mixed_type:
            self.descrpt.compute_input_stats(all_stat['coord'],
                                             all_stat['box'],
                                             all_stat['type'],
                                             all_stat['natoms_vec'],
                                             all_stat['default_mesh'],
                                             all_stat,
                                             mixed_type,
                                             all_stat['real_natoms_vec'])
        else:
            self.descrpt.compute_input_stats(all_stat['coord'],
                                             all_stat['box'],
                                             all_stat['type'],
                                             all_stat['natoms_vec'],
                                             all_stat['default_mesh'],
                                             all_stat)

    def _compute_output_stat(self, all_stat, mixed_type=False):
        pass

    def build(self,
              coord_,
              atype_,
              natoms,
              box,
              mesh,
              input_dict,
              frz_model=None,
              ckpt_meta: Optional[str] = None,
              suffix='',
              reuse=None):

        if input_dict is None:
            input_dict = {}
        with tf.variable_scope('model_attr' + suffix, reuse=reuse):
            t_tmap = tf.constant(' '.join(self.type_map),
                                 name='tmap',
                                 dtype=tf.string)
            t_mt = tf.constant(self.model_type,
                               name='model_type',
                               dtype=tf.string)
            t_ver = tf.constant(MODEL_VERSION,
                                name='model_version',
                                dtype=tf.string)

        coord = tf.reshape(coord_, [-1, natoms[1] * 3])
        atype = tf.reshape(atype_, [-1, natoms[1]])
        box = tf.reshape(box, [-1, 9])
        input_dict['nframes'] = tf.shape(coord)[0]

        # type embedding if any
        if self.typeebd is not None:
            type_embedding = self.typeebd.build(
                self.ntypes,
                reuse=reuse,
                suffix=suffix,
            )
            input_dict['type_embedding'] = type_embedding
        input_dict['atype'] = atype_
        input_dict['nnei'] = self.descrpt.sel_all_a[0]

        self.atomic_rep, self.pair_rep = self.build_descrpt(
            coord, atype, natoms, box, mesh, input_dict,
            frz_model=frz_model,
            ckpt_meta=ckpt_meta,
            suffix=suffix,
            reuse=reuse)

        self.atomic_rep_out, self.pair_rep_out, self.coord_update, self.logits, self.norm_x, self.norm_delta_pair_rep = self.backbone.build(self.atomic_rep,
                                                                                                      self.pair_rep,
                                                                                                      natoms,
                                                                                                      input_dict,
                                                                                                      reuse=reuse,
                                                                                                      suffix=suffix)
        self.coord_output = coord + tf.reshape(self.coord_update, [-1, natoms[0] * 3])

        model_dict = {}
        model_dict['coord'] = coord
        model_dict['coord_output'] = self.coord_output
        model_dict['atype'] = atype
        model_dict['logits'] = self.logits
        model_dict['norm_x'] = self.norm_x
        model_dict['norm_delta_pair_rep'] = self.norm_delta_pair_rep

        return model_dict

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       model_type: str = "original_model",
                       suffix: str = "",
                       ) -> None:
        """
        Init the embedding net variables with the given frozen model

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        model_type : str
            the type of the model
        suffix : str
            suffix to name scope
        """
        # self.frz_model will control the self.model to import the descriptor from the given frozen model instead of building from scratch...
        # initialize fitting net with the given compressed frozen model
        if model_type == 'original_model':
            self.descrpt.init_variables(graph, graph_def, suffix=suffix)
            self.backbone.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("original_model", name='model_type', dtype=tf.string)
        elif model_type == 'compressed_model':
            self.backbone.init_variables(graph, graph_def, suffix=suffix)
            tf.constant("compressed_model", name='model_type', dtype=tf.string)
        else:
            raise RuntimeError("Unknown model type %s" % model_type)
        if self.typeebd is not None:
            self.typeebd.init_variables(graph, graph_def, suffix=suffix)
