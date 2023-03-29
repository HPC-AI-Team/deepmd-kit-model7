import json
import os
import platform
import shutil
import subprocess as sp
import sys
import unittest

import dpdata
import numpy as np
from common import (
    j_loader,
    run_dp,
    tests_path,
)

from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    tf,
)
from deepmd.train.run_options import (
    RunOptions,
)
from deepmd.train.trainer import (
    DPTrainer,
)
from deepmd.utils.argcheck import (
    normalize,
)
from deepmd.utils.compat import (
    update_deepmd_input,
)
from deepmd.utils.sess import (
    run_sess,
)
from deepmd.utils.data_system import (
    DeepmdDataSystem,
)

if GLOBAL_NP_FLOAT_PRECISION == np.float32:
    default_places = 4
else:
    default_places = 10

from IPython import embed

def _file_delete(file):
    if os.path.isdir(file):
        os.rmdir(file)
    elif os.path.isfile(file):
        os.remove(file)


def _init_models():
    data_file = str(tests_path / os.path.join("unsup", "data", "one_frame_unimol"))
    jdata = j_loader(str(tests_path / os.path.join("unsup", "input.json")))
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata)
    model_ckpt = DPTrainer(jdata, RunOptions(log_level=20))
    rcut = model_ckpt.model.get_rcut()
    type_map = model_ckpt.model.get_type_map()
    data = DeepmdDataSystem(
        systems=[data_file],
        batch_size=1,
        test_size=1,
        rcut=rcut,
        type_map=type_map,
        trn_all_set=True,
    )
    stop_batch = jdata["training"]["numb_steps"]

    return model_ckpt, data, stop_batch


(
    CKPT_TRAINER,
    VALID_DATA,
    STOP_BATCH,
) = _init_models()


class TestInitFrzModelA(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dp = CKPT_TRAINER
        cls.valid_data = VALID_DATA
        cls.stop_batch = STOP_BATCH

    @classmethod
    def tearDownClass(cls):
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_shift(self):
        valid_batch = self.valid_data.get_batch()
        natoms = valid_batch["natoms_vec"]

        tf.reset_default_graph()
        self.dp.build(self.valid_data, self.stop_batch)
        run_data = [
            self.dp.model.descrpt.atomic_rep,
            self.dp.model.descrpt.pair_rep,

            self.dp.model.backbone.in_atomic_rep,  # 2
            self.dp.model.backbone.in_pair_rep,  # 3
            self.dp.model.backbone.out_layer_q,  # 4
            self.dp.model.backbone.out_layer_k,  # 5
            self.dp.model.backbone.out_layer_v,  # 6
            self.dp.model.backbone.out_layer_pair_rep,  # 7
            self.dp.model.backbone.out_layer_attn_weights_before,  # 8
            self.dp.model.backbone.out_layer_attn_weights_after,  # 9
            self.dp.model.backbone.out_layer_attn_weights_after_softmax,  # 10
            self.dp.model.backbone.out_layer_o,  # 11
            self.dp.model.backbone.evo_layer_atomic_rep,  # 12
            self.dp.model.backbone.evo_layer_pair_rep,  # 13

            # out_layer_pair_rep[0] + out_layer_attn_weights_before[0] == out_layer_attn_weights_after[0]
            # out_layer_pair_rep[1] == out_layer_attn_weights_after[0]
            # evo_layer_pair_rep[0] == out_layer_pair_rep[1]
            # in_pair_rep == out_layer_pair_rep[0]

            self.dp.model.backbone.out_pair_rep,  # 14
            self.dp.model.backbone.out_delta_pair_rep,  # 15
            self.dp.model.backbone.out_delta_pair_rep_fill0,  # 16
            self.dp.model.backbone.delta_pair_rep_before_ln,  # 18
            self.dp.model.backbone.delta_pair_rep_after_ln,  # 17
            self.dp.model.backbone.attn_probs,  # 19
            self.dp.model.backbone.rij,  # 20

            self.dp.model.atomic_rep_out,
            self.dp.model.pair_rep_out,
            self.dp.model.coord_update,
            self.dp.model.logits,
            self.dp.model.norm_x,
            self.dp.model.norm_delta_pair_rep,
        ]

        self.dp._init_session()
        noise_dict_0 = {'noise_idx': np.array([0], dtype=np.int),
                      'token_idx': np.array([], dtype=np.int),
                      'coord_noise': np.array([0.1, 0.5, 0.2])
                      }
        noise_dict_1 = {'noise_idx': np.array([1], dtype=np.int),
                      'token_idx': np.array([], dtype=np.int),
                      'coord_noise': np.array([0.1, 0.5, 0.2])
                      }

        feed_dict_ckpt_0 = self.dp.get_feed_dict(valid_batch, is_training=False, noise_dict=noise_dict_0)
        feed_dict_ckpt_1 = self.dp.get_feed_dict(valid_batch, is_training=False, noise_dict=noise_dict_1)
        out_0 = run_sess(self.dp.sess, run_data, feed_dict=feed_dict_ckpt_0)
        out_1 = run_sess(self.dp.sess, run_data, feed_dict=feed_dict_ckpt_1)
        embed()

        # # check values
        # np.testing.assert_almost_equal(
        #     ckpt_rmse_ckpt["rmse_e"], ckpt_rmse_frz["rmse_e"], default_places
        # )