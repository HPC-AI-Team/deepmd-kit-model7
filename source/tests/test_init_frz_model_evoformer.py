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
    data_file = str(tests_path / os.path.join("unsup", "data", "one_frame"))
    jdata = j_loader(str(tests_path / os.path.join("unsup", "input.json")))
    frozen_model = str(tests_path / "init_frz_evoformer.pb")
    ckpt = str(tests_path / "init_frz_evoformer.ckpt")
    run_opt_ckpt = RunOptions(init_model=ckpt, log_level=20)
    run_opt_frz = RunOptions(init_frz_model=frozen_model, log_level=20)
    INPUT = str(tests_path / "input.json")
    jdata["training"]["training_data"]["systems"] = data_file
    jdata["training"]["validation_data"]["systems"] = data_file
    jdata["training"]["save_ckpt"] = ckpt
    with open(INPUT, "w") as fp:
        json.dump(jdata, fp, indent=4)
    ret = run_dp("dp train " + INPUT)
    np.testing.assert_equal(ret, 0, "DP train failed!")
    ret = run_dp("dp freeze -c " + str(tests_path) + " -o " + frozen_model)
    np.testing.assert_equal(ret, 0, "DP freeze failed!")

    jdata = update_deepmd_input(jdata, warning=True, dump="input_v2_compat.json")
    jdata = normalize(jdata)
    model_ckpt = DPTrainer(jdata, run_opt=run_opt_ckpt)
    model_frz = DPTrainer(jdata, run_opt=run_opt_frz)
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

    return INPUT, ckpt, frozen_model, model_ckpt, model_frz, data, stop_batch


(
    INPUT,
    CKPT,
    FROZEN_MODEL,
    CKPT_TRAINER,
    FRZ_TRAINER,
    VALID_DATA,
    STOP_BATCH,
) = _init_models()


class TestInitFrzModelEvoformer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dp_ckpt = CKPT_TRAINER
        cls.dp_frz = FRZ_TRAINER
        cls.valid_data = VALID_DATA
        cls.stop_batch = STOP_BATCH

    @classmethod
    def tearDownClass(cls):
        _file_delete(INPUT)
        _file_delete(FROZEN_MODEL)
        _file_delete("out.json")
        _file_delete(str(tests_path / "checkpoint"))
        _file_delete(CKPT + ".meta")
        _file_delete(CKPT + ".index")
        _file_delete(CKPT + ".data-00000-of-00001")
        _file_delete(CKPT + "-0.meta")
        _file_delete(CKPT + "-0.index")
        _file_delete(CKPT + "-0.data-00000-of-00001")
        _file_delete(CKPT + "-1.meta")
        _file_delete(CKPT + "-1.index")
        _file_delete(CKPT + "-1.data-00000-of-00001")
        _file_delete("input_v2_compat.json")
        _file_delete("lcurve.out")

    def test_single_frame(self):
        valid_batch = self.valid_data.get_batch()
        natoms = valid_batch["natoms_vec"]
        tf.reset_default_graph()
        self.dp_ckpt.build(self.valid_data, self.stop_batch)
        self.dp_ckpt._init_session()
        noise_dict_0 = {'noise_idx': np.array([0], dtype=np.int),
                        'token_idx': np.array([], dtype=np.int),
                        'coord_noise': np.array([0.1, 0.5, 0.2])
                        }
        feed_dict_ckpt = self.dp_ckpt.get_feed_dict(valid_batch, is_training=False, noise_dict=noise_dict_0)
        ckpt_rmse_ckpt = self.dp_ckpt.loss.eval(
            self.dp_ckpt.sess, feed_dict_ckpt, natoms
        )
        tf.reset_default_graph()

        self.dp_frz.build(self.valid_data, self.stop_batch)
        self.dp_frz._init_session()
        feed_dict_frz = self.dp_frz.get_feed_dict(valid_batch, is_training=False, noise_dict=noise_dict_0)
        ckpt_rmse_frz = self.dp_frz.loss.eval(self.dp_frz.sess, feed_dict_frz, natoms)
        tf.reset_default_graph()

        # check values
        np.testing.assert_almost_equal(
            ckpt_rmse_ckpt["stru_loss"], ckpt_rmse_frz["stru_loss"], default_places
        )
        np.testing.assert_almost_equal(
            ckpt_rmse_ckpt["token_loss"], ckpt_rmse_frz["token_loss"], default_places
        )
        np.testing.assert_almost_equal(
            ckpt_rmse_ckpt["norm_loss"], ckpt_rmse_frz["norm_loss"], default_places
        )
