#!/usr/bin/env python3
import glob
import logging
import os
import platform
import shutil
import time
import math

import google.protobuf.message
import numpy as np
from packaging.version import (
    Version,
)
from tensorflow.python.client import (
    timeline,
)

# load grad of force module
import deepmd.op
from deepmd.common import (
    data_requirement,
    get_precision,
    j_must_have,
)
from deepmd.descriptor.descriptor import (
    Descriptor,
)
from deepmd.env import (
    GLOBAL_NP_FLOAT_PRECISION,
    GLOBAL_ENER_FLOAT_PRECISION,
    GLOBAL_TF_FLOAT_PRECISION,
    TF_VERSION,
    get_tf_session_config,
    default_tf_session_config,
    op_module,
    tf,
    tfv2,
)
from deepmd.fit import (
    DipoleFittingSeA,
    EnerFitting,
    PolarFittingSeA,
    AttrFitting,
)
from deepmd.backbone import (
    EvoformerBackbone,
)
from deepmd.loss import (
    EnerDipoleLoss,
    EnerStdLoss,
    TensorLoss,
    StruReconLoss,
    AttrStdLoss,
)
from deepmd.model import (
    DipoleModel,
    EnerModel,
    GlobalPolarModel,
    MultiModel,
    PolarModel,
    WFCModel,
    EvoformerModel,
)
from deepmd.utils import random as dp_random
from deepmd.utils.argcheck import (
    type_embedding_args,
)
from deepmd.utils.errors import (
    GraphTooLargeError,
    GraphWithoutTensorError,
)
from deepmd.utils.graph import (
    get_tensor_by_name_from_graph,
    load_graph_def,
)
from deepmd.utils.learning_rate import (
    LearningRateExp,
)
from deepmd.utils.neighbor_stat import (
    NeighborStat,
)
from deepmd.utils.sess import (
    run_sess,
)
from deepmd.utils.type_embed import (
    TypeEmbedNet,
)

log = logging.getLogger(__name__)

# nvnmd
from deepmd.nvnmd.utils.config import (
    nvnmd_cfg,
)

# import wandb as wb
# from IPython import embed


def _is_subdir(path, directory):
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    if path == directory:
        return False
    relative = os.path.relpath(path, directory) + os.sep
    return not relative.startswith(os.pardir + os.sep)


class DPTrainer(object):
    def __init__(self, jdata, run_opt, is_compress=False):
        self.run_opt = run_opt
        self._init_param(jdata)
        self.is_compress = is_compress

    def _init_param(self, jdata):
        # model config
        model_param = j_must_have(jdata, "model")
        self.use_backbone = "backbone" in model_param
        self.multi_task_mode = "fitting_net_dict" in model_param
        descrpt_param = j_must_have(model_param, "descriptor")
        fitting_param = None
        if not self.use_backbone:
            fitting_param = (
                j_must_have(model_param, "fitting_net")
                if not self.multi_task_mode
                else j_must_have(model_param, "fitting_net_dict")
            )
        else:
            backbone_param = j_must_have(model_param, "backbone")
            self.backbone_param = backbone_param
            if "fitting_net" in model_param or "fitting_net_dict" in model_param:
                fitting_param = (
                    j_must_have(model_param, "fitting_net")
                    if not self.multi_task_mode
                    else j_must_have(model_param, "fitting_net_dict")
                )
        typeebd_param = model_param.get("type_embedding", None)
        self.model_param = model_param
        self.descrpt_param = descrpt_param

        # nvnmd
        self.nvnmd_param = jdata.get("nvnmd", {})
        nvnmd_cfg.init_from_jdata(self.nvnmd_param)
        if nvnmd_cfg.enable:
            nvnmd_cfg.init_from_deepmd_input(model_param)
            nvnmd_cfg.disp_message()
            nvnmd_cfg.save()

        # descriptor
        try:
            descrpt_type = descrpt_param["type"]
            self.descrpt_type = descrpt_type
        except KeyError:
            raise KeyError("the type of descriptor should be set by `type`")

        explicit_ntypes_descrpt = ["se_atten", "gaussian"]
        hybrid_with_tebd = False
        if descrpt_param["type"] in explicit_ntypes_descrpt:
            descrpt_param["ntypes"] = len(model_param["type_map"])
        elif descrpt_param["type"] == "hybrid":
            for descrpt_item in descrpt_param["list"]:
                if descrpt_item["type"] in explicit_ntypes_descrpt:
                    descrpt_item["ntypes"] = len(model_param["type_map"])
                    hybrid_with_tebd = True
        if self.use_backbone:
            assert self.descrpt_type in ["se_atten", "gaussian"], "backbones only support se_atten, gaussian descriptor!"
            descrpt_param["return_G"] = True
            if self.descrpt_type in ["gaussian"]:
                assert typeebd_param is not None
                descrpt_param["tebd_size"] = typeebd_param["neuron"][-1]
        elif self.multi_task_mode:
            descrpt_param["multi_task"] = True
        self.descrpt = Descriptor(**descrpt_param)

        # fitting net
        def fitting_net_init(fitting_type_, descrpt_type_, params):
            if fitting_type_ == "ener":
                return EnerFitting(**params)
            elif fitting_type_ == "dipole":
                return DipoleFittingSeA(**params)
            elif fitting_type_ == "polar":
                return PolarFittingSeA(**params)
            elif fitting_type_ == "attr":
                return AttrFitting(**params)
            # elif fitting_type_ == 'global_polar':
            #     if descrpt_type_ == 'se_e2_a':
            #         return GlobalPolarFittingSeA(**params)
            #     else:
            #         raise RuntimeError('fitting global_polar only supports descrptors: loc_frame and se_e2_a')
            else:
                raise RuntimeError("unknown fitting type " + fitting_type_)

        def backbone_init(backbone_type_, params):
            if backbone_type_ == "evoformer":
                return EvoformerBackbone(**params)
            else:
                raise RuntimeError("unknown backbone type " + backbone_type_)

        if not self.use_backbone:
            if not self.multi_task_mode:
                fitting_type = fitting_param.get("type", "ener")
                self.fitting_type = fitting_type
                fitting_param.pop("type", None)
                fitting_param["descrpt"] = self.descrpt
                self.fitting = fitting_net_init(fitting_type, descrpt_type, fitting_param)
            else:
                self.fitting_dict = {}
                self.fitting_type_dict = {}
                self.nfitting = len(fitting_param)
                for item in fitting_param:
                    item_fitting_param = fitting_param[item]
                    item_fitting_type = item_fitting_param.get("type", "ener")
                    self.fitting_type_dict[item] = item_fitting_type
                    item_fitting_param.pop("type", None)
                    item_fitting_param["descrpt"] = self.descrpt
                    self.fitting_dict[item] = fitting_net_init(
                        item_fitting_type, descrpt_type, item_fitting_param
                    )
        else:
            # backbone
            backbone_type = backbone_param.pop("type", "evoformer")
            backbone_param["descrpt"] = self.descrpt
            self.backbone = backbone_init(backbone_type, backbone_param)
            if fitting_param is not None:
                if not self.multi_task_mode:
                    fitting_type = fitting_param.get("type", "ener")
                    self.fitting_type = fitting_type
                    fitting_param.pop("type", None)
                    fitting_param["descrpt"] = self.descrpt
                    self.fitting = fitting_net_init(fitting_type, descrpt_type, fitting_param)
            else:
                self.fitting = None
                self.fitting_type = None
        # type embedding
        padding = False
        if descrpt_type == "se_atten" or hybrid_with_tebd:
            padding = True
        if typeebd_param is not None:
            self.typeebd = TypeEmbedNet(
                neuron=typeebd_param["neuron"],
                resnet_dt=typeebd_param["resnet_dt"],
                activation_function=typeebd_param["activation_function"],
                precision=typeebd_param["precision"],
                trainable=typeebd_param["trainable"],
                seed=typeebd_param["seed"],
                padding=padding,
            )
        elif descrpt_type == "se_atten" or hybrid_with_tebd:
            default_args = type_embedding_args()
            default_args_dict = {i.name: i.default for i in default_args}
            self.typeebd = TypeEmbedNet(
                neuron=default_args_dict["neuron"],
                resnet_dt=default_args_dict["resnet_dt"],
                activation_function=None,
                precision=default_args_dict["precision"],
                trainable=default_args_dict["trainable"],
                seed=default_args_dict["seed"],
                padding=padding,
            )
        else:
            self.typeebd = None

        # init model
        # infer model type by fitting_type
        if not self.use_backbone:
            if not self.multi_task_mode:
                if self.fitting_type == "ener":
                    self.model = EnerModel(
                        self.descrpt,
                        self.fitting,
                        self.typeebd,
                        model_param.get("type_map"),
                        model_param.get("data_stat_nbatch", 10),
                        model_param.get("data_stat_protect", 1e-2),
                        model_param.get("use_srtab"),
                        model_param.get("smin_alpha"),
                        model_param.get("sw_rmin"),
                        model_param.get("sw_rmax")
                    )
                # elif fitting_type == "wfc":
                #     self.model = WFCModel(model_param, self.descrpt, self.fitting)
                elif self.fitting_type == "dipole":
                    self.model = DipoleModel(
                        self.descrpt,
                        self.fitting,
                        self.typeebd,
                        model_param.get("type_map"),
                        model_param.get("data_stat_nbatch", 10),
                        model_param.get("data_stat_protect", 1e-2)
                    )
                elif self.fitting_type == "polar":
                    self.model = PolarModel(
                        self.descrpt,
                        self.fitting,
                        self.typeebd,
                        model_param.get("type_map"),
                        model_param.get("data_stat_nbatch", 10),
                        model_param.get("data_stat_protect", 1e-2)
                    )
                # elif self.fitting_type == "global_polar":
                #     self.model = GlobalPolarModel(
                #         self.descrpt,
                #         self.fitting,
                #         model_param.get("type_map"),
                #         model_param.get("data_stat_nbatch", 10),
                #         model_param.get("data_stat_protect", 1e-2)
                #     )
                else:
                    raise RuntimeError("get unknown fitting type when building model")
            else:  # multi-task mode
                self.model = MultiModel(
                    self.descrpt,
                    self.fitting_dict,
                    self.fitting_type_dict,
                    self.typeebd,
                    model_param.get("type_map"),
                    model_param.get("data_stat_nbatch", 10),
                    model_param.get("data_stat_protect", 1e-2),
                    model_param.get("use_srtab"),
                    model_param.get("smin_alpha"),
                    model_param.get("sw_rmin"),
                    model_param.get("sw_rmax")
                )
        else:
            self.model = EvoformerModel(
                self.descrpt,
                self.backbone,
                self.typeebd,
                self.fitting,
                self.fitting_type,
                model_param.get("type_map"),
                model_param.get("data_stat_nbatch", 10),
                model_param.get("data_stat_protect", 1e-2),
                model_param.get("noise", 1.00),
                model_param.get("noise_type", "uniform"),
                model_param.get("ener_style", "atomic_out"),
            )

        # learning rate
        lr_param = j_must_have(jdata, "learning_rate")
        scale_by_worker = lr_param.get("scale_by_worker", "linear")
        if scale_by_worker == "linear":
            self.scale_lr_coef = float(self.run_opt.world_size)
        elif scale_by_worker == "sqrt":
            self.scale_lr_coef = np.sqrt(self.run_opt.world_size).real
        else:
            self.scale_lr_coef = 1.0
        lr_type = lr_param.get("type", "exp")
        if lr_type == "exp":
            self.lr = LearningRateExp(
                lr_param["start_lr"], lr_param["stop_lr"], lr_param["decay_steps"], warm_up_num = lr_param["warm_up"]
            )
        else:
            raise RuntimeError("unknown learning_rate type " + lr_type)

        # loss
        # infer loss type by fitting_type
        def loss_init(_loss_param, _fitting_type, _fitting, _lr, _backbone=None, ntypes=None):
            if _backbone is None:
                _loss_type = _loss_param.get("type", "ener")
                if _fitting_type == "ener":
                    _loss_param.pop("type", None)
                    _loss_param["starter_learning_rate"] = _lr.start_lr()
                    if _loss_type == "ener":
                        loss = EnerStdLoss(**_loss_param)
                    elif _loss_type == "ener_dipole":
                        loss = EnerDipoleLoss(**_loss_param)
                    else:
                        raise RuntimeError("unknown loss type")
                elif _fitting_type == "wfc":
                    loss = TensorLoss(
                        _loss_param,
                        model=_fitting,
                        tensor_name="wfc",
                        tensor_size=_fitting.get_out_size(),
                        label_name="wfc",
                    )
                elif _fitting_type == "dipole":
                    loss = TensorLoss(
                        _loss_param,
                        model=_fitting,
                        tensor_name="dipole",
                        tensor_size=3,
                        label_name="dipole",
                    )
                elif _fitting_type == "polar":
                    loss = TensorLoss(
                        _loss_param,
                        model=_fitting,
                        tensor_name="polar",
                        tensor_size=9,
                        label_name="polarizability",
                    )
                elif _fitting_type == "global_polar":
                    loss = TensorLoss(
                        _loss_param,
                        model=_fitting,
                        tensor_name="global_polar",
                        tensor_size=9,
                        atomic=False,
                        label_name="polarizability",
                    )
                else:
                    raise RuntimeError("get unknown fitting type when building loss function")
            else:
                _loss_type = _loss_param.get("type", "stru_recon")
                _loss_param.pop("type", None)
                if _loss_type == "stru":
                    _loss_param["ntypes"] = ntypes
                    loss = StruReconLoss(_loss_param)
                elif _loss_type == "ener":
                    _loss_param["starter_learning_rate"] = _lr.start_lr()
                    loss = EnerStdLoss(**_loss_param)
                elif _loss_type == "attr":
                    loss = AttrStdLoss(**_loss_param)
                else:
                    raise RuntimeError("get unknown backbone type when building loss function")
            return loss

        if not self.use_backbone:
            if not self.multi_task_mode:
                loss_param = jdata.get("loss", {})
                self.loss_type = loss_param.get("type", "ener")
                self.loss = loss_init(loss_param, self.fitting_type, self.fitting, self.lr)
            else:
                self.loss_dict = {}
                loss_param_dict = jdata.get("loss_dict", {})
                for fitting_key in self.fitting_type_dict:
                    loss_param = loss_param_dict.get(fitting_key, {})
                    self.loss_dict[fitting_key] = loss_init(
                        loss_param,
                        self.fitting_type_dict[fitting_key],
                        self.fitting_dict[fitting_key],
                        self.lr,
                    )
        else:
            loss_param = jdata.get("loss", {})
            self.loss_type = loss_param.get("type", "stru")
            self.loss = loss_init(loss_param, "", "", self.lr, "stru_recon", self.model.ntypes)

        # training
        tr_data = jdata["training"]
        self.fitting_weight = tr_data.get("fitting_weight", None)
        if self.multi_task_mode:
            self.fitting_key_list = []
            self.fitting_prob = []
            for fitting_key in self.fitting_type_dict:
                self.fitting_key_list.append(fitting_key)
                # multi-task mode must have self.fitting_weight
                self.fitting_prob.append(self.fitting_weight[fitting_key])
        self.disp_file = tr_data.get("disp_file", "lcurve.out")
        self.disp_freq = tr_data.get("disp_freq", 1000)
        self.save_freq = tr_data.get("save_freq", 1000)
        self.save_ckpt = tr_data.get("save_ckpt", "model.ckpt")
        self.display_in_training = tr_data.get("disp_training", True)
        self.timing_in_training = tr_data.get("time_training", True)
        self.mask_mode = ''
        if self.use_backbone:
            self.noise = float(tr_data.get("noise", 1.00))
            self.noise_type = tr_data.get("noise_type", "uniform")
            self.coord_noise_num = tr_data.get("coord_noise_num", 10)
            self.masked_token_num = tr_data.get("masked_token_num", 10)
            self.mask_mode = tr_data.get("coord_mask_mode", '')
            self.same_mask = tr_data.get("same_mask", False)
            self._min_dist_sub_graph()
        self.profiling = self.run_opt.is_chief and tr_data.get("profiling", False)
        self.profiling_file = tr_data.get("profiling_file", "timeline.json")
        self.enable_profiler = tr_data.get("enable_profiler", False)
        self.tensorboard = self.run_opt.is_chief and tr_data.get("tensorboard", False)
        self.tensorboard_log_dir = tr_data.get("tensorboard_log_dir", "log")
        self.tensorboard_freq = tr_data.get("tensorboard_freq", 1)
        self.mixed_prec = tr_data.get("mixed_precision", None)
        # name_path = os.path.abspath('.').split('/')
        # wb.init(project="DPA", entity="dp_model_engineering", config=model_param,
        #         name=name_path[-2] + '/' + name_path[-1], settings=wb.Settings(start_method="fork"))
        if self.mixed_prec is not None:
            if (
                self.mixed_prec["compute_prec"] not in ("float16", "bfloat16")
                or self.mixed_prec["output_prec"] != "float32"
            ):
                raise RuntimeError(
                    "Unsupported mixed precision option [output_prec, compute_prec]: [%s, %s], "
                    " Supported: [float32, float16/bfloat16], Please set mixed precision option correctly!"
                    % (self.mixed_prec["output_prec"], self.mixed_prec["compute_prec"])
                )
        # self.sys_probs = tr_data['sys_probs']
        # self.auto_prob_style = tr_data['auto_prob']
        self.useBN = False
        if not self.use_backbone:
            if not self.multi_task_mode:
                if self.fitting_type == "ener" and self.fitting.get_numb_fparam() > 0:
                    self.numb_fparam = self.fitting.get_numb_fparam()
                else:
                    self.numb_fparam = 0

                if tr_data.get("validation_data", None) is not None:
                    self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
                else:
                    self.valid_numb_batch = 1
            else:
                self.numb_fparam_dict = {}
                self.valid_numb_batch_dict = {}
                for fitting_key in self.fitting_type_dict:
                    if (
                        self.fitting_type_dict[fitting_key] == "ener"
                        and self.fitting_dict[fitting_key].get_numb_fparam() > 0
                    ):
                        self.numb_fparam_dict[fitting_key] = self.fitting_dict[
                            fitting_key
                        ].get_numb_fparam()
                    else:
                        self.numb_fparam_dict[fitting_key] = 0
                data_dict = tr_data.get("data_dict", None)
                for systems in data_dict:
                    if data_dict[systems].get("validation_data", None) is not None:
                        self.valid_numb_batch_dict[systems] = data_dict[systems][
                            "validation_data"
                        ].get("numb_btch", 1)
                    else:
                        self.valid_numb_batch_dict[systems] = 1
        else:
            if tr_data.get("validation_data", None) is not None:
                self.valid_numb_batch = tr_data["validation_data"].get("numb_btch", 1)
            else:
                self.valid_numb_batch = 1

        # if init the graph with the frozen model
        self.frz_model = None
        self.ckpt_meta = None
        self.model_type = None

    def build(self, data=None, stop_batch=0, origin_type_map=None, suffix=""):
        self.ntypes = self.model.get_ntypes()
        self.stop_batch = stop_batch

        if not self.multi_task_mode:
            if not self.use_backbone and not self.is_compress and data.mixed_type:
                assert self.descrpt_type in [
                    "se_atten"
                ], "Data in mixed_type format must use attention descriptor!"
                assert self.fitting_type in [
                    "ener"
                ], "Data in mixed_type format must use ener fitting!"

            if not self.use_backbone and self.numb_fparam > 0:
                log.info("training with %d frame parameter(s)" % self.numb_fparam)
            else:
                log.info("training without frame parameter")
        else:
            assert (
                not self.is_compress
            ), "You should not reach here, multi-task input could not be compressed! "
            self.valid_fitting_key = []
            for fitting_key in data:
                self.valid_fitting_key.append(fitting_key)
                if data[fitting_key].mixed_type:
                    assert self.descrpt_type in ["se_atten"], (
                        "Data for fitting net {} in mixed_type format "
                        "must use attention descriptor!".format(fitting_key)
                    )
                    assert self.fitting_type_dict[fitting_key] in [
                        "ener"
                    ], "Data for fitting net {} in mixed_type format must use ener fitting!".format(
                        fitting_key
                    )

                if self.numb_fparam_dict[fitting_key] > 0:
                    log.info(
                        "fitting net %s training with %d frame parameter(s)"
                        % (fitting_key, self.numb_fparam_dict[fitting_key])
                    )
                else:
                    log.info(
                        "fitting net %s training without frame parameter" % fitting_key
                    )

        if not self.is_compress:
            # Usually, the type number of the model should be equal to that of the data
            # However, nt_model > nt_data should be allowed, since users may only want to
            # train using a dataset that only have some of elements
            if not self.multi_task_mode:
                single_data = data
            else:
                single_data = data[list(data.keys())[0]]
            if self.ntypes < single_data.get_ntypes():
                raise ValueError(
                    "The number of types of the training data is %d, but that of the "
                    "model is only %d. The latter must be no less than the former. "
                    "You may need to reset one or both of them. Usually, the former "
                    "is given by `model/type_map` in the training parameter (if set) "
                    "or the maximum number in the training data. The latter is given "
                    "by `model/descriptor/sel` in the training parameter."
                    % (single_data.get_ntypes(), self.ntypes)
                )
            self.type_map = single_data.get_type_map()
            if not self.multi_task_mode:
                self.batch_size = data.get_batch_size()
            else:
                self.batch_size = {}
                for fitting_key in data:
                    self.batch_size[fitting_key] = data[fitting_key].get_batch_size()
            if self.run_opt.init_mode not in (
                "init_from_model",
                "restart",
                "init_from_frz_model",
                "finetune",
            ):
                # self.saver.restore (in self._init_session) will restore avg and std variables, so data_stat is useless
                # init_from_frz_model will restore data_stat variables in `init_variables` method
                log.info("data stating... (this step may take long time)")
                self.model.data_stat(data)

            # config the init_frz_model command
            if self.run_opt.init_mode == "init_from_frz_model":
                self._init_from_frz_model()
            elif self.run_opt.init_mode == "init_model":
                self._init_from_ckpt(self.run_opt.init_model)
            elif self.run_opt.init_mode == "restart":
                self._init_from_ckpt(self.run_opt.restart)
            elif self.run_opt.init_mode == "finetune":
                self._init_from_pretrained_model(
                    data=data, origin_type_map=origin_type_map
                )

            # neighbor_stat is moved to train.py as duplicated
            # TODO: this is a simple fix but we should have a clear
            #       architecture to call neighbor stat
        else:
            graph, graph_def = load_graph_def(
                self.model_param["compress"]["model_file"]
            )
            self.descrpt.enable_compression(
                self.model_param["compress"]["min_nbor_dist"],
                graph,
                graph_def,
                self.model_param["compress"]["table_config"][0],
                self.model_param["compress"]["table_config"][1],
                self.model_param["compress"]["table_config"][2],
                self.model_param["compress"]["table_config"][3],
            )
            # for fparam or aparam settings in 'ener' type fitting net
            self.fitting.init_variables(graph, graph_def)

        if self.is_compress or self.model_type == "compressed_model":
            tf.constant("compressed_model", name="model_type", dtype=tf.string)
        else:
            tf.constant("original_model", name="model_type", dtype=tf.string)

        if self.mixed_prec is not None:
            self.descrpt.enable_mixed_precision(self.mixed_prec)
            if not self.multi_task_mode:
                self.fitting.enable_mixed_precision(self.mixed_prec)
            else:
                for fitting_key in self.fitting_dict:
                    self.fitting_dict[fitting_key].enable_mixed_precision(
                        self.mixed_prec
                    )

        self._build_lr()
        self._build_network(data, suffix)
        self._build_training()

    def _build_lr(self):
        self._extra_train_ops = []
        self.global_step = tf.train.get_or_create_global_step()
        self.learning_rate = self.lr.build(self.global_step, self.stop_batch)
        log.info("built lr")

    def _build_network(self, data, suffix=""):
        self.place_holders = {}
        if self.is_compress:
            for kk in ["coord", "box"]:
                self.place_holders[kk] = tf.placeholder(
                    GLOBAL_TF_FLOAT_PRECISION, [None], "t_" + kk
                )
            self._get_place_horders(data_requirement)
        else:
            if not self.multi_task_mode:
                self._get_place_horders(data.get_data_dict())
            else:
                self._get_place_horders(data[list(data.keys())[0]].get_data_dict())

        self.place_holders["type"] = tf.placeholder(tf.int32, [None], name="t_type")
        self.place_holders["natoms_vec"] = tf.placeholder(
            tf.int32, [self.ntypes + 2], name="t_natoms"
        )
        self.place_holders["default_mesh"] = tf.placeholder(
            tf.int32, [None], name="t_mesh"
        )
        self.place_holders["is_training"] = tf.placeholder(tf.bool)
        if self.use_backbone:
            self.place_holders["clean_type"] = tf.placeholder(tf.int32, [None], name="t_clean_type")
            self.place_holders["clean_coord"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_clean_coord")
            self.place_holders["noise_mask"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_noise_mask")
            self.place_holders["token_mask"] = tf.placeholder(GLOBAL_TF_FLOAT_PRECISION, [None], name="t_token_mask")
        self.model_pred = self.model.build(
            self.place_holders["coord"],
            self.place_holders["type"],
            self.place_holders["natoms_vec"],
            self.place_holders["box"],
            self.place_holders["default_mesh"],
            self.place_holders,
            frz_model=self.frz_model,
            ckpt_meta=self.ckpt_meta,
            suffix=suffix,
            reuse=False,
        )

        if not self.multi_task_mode:
            self.l2_l, self.l2_more = self.loss.build(
                self.learning_rate,
                self.place_holders["natoms_vec"],
                self.model_pred,
                self.place_holders,
                suffix="test",
            )

            if self.mixed_prec is not None:
                self.l2_l = tf.cast(
                    self.l2_l, get_precision(self.mixed_prec["output_prec"])
                )
        else:
            self.l2_l, self.l2_more = {}, {}
            for fitting_key in self.fitting_type_dict:
                self.l2_l[fitting_key], self.l2_more[fitting_key] = self.loss_dict[
                    fitting_key
                ].build(
                    self.learning_rate,
                    self.place_holders["natoms_vec"],
                    self.model_pred[fitting_key],
                    self.place_holders,
                    suffix=fitting_key,
                )
                if self.mixed_prec is not None:
                    self.l2_l[fitting_key] = tf.cast(
                        self.l2_l[fitting_key],
                        get_precision(self.mixed_prec["output_prec"]),
                    )

        log.info("built network")

    def _build_training(self):
        trainable_variables = tf.trainable_variables()
        if self.run_opt.is_distrib:
            if self.scale_lr_coef > 1.0:
                log.info("Scale learning rate by coef: %f", self.scale_lr_coef)
                optimizer = tf.train.AdamOptimizer(
                    self.learning_rate * self.scale_lr_coef
                )
            else:
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = self.run_opt._HVD.DistributedOptimizer(optimizer)
        else:
            optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        if self.mixed_prec is not None:
            _TF_VERSION = Version(TF_VERSION)
            # check the TF_VERSION, when TF < 1.12, mixed precision is not allowed
            if _TF_VERSION < Version("1.14.0"):
                raise RuntimeError(
                    "TensorFlow version %s is not compatible with the mixed precision setting. Please consider upgrading your TF version!"
                    % TF_VERSION
                )
            elif _TF_VERSION < Version("2.4.0"):
                optimizer = tf.train.experimental.enable_mixed_precision_graph_rewrite(
                    optimizer
                )
            else:
                optimizer = tf.mixed_precision.enable_mixed_precision_graph_rewrite(
                    optimizer
                )
        if not self.multi_task_mode:
            apply_op = optimizer.minimize(
                loss=self.l2_l,
                global_step=self.global_step,
                var_list=trainable_variables,
                name="train_step",
            )
            aa = tf.debugging.assert_all_finite(
                t=self.l2_l, msg='NAN or infinite in loss'
            )
            train_ops = [apply_op, aa] + self._extra_train_ops
            self.train_op = tf.group(*train_ops)
        else:
            self.train_op = {}
            for fitting_key in self.fitting_type_dict:
                apply_op = optimizer.minimize(
                    loss=self.l2_l[fitting_key],
                    global_step=self.global_step,
                    var_list=trainable_variables,
                    name="train_step_{}".format(fitting_key),
                )
                train_ops = [apply_op] + self._extra_train_ops
                self.train_op[fitting_key] = tf.group(*train_ops)
        log.info("built training")

    def _init_session(self):
        config = get_tf_session_config()
        device, idx = self.run_opt.my_device.split(":", 1)
        if device == "gpu":
            config.gpu_options.visible_device_list = idx
        self.sess = tf.Session(config=config)

        # Initializes or restore global variables
        init_op = tf.global_variables_initializer()
        if self.run_opt.is_chief:
            self.saver = tf.train.Saver(save_relative_paths=True)
            if self.run_opt.init_mode == "init_from_scratch":
                log.info("initialize model from scratch")
                run_sess(self.sess, init_op)
                if not self.is_compress:
                    fp = open(self.disp_file, "w")
                    fp.close()
            elif self.run_opt.init_mode == "init_from_model":
                log.info("initialize from model %s" % self.run_opt.init_model)
                run_sess(self.sess, init_op)
                self.saver.restore(self.sess, self.run_opt.init_model)
                run_sess(self.sess, self.global_step.assign(0))
                fp = open(self.disp_file, "w")
                fp.close()
            elif self.run_opt.init_mode == "restart":
                log.info("restart from model %s" % self.run_opt.restart)
                run_sess(self.sess, init_op)
                self.saver.restore(self.sess, self.run_opt.restart)
            elif self.run_opt.init_mode == "init_from_frz_model":
                log.info("initialize training from the frozen model")
                run_sess(self.sess, init_op)
                fp = open(self.disp_file, "w")
                fp.close()
            elif self.run_opt.init_mode == "finetune":
                log.info("initialize training from the frozen pretrained model")
                run_sess(self.sess, init_op)
                fp = open(self.disp_file, "w")
                fp.close()
            else:
                raise RuntimeError("unknown init mode")
        else:
            run_sess(self.sess, init_op)
            self.saver = None

        # Ensure variable consistency among tasks when training starts
        if self.run_opt.is_distrib:
            bcast_op = self.run_opt._HVD.broadcast_global_variables(0)
            if self.run_opt.is_chief:
                log.info("broadcast global variables to other tasks")
            else:
                log.info("receive global variables from task#0")
            run_sess(self.sess, bcast_op)

    def train(self, train_data=None, valid_data=None):

        # if valid_data is None:  # no validation set specified.
        #     valid_data = train_data  # using training set as validation set.

        stop_batch = self.stop_batch
        self._init_session()

        # Before data shard is enabled, only cheif do evaluation and record it
        # self.print_head()
        fp = None
        if self.run_opt.is_chief:
            fp = open(self.disp_file, "a")

        cur_batch = run_sess(self.sess, self.global_step)
        is_first_step = True
        self.cur_batch = cur_batch
        log.info(
            "start training at lr %.2e (== %.2e), decay_step %d, decay_rate %f, final lr will be %.2e"
            % (
                run_sess(self.sess, self.learning_rate),
                self.lr.value(cur_batch),
                self.lr.decay_steps_,
                self.lr.decay_rate_,
                self.lr.value(stop_batch),
            )
        )

        prf_options = None
        prf_run_metadata = None
        if self.profiling:
            prf_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            prf_run_metadata = tf.RunMetadata()

        # set tensorboard execution environment
        if self.tensorboard:
            summary_merged_op = tf.summary.merge_all()
            # Remove TB old logging directory from previous run
            try:
                shutil.rmtree(self.tensorboard_log_dir)
            except FileNotFoundError:
                pass  # directory does not exist, this is OK
            except Exception as e:
                # general error when removing directory, warn user
                log.exception(
                    f"Could not remove old tensorboard logging directory: "
                    f"{self.tensorboard_log_dir}. Error: {e}"
                )
            else:
                log.debug("Removing old tensorboard log directory.")
            tb_train_writer = tf.summary.FileWriter(
                self.tensorboard_log_dir + "/train", self.sess.graph
            )
            tb_valid_writer = tf.summary.FileWriter(self.tensorboard_log_dir + "/test")
        else:
            tb_train_writer = None
            tb_valid_writer = None
        if self.enable_profiler:
            # https://www.tensorflow.org/guide/profiler
            tfv2.profiler.experimental.start(self.tensorboard_log_dir)

        train_time = 0
        total_train_time = 0.0

        while cur_batch < stop_batch:

            # first round validation:
            if not self.multi_task_mode:
                train_batch = train_data.get_batch()
                batch_train_op = self.train_op
            else:
                fitting_idx = dp_random.choice(
                    np.arange(self.nfitting), p=np.array(self.fitting_prob)
                )
                fitting_key = self.fitting_key_list[fitting_idx]
                train_batch = train_data[fitting_key].get_batch()
                batch_train_op = self.train_op[fitting_key]
            if self.display_in_training and is_first_step:
                if self.run_opt.is_chief:
                    if not self.multi_task_mode:
                        valid_batches = (
                            [
                                valid_data.get_batch()
                                for ii in range(self.valid_numb_batch)
                            ]
                            if valid_data is not None
                            else None
                        )
                        self.valid_on_the_fly(
                            fp, [train_batch], valid_batches, print_header=True
                        )
                    else:
                        train_batches = {}
                        valid_batches = {}
                        # valid_numb_batch_dict
                        for fitting_key in train_data:
                            train_batches[fitting_key] = [
                                train_data[fitting_key].get_batch()
                            ]
                            valid_batches[fitting_key] = (
                                [
                                    valid_data[fitting_key].get_batch()
                                    for ii in range(
                                        self.valid_numb_batch_dict[fitting_key]
                                    )
                                ]
                                if fitting_key in valid_data
                                else None
                            )
                        self.valid_on_the_fly(
                            fp, train_batches, valid_batches, print_header=True
                        )
                is_first_step = False

            if self.timing_in_training:
                tic = time.time()
            train_feed_dict = self.get_feed_dict(train_batch, is_training=True)
            # use tensorboard to visualize the training of deepmd-kit
            # it will takes some extra execution time to generate the tensorboard data
            if self.tensorboard and (cur_batch % self.tensorboard_freq == 0):
                summary, _ = run_sess(
                    self.sess,
                    [summary_merged_op, batch_train_op],
                    feed_dict=train_feed_dict,
                    options=prf_options,
                    run_metadata=prf_run_metadata,
                )
                tb_train_writer.add_summary(summary, cur_batch)
            else:
                # if True:
                run_sess(
                    self.sess,
                    [batch_train_op],
                    feed_dict=train_feed_dict,
                    options=prf_options,
                    run_metadata=prf_run_metadata,
                )
                # else:
                #     backbone_debug_keys = list(self.model.backbone.debug_dict.keys())
                #     debug_out = run_sess(
                #         self.sess,
                #         [self.model.backbone.debug_dict[item_key] for item_key in backbone_debug_keys],
                #         feed_dict=train_feed_dict,
                #         options=prf_options,
                #         run_metadata=prf_run_metadata,
                #     )
                #     run_sess(
                #         self.sess,
                #         [batch_train_op],
                #         feed_dict=train_feed_dict,
                #         options=prf_options,
                #         run_metadata=prf_run_metadata,
                #     )
                #     backbone_debug_out_dict = {item_key: debug_out[i] for i, item_key in enumerate(backbone_debug_keys)}
                #     backbone_debug_out_dict_mean = {}
                #     backbone_debug_out_dict_std = {}
                #     for item_key in backbone_debug_keys:
                #         if isinstance(backbone_debug_out_dict[item_key], list):
                #             for layer_num in range(len(backbone_debug_out_dict[item_key])):
                #                 backbone_debug_out_dict_mean[item_key + '_layer_{}'.format(layer_num)] \
                #                     = backbone_debug_out_dict[item_key][layer_num].mean()
                #                 backbone_debug_out_dict_std[item_key + '_layer_{}'.format(layer_num)] \
                #                     = backbone_debug_out_dict[item_key][layer_num].std()
                #         else:
                #             backbone_debug_out_dict_mean[item_key] \
                #                 = backbone_debug_out_dict[item_key].mean()
                #             backbone_debug_out_dict_std[item_key] \
                #                 = backbone_debug_out_dict[item_key].std()
                #     embed()
            if self.timing_in_training:
                toc = time.time()
            if self.timing_in_training:
                train_time += toc - tic
            cur_batch = run_sess(self.sess, self.global_step)
            self.cur_batch = cur_batch

            # on-the-fly validation
            if self.display_in_training and (cur_batch % self.disp_freq == 0):
                if self.timing_in_training:
                    tic = time.time()
                if self.run_opt.is_chief:
                    if not self.multi_task_mode:
                        valid_batches = (
                            [
                                valid_data.get_batch()
                                for ii in range(self.valid_numb_batch)
                            ]
                            if valid_data is not None
                            else None
                        )
                        self.valid_on_the_fly(fp, [train_batch], valid_batches)
                    else:
                        train_batches = {}
                        valid_batches = {}
                        for fitting_key in train_data:
                            train_batches[fitting_key] = [
                                train_data[fitting_key].get_batch()
                            ]
                            valid_batches[fitting_key] = (
                                [
                                    valid_data[fitting_key].get_batch()
                                    for ii in range(
                                        self.valid_numb_batch_dict[fitting_key]
                                    )
                                ]
                                if fitting_key in valid_data
                                else None
                            )
                        self.valid_on_the_fly(fp, train_batches, valid_batches)
                if self.timing_in_training:
                    toc = time.time()
                    test_time = toc - tic
                    log.info(
                        "batch %7d training time %.2f s, testing time %.2f s"
                        % (cur_batch, train_time, test_time)
                    )
                    # the first training time is not accurate
                    if cur_batch > self.disp_freq or stop_batch < 2 * self.disp_freq:
                        total_train_time += train_time
                    train_time = 0
                if (
                    self.save_freq > 0
                    and cur_batch % self.save_freq == 0
                    and self.saver is not None
                ):
                    self.save_checkpoint(cur_batch)
        if (
            self.save_freq == 0 or cur_batch == 0 or cur_batch % self.save_freq != 0
        ) and self.saver is not None:
            self.save_checkpoint(cur_batch)
        if self.run_opt.is_chief:
            fp.close()
        if self.timing_in_training and stop_batch // self.disp_freq > 0:
            if stop_batch >= 2 * self.disp_freq:
                log.info(
                    "average training time: %.4f s/batch (exclude first %d batches)",
                    total_train_time
                    / (stop_batch // self.disp_freq * self.disp_freq - self.disp_freq),
                    self.disp_freq,
                )
            else:
                log.info(
                    "average training time: %.4f s/batch",
                    total_train_time / (stop_batch // self.disp_freq * self.disp_freq),
                )

        if self.profiling and self.run_opt.is_chief:
            fetched_timeline = timeline.Timeline(prf_run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open(self.profiling_file, "w") as f:
                f.write(chrome_trace)
        if self.enable_profiler and self.run_opt.is_chief:
            tfv2.profiler.experimental.stop()

    def save_checkpoint(self, cur_batch: int):
        try:
            ckpt_prefix = self.saver.save(
                self.sess,
                os.path.join(os.getcwd(), self.save_ckpt),
                global_step=cur_batch,
            )
        except google.protobuf.message.DecodeError as e:
            raise GraphTooLargeError(
                "The graph size exceeds 2 GB, the hard limitation of protobuf."
                " Then a DecodeError was raised by protobuf. You should "
                "reduce the size of your model."
            ) from e
        # make symlinks from prefix with step to that without step to break nothing
        # get all checkpoint files
        original_files = glob.glob(ckpt_prefix + ".*")
        for ori_ff in original_files:
            new_ff = self.save_ckpt + ori_ff[len(ckpt_prefix) :]
            try:
                # remove old one
                os.remove(new_ff)
            except OSError:
                pass
            if platform.system() != "Windows":
                # by default one does not have access to create symlink on Windows
                os.symlink(ori_ff, new_ff)
            else:
                shutil.copyfile(ori_ff, new_ff)
        log.info("saved checkpoint %s" % self.save_ckpt)

    def get_feed_dict(self, batch, is_training, noise_dict=None, mask_mode=''):
        if self.mask_mode != '':
            mask_mode = self.mask_mode
        feed_dict = {}
        for kk in batch.keys():
            if kk == "find_type" or kk == "type" or kk == "real_natoms_vec":
                continue
            if "find_" in kk:
                feed_dict[self.place_holders[kk]] = batch[kk]
            else:
                feed_dict[self.place_holders[kk]] = np.reshape(batch[kk], [-1])
        for ii in ["type"]:
            feed_dict[self.place_holders[ii]] = np.reshape(batch[ii], [-1])
        for ii in ["natoms_vec", "default_mesh"]:
            feed_dict[self.place_holders[ii]] = batch[ii]
        feed_dict[self.place_holders["is_training"]] = is_training
        if self.use_backbone and self.loss_type == "stru":
            # add noise and mask here
            clean_coord = feed_dict[self.place_holders["coord"]]
            feed_dict[self.place_holders["clean_coord"]] = clean_coord
            clean_type = feed_dict[self.place_holders["type"]]
            feed_dict[self.place_holders["clean_type"]] = clean_type
            (
                feed_dict[self.place_holders["coord"]],
                feed_dict[self.place_holders["type"]],
                feed_dict[self.place_holders["noise_mask"]],
                feed_dict[self.place_holders["token_mask"]],
             ) = self.add_noise_mask(
                clean_coord,
                clean_type,
                batch,
                _coord_noise_num=self.coord_noise_num,
                _masked_token_num=self.masked_token_num,
                noise_dict=noise_dict,
                mask_mode=mask_mode,
            )
        return feed_dict

    def add_noise_mask(self, _clean_coord, _clean_type, _batch, _coord_noise_num=10, _masked_token_num=10, noise_dict=None, mask_mode=''):
        natom = _batch["natoms_vec"][1]
        box = _batch["box"].reshape([-1, 9])
        natoms_vec = _batch["natoms_vec"]
        default_mesh = _batch["default_mesh"]
        _clean_coord = _clean_coord.reshape([-1, natom * 3])
        _clean_type = _clean_type.reshape([-1, natom])
        nframes = _clean_coord.shape[0]
        if noise_dict is None:
            if mask_mode not in ['nloc_mask_3x3']:
                coord_noise_num = _coord_noise_num if _coord_noise_num < natom else natom
                noise_idx = np.random.choice(natom, coord_noise_num, replace=False)
                noise_mask = np.full(natom, False)
                noise_mask[noise_idx] = True
            else:
                if mask_mode == 'nloc_mask_3x3':
                    noise_mask = np.full(natom, False)
                    assert natom % 27 == 0, 'natom not in 3x3 copy mode!'
                    nloc_natom = int(natom / 27)
                    noise_mask[:nloc_natom] = True
        else:
            noise_idx = noise_dict['noise_idx'] if "noise_idx" in noise_dict else 0
            noise_mask = np.full(natom, False)
            noise_mask[noise_idx] = True

        if noise_dict is None:
            if not self.same_mask:
                masked_token_num = _masked_token_num if _masked_token_num < natom else natom
                token_idx = np.random.choice(natom, masked_token_num, replace=False)
                token_mask = np.full(natom, False)
                token_mask[token_idx] = True
            else:
                token_mask = noise_mask
        else:
            token_idx = noise_dict['token_idx'] if "token_idx" in noise_dict else 0
            token_mask = np.full(natom, False)
            token_mask[token_idx] = True

        def add_noise(__clean_coord, _noise_mask):
            noise_c = None
            if self.noise_type == "trunc_normal":
                noise_c = np.clip(
                    np.random.randn(nframes, natom, 3) * self.noise,
                    a_min=-self.noise * 2.0,
                    a_max=self.noise * 2.0,
                )
            elif self.noise_type == "normal":
                noise_c = np.random.randn(nframes, natom, 3) * self.noise
            elif self.noise_type == "uniform":
                noise_c = np.random.uniform(
                    low=-self.noise, high=self.noise, size=(nframes, natom, 3)
                )
            else:
                RuntimeError("Unknown noise type !")
            noise_c[:, _noise_mask == False, :] = 0.0
            noise_c = noise_c.reshape([-1, natom * 3])
            return __clean_coord + noise_c

        temp_type = _clean_type.copy()
        temp_type[:, token_mask] = self.model.ntypes - 1  # masked type
        if noise_dict is None:
            while True:
                temp_coord = add_noise(_clean_coord, noise_mask)
                temp_feed_dict = {}
                temp_feed_dict[self.place_holders_temp["coord"]] = temp_coord
                temp_feed_dict[self.place_holders_temp["box"]] = box
                temp_feed_dict[self.place_holders_temp["type"]] = temp_type
                temp_feed_dict[self.place_holders_temp["natoms_vec"]] = natoms_vec
                temp_feed_dict[self.place_holders_temp["default_mesh"]] = default_mesh
                temp_min_nbor_dist = run_sess(self.sub_sess, [self.temp_min_nbor_dist], feed_dict=temp_feed_dict)[0]
                if not math.isclose(temp_min_nbor_dist.min(), 0.0, rel_tol=1e-6):
                    return temp_coord.reshape(-1), temp_type.reshape(-1), \
                           noise_mask.astype(GLOBAL_NP_FLOAT_PRECISION), token_mask.astype(GLOBAL_NP_FLOAT_PRECISION)
        else:
            temp_coord = _clean_coord.reshape([-1, natom, 3]).copy()
            temp_coord[:, noise_mask == True, :] += noise_dict['coord_noise'] if "coord_noise" in noise_dict else 0.0
            return temp_coord.reshape(-1), temp_type.reshape(-1), \
                   noise_mask.astype(GLOBAL_NP_FLOAT_PRECISION), token_mask.astype(GLOBAL_NP_FLOAT_PRECISION)

    def get_global_step(self):
        return run_sess(self.sess, self.global_step)

    # def print_head (self) :  # depreciated
    #     if self.run_opt.is_chief:
    #         fp = open(self.disp_file, "a")
    #         print_str = "# %5s" % 'batch'
    #         print_str += self.loss.print_header()
    #         print_str += '   %8s\n' % 'lr'
    #         fp.write(print_str)
    #         fp.close ()

    def valid_on_the_fly(self, fp, train_batches, valid_batches, print_header=False):
        train_results = self.get_evaluation_results(train_batches)
        valid_results = self.get_evaluation_results(valid_batches)
        # train_logs = {}
        # valid_logs = {}
        # if not self.multi_task_mode:
        #     for k, v in train_results.items():
        #         train_logs[k + '_train'] = v
        #     wb.log(train_logs, step=self.cur_batch)
        #     for k, v in valid_results.items():
        #         valid_logs[k + '_valid'] = v
        #     wb.log(valid_logs, step=self.cur_batch)
        # else:
        #     for item in train_results:
        #         for k, v in train_results[item].items():
        #             train_logs[k + '_train'] = v
        #     wb.log(train_logs, step=self.cur_batch)
        #     for item in valid_results:
        #         for k, v in valid_results[item].items():
        #             valid_logs[k + '_valid'] = v
        #     wb.log(valid_logs, step=self.cur_batch)

        cur_batch = self.cur_batch
        current_lr = run_sess(self.sess, self.learning_rate)
        # wb.log({'lr': current_lr}, step=self.cur_batch)
        if print_header:
            self.print_header(fp, train_results, valid_results, self.multi_task_mode)
        self.print_on_training(
            fp,
            train_results,
            valid_results,
            cur_batch,
            current_lr,
            self.multi_task_mode,
        )

    @staticmethod
    def print_header(fp, train_results, valid_results, multi_task_mode=False):
        print_str = ""
        print_str += "# %5s" % "step"
        if not multi_task_mode:
            if valid_results is not None:
                prop_fmt = "   %11s %11s"
                for k in train_results.keys():
                    print_str += prop_fmt % (k + "_val", k + "_trn")
            else:
                prop_fmt = "   %11s"
                for k in train_results.keys():
                    print_str += prop_fmt % (k + "_trn")
        else:
            for fitting_key in train_results:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11s %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_val", k + "_trn")
                else:
                    prop_fmt = "   %11s"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (k + "_trn")
        print_str += "   %8s\n" % "lr"
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def print_on_training(
        fp, train_results, valid_results, cur_batch, cur_lr, multi_task_mode=False
    ):
        print_str = ""
        print_str += "%7d" % cur_batch
        if not multi_task_mode:
            if valid_results is not None:
                prop_fmt = "   %11.2e %11.2e"
                for k in valid_results.keys():
                    # assert k in train_results.keys()
                    print_str += prop_fmt % (valid_results[k], train_results[k])
            else:
                prop_fmt = "   %11.2e"
                for k in train_results.keys():
                    print_str += prop_fmt % (train_results[k])
        else:
            for fitting_key in train_results:
                if valid_results[fitting_key] is not None:
                    prop_fmt = "   %11.2e %11.2e"
                    for k in valid_results[fitting_key].keys():
                        # assert k in train_results[fitting_key].keys()
                        print_str += prop_fmt % (
                            valid_results[fitting_key][k],
                            train_results[fitting_key][k],
                        )
                else:
                    prop_fmt = "   %11.2e"
                    for k in train_results[fitting_key].keys():
                        print_str += prop_fmt % (train_results[fitting_key][k])
        print_str += "   %8.1e\n" % cur_lr
        fp.write(print_str)
        fp.flush()

    @staticmethod
    def eval_single_list(single_batch_list, loss, sess, get_feed_dict_func, prefix=""):
        if single_batch_list is None:
            return None
        numb_batch = len(single_batch_list)
        sum_results = {}  # sum of losses on all atoms
        sum_natoms = 0
        for i in range(numb_batch):
            batch = single_batch_list[i]
            natoms = batch["natoms_vec"]
            feed_dict = get_feed_dict_func(batch, is_training=False)
            results = loss.eval(sess, feed_dict, natoms)

            for k, v in results.items():
                if k == "natoms":
                    sum_natoms += v
                else:
                    sum_results[k] = sum_results.get(k, 0.0) + v * results["natoms"]
        single_results = {
            prefix + k: v / sum_natoms
            for k, v in sum_results.items()
            if not k == "natoms"
        }
        return single_results

    def get_evaluation_results(self, batch_list):
        if not self.multi_task_mode:
            avg_results = self.eval_single_list(
                batch_list, self.loss, self.sess, self.get_feed_dict
            )
        else:
            avg_results = {}
            for fitting_key in batch_list:
                avg_results[fitting_key] = self.eval_single_list(
                    batch_list[fitting_key],
                    self.loss_dict[fitting_key],
                    self.sess,
                    self.get_feed_dict,
                    prefix="{}_".format(fitting_key),
                )
        return avg_results

    def save_compressed(self):
        """
        Save the compressed graph
        """
        self._init_session()
        if self.is_compress:
            self.saver.save(self.sess, os.path.join(os.getcwd(), self.save_ckpt))

    def _get_place_horders(self, data_dict):
        for kk in data_dict.keys():
            if kk == "type":
                continue
            prec = GLOBAL_TF_FLOAT_PRECISION
            if data_dict[kk]["high_prec"]:
                prec = GLOBAL_ENER_FLOAT_PRECISION
            self.place_holders[kk] = tf.placeholder(prec, [None], name="t_" + kk)
            self.place_holders["find_" + kk] = tf.placeholder(
                tf.float32, name="t_find_" + kk
            )

    def _init_from_frz_model(self):
        try:
            graph, graph_def = load_graph_def(self.run_opt.init_frz_model)
        except FileNotFoundError as e:
            # throw runtime error if there's no frozen model
            raise RuntimeError(
                "The input frozen model %s (%s) does not exist! Please check the path of the frozen model. "
                % (
                    self.run_opt.init_frz_model,
                    os.path.abspath(self.run_opt.init_frz_model),
                )
            ) from e
        # get the model type from the frozen model(self.run_opt.init_frz_model)
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            # throw runtime error if the frozen_model has no model type information...
            raise RuntimeError(
                "The input frozen model: %s has no 'model_type' information, "
                "which is not supported by the 'dp train init-frz-model' interface. "
                % self.run_opt.init_frz_model
            ) from e
        else:
            self.model_type = bytes.decode(t_model_type)
        if self.model_type == "compressed_model":
            self.frz_model = self.run_opt.init_frz_model
        self.model.init_variables(graph, graph_def, model_type=self.model_type)

    def _init_from_ckpt(self, ckpt_meta: str):
        with tf.Graph().as_default() as graph:
            tf.train.import_meta_graph(f"{ckpt_meta}.meta", clear_devices=True)
        # get the model type from the model
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            self.model_type = "original_model"
        else:
            self.model_type = bytes.decode(t_model_type)
        if self.model_type == "compressed_model":
            self.ckpt_meta = ckpt_meta

    def _init_from_pretrained_model(
        self, data, origin_type_map=None, bias_shift="delta"
    ):
        """
        Init the embedding net variables with the given frozen model

        Parameters
        ----------
        data : DeepmdDataSystem
            The training data.
        origin_type_map : list
            The original type_map in dataset, they are targets to change the energy bias.
        bias_shift : str
            The mode for changing energy bias : ['delta', 'statistic']
            'delta' : perform predictions on energies of target dataset,
                    and do least sqaure on the errors to obtain the target shift as bias.
            'statistic' : directly use the statistic energy bias in the target dataset.
        """
        try:
            graph, graph_def = load_graph_def(self.run_opt.finetune)
        except FileNotFoundError as e:
            # throw runtime error if there's no frozen model
            raise RuntimeError(
                "The input frozen pretrained model %s (%s) does not exist! "
                "Please check the path of the frozen pretrained model. "
                % (self.run_opt.finetune, os.path.abspath(self.run_opt.finetune))
            ) from e
        # get the model type from the frozen model(self.run_opt.finetune)
        try:
            t_model_type = get_tensor_by_name_from_graph(graph, "model_type")
        except GraphWithoutTensorError as e:
            # throw runtime error if the frozen_model has no model type information...
            raise RuntimeError(
                "The input frozen pretrained model: %s has no 'model_type' information, "
                "which is not supported by the 'dp train finetune' interface. "
                % self.run_opt.finetune
            ) from e
        else:
            self.model_type = bytes.decode(t_model_type)
        assert (
            self.model_type != "compressed_model"
        ), "Compressed models are not supported for finetuning!"
        self.model.init_variables(graph, graph_def, model_type=self.model_type)
        log.info(
            "Changing energy bias in pretrained model for types {}... "
            "(this step may take long time)".format(str(origin_type_map))
        )
        self._change_energy_bias(
            data, self.run_opt.finetune, origin_type_map, bias_shift
        )

    def _change_energy_bias(
        self, data, frozen_model, origin_type_map, bias_shift="delta"
    ):
        full_type_map = data.get_type_map()
        assert (
            self.fitting_type == "ener"
        ), "energy bias changing only supports 'ener' fitting net!"
        self.model.fitting.change_energy_bias(
            data,
            frozen_model,
            origin_type_map,
            full_type_map,
            bias_shift=bias_shift,
            ntest=self.model_param.get("data_bias_nsample", 10),
        )

    def _min_dist_sub_graph(self):
        self.place_holders_temp = {}
        sub_graph = tf.Graph()
        with sub_graph.as_default():
            name_pfx = "min_dist_"
            for ii in ["coord", "box"]:
                self.place_holders_temp[ii] = tf.placeholder(
                    GLOBAL_NP_FLOAT_PRECISION, [None, None], name=name_pfx + "t_" + ii
                )
            self.place_holders_temp["type"] = tf.placeholder(
                tf.int32, [None, None], name=name_pfx + "t_type"
            )
            self.place_holders_temp["natoms_vec"] = tf.placeholder(
                tf.int32, [self.model.ntypes + 2], name=name_pfx + "t_natoms"
            )
            self.place_holders_temp["default_mesh"] = tf.placeholder(
                tf.int32, [None], name=name_pfx + "t_mesh"
            )
            self.temp_max_nbor_size, self.temp_min_nbor_dist = op_module.neighbor_stat(
                self.place_holders_temp["coord"],
                self.place_holders_temp["type"],
                self.place_holders_temp["natoms_vec"],
                self.place_holders_temp["box"],
                self.place_holders_temp["default_mesh"],
                rcut=self.model.rcut,
            )
        self.sub_sess = tf.Session(graph=sub_graph, config=default_tf_session_config)
