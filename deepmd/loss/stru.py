import numpy as np

from deepmd.common import (
    add_data_requirement,
)
from deepmd.env import (
    global_cvt_2_ener_float,
    global_cvt_2_tf_float,
    tf,
)
from deepmd.utils.sess import (
    run_sess,
)

from .loss import (
    Loss,
)


class StruReconLoss(Loss):
    """
    Loss function for structure reconstruction

    Parameters
    ----------
    enable_atom_ener_coeff : bool
        if true, the energy will be computed as \\sum_i c_i E_i
    """

    def __init__(self, jdata, **kwarg) -> None:
        self.ntypes = jdata["ntypes"]
        self.masked_token_loss = float(jdata.get("masked_token_loss", 1.00))
        self.masked_coord_loss = float(jdata.get("masked_coord_loss", 1.00))
        self.norm_loss = float(jdata.get("norm_loss", 0.01))
        self.has_coord = self.masked_coord_loss != 0.0
        self.has_token = self.masked_token_loss != 0.0
        self.has_norm = self.norm_loss != 0.0
        self.use_l1 = jdata.get("use_l1", True)
        self.beta = float(jdata.get("beta", 1.00))
        self.frac_beta = 1.00 / self.beta
        self.mask_loss_coord = jdata.get("mask_loss_coord", True)
        self.mask_loss_token = jdata.get("mask_loss_token", True)
        # data required
        # add_data_requirement('energy', 1, atomic=False, must=False, high_prec=True)
        # add_data_requirement('force', 3, atomic=True, must=False, high_prec=False)

    def build(self,
              learning_rate,
              natoms,
              model_dict,
              label_dict,
              suffix):
        self.coord_output = tf.reshape(model_dict['coord_output'], [-1, natoms[0], 3])
        self.coord_hat = tf.reshape(label_dict['clean_coord'], [-1, natoms[0], 3])
        self.type_output = tf.reshape(model_dict['logits'], [-1, natoms[0], self.ntypes - 1])
        self.type_hat = tf.reshape(label_dict['clean_type'], [-1, natoms[0]])
        self.noise_mask = tf.reshape(label_dict['noise_mask'], [1, -1])  # 1 x natoms
        self.token_mask = tf.reshape(label_dict['token_mask'], [1, -1])  # 1 x natoms
        self.norm_x = model_dict['norm_x']
        self.norm_delta_pair_rep = model_dict['norm_delta_pair_rep']
        loss = 0
        more_loss = {}
        if self.has_coord:
            coord_diff = self.coord_output - self.coord_hat  # nframes x natoms x 3
            if not self.use_l1:
                if not self.mask_loss_coord:
                    stru_loss = tf.reduce_mean(tf.square(coord_diff), name='l2_' + suffix)
                else:
                    stru_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.reduce_mean(
                                tf.square(coord_diff),
                                axis=-1,
                            ) * self.noise_mask,
                            axis=-1,
                        ) / tf.reduce_sum(self.noise_mask),
                        name='l2_' + suffix,
                    )
            else:
                if not self.mask_loss_coord:
                    stru_loss = tf.reduce_mean(tf.where(tf.abs(coord_diff) < self.beta,
                                                        0.5 * self.frac_beta * tf.square(coord_diff),
                                                        tf.abs(coord_diff) - 0.5 * self.beta,
                                                        ), name='l1_' + suffix)
                else:
                    stru_loss = tf.reduce_mean(
                        tf.reduce_sum(
                            tf.reduce_mean(
                                tf.where(tf.abs(coord_diff) < self.beta,
                                         0.5 * self.frac_beta * tf.square(coord_diff),
                                         tf.abs(coord_diff) - 0.5 * self.beta,
                                         ),
                                axis=-1,
                            ) * self.noise_mask,
                            axis=-1,
                        ) / tf.reduce_sum(self.noise_mask),
                        name='l2_' + suffix,
                    )
            loss += self.masked_coord_loss * stru_loss
            more_loss['stru_loss'] = stru_loss

        if self.has_token:
            if not self.mask_loss_token:
                token_loss = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        labels=self.type_hat,
                        logits=self.type_output,
                    ),
                )
            else:
                token_loss = tf.reduce_mean(
                    tf.reduce_sum(
                        tf.nn.sparse_softmax_cross_entropy_with_logits(
                            labels=self.type_hat,
                            logits=self.type_output,
                        ) * self.token_mask,  # nframes x natoms
                        axis=-1,
                    ) / tf.reduce_sum(self.token_mask),
                )

            loss += self.masked_token_loss * token_loss
            more_loss['token_loss'] = token_loss

        if self.has_norm:
            norm_loss = self.norm_x + self.norm_delta_pair_rep
            loss += self.norm_loss * norm_loss
            more_loss['norm_loss'] = norm_loss

        self.loss = loss
        self.l2_more = more_loss
        return loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        run_data = [
            self.loss,
            self.l2_more['stru_loss'],
            self.l2_more['token_loss'],
            self.l2_more['norm_loss'],
        ]
        error, error_stru, error_token, error_norm = run_sess(sess, run_data, feed_dict=feed_dict)
        results = {"natoms": natoms[0]}
        results["stru_loss"] = error_stru
        results["token_loss"] = error_token
        results["norm_loss"] = error_norm
        return results
