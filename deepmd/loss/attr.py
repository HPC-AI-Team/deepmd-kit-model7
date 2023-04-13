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


class AttrStdLoss(Loss):
    """
    Standard loss function for DP models

    Parameters
    ----------
    """

    def __init__(
        self,
        attribute: str = "sum",
    ) -> None:
        self.attribute = attribute
        # data required
        add_data_requirement("energy", 1, atomic=False, must=False, high_prec=True)

    def build(self, learning_rate, natoms, model_dict, label_dict, suffix):
        attr_out = model_dict["attr_out"]
        attr_hat = label_dict["energy"]
        l2_attr_loss = tf.reduce_mean(
            tf.square(attr_out - attr_hat), name="l2_" + suffix
        )

        atom_norm = 1.0 / global_cvt_2_tf_float(natoms[0])
        atom_norm_attr = 1.0 / global_cvt_2_ener_float(natoms[0])
        l2_loss = 0
        more_loss = {}
        if self.attribute == 'sum':
            l2_loss += atom_norm_attr * l2_attr_loss
        elif self.attribute == 'mean':
            l2_loss += l2_attr_loss
        else:
            raise RuntimeError(f"Unknown attribute type {self.attribute}!")
        more_loss["l2_attr_loss"] = l2_attr_loss

        # only used when tensorboard was set as true
        self.l2_l = l2_loss
        self.l2_more = more_loss
        return l2_loss, more_loss

    def eval(self, sess, feed_dict, natoms):
        placeholder = self.l2_l
        run_data = [
            self.l2_l,
            self.l2_more["l2_attr_loss"],
        ]
        error, error_attr = run_sess(
            sess, run_data, feed_dict=feed_dict
        )
        results = {"natoms": natoms[0], "rmse": np.sqrt(error)}
        if self.attribute == 'sum':
            results["rmse_per_atom"] = np.sqrt(error_attr) / natoms[0]
        elif self.attribute == 'mean':
            results["rmse_mean"] = np.sqrt(error_attr)
        else:
            raise RuntimeError(f"Unknown attribute type {self.attribute}!")

        return results

