from deepmd.env import tf
from deepmd.utils import Plugin, PluginVariant


class Backbone:
    @property
    def precision(self) -> tf.DType:
        """Precision of backbone network."""
        return self.backbone_precision

    def init_variables(self,
                       graph: tf.Graph,
                       graph_def: tf.GraphDef,
                       suffix: str = "",
                       ) -> None:
        """
        Init the backbone net variables with the given dict

        Parameters
        ----------
        graph : tf.Graph
            The input frozen model graph
        graph_def : tf.GraphDef
            The input frozen model graph_def
        suffix : str
            suffix to name scope

        Notes
        -----
        This method is called by others when the backbone supported initialization from the given variables.
        """
        raise NotImplementedError(
            "Backbone %s doesn't support initialization from the given variables!" % type(self).__name__)