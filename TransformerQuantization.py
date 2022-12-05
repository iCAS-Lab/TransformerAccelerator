#
# Author(s): Brendan Reidy
# Email: bcreidy@email.sc.edu
# Date created: Sep 3 2022
# Last Modified: Sep 11 2022
#

import tensorflow_model_optimization as tfmot
import tensorflow as tf
import TransformerModel


WEIGHT_BITS = 8
ACTIVATION_BITS = 8

LastValueQuantizer = tfmot.quantization.keras.quantizers.LastValueQuantizer
AllValuesQuantizer = tfmot.quantization.keras.quantizers.AllValuesQuantizer
MovingAverageQuantizer = tfmot.quantization.keras.quantizers.MovingAverageQuantizer
quantize_annotate_layer = tfmot.quantization.keras.quantize_annotate_layer
quantize_annotate_model = tfmot.quantization.keras.quantize_annotate_model
quantize_scope = tfmot.quantization.keras.quantize_scope

class ScaledDotProductQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [
        (layer.kernel_q, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        (layer.kernel_k, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        (layer.kernel_v, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return []
      
    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel_q = quantize_weights[0]
      layer.kernel_k = quantize_weights[1]
      layer.kernel_v = quantize_weights[2]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      pass

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

class ScaledConvolutionalDotProductQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return [
        (layer.kernel_q, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        (layer.kernel_k, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        (layer.kernel_v, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
        ]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return [
        (layer.relu1, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)),
        (layer.relu2, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)),
        (layer.relu3, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)),
        (layer.softmax, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False))
        ]
    
    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      layer.kernel_q = quantize_weights[0]
      layer.kernel_k = quantize_weights[1]
      layer.kernel_v = quantize_weights[2]

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      layer.relu1 = quantize_activations[0]
      layer.relu2 = quantize_activations[1]
      layer.relu3 = quantize_activations[2]
      layer.softmax = quantize_activations[3]

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []
      #return [MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
      return {}

class MultiHeadedAttentionQuantizeConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
        retVals = []
        sdp_quant_config = ScaledDotProductQuantizeConfig()
        for head in layer.attention_heads:
            retVals += sdp_quant_config.get_weights_and_quantizers(head)
        return [
            (layer.kernel, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
            ] + retVals

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        sdp_quant_config = ScaledDotProductQuantizeConfig()
        retVals = []
        for head in layer.attention_heads:
                retVals += sdp_quant_config.get_activations_and_quantizers(head)
        return [] + retVals

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
        layer.kernel = quantize_weights[0]
        sdp_quant_config = ScaledDotProductQuantizeConfig()
        offset = 1
        for head in layer.attention_heads:
            lookAhead = len(sdp_quant_config.get_weights_and_quantizers(head))
            sdp_quant_config.set_quantize_weights(head, quantize_weights[offset:offset+lookAhead])
            offset+=lookAhead

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`
        # , in the same order.
        sdp_quant_config = ScaledDotProductQuantizeConfig()
        offset = 0
        for head in layer.attention_heads:
            lookAhead = len(sdp_quant_config.get_activations_and_quantizers(head))
            sdp_quant_config.set_quantize_activations(head, quantize_activations[offset:offset+lookAhead])
            offset+=lookAhead

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []

    def get_config(self):
      return {}

class NoQuantizationConfig(tfmot.quantization.keras.QuantizeConfig):
    # Configure how to quantize weights.
    def get_weights_and_quantizers(self, layer):
      return []
      #return [(layer.embeddings, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False))]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
      return []
      #return [(layer.relu, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False))]

    def set_quantize_weights(self, layer, quantize_weights):
      # Add this line for each item returned in `get_weights_and_quantizers`
      # , in the same order
      #layer.embeddings = quantize_weights[0]
      pass

    def set_quantize_activations(self, layer, quantize_activations):
      # Add this line for each item returned in `get_activations_and_quantizers`
      # , in the same order.
      #layer.relu = quantize_activations[0]
      pass
      

    # Configure how to quantize outputs (may be equivalent to activations).
    def get_output_quantizers(self, layer):
      return []
      #return [MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)]

    def get_config(self):
      return {}

class BERTEncoderQuantizationConfig(tfmot.quantization.keras.QuantizeConfig):
  def get_weights_and_quantizers(self, layer):
    retVals = []

    mha_quant_config = MultiHeadedAttentionQuantizeConfig()
    retVals = mha_quant_config.get_weights_and_quantizers(layer.mha)
    return [
            (layer.kernel_dff, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
            (layer.kernel_out, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
            (layer.layernorm1.gamma, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False)),
            (layer.layernorm1.beta, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False))
            ] + retVals
    
  def get_activations_and_quantizers(self, layer):
    return []
    #return [
    #    (layer.activation1, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False))
    #    ]

  def set_quantize_weights(self, layer, quantize_weights):
    # Add this line for each item returned in `get_weights_and_quantizers`
    # , in the same order
    layer.kernel_dff = quantize_weights[0]
    layer.kernel_out = quantize_weights[1]
    layer.layernorm1.gamma = quantize_weights[2]
    layer.layernorm1.beta = quantize_weights[3]
    mha_quant_config = MultiHeadedAttentionQuantizeConfig()
    offset = 4
    mha_quant_config.set_quantize_weights(layer.mha, quantize_weights[offset:])

  def set_quantize_activations(self, layer, quantize_activations):
    #layer.activation1 = quantize_activations[0]
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}

class BERTQuantizationConfig(tfmot.quantization.keras.QuantizeConfig):
  def get_weights_and_quantizers(self, layer):
    layer.embedding.norm.quantized = True
    return []
    
  def get_activations_and_quantizers(self, layer):
    return []

  def set_quantize_weights(self, layer, quantize_weights):
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    pass

  def get_output_quantizers(self, layer):
    return []

  def get_config(self):
    return {}

class EmbeddingQuantizationConfig(tfmot.quantization.keras.QuantizeConfig):
  # Configure how to quantize weights.
  def get_weights_and_quantizers(self, layer):
    return [(layer.embeddings, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False))]
    #return [(layer.embeddings, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False))]

  # Configure how to quantize activations.
  def get_activations_and_quantizers(self, layer):
    return []
    #return [(layer.relu, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False))]

  def set_quantize_weights(self, layer, quantize_weights):
    # Add this line for each item returned in `get_weights_and_quantizers`
    # , in the same order
    #layer.embeddings = quantize_weights[0]
    layer.embeddings = quantize_weights[0]

  def set_quantize_activations(self, layer, quantize_activations):
    # Add this line for each item returned in `get_activations_and_quantizers`
    # , in the same order.
    #layer.relu = quantize_activations[0]
    pass
    

  # Configure how to quantize outputs (may be equivalent to activations).
  def get_output_quantizers(self, layer):
    return [MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)]

  def get_config(self):
    return {}


class OutputQuantizationConfig(tfmot.quantization.keras.QuantizeConfig):
  # Configure how to quantize weights.
  def get_weights_and_quantizers(self, layer):
    return []
    #return [(layer.embeddings, LastValueQuantizer(num_bits=WEIGHT_BITS, symmetric=True, narrow_range=False, per_axis=False))]

  # Configure how to quantize activations.
  def get_activations_and_quantizers(self, layer):
    return []
    #return [(layer.relu, MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False))]

  def set_quantize_weights(self, layer, quantize_weights):
    # Add this line for each item returned in `get_weights_and_quantizers`
    # , in the same order
    #layer.embeddings = quantize_weights[0]
    pass

  def set_quantize_activations(self, layer, quantize_activations):
    # Add this line for each item returned in `get_activations_and_quantizers`
    # , in the same order.
    #layer.relu = quantize_activations[0]
    pass
    

  # Configure how to quantize outputs (may be equivalent to activations).
  def get_output_quantizers(self, layer):
    return [MovingAverageQuantizer(num_bits=ACTIVATION_BITS, symmetric=False, narrow_range=False, per_axis=False)]

  def get_config(self):
    return {}

def apply_quantization_to_custom(layer):
  if isinstance(layer, TransformerModel.BERT):
    return tfmot.quantization.keras.quantize_annotate_layer(layer, OutputQuantizationConfig())
  return tfmot.quantization.keras.quantize_annotate_layer(layer)

def QuantizeTransformer(model):
    annotated_model = tf.keras.models.clone_model(model, clone_function=apply_quantization_to_custom)
    with quantize_scope(
        {'ScaledDotProduct':TransformerModel.ScaledDotProduct,
        'MultiHeadedAttention':TransformerModel.MultiHeadedAttention,
        'Dense':TransformerModel.Dense,
        'PartitionLayer':TransformerModel.PartitionLayer,
        'PartitionEmbedding':TransformerModel.PartitionEmbedding,
        'BertEmbedding':TransformerModel.BertEmbedding,
        'BERT':TransformerModel.BERT,
        'BertEncoder':TransformerModel.BertEncoder,
        'OutputQuantizationConfig':OutputQuantizationConfig}
        ):
        quant_aware_model = tfmot.quantization.keras.quantize_apply(annotated_model)
    return quant_aware_model