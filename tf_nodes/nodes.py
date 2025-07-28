from ryven.node_env import *
from random import random
import tensorflow as tf
import logging
import ast
import numpy as np
import sys
import ryven
from qtpy.QtWidgets import QLineEdit
import os
import re

logger = logging.getLogger(__name__)

class NameRegistry:
    def __init__(self):
        self.used_names = set()
        self.name_counts = {}

    def reset(self):
        self.used_names.clear()
        self.name_counts.clear()

    def get_unique_name(self, base):
        count = self.name_counts.get(base, 0)
        name = f"{base}_{count}"
        self.name_counts[base] = count + 1
        self.used_names.add(name)
        return name

    def has_name(self, name):
        return name in self.used_names

# Initialise an instance
name_registry = NameRegistry()


# your node definitions go here
class InputNode(Node):
    title = "Input Node"
    init_inputs = [NodeInputType(label='Shape')]
    init_outputs = [NodeOutputType(label='Tensor')]

    def update_event(self, inp=-1):
        logger.info("InputNode:")
        shape = self.input(0).payload
        try:
            tensor = tf.keras.Input(shape=shape)
            layer_name = name_registry.get_unique_name("input")
            code = f"{layer_name} = tf.keras.Input(shape={shape})"
            print(f"Successful")
            print(f"----------------------------------------")
            self.set_output_val(0, Data((tensor, code)))
        except Exception as e:
            print("InputNode error:", e)
            self.set_output_val(0, Data((None, "")))


class AddNode(Node):
    title = "Add Node"
    init_inputs = [
        NodeInputType(label='Tensor 1'),
        NodeInputType(label='Tensor 2')
    ]
    init_outputs = [NodeOutputType(label='Result')]

    def update_event(self, inp=-1):
        print(f"AddNode:\n")
        try:
            # Get the previous node code and index
            input_valA, input_codeA = self.input(0).payload
            input_valB, input_codeB = self.input(1).payload

            print("Attempting to add Tensor 1 and Tensor 2...")
            layer = tf.keras.layers.Add()
            result = layer([input_valA, input_valB])
            
            # Generate code
            code = self.generate_code(input_codeA, input_codeB)

            print("Successful. Result shape:", result.shape)
            print("----------------------------------------")
            self.set_output_val(0, Data((result, code)))

        except Exception as e:
            print("AddNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_codeA, input_codeB):
        lines_A = input_codeA.strip().split('\n')
        lines_B = input_codeB.strip().split('\n')

        seen = set()
        merged_lines = []
        for line in lines_A + lines_B:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        varA = lines_A[-1].split('=')[0].strip()
        varB = lines_B[-1].split('=')[0].strip()

        layer_name = name_registry.get_unique_name("add")
        merged_lines.append(f"{layer_name} = tf.keras.layers.Add()([{varA}, {varB}])")

        return "\n".join(merged_lines)


"""class PrintShapeNode(Node):
    title = "Print Shape"
    init_inputs = [NodeInputType(label="Tensor")]

    def update_event(self, inp=-1):
        print(f"PrintShapeNode:\n")

        tensor = self.input(0).payload
        if hasattr(tensor, 'shape'):
            print("Shape:", tensor.shape)
            print(f"----------------------------------------")
        else:
            print("Not a tensor or no shape")
            print(f"----------------------------------------")"""


# Input: activation units
class DenseNode(Node):
    title = 'Dense Node'
    init_inputs = [NodeInputType(label='Input Tensor'),
                   NodeInputType(label='Units'),
                   NodeInputType(label='Activation (Optional)'),
                   NodeInputType(label='Kernel Initializer (Optional)')]
    init_outputs = [NodeOutputType(label='Output Tensor')]

    def update_event(self, inp=-1):
        try:
            print(f"DenseNode:\n")
            input_val, input_code = self.input(0).payload
            units = self.input(1).payload
            #activation = self.input(2).payload if self.input(2) and self.input(2).payload is not None else None
            """activation = self.input(2).payload
            if activation is None or activation == "":
                activation = None"""
            activation = self.input(2).payload if self.input(2) and self.input(2).payload not in [None, ""] else None

            # Default initializer values
            kernel_initializer = 'glorot_uniform'
            kernel_initializer_code = "'glorot_uniform'"

            # Override if custom initializer provided
            """if self.input(3) and self.input(3).payload is not None:
                initializer_obj, initializer_code = self.input(3).payload
                kernel_initializer = initializer_obj
                kernel_initializer_code = initializer_code"""

            if self.input(3) and self.input(3).payload:
                initializer_payload = self.input(3).payload
                print("Received initializer:", initializer_payload)

                initializer_obj, initializer_code = initializer_payload
                kernel_initializer = initializer_obj
                kernel_initializer_code = initializer_code
            else:
                print("Using default initializer: glorot_uniform")

            layer = tf.keras.layers.Dense(units=units, activation=activation, kernel_initializer=kernel_initializer)
            result = layer(input_val)
            print(f"Sucessful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code, activation, units, kernel_initializer_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("DenseNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, activation, units, kernel_initializer_code):
        layer_name = name_registry.get_unique_name("dense")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()
        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Dense({units}, activation='{activation}', kernel_initializer={kernel_initializer_code})({input_var_name})"

        return code
    

class DropoutNode(Node):
    title = "Dropout"
    init_inputs = [
        NodeInputType(label="Input Tensor"),
        NodeInputType(label="Rate")
    ]
    init_outputs = [NodeOutputType(label="Result")]

    def update_event(self, inp=-1):
        try:
            print(f"Dropout")
            input_val, input_code = self.input(0).payload
            rate = self.input(1).payload

            layer = tf.keras.layers.Dropout(rate=rate)
            result = layer(input_val, training=True)
            print(f"success")

            code = self.generate_code(input_code, rate)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("DropoutNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, rate):
        layer_name = name_registry.get_unique_name("dropout")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Dropout({rate})({input_var_name})"
        return code


# Check if you can input a list into 
class ReLUNode(Node):
    title = "ReLU Activation"
    init_inputs = [NodeInputType(label='Input Tensor')]
    init_outputs = [NodeOutputType(label='Activated Tensor')]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            result = tf.keras.activations.relu(input_val)

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("ReLUNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code):
        layer_name = name_registry.get_unique_name("relu")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.activations.relu({input_var_name})"
        return code


class ZeroPadding2D(Node):
    title = "ZeroPadding2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label="Padded Tensor")]

    def update_event(self, inp=-1):
        try:
            print(f"ZeroPadding2D Node")
            input_val, input_code = self.input(0).payload
            padding = self.input(1).payload

            # Only pass through if input is a Keras symbolic tensor
            if not hasattr(input_val, '_keras_history'):
                raise ValueError("Input is not a Keras symbolic tensor")

            layer = tf.keras.layers.ZeroPadding2D(padding=padding)
            result = layer(input_val)
            print(f"Successful")

            code = self.generate_code(input_code, padding)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("ZeroPaddingNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, padding):
        layer_name = name_registry.get_unique_name("zero_padding2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.ZeroPadding2D(padding={padding})({input_var_name})"
        return code


class BatchNormalization(Node):
    title = "BatchNormalization"
    init_inputs = [NodeInputType(label="Input Tensor"),
                   NodeInputType(label="Axis")]
    init_outputs = [NodeOutputType(label="Normalized Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            axis = self.input(1).payload

            layer = tf.keras.layers.BatchNormalization(axis=axis)
            result = layer(input_val)
            print(f"Successful")

            code = self.generate_code(input_code, axis)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("BatchNormalizationNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, axis):
        layer_name = name_registry.get_unique_name("batch_normalization")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.BatchNormalization(axis={axis})({input_var_name})"
        return code
    

class FlattenNode(Node):
    title = "Flatten"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Flattened Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            layer = tf.keras.layers.Flatten()
            result = layer(input_val)

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("FlattenNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code):
        layer_name = name_registry.get_unique_name("flatten")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Flatten()({input_var_name})"
        return code


class Conv2DNode(Node):
    title = "Conv2D Node"
    init_inputs = [NodeInputType(label="Tensor"), 
                   NodeInputType(label="Filters"), 
                   NodeInputType(label="Kernel Size"),
                   NodeInputType(label="Strides (Optional)"),
                   NodeInputType(label="Padding (Optional)"),
                   NodeInputType(label="Kernel Initializer (Optional)")]
    init_outputs = [NodeOutputType()]

    def update_event(self, inp=-1):
        try: 
            print(f"Conv2D Node:\n")
            input_val, input_code = self.input(0).payload
            filters = self.input(1).payload
            kernel_size = self.input(2).payload
            
            # Optional inputs with defaults
            strides = self.input(3).payload if self.input(3) and self.input(3).payload is not None else (1, 1)
            padding = self.input(4).payload if self.input(4) and self.input(4).payload is not None else 'valid'

            # Default initializer values
            kernel_initializer = 'glorot_uniform'
            kernel_initializer_code = "'glorot_uniform'"

            # Override if custom initializer provided
            if self.input(5) and self.input(5).payload is not None:
                initializer_obj, initializer_code = self.input(5).payload
                kernel_initializer = initializer_obj
                kernel_initializer_code = initializer_code

            layer = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=kernel_size, 
                                           strides=strides, padding=padding, 
                                           kernel_initializer=kernel_initializer)
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code, filters, kernel_size, strides, padding, kernel_initializer_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[Conv2DNode error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, filters, kernel_size, strides, padding, kernel_initializer_code):
        layer_name = name_registry.get_unique_name("conv2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, strides={strides}, padding='{padding}', kernel_initializer={kernel_initializer_code})({input_var_name})"
        return code


# Performs average pooling on a 2D input tensor, calculates the mean of all values in the window
class AveragePooling2DNode(Node):
    title = "AveragePooling2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Pool Size"), 
                   NodeInputType(label="Strides"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            print(f"AveragePooling2D:\n")
            pool_size = self.input(1).payload
            strides = self.input(2).payload
            padding = self.input(3).payload
            
            layer = tf.keras.layers.AveragePooling2D(pool_size=pool_size,
                                                     strides=strides,
                                                     padding=padding)
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code, pool_size, strides, padding)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[AveragePooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))
    
    def generate_code(self, input_code, pool_size, strides, padding):
        layer_name = name_registry.get_unique_name("average_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.AveragePooling2D(pool_size={pool_size}, strides={strides}, padding='{padding}')({input_var_name})"
        return code


# MaxPooling is a downsampling technique that extracts the maximum value from each window
class MaxPooling2DNode(Node):
    title = "MaxPooling2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Pool Size"), 
                   NodeInputType(label="Strides"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label='Pooled Tensor')]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            print(f"MaxPooling2D")
            pool_size = self.input(1).payload
            strides = self.input(2).payload
            padding = self.input(3).payload

            layer = tf.keras.layers.MaxPooling2D(pool_size=pool_size,
                                                strides=strides,
                                                padding=padding)
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code, pool_size, strides, padding)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[MaxPooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, pool_size, strides, padding):
        layer_name = name_registry.get_unique_name("max_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.MaxPooling2D(pool_size={pool_size}, strides={strides}, padding={padding})({input_var_name})"
        return code


class GlobalMaxPooling2DNode(Node):
    title = "GlobalMaxPooling2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            print(f"GlobalMaxPooling2D")

            layer = tf.keras.layers.GlobalMaxPooling2D()
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[GlobalMaxPooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code):
        layer_name = name_registry.get_unique_name("global_max_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.GlobalMaxPooling2D()({input_var_name})"
        return code
    

class GlobalAveragePooling2DNode(Node):
    title = "GlobalAveragePooling2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code = self.input(0).payload
            print(f"GlobalAveragePooling2D Node")

            layer = tf.keras.layers.GlobalAveragePooling2D()
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[GlobalAveragePooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code):
        layer_name = name_registry.get_unique_name("global_average_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.GlobalAveragePooling2D()({input_var_name})"
        return code


class ConcatenateNode(Node):
    title = 'Concatenate'
    init_inputs = [NodeInputType('Tensor 1'),
                   NodeInputType('Tensor 2'), 
                   NodeInputType("Axis")]
    init_outputs = [NodeOutputType('Result')]

    def update_event(self, inp=-1):
        try:
            print(f"Concatenate")
            input_val_a, input_code_a = self.input(0).payload
            input_val_b, input_code_b = self.input(1).payload
            axis = self.input(2).payload

            result = tf.keras.layers.concatenate([input_val_a, input_val_b], axis=axis)

            code = self.generate_code(input_code_a, input_code_b, axis)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("ConcatenateNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code_a, input_code_b, axis):
        layer_name = name_registry.get_unique_name("concatenate")
        input_var_name_a = input_code_a.strip().split("\n")[-1].split("=")[0].strip()
        input_var_name_b = input_code_b.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code_a}\n{input_code_b}\n{layer_name} = tf.keras.layers.Concatenate(axis={axis})([{input_var_name_a}, {input_var_name_b}])"
        return code


class ReshapeNode(Node):
    title = "Reshape"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Target Shape")]
    init_outputs = [NodeOutputType(label="Result")]

    def update_event(self, inp=-1):
        try:
            print(f"Reshape Node")
            input_val, input_code = self.input(0).payload
            target = self.input(1).payload

            layer = tf.keras.layers.Reshape(target_shape=target)
            result = layer(input_val)

            code = self.generate_code(input_code, target)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("Reshape Node error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, target):
        layer_name = name_registry.get_unique_name("reshape")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Reshape(target_shape={target})({input_var_name})"
        return code
    

class RealTensorNode(Node):
    title = "Real Tensor"
    init_inputs = [NodeInputType(label='Shape (tuple)')]
    init_outputs = [NodeOutputType(label='Real Tensor')]

    def update_event(self, inp=-1):
        try:
            shape_input = self.input(0).payload

            # If input is from another node
            if isinstance(shape_input, tuple) and len(shape_input) == 3:
                shape_val, shape_code, idx = shape_input
            else:
                shape_val = shape_input
                shape_code = f"shape = {shape_val}"
                idx = 0

            if not isinstance(shape_val, (list, tuple)):
                raise ValueError("Shape must be a tuple or list")

            tensor = tf.constant(np.arange(np.prod(shape_val)).reshape(shape_val), dtype=tf.float32)
            print(f"RealTensorNode: Generated tensor with shape {shape_val}")

            code = self.generate_code(shape_code)
            self.set_output_val(0, Data((tensor, code)))

        except Exception as e:
            print("RealTensorNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, shape_code):
        var_name = name_registry.get_unique_name("real")
        full_code = (
            f"{shape_code}\n"
            f"{var_name} = tf.constant(np.arange(np.prod(shape)).reshape(shape), dtype=tf.float32)"
        )
        return full_code

class FlushWrapper:
    def __init__(self, wrapped):
        self._wrapped = wrapped
    def write(self, *args, **kwargs):
        return self._wrapped.write(*args, **kwargs)
    def flush(self):
        pass  # no-op
    def __getattr__(self, attr):
        return getattr(self._wrapped, attr)

class ModelNode(Node):
    title = "Model"
    init_inputs = [NodeInputType(label="Input"),
                   NodeInputType(label="Output")]
    init_outputs = [NodeOutputType(label="Keras Model")]

    def update_event(self, inp=-1):
        try:
            input_tensor, input_code = self.input(0).payload
            output_tensor, output_code = self.input(1).payload
        
            if input_tensor is None or output_tensor is None:
                raise ValueError("Missing input or output tensor.")

            # Patch stdout to prevent flush error
            sys.stdout = FlushWrapper(sys.stdout)

            model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
            code = self.generate_code(input_code, output_code)
            print(f"Successful")
            self.set_output_val(0, Data((model, code)))
        except Exception as e:
            print("[KerasModelNode Error]:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, output_code):
        lines_input = input_code.strip().split('\n')
        lines_output = output_code.strip().split('\n')

        seen = set()
        merged_lines = []

        for line in lines_input + lines_output:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        input_var = lines_input[-1].split('=')[0].strip()
        output_var = lines_output[-1].split('=')[0].strip()
        model_name = name_registry.get_unique_name("model")

        merged_lines.append(
            f"{model_name} = tf.keras.Model(inputs={input_var}, outputs={output_var})"
        )

        return "\n".join(merged_lines)

    
    """def generate_code(self, input_code, output_code, input_idx, output_idx):
        idx = max(input_idx, output_idx)
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()
        output_var_name = output_code.strip().split("\n")[-1].split("=")[0].strip()
        layer_name = name_registry.get_unique_name("add")

        full_code = (
            f"{input_code}\n"
            f"{output_code}\n"
            f"{layer_name} = tf.keras.Model(inputs={input_var_name}, outputs={output_var_name})"
        )
        return full_code, idx + 1"""
    

"""class ModelSummaryNode(Node):
    title = "Model Summary"
    init_inputs = [NodeInputType(label="Keras Model")]
    init_outputs = [NodeOutputType(label="Summary")]

    def update_event(self, inp=-1):
        try:
            model_data = self.input(0)
            if model_data is None or model_data.payload is None:
                raise ValueError("No model provided.")

            model_val, model_code, idx, model_ids = model_data.payload

            if not hasattr(model_val, 'summary'):
                raise TypeError("Provided object is not a Keras model.")

            # Capture the printed summary
            summary_lines = []
            model_val.summary(print_fn=lambda line: summary_lines.append(line))
            summary_text = "\n".join(summary_lines)

            code, new_idx = self.generate_code(model_code, idx)
            new_ids = model_ids.union({id(self)})
            print("Model Summary:\n" + summary_text)
            self.set_output_val(0, Data((summary_text, code, new_idx, new_ids)))
        except Exception as e:
            print("[ModelSummaryNode Error]:", e)
            self.set_output_val(0, Data(("Error generating summary", "", 0)))

    def generate_code(self, model_code, idx):
        lines = model_code.strip().split("\n")
        model_var = lines[-1].split("=")[0].strip() if "=" in lines[-1] else "model"
        new_code = f"{model_code}\n{model_var}.summary()"
        return new_code, idx + 1"""

class ModelSummaryNode(Node):
    title = "Model Summary"
    init_inputs = [NodeInputType(label="Keras Model")]
    init_outputs = [NodeOutputType(label="Summary")]

    def update_event(self, inp=-1):
        try:
            model_data = self.input(0)
            if model_data is None or model_data.payload is None:
                raise ValueError("No model provided.")

            model_val, model_code = model_data.payload

            if not hasattr(model_val, 'summary'):
                raise TypeError("Provided object is not a Keras model.")

            # Capture the summary text
            summary_lines = []
            model_val.summary(print_fn=lambda line: summary_lines.append(line))
            summary_text = "\n".join(summary_lines)

            code = self.generate_code(model_code)

            print("Model Summary:\n" + summary_text)
            print("----------------------------------------")

            self.set_output_val(0, Data((summary_text, code)))
        except Exception as e:
            print("[ModelSummaryNode Error]:", e)
            self.set_output_val(0, Data(("Error generating summary", "")))

    def generate_code(self, model_code):
        lines = model_code.strip().split("\n")
        model_var = lines[-1].split("=")[0].strip() if "=" in lines[-1] else "model"
        new_code = f"{model_code}\n{model_var}.summary()"
        return new_code


"""class ModelCompileNode(Node):
    title = "Model Compile"
    init_inputs = [
        NodeInputType(label="Keras Model"),
        NodeInputType(label="Optimizer"),
        NodeInputType(label="Loss"),
        NodeInputType(label="Metrics"),
    ]
    init_outputs = [NodeOutputType(label="Compiled Model")]

    def update_event(self, inp=-1):
        try:
            print(f"ModelCompile")
            model_val, model_code = self.input(0).payload
            optimizer, opt_code = self.input(1).payload
            loss = self.input(2).payload
            metrics = self.input(3).payload

            model_val.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            code = self.generate_code(model_code, optimizer, loss, metrics)
            print(f"model compile: success")
            self.set_output_val(0, Data((model_val, code)))
        except Exception as e:
            print("[ModelCompileNode Error]:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, model_code, optimizer, loss, metrics):
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()

        if isinstance(metrics, (list, tuple)):
            metrics_strs = [f'"{m}"' for m in metrics]
            metrics_list = "[" + ", ".join(metrics_strs) + "]"
        else:
            metrics_list = f'"{metrics}"'

        compile_code = f"{model_code.strip()}\n{model_var}.compile(optimizer=\"{optimizer}\", loss=\"{loss}\", metrics={metrics_list})"
        return compile_code"""


class ModelCompileNode(Node):
    title = "Model Compile"
    init_inputs = [
        NodeInputType(label="Keras Model"),
        NodeInputType(label="Optimizer"),
        NodeInputType(label="Loss"),
        NodeInputType(label="Metrics"),
    ]
    init_outputs = [NodeOutputType(label="Compiled Model")]

    def update_event(self, inp=-1):
        try:
            print(f"ModelCompile")
            model_val, model_code = self.input(0).payload
            # Get optimizer and its code safely
            optimizer_data = self.input(1).payload
            if isinstance(optimizer_data, tuple):
                optimizer, opt_code = optimizer_data
            else:
                optimizer = optimizer_data
                opt_code = f'"{optimizer}"'  # Fallback

            loss = self.input(2).payload
            metrics = self.input(3).payload

            # Compile the model
            model_val.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            # Generate code
            code = self.generate_code(model_code, opt_code, loss, metrics)
            print(f"Model compile: success")

            self.set_output_val(0, Data((model_val, code)))
        except Exception as e:
            print("[ModelCompileNode Error]:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, model_code, optimizer_code, loss, metrics):
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()

        # Handle metrics formatting
        if isinstance(metrics, (list, tuple)):
            metrics_strs = [f'"{m}"' for m in metrics]
            metrics_list = "[" + ", ".join(metrics_strs) + "]"
        else:
            metrics_list = f'"{metrics}"'

        compile_code = (
            f"{model_code.strip()}\n"
            f"{model_var}.compile(optimizer={optimizer_code}, loss=\"{loss}\", metrics={metrics_list})"
        )
        return compile_code



# Model.evaluate: Returns the loss value & metrics values for the model in test mode.
class ModelEvaluateNode(Node):
    title = "Model Evaluate"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="x (Inputs)"),
                   NodeInputType(label="y (Targets)")]
    
    init_outputs = [NodeOutputType(label="Loss and Metrics")]

    def update_event(self, inp=-1):
        try:
            model_val, model_code = self.input(0).payload
            x_val, x_code, x_idx = self.input(1).payload
            y_val, y_code, y_idx = self.input(2).payload

            results = model_val.evaluate(x_val, y_val, verbose=0)
            code = self.generate_code(model_code, x_code, y_code, x_idx, y_idx)
            self.set_output_val(0, Data((results, code)))
        except Exception as e:
            print("ModelEvaluateNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, model_code, x_code, y_code):
        lines_model = model_code.strip().split("\n")
        lines_x = x_code.strip().split("\n")
        lines_y = y_code.strip().split("\n")

        # Deduplicate all lines
        seen = set()
        merged_lines = []
        for line in lines_model + lines_x + lines_y:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        # Extract variable names
        model_var = lines_model[-1].split("=")[0].strip()
        x_var = lines_x[-1].split("=")[0].strip()
        y_var = lines_y[-1].split("=")[0].strip()

        eval_var = name_registry.get_unique_name("evaluate")
        merged_lines.append(f"{eval_var} = {model_var}.evaluate({x_var}, {y_var}, verbose=0)")

        return "\n".join(merged_lines)
    
    """def generate_code(self, model_code, x_code, y_code):
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()
        x_var = x_code.strip().split("\n")[-1].split("=")[0].strip()
        y_var = y_code.strip().split("\n")[-1].split("=")[0].strip()

        eval_var = name_registry.get_unique_name("evaluate")

        full_code = (
            f"{model_code}\n"
            f"{x_code}\n"
            f"{y_code}\n"
            f"{eval_var} = {model_var}.evaluate({x_var}, {y_var}, verbose=0)"
        )
        return full_code"""
    


# Model.fit: Trains the model for a fixed number of epochs (dataset iterations).
class ModelFitNode(Node):
    title = "Model Fit"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="x"),
                   NodeInputType(label="y"),
                   NodeInputType(label="Epochs"),
                   NodeInputType(label="Batch Size"),
                   NodeInputType(label="Validation Data X(Optional)"),
                   NodeInputType(label="Validation Data Y(Optional)"),
                   NodeInputType(label="Verbose"),
                   NodeInputType(label="Trigger (Button)", type_ = 'exec')]
    
    init_outputs = [NodeOutputType(label="History")]

    def update_event(self, inp=-1):
        try:

            if inp == 8:
                model_val, model_code = self.input(0).payload
                x_val, x_code = self.input(1).payload
                y_val, y_code = self.input(2).payload
                epochs = int(self.input(3).payload)
                batch_size = int(self.input(4).payload)
                #val_data = self.input(5).payload if self.input(5) is not None else None
                #val_data = self.input(6).payload if self.input(6) is not None else None
                verbose = int(self.input(7).payload if self.input(7) is not None else 0)

                val_data = None
                val_code_x, val_code_y = "", ""

                if self.input(5) and self.input(5).payload and self.input(6) and self.input(6).payload:
                    val_data_x, val_code_x = self.input(5).payload
                    val_data_y, val_code_y = self.input(6).payload
                    val_data = (val_data_x, val_data_y)

                history = model_val.fit(x_val, y_val, epochs=epochs, batch_size=batch_size, validation_data=val_data, verbose=verbose)
                code = self.generate_code(model_code, x_code, y_code, epochs, batch_size, val_code_x, val_code_y, verbose)
                self.set_output_val(0, Data((model_val, code)))
        except Exception as e:
            print("ModelFitNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, model_code, x_code, y_code, epochs, batch_size, val_code_x, val_code_y, verbose):
        lines_model = model_code.strip().split("\n")
        lines_x = x_code.strip().split("\n")
        lines_y = y_code.strip().split("\n")
        lines_val_x = val_code_x.strip().split("\n")
        lines_val_y = val_code_y.strip().split("\n")

        seen = set()
        merged_lines = []
        for line in lines_model + lines_x + lines_y + lines_val_x + lines_val_y:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        # Detect model name like "model_0 = ..."
        model_var = "model"
        for line in reversed(lines_model):
            match = re.match(r"(model_\d+)\s*=", line)
            if match:
                model_var = match.group(1)
                break

        hist_var = name_registry.get_unique_name("fit")

        merged_lines.append(
            f"{hist_var} = {model_var}.fit(trainX, trainY, epochs={epochs}, batch_size={batch_size}, validation_data=(testX, testY), verbose={verbose})"
        )

        return "\n".join(merged_lines)


class ModelSaveNode(Node):
    title = "Model Save"
    init_inputs = [
        NodeInputType(label="Model"),
        NodeInputType(label="Filepath"),
    ]
    init_outputs = [
        NodeOutputType(label="Status")
    ]

    def update_event(self, inp=-1):
        try:
            model_val, model_code = self.input(0).payload
            filename = self.input(1).payload or "saved_model.keras"

            # Ensure 'models' folder exists
            os.makedirs("models", exist_ok=True)

            # Join the folder with filename if it's just a filename (no slash or path given)
            if not os.path.dirname(filename):  # no directory in filename
                filepath = os.path.join("models", filename)
            else:
                filepath = filename

            model_val.save(filepath)
            print(f"Model saved to {os.path.abspath(filepath)}")

            code = self.generate_code(model_code, filepath)
            self.set_output_val(0, Data(("Model saved successfully.", code)))
        except Exception as e:
            print("[ModelSaveNode Error]:", e)
            self.set_output_val(0, Data("Model save failed."))

    def generate_code(self, model_code, filepath):
        lines = model_code.strip().split("\n")
        model_var_name = None

        # Find the last variable assignment that matches "model_# ="
        for line in reversed(lines):
            match = re.match(r"(model_\d+)\s*=", line)
            if match:
                model_var_name = match.group(1)
                break

        # Fallback name
        if not model_var_name:
            model_var_name = "model"

        abs_path = os.path.abspath(filepath)

        return f"{model_code}\n{model_var_name}.save(r\"{abs_path}\")"


    """def generate_code(self, model_code, filepath):
        # Try to find a line with model =
        lines = model_code.strip().split("\n")
        model_var_name = None
        for line in reversed(lines):
            if "=" in line:
                model_var_name = line.split("=")[0].strip()
                break

        # Fallback
        if not model_var_name:
            model_var_name = "model"

        abs_path = os.path.abspath(filepath)

        save_code = (
            f"{model_code}\n"
            f"{model_var_name}.save(r\"{abs_path}\")"
        )
        return save_code"""


    """def generate_code(self, model_code, filepath):
        model_var_name = model_code.strip().split("\n")[-1].split("=")[0].strip()
        
        # Default or sanitized path
        if not os.path.dirname(filepath):
            filepath = os.path.join("models", filepath)
        abs_path = os.path.abspath(filepath)

        save_code = (
            f"{model_code}\n"
            f"{model_var_name}.save(r\"{abs_path}\")  # Save model"
        )
        return save_code"""


class GlorotUniformInitializerNode(Node):
    title = "GlorotUniform Initializer"
    init_inputs = [NodeInputType(label="Seed (optional)")]
    init_outputs = [NodeOutputType(label="Initializer")]

    def place_event(self):
        # Called when node is placed in the workspace
        self.update_event()

    def update_event(self, inp=-1):
        try:
            # Cannot just leave out the 'is not None' because if the input is zero, it will not pass too
            if (
                self.input(0) is not None and
                self.input(0).payload is not None and
                self.input(0).payload != ''
            ):

                seed = self.input(0).payload
            else:
                seed = None
            initializer = tf.keras.initializers.GlorotUniform(seed=seed)

            # Code string to use in generate_code
            if seed is not None:
                code = f"tf.keras.initializers.GlorotUniform(seed={seed})"
            else:
                code = "'glorot_uniform'"  # Use string for default case

            self.set_output_val(0, Data((initializer, code)))
        except Exception as e:
            print("[GlorotUniformInitializerNode error]:", e)
            self.set_output_val(0, Data(None))

class HeUniformInitializerNode(Node):
    title = "HeUniform Initializer"
    init_inputs = []
    init_outputs = [NodeOutputType(label="Initializer")]

    def place_event(self):
        # Called when node is placed in the workspace
        self.update_event()

    def update_event(self, inp=-1):
        try:
            initializer = tf.keras.initializers.HeUniform()
            code = f"tf.keras.initializers.HeUniform()"
            print("HeUniform output set successfully:", code)
            self.set_output_val(0, Data((initializer, code)))
        except Exception as e:
            print("[HeUniformInitializerNode error]:", e)
            self.set_output_val(0, Data(None))


# havent do the unpacking of tuple
class SGDOptimizerNode(Node):
    title = "SGD Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.SGD(learning_rate={lr})"
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[SGDOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))

class AdamOptimizerNode(Node):
    title = "Adam Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.Adam(learning_rate={lr})"
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[AdamOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))

class RMSpropOptimizerNode(Node):
    title = "RMSprop Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.RMSprop(learning_rate={lr})"
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[RMSpropOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))


class PrintCodeNode(Node):
    title = "Print Code"
    init_inputs = [NodeInputType(label='Input')]
    init_outputs = []

    def update_event(self, inp=-1):
        payload = self.input(0).payload

        # Reset name registry before new code generation
        name_registry.reset()

        if isinstance(payload, tuple) and len(payload) >= 2:
            code = payload[1]
            print("Generated Code:\n", code)
            print(f"---------------------------------------")
        else:
            print("Error: No code string found in input payload")

"""class DummyImageTrainTestNode(Node):
    title = "Dummy Train/Test Data"
    init_inputs = [NodeInputType(label="Num Samples (total)")]
    init_outputs = [
        NodeOutputType(label="x_train"),
        NodeOutputType(label="y_train"),
        NodeOutputType(label="x_test"),
        NodeOutputType(label="y_test"),
    ]

    def update_event(self, inp=-1):
        try:
            num_samples = self.input(0).payload or 100
            num_classes = 10
            x = np.random.rand(num_samples, 224, 224, 3).astype('float32')
            y = np.random.randint(0, num_classes, size=(num_samples,))
            y = tf.keras.utils.to_categorical(y, num_classes)

            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

            self.set_output_val(0, Data((x_train, "x_train = ...", 0)))
            self.set_output_val(1, Data((y_train, "y_train = ...", 0)))
            self.set_output_val(2, Data((x_test, "x_test = ...", 0)))
            self.set_output_val(3, Data((y_test, "y_test = ...", 0)))
        except Exception as e:
            print("[DummyImageTrainTestNode Error]:", e)
            for i in range(4):
                self.set_output_val(i, Data((None, "", 0)))"""

class MNISTLoaderNode(Node):
    title="MNIST Loader"
    init_inputs = []
    init_outputs = [
        NodeOutputType(label="trainX"),
        NodeOutputType(label="trainY"),
        NodeOutputType(label="testX"),
        NodeOutputType(label="testY"),
    ]

    def place_event(self):
        # Called when node is placed in the workspace
        self.update_event()

    def update_event(self, inp=-1):
        try:
            # Load dataset
            (trainX, trainY), (testX, testY) = tf.keras.datasets.mnist.load_data()

            # Reshape and normalize
            trainX = trainX.reshape((trainX.shape[0], 28 * 28, -1)) / 255.0
            testX = testX.reshape((testX.shape[0], 28 * 28, -1)) / 255.0

            # One-hot encode the labels
            trainY = tf.keras.utils.to_categorical(trainY)
            testY = tf.keras.utils.to_categorical(testY)

            # Code representation
            code = (
                "(trainX, trainY), (testX, testY) = mnist.load_data()\n"
                "trainX = trainX.reshape((trainX.shape[0], 28*28, -1)) / 255.0\n"
                "testX = testX.reshape((testX.shape[0], 28*28, -1)) / 255.0\n"
                "trainY = to_categorical(trainY)\n"
                "testY = to_categorical(testY)\n"
            )

            # Set outputs (each paired with code)
            self.set_output_val(0, Data((trainX, code)))
            self.set_output_val(1, Data((trainY, code)))
            self.set_output_val(2, Data((testX, code)))
            self.set_output_val(3, Data((testY, code)))

            print("MNIST data loaded and processed.")

        except Exception as e:
            print("MNISTLoaderNode error:", e)
            self.set_output_val(0, Data((None, "")))
            self.set_output_val(1, Data((None, "")))
            self.set_output_val(2, Data((None, "")))
            self.set_output_val(3, Data((None, "")))


export_nodes([
    # list your node classes here
    DenseNode,
    DropoutNode,
    AddNode,
    InputNode,
    ReLUNode,
    ZeroPadding2D,
    BatchNormalization,
    FlattenNode,
    Conv2DNode,
    AveragePooling2DNode,
    MaxPooling2DNode,
    GlobalAveragePooling2DNode,
    GlobalMaxPooling2DNode,
    ConcatenateNode,
    ReshapeNode,
    RealTensorNode,
    ModelNode,
    ModelSummaryNode,
    ModelCompileNode,
    ModelEvaluateNode,
    ModelFitNode,
    ModelSaveNode,
    GlorotUniformInitializerNode,
    HeUniformInitializerNode,
    SGDOptimizerNode,
    AdamOptimizerNode,
    RMSpropOptimizerNode,
    PrintCodeNode,
    MNISTLoaderNode,
])


@on_gui_load
def load_gui():
    # import gui sources here only
    from . import gui