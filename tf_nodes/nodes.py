from ryven.node_env import *
from random import random
import tensorflow as tf
from random import random
import logging
import ast
import numpy as np
import sys
import ryven
from qtpy.QtWidgets import QLineEdit

logger = logging.getLogger(__name__)


# Global variable to count the names
input_count = 0

# your node definitions go here

class RandNode(Node):
    """Generates scaled random float values"""

    title = 'Rand'
    tags = ['random', 'numbers']
    init_inputs = [NodeInputType()]
    init_outputs = [NodeOutputType()]

    def update_event(self, inp=-1):
        self.set_output_val(0, 
            Data(random() * self.input(0).payload)
        )

class PrintNode(Node):
    title = 'Print Node'
    init_inputs = [NodeInputType()]

    def update_event(self, inp=-1):
        print(f"Printing result: ")
        print(self.input(0).payload)
        print(f"----------------------------------------")

class InputNode(Node):
    title = "Input Node"
    init_inputs = [NodeInputType(label='Shape')]
    init_outputs = [NodeOutputType(label='Tensor')]

    def update_event(self, inp=-1):
        logger.info("InputNode:")
        shape = self.input(0).payload
        try:
            tensor = tf.keras.Input(shape=shape)
            code = f"input_layer_{input_count} = tf.keras.Input(shape={shape})"
            print(f"Successful")
            print(f"----------------------------------------")
            self.set_output_val(0, Data((tensor, code, 1)))
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
            input_valA, input_codeA, idxA = self.input(0).payload
            input_valB, input_codeB, idxB = self.input(1).payload

            print("Attempting to add Tensor 1 and Tensor 2...")
            layer = tf.keras.layers.Add()
            result = layer([input_valA, input_valB])
            
            # Generate code
            code, new_idx = self.generate_code(input_codeA, input_codeB, idxA, idxB)

            print("Successful. Result shape:", result.shape)
            print("----------------------------------------")
            self.set_output_val(0, Data((result, code, new_idx)))

        except Exception as e:
            print("AddNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_codeA, input_codeB, idxA, idxB):
        # Name
        max_index = max(idxA, idxB)
        layer_name = f"add_{max_index}"
        input_var_nameA = input_codeA.strip().split("\n")[-1].split("=")[0].strip() # get the previous name
        input_var_nameB = input_codeB.strip().split("\n")[-1].split("=")[0].strip()

        code = (
            f"{input_codeA}\n{input_codeB}\n"
            f"{layer_name} = tf.keras.layers.Add()([{input_var_nameA}, {input_var_nameB}])"
        )
        return code, (max_index + 1)

class PrintShapeNode(Node):
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
            print(f"----------------------------------------")

# Input: activation units
class DenseNode(Node):
    title = 'Dense Node'
    init_inputs = [NodeInputType(label='Input Tensor'),
                   NodeInputType(label='Units'),
                   NodeInputType(label='Activation (Optional)')]
    init_outputs = [NodeOutputType(label='Output Tensor')]

    def update_event(self, inp=-1):
        try:
            print(f"DenseNode:\n")
            #input_tensor = self.input(0).payload
            input_val, input_code, idx = self.input(0).payload
            units = self.input(1).payload
            activation = self.input(2).payload if self.input(2) and self.input(2).payload is not None else None

            layer = tf.keras.layers.Dense(units=units, activation=activation)
            result = layer(input_val)
            print(f"Sucessful")
            print(f"----------------------------------------")

            code, new_idx = self.generate_code(input_code, idx, activation, units)
            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("DenseNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, activation, units):
        layer_name = f"dense_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        if activation is None:
            code = f"{input_code}\n{layer_name} = tf.keras.layers.Dense({units})({input_var_name})"
        else:
            activation_code = f"'{activation}'" if isinstance(activation, str) else activation
            code = f"{input_code}\n{layer_name} = tf.keras.layers.Dense({units}, activation={activation_code})({input_var_name})"

        return code, (idx + 1)
    

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
            #x = self.input(0).payload
            input_val, input_code, idx = self.input(0).payload
            rate = self.input(1).payload

            layer = tf.keras.layers.Dropout(rate=rate)
            result = layer(input_val, training=True)
            print(f"success")

            code, new_idx = self.generate_code(input_code, idx, rate)
            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("DropoutNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, rate):
        layer_name = f"dropout_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.Dropout({rate})({input_var_name})"
        return code, (idx + 1)


# Check if you can input a list into 
class ReLUNode(Node):
    title = "ReLU Activation"
    init_inputs = [NodeInputType(label='Input Tensor')]
    init_outputs = [NodeOutputType(label='Activated Tensor')]

    def update_event(self, inp=-1):
        try:
            input_val, input_code, idx = self.input(0).payload
            result = tf.keras.activations.relu(input_val)

            code, new_idx = self.generate_code(input_code, idx)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("ReLUNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx):
        layer_name = f"relu_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.activations.relu({input_var_name})"
        return code, (idx + 1)


class ZeroPadding2D(Node):
    title = "ZeroPadding2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label="Padded Tensor")]

    def update_event(self, inp=-1):
        try:
            print(f"ZeroPadding2D Node")
            input_val, input_code, idx = self.input(0).payload
            padding = self.input(1).payload

            # Only pass through if input is a Keras symbolic tensor
            if not hasattr(input_val, '_keras_history'):
                raise ValueError("Input is not a Keras symbolic tensor")

            layer = tf.keras.layers.ZeroPadding2D(padding=padding)
            result = layer(input_val)
            print(f"Successful")

            code, new_idx = self.generate_code(input_code, idx, padding)
            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("ZeroPaddingNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, padding):
        layer_name = f"zero_padding2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.ZeroPadding2D(padding={padding})({input_var_name})"
        return code, (idx + 1)


"""def normalize_tensor_input(x):
    # Case 1: Input is a string (from a Val node)
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
            x = np.array(x, dtype=np.float32)
        except Exception as e:
            raise ValueError(f"Failed to parse string input: {e}")
    
    # Case 2: Input is a list or tuple
    elif isinstance(x, (list, tuple)):
        x = np.array(x, dtype=np.float32)
    
    # Case 3: Input is a NumPy array
    elif isinstance(x, np.ndarray):
        x = x.astype(np.float32)
    
    # Case 4: Input is already a Tensor
    elif isinstance(x, tf.Tensor):
        return x
    
    else:
        raise ValueError(f"Unsupported input type: {type(x)}")
    
    # Convert to Tensor if not already
    return tf.convert_to_tensor(x)"""

class BatchNormalization(Node):
    title = "BatchNormalization"
    init_inputs = [NodeInputType(label="Input Tensor"),
                   NodeInputType(label="Axis")]
    init_outputs = [NodeOutputType(label="Normalized Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code, idx = self.input(0).payload
            axis = self.input(1).payload

            layer = tf.keras.layers.BatchNormalization(axis=axis)
            result = layer(input_val)
            print(f"Successful")

            code, new_idx = self.generate_code(input_code, idx, axis)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("BatchNormalizationNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, axis):
        layer_name = f"batch_normalization_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.BatchNormalization(axis={axis})({input_var_name})"
        return code, (idx + 1)
    

class FlattenNode(Node):
    title = "Flatten"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Flattened Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code, idx = self.input(0).payload
            layer = tf.keras.layers.Flatten()
            result = layer(input_val)

            code, new_idx = self.generate_code(input_code, idx)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("FlattenNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx):
        layer_name = f"flatten_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.Flatten()({input_var_name})"
        return code, (idx + 1)


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
            input_val, input_code, idx = self.input(0).payload
            filters = self.input(1).payload
            kernel_size = self.input(2).payload
            
            # Optional inputs with defaults
            strides = self.input(3).payload if self.input(3) and self.input(3).payload is not None else (1, 1)
            padding = self.input(4).payload if self.input(4) and self.input(4).payload is not None else 'valid'
            kernel_initializer = self.input(5).payload if self.input(5) and self.input(5).payload is not None else 'glorot_uniform'

            layer = tf.keras.layers.Conv2D(filters=filters, 
                                           kernel_size=kernel_size, 
                                           strides=strides, padding=padding, 
                                           kernel_initializer=kernel_initializer)
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code, new_idx = self.generate_code(input_code, idx, filters, kernel_size, strides, padding, kernel_initializer)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("[Conv2DNode error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, filters, kernel_size, strides, padding, kernel_initializer):
        layer_name = f"conv2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, strides={strides}, padding='{padding}', kernel_initializer={kernel_initializer})({input_var_name})"
        return code, (idx + 1)


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
            input_val, input_code, idx = self.input(0).payload
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

            code, new_idx = self.generate_code(input_code, idx, pool_size, strides, padding)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("[AveragePooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))
    
    def generate_code(self, input_code, idx, pool_size, strides, padding):
        layer_name = f"average_pooling2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.AveragePooling2D(pool_size={pool_size}, strides={strides}, padding='{padding}')({input_var_name})"
        return code, (idx + 1)


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
            input_val, input_code, idx = self.input(0).payload
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

            code, new_idx = self.generate_code(input_code, idx, pool_size, strides, padding)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("[MaxPooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, pool_size, strides, padding):
        layer_name = f"max_pooling2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.MaxPooling2D(pool_size={pool_size}, strides={strides}, padding={padding})({input_var_name})"
        return code, (idx + 1)


class GlobalMaxPooling2DNode(Node):
    title = "GlobalMaxPooling2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code, idx = self.input(0).payload
            print(f"GlobalMaxPooling2D")

            layer = tf.keras.layers.GlobalMaxPooling2D()
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code, new_idx = self.generate_code(input_code, idx)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("[GlobalMaxPooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx):
        layer_name = f"global_max_pooling2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.GlobalMaxPooling2D()({input_var_name})"
        return code, (idx + 1)
    

class GlobalAveragePooling2DNode(Node):
    title = "GlobalAveragePooling2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            input_val, input_code, idx = self.input(0).payload
            print(f"GlobalAveragePooling2D Node")

            layer = tf.keras.layers.GlobalAveragePooling2D()
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code, new_idx = self.generate_code(input_code, idx)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("[GlobalAveragePooling2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx):
        layer_name = f"global_average_pooling2d_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.GlobalAveragePooling2D()({input_var_name})"
        return code, (idx + 1)


class ConcatenateNode(Node):
    title = 'Concatenate'
    init_inputs = [NodeInputType('Tensor 1'),
                   NodeInputType('Tensor 2'), 
                   NodeInputType("Axis")]
    init_outputs = [NodeOutputType('Result')]

    def update_event(self, inp=-1):
        try:
            print(f"Concatenate")
            input_val_a, input_code_a, idx_a = self.input(0).payload
            input_val_b, input_code_b, idx_b = self.input(1).payload
            axis = self.input(2).payload

            result = tf.keras.layers.concatenate([input_val_a, input_val_b], axis=axis)

            code, new_idx = self.generate_code(input_code_a, input_code_b, idx_a, idx_b, axis)

            self.set_output_val(0, Data((result, code, new_idx)))
        except Exception as e:
            print("ConcatenateNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code_a, input_code_b, idx_a, idx_b, axis):
        idx = max(idx_a, idx_b)
        layer_name = f"concatenate_{idx}"
        input_var_name_a = input_code_a.strip().split("\n")[-1].split("=")[0].strip()
        input_var_name_b = input_code_b.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code_a}\n{input_code_b}\n{layer_name} = tf.keras.layers.Concatenate(axis={axis})([{input_var_name_a}, {input_var_name_b}])"
        return code, (idx + 1)


class ReshapeNode(Node):
    title = "Reshape"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Target Shape")]
    init_outputs = [NodeOutputType(label="Result")]

    def update_event(self, inp=-1):
        try:
            print(f"Reshape Node")
            input_val, input_code, idx = self.input(0).payload
            target = self.input(1).payload

            layer = tf.keras.layers.Reshape(target_shape=target)
            result = layer(input_val)

            code, new_idx = self.generate_code(input_code, idx, target)

            self.set_output_val(0, Data((result, code, idx)))
        except Exception as e:
            print("Reshape Node error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, input_code, idx, target):
        layer_name = f"reshape_{idx}"
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code}\n{layer_name} = tf.keras.layers.Reshape(target_shape={target})({input_var_name})"
        return code, (idx + 1)
    

"""class RealTensorNode(Node):
    title = "Real Tensor"
    init_inputs = [NodeInputType(label='Shape (tuple)')]
    init_outputs = [NodeOutputType(label='Real Tensor')]

    def update_event(self, inp=-1):
        try:
            shape = self.input(0).payload

            if not isinstance(shape, (list, tuple)):
                raise ValueError("Shape must be a tuple or list")

            # Generate test data (customize as needed)
            tensor = tf.constant(np.arange(np.prod(shape)).reshape(shape), dtype=tf.float32)
            print(f"RealTensorNode: Generated tensor with shape {shape}")
            self.set_output_val(0, Data(tensor))
        except Exception as e:
            print("RealTensorNode error:", e)
            self.set_output_val(0, Data(None))"""


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

            code, new_idx = self.generate_code(shape_code, idx)
            self.set_output_val(0, Data((tensor, code, new_idx)))

        except Exception as e:
            print("RealTensorNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, shape_code, idx):
        var_name = f"real_tensor_{idx}"
        full_code = (
            f"{shape_code}\n"
            f"{var_name} = tf.constant(np.arange(np.prod(shape)).reshape(shape), dtype=tf.float32)"
        )
        return full_code, idx + 1


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
            input_tensor, input_code, input_idx = self.input(0).payload
            output_tensor, output_code, output_idx = self.input(1).payload
        
            if input_tensor is None or output_tensor is None:
                raise ValueError("Missing input or output tensor.")

            # Patch stdout to prevent flush error
            sys.stdout = FlushWrapper(sys.stdout)

            model = tf.keras.Model(inputs=input_tensor, outputs=output_tensor)
            code, new_idx = self.generate_code(input_code, output_code, input_idx, output_idx)

            self.set_output_val(0, Data((model, code, new_idx)))

        except Exception as e:
            print("[KerasModelNode Error]:", e)
            self.set_output_val(0, Data((None, "", 0)))
    
    def generate_code(self, input_code, output_code, input_idx, output_idx):
        idx = max(input_idx, output_idx)
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()
        output_var_name = output_code.strip().split("\n")[-1].split("=")[0].strip()
        model_name = f"model_{idx}"

        full_code = (
            f"{input_code}\n"
            f"{output_code}\n"
            f"{model_name} = tf.keras.Model(inputs={input_var_name}, outputs={output_var_name})"
        )
        return full_code, idx + 1
    
"""class ModelNode(Node):
    title = "Model"
    init_inputs = [
        NodeInputType(label="Input"),
        NodeInputType(label="Output")
    ]
    init_outputs = [NodeOutputType(label="Keras Model")]

    def update_event(self, inp=-1):
        try:
            input_tensor = self.input(0)
            output_tensor = self.input(1)

            if input_tensor is None or output_tensor is None:
                raise ValueError("Input or output tensor is missing.")

            input_val = input_tensor.payload
            output_val = output_tensor.payload

            if input_val is None or output_val is None:
                raise ValueError("One or both tensor payloads are None.")

            from tensorflow.keras import Model
            model = Model(inputs=input_val, outputs=output_val)

            # Generate code
            code = f"model_{self.id} = tf.keras.Model(inputs=input_{self.id}, outputs=output_{self.id})"
            model_package = (model, code, self.id)

            self.set_output_val(0, Data(model_package))
        except Exception as e:
            print("[ModelNode Error]:", e)
            self.set_output_val(0, Data((None, "", self.id)))"""


"""class ModelNode(Node):
    title = "Model"
    init_inputs = [
        NodeInputType(label="Input"),
        NodeInputType(label="Output")
    ]
    init_outputs = [NodeOutputType(label="Keras Model")]

    def update_event(self, inp=-1):
        try:
            input_data = self.input(0)
            output_data = self.input(1)

            if input_data is None or output_data is None:
                raise ValueError("Input or output is missing.")

            # Unpack payloads: (tensor, code, idx)
            input_tensor, input_code, input_idx = input_data.payload
            output_tensor, output_code, output_idx = output_data.payload

            if input_tensor is None or output_tensor is None:
                raise ValueError("One or both tensor values are None.")

            from tensorflow.keras import Model
            model = Model(inputs=input_tensor, outputs=output_tensor)

            # Generate code
            code = (
                input_code + "\n" +
                output_code + "\n" +
                f"model_{self.id} = tf.keras.Model(inputs=input_{self.id}, outputs=output_{self.id})"
            )
            model_package = (model, code, self.id)

            self.set_output_val(0, Data(model_package))
        except Exception as e:
            print("[ModelNode Error]:", e)
            self.set_output_val(0, Data((None, "", self.id)))"""



    

"""class ModelSummaryNode(Node):
    title = "Model Summary"
    init_inputs = [NodeInputType(label="Keras Model")]
    init_outputs = [NodeOutputType(label="Summary")]

    def update_event(self, inp=-1):
        try:
            model_val, model_code, idx = self.input(0).payload
            
            if model_val is None:
                raise ValueError("No model provided.")

            # Capture the printed summary
            summary_lines = []
            model_val.summary(print_fn=lambda line: summary_lines.append(line))
            summary_text = "\n".join(summary_lines)

            code, new_idx = self.generate_code(model_code, idx)

            print("Model Summary:\n" + summary_text)
            self.set_output_val(0, Data((summary_text, code, new_idx)))
        except Exception as e:
            print("[ModelSummaryNode Error]:", e)
            self.set_output_val(0, Data(("Error generating summary", "", 0)))

    def generate_code(self, model_code, idx):
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()
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

            model_val, model_code, idx = model_data.payload

            if not hasattr(model_val, 'summary'):
                raise TypeError("Provided object is not a Keras model.")

            # Capture the printed summary
            summary_lines = []
            model_val.summary(print_fn=lambda line: summary_lines.append(line))
            summary_text = "\n".join(summary_lines)

            code, new_idx = self.generate_code(model_code, idx)

            print("Model Summary:\n" + summary_text)
            self.set_output_val(0, Data((summary_text, code, new_idx)))
        except Exception as e:
            print("[ModelSummaryNode Error]:", e)
            self.set_output_val(0, Data(("Error generating summary", "", 0)))

    def generate_code(self, model_code, idx):
        lines = model_code.strip().split("\n")
        model_var = lines[-1].split("=")[0].strip() if "=" in lines[-1] else "model"
        new_code = f"{model_code}\n{model_var}.summary()"
        return new_code, idx + 1


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
            model_val, model_code, idx = self.input(0).payload
            optimizer = self.input(1).payload
            loss = self.input(2).payload
            metrics = self.input(3).payload

            model_val.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            code, new_idx = self.generate_code(model_code, idx, optimizer, loss, metrics)
            self.set_output_val(0, Data((model_val, code, new_idx)))
        except Exception as e:
            print("[ModelCompileNode Error]:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, model_code, idx, optimizer, loss, metrics):
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()

        if isinstance(metrics, (list, tuple)):
            metrics_strs = [f'"{m}"' for m in metrics]
            metrics_list = "[" + ", ".join(metrics_strs) + "]"
        else:
            metrics_list = f'"{metrics}"'

        compile_code = f"{model_code}\n{model_var}.compile(optimizer=\"{optimizer}\", loss=\"{loss}\", metrics={metrics_list})"
        return compile_code, idx + 1


# Model.evaluate: Returns the loss value & metrics values for the model in test mode.
class ModelEvaluateNode(Node):
    title = "Model Evaluate"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="x (Inputs)"),
                   NodeInputType(label="y (Targets)")]
    
    init_outputs = [NodeOutputType(label="Loss and Metrics")]

    def update_event(self, inp=-1):
        try:
            (model_val, model_code, model_idx) = self.input(0).payload
            (x_val, x_code, x_idx) = self.input(1).payload
            (y_val, y_code, y_idx) = self.input(2).payload

            results = model_val.evaluate(x_val, y_val, verbose=0)
            code, new_idx = self.generate_code(model_code, x_code, y_code, model_idx, x_idx, y_idx)

            self.set_output_val(0, Data((results, code, new_idx)))
        except Exception as e:
            print("ModelEvaluateNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))
    
    def generate_code(self, model_code, x_code, y_code, model_idx, x_idx, y_idx):
        idx = max(model_idx, x_idx, y_idx)
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()
        x_var = x_code.strip().split("\n")[-1].split("=")[0].strip()
        y_var = y_code.strip().split("\n")[-1].split("=")[0].strip()

        eval_var = f"eval_results_{idx}"

        full_code = (
            f"{model_code}\n"
            f"{x_code}\n"
            f"{y_code}\n"
            f"{eval_var} = {model_var}.evaluate({x_var}, {y_var}, verbose=0)"
        )
        return full_code, idx + 1

# Model.fit: Trains the model for a fixed number of epochs (dataset iterations).
class ModelFitNode(Node):
    title = "Model Fit"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="x (Inputs)"),
                   NodeInputType(label="y (Targets)"),
                   NodeInputType(label="Epochs"),
                   NodeInputType(label="Batch Size")]
    
    init_outputs = [NodeOutputType(label="History")]

    def update_event(self, inp=-1):
        try:
            model_val, model_code, model_idx = self.input(0).payload
            x_val, x_code, x_idx = self.input(1).payload
            y_val, y_code, y_idx = self.input(2).payload
            epochs = self.input(3).payload
            batch_size = self.input(4).payload

            history = model_val.fit(x_val, y_val, epochs=epochs, batch_size=batch_size, verbose=0)
            code, new_idx = self.generate_code(model_code, x_code, y_code, model_idx, x_idx, y_idx, epochs, batch_size)
            self.set_output_val(0, Data((history, code, new_idx)))
        except Exception as e:
            print("ModelFitNode error:", e)
            self.set_output_val(0, Data((None, "", 0)))

    def generate_code(self, model_code, x_code, y_code, model_idx, x_idx, y_idx, epochs, batch_size):
        idx = max(model_idx, x_idx, y_idx)
        model_var = model_code.strip().split("\n")[-1].split("=")[0].strip()
        x_var = x_code.strip().split("\n")[-1].split("=")[0].strip()
        y_var = y_code.strip().split("\n")[-1].split("=")[0].strip()
        hist_var = f"history_{idx}"

        full_code = (
            f"{model_code}\n"
            f"{x_code}\n"
            f"{y_code}\n"
            f"{hist_var} = {model_var}.fit({x_var}, {y_var}, epochs={epochs}, batch_size={batch_size}, verbose=0)"
        )
        return full_code, idx + 1

# Saves a model as a .keras file
class ModelSaveNode(Node):
    title = "Model Save"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="Filepath")]
    
    init_outputs = [NodeOutputType(label="Status")]

    def update_event(self, inp=-1):
        try:
            model = self.input(0).payload
            filepath = self.input(1).payload
            model.save(filepath)
            self.set_output_val(0, Data("Model saved successfully."))
        except Exception as e:
            print("ModelSaveNode error:", e)
            self.set_output_val(0, Data("Model save failed."))

class GlorotUniformInitializerNode(Node):
    title = "GlorotUniform Initializer"
    init_inputs = []
    init_outputs = [NodeOutputType(label="Initializer")]

    def update_event(self, inp=-1):
        try:
            initializer = tf.keras.initializers.GlorotUniform()
            self.set_output_val(0, Data(initializer))
        except Exception as e:
            print("[GlorotUniformInitializerNode error]:", e)
            self.set_output_val(0, Data(None))

class HeUniformInitializerNode(Node):
    title = "HeUniform Initializer"
    init_inputs = []
    init_outputs = [NodeOutputType(label="Initializer")]

    def update_event(self, inp=-1):
        try:
            initializer = tf.keras.initializers.HeUniform()
            self.set_output_val(0, Data(initializer))
        except Exception as e:
            print("[HeUniformInitializerNode error]:", e)
            self.set_output_val(0, Data(None))

class SGDOptimizerNode(Node):
    title = "SGD Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
            self.set_output_val(0, Data(optimizer))
        except Exception as e:
            print("[SGDOptimizerNode error]:", e)
            self.set_output_val(0, Data(None))

class AdamOptimizerNode(Node):
    title = "Adam Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.set_output_val(0, Data(optimizer))
        except Exception as e:
            print("[AdamOptimizerNode error]:", e)
            self.set_output_val(0, Data(None))

class RMSpropOptimizerNode(Node):
    title = "RMSprop Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        try:
            lr = self.input(0).payload
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            self.set_output_val(0, Data(optimizer))
        except Exception as e:
            print("[RMSpropOptimizerNode error]:", e)
            self.set_output_val(0, Data(None))


class PrintCodeNode(Node):
    title = "Print Code"
    init_inputs = [NodeInputType(label='Input')]
    init_outputs = []

    def update_event(self, inp=-1):
        payload = self.input(0).payload
        if isinstance(payload, tuple) and len(payload) == 3:
            code = payload[1]
            print("Generated Code:\n", code)
            print(f"---------------------------------------")
        else:
            print("Error: No code string found in input payload")



export_nodes([
    # list your node classes here
    DenseNode,
    DropoutNode,
    AddNode,
    InputNode,
    ReLUNode,
    RandNode,
    ZeroPadding2D,
    BatchNormalization,
    FlattenNode,
    PrintNode,
    PrintShapeNode,
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
])


@on_gui_load
def load_gui():
    # import gui sources here only
    from . import gui