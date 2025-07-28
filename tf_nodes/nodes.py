from ryven.node_env import *
import tensorflow as tf
import sys
import os
import re


class NameRegistry():
    """A class to generate and manage unique variable names"""

    def __init__(self):
        """Initializes the registry with empty name sets and counters."""
        self.used_names = set()
        self.name_counts = {}

    def reset(self):
        """Clears all stored names and counters."""
        self.used_names.clear()
        self.name_counts.clear()

    def get_unique_name(self, base):
        """
        Generates a unique variable name based on the base.

        Args:
            base (str): The prefix for the variable name (e.g., 'dense').

        Returns:
            str: A unique name like 'dense_0', 'dense_1', etc.
        """
        count = self.name_counts.get(base, 0)
        name = f"{base}_{count}"
        self.name_counts[base] = count + 1
        self.used_names.add(name)
        return name

    def has_name(self, name):
        return name in self.used_names


# Initialise an instance
name_registry = NameRegistry()


# Nodes
class InputNode(Node):
    """
    A Ryven node that creates a Keras Input layer with a specified shape.

    This node takes a shape (e.g. (224, 224, 3)) and generates a Keras Input tensor.
    It also outputs the equivalent Python code to recreate the layer.

    Inputs:
        Shape (tuple): The shape of the input tensor (excluding batch size).

    Outputs:
        Tensor (Data): A tuple containing:
            - The created tf.keras.Input tensor.
            - A string of Python code to define the layer.
    """
    
    title = "Input Node"
    init_inputs = [NodeInputType(label='Shape')]
    init_outputs = [NodeOutputType(label='Tensor')]

    def update_event(self, inp=-1):
        """
        Creates the Keras Input layer when the shape input changes.

        Reads the shape input, creates the Input tensor, and outputs both
        the tensor and the code to define it.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((tensor, code)) if successful.
                       Data((None, "")) if an error occurs.
        """
        
        shape = self.input(0).payload
        try:
            tensor = tf.keras.Input(shape=shape)
            layer_name = name_registry.get_unique_name("input")
            code = f"import tensorflow as tf\n{layer_name} = tf.keras.Input(shape={shape})"
            print(f"InputNode successful\n")
            self.set_output_val(0, Data((tensor, code)))
        except Exception as e:
            print("InputNode error:", e)
            self.set_output_val(0, Data((None, "")))


class AddNode(Node):
    """
    A Ryven node that adds two input tensors using Keras Add layer.

    Inputs:
        Tensor 1 (Data): First input tensor and its generated code.
        Tensor 2 (Data): Second input tensor and its generated code.

    Outputs:
        Result (Data): A tuple containing:
            - The result of tf.keras.layers.Add()([Tensor 1, Tensor 2]).
            - A combined Python code string to recreate the operation.
    """
    
    title = "Add Node"
    init_inputs = [
        NodeInputType(label='Tensor 1'),
        NodeInputType(label='Tensor 2')
    ]
    init_outputs = [NodeOutputType(label='Result')]

    def update_event(self, inp=-1):
        """
        Combines two tensors using the Keras Add layer and outputs the result.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((added_tensor, generated_code)).
                       If an error occurs, sets output to (None, "").
        """

        print(f"AddNode:\n")
        try:
            input_valA, input_codeA = self.input(0).payload
            input_valB, input_codeB = self.input(1).payload

            layer = tf.keras.layers.Add()
            result = layer([input_valA, input_valB])
            
            code = self.generate_code(input_codeA, input_codeB)

            print(f"AddNode successful\n")
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("AddNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_codeA, input_codeB):
        """
        Merges input code snippets and generates the Add layer code.

        Args:
            input_codeA (str): Python code for the first input tensor.
            input_codeB (str): Python code for the second input tensor.

        Returns:
            str: Combined code with a unique Add layer operation.
        """

        lines_A = input_codeA.strip().split('\n')
        lines_B = input_codeB.strip().split('\n')

        # Ensures no duplicates
        seen = set()
        merged_lines = []
        for line in lines_A + lines_B:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        # Get the variable names of the 2 input tensor
        varA = lines_A[-1].split('=')[0].strip()
        varB = lines_B[-1].split('=')[0].strip()

        layer_name = name_registry.get_unique_name("add")
        merged_lines.append(f"{layer_name} = tf.keras.layers.Add()([{varA}, {varB}])")

        return "\n".join(merged_lines)


class DenseNode(Node):
    """
    A Ryven node that applies a Keras Dense layer.

    Inputs:
        Input Tensor (Data): The input tensor and its code.
        Units (int): Number of neurons in the Dense layer.
        Activation (str or Data, optional): A tuple of (activation_name, code string), 
                                          or a raw string if entered manually.
        Kernel Initializer (Data, optional): Initializer object and its code.

    Outputs:
        Output Tensor (Data): A tuple containing:
            - The resulting tensor after applying the Dense layer.
            - The generated Python code for this layer.
    """
    
    title = 'Dense Node'
    init_inputs = [NodeInputType(label='Input Tensor'),
                   NodeInputType(label='Units'),
                   NodeInputType(label='Activation (Optional)'),
                   NodeInputType(label='Kernel Initializer (Optional)')]
    init_outputs = [NodeOutputType(label='Output Tensor')]

    def update_event(self, inp=-1):
        """
        Applies a Dense layer to the input tensor and outputs the result.

        Args:
            inp (int): Index of the updated input. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
        try:
            print(f"DenseNode:\n")
            input_val, input_code = self.input(0).payload
            units = self.input(1).payload
            activation = self.input(2).payload if self.input(2) and self.input(2).payload not in [None, ""] else None

            # Default initializer values
            kernel_initializer = 'glorot_uniform'
            kernel_initializer_code = "'glorot_uniform'"

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
            print(f"Sucessful\n")

            code = self.generate_code(input_code, activation, units, kernel_initializer_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("DenseNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, activation, units, kernel_initializer_code):
        """
        Generates Python code for the Dense layer.

        Args:
            input_code (str): Code from the input tensor.
            activation (str): Activation function name or None.
            units (int): Number of neurons.
            kernel_initializer_code (str): Code string for the initializer.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Dense layer operation.
        """
        layer_name = name_registry.get_unique_name("dense")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()
        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Dense({units}, activation='{activation}', kernel_initializer={kernel_initializer_code})({input_var_name})"

        return code
    

class DropoutNode(Node):
    """
    A Ryven node that applies a Keras Dropout layer to the input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code.
        Rate (float): Dropout rate between 0 and 1.

    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying dropout.
            - The generated Python code for this layer.
    """

    title = "Dropout"
    init_inputs = [
        NodeInputType(label="Input Tensor"),
        NodeInputType(label="Rate")
    ]
    init_outputs = [NodeOutputType(label="Result")]

    def update_event(self, inp=-1):
        """
        Applies the Dropout layer with the given rate to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
        try:
            input_val, input_code = self.input(0).payload
            rate = self.input(1).payload

            layer = tf.keras.layers.Dropout(rate=rate)
            result = layer(input_val, training=True)
            print(f"DropoutNode successful\n")

            code = self.generate_code(input_code, rate)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("DropoutNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, rate):
        """
        Generates Python code for the Dropout layer.

        Args:
            input_code (str): Code for the input tensor.
            rate (float): Dropout rate used in the layer.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Dropout layer operation.

        """
        layer_name = name_registry.get_unique_name("dropout")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Dropout({rate})({input_var_name})"
        return code


class ReLUNode(Node):
    """
    A Ryven node that apples ReLU activation to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying Relu activation.
            - The generated Python code for this layer.
    """
    title = "ReLU Activation"
    init_inputs = [NodeInputType(label='Input Tensor')]
    init_outputs = [NodeOutputType(label='Activated Tensor')]

    def update_event(self, inp=-1):
        """
        Applies ReLU activation to input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
  
        try:
            input_val, input_code = self.input(0).payload
            result = tf.keras.activations.relu(input_val)

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("ReLUNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code):
        """
        Generates Python code for the Relu activation.

        Args:
            input_code (str): Code for the input tensor.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Relu activation operation.

        """
        layer_name = name_registry.get_unique_name("relu")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.activations.relu({input_var_name})"
        return code


class ZeroPadding2D(Node):
    """
    A Ryven node that applies ZeroPadding2D layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
        Padding (Int, or tuple of 2 ints/tuples): Padding size.
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying ZeroPadding2D layer.
            - The generated Python code for this layer.
    """
    title = "ZeroPadding2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label="Padded Tensor")]

    def update_event(self, inp=-1):
        """
        Applies the ZeroPadding2D layer with the given padding size to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
        try:
            input_val, input_code = self.input(0).payload
            padding = self.input(1).payload

            if not hasattr(input_val, '_keras_history'):
                raise ValueError("Input is not a Keras symbolic tensor")

            layer = tf.keras.layers.ZeroPadding2D(padding=padding)
            result = layer(input_val)
            print(f"ZeroPaddingNode successful")

            code = self.generate_code(input_code, padding)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("ZeroPaddingNode error:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, padding):
        """
        Generates Python code for the ZeroPadding2D layer.

        Args:
            input_code (str): Code for the input tensor.
            padding (int or tuple of 2 int/tuples): Padding size used in the layer.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the ZeroPadding2D layer operation.

        """
        layer_name = name_registry.get_unique_name("zero_padding2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.ZeroPadding2D(padding={padding})({input_var_name})"
        return code


class BatchNormalization(Node):
    """
    A Ryven node that applies Batch Normalization layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
        Axis (int): The axis that should be normalized. 
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying batch normalization.
            - The generated Python code for this layer.
    """
    title = "BatchNormalization"
    init_inputs = [NodeInputType(label="Input Tensor"),
                   NodeInputType(label="Axis")]
    init_outputs = [NodeOutputType(label="Normalized Tensor")]

    def update_event(self, inp=-1):
        """
        Applies the Batch Normalization layer to the input tensor on the given axis.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for the Batch Normalization layer.

        Args:
            input_code (str): Code for the input tensor.
            axis (int): Axis to apply batch normalization to.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the batch normalization layer operation.

        """
        layer_name = name_registry.get_unique_name("batch_normalization")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.BatchNormalization(axis={axis})({input_var_name})"
        return code
    

class FlattenNode(Node):
    """
    A Ryven node that applies Flatten layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying flatten layer.
            - The generated Python code for this layer.
    """
    title = "Flatten"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Flattened Tensor")]

    def update_event(self, inp=-1):
        """
        Applies the Flatten layer to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for the Flatten layer.

        Args:
            input_code (str): Code for the input tensor.
            
        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Flatten layer operation.

        """
        layer_name = name_registry.get_unique_name("flatten")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Flatten()({input_var_name})"
        return code


class Conv2DNode(Node):
    """
    A Ryven node that applies Conv2D layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
        Filters (int): The number of filters in the convolution.
        Kernel Size (int or tuple of 2 int): Specifying the size of convolution window.
        Strides (int or tuple of 2 int): Specifying the stride length of convolution.
        Padding (str): either "valid" or "same".
        Kernel Initializer (Data): Initializer for convolution kernel.
            If None, "glorot_uniform" initializer will be used.
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying Conv2D.
            - The generated Python code for this layer.
    """
    title = "Conv2D Node"
    init_inputs = [NodeInputType(label="Tensor"), 
                   NodeInputType(label="Filters"), 
                   NodeInputType(label="Kernel Size"),
                   NodeInputType(label="Strides (Optional)"),
                   NodeInputType(label="Padding (Optional)"),
                   NodeInputType(label="Kernel Initializer (Optional)")]
    init_outputs = [NodeOutputType()]

    def update_event(self, inp=-1):
        """
        Applies the Conv2D layer with the given arguments to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for the Conv2D layer.

        Args:
            input_code (str): Code for the input tensor.
            filters (int): Number of filters.
            kernel_size (int or tuple of 2 int): Size of convolution window.
            strides (int or tuple of 2 int): Stride length.
            padding (str): Either "valid" or "same"
            kernel_initializer_code (str): String containing the code for initializer.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Conv2D layer operation.

        """
        layer_name = name_registry.get_unique_name("conv2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Conv2D(filters={filters}, kernel_size={kernel_size}, strides={strides}, padding='{padding}', kernel_initializer={kernel_initializer_code})({input_var_name})"
        return code


class AveragePooling2DNode(Node):
    """
    A Ryven node that applies Average pooling2D layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
        Pool Size (int or tuple of 2 int): Factors by which to downscale to.
        Strides (int or tuple of 2 int): Strides values.
        Padding (str): Either "valid" or "same".
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying average pooling2d.
            - The generated Python code for this layer.
    """

    title = "AveragePooling2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Pool Size"), 
                   NodeInputType(label="Strides"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        """
        Applies the AveragePooling2D layer with the given arguments to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """

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
        """
        Generates Python code for the AveragePooling2D layer.

        Args:
            input_code (str): Code for the input tensor.
            pool_size (int or tuple of 2 int): Given factor to downscale.
            strides (int or tuple of 2 int): Given strides value.
            padding (str): Given padding type.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the AveragePooling2D layer operation.

        """
        layer_name = name_registry.get_unique_name("average_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.AveragePooling2D(pool_size={pool_size}, strides={strides}, padding='{padding}')({input_var_name})"
        return code


class MaxPooling2DNode(Node):
    """
    A Ryven node that applies Max pool2D layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
        Pool Size (int or tuple of 2 int): Factors by which to downscale to.
        Strides (int or tuple of 2 int): Strides values.
        Padding (str): Either "valid" or "same".
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying MaxPool2d.
            - The generated Python code for this layer.
    """
    title = "MaxPool2D"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Pool Size"), 
                   NodeInputType(label="Strides"), 
                   NodeInputType(label="Padding")]
    init_outputs = [NodeOutputType(label='Pooled Tensor')]

    def update_event(self, inp=-1):
        """
        Applies the MaxPool2D layer with the given arguments to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
        try:
            input_val, input_code = self.input(0).payload
            print(f"MaxPooling2D")
            pool_size = self.input(1).payload
            strides = self.input(2).payload
            padding = self.input(3).payload

            layer = tf.keras.layers.MaxPool2D(pool_size=pool_size,
                                                strides=strides,
                                                padding=padding)
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code, pool_size, strides, padding)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[MaxPool2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code, pool_size, strides, padding):
        """
        Generates Python code for the MaxPool2D layer.

        Args:
            input_code (str): Code for the input tensor.
            pool_size (int or tuple of 2 int): Given factor to downscale.
            strides (int or tuple of 2 int): Given strides value.
            padding (str): Given padding type.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the MaxPool2D layer operation.

        """
        layer_name = name_registry.get_unique_name("max_pool2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.MaxPool2D(pool_size={pool_size}, strides={strides}, padding={padding})({input_var_name})"
        return code


class GlobalMaxPooling2DNode(Node):
    """
    A Ryven node that applies GlobalMaxPool2D layer to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying GlobalMaxPool2D.
            - The generated Python code for this layer.
    """
    title = "GlobalMaxPool2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        try:
            """
            Applies the GlobalMaxPool2D layer to the input tensor.

            Args:
                inp (int): Index of the input that triggered the update. Defaults to -1.

            Outputs:
                Output[0]: Data((output_tensor, generated_code)).
                        If an error occurs, outputs (None, "").
            """
            input_val, input_code = self.input(0).payload
            print(f"GlobalMaxPool2D")

            layer = tf.keras.layers.GlobalMaxPool2D()
            result = layer(input_val)
            print(f"Successful")
            print(f"----------------------------------------")

            code = self.generate_code(input_code)
            self.set_output_val(0, Data((result, code)))
        except Exception as e:
            print("[GlobalMaxPool2D error]:", e)
            print(f"----------------------------------------")
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, input_code):
        """
        Generates Python code for the GlobalMaxPool2D layer.

        Args:
            input_code (str): Code for the input tensor.
            
        Returns:
            str: A Python code string that combines the previous input tensor
                code and the GlobalMaxPool2D layer operation.

        """
        layer_name = name_registry.get_unique_name("global_max_pool2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.GlobalMaxPool2D()({input_var_name})"
        return code
    

class GlobalAveragePooling2DNode(Node):
    """
    A Ryven node that applies GlobalAveragePooling2D to input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying GlobalAveragePooling2D.
            - The generated Python code for this layer.
    """

    title = "GlobalAveragePooling2D"
    init_inputs = [NodeInputType(label="Input Tensor")]
    init_outputs = [NodeOutputType(label="Pooled Tensor")]

    def update_event(self, inp=-1):
        """
        Applies the GlobalAveragePooling2D layer to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for the GlobalAveragePooling2D layer.

        Args:
            input_code (str): Code for the input tensor.
            
        Returns:
            str: A Python code string that combines the previous input tensor
                code and the GlobalAveragePooling2D layer operation.

        """
        layer_name = name_registry.get_unique_name("global_average_pooling2d")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.GlobalAveragePooling2D()({input_var_name})"
        return code


class ConcatenateNode(Node):
    """
    A Ryven node that concatenates two tensors along a specified axis using 
    Keras's Concatenate layer.

    Inputs:
        Tensor 1 (Data): First input tensor and its code.
        Tensor 2 (Data): Second input tensor and its code.
        Axis (int): Axis along which to concatenate.

    Outputs:
        Result (Data): A tuple containing:
            - The resulting concatenated tensor.
            - The generated Python code for this operation.
    """
    title = 'Concatenate'
    init_inputs = [NodeInputType('Tensor 1'),
                   NodeInputType('Tensor 2'), 
                   NodeInputType("Axis")]
    init_outputs = [NodeOutputType('Result')]

    def update_event(self, inp=-1):
        """
        Concatenates two input tensors along the specified axis.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((concatenated_tensor, generated_code)).
                       Outputs (None, "") if an error occurs.
        """
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
        """
        Generates Python code to concatenate two tensors.

        Args:
            input_code_a (str): Code for the first tensor.
            input_code_b (str): Code for the second tensor.
            axis (int): Axis to concatenate along.

        Returns:
            str: Combined code string that includes the Concatenate layer.
        """
        layer_name = name_registry.get_unique_name("concatenate")

        lines_a = input_code_a.strip().split('\n')
        lines_b = input_code_b.strip().split('\n')

        seen = set()
        merged_lines = []
        for line in lines_a + lines_b:
            if line not in seen:
                seen.add(line)
                merged_lines.append(line)

        var_a = lines_a[-1].split('=')[0].strip()
        var_b = lines_b[-1].split('=')[0].strip()

        merged_lines.append(
            f"{layer_name} = tf.keras.layers.Concatenate(axis={axis})([{var_a}, {var_b}])"
        )

        return "\n".join(merged_lines)


class ReshapeNode(Node):
    """
    A Ryven node that reshapes input tensor.

    Inputs:
        Input Tensor (Data): The input tensor and its code.
        Target Shape (tuple of int): Target shape.
    
    Outputs:
        Result (Data): A tuple containing:
            - The resulting tensor after applying Reshape.
            - The generated Python code for this layer.
    """

    title = "Reshape"
    init_inputs = [NodeInputType(label="Input Tensor"), 
                   NodeInputType(label="Target Shape")]
    init_outputs = [NodeOutputType(label="Result")]

    def update_event(self, inp=-1):
        """
        Applies the Reshape layer with the given target shape to the input tensor.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((output_tensor, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for the Reshape layer.

        Args:
            input_code (str): Code for the input tensor.
            target (tuple of int): The target shape.

        Returns:
            str: A Python code string that combines the previous input tensor
                code and the Reshape layer operation.

        """
        layer_name = name_registry.get_unique_name("reshape")
        input_var_name = input_code.strip().split("\n")[-1].split("=")[0].strip()

        code = f"{input_code.strip()}\n{layer_name} = tf.keras.layers.Reshape(target_shape={target})({input_var_name})"
        return code


class FlushWrapper:
    """
    A wrapper for sys.stdout that ignores flush() calls.

    Used to prevent errors in environments where sys.stdout.flush()
    is not implemented.
    """

    def __init__(self, wrapped):
        """Initialises the wrapper."""
        self._wrapped = wrapped

    def write(self, *args, **kwargs):
        """Writes to the wrapped stream."""
        return self._wrapped.write(*args, **kwargs)
    
    def flush(self):
        """No-op to override flush without raising errors."""
        pass  # no-op

    def __getattr__(self, attr):
        """Delegates attribute access to the wrapped stream."""
        return getattr(self._wrapped, attr)

class ModelNode(Node):
    """
    A Ryven node that builds a Keras functional Model from input and output tensors.

    Inputs:
        Input (Data): The model's input tensor and its generated code.
        Output (Data): The model's output tensor and its generated code.

    Outputs:
        Keras Model (Data): A tuple containing:
            - The compiled tf.keras.Model.
            - The complete Python code to define the model.
    """

    title = "Model"
    init_inputs = [NodeInputType(label="Input"),
                   NodeInputType(label="Output")]
    init_outputs = [NodeOutputType(label="Keras Model")]

    def update_event(self, inp=-1):
        """
        Creates a Keras Model from the input and output tensors.

        Args:
            inp (int): Index of the updated input. Defaults to -1.

        Outputs:
            Output[0]: Data((model, code)) if successful.
                       If there's an error, outputs (None, "", 0).
        """
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
        """
        Merges input and output code and generates full model definition.

        Args:
            input_code (str): Code string for the input tensor.
            output_code (str): Code string for the output tensor.

        Returns:
            str: Python code that defines the full tf.keras.Model.
        """
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


class ModelSummaryNode(Node):
    """
    A Ryven node that prints and returns the summary of a Keras model.

    Inputs:
        Keras Model (Data): A tuple of (model object, generated code).

    Outputs:
        Summary (Data): A tuple containing:
            - The model summary text.
            - The code that includes the model and the .summary() call.
    """

    title = "Model Summary"
    init_inputs = [NodeInputType(label="Keras Model")]
    init_outputs = [NodeOutputType(label="Summary")]

    def update_event(self, inp=-1):
        """
        Extracts the summary from the input model and outputs the text and code.

        Args:
            inp (int): Index of the input that triggered the update.

        Outputs:
            Output[0]: Data((summary_text, generated_code)).
                       Outputs an error message if model is invalid.
        """
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
        """
        Appends a .summary() call to the given model code.

        Args:
            model_code (str): Code defining the Keras model.

        Returns:
            str: The code including the .summary() call.
        """
        lines = model_code.strip().split("\n")
        model_var = lines[-1].split("=")[0].strip() if "=" in lines[-1] else "model"
        new_code = f"{model_code}\n{model_var}.summary()"
        return new_code


class ModelCompileNode(Node):
    """
    A Ryven node that compiles a Keras model with specified optimizer, loss, and metrics.

    Inputs:
        Keras Model (Data): A tuple of (model object, generated code).
        Optimizer (Data or str): Optimizer object and code, or string name.
        Loss (str): Loss function name (e.g., "categorical_crossentropy").
        Metrics (list or str): List of metric names or a single metric.

    Outputs:
        Compiled Model (Data): A tuple containing:
            - The compiled model.
            - The generated Python code to compile the model.
    """
    title = "Model Compile"
    init_inputs = [
        NodeInputType(label="Keras Model"),
        NodeInputType(label="Optimizer"),
        NodeInputType(label="Loss"),
        NodeInputType(label="Metrics"),
    ]
    init_outputs = [NodeOutputType(label="Compiled Model")]

    def update_event(self, inp=-1):
        """
        Compiles the Keras model using the given optimizer, loss, and metrics.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((compiled_model, generated_code)).
                       Outputs (None, "") if compilation fails.
        """
        try:
            print(f"ModelCompile")
            model_val, model_code = self.input(0).payload
            
            optimizer_data = self.input(1).payload
            if isinstance(optimizer_data, tuple):
                optimizer, opt_code = optimizer_data
            else:
                optimizer = optimizer_data
                opt_code = f'"{optimizer}"'  # Fallback

            loss = self.input(2).payload
            metrics = self.input(3).payload

            model_val.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            code = self.generate_code(model_code, opt_code, loss, metrics)
            print(f"Model compile: success")

            self.set_output_val(0, Data((model_val, code)))
        except Exception as e:
            print("[ModelCompileNode Error]:", e)
            self.set_output_val(0, Data((None, "")))

    def generate_code(self, model_code, optimizer_code, loss, metrics):
        """
        Generates Python code for compiling a Keras model.

        Args:
            model_code (str): Code that defines the model.
            optimizer_code (str): Code or string for the optimizer.
            loss (str): Loss function name.
            metrics (list or str): List or string of metrics.

        Returns:
            str: Full Python code including the model and its compile statement.
        """
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


class ModelEvaluateNode(Node):
    """
    A Ryven node that evaluates a compiled Keras model on test data.

    Inputs:
        Model (Data): A tuple of (compiled Keras model, model code).
        x (Data): Evaluation input data and its code.
        y (Data): Evaluation target data and its code.

    Outputs:
        Loss and Metrics (Data): A tuple containing:
            - Evaluation results (loss and metric values).
            - Generated Python code to evaluate the model.
    """
    title = "Model Evaluate"
    init_inputs = [NodeInputType(label="Model"),
                   NodeInputType(label="x (Inputs)"),
                   NodeInputType(label="y (Targets)")]
    
    init_outputs = [NodeOutputType(label="Loss and Metrics")]

    def update_event(self, inp=-1):
        """
        Evaluates the model using provided x and y test data.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((evaluation_results, generated_code)).
                       If an error occurs, outputs (None, "").
        """
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
        """
        Generates Python code for evaluating the model.

        Args:
            model_code (str): Model creation and compilation code.
            x_code (str): Code for input test data.
            y_code (str): Code for target test data.

        Returns:
            str: Full evaluation code string.
        """
        lines_model = model_code.strip().split("\n")
        lines_x = x_code.strip().split("\n")
        lines_y = y_code.strip().split("\n")

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
    

class ModelFitNode(Node):
    """
    A Ryven node that trains a compiled Keras model on training data.

    Inputs:
        Model (Data): A tuple of (compiled model, code).
        x (Data): Training input data and code.
        y (Data): Training target data and code.
        Epochs (int): Number of training epochs.
        Batch Size (int): Batch size for training.
        Validation Data X (Optional, Data): Validation inputs and code.
        Validation Data Y (Optional, Data): Validation targets and code.
        Verbose (int): Verbosity level (0 = silent, 1 = progress bar).
        Trigger (exec): Button or manual trigger to start training.

    Outputs:
        History (Data): A tuple of:
            - The trained model.
            - Python code for training the model.
    """

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
        """
        Trains the model when the trigger input is activated.

        Args:
            inp (int): Index of the updated input. Only proceeds if it's the trigger.

        Outputs:
            Output[0]: Data((trained_model, generated_code)).
                       If training fails, outputs (None, "").
        """

        try:
            if inp == 8:
                model_val, model_code = self.input(0).payload
                x_val, x_code = self.input(1).payload
                y_val, y_code = self.input(2).payload
                epochs = int(self.input(3).payload)
                batch_size = int(self.input(4).payload)
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
        """
        Generates Python code for training the model.

        Args:
            model_code (str): Code for the input model.
            x_code (str): Code for the input data.
            y_code (str): Code for the target data.
            epochs (int): Number of epochs to train the model.
            batch_size (int): Bumber of samples per gradient update.
            val_code_x (str): Code for validation input data.
            val_code_y (str): Code for validation target data.
            verbose (int or str): Verbosity mode.

        Returns:
            str: Complete code for fitting the model, including training and optional validation data.
        """
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
            f"{model_var}.fit(trainX, trainY, epochs={epochs}, batch_size={batch_size}, validation_data=(testX, testY), verbose={verbose})"
        )

        return "\n".join(merged_lines)


class ModelSaveNode(Node):
    """
    A Ryven node that saves a Keras model to disk.

    Inputs:
        Model (Data): A tuple of (trained model, model code).
        Filepath (str): Path or filename to save the model.

    Outputs:
        Status (Data): A message and Python code used to save the model.
    """

    title = "Model Save"
    init_inputs = [
        NodeInputType(label="Model"),
        NodeInputType(label="Filepath"),
    ]
    init_outputs = [
        NodeOutputType(label="Status")
    ]

    def update_event(self, inp=-1):
        """
        Saves the provided Keras model to the specified file path.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data(("Model saved successfully.", generated_code)).
                       If an error occurs, outputs ("Model save failed.").
        """
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
        """
        Generates Python code to save the model.

        Args:
            model_code (str): Code that defines the model.
            filepath (str): Path to save the model to.

        Returns:
            str: Model save code with full path included.
        """
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


class GlorotUniformInitializerNode(Node):
    """
    A Ryven node that creates a GlorotUniform kernel initializer.

    Inputs:
        Seed (optional, int): An optional seed for reproducibility.

    Outputs:
        Initializer (Data): A tuple containing:
            - The tf.keras.initializers.GlorotUniform instance.
            - The code to recreate it.
    """

    title = "GlorotUniform Initializer"
    init_inputs = [NodeInputType(label="Seed (optional)")]
    init_outputs = [NodeOutputType(label="Initializer")]

    def place_event(self):
        """Initializes the node when placed on the canvas"""
        self.update_event()

    def update_event(self, inp=-1):
        """
        Creates the initializer with the given seed (if any).

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((initializer, code)).
                       If an error occurs, outputs (None, "").
        """
        try:
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
    """
    A Ryven node that creates a HeUniform kernel initializer.

    Inputs:
        None

    Outputs:
        Initializer (Data): A tuple containing:
            - The tf.keras.initializers.HeUniform instance.
            - The code to recreate it.
    """

    title = "HeUniform Initializer"
    init_inputs = []
    init_outputs = [NodeOutputType(label="Initializer")]

    def place_event(self):
        """Initializes the node when placed on the canvas."""
        self.update_event()

    def update_event(self, inp=-1):
        """
        Creates the HeUniform initializer.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((initializer, code)).
                       If an error occurs, outputs (None, "").
        """

        try:
            initializer = tf.keras.initializers.HeUniform()
            code = f"tf.keras.initializers.HeUniform()"
            print("HeUniform output set successfully:", code)
            self.set_output_val(0, Data((initializer, code)))
        except Exception as e:
            print("[HeUniformInitializerNode error]:", e)
            self.set_output_val(0, Data(None))


class SGDOptimizerNode(Node):
    """
    A Ryven node that creates an SGD optimizer.

    Inputs:
        Learning Rate (float): Learning rate for the optimizer.

    Outputs:
        Optimizer (Data): A tuple containing:
            - The tf.keras.optimizers.SGD instance.
            - The code to recreate it.
    """

    title = "SGD Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        """
        Creates an SGD optimizer using the specified learning rate.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((optimizer, code)).
            If an error occurs, outputs (None, "").
        """

        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.SGD(learning_rate={lr})"
            optimizer = tf.keras.optimizers.SGD(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[SGDOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))


class AdamOptimizerNode(Node):
    """
    A Ryven node that creates an Adam optimizer.

    Inputs:
        Learning Rate (float): Learning rate for the optimizer.

    Outputs:
        Optimizer (Data): A tuple containing:
            - The tf.keras.optimizers.Adam instance.
            - The code to recreate it.
    """

    title = "Adam Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        """
        Creates an Adam optimizer using the specified learning rate.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((optimizer, code)).
            If an error occurs, outputs (None, "").
        """
        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.Adam(learning_rate={lr})"
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[AdamOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))

class RMSpropOptimizerNode(Node):
    """
    A Ryven node that creates an RMSprop optimizer.

    Inputs:
        Learning Rate (float): Learning rate for the optimizer.

    Outputs:
        Optimizer (Data): A tuple containing:
            - The tf.keras.optimizers.RMSprop instance.
            - The code to recreate it.
    """

    title = "RMSprop Optimizer"
    init_inputs = [NodeInputType(label="Learning Rate")]
    init_outputs = [NodeOutputType(label="Optimizer")]

    def update_event(self, inp=-1):
        """
        Creates an RMSprop optimizer using the specified learning rate.

        Args:
            inp (int): Index of the input that triggered the update. Defaults to -1.

        Outputs:
            Output[0]: Data((optimizer, code)).
            If an error occurs, outputs (None, "").
        """
        try:
            lr = self.input(0).payload
            code = f"tf.keras.optimizers.RMSprop(learning_rate={lr})"
            optimizer = tf.keras.optimizers.RMSprop(learning_rate=lr)
            self.set_output_val(0, Data((optimizer, code)))
        except Exception as e:
            print("[RMSpropOptimizerNode error]:", e)
            self.set_output_val(0, Data((None, "")))


class PrintCodeNode(Node):
    """
    A Ryven node that prints the generated Python code from a model or layer node.

    Inputs:
        Input (Data): A tuple containing (Tensor or Model, generated code as str).

    Outputs:
        None. The code is printed to the console.
    """

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


class MNISTLoaderNode(Node):
    """
    A Ryven node that loads and preprocesses the MNIST dataset.

    Outputs:
        trainX (Data): Tuple of (training images, preprocessing code).
        trainY (Data): Tuple of (one-hot training labels, code).
        testX (Data): Tuple of (test images, preprocessing code).
        testY (Data): Tuple of (one-hot test labels, code).
    """

    title="MNIST Loader"
    init_inputs = []
    init_outputs = [
        NodeOutputType(label="trainX"),
        NodeOutputType(label="trainY"),
        NodeOutputType(label="testX"),
        NodeOutputType(label="testY"),
    ]

    def place_event(self):
        """Automatically runs when the node is added to the workspace."""
        self.update_event()

    def update_event(self, inp=-1):
        """
        Loads and processes the MNIST dataset.

        Args:
            inp (int): Index of the triggered input (unused).

        Outputs:
            Output[0]: trainX (Data)
            Output[1]: trainY (Data)
            Output[2]: testX (Data)
            Output[3]: testY (Data)
        """
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