from ryven.gui_env import *
from qtpy.QtWidgets import QLineEdit
from qtpy.QtCore import Qt
from . import nodes
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QComboBox
from qtpy.QtWidgets import QVBoxLayout, QWidget
from qtpy.QtWidgets import QPushButton



class LineEditWidget(NodeInputWidget, QLineEdit):
    """A simple text input that updates the node when value changes."""

    def __init__(self, params):
        NodeInputWidget.__init__(self, params)
        QLineEdit.__init__(self)

        self.setMinimumWidth(150)
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(8)
        self.setFont(font)
        # self.setText("0")  # Default value
        # self.returnPressed.connect(self.on_return)
        self.textChanged.connect(self.on_text_changed)

    def on_text_changed(self, text):
        try:
            val = eval(text)
        except:
            val = text  # fallback to string if eval fails
        self.update_node_input(Data(val))

    def on_return(self):
        """Evaluate the text and update the node input."""
        try:
            val = eval(self.text())
        except:
            val = self.text()  # Fallback to raw string
        self.update_node_input(Data(val))

    def get_state(self) -> dict:
        return {'text': self.text()}

    def set_state(self, state: dict):
        self.setText(state.get('text', ''))  # Restore previous text


"""class ComboBoxWidget(QWidget):
    def __init__(self, options, default=None):
        super().__init__()
        self.position = 'below'  # <--- THIS IS REQUIRED
        self.combo = QComboBox()
        self.combo.addItems(options)
        #self.combo.setFont(QFont("Arial", 10))  # Set readable font and size
        #self.combo.setStyleSheet("color: black; background-color: white;")  # Ensure text is visible
        if default in options:
            self.combo.setCurrentText(default)
        
        layout = QVBoxLayout()
        layout.addWidget(self.combo)
        self.setLayout(layout)

    def get_val(self):
        return self.combo.currentText()"""



@node_gui(nodes.InputNode)
class InputNodeGui(NodeGUI):

    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        0: {'name': 'line_edit', 'pos': 'below'}  # Attach to input 0
    }

@node_gui(nodes.ZeroPadding2D)
class ZeroPadding2DGui(NodeGUI):

    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.BatchNormalization)
class BatchNormalizationGui(NodeGUI):

    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.Conv2DNode)
class Conv2DNodeGui(NodeGUI):

    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'},
        2: {'name': 'line_edit', 'pos': 'below'},
        3: {'name': 'line_edit', 'pos': 'below'},
        4: {'name': 'line_edit', 'pos': 'below'},
    }

@node_gui(nodes.DenseNode)
class DenseNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget,
        #'combo_box': ComboBoxWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'},
        2: {'name': 'line_edit', 'pos': 'below'},
        #2: {'name': 'combo_box', 'pos': 'below', 'args': ['relu', 'sigmoid', 'softmax']}
    }

@node_gui(nodes.ModelFitNode)
class ModelFitNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        3: {'name': 'line_edit', 'pos': 'below'},
        4: {'name': 'line_edit', 'pos': 'below'},
        7: {'name': 'line_edit', 'pos': 'below'},
    }

@node_gui(nodes.DropoutNode)
class DropoutNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.AveragePooling2DNode)
class AvergaePooling2DNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'},
        2: {'name': 'line_edit', 'pos': 'below'},
        3: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.MaxPooling2DNode)
class MaxPooling2DNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'},
        2: {'name': 'line_edit', 'pos': 'below'},
        3: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.ConcatenateNode)
class ConcatenateNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        2: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.ReshapeNode)
class ReshapeNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'}
    }  

@node_gui(nodes.SGDOptimizerNode)
class SGDOptimizerNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        0: {'name': 'line_edit', 'pos': 'below'}
    }  

@node_gui(nodes.AdamOptimizerNode)
class AdamOptimizerNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        0: {'name': 'line_edit', 'pos': 'below'}
    }  

@node_gui(nodes.RMSpropOptimizerNode)
class RMSpropOptimizerNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        0: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.GlorotUniformInitializerNode)
class GlorotUniformInitializerNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        0: {'name': 'line_edit', 'pos': 'below'}
    }

@node_gui(nodes.ModelCompileNode)
class ModelCompileNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        2: {'name': 'line_edit', 'pos': 'below'},
        3: {'name': 'line_edit', 'pos': 'below'}
    }  

@node_gui(nodes.ModelSaveNode)
class ModelSaveNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        1: {'name': 'line_edit', 'pos': 'below'}
    } 

@node_gui(nodes.ModelEvaluateNode)
class ModelEvaluateNodeGui(NodeGUI):
    
    input_widget_classes = {
        'line_edit': LineEditWidget
    }

    init_input_widgets = {
        3: {'name': 'line_edit', 'pos': 'below'}
    } 