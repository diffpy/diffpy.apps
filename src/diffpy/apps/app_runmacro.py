import inspect
from collections import OrderedDict
from pathlib import Path

import yaml
from textx import metamodel_from_str

from diffpy.apps.pdfadapter import PDFAdapter

grammar = r"""
Program:
    commands*=Command
    variable=VariableBlock
;

Command:
    LoadCommand | SetCommand | CreateCommand | SaveCommand
;

LoadCommand:
    'load' component=ID name=ID 'from' source=STRING
;

SetCommand:
    'set' name=ID attribute=ID 'as' value+=Value[eolterm]
    | 'set' name=ID 'as' value+=Value[eolterm]
;

CreateCommand:
    'create' 'equation' 'variables' value+=Value[eolterm]
;

SaveCommand:
    'save' 'to' source=STRING
;

VariableBlock:
    'variables:' '---' content=/[\s\S]*?(?=---)/ '---'
;

Value:
    STRICTFLOAT | INT | STRING | RawValue
;

RawValue:
    /[^\s]+/
;
"""


class MacroParser:
    def __init__(self):
        self.pdfadapter = PDFAdapter()
        self.meta_model = metamodel_from_str(grammar)
        self.meta_model.register_obj_processors(
            {
                "SetCommand": self.set_command_processor,
                "LoadCommand": self.load_command_processor,
                "VariableBlock": self.parameter_block_processor,
                "CreateCommand": self.create_command_processor,
                "SaveCommand": self.save_command_processor,
            }
        )
        # key: method_name.argument_name
        # value: argument_value
        self.inputs = {}
        # key: structure name or profile name set in the macro
        # value: 'structure' or 'profile'
        self.variables = OrderedDict()

    def parse(self, code):
        self.meta_model.model_from_str(code)

    def input_as_list(self, key, value):
        if key in self.inputs:
            if not isinstance(self.inputs[key], list):
                self.inputs[key] = [self.inputs[key]]
            else:
                self.inputs[key].append(value)
        else:
            if isinstance(value, list):
                self.inputs[key] = value
            else:
                self.inputs[key] = [value]

    def load_command_processor(self, command):
        if command.component == "structure":
            # TODO: support multiple structures input in the future
            key = "initialize_structures.structure_paths"
            variable = "structure"
        elif command.component == "profile":
            key = "initialize_profile.profile_path"
            variable = "profile"
        else:
            raise ValueError(
                f"Unknown component type: {command.component} "
                "Please use 'structure' or 'profile'."
            )
        source_path = Path(command.source)
        if not source_path.exists():
            raise FileNotFoundError(
                f"{command.component} {source_path} not found. "
                "Please ensure the path is correct and the file exists."
            )
        self.inputs[key] = str(source_path)
        self.variables[command.name] = variable
        if variable == "structure":
            self.input_as_list("initialize_structures.names", command.name)

    def set_command_processor(self, command):
        if command.name == "equation":
            key = "initialize_contribution.equation"
        elif command.name in self.variables:
            if self.variables[command.name] == "structure":
                if command.attribute == "spacegroup":
                    key = "initialize_structures.spacegroups"
                else:
                    key = "initialize_structures." + command.attribute
            elif self.variables[command.name] == "profile":
                key = "initialize_profile." + command.attribute
            else:
                raise ValueError(
                    f"Unknown variable type for name: {command.name}. "
                    "This is an internal error. Please report this issue to "
                    "the developers."
                )
        else:
            raise ValueError(
                f"Unknown name in set command: {command.name}. "
                "Please ensure that it is typed correctly as 'equation' or "
                "it matches a previously loaded structure or "
                "profile. "
            )
        self.input_as_list(key, command.value)

    def parameter_block_processor(self, variable_block):
        self.inputs["set_initial_variable_values.variable_name_to_value"] = {}
        self.inputs["refine_variables.variable_names"] = []
        parameters = yaml.safe_load(variable_block.content)
        if not isinstance(parameters, list):
            raise ValueError(
                "Parameter block should contain a list of parameters. "
                "Please use the following format:\n"
                "- param1  # use default initial value\n"
                "- param2: initial_value\n"
            )
        for item in parameters:
            if isinstance(item, str):
                self.inputs["refine_variables.variable_names"].append(
                    item.replace(".", "_")
                )
            elif isinstance(item, dict):
                pname, pvalue = list(item.items())[0]
                self.inputs[
                    "set_initial_variable_values.variable_name_to_value"
                ][pname.replace(".", "_")] = pvalue
                self.inputs["refine_variables.variable_names"].append(
                    pname.replace(".", "_")
                )
            else:
                raise ValueError(
                    "Variables block items are not correctly formatted. "
                    "Please use the following format:\n"
                    "- param1  # use default initial value\n"
                    "- param2: initial_value\n"
                )

    def create_command_processor(self, command):
        self.inputs["add_contribution_variables.variable_names"] = (
            command.value
        )

    def save_command_processor(self, command):
        self.inputs["save_results.result_path"] = command.source

    def required_args(self, func):
        sig = inspect.signature(func)
        return [
            name
            for name, p in sig.parameters.items()
            if p.default is inspect.Parameter.empty
            and p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            )
        ]

    def call_pdfadapter_method(self, method_name, function_requirement):
        func = getattr(self.pdfadapter, method_name)
        required_arguments = self.required_args(func)
        arguments = {
            key.split(".")[1]: value
            for key, value in self.inputs.items()
            if key.startswith(method_name)
        }
        if not all(arg in arguments for arg in required_arguments):
            missing_args = [
                arg for arg in required_arguments if arg not in arguments
            ]
            if function_requirement == "required":
                raise ValueError(
                    "Missing required arguments for function "
                    f"'{method_name}' {', '.join(missing_args)}. "
                    "Please provide these arguments in the macro file."
                )
            elif function_requirement == "optional":
                print(
                    "Missing required arguments for function "
                    f"'{method_name}' {', '.join(missing_args)}. "
                    "This function will be skipped. "
                    "Please provide these arguments in the macro file "
                    "to activate this function."
                )
                return
        func(**arguments)

    def preprocess(self):
        methods_to_call = [
            ("initialize_profile", "required"),
            ("initialize_structures", "required"),
            ("initialize_contribution", "required"),
            ("initialize_recipe", "required"),
            ("add_contribution_variables", "optional"),
            ("set_initial_variable_values", "optional"),
        ]
        for method in methods_to_call:
            self.call_pdfadapter_method(*method)

    def run(self):
        methods_to_call = [
            ("refine_variables", "required"),
            ("save_results", "optional"),
        ]
        for method in methods_to_call:
            self.call_pdfadapter_method(*method)
        return self.pdfadapter.get_results()


def runmacro(args):
    dpin_path = Path(args.file)
    if not dpin_path.exists():
        raise FileNotFoundError(
            f"{str(dpin_path)} not found. Please check if this file "
            "exists and provide the correct path to it."
        )
    dsl_code = dpin_path.read_text()
    parser = MacroParser()
    parser.parse(dsl_code)
    parser.preprocess()
    return parser.run()


if __name__ == "__main__":
    parser = MacroParser()
    code = f"""
load structure G1 from "{str(Path(__file__).parents[3] / "tests/data/Ni.cif")}"
load profile exp_ni from "{str(Path(__file__).parents[3] / "tests/data/Ni.gr")}"

set G1 spacegroup as auto
set exp_ni q_range as 0.1 25
set exp_ni calculation_range as 1.5 50 0.01
create equation variables s0
set equation as "s0*G1"
save to "results.json"

variables:
---
- G1.a: 3.52
- s0: 0.4
- G1.Uiso_0: 0.005
- G1.delta2: 2
- qdamp: 0.04
- qbroad: 0.02
---
"""  # noqa: E501
    parser.parse(code)
    parser.preprocess()
    recipe = parser.pdfadapter.recipe
    for pname, param in recipe._parameters.items():
        print(f"{pname}: {param.value}")
