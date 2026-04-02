import json
from pathlib import Path

import yaml
from scipy.optimize import least_squares
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


class DiffpyInterpreter:
    def __init__(self):
        self.pdfadapter = PDFAdapter()
        self.meta_model = metamodel_from_str(grammar)
        self.meta_model.register_obj_processors(
            {
                "SetCommand": self.set_command_processor,
                "LoadCommand": self.load_command_processor,
                "VariableBlock": self.variable_block_processor,
                "CreateCommand": self.create_command_processor,
                "SaveCommand": self.save_command_processor,
            }
        )
        self.inputs = {}
        self.profile_name = ""
        self.structure_name = (
            ""  # TODO: support multiple structures in the future
        )

    def interpret(self, code):
        self.meta_model.model_from_str(code)

    def load_command_processor(self, command):
        if command.component == "structure":
            source_entry = "structure_path"
            attribute_name = "structure_name"
        elif command.component == "profile":
            source_entry = "profile_path"
            attribute_name = "profile_name"
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
        self.inputs[source_entry] = str(source_path)
        setattr(self, attribute_name, command.name)

    def set_command_processor(self, command):
        if "structures_config" not in self.inputs:
            self.inputs["structures_config"] = {}
        if "profiles_config" not in self.inputs:
            self.inputs["profiles_config"] = {}
        if command.name in ["equation"]:
            self.inputs[command.name] = command.value
        elif command.name == self.structure_name:
            self.inputs["structures_config"][
                self.structure_name + "_" + command.attribute
            ] = command.value
        elif command.name == self.profile_name:
            self.inputs["profiles_config"][command.attribute] = command.value
        else:
            raise ValueError(
                f"Unknown name in set command: {command.name}. "
                "Please ensure it matches a previously loaded structure or "
                "profile."
            )

    def variable_block_processor(self, variable_block):
        self.inputs["variables"] = []
        self.inputs["initial_values"] = {}
        variables = yaml.safe_load(variable_block.content)
        if not isinstance(variables, list):
            raise ValueError(
                "Variables block should contain a list of variables. "
                "Please use the following format:\n"
                "- var1  # use default initial value\n"
                "- var2: initial_value\n"
            )
        for item in variables:
            if isinstance(item, str):
                self.inputs["variables"].append(item.replace(".", "_"))
            elif isinstance(item, dict):
                pname, pvalue = list(item.items())[0]
                self.inputs["variables"].append(pname.replace(".", "_"))
                self.inputs["initial_values"][pname.replace(".", "_")] = pvalue
            else:
                raise ValueError(
                    "Variables block items are not correctly formatted. "
                    "Please use the following format:\n"
                    "- var1  # use default initial value\n"
                    "- var2: initial_value\n"
                )

    def create_command_processor(self, command):
        self.inputs["equation_variable"] = [
            v for v in command.value if isinstance(v, str)
        ]

    def save_command_processor(self, command):
        self.inputs["result_path"] = command.source

    def configure_adapter(self):
        self.pdfadapter.initialize_profile(
            self.inputs["profile_path"], **self.inputs["profiles_config"]
        )
        spacegroups = self.inputs["structures_config"].get(
            f"{self.structure_name}_spacegroup", None
        )
        spacegroups = None if spacegroups == ["auto"] else spacegroups
        self.pdfadapter.initialize_structures(
            [self.inputs["structure_path"]],
            run_parallel=True,
            spacegroups=spacegroups,
            names=[self.structure_name],
        )
        self.pdfadapter.initialize_contribution(self.inputs["equation"][0])
        self.pdfadapter.initialize_recipe()
        for i in range(len(self.inputs["equation_variable"])):
            self.pdfadapter.recipe.addVar(
                getattr(
                    list(self.pdfadapter.recipe._contributions.values())[0],
                    self.inputs["equation_variable"][i],
                )
            )
        self.pdfadapter.set_initial_variable_values(
            self.inputs["initial_values"]
        )

    def run(self):
        self.pdfadapter.recipe.fix("all")
        for var in self.inputs["variables"]:
            self.pdfadapter.recipe.free(var)
            least_squares(
                self.pdfadapter.recipe.residual, self.pdfadapter.recipe.values
            )
        if "result_path" in self.inputs:
            with open(self.inputs["result_path"], "w") as f:
                json.dump(self.pdfadapter.get_results(), f, indent=4)
        return self.pdfadapter.get_results()

    def run_app(self, args):
        dpin_path = Path(args.input_file)
        if not dpin_path.exists():
            raise FileNotFoundError(
                f"{str(dpin_path)} not found. Please check if this file "
                "exists and provide the correct path to it."
            )
        dsl_code = dpin_path.read_text()
        self.interpret(dsl_code)
        self.configure_adapter()
        self.run()


if __name__ == "__main__":
    interpreter = DiffpyInterpreter()
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
    interpreter.interpret(code)
    interpreter.configure_adapter()
    recipe = interpreter.pdfadapter.recipe
    for pname, param in recipe._parameters.items():
        print(f"{pname}: {param.value}")
    print(interpreter.inputs["variables"])
