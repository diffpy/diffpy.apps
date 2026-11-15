from pathlib import Path

from textx import metamodel_from_str

from diffpy.apps.parametric_refinement.refinement_session import (
    RefinementSession,
)

grammar = r"""
Program:
    calls*=Call
;

Call:
    name=ID '(' args*=Argument[','] ','? ')'
;

Argument:
    name=ID '=' value=Value | value=Value
;

Value:
    Bool | NoneValue | List | Dict | NUMBER | STRING
;

Bool:
    value=/True|False/
;

NoneValue:
    value='None'
;

List:
    '[' items*=Value[','] ','? ']'
;

Dict:
    '{' items*=DictItem[','] ','? '}'
;

DictItem:
    key=STRING ':' value=Value
;

Comment:
    /#[^\n]*/
;
"""


def to_python(node):
    kind = type(node).__name__
    if kind == "Bool":
        return node.value == "True"
    if kind == "NoneValue":
        return None
    if kind == "List":
        return [to_python(item) for item in node.items]
    if kind == "Dict":
        return {item.key: to_python(item.value) for item in node.items}
    return node


class MacroRunner:
    def __init__(self, session=None):
        self.session = session if session is not None else RefinementSession()
        self.meta_model = metamodel_from_str(grammar)

    def run(self, code):
        calls = self.meta_model.model_from_str(code).calls
        for call in calls:
            if call.name.startswith("_") or not callable(
                getattr(self.session, call.name, None)
            ):
                raise ValueError(
                    f"Unknown session method '{call.name}'. "
                    "Use a public method of RefinementSession."
                )
        results = []
        for call in calls:
            args = []
            kwargs = {}
            for arg in call.args:
                if arg.name:
                    kwargs[arg.name] = to_python(arg.value)
                elif kwargs:
                    raise ValueError(
                        f"In '{call.name}', positional arguments must come "
                        "before keyword arguments."
                    )
                else:
                    args.append(to_python(arg.value))
            results.append(getattr(self.session, call.name)(*args, **kwargs))
        return results

    def run_file(self, path):
        macro_path = Path(path)
        if not macro_path.exists():
            raise FileNotFoundError(
                f"{macro_path} not found. Please check the path to the "
                "macro file."
            )
        return self.run(macro_path.read_text())


def runmacro(args):
    results = MacroRunner().run_file(args.file)
    for result in results:
        if isinstance(result, dict) and "result" in result:
            print(result["result"])
    return results
