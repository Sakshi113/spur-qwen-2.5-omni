import torch
import torch.fx
from torch.fx.node import Node, Argument, Target

import os
import sys

from typing import Any, Dict, Tuple, Optional

from spatpy.signal_path.primitives import forward_single_frame
from spatpy.signal_path.io import mermaid_html


class SignalPathTracer(torch.fx.Tracer):
    def create_arg(self, a):
        if isinstance(a, complex):
            return a
        return super().create_arg(a)

    def create_proxy(self, kind, target, args, kwargs, name=None, type_expr=None):
        # print(f"create_proxy: {target}")
        return super().create_proxy(
            kind, target, args, kwargs, name=name, type_expr=type_expr
        )

    def create_node(self, kind, target, args, kwargs, name=None, type_expr=None):
        # print(f"create_node: {target}")
        return super().create_node(
            kind, target, args, kwargs, name=name, type_expr=type_expr
        )

    def call_module(self, m, forward, args, kwargs):
        # print(f"call_module: {m}")
        return super().call_module(m, forward, args, kwargs)

    @staticmethod
    def signal_path_graph(
        m: torch.nn.Module, concrete_args=None
    ) -> torch.fx.GraphModule:
        tracer = SignalPathTracer()
        return torch.fx.GraphModule(m, tracer.trace(m, concrete_args=concrete_args))

    @classmethod
    def signal_path_to_mermaid(
        cls, m: torch.nn.Module, show=False, filename=None
    ) -> str:
        gm = cls.signal_path_graph(m)

        # gm.graph.lint()
        mm = to_mermaid(gm, show=show, filename=filename)
        return gm, mm


class SignalPathInterpreter(torch.fx.Interpreter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._mm = []
        self.prev = None
        self._mmid = 0

    def call_module(
        module_name: str,
        args: Optional[Tuple[Argument, ...]] = None,
        kwargs: Optional[Dict[str, Argument]] = None,
        type_expr: Optional[Any] = None,
    ) -> Any:
        # print(f"call_module: {module_name}")
        return super().call_module(module_name, args, kwargs, type_expr)

    def call_function(
        self,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        fname = target.__name__
        if fname == "forward_single_frame":
            state = None
            args = (args[0], state) + args[2:]

        ann = None
        if fname != "cat":
            ann = self._get_annotation(args)
        if fname not in ["getattr", "getitem"]:
            # print(f"call_function: {fname}({ann})")
            self._append_node(fname, annotation=ann)
        result = super().call_function(target, args, kwargs)
        return result

    def _append_node(self, name, annotation=None):
        s = ""
        if self.prev:
            s = f"{self._mmid} -->"
            if annotation is not None:
                s += f'|"{annotation}"|'
            s += " "
        self._mmid += 1
        self._mm.append(s + f'{self._mmid}["{name}"]')
        self.prev = name

    def call_method(
        self,
        target: Target,
        args: Tuple[Argument, ...],
        kwargs: Dict[str, Any],
    ) -> Any:
        classname = args[0].__class__.__name__
        ann = self._get_annotation(args)
        if target not in ["rename", "align_to", "to", "float", "refine_names"]:
            # print(f"call_method: {target}")
            self._append_node(f"{classname}.{target}()", annotation=ann)
        return super().call_method(target, args, kwargs)

    @property
    def mermaid(self):
        return "graph TD\n   " + "\n   ".join(self._mm)


def get_annotation(args):
    first = True
    annotation = ""
    for arg in args:
        if not first:
            annotation += ", "
        if hasattr(arg, "names") and arg.dim() > 0:
            if not all([n == None for n in arg.names]):
                annotation += (
                    "["
                    + ", ".join(
                        [f"{dim}:{sz}" for (dim, sz) in zip(arg.names, arg.shape)]
                    )
                    + "]"
                )
            else:
                annotation += "[" + ",".join([f"{sz}" for sz in arg.shape]) + "]"
        else:
            annotation += str(arg)
        first = False
    return annotation


def to_mermaid(m: torch.fx.GraphModule, indent=None, filename=None, show=False):
    # FX represents its Graph as an ordered list of
    # nodes, so we can iterate through them.
    mm = "graph TD\n   "
    for node in m.graph.nodes:
        for (k, v) in node.users.items():
            target_name = (
                node.target if isinstance(node.target, str) else node.target.__name__
            )
            # print(node.args)
            node_str = target_name
            if target_name == "getattr":
                node_str = f".{node.args[1]}"
            elif target_name == "getitem":
                args = node.args[1:]
                if isinstance(args[0], tuple):
                    args = args[0]
                arg_strs = []
                for arg in args:
                    if arg == Ellipsis:
                        arg_strs.append("...")
                    elif isinstance(arg, slice):
                        if arg.start is None and arg.step is None:
                            s = f":{arg.stop}"
                        elif arg.step is None and arg.stop is None:
                            s = f"{arg.start}:"
                        elif arg.step is None:
                            s = f"{arg.start}:{arg.stop}"
                        else:
                            s = f"{arg.start}:{arg.stop}:{arg.step}"
                        arg_strs.append(s)
                    else:
                        arg_strs.append(str(arg))

                node_str = "[" + ", ".join(arg_strs) + "]"
            else:
                arg_str = ""
                n_inputs = len(node.all_input_nodes)
                if len(node.args) > n_inputs:
                    if target_name == "mul":
                        node_str = f"* {node.args[0]}"
                    elif target_name == "truediv":
                        node_str = f"{node.args[0]} / x"
                    else:
                        if target_name == "apply_along_axis":
                            args = node.args[1:2]
                        else:
                            args = node.args[n_inputs:]
                        arg_str = (
                            "("
                            + ", ".join(
                                ["..." if arg == Ellipsis else str(arg) for arg in args]
                            )
                            + ")"
                        )
                        node_str = f"{target_name}{arg_str}"
            mm += f'   {node.name}["{node_str}"] --> {k}\n'

    if filename or show:
        mermaid_html(mm, filename=filename, show=show)
    return mm
