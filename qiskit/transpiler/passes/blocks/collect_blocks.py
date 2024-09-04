# This code is part of Qiskit.
#
# (C) Copyright IBM 2017, 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""Calculate the width of a DAG circuit."""

from qiskit.transpiler.basepasses import TransformationPass
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.block import CircuitBlock


class CollectBlocks(TransformationPass):
    """Collects the gates of a DAG circuit into :class:`.CircuitBlock`\\s.
    """

    def run(self, dag):
        circuit = QuantumCircuit
        # for node in dag.op_nodes():
        #     if isinstance(node.op, ISA_SUPPORTED_GATES + SUPPORTED_INSTRUCTIONS):
        #         # Fast handling of ISA gates. It rounds the angle of `RZ`s to a multiple of pi/2,
        #         # while skipping every other supported gates and instructions.
        #         if isinstance(node.op, RZGate):
        #             if isinstance(angle := node.op.params[0], float):
        #                 rem = angle % (np.pi / 2)
        #                 new_angle = angle - rem if rem < np.pi / 4 else angle + np.pi / 2 - rem
        #             else:
        #                 # special handling of parametric gates
        #                 new_angle = choices([0, np.pi / 2, np.pi, 3 * np.pi / 2])[0]
        #             dag.substitute_node(node, RZGate(new_angle), inplace=True)
        #     else:
        #         # Handle non-ISA gates, which may be either Clifford or non-Clifford.
        #         if not _is_clifford(node.op):
        #             raise ValueError(f"Operation ``{node.op.name}`` not supported.")
        #         dag.substitute_node(node, node.op, inplace=True)

        return dag
