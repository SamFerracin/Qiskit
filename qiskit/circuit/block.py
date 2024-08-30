# This code is part of Qiskit.
#
# (C) Copyright IBM 2020.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

"""
Delay instruction (for circuit module).
"""
from typing import Optional, Sequence, Tuple
import numpy as np

from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit.instruction import Instruction
from qiskit.circuit.gate import Gate
from qiskit.circuit import _utils
from qiskit.circuit.parameterexpression import ParameterExpression


class CircuitBlock(Instruction):
    """A container for an isolated block of operations in a larger circuit.
    
    Args:
        circuit: A quantum circuit containing all the operations in the block.
        label: A label.
    """

    def __init__(self, circuit: QuantumCircuit, label: str = ""):
        self._circuit = circuit

        super().__init__(
            "block",
            self.circuit.num_qubits,
            self.circuit.num_clbits,
            params=self.circuit.parameters,
            label=label,
        )

    @property
    def circuit(self) -> QuantumCircuit:
        r"""
        The circuit in this block.
        """
        return self._circuit
    
    def draw(self):
        return self.circuit.draw()
