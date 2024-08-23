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

    This is roughly equivalent to a qiskit ``CircuitOperation`` but on abstract qubit
    indices instead of specific qubit objects in a QuantumCircuits quantum registers.

    It is intended to store sub-circuits from a larger circuit to enable hashing
    and equivalency comparison.

    .. note::

        The circuit contained in this block is assumed to be immutable. If it is
        modified in anyway it will invalidate the use of this block for equivalence
        checking.
    """

    def __init__(
        self, circuit: QuantumCircuit, qubits: Optional[Sequence[int]] = None, label: str = ""
    ):
        """Initialize a circuit block.

        Args:
            circuit: A quantum circuit.
            qubits: The physical qubit indices corresponding to the circuit's abstract qubits. If
                ``None``, it is set to ``range(circuit.num_qubits)``.
        """
        self._circuit = circuit
        self._qubits = tuple(qubits if qubits else range(circuit.num_qubits))

        if self.circuit.num_qubits != len(self.qubits):
            raise ValueError("Input circuit and qubits have different number of qubits.")

        super().__init__(
            "block",
            self.circuit.num_qubits,
            self.circuit.num_clbits,
            params=self.circuit.parameters,
            label=label
        )

    @property
    def circuit(self) -> QuantumCircuit:
        r"""
        The circuit in this block.
        """
        return self._circuit

    @property
    def qubits(self) -> Tuple[int]:
        r"""
        The physical qubit indices corresponding to the ``circuit``'s abstract qubits.
        """
        return self._qubits
