# This code is part of Qiskit.
#
# (C) Copyright IBM 2024.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.


"""
ND-Array container class for Estimator observables.
"""
from __future__ import annotations

import re
from collections import defaultdict
from collections.abc import Iterable, Mapping as _Mapping
from functools import lru_cache
from typing import Union, Mapping, Tuple, Sequence, overload
from numbers import Complex

import numpy as np
from numpy.typing import ArrayLike

from qiskit.quantum_info import Pauli, PauliList, SparsePauliOp

from .object_array import object_array
from .shape import ShapedMixin, shape_tuple

# Public API classes
__all__ = ["ObservableLike", "ObservablesArrayLike"]

ObservableLike = Union[
    str,
    Pauli,
    SparsePauliOp,
    Mapping[Union[str, Pauli], float],
    Tuple[Union[str, Pauli, SparsePauliOp], Sequence[int], int],
]
"""Types that can be natively used to construct a Hermitian Estimator observable.

The following formats are allowed:
- A string, e.g. "XlXZO+".
- A :class:`.Pauli` or a :class:`.SparsePauliOp` object.
- A map between Paulis and floats, where each float represents the weight of the
  corresponding pauli, e.g. ``{"ZI": 0.5, "IZ": 0.5}``.
- A tuple of the type ``(pauli, indices, num_qubits)`` representing a Pauli with support on a
  subset of qubits. For example, ``(XYZ, [0, 2, 4], 6)`` is equivalent to the six-qubit Pauli
  ``IIXIYIZ``.
"""


ObservablesArrayLike = Union[ObservableLike, ArrayLike]
"""Types that can be natively converted to an array of Hermitian Estimator observables."""


class ObservablesArray(ShapedMixin):
    """An ND-array of Hermitian observables for an :class:`.Estimator` primitive."""

    __slots__ = ("_array", "_shape")
    ALLOWED_BASIS: str = "IXYZ01+-lr"
    """The allowed characters in basis strings."""

    def __init__(
        self,
        observables: ObservablesArrayLike,
        copy: bool = True,
        validate: bool = True,
    ):
        """Initialize an observables array.

        Args:
            observables: An array-like of basis observable compatible objects.
            copy: Specify the ``copy`` kwarg of the :func:`.object_array` function
                when initializing observables.
            validate: If true, coerce entries into the internal format and validate them. If false,
                the input should already be an array-like.

        Raises:
            ValueError: If ``validate=True`` and the input observables is not valid.
        """
        super().__init__()
        if isinstance(observables, ObservablesArray):
            observables = observables._array
        if isinstance(observables, Sequence) and len(observables) == 3:
            # Special handling of observable with support on a subset of qubits.
            # They need to turned into a tuple before passing it to `object_array`
            # TODO: Can this be avoided?
            observables = (observables[0], tuple(observables[1]), observables[2])
        self._array = object_array(observables, copy=copy, list_types=(PauliList,))
        self._shape = self._array.shape
        if validate:
            num_qubits = None
            for ndi, obs in np.ndenumerate(self._array):
                basis_obs = self.coerce_observable(obs)
                basis_num_qubits = len(next(iter(basis_obs)))
                if num_qubits is None:
                    num_qubits = basis_num_qubits
                elif basis_num_qubits != num_qubits:
                    raise ValueError(
                        "The number of qubits must be the same for all observables in the "
                        "observables array."
                    )
                self._array[ndi] = basis_obs

    def __repr__(self):
        prefix = f"{type(self).__name__}("
        suffix = f", shape={self.shape})"
        array = np.array2string(self._array, prefix=prefix, suffix=suffix, threshold=50)
        return prefix + array + suffix

    def tolist(self) -> list:
        """Convert to a nested list"""
        return self._array.tolist()

    def __array__(self, dtype=None, copy=None):
        """Convert to an Numpy.ndarray"""
        if dtype is None or dtype == object:
            return self._array.copy() if copy else self._array
        raise ValueError("Type must be 'None' or 'object'")

    @overload
    def __getitem__(self, args: int | tuple[int, ...]) -> Mapping[str, float]: ...

    @overload
    def __getitem__(self, args: slice) -> ObservablesArray: ...

    def __getitem__(self, args):
        item = self._array[args]
        if not isinstance(item, np.ndarray):
            return item
        return ObservablesArray(item, copy=False, validate=False)

    def reshape(self, *shape: int | Iterable[int]) -> ObservablesArray:
        """Return a new array with a different shape.

        This results in a new view of the same arrays.

        Args:
            shape: The shape of the returned array.

        Returns:
            A new array.
        """
        shape = shape_tuple(*shape)
        return ObservablesArray(self._array.reshape(shape), copy=False, validate=False)

    def ravel(self) -> ObservablesArray:
        """Return a new array with one dimension.

        The returned array has a :attr:`shape` given by ``(size, )``, where
        the size is the :attr:`~size` of this array.

        Returns:
            A new flattened array.
        """
        return self.reshape(self.size)

    @classmethod
    def coerce_observable(cls, observable: ObservableLike) -> Mapping[str, float]:
        """Format an observable-like object into the internal format.

        Args:
            observable: The observable-like to format.

        Returns:
            The coerced observable.

        Raises:
            TypeError: If the input cannot be formatted because its type is not valid.
            ValueError: If the input observable is invalid.
        """
        # Pauli-type conversions
        if isinstance(observable, SparsePauliOp):
            observable = observable.simplify(atol=0)
            # Check that the operator is Hermitian and has real coeffs
            coeffs = np.real_if_close(observable.coeffs)
            if np.iscomplexobj(coeffs):
                raise ValueError(
                    "Non-Hermitian input observable: the input SparsePauliOp has non-zero"
                    " imaginary part in its coefficients."
                )
            paulis = observable.paulis.to_labels()
            # Call simplify to combine duplicate keys before converting to a mapping
            return dict(zip(paulis, coeffs))

        if isinstance(observable, Pauli):
            label, phase = observable[:].to_label(), observable.phase
            if phase % 2:
                raise ValueError(
                    "Non-Hermitian input observable: the input Pauli has an imaginary phase."
                )
            return {label: 1} if phase == 0 else {label: -1}

        # String conversion
        if isinstance(observable, str):
            cls._validate_basis(observable)
            return {observable: 1}

        # Mapping conversion (with possible Pauli keys)
        if isinstance(observable, _Mapping):
            num_qubits = len(next(iter(observable)))
            unique = defaultdict(float)
            for basis, coeff in observable.items():
                if isinstance(basis, Pauli):
                    basis, phase = basis[:].to_label(), basis.phase
                    if phase % 2:
                        raise ValueError(
                            "Non-Hermitian input observable: the input Pauli has an imaginary phase."
                        )
                    if phase == 2:
                        coeff = -coeff
                # Truncate complex numbers to real
                if isinstance(coeff, Complex):
                    if abs(coeff.imag) > 1e-7:
                        raise TypeError(
                            f"Non-Hermitian input observable: {basis} term has a complex value"
                            " coefficient."
                        )
                    coeff = coeff.real

                # Validate basis
                cls._validate_basis(basis)
                if len(basis) != num_qubits:
                    raise ValueError(
                        "Number of qubits must be the same for all observable basis elements."
                    )
                unique[basis] += coeff
            return dict(unique)
        
        # Tuple conversion, performed by inserting identities and calling `coerce_observable` on
        # the resulting SparsePauliOp
        if isinstance(observable, Tuple):
            if len(observable) != 3:
                raise ValueError("Incorrect length.")
            
            obs = observable[0]
            indices = observable[1]
            num_qubits = observable[2]

            # Turn `obs` into a SparsePauliOp
            if isinstance(obs, str):
                obs = Pauli(obs)
            if isinstance(obs, Pauli):
                obs = SparsePauliOp(obs)  
            if not isinstance(obs, SparsePauliOp):
                raise ValueError("Failed to coerce tuple into `SparsePauliOp`.")

            if obs.num_qubits != len(indices):
                raise ValueError(f"Expected {obs.num_qubits} indices, found {len(indices)}.")
            if obs.num_qubits > num_qubits:
                raise ValueError(f"Invalid number of qubits.")

            new_paulis = []
            for pauli in obs.paulis:
                pos = [p for p in range(num_qubits) if p not in indices]
                new_paulis.append(pauli.insert(pos, Pauli("I" * len(pos))))
            return cls.coerce_observable(SparsePauliOp(new_paulis, obs.coeffs))

        raise TypeError(f"Invalid observable type: {type(observable)}")

    @classmethod
    def coerce(cls, observables: ObservablesArrayLike) -> ObservablesArray:
        """Coerce ObservablesArrayLike into ObservableArray.

        Args:
            observables: an object to be observables array.

        Returns:
            A coerced observables array.
        """
        if isinstance(observables, ObservablesArray):
            return observables
        return cls(observables)

    def validate(self):
        """Validate the consistency in observables array."""
        num_qubits = None
        for obs in self._array.reshape(-1):
            basis_num_qubits = len(next(iter(obs)))
            if num_qubits is None:
                num_qubits = basis_num_qubits
            elif basis_num_qubits != num_qubits:
                raise ValueError(
                    "The number of qubits must be the same for all observables in the "
                    "observables array."
                )

    @classmethod
    def _validate_basis(cls, basis: str) -> None:
        """Validate a basis string.

        Args:
            basis: a basis string to validate.

        Raises:
            ValueError: If basis string contains invalid characters
        """
        # NOTE: the allowed basis characters can be overridden by modifying the class
        # attribute ALLOWED_BASIS
        allowed_pattern = _regex_match(cls.ALLOWED_BASIS)
        if not allowed_pattern.match(basis):
            invalid_pattern = _regex_invalid(cls.ALLOWED_BASIS)
            invalid_chars = list(set(invalid_pattern.findall(basis)))
            raise ValueError(
                f"Observable basis string '{basis}' contains invalid characters {invalid_chars},"
                f" allowed characters are {list(cls.ALLOWED_BASIS)}.",
            )


@lru_cache(1)
def _regex_match(allowed_chars: str) -> re.Pattern:
    """Return pattern for matching if a string contains only the allowed characters."""
    return re.compile(f"^[{re.escape(allowed_chars)}]*$")


@lru_cache(1)
def _regex_invalid(allowed_chars: str) -> re.Pattern:
    """Return pattern for selecting invalid strings"""
    return re.compile(f"[^{re.escape(allowed_chars)}]")
