# Numerical methods lab

How far can a least-squares solution be from the right coefficients while still fitting the data? In a seeded example with condition number near 10⁸, the normal equations returned a relative residual of about 6.3 × 10⁻⁸ and a relative coefficient error of about 3. QR recovered the coefficients to about 1.2 × 10⁻¹⁰ on the same problem.

`conditioning.py` builds a 96-by-8 matrix with prescribed singular values and uses a known coefficient vector to generate the target. It compares the normal equations, reduced QR, and NumPy's `lstsq` as the condition number increases. Forming XᵀX squares the condition number. Reporting the residual and coefficient error side by side shows what the residual alone can hide.

```sh
python -m pip install -r requirements.txt
python conditioning.py
python -m unittest -v
```

## Coursework examples

`coursework/` contains seven standalone Python examples from numerical analysis. They cover bracketed roots, Newton and secant iteration, multiple roots, interpolation, finite differences, trapezoid quadrature, and pivoting. Run any script directly. The interpolation and finite-difference examples open Matplotlib figures.

```sh
python coursework/pivoting.py
python coursework/interpolation.py
```
