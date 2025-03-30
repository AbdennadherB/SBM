# README.md

#  GPU-Accelerated Electromagnetic Field Solver using the Singular Boundary Method

This project implements a **vectorized, meshless 3D solver** for the **Helmholtz equation** using the **Singular Boundary Method (SBM)**. It models electromagnetic scattering on **Perfect Electric Conductors (PEC)** by computing Green's functions and their derivatives, leveraging **PyTorch** and **GPU acceleration** for high performance.

---

##  Key Features
- Works for **3D structures** — not limited to 2D approximations
- **Meshless method** — uses SBM, avoids traditional meshing
- **STL-to-point cloud** conversion — for defining surface geometry
- **GPU-accelerated** via PyTorch (up to **10x faster** than CPU)
- **Green's function formalism** for Helmholtz equation
- Full computation of scattered **electric fields**
- Fast, vectorized matrix construction and solving

---

##  Project Structure

```
SBM/
├── src/
│   ├── solver_functions.py  # SBM matrix builder and field computation
│   └── main.py              # Main execution & plot scripts 
├── data/                    # STL models and .mat files
├── README.md
└── .gitignore
```

---

## Dependencies
- `torch`
- `open3d`
- `numpy`
- `matplotlib`

---

## Output

- Electric field (Ex, Ey, Ez) distributions on a 2D grid
- Magnitude and phase contour plots
- Comparison of real and imaginary field components

---

## Notes

- Works on **Perfect Electric Conductors (PEC)**
- Can be extended for more general boundary conditions
- Designed for research-level simulations in electromagnetics and optics

---

## Author

**Bilel Abdennadher**  
_PhD Candidate in Physics / Computational Electromagnetics_  
Adolphe Merkle Institute AMI | Massachusetts Institute Of Technology MIT
