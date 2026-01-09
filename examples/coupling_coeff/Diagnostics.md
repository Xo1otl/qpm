### Issue: Coordinate System Mismatch

**Observation**
There is a conflict between the coordinate system defined in the documentation and the standard convention used by the solver scripts.

* **Documentation:** Defines $x$ **as Depth** (Vertical/Extraordinary axis).
* **Code/Solver:** Follows standard simulation convention where $y$ **is Depth**.

**Diagnostic Evidence**

* `calc_efields_ey.py` **succeeded**: It correctly solved for the vertical field ($E_y$), which corresponds to the guided extraordinary index (n_e).
* `calc_efields_ex.py` **failed**: It attempted to solve for the horizontal field ($E_x$), which sees the unguided ordinary index (n_o).

**Action Required**
**Update the documentation** to align with standard conventions and the current codebase:

1. Redefine ** as Depth** (Vertical).
2. Redefine ** as Width** (Transverse).
3. Map the guided TM mode to $E_y$ instead of $E_x$.
