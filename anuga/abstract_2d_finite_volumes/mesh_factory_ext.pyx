#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False, initializedcheck=False, language_level=3
import cython
from libc.stdint cimport int64_t

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np


def rectangular_cross_construct(double[::1] params not None,
								double[::1] origin not None,
								double[:,::1] points not None,
								int64_t[:,::1] elements not None,
								int64_t[:,::1] neighbours not None,
								int64_t[:,::1] neighbour_edges not None):
	"""
	Construct a rectangular cross mesh with 4 triangles per cell.

	For each cell (i,j) four triangles are created:
	  base+0 = left   [v2, v5, v1]
	  base+1 = bottom [v4, v5, v2]
	  base+2 = right  [v3, v5, v4]
	  base+3 = top    [v1, v5, v3]
	where v1..v4 are the cell corners and v5 is the cell centre.

	Corner indices: v2=i*(n+1)+j, v1=i*(n+1)+(j+1),
	                v4=(i+1)*(n+1)+j, v3=(i+1)*(n+1)+(j+1)
	Centre index:   v5 = (m+1)*(n+1) + i*n + j

	Neighbour_edges analytical pattern (edge k opposite vertex k):
	  For all valid (non-boundary) edges, the reciprocal edge index is 2 - k.
	  Boundary edges retain -1.
	"""

	cdef int64_t m, n, i, j, v1, v2, v3, v4, v5, base, num_corners
	cdef double len1, len2, delta1, delta2, ox, oy

	m = <int64_t>params[0]
	n = <int64_t>params[1]
	len1 = params[2]
	len2 = params[3]
	ox = origin[0]
	oy = origin[1]

	delta1 = len1 / m
	delta2 = len2 / n
	num_corners = (m + 1) * (n + 1)

	# -------------------------------------------------------
	# Build boundary dict (Python object; must hold the GIL).
	# Only O(m+n) entries, so this is negligible vs the main loop.
	# -------------------------------------------------------
	boundary = {}
	for j in range(n):
		boundary[(4 * (0     * n + j) + 0, 1)] = 'left'    # i == 0
		boundary[(4 * ((m-1) * n + j) + 2, 1)] = 'right'   # i == m-1
	for i in range(m):
		boundary[(4 * (i * n + 0    ) + 1, 1)] = 'bottom'  # j == 0
		boundary[(4 * (i * n + (n-1)) + 3, 1)] = 'top'     # j == n-1

	# -------------------------------------------------------
	# Fill points, elements, neighbours, neighbour_edges.
	# All operations are on C arrays — release the GIL.
	# -------------------------------------------------------
	with nogil:
		# Corner points: index(i,j) = i*(n+1) + j
		for i in range(m + 1):
			for j in range(n + 1):
				points[i*(n+1) + j, 0] = i * delta1 + ox
				points[i*(n+1) + j, 1] = j * delta2 + oy

		for i in range(m):
			for j in range(n):
				# Corner vertex indices (direct formula, no lookup array)
				v2 = i    *(n+1) + j        # (i,   j  )  bottom-left
				v1 = i    *(n+1) + (j+1)    # (i,   j+1)  top-left
				v4 = (i+1)*(n+1) + j        # (i+1, j  )  bottom-right
				v3 = (i+1)*(n+1) + (j+1)    # (i+1, j+1)  top-right

				# Centre point: deterministic index, direct coordinate
				v5 = num_corners + i*n + j
				points[v5, 0] = (i + 0.5) * delta1 + ox
				points[v5, 1] = (j + 0.5) * delta2 + oy

				base = 4 * (i*n + j)

				# Elements
				elements[base+0, 0] = v2; elements[base+0, 1] = v5; elements[base+0, 2] = v1  # left
				elements[base+1, 0] = v4; elements[base+1, 1] = v5; elements[base+1, 2] = v2  # bottom
				elements[base+2, 0] = v3; elements[base+2, 1] = v5; elements[base+2, 2] = v4  # right
				elements[base+3, 0] = v1; elements[base+3, 1] = v5; elements[base+3, 2] = v3  # top

				# --- left triangle [v2, v5, v1] ---
				# edge 0 (opp v2): v5-v1  → top tri of same cell
				neighbours[base+0, 0] = base+3
				# edge 1 (opp v5): v2-v1  → right tri of cell (i-1,j), or boundary
				if i == 0:
					neighbours[base+0, 1] = -1
				else:
					neighbours[base+0, 1] = 4*((i-1)*n+j)+2
				# edge 2 (opp v1): v2-v5  → bottom tri of same cell
				neighbours[base+0, 2] = base+1

				# --- bottom triangle [v4, v5, v2] ---
				# edge 0 (opp v4): v5-v2  → left tri of same cell
				neighbours[base+1, 0] = base+0
				# edge 1 (opp v5): v4-v2  → top tri of cell (i,j-1), or boundary
				if j == 0:
					neighbours[base+1, 1] = -1
				else:
					neighbours[base+1, 1] = 4*(i*n+(j-1))+3
				# edge 2 (opp v2): v4-v5  → right tri of same cell
				neighbours[base+1, 2] = base+2

				# --- right triangle [v3, v5, v4] ---
				# edge 0 (opp v3): v5-v4  → bottom tri of same cell
				neighbours[base+2, 0] = base+1
				# edge 1 (opp v5): v3-v4  → left tri of cell (i+1,j), or boundary
				if i == m-1:
					neighbours[base+2, 1] = -1
				else:
					neighbours[base+2, 1] = 4*((i+1)*n+j)+0
				# edge 2 (opp v4): v3-v5  → top tri of same cell
				neighbours[base+2, 2] = base+3

				# --- top triangle [v1, v5, v3] ---
				# edge 0 (opp v1): v5-v3  → right tri of same cell
				neighbours[base+3, 0] = base+2
				# edge 1 (opp v5): v1-v3  → bottom tri of cell (i,j+1), or boundary
				if j == n-1:
					neighbours[base+3, 1] = -1
				else:
					neighbours[base+3, 1] = 4*(i*n+(j+1))+1
				# edge 2 (opp v3): v1-v5  → left tri of same cell
				neighbours[base+3, 2] = base+0

				# -------------------------------------------------------
				# Neighbour_edges: for all valid edges the reciprocal edge
				# index is always 2 - k (edge 0 <-> edge 2, edge 1 <-> edge 1).
				# Boundary edges (neighbours < 0) keep -1.
				# -------------------------------------------------------
				neighbour_edges[base+0, 0] = 2
				if i == 0:
					neighbour_edges[base+0, 1] = -1
				else:
					neighbour_edges[base+0, 1] = 1
				neighbour_edges[base+0, 2] = 0

				neighbour_edges[base+1, 0] = 2
				if j == 0:
					neighbour_edges[base+1, 1] = -1
				else:
					neighbour_edges[base+1, 1] = 1
				neighbour_edges[base+1, 2] = 0

				neighbour_edges[base+2, 0] = 2
				if i == m-1:
					neighbour_edges[base+2, 1] = -1
				else:
					neighbour_edges[base+2, 1] = 1
				neighbour_edges[base+2, 2] = 0

				neighbour_edges[base+3, 0] = 2
				if j == n-1:
					neighbour_edges[base+3, 1] = -1
				else:
					neighbour_edges[base+3, 1] = 1
				neighbour_edges[base+3, 2] = 0

	return boundary


def rectangular_construct(double[::1] params not None,
						double[::1] origin not None,
						double[:,::1] points not None,
						int64_t[:,::1] elements not None,
						int64_t[:,::1] neighbours not None,
						int64_t[:,::1] neighbour_edges not None):
	"""
	Construct a rectangular mesh with 2 triangles per cell.

	For each cell (i,j) two triangles are created:
	  lower [c, d, a]  (bottom-right half of cell)
	  upper [b, a, d]  (top-left half of cell)
	where a=index(i,j), b=index(i,j+1), c=index(i+1,j), d=index(i+1,j+1)
	and index(i,j) = j + i*(n+1).

	Boundary tags:
	  lower edge 1 -> 'bottom' (j==0)
	  lower edge 2 -> 'right'  (i==m-1)
	  upper edge 1 -> 'top'    (j==n-1)
	  upper edge 2 -> 'left'   (i==0)

	Neighbour connectivity (edge k is opposite vertex k):
	  lower edge 0 <-> upper edge 0  (shared diagonal, same cell)
	  lower edge 1 <-> upper edge 1 of cell (i,j-1), or -1
	  lower edge 2 <-> upper edge 2 of cell (i+1,j), or -1
	  upper edge 1 <-> lower edge 1 of cell (i,j+1), or -1
	  upper edge 2 <-> lower edge 2 of cell (i-1,j), or -1

	Neighbour_edges analytical pattern:
	  For all valid edges, the reciprocal edge index equals k (edge k <-> edge k).
	  Boundary edges retain -1.
	"""

	cdef int64_t m, n, i, j
	cdef int64_t a, b, c, d       # corner vertex indices for cell (i,j)
	cdef int64_t nt_lower, nt_upper
	cdef double len1, len2, delta1, delta2, ox, oy

	m = <int64_t>params[0]
	n = <int64_t>params[1]
	len1 = params[2]
	len2 = params[3]
	ox = origin[0]
	oy = origin[1]

	delta1 = len1 / m
	delta2 = len2 / n

	# -------------------------------------------------------
	# Build boundary dict (Python object; must hold the GIL).
	# -------------------------------------------------------
	boundary = {}
	for i in range(m):
		boundary[(2*(i*n + 0),       1)] = 'bottom'  # j == 0,   lower edge 1
		boundary[(2*(i*n + (n-1))+1, 1)] = 'top'    # j == n-1, upper edge 1
	for j in range(n):
		boundary[(2*((m-1)*n + j),   2)] = 'right'  # i == m-1, lower edge 2
		boundary[(2*(j)          +1, 2)] = 'left'   # i == 0,   upper edge 2

	# -------------------------------------------------------
	# Fill points, elements, neighbours, neighbour_edges.
	# -------------------------------------------------------
	with nogil:
		# Corner points: index(i,j) = j + i*(n+1)
		for i in range(m + 1):
			for j in range(n + 1):
				points[j + i*(n+1), 0] = i * delta1 + ox
				points[j + i*(n+1), 1] = j * delta2 + oy

		for i in range(m):
			for j in range(n):
				# Corner vertex indices
				a = j     +  i   *(n+1)   # index(i,   j  )  bottom-left
				b = j + 1 +  i   *(n+1)   # index(i,   j+1)  top-left
				c = j     + (i+1)*(n+1)   # index(i+1, j  )  bottom-right
				d = j + 1 + (i+1)*(n+1)   # index(i+1, j+1)  top-right

				nt_lower = 2*(i*n + j)
				nt_upper = nt_lower + 1

				# Elements
				elements[nt_lower, 0] = c; elements[nt_lower, 1] = d; elements[nt_lower, 2] = a
				elements[nt_upper, 0] = b; elements[nt_upper, 1] = a; elements[nt_upper, 2] = d

				# lower edge 0 (diagonal d-a) <-> upper edge 0 (same cell)
				neighbours[nt_lower, 0] = nt_upper
				# lower edge 1 (bottom c-a) <-> upper edge 1 of cell (i,j-1), or boundary
				if j == 0:
					neighbours[nt_lower, 1] = -1
				else:
					neighbours[nt_lower, 1] = 2*(i*n + (j-1)) + 1
				# lower edge 2 (right c-d) <-> upper edge 2 of cell (i+1,j), or boundary
				if i == m-1:
					neighbours[nt_lower, 2] = -1
				else:
					neighbours[nt_lower, 2] = 2*((i+1)*n + j) + 1

				# upper edge 0 (diagonal a-d) <-> lower edge 0 (same cell)
				neighbours[nt_upper, 0] = nt_lower
				# upper edge 1 (top b-d) <-> lower edge 1 of cell (i,j+1), or boundary
				if j == n-1:
					neighbours[nt_upper, 1] = -1
				else:
					neighbours[nt_upper, 1] = 2*(i*n + (j+1))
				# upper edge 2 (left b-a) <-> lower edge 2 of cell (i-1,j), or boundary
				if i == 0:
					neighbours[nt_upper, 2] = -1
				else:
					neighbours[nt_upper, 2] = 2*((i-1)*n + j)

				# -------------------------------------------------------
				# Neighbour_edges: for all valid edges the reciprocal edge
				# index equals k (lower edge k <-> upper edge k, and vice versa).
				# Boundary edges (neighbours < 0) keep -1.
				# -------------------------------------------------------
				neighbour_edges[nt_lower, 0] = 0   # diagonal: lower edge 0 <-> upper edge 0
				if j == 0:
					neighbour_edges[nt_lower, 1] = -1
				else:
					neighbour_edges[nt_lower, 1] = 1
				if i == m-1:
					neighbour_edges[nt_lower, 2] = -1
				else:
					neighbour_edges[nt_lower, 2] = 2

				neighbour_edges[nt_upper, 0] = 0   # diagonal: upper edge 0 <-> lower edge 0
				if j == n-1:
					neighbour_edges[nt_upper, 1] = -1
				else:
					neighbour_edges[nt_upper, 1] = 1
				if i == 0:
					neighbour_edges[nt_upper, 2] = -1
				else:
					neighbour_edges[nt_upper, 2] = 2

	return boundary
