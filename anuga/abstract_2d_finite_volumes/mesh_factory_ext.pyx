#cython: wraparound=False, boundscheck=False, cdivision=True, profile=False, nonecheck=False, overflowcheck=False, cdivision_warnings=False, unraisable_tracebacks=False
import cython
from libc.stdint cimport int64_t

# import both numpy and the Cython declarations for numpy
import numpy as np
cimport numpy as np

def rectangular_cross_construct(np.ndarray[double, ndim=1, mode="c"] params not None,\
								np.ndarray[double, ndim=1, mode="c"] origin not None,\
								np.ndarray[double, ndim=2, mode="c"] points not None,\
								np.ndarray[int64_t, ndim=2, mode="c"] elements not None,\
								np.ndarray[int64_t, ndim=2, mode="c"] neighbours not None):


	cdef int64_t m, n, i, j, v1, v2 ,v3 ,v4, v5
	cdef int64_t numPoints, numElements, base
	cdef double len1, len2, delta1, delta2, x, y

	m = int(params[0])
	n = int(params[1])
	len1 = params[2]
	len2 = params[3]

	cdef np.ndarray[int64_t, ndim=2, mode="c"] vertices = np.ascontiguousarray(np.zeros((m+1,n+1),dtype=np.int64))

	delta1 = len1/m
	delta2 = len2/n

	numPoints = 0
	for i in range(m+1):
		for j in range(n+1):
			vertices[i,j] = numPoints
			points[numPoints,0] = i*delta1 + origin[0]
			points[numPoints,1] = j*delta2 + origin[1]
			numPoints += 1

	boundary = {}
	numElements = 0
	for i in range(m):
		for j in range(n):
			v1 = vertices[i,j+1]
			v2 = vertices[i,j]
			v3 = vertices[i+1,j+1]
			v4 = vertices[i+1,j]
			x = (points[v1,0] + points[v2,0] + points[v3,0] + points[v4,0])*0.25
			y = (points[v1,1] + points[v2,1] + points[v3,1] + points[v4,1])*0.25

			# Create centre point
			v5 = numPoints
			points[numPoints,0] = x
			points[numPoints,1] = y
			numPoints += 1

			# base index for the 4 triangles of this cell
			base = numElements

			# Create left triangle  [v2, v5, v1]
			if i == 0:
				boundary[(numElements,1)] = "left"

			elements[numElements,0] = v2
			elements[numElements,1] = v5
			elements[numElements,2] = v1
			numElements += 1

			# Create bottom triangle  [v4, v5, v2]
			if j == 0:
				boundary[(numElements,1)] = "bottom"

			elements[numElements,0] = v4
			elements[numElements,1] = v5
			elements[numElements,2] = v2
			numElements += 1

			# Create right triangle  [v3, v5, v4]
			if i == m-1:
				boundary[(numElements,1)] = "right"

			elements[numElements,0] = v3
			elements[numElements,1] = v5
			elements[numElements,2] = v4
			numElements += 1

			# Create top triangle  [v1, v5, v3]
			if j == n-1:
				boundary[(numElements,1)] = "top"

			elements[numElements,0] = v1
			elements[numElements,1] = v5
			elements[numElements,2] = v3
			numElements += 1

			# -------------------------------------------------------
			# Neighbour connectivity for the 4 triangles of cell (i,j)
			# base+0 = left   [v2,v5,v1]
			# base+1 = bottom [v4,v5,v2]
			# base+2 = right  [v3,v5,v4]
			# base+3 = top    [v1,v5,v3]
			#
			# Edge k is opposite vertex k:
			#   edge 0: v1-v2 side, edge 1: v0-v2 side, edge 2: v0-v1 side
			# -------------------------------------------------------

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

	return boundary


def rectangular_construct(np.ndarray[double, ndim=1, mode="c"] params not None,\
						np.ndarray[double, ndim=1, mode="c"] origin not None,\
						np.ndarray[double, ndim=2, mode="c"] points not None,\
						np.ndarray[int64_t, ndim=2, mode="c"] elements not None,\
						np.ndarray[int64_t, ndim=2, mode="c"] neighbours not None):
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
	"""

	cdef int64_t m, n, i, j
	cdef int64_t a, b, c, d       # corner vertex indices for cell (i,j)
	cdef int64_t nt_lower, nt_upper
	cdef double len1, len2, delta1, delta2

	m = int(params[0])
	n = int(params[1])
	len1 = params[2]
	len2 = params[3]

	delta1 = len1/m
	delta2 = len2/n

	# Fill point coordinates: index(i,j) = j + i*(n+1)
	for i in range(m+1):
		for j in range(n+1):
			points[j + i*(n+1), 0] = i*delta1 + origin[0]
			points[j + i*(n+1), 1] = j*delta2 + origin[1]

	boundary = {}

	for i in range(m):
		for j in range(n):
			# Corner vertex indices
			a = j     + i    *(n+1)   # index(i,   j  )  bottom-left
			b = j + 1 + i    *(n+1)   # index(i,   j+1)  top-left
			c = j     + (i+1)*(n+1)   # index(i+1, j  )  bottom-right
			d = j + 1 + (i+1)*(n+1)   # index(i+1, j+1)  top-right

			nt_lower = 2*(i*n + j)
			nt_upper = nt_lower + 1

			# --- lower triangle [c, d, a] ---
			if j == 0:
				boundary[(nt_lower, 1)] = 'bottom'
			if i == m-1:
				boundary[(nt_lower, 2)] = 'right'
			elements[nt_lower, 0] = c
			elements[nt_lower, 1] = d
			elements[nt_lower, 2] = a

			# --- upper triangle [b, a, d] ---
			if i == 0:
				boundary[(nt_upper, 2)] = 'left'
			if j == n-1:
				boundary[(nt_upper, 1)] = 'top'
			elements[nt_upper, 0] = b
			elements[nt_upper, 1] = a
			elements[nt_upper, 2] = d

			# -------------------------------------------------------
			# Neighbour connectivity
			# lower [c,d,a]: edge 0 opp c (d-a), edge 1 opp d (c-a), edge 2 opp a (c-d)
			# upper [b,a,d]: edge 0 opp b (a-d), edge 1 opp a (b-d), edge 2 opp d (b-a)
			# -------------------------------------------------------

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

	return boundary
