
from anuga.coordinate_transforms.geo_reference import Geo_reference, DEFAULT_ZONE
from anuga.geometry.polygon import point_in_polygon, populate_polygon
from anuga.utilities.numerical_tools import ensure_numeric
import numpy as num
from anuga.geometry.polygon import inside_polygon
from anuga.geometry.polygon import polylist2points_verts
import anuga.utilities.log as log
import datetime
import warnings

# This is due to pmesh being a package and a module and
# the current dir being unknown
try:
    from anuga.pmesh.mesh import Pmesh
except ImportError:
    from .mesh import Pmesh


class PolygonError(Exception):
    pass


class SegmentError(Exception):
    pass


def create_mesh_from_regions(bounding_polygon,
                             boundary_tags,
                             maximum_triangle_area=None,
                             filename=None,
                             interior_regions=None,
                             interior_holes=None,
                             hole_tags=None,
                             poly_geo_reference=None,
                             mesh_geo_reference=None,
                             breaklines=None,
                             regionPtArea=None,
                             minimum_triangle_angle=28.0,
                             fail_if_polygons_outside=True,
                             use_cache=False,
                             verbose=True):

    """Create mesh from bounding polygons, and resolutions.

        Parameters
        ----------
        bounding_polygon : list of points
            Points in Eastings and Northings, relative to the poly_geo_reference.
        boundary_tags : dict
            Symbolic tags with lists of indices referring to segments associated 
            with each tag. If a segment is omitted an Exception will be raised.
        maximum_triangle_area : float, optional
            Maximal area per triangle for the bounding polygon, excluding the 
            interior regions.
        interior_regions : list of tuples, optional
            List of (polygon, resolution) tuples for each region to be separately 
            refined. Polygon lines should not cross or overlap, and should not be 
            close to each other.
        interior_holes : list of polygons, optional
            List of polygons for each hole.
        hole_tags : list, optional
            Boundary tags for the holes, see boundary_tags parameter.
        poly_geo_reference : Geo_reference, optional
            Geo reference of the bounding polygon and interior polygons.
            If None, assume absolute. Please pass one though, since absolute 
            references have a zone.
        mesh_geo_reference : Geo_reference, optional
            Geo reference of the mesh to be created. If None, one will be 
            automatically generated using the lower left hand corner of 
            bounding_polygon (absolute) as x and y values.
        breaklines : list of polygons, optional
            Lines to be preserved by the triangulation algorithm (e.g., coastlines, 
            walls). The polygons are not closed.
        regionPtArea : list, optional
            User-specified point-based regions with max area.
        minimum_triangle_angle : float, optional
            Minimum angle for triangles (default: 28.0).
        fail_if_polygons_outside : bool, optional
            If True (default), raise Exception when interior polygons fall outside 
            bounding polygon. If False, ignore these polygons.
        use_cache : bool, optional
            Whether to use caching (default: False).
        verbose : bool, optional
            Verbosity flag (default: True).

        Returns
        -------
        mesh : Mesh
            The generated mesh instance if no filename is given.

        Notes
        -----
        Interior regions should be fully nested, as overlaps may cause unintended 
        resolutions. This function does not allow segments to share points - use 
        underlying pmesh functionality for that.
        """

    if verbose:
        log.resource_usage_timing(log.logging.CRITICAL, "start_")
    if verbose:
        log.timingInfo("maximum_triangle_area, " + str(maximum_triangle_area))
    if verbose:
        log.timingInfo("minimum_triangle_angle, " +
                       str(minimum_triangle_angle))
    if verbose:
        log.timingInfo("startMesh, '%s'" % log.CurrentDateTime())

    # Build arguments and keyword arguments for use with caching or apply.
    args = (bounding_polygon,
            boundary_tags)

    kwargs = {'maximum_triangle_area': maximum_triangle_area,
              'filename': filename,
              'interior_regions': interior_regions,
              'interior_holes': interior_holes,
              'hole_tags': hole_tags,
              'poly_geo_reference': poly_geo_reference,
              'mesh_geo_reference': mesh_geo_reference,
              'minimum_triangle_angle': minimum_triangle_angle,
              'fail_if_polygons_outside': fail_if_polygons_outside,
              'breaklines': breaklines,
              'verbose': verbose,
              'regionPtArea': regionPtArea}   # FIXME (Ole): Should be bypassed one day. See ticket:14

    # Call underlying engine with or without caching
    if use_cache is True:
        try:
            from anuga.caching import cache
        except:
            msg = 'Caching was requested, but caching module' +\
                  'could not be imported'
            raise Exception(msg)

        m = cache(_create_mesh_from_regions,
                  args, kwargs,
                  verbose=verbose,
                  compression=False)
    else:
        m = _create_mesh_from_regions(*args, **kwargs)

    return m


def _create_mesh_from_regions(bounding_polygon,
                              boundary_tags,
                              maximum_triangle_area=None,
                              filename=None,
                              interior_regions=None,
                              interior_holes=None,
                              hole_tags=None,
                              poly_geo_reference=None,
                              mesh_geo_reference=None,
                              minimum_triangle_angle=28.0,
                              fail_if_polygons_outside=True,
                              breaklines=None,
                              verbose=True,
                              regionPtArea=None):
    """_create_mesh_from_regions - internal function.

    See create_mesh_from_regions for documentation.
    """

    # check the segment indexes - throw an error if they are out of bounds
    if boundary_tags is not None:
        max_points = len(bounding_polygon)
        for key in list(boundary_tags.keys()):
            if len([x for x in boundary_tags[key] if x > max_points-1]) >= 1:
                msg = 'Boundary tag %s has segment out of bounds. '\
                      % (str(key))
                msg += 'Number of points in bounding polygon = %d' % max_points
                raise SegmentError(msg)

        for i in range(max_points):
            found = False
            for tag in boundary_tags:
                if i in boundary_tags[tag]:
                    found = True
            if found is False:
                msg = 'Segment %d was not assigned a boundary_tag.' % i
                msg += 'Default tag "exterior" will be assigned to missing segment'
                warnings.warn(msg, UserWarning)
                if verbose:
                    log.critical('WARNING: %s' % msg)
                

    # In addition I reckon the polygons could be of class Geospatial_data
    # (DSG) If polygons were classes caching would break in places.

    # Simple check
    bounding_polygon = ensure_numeric(bounding_polygon, float)
    msg = 'Bounding polygon must be a list of points or an Nx2 array'
    assert len(bounding_polygon.shape) == 2, msg
    assert bounding_polygon.shape[1] == 2, msg

    #
    if interior_regions is not None:

        # Test that all the interior polygons are inside the
        # bounding_poly and throw out those that aren't fully
        # included.  #Note, Both poly's have the same geo_ref,
        # therefore don't take into account # geo_ref

        polygons_inside_boundary = []
        for interior_polygon, res in interior_regions:
            indices = inside_polygon(interior_polygon, bounding_polygon,
                                     closed=True, verbose=False)

            if len(indices) != len(interior_polygon):
                msg = 'Interior polygon %s is not fully inside'\
                      % (str(interior_polygon))
                msg += ' bounding polygon: %s.' % (str(bounding_polygon))

                if fail_if_polygons_outside is True:
                    raise PolygonError(msg)
                else:
                    msg += ' I will ignore it.'
                    log.critical(msg)

            else:
                polygons_inside_boundary.append([interior_polygon, res])

        # Record only those that were fully contained
        interior_regions = polygons_inside_boundary


    if interior_holes is not None:
        # Test that all the interior polygons are inside the bounding_poly
        for interior_polygon in interior_holes:

            # Test that we have a polygon
            if len(num.array(interior_polygon).flat) < 6:
                msg = 'Interior hole polygon %s has too few (<3) points.\n' \
                    % (str(interior_polygon))
                msg = msg + \
                    '(Insure that you have specified a LIST of interior hole polygons)'
                raise PolygonError(msg)

            indices = inside_polygon(interior_polygon, bounding_polygon,
                                     closed=True, verbose=False)

            if len(indices) != len(interior_polygon):
                msg = 'Interior polygon %s is outside bounding polygon: %s'\
                      % (str(interior_polygon), str(bounding_polygon))
                raise PolygonError(msg)

    # Resolve geo referencing
    if mesh_geo_reference is None:
        xllcorner = min(bounding_polygon[:, 0])
        yllcorner = min(bounding_polygon[:, 1])
        #
        if poly_geo_reference is None:
            zone = DEFAULT_ZONE
        else:
            zone = poly_geo_reference.get_zone()
            [(xllcorner, yllcorner)] = poly_geo_reference.get_absolute(
                [(xllcorner, yllcorner)])
        # create a geo_ref, based on the llc of the bounding_polygon
        mesh_geo_reference = Geo_reference(xllcorner=xllcorner,
                                           yllcorner=yllcorner,
                                           zone=zone)

    m = Pmesh(geo_reference=mesh_geo_reference)

    # build a list of discrete segments from the breakline polygons
    if breaklines is not None:
        points, verts = polylist2points_verts(breaklines)
        m.add_points_and_segments(points, verts)

    # Do bounding polygon
    m.add_region_from_polygon(bounding_polygon,
                              segment_tags=boundary_tags,
                              geo_reference=poly_geo_reference)

    # Find one point inside region automatically
    if interior_regions is not None:
        excluded_polygons = []
        for polygon, res in interior_regions:
            excluded_polygons.append(polygon)
    else:
        excluded_polygons = None

    # Convert bounding poly to absolute values
    # this sort of thing can be fixed with the geo_points class
    if poly_geo_reference is not None:
        bounding_polygon_absolute = \
            poly_geo_reference.get_absolute(bounding_polygon)
    else:
        bounding_polygon_absolute = bounding_polygon

    inner_point = point_in_polygon(bounding_polygon_absolute)
    inner = m.add_region(inner_point[0], inner_point[1])
    inner.setMaxArea(maximum_triangle_area)

    # Do interior regions
    if interior_regions is not None:
        for polygon, res in interior_regions:
            m.add_region_from_polygon(polygon,
                                      max_triangle_area=res,
                                      geo_reference=poly_geo_reference)

    # Do interior holes
    if interior_holes is not None:
        for n, polygon in enumerate(interior_holes):
            try:
                tags = hole_tags[n]
            except:
                tags = {}
            m.add_hole_from_polygon(polygon,
                                    segment_tags=tags,
                                    geo_reference=poly_geo_reference)

    # 22/04/2014
    # Add user-specified point-based regions with max area
    if(regionPtArea is not None):
        for i in range(len(regionPtArea)):
            inner = m.add_region(regionPtArea[i][0], regionPtArea[i][1])
            inner.setMaxArea(regionPtArea[i][2])

    # NOTE (Ole): This was moved here as it is annoying if mesh is always
    # stored irrespective of whether the computation
    # was cached or not. This caused Domain to
    # recompute as it has meshfile as a dependency

    # Decide whether to store this mesh or return it

    if filename is None:
        return m
    else:
        if verbose:
            log.critical("Generating mesh to file '%s'" % filename)

        m.generate_mesh(minimum_triangle_angle=minimum_triangle_angle,
                        verbose=verbose)
        m.export_mesh_file(filename)

        return m
