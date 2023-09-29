# Julian's helpers functions for project 3
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
from skimage.draw import polygon2mask
from skimage.draw import polygon
from scipy.interpolate import RegularGridInterpolator
import json

def read_im(filepath):
    if filepath[len(filepath)-3:] == 'tif':
        return mpimg.imread(filepath)/65536.0 
    if filepath[len(filepath)-3:] == 'jpg' or filepath[len(filepath)-4:] == 'jpeg':
        return mpimg.imread(filepath)/255.0
    else:
        return mpimg.imread(filepath)

def Display2Images(im1, im2, name1, name2, size):
    # Set the desired width and height for the figure
    fig_width = 2*size # Width in inches
    fig_height = size  # Height in inches

    # Create a figure with two subplots and set the figsize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))

    # Display image1 in the first subplot
    ax1.imshow(im1)
    ax1.set_title(name1)

    # Display image2 in the second subplot
    ax2.imshow(im2)
    ax2.set_title(name2)

    # Adjust spacing between subplots 
    plt.tight_layout()

    # Show the figure
    plt.show()

def Display3Images(im1, im2, im3, name1, name2, name3, size):
    # Set the desired width and height for the figure
    fig_width = 2 * size# Width in inches
    fig_height = size # Height in inches

    # Create a figure with three subplots and set the figsize
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(fig_width, fig_height))

    # Display image1 in the first subplot
    ax1.imshow(im1)
    ax1.set_title(name1)

    # Display image2 in the second subplot
    ax2.imshow(im2)
    ax2.set_title(name2)

    # Display image3 in the third subplot
    ax3.imshow(im3)
    ax3.set_title(name3)

    # Adjust spacing between subplots (optional)
    plt.tight_layout()

    # Show the figure
    plt.show()

def GetPointsFromJSON(json_file_path):
    try:
        # Read the JSON file
        with open(json_file_path, 'r') as file:
            data = json.load(file)

        # Extract the points from the JSON data
        im1_points = data.get('im1Points', [])
        im2_points = data.get('im2Points', [])

        return im1_points, im2_points
    except FileNotFoundError:
        print(f"The file '{json_file_path}' was not found.")
        return [], []

def GetTriangulationObject(pts):
    """
    Returns a maplotlib triangulation object. 
    Assumes pts is a NumPy array.
    """
    tris = Delaunay(pts).simplices
    triangulation = mtri.Triangulation(pts[:, 0], pts[:, 1], tris) 
    return triangulation

def GetTriangulationObjectFromExistingDelaunay(pts, delaunay):
    """
    Returns a maplotlib triangulation object. 
    Assumes pts is a NumPy array.
    Assumes scipy Delaunay triangulation object.
    """
    tris = delaunay.simplices
    triangulation = mtri.Triangulation(pts[:, 0], pts[:, 1], tris) 
    # Can you explain why the previous line works? Specifically the array slicing. 
    return triangulation

def DisplayFaceTrisAndPts(face_im, tris, pts, title):
    """
    Assumes tris is a matplotlib triangulation object.
    Assumes pts is a NumPy array.
    Assumes title is a string.
    """

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(face_im)

    ax.triplot(tris, 'go--', alpha=0.5)

    # Plot the original points
    ax.plot(pts[:, 1], pts[:, 0], 'ro')

    # Add labels and title
    ax.set_xlabel('X-Axis')
    ax.set_ylabel('Y-Axis')
    ax.set_title(title)

    # Show the plot
    plt.show()

def GetMidwayFace(pts1, pts2):
    """
    Returns the average of two sets of points.
    Assumes pts1 and pts2 are NumPy arrays.
    """
    midway_pts = np.zeros_like(pts1)
    for i in range(len(midway_pts)):
        mean_point = [pts1[i][0] + pts2[i][0], pts1[i][1] + pts2[i][1]]
        mean_point = [mean_point[0]//2, mean_point[1]//2]
        midway_pts[i] = mean_point
    return midway_pts

def ComputeAffine(tri1, tri2):
    """
    Assumes pts1 and pts2 are NumPy arrays that contain 3 points (tuples like (x,y)) 
        that represent the vertices of triangles for face1 and face2. Also, it is assumed
        that the triangles at the same index of pts1 and pts2 have the same "meaning" in both
        images/faces. I.e. both correspond to the same feature of the face.

    Assumes that point A and point A' are correpsndences of one another, and the same for B, B' and C, C'.
    
    Returns the transformation matrix from tri1->tri2.
    Assumes that tri1 and 2 are Np arrays where each point is a row,
        and the points are represented in homogenous coordinates.
    """
    # Solve the system of linear equation that maps triangle 1 to triangle 2
    A = tri1.T
    B = tri2.T 

    # Need to get the pseudo-inverse because normal inverse is too strict.
    A_pseudo_inv = np.linalg.pinv(A)

    # Calculate the transformation matrix T
    # WHY DOES THIS WORK SO WELL?
    T = B @ A_pseudo_inv

    # Remove projective component; turn into affine transformation.
    T[2][0] = 0
    T[2][1] = 0
    T[2][2] = 1

    return T 

def ConvertPointsToHomogenous(points):
    """
    Takes a list of points, assumed to be a NumPy array. 
        Each point is a vector of size 2.
    Returns the same list of points, but as homogenous coordinates.
        Each homogenous point is a vector of size 3, where the 3rd entry is w=1.
    """
    # Append a column of ones to the array
    homogeneous_points = np.column_stack((points, np.ones(points.shape[0])))
    
    return homogeneous_points

def ConvertTrianglesToHomogenous(triangles):
    """
    Takes a list of triangles, each of which is a list of points.
        Assumes that the points aren't homogenous, and that everything is a NumPy array.
    Returns the same list of triangles, with all points replaced by homogenous versions.
    """
    for tri in range(triangles.shape[0]):
        triangles[tri] = ConvertPointsToHomogenous(triangles[tri])
    
    return triangles

def warp(im, im_pts, target_pts, tri):
    """
    Does the image warping from im to a target shape.
    tri should be a scipy Delaunay triangulation object.
    """
    x_dim = np.arange(0, im.shape[1])
    y_dim = np.arange(0, im.shape[0])

    r_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,0], method='linear', bounds_error=False, fill_value=None)
    g_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,1], method='linear', bounds_error=False, fill_value=None)
    b_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,2], method='linear', bounds_error=False, fill_value=None)

    def interpolate_rgb(yx_coord):
        r = r_interpolator(yx_coord)
        g = g_interpolator(yx_coord)
        b = b_interpolator(yx_coord)
        return np.array((r, g, b)).reshape(1,3)

    def interpolate_rgb_h(homogenous):
        return interpolate_rgb((homogenous[0], homogenous[1]))

    transformed_tris = []
    target_tri_masks = []
    # For all triangles in the triangulation:
    for i in range(len(tri.simplices)):
        triangle_simplex = tri.simplices[i]
        # Get the corresponding triangle shape from george's face.
        # TODO: Replace this with a call to polygon(triangle) which returns the pixels in the triangle.
        # Then just do the inverse warp for that array and set those pixels in the result image to the interpolation of the inverse coordinates. 
        #target_tri_mask = polygon2mask(im.shape[:2], target_pts[triangle_simplex])
        #target_tri_masks.append(target_tri_mask)
        # Compute transformation from one of my triangles to the corresponding triangle in george's face.
        im_tri = im_pts[triangle_simplex]
        target_tri = target_pts[triangle_simplex]
        target_tri_coords = polygon()
        # Convert triangle points to homogenous coordinates.
        im_tri_h = ConvertPointsToHomogenous(im_tri)
        target_tri_h = ConvertPointsToHomogenous(target_tri)

        # Compute affine transfromation matrix from my triangle to george's triangle.
        T = ComputeAffine(im_tri_h, target_tri_h)
        # Get the inverse of the transformation matrix.
        T_inv = np.linalg.inv(T)

        # Initialize transformed image to zeros.
        transformed_tri = np.zeros_like(im)
        # For every pixel coordinate position that is True in the mask of the transformed image (g_mask1)
        for y in range(target_tri_mask.shape[0]):
            for x in range(target_tri_mask.shape[1]):
                if target_tri_mask[y][x]:
                    # Get inverse coord
                    coord = np.array([y, x, 1])
                    inverse_coord = T_inv @ coord
                    # Interpolate the value of the pixel at the inverse_coord.
                    pixel_value = interpolate_rgb_h(inverse_coord)
                    # set this coord to px value of inverse_coord
                    transformed_tri[y][x] = pixel_value                 
        transformed_tris.append(transformed_tri)

    im_transformed = np.zeros_like(im)
    for tri_im in transformed_tris:
        im_transformed += tri_im

    # Add up the masks
    mask_weights = np.zeros_like(target_tri_masks[0])
    for mask in target_tri_masks:
        mask_weights += mask
    mask_weights[mask_weights < 1] = 1
    # Divide the result image by the mask weights (overlapping masks)
    im_transformed[:, :, 0] /= mask_weights
    im_transformed[:, :, 1] /= mask_weights
    im_transformed[:, :, 2] /= mask_weights
    return im_transformed

def cross_dissolve(im1, im2, frac):
    """
    Compute the weighted average of two images by amount frac.
    res = (1-frac)*im1 + frac*im2
    """
    return (1-frac)*im1 + frac*im2

def morph(im1, im2, im1_pts, im2_pts, tri, warp_frac, dissolve_frac):
    """
    Inputs:
        im1, im2: Images of the same size to be morphed together.
        im1_pts, im2_pts: Corresponding points in each image. 
            They should be NumPy arrays of shape (N, 2).
        tri: Triangulation object returned by scipy.spatial.Delaunay(). 
            Note: The whether the triangulation is for im1 or im2 doesn't matter, what's important is that the indices
                of the triangles are consistent between the two correspondance point lists.
        warp_frac: Fraction of "warping" to apply to im1_pts and im2_pts. Lies in range [0, 1] for interpolation.
        dissolve_frac: Fraction of cross-dissolve to apply to image. Lies in range [0, 1] for interpolation.
    Returns:
        Intermediate image morphed from im1 and im2 by fractional ammounts warp_frac and dissolve_frac.
    """
    # 1. Warp im1 and im2 to the shape of tri using the weighted average im1_pts and im2_pts by ammount warp_frac.
    # Take the weighted average of the correspondance points 
    intermediate_pts = (1-warp_frac)*im1_pts + warp_frac*im2_pts
    # Warp the images to the intermediate points 
    im1_warp = warp(im1, im1_pts, intermediate_pts, tri) 
    im2_warp = warp(im2, im2_pts, intermediate_pts, tri) 

    # Check the data types of the images

    # 2. Cross-dissolve the two warped images using dissolve_frac.
    #   Note: Cross-dissolve means that at dissolve_frac=0, the output is only im1, and at dissolve_frac=1, the output is only im2.
    result = cross_dissolve(im1_warp, im2_warp, dissolve_frac)
    # Check the data type
    return result