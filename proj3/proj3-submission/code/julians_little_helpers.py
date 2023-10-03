# Julian's helpers functions for project 3
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
from skimage.draw import polygon2mask
from skimage.draw import polygon
from skimage.transform import resize
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
    # Matplot uses xy format
    triangulation = mtri.Triangulation(pts[:, 1], pts[:, 0], tris) 
    return triangulation

def GetTriangulationObjectFromExistingDelaunay(pts, delaunay):
    """
    Returns a maplotlib triangulation object. 
    Assumes pts is a NumPy array.
    Assumes scipy Delaunay triangulation object.
    """
    tris = delaunay.simplices
    triangulation = mtri.Triangulation(pts[:, 1], pts[:, 0], tris) 
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

    # Save the plot
    #plt.savefig(f"{title}.jpg")

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

def get_interpolator(im):
    x_dim = np.arange(0, im.shape[1])
    y_dim = np.arange(0, im.shape[0])

    r_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,0], method='linear', bounds_error=False, fill_value=None)
    g_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,1], method='linear', bounds_error=False, fill_value=None)
    b_interpolator = RegularGridInterpolator((y_dim, x_dim), im[:,:,2], method='linear', bounds_error=False, fill_value=None)

    def interpolate_rgb(yx_coord):
        r = r_interpolator(yx_coord)
        g = g_interpolator(yx_coord)
        b = b_interpolator(yx_coord)
        return np.column_stack((r, g, b))

    def interpolate_rgb_h(homogenous):
        return interpolate_rgb((homogenous[0], homogenous[1]))
    
    return interpolate_rgb

def warp(im, im_pts, target_pts, target_shape, tri):
    """
    Does the image warping from im to a target shape.
    tri should be a scipy Delaunay triangulation object.
    """
    interpolate_rgb = get_interpolator(im)
    transformed_im = np.zeros(target_shape)
    # For all triangles in the triangulation:
    for i in range(len(tri.simplices)):
        triangle_simplex = tri.simplices[i]
        # Get the corresponding triangles.
        im_tri = im_pts[triangle_simplex]
        target_tri = target_pts[triangle_simplex]
        target_tri_pixels = polygon(target_tri[:, 0], target_tri[:, 1])
        # Convert triangle points to homogenous coordinates.
        im_tri_h = ConvertPointsToHomogenous(im_tri)
        target_tri_h = ConvertPointsToHomogenous(target_tri)
        # Compute affine transfromation matrix from the target triangle to the triangle in the original. 
        T = ComputeAffine(target_tri_h, im_tri_h)
        # Setup the pixels array as 2xN array of points
        target_tri_pixels = np.array(target_tri_pixels)
        # Append a row of all ones to the array to turn into homogeneous coordinates.
        target_tri_pixels = np.vstack((target_tri_pixels, np.ones(target_tri_pixels.shape[1])))
        # Compute the inverse transformation of the pixel array, elementwise.
        inverse_tri_pixels = T @ target_tri_pixels # Not sure if this sytax is correct
        # Cut off the last row of the inverse pixels array to turn into cartesian coordinates.
        inverse_tri_pixels = inverse_tri_pixels[:2, :]
        # Put the inverse coordinates into a format that RegularGridInterpolator can use.
        float_points = inverse_tri_pixels.T
        # Make a new array that contains the interpolated RGB values from the original image of the inverse pixels
        interpolated_rgb = interpolate_rgb(float_points)
        # Set the pixels in the transformed image to the interpolated values.
        target_tri_pixels = target_tri_pixels.astype(int)
        # Remove homogenous coordinate
        target_tri_pixels = target_tri_pixels[:2, :]
        target_tri_pixels = target_tri_pixels.T
        # Clamp the values to be within the image
        target_tri_pixels[:, 0] = np.clip(target_tri_pixels[:, 0], 0, transformed_im.shape[0]-1)
        target_tri_pixels[:, 1] = np.clip(target_tri_pixels[:, 1], 0, transformed_im.shape[1]-1)
        transformed_im[target_tri_pixels[:, 0], target_tri_pixels[:, 1]] = interpolated_rgb

    return transformed_im 

def cross_dissolve(im1, im2, frac):
    """
    Compute the weighted average of two images by amount frac.
    res = (1-frac)*im1 + frac*im2
    """
    # Resize the smaller image to be the same size as the larger image
    if (im1.shape[0] * im1.shape[1]) < (im2.shape[0] * im2.shape[1]):
        im2 = resize(im2, im1.shape)
    else:
        im1 = resize(im1, im2.shape)
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
    # Make the interpolators
    # Warp the images to the intermediate points 
    im1_warp = warp(im1, im1_pts, intermediate_pts, im2.shape, tri) 
    im2_warp = warp(im2, im2_pts, intermediate_pts, im1.shape, tri) 

    # 2. Cross-dissolve the two warped images using dissolve_frac.
    #   Note: Cross-dissolve means that at dissolve_frac=0, the output is only im1, and at dissolve_frac=1, the output is only im2.
    result = cross_dissolve(im1_warp, im2_warp, dissolve_frac)
    # Check the data type
    return result


