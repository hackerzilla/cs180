# Julian's helpers functions for project 3
import numpy as np
import cv2
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.tri as mtri
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
    ax.plot(pts[:, 0], pts[:, 1], 'ro')

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

