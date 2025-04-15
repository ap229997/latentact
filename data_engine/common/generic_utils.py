import random

import cv2
from cv2.dnn import Model
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader
import alphashape
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon


def to_tensor(data):
    # convert all_actions to tensor recursively
    if isinstance(data, dict):
        return {k: to_tensor(v) for k, v in data.items()}
    if isinstance(data, list):
        return [to_tensor(v) for v in data]
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    if isinstance(data, Image.Image):
        return to_tensor(np.array(data))
    return data


def squeeze_float(data):
    if isinstance(data, dict):
        return {k: squeeze_float(v) for k, v in data.items()}
    if isinstance(data, list):
        return [squeeze_float(v) for v in data]
    if isinstance(data, torch.Tensor):
        return data.squeeze().float()
    return data


def reset_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mul_loss_dict(loss_dict):
    for key, val in loss_dict.items():
        loss, weight = val
        loss_dict[key] = loss * weight
    return loss_dict


def jitter_mask(mask, jitter_range=2):
    """
    Jitters the boundary of a binary mask.
    
    Parameters:
    - mask: A binary mask (numpy array) where the object/hand is 1 and the background is 0.
    - jitter_range: The maximum distance in pixels to jitter a boundary pixel.
    
    Returns:
    - A new mask with the jittered boundary.
    """
    # Find the boundary by dilating the mask and then finding the difference with erosion
    dilated_mask = binary_dilation(mask)
    eroded_mask = binary_erosion(mask)
    boundary_mask = dilated_mask ^ eroded_mask
    
    # Initialize the jittered mask with the original mask
    jittered_mask = np.copy(mask)
    
    # Get the coordinates of the boundary pixels
    y_indices, x_indices = np.where(boundary_mask)
    
    # For each boundary pixel, jitter its position
    for y, x in zip(y_indices, x_indices):
        # Calculate random offsets within the jitter range
        y_jitter = np.random.randint(-jitter_range, jitter_range + 1)
        x_jitter = np.random.randint(-jitter_range, jitter_range + 1)
        
        # Calculate the new position ensuring it's within the mask bounds
        y_new = np.clip(y + y_jitter, 0, mask.shape[0] - 1)
        x_new = np.clip(x + x_jitter, 0, mask.shape[1] - 1)
        
        # Set the jittered position in the new mask
        jittered_mask[y, x] = 0  # Clear the original boundary pixel
        jittered_mask[y_new, x_new] = 1  # Set the new jittered position
    
    return jittered_mask


def pad_mask_at_boundary(mask, padding=5):
    """
    Pads a binary mask at the boundary by a specified amount.
    
    Parameters:
    - mask: A binary mask (numpy array) where the object/hand is 1 and the background is 0.
    - padding: The amount of padding to add at the boundary.
    
    Returns:
    - A new mask with the padded boundary.
    """
    # Find the boundary by dilating the mask and then finding the difference with erosion
    try:
        dilated_mask = binary_dilation(mask)
    except:
        # print ('Error in processing mask of type ', type(mask))
        return None
    eroded_mask = binary_erosion(mask)
    boundary_mask = dilated_mask ^ eroded_mask
    
    # Initialize the padded mask with the original mask
    padded_mask = np.copy(mask)
    
    # Get the coordinates of the boundary pixels
    y_indices, x_indices = np.where(boundary_mask)
    
    # For each boundary pixel, pad the boundary by the specified amount
    for y, x in zip(y_indices, x_indices):
        # Pad the boundary pixel and its neighbors by the specified amount
        y_start = max(y - padding, 0)
        y_end = min(y + padding + 1, mask.shape[0])
        x_start = max(x - padding, 0)
        x_end = min(x + padding + 1, mask.shape[1])
        
        # Set the padded boundary pixels in the new mask
        padded_mask[y_start:y_end, x_start:x_end] = 1
    
    return padded_mask


def create_gussian_mask(center, sigma, im_w, im_h):
    x = np.arange(0, im_w, 1, float)
    y = np.arange(0, im_h, 1, float)
    x, y = np.meshgrid(x, y)
    # create a gaussian mask
    gauss = np.exp(-((x - center[0])**2 + (y - center[1])**2) / (2.0 * sigma**2))
    
    # draw gauss on raw_image
    gauss_image = Image.fromarray(np.uint8(gauss / gauss.max() * 255), 'L')
    # Resize the Gaussian image to match the original image dimensions if necessary
    gauss_image = gauss_image.resize((im_w, im_h), Image.Resampling.BILINEAR)
    # Create a new image for the overlay with the same size and RGBA mode
    overlay_image = Image.new('RGBA', (im_w, im_h), (255, 255, 255, 0))  # Transparent background
    # Paste the Gaussian mask onto the overlay image using itself as the mask
    overlay_image.paste(gauss_image, (0, 0), gauss_image)

    return overlay_image


def create_mask_with_convex_hull(pixels, size, return_numpy=False):
    # Create a convex hull mask from 2D pixels (numpy array)
    hull = ConvexHull(pixels)
    hull_verts = pixels[hull.vertices]
    hull_mask = Image.new('L', (size[0], size[1]), 0)
    ImageDraw.Draw(hull_mask).polygon([tuple(x) for x in hull_verts], outline='white', fill='white')
    hull_mask = Image.fromarray((np.array(hull_mask)*255).astype(np.uint8))
    if return_numpy:
        hull_mask = np.array(hull_mask)
    return hull_mask


def create_mask_with_polygon_outline(pixels, size, return_numpy=False):
    verts_list = [tuple(point) for point in pixels]
    ordered_points = [pixels[0]]  # Start with the first point
    remaining_points = pixels[1:].tolist()  # Convert remaining points to list for easier manipulation
    # Repeatedly find the nearest neighbor and append to ordered_points
    while remaining_points:
        last_point = ordered_points[-1]
        # Calculate distances from the last point in ordered_points to all remaining points
        distances = np.linalg.norm(np.array(remaining_points) - last_point, axis=1)
        nearest_idx = np.argmin(distances)  # Index of the nearest point
        # Append the nearest point to ordered_points and remove it from remaining_points
        ordered_points.append(remaining_points.pop(nearest_idx))
    # Convert ordered_points back to numpy array if needed
    verts_list = [tuple(point) for point in ordered_points]
    # Draw the polygon outline on curr_img
    curr_img = Image.new('L', size, 0)
    draw = ImageDraw.Draw(curr_img)
    draw.line(verts_list + [verts_list[0]], fill='white', width=5)  # Ensures the polygon is closed
    if return_numpy:
        curr_img = np.array(curr_img)
    return curr_img


def create_mask_with_concave_hull(pixels, size, alpha=0.05, return_numpy=False):
    # Compute the alpha shape (concave hull)
    # Adjust alpha parameter as needed (smaller values = more detailed boundary)
    boundary = alphashape.alphashape(pixels, alpha)
    # Extract the exterior coordinates of the boundary polygon if it's not empty
    if boundary and isinstance(boundary, Polygon):
        boundary_points = np.array(boundary.exterior.coords)
        # Create an empty mask
        mask = Image.new('L', size, 0)
        boundary_points = [tuple(point) for point in boundary_points]
        # Draw the concave hull on the mask
        ImageDraw.Draw(mask).polygon(boundary_points, outline='white', fill='white')
    else:
        print("No valid boundary found. Try adjusting the alpha parameter.")
        return None
    # alpha blend mask with curr_img
    if return_numpy:
        mask = np.array(mask)
    return mask


def get_bbox_from_mask(mask):
    # use cv2 to get the bounding box of the mask
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    if mask.dtype == bool:
        mask = (mask*255).astype(np.uint8)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Check if any contours were found
    if contours:
        # Get the largest contour by area
        largest_contour = max(contours, key=cv2.contourArea)
        # Get the bounding box for the largest contour
        x, y, w, h = cv2.boundingRect(largest_contour)
        bbox = [x, y, x + w, y + h]
        return bbox
    else:
        return None


def blend_mask_with_image(mask, image, alpha=128, return_numpy=False):
    # Blend a mask with an image using alpha blending
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray((mask*255).astype(np.uint8))
    mask.putalpha(alpha)
    curr_img = image.copy()
    curr_img.paste(mask, (0, 0), mask)
    if return_numpy:
        curr_img = np.array(curr_img)
    return curr_img