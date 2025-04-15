import os
if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from PIL import Image
from yacs.config import CfgNode
from typing import List, Optional

def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses

def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)

def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def get_text_image(text_input, img_resized, font_scale=1, thickness=1): # TODO: modify code to handle different font sizes
    tmp_img = np.array(img_resized.convert("RGB"))      # PIL to RGB
    # Lowercase text
    text_input = text_input.lower()
    # Get image dimensions
    h, w = tmp_img.shape[:2]

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = font_scale
    thickness = thickness
    text_color = (0, 0, 0)  # black
    text_padding = 10

    # Measure text size
    (text_width, text_height), baseline = cv2.getTextSize(text_input, font, font_scale, thickness)
    text_box_height = text_height + 2 * text_padding

    # Create a white image for the text area
    text_img = np.ones((text_box_height, w, 3), dtype=np.uint8) * 255

    # Center the text
    x = (w - text_width) // 2
    y = text_height + text_padding // 2

    # Draw text on the white strip
    cv2.putText(text_img, text_input, (x, y), font, font_scale, text_color, thickness, cv2.LINE_AA)

    # Stack original image (img) with text strip below
    combined_img = np.vstack((tmp_img, text_img))
    combined_img = Image.fromarray(combined_img.astype('uint8'))  # Convert back to PIL Image
    return combined_img


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )
    

def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3,:3] = np.c_[x,y,z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def checkerboard_geometry(
    length=12.0,
    color0=[172/255, 172/255, 172/255],
    color1=[215/255, 215/255, 215/255],
    tile_width=0.5,
    alpha=1.0,
    up="y",
    c1=0.0,
    c2=0.0,
):
    assert up == "y" or up == "z"
    color0 = np.array(color0 + [alpha])
    color1 = np.array(color1 + [alpha])
    radius = length / 2.0
    num_rows = num_cols = max(2, int(length / tile_width))
    vertices = []
    vert_colors = []
    faces = []
    face_colors = []
    for i in range(num_rows):
        for j in range(num_cols):
            u0, v0 = j * tile_width - radius, i * tile_width - radius
            us = np.array([u0, u0, u0 + tile_width, u0 + tile_width])
            vs = np.array([v0, v0 + tile_width, v0 + tile_width, v0])
            zs = np.zeros(4)
            if up == "y":
                cur_verts = np.stack([us, zs, vs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 2] += c2
            else:
                cur_verts = np.stack([us, vs, zs], axis=-1)  # (4, 3)
                cur_verts[:, 0] += c1
                cur_verts[:, 1] += c2

            cur_faces = np.array(
                [[0, 1, 3], [1, 2, 3], [0, 3, 1], [1, 3, 2]], dtype=np.int64
            )
            cur_faces += 4 * (i * num_cols + j)  # the number of previously added verts
            use_color0 = (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1)
            cur_color = color0 if use_color0 else color1
            cur_colors = np.array([cur_color, cur_color, cur_color, cur_color])

            vertices.append(cur_verts)
            faces.append(cur_faces)
            vert_colors.append(cur_colors)
            face_colors.append(cur_colors)

    vertices = np.concatenate(vertices, axis=0).astype(np.float32)
    vert_colors = np.concatenate(vert_colors, axis=0).astype(np.float32)
    faces = np.concatenate(faces, axis=0).astype(np.float32)
    face_colors = np.concatenate(face_colors, axis=0).astype(np.float32)

    return vertices, faces, vert_colors, face_colors


class Renderer:

    def __init__(self, faces: np.array, focal_length: float = 685.88, img_res: int = 256):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.focal_length = focal_length
        self.img_res = img_res
        if not isinstance(img_res, tuple):
            self.img_res = [int(img_res), int(img_res)]

        # add faces that make the hand mesh watertight
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)
        
        self.camera_center = [self.img_res[0] // 2, self.img_res[0] // 2]
        self.faces = faces
        self.faces_left = self.faces[:,[0,2,1]]

    def __call__(self,
                vertices: np.array,
                camera_translation: np.array,
                image: torch.Tensor,
                full_frame: bool = False,
                imgname: Optional[str] = None,
                side_view=False, rot_angle=90,
                mesh_base_color=(1.0, 1.0, 0.9),
                scene_bg_color=(0,0,0),
                return_rgba=False,
                ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (np.array): Array of shape (V, 3) containing the mesh vertices.
            camera_translation (np.array): Array of shape (3,) with the camera translation.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
        """
        
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            image = image.clone() * torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3,1,1)
            image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3,1,1)
            image = image.permute(1, 2, 0).cpu().numpy()

        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=(*mesh_base_color, 1.0))

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=self.focal_length, fy=self.focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)
        scene.add(camera, pose=camera_pose)


        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        if return_rgba:
            return color

        valid_mask = (color[:, :, -1])[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]

        output_img = output_img.astype(np.float32)
        return output_img

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9), 
                            rot_axis=[1,0,0], rot_angle=0, is_right=1, vertex_colors=None):
        if vertex_colors is None:
            vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        
        if is_right:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        else:
            mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces_left.copy(), vertex_colors=vertex_colors)
        
        # rotate along x-axis by 180
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=1,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, is_right=is_right)
        mesh = pyrender.Mesh.from_trimesh(mesh)
        
        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color
    
    def render_rgba_contact(
        self,
        vertices: np.array,
        cam_t=None,
        rot=None,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        camera_z=3,
        mesh_base_color=(1.0, 1.0, 0.9),
        scene_bg_color=(0, 0, 0),
        render_res=[256, 256],
        focal_length=None,
        is_right=1,
        contact_vertices=None,
        contact_point=None,
        bg_image=None,
        checkerboard=False,
        camera_top_down=False,
    ):
        # Create offscreen renderer
        renderer = pyrender.OffscreenRenderer(
            viewport_width=render_res[0],
            viewport_height=render_res[1],
            point_size=1.0
        )

        # Use the given focal length or a default
        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        # ------------------------------------------------------------------
        # 1) Set up the camera for a top-down view from y=1.0
        #    looking down -Y.
        # ------------------------------------------------------------------
        camera_pose = np.eye(4, dtype=np.float32)
        camera_pose[2, 3] = camera_translation[2]

        if camera_top_down:
            camera_pose[1, 3] = 0.5  # 1 meter above the ground
            # Rotate camera -90° around X, so it looks along -Y instead of -Z
            R = trimesh.transformations.rotation_matrix(
                np.radians(-90), [1, 0, 0]
            )
            camera_pose[:3, :3] = R[:3, :3]

        # ------------------------------------------------------------------
        # 2) Prepare hand mesh vertices & colors
        #    (the 180° and rot_angle rotations occur inside vertices_to_trimesh)
        # ------------------------------------------------------------------
        # Base color for all vertices, unless marked as contact
        vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        if contact_vertices is not None:
            vertex_colors[np.where(contact_vertices)] = (1.0, 0.0, 0.0, 1.0)

        # Convert the hand to a PyRender mesh
        # Note: This automatically applies 180° around X plus user rot_angle around rot_axis
        mesh_trimesh = self.vertices_to_trimesh(
            vertices=vertices,
            camera_translation=np.array([0, 0, 0]),  # no translation
            mesh_base_color=mesh_base_color,
            rot_axis=rot_axis,
            rot_angle=rot_angle,
            is_right=is_right,
            vertex_colors=vertex_colors
        )
        hand_mesh = pyrender.Mesh.from_trimesh(mesh_trimesh)

        # Create a scene
        scene = pyrender.Scene(
            bg_color=[*scene_bg_color, 0.0],
            ambient_light=(0.3, 0.3, 0.3)
        )
        scene.add(hand_mesh, 'hand_mesh')

        # ------------------------------------------------------------------
        # 3) Contact sphere (if any)
        #    We must replicate the same transformations as in vertices_to_trimesh
        #    so that it appears in the correct location/orientation relative to the hand.
        # ------------------------------------------------------------------
        if contact_point is not None:
            contact_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.01)

            # Color it cyan
            colors = np.array([[0, 255, 255, 255] for _ in range(len(contact_sphere.vertices))],
                            dtype=np.uint8)
            contact_sphere.visual.vertex_colors = colors

            # First shift it to the user's specified contact_point
            contact_sphere.apply_translation(contact_point)

            # Now apply the same transforms done inside vertices_to_trimesh:
            #  (1) 180° about X, and (2) user rotation about rot_axis
            rot_180 = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )
            contact_sphere.apply_transform(rot_180)

            rot_user = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis
            )
            contact_sphere.apply_transform(rot_user)

            # Convert to a PyRender mesh
            contact_sphere_pyrender = pyrender.Mesh.from_trimesh(contact_sphere)
            scene.add(contact_sphere_pyrender, 'contact_sphere')

        # ------------------------------------------------------------------
        # 4) Checkerboard geometry (optional)
        #    Because the hand is being flipped 180° around X, we must do the same
        #    to keep the ground plane aligned visually. We also apply the same
        #    user rotation if desired.
        # ------------------------------------------------------------------
        if checkerboard:
            v, f, vc, fc = checkerboard_geometry(
                length=12.0,
                # color0=[0.0, 0.0, 0.0],  # black
                # color1=[1.0, 1.0, 1.0],  # white
                # color0=[0.0, 0.0, 139 / 255.0],                          # dark blue
                # color1=[173 / 255.0, 216 / 255.0, 230 / 255.0],          # light blue
                color0=[0.0, 100 / 255.0, 0.0],                         # dark green
                color1=[144 / 255.0, 238 / 255.0, 144 / 255.0],         # light green
                tile_width=0.1,
                alpha=1.0,
                up="y",    # squares lie in x–z plane
                c1=0.0,
                c2=0.0
            )

            # Shift the board downward on y so it is below the hand
            v[:, 1] -= -1.0  # e.g., place it at y=0 if your hand is above y=0
            ground_mesh = trimesh.Trimesh(vertices=v, faces=f, vertex_colors=vc, face_colors=fc,
                                          process=False,
                                          validate=False,
                                          draw_edges=False,
                                          flat=False  # flat=False disables flat shading (i.e., use smooth)
                                        )

            # Apply the same transformations as the hand:
            rot_180 = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0]
            )
            ground_mesh.apply_transform(rot_180)

            rot_user = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), rot_axis
            )
            ground_mesh.apply_transform(rot_user)

            ground_mesh_pyrender = pyrender.Mesh.from_trimesh(ground_mesh, smooth=False)
            scene.add(ground_mesh_pyrender, 'ground')

        # ------------------------------------------------------------------
        # 5) Intrinsics camera, placed at y=1.0, looking along -Y
        # ------------------------------------------------------------------
        camera_center = [render_res[0] / 2.0, render_res[1] / 2.0]
        camera = pyrender.IntrinsicsCamera(
            fx=focal_length,
            fy=focal_length,
            cx=camera_center[0],
            cy=camera_center[1],
            zfar=1e12
        )
        scene.add(camera, pose=camera_pose)

        # ------------------------------------------------------------------
        # 6) Lights
        # ------------------------------------------------------------------
        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        # ------------------------------------------------------------------
        # 7) Render
        # ------------------------------------------------------------------
        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        # ------------------------------------------------------------------
        # 8) Optionally blend with a background image
        # ------------------------------------------------------------------
        if bg_image is not None:
            bg_image = bg_image.astype(np.float32) / 255.0
            alpha = color[:, :, 3]
            alpha[alpha > 0] = 1
            color = (
                color[:, :, :3] * alpha[:, :, np.newaxis]
                + bg_image * (1 - alpha[:, :, np.newaxis])
            )

        renderer.delete()
        return color
    
    def render_multiple_contacts(
            self,
            vertices: List[np.array],
            cam_t: List[np.array] = None,
            rot=None,
            rot_axis=[1,0,0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=1,
            contact_vertices=None,
            fading_factor=1.0,
            contact_point=None,
            bg_image=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length/render_res[1]])

        num_iters = len(vertices)
        meshes: List[pyrender.Mesh] = []
        for idx, vertex in enumerate(vertices):
            vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertex.shape[0])
            if contact_vertices is not None:
                vertex_colors[np.where(contact_vertices[idx])] = (1.0, 0.0, 0.0, 1.0)
            vertex_colors[:, 3] *= (fading_factor ** idx)
            mesh = self.vertices_to_trimesh(vertex, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle, 
                                            is_right=is_right, vertex_colors=vertex_colors)
            mesh = pyrender.Mesh.from_trimesh(mesh)
            # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
            meshes.append(mesh)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0], ambient_light=(0.3, 0.3, 0.3))
        # scene.add(mesh, 'mesh')
        for i,mesh in enumerate(meshes):
            scene.add(mesh, f'mesh_{i}')

        if contact_point is not None:
            # create a sphere at the contact point
            contact_sphere = trimesh.creation.icosphere(subdivisions=3, radius=0.01)
            # Ensure vertex colors are correctly formatted (RGBA)
            colors = np.array([[0, 255, 255, 255] for _ in range(len(contact_sphere.vertices))], dtype=np.uint8) # cyan color
            contact_sphere.visual.vertex_colors = colors
            # Move the sphere to the contact point
            contact_sphere.apply_translation(contact_point)

            # match the renderer convention
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            contact_sphere.apply_transform(rot)
            rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), rot_axis)
            contact_sphere.apply_transform(rot)

            # Convert the trimesh sphere to a pyrender mesh
            contact_sphere_pyrender = pyrender.Mesh.from_trimesh(contact_sphere)
            scene.add(contact_sphere_pyrender, 'contact_sphere')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12) # zfar=1e12, check what needs to be set here for znear and zfar

        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0

        if bg_image is not None:
            # blend the rendered image with the background image
            bg_image = bg_image.astype(np.float32) / 255.0
            alpha = color[:, :, 3]
            alpha[alpha > 0] = 1
            color = color[:, :, :3] * alpha[:, :, np.newaxis] + bg_image * (1 - alpha[:, :, np.newaxis])

        renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color
    
    def render_mask_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1,0,0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0,0,0),
            render_res=[256, 256],
            focal_length=None,
            is_right=None,
        ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        if is_right is None:
            is_right = [1 for _ in range(len(vertices))]

        mesh_list = [pyrender.Mesh.from_trimesh(self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle, is_right=sss)) for vvv,ttt,sss in zip(vertices, cam_t, is_right)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0])
        for i,mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)

        nm = {node: 1*(i + 1) for i, node in enumerate(scene.mesh_nodes)}
        mask, _ = renderer.render(scene, pyrender.RenderFlags.SEG, nm)
        renderer.delete()

        return mask

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)
