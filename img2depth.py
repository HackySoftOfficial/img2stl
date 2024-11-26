import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from stl import mesh
from tqdm import tqdm
import asyncio
import sys

async def update_ui(root):
    while True:
        try:
            root.update()
            await asyncio.sleep(0.01)
        except tk.TclError:
            break

async def generate_3d_model(image_path, depth_levels=10, target_width=100, base_height=2, max_dimension=1000):
    """
    Convert an image to a 3D STL model where:
    - Model width is fixed to target_width (mm)
    - Height is scaled proportionally 
    - Depth is subtracted from infill layers
    - Base layer + depth_levels additional layers
    - Shows depth visualization
    - Each layer is 0.2mm thick
    - Smooth walls with infill subtraction
    
    :param image_path: Path to the input image
    :param depth_levels: Number of depth levels (excluding base)
    :param target_width: Desired width of the model in mm
    :param base_height: Height of the base plate in mm
    :param max_dimension: Maximum dimension for image processing
    """
    print("Loading and processing image...")
    # Read and process image
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print("Error: Unable to load image.")
        return

    # Store original image for display
    if len(img.shape) == 3:
        original_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        original_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    # Handle transparency and convert to grayscale
    if len(img.shape) == 3 and img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    elif len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Store grayscale image for display
    grayscale_img = img.copy()

    # Resize if image is too large
    height, width = img.shape
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img = cv2.resize(img, (new_width, new_height))
        original_img = cv2.resize(original_img, (new_width, new_height))
        grayscale_img = cv2.resize(grayscale_img, (new_width, new_height))
        height, width = img.shape

    # Reduce detail by downscaling and then upscaling
    scale_factor = 0.25  # Reduce to 25% of original size
    small_width = int(width * scale_factor)
    small_height = int(height * scale_factor)
    img = cv2.resize(img, (small_width, small_height))
    img = cv2.resize(img, (width, height))

    # Calculate mm per pixel to achieve target width
    mm_per_pixel = target_width / width
    mm_height = height * mm_per_pixel
    mm_width = width * mm_per_pixel

    # Apply preprocessing to remove noise and small particles
    print("Preprocessing image...")
    kernel = np.ones((5,5), np.uint8)  # Increased kernel size for more smoothing
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.GaussianBlur(img, (5,5), 0)  # Additional smoothing
    
    # Create infill pattern
    print("Generating infill pattern...")
    infill_layers = 10
    layer_height = 0.2  # mm per layer
    total_height = infill_layers * layer_height + base_height  # Add base height
    
    # Normalize image for depth calculation
    print("Normalizing image depth...")
    img_normalized = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    # Calculate depth and add base height
    Z = total_height * (1 - img_normalized) + base_height  # Add base height to all Z values
    
    # Show depth visualization using Tkinter
    print("Creating depth visualization...")
    root = tk.Tk()
    root.title("Image Processing Visualization")

    # Convert Z values to display image
    depth_img = ((Z / total_height) * 255).astype(np.uint8)
    depth_img = cv2.applyColorMap(depth_img, cv2.COLORMAP_VIRIDIS)
    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2RGB)

    # Scale for display
    max_display_size = 400
    scale = min(max_display_size / depth_img.shape[1], max_display_size / depth_img.shape[0])
    if scale < 1:
        new_width = int(depth_img.shape[1] * scale)
        new_height = int(height * scale)
        depth_img = cv2.resize(depth_img, (new_width, new_height))
        original_img = cv2.resize(original_img, (new_width, new_height))
        grayscale_img = cv2.resize(grayscale_img, (new_width, new_height))

    # Create frames for each image
    frame1 = tk.Frame(root)
    frame1.pack(side=tk.LEFT, padx=5)
    frame2 = tk.Frame(root)
    frame2.pack(side=tk.LEFT, padx=5)
    frame3 = tk.Frame(root)
    frame3.pack(side=tk.LEFT, padx=5)

    # Create labels for each image
    tk.Label(frame1, text="Original Image").pack()
    canvas1 = tk.Canvas(frame1, width=new_width, height=new_height)
    canvas1.pack()

    tk.Label(frame2, text="Grayscale Image").pack()
    canvas2 = tk.Canvas(frame2, width=new_width, height=new_height)
    canvas2.pack()

    tk.Label(frame3, text="Depth Visualization").pack()
    canvas3 = tk.Canvas(frame3, width=new_width, height=new_height)
    canvas3.pack()

    # Convert and display images with proper references
    img_pil_original = Image.fromarray(original_img)
    img_tk_original = ImageTk.PhotoImage(image=img_pil_original)
    canvas1.create_image(0, 0, anchor=tk.NW, image=img_tk_original)
    canvas1.image = img_tk_original  # Keep reference
    
    img_pil_gray = Image.fromarray(grayscale_img)
    img_tk_gray = ImageTk.PhotoImage(image=img_pil_gray)
    canvas2.create_image(0, 0, anchor=tk.NW, image=img_tk_gray)
    canvas2.image = img_tk_gray  # Keep reference
    
    img_pil_depth = Image.fromarray(depth_img)
    img_tk_depth = ImageTk.PhotoImage(image=img_pil_depth)
    canvas3.create_image(0, 0, anchor=tk.NW, image=img_tk_depth)
    canvas3.image = img_tk_depth  # Keep reference

    # Start UI update task
    ui_task = asyncio.create_task(update_ui(root))

    print("Creating vertex grid...")
    # Create vertex grid - flip X coordinates to mirror the model
    x = np.linspace(mm_width, 0, width)  # Reversed x coordinates
    y = np.linspace(0, mm_height, height)
    X, Y = np.meshgrid(x, y)
    
    # Create vertices for top surface and base
    print("Generating vertices...")
    vertices_top = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
    vertices_bottom = np.column_stack((X.flatten(), Y.flatten(), np.zeros_like(Z.flatten())))
    
    print("Creating faces...")
    # Create faces for top and bottom surfaces
    faces = []
    for i in range(height - 1):
        for j in range(width - 1):
            # Top surface triangles
            v0 = i * width + j
            v1 = v0 + 1
            v2 = (i + 1) * width + j
            v3 = v2 + 1
            
            # Add two triangles for top surface (corrected winding order)
            faces.append([v0, v2, v1])
            faces.append([v1, v2, v3])
            
            # Bottom surface triangles
            v0_b = v0 + (height * width)  # Offset for bottom vertices
            v1_b = v1 + (height * width)
            v2_b = v2 + (height * width)
            v3_b = v3 + (height * width)
            
            # Add two triangles for bottom surface (corrected winding order)
            faces.append([v0_b, v1_b, v2_b])
            faces.append([v1_b, v3_b, v2_b])
            
            # Add side wall triangles
            if j == 0:  # Left wall
                faces.append([v0, v0_b, v2])
                faces.append([v2, v0_b, v2_b])
            if j == width - 2:  # Right wall
                faces.append([v1, v3, v1_b])
                faces.append([v3, v3_b, v1_b])
            if i == 0:  # Front wall
                faces.append([v0, v1, v0_b])
                faces.append([v1, v1_b, v0_b])
            if i == height - 2:  # Back wall
                faces.append([v2, v2_b, v3])
                faces.append([v3, v2_b, v3_b])

    faces = np.array(faces)
    
    print("Creating final mesh...")
    # Combine vertices and create mesh
    all_vertices = np.vstack([vertices_top, vertices_bottom])
    model = mesh.Mesh(np.zeros(len(faces), dtype=mesh.Mesh.dtype))
    for i, face in enumerate(faces):
        for j in range(3):
            model.vectors[i][j] = all_vertices[face[j]]
    
    # Fix mesh normals
    model.update_normals()
    
    # Save the STL file
    output_path = image_path.rsplit('.', 1)[0] + '_3d.stl'
    print(f"Saving STL file to {output_path}...")
    model.save(output_path)
    
    print(f"\nModel successfully created!")
    print(f"Model dimensions: {mm_width:.1f}mm x {mm_height:.1f}mm")
    print(f"Total height: {total_height}mm (Base: {base_height}mm + {infill_layers} layers * {layer_height}mm)")
    
    # Keep window open until user closes it
    root.mainloop()
    
    # Cancel UI update task when window is closed
    ui_task.cancel()
    try:
        await ui_task
    except asyncio.CancelledError:
        pass
    
    return model

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python img2depth.py <image_path>")
        sys.exit(1)
        
    image_path = sys.argv[1]
    asyncio.run(generate_3d_model(image_path, depth_levels=10, target_width=100, base_height=2))