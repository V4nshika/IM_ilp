import matplotlib.pyplot as plt
from pm4py.visualization.petri_net import visualizer as pn_visualizer
import tempfile
from PIL import Image
import os

def save_multiple_petri_nets_to_pdf(petri_nets_list, filename="multiple_petri_nets.pdf"):
    """
    Save multiple Petri nets to a single PDF
    petri_nets_list: List of tuples [(net1, im1, fm1), (net2, im2, fm2), ...]
    """
    images = []
    
    # Create temporary images for each Petri net
    for i, (net, im, fm) in enumerate(petri_nets_list):
        # Generate visualization
        gviz = pn_visualizer.apply(net, im, fm)
        
        # Save as temporary PNG
        temp_path = f"temp_petri_net_{i}.png"
        pn_visualizer.save(gviz, temp_path)
        images.append(temp_path)
    
    # Convert images to PDF
    first_image = Image.open(images[0])
    other_images = [Image.open(img) for img in images[1:]]
    
    first_image.save(filename, "PDF", resolution=100.0, save_all=True, 
                    append_images=other_images)
    
    # Clean up temporary files
    for img_path in images:
        os.remove(img_path)
    
    print(f"Saved {len(petri_nets_list)} Petri nets to {filename}")
