import numpy as np
from scipy import ndimage
from display_npy import Viewer

def analyze_lesions(
    image_path: str,
    mask_path: str,
    prediction: str, 
    voxel_spacing=(1.0, 1.0, 1.0)
):
    # --- load data ---
    image = np.load(image_path)               # raw brain volume
    pred_mask  = np.load(prediction)
    mask = np.load(mask_path)
    mask  = np.argmax(mask, axis=-1)          # collapse one-hot → 3D labels
    image  = np.argmax(image, axis=-1)          # collapse one-hot → 3D labels
    # physical volume of one voxel
    voxel_vol_mm3 = np.prod(voxel_spacing)

    # find all distinct class‐labels
    labels = np.unique(pred_mask)
    mask_labels = np.unique(mask)
    print("Classes in prediction:", labels)

    name_map = {
        1: "Metastasis",
        2: "Edema",
        3: "Tumor core"
    }
    # set up napari
    viewer = Viewer(image_path=image_path, mask_path=mask_path, mode_3d=True)
    
    # add the raw brain image underneath everything
    viewer.viewer.add_image(
        image,
        name="Raw Image",
        opacity=.5
    )

    # define colors for each class (RGBA)
    color_map = {
        1: (1.0, 0.0, 0.0, 0.6),   # red
        2: (0.0, 1.0, 0.0, 0.6),   # green
        3: (0.0, 0.0, 1.0, 0.6),   # blue
        # etc.
    }
    for class_label in mask_labels:
        if class_label == 0:
            continue  # skip background

        # binary mask for this class
        binary = (mask == class_label)

        # split into connected components
        cc_map, num_instances = ndimage.label(binary)
        cmap = np.zeros((2, 4), float)
        col = color_map.get(class_label, (1.0, 1.0, 0.0, 0.6))
        cmap[1] = col
        viewer.viewer.add_labels(
                binary,
                name=f"{name_map[class_label]}_GT_mask_total",
                colormap = cmap,
                opacity=0.6        
            ) 
            
            
    for class_label in labels:
        if class_label == 0:
            continue  # skip background

        # binary mask for this class
        binary = (pred_mask == class_label)

        # split into connected components
        cc_map, num_instances = ndimage.label(binary)
        
        
        cmap = np.zeros((2, 4), float)
        col = color_map.get(class_label, (1.0, 1.0, 0.0, 0.6))
        cmap[1] = col
        print(f"Class {name_map[class_label]}: {num_instances} instance(s)")
        viewer.viewer.add_labels(
                binary,
                name=f"{name_map[class_label]}_total",
                colormap = cmap,
                opacity=0.6        
            )
            
        for inst_idx in range(1, num_instances + 1):
            lesion_mask = (cc_map == inst_idx).astype(np.uint16)

            # compute volume if you like
            vox = int(lesion_mask.sum())
            mm3 = vox * voxel_vol_mm3
            print(f"  • label_{class_label}_{inst_idx}: {vox} voxels ({mm3:.2f} mm³)")


            # add the layer with our custom colormap
            #viewer.viewer.add_labels(
            #   lesion_mask,
            #   name=f"{name_map[class_label]}_{inst_idx}",
            #   colormap=cmap,      
            #   opacity=0.6, 
            #)

    #viewer.visualize_mask(mask_path)
    for layer in viewer.viewer.layers:
        if layer.name != "Raw Image" and layer.name !="GT MASK" and not "total" in layer.name:
            layer.visible = False

    viewer.run()

sample_number = 2
analyze_lesions(


    image_path=f"MET_samples/images/image_{sample_number}.npy",
    mask_path=f"MET_samples/masks/mask_{sample_number}.npy",
    prediction=f"MET_samples/prediction_boundary/prediction_{sample_number}.npy",
    #target_label=3
)
