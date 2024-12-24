import numpy as np
import torch
import math
import lpips
import cv2
import os
import json
from torchmetrics.functional import structural_similarity_index_measure


def img_psnr(image1,image2):# calculate PSNR for 2 images
    mse = np.mean((image1/1.0 - image2/1.0)**2)
    if mse < 1e-10:
        return 100
    psnr = 20 * math.log10(255/math.sqrt(mse))
    return psnr

def Clpips(img1,img2):
    global device,loss_fn
    loss_fn.cuda()
    img1 = lpips.im2tensor(img1)
    img2 = lpips.im2tensor(img2)
    img1 = img1.cuda()
    img2 = img2.cuda()
    dist01 = loss_fn.forward(img1, img2).mean().detach().cpu().tolist()
    return dist01

def calculate_metrics(path):
    global device,loss_fn
    device = torch.device("cuda") # cuda/cpu
    loss_fn = lpips.LPIPS(net='alex', spatial=True)
    ssim = structural_similarity_index_measure

    neural_path = os.path.join(path, "dataset_export")
    reference_path = os.path.join(path, "reference")       

    neural = os.listdir(neural_path)
    reference = os.listdir(reference_path)

    OverAllMetrics = {}

    PSNR = []

    for file1,file2 in zip(neural,reference):
        image1 = cv2.imread(os.path.join(neural_path,file1))
        image2 = cv2.imread(os.path.join(reference_path,file2))

        # Convert BGR images to RGB (OpenCV loads images in BGR format)
        image1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        original_image = torch.tensor(image1).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        generated_image = torch.tensor(image2).permute(2, 0, 1).unsqueeze(0).float() / 255.0

        # Call the functions with the loaded images
        psnr_val = img_psnr(image1_rgb, image2_rgb)
        ssim_torch_val = ssim(original_image, generated_image)
        lpips_val = Clpips(image1_rgb, image2_rgb)
        
        file_name = file1 +" & "+ file2
        OverAllMetrics[file_name] = {"PSNR": psnr_val, "SSIM_torch": float(ssim_torch_val),"LPIPS": lpips_val}

    results = {}
    mean_psnr = sum(info["PSNR"] for info in OverAllMetrics.values()) / len(OverAllMetrics)
    mean_ssim_torch = sum(info["SSIM_torch"] for info in OverAllMetrics.values()) / len(OverAllMetrics)
    mean_lpips = sum(info["LPIPS"] for info in OverAllMetrics.values()) / len(OverAllMetrics)

    max_psnr_images = max(OverAllMetrics, key=lambda x: OverAllMetrics[x]["PSNR"])
    max_ssim_images = max(OverAllMetrics, key=lambda x: OverAllMetrics[x]["SSIM_torch"])
    max_lpips_images = max(OverAllMetrics, key=lambda x: OverAllMetrics[x]["LPIPS"])

    min_psnr_images = min(OverAllMetrics, key=lambda x: OverAllMetrics[x]["PSNR"])
    min_ssim_images = min(OverAllMetrics, key=lambda x: OverAllMetrics[x]["SSIM_torch"])
    min_lpips_images = min(OverAllMetrics, key=lambda x: OverAllMetrics[x]["LPIPS"])

    results["mean_values"] = {"PSNR": mean_psnr,"SSIM_torch": float(mean_ssim_torch), "LPIPS": mean_lpips}
    results["max_values"] = {"PSNR": {"image" :max_psnr_images,"value":OverAllMetrics[max_psnr_images]["PSNR"]},
                             "SSIM_torch": {"image" :max_ssim_images,"value":OverAllMetrics[max_ssim_images]["SSIM_torch"]},
                             "LPIPS": {"image" :max_lpips_images,"value":OverAllMetrics[max_lpips_images]["LPIPS"]}}
    results["min_values"] = {"PSNR": {"image" :min_psnr_images,"value":OverAllMetrics[min_psnr_images]["PSNR"]},
                             "SSIM_torch": {"image" :min_ssim_images,"value":OverAllMetrics[min_ssim_images]["SSIM_torch"]},
                             "LPIPS": {"image" :min_lpips_images,"value":OverAllMetrics[min_lpips_images]["LPIPS"]}}
    results.update(OverAllMetrics)

    # Print the resulting dictionary
    outstr = json.dumps(results, indent=4)
    with open(os.path.join(path, "eval.json"), mode="w") as f:
        f.write(outstr)
    print(f"\nEval saved :{path}\nPSNR-{mean_psnr},SSIM-{mean_ssim_torch},LPIPS-{mean_lpips}\n")

