import argparse
#from src.transform import transform_2_camera
from src.metrics_functions import calculate_metrics

def main():
    parser = argparse.ArgumentParser(description="Script created for Thesis project 2024 - Calculating metrics (PSNR) (SSIM) (LPIPS)")
    '''Calculating metrics'''
    parser.add_argument("--metrics", action="store_true", default=False,
                        help="If set, script expects two additional arguments path to transform.json for new route  (-new_path) and model paths to dataparser_transforms.json (-existed_model)")
    parser.add_argument("-data", type=str,
                        help="Path to the folder that contains dataset_export folder that has views from models, and reference folder thas has data train data. Number and order in each folders has to be same")
    


    args = parser.parse_args()
    if args.metrics:
        if not args.data:
            print("Error: Both -data argument is required with --metrics")
            exit(1)
        try:
            calculate_metrics(args.data)
        except:
            print("Error: Check the corect form of the input paths")

if __name__ == "__main__":
  main()