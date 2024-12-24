import os
import json


class ScenarioManager:
    def __init__(self):
        pass

    def generate_rf_models(self, path, model_names, variants):
        rf_variants = []

        for variant in variants:
            rf_models = []
            for model in model_names:
                gt_path = os.path.join(path, model, variant, "metrics", "GT_Carla")
                rf_path = os.path.join(path, model, variant, "metrics", f"RF_{variant.capitalize()}")
                if not os.path.exists(gt_path) or not os.path.exists(rf_path):
                    print("THIS PATH=", gt_path)
                    print("THIS PATH=", rf_path)
                    return None
                rf_model = {
                    "rf_name": model,
                    "gt_dataset_path": gt_path,
                    "rf_dataset_path": rf_path
                }
                rf_models.append(rf_model)

            rf_variant_entry = {
                "rf_variant": variant,
                "rf_models": rf_models
            }
            rf_variants.append(rf_variant_entry)

        return rf_variants


def main():
    base_path = "C:\\Users\\mDopiriak\\Desktop\\carla_city"
    model_names = [f"a{i}" for i in range(1, 20)]
    variants = ["nerfacto", "splatfacto"]

    scenario_manager = ScenarioManager()

    data = {
        "rf_variants": scenario_manager.generate_rf_models(base_path, model_names, variants)
    }

    with open('output.json', 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    main()
