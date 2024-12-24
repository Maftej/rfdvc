import argparse

from ui.command_enum import CommandEnum


class ConsoleManager:
    def __init__(self):
        pass

    def parse_args(self):
        parser = argparse.ArgumentParser(
            description="Run VC-RFs")

        # parser.add_argument("--nerf_variant",
        #                     choices=[NeRFVariantsEnum.D_NERF.value, NeRFVariantsEnum.INSTANT_NGP.value],
        #                     help="")

        parser.add_argument("--command", choices=[CommandEnum.EVAL_RF_MODEL.value,
                                                  CommandEnum.EVAL_ENCODED_IPFRAME.value,
                                                  CommandEnum.PLOT_EVAL_RF_MODEL.value,
                                                  CommandEnum.PLOT_DEMO_DATA.value,
                                                  CommandEnum.VIDEO_DEMO_DATA.value,
                                                  CommandEnum.ENCODE_BATCH.value,
                                                  CommandEnum.EVAL_ENCODED_BATCH.value,
                                                  CommandEnum.PLOT_ENCODED_BATCH.value,
                                                  CommandEnum.ENCODE_IPFRAME_DATASET.value,
                                                  CommandEnum.SIMULATE_PACKET_LOSS.value,
                                                  CommandEnum.PLOT_PACKET_LOSS.value,
                                                  CommandEnum.CREATE_GT_MASK.value,
                                                  CommandEnum.EVAL_DOWNSTREAM_TASK.value],
                            help="")

        # parser.add_argument("--dataset_type", default=None,
        #                     choices=[DatasetTypeEnum.TRAIN.value, DatasetTypeEnum.TEST.value,
        #                              DatasetTypeEnum.VAL.value],
        #                     help="")

        parser.add_argument("--path", default="",
                            help="")

        return parser.parse_args()
