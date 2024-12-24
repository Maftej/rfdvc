import json
from abc import ABC

from ui.dsl_manager import DslManager
from ui.command_enum import CommandEnum
from utils.plot_manager import PlotManager
from gt_mask.gt_mask_manager import GTMaskManager
from utils.demo_plot_manager import DemoPlotManager
from utils.demo_video_manager import DemoVideoManager
from packet_loss.pl_eval_manager import PacketLossManager
from packet_loss.pl_plot_manager import PacketLossPlotManager
from video_compression.evaluation_manager import EvaluationManager
from video_compression.video_compression_manager import VideoCompressionManager
from downstream_task_manager.downstream_task_manager import DownStreamTaskManager


class JsonDslManager(DslManager, ABC):
    def __init__(self):
        super().__init__()
        self.plot_manager = PlotManager()
        self.gt_mask_manager = GTMaskManager()
        self.demo_plot_manager = DemoPlotManager()
        self.demo_video_manager = DemoVideoManager()
        self.evaluation_manager = EvaluationManager()
        self.packet_loss_manager = PacketLossManager()
        self.packet_loss_plot_manager = PacketLossPlotManager()
        self.video_compression_manager = VideoCompressionManager()
        self.downstream_task_manager = DownStreamTaskManager()

    def _switch_command(self, command, command_data):
        switcher = {
            CommandEnum.EVAL_RF_MODEL.value: self.evaluation_manager.eval_rf_model,
            CommandEnum.EVAL_ENCODED_IPFRAME.value: self.evaluation_manager.evaluate_encoded_ipframe,
            CommandEnum.PLOT_EVAL_RF_MODEL.value: self.plot_manager.plot_eval_rf_model,
            CommandEnum.PLOT_DEMO_DATA.value: self.demo_plot_manager.plot_demo_data,
            CommandEnum.ENCODE_BATCH.value: self.video_compression_manager.encode_batch,
            CommandEnum.EVAL_ENCODED_BATCH.value: self.evaluation_manager.eval_encoded_batch,
            CommandEnum.PLOT_ENCODED_BATCH.value: self.plot_manager.plot_encoded_batch,
            CommandEnum.ENCODE_IPFRAME_DATASET.value: self.video_compression_manager.encode_ipframe_dataset,
            CommandEnum.VIDEO_DEMO_DATA.value: self.demo_video_manager.images2demo_video,
            CommandEnum.SIMULATE_PACKET_LOSS.value: self.packet_loss_manager.simulate_packet_loss,
            CommandEnum.PLOT_PACKET_LOSS.value: self.packet_loss_plot_manager.plot_packet_loss,
            CommandEnum.CREATE_GT_MASK.value: self.gt_mask_manager.create_gt_mask,
            CommandEnum.EVAL_DOWNSTREAM_TASK.value: self.downstream_task_manager.eval_downstream_task
        }

        func = switcher.get(command, lambda: "Invalid command!")
        return func(command_data)

    def trigger_command(self, command, path):
        data = None
        with open(path, 'r') as f:
            # Load the JSON data into a Python dictionary
            data = json.load(f)

        commands = data["commands"]
        command_data = next(filter(lambda command_enum: command == command_enum["command"], commands), None)
        self._switch_command(command, command_data)
