from enum import Enum


class CommandEnum(Enum):
    EVAL_RF_MODEL = "EVAL_RF_MODEL"
    EVAL_ENCODED_IPFRAME = "EVAL_ENCODED_IPFRAME"
    EVAL_ENCODED_BATCH = "EVAL_ENCODED_BATCH"
    # EVAL_TRAIN_DATASET = "EVAL_TRAIN_DATASET"
    # EVAL_TEST_DATASET = "EVAL_TEST_DATASET"
    ENCODE_BATCH = "ENCODE_BATCH"
    ENCODE_IPFRAME_DATASET = "ENCODE_IPFRAME_DATASET"
    PLOT_EVAL_RF_MODEL = "PLOT_EVAL_RF_MODEL"
    # PLOT_BATCH = "PLOT_BATCH"
    PLOT_DEMO_DATA = "PLOT_DEMO_DATA"
    PLOT_ENCODED_BATCH = "PLOT_ENCODED_BATCH"
    VIDEO_DEMO_DATA = "VIDEO_DEMO_DATA"

    SIMULATE_PACKET_LOSS = "SIMULATE_PACKET_LOSS"
    PLOT_PACKET_LOSS = "PLOT_PACKET_LOSS"

    CREATE_GT_MASK = "CREATE_GT_MASK"
    EVAL_DOWNSTREAM_TASK = "EVAL_DOWNSTREAM_TASK"

    # IPFRAME_DATASET = "IPFRAME_DATASET"
    # EVAL_TRAJECTORY = "EVAL_TRAJECTORY"
    # EVAL_ENCODED_H264 = "EVAL_ENCODED_H264"

    # PLOT_ALL_MAPS = "PLOT_ALL_MAPS"
    # PLOT_DATA = "PLOT_DATA"
