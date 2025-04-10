from components.detection_model import Driver, DetectionModelConfigFactory, DetectionModelLayerConfig
from components.llm_utils import LLMModels
from components.localization_model import ConfigFactory


def run_training_job(config):
    # star the training job by giving configuration object as input
    # training device is set as 'cuda:0' in the config
    config.device = 'cuda:0'
    config.train_epochs = 25  # default is set to 25, change if you want
    try:
        Driver(config)
        return f"Finished the execution of the configuration: {config.exp_config}"
    except Exception as e:
        return f"Exception occurred running the configuration: {config.exp_config} Error: {e}"


base_model_fold = 8  # replace the 0 with the fold you want to load
base_model_config = ConfigFactory(exp_config='exp10',
                                  dataset_version='v2',
                                  dataset_name='solidity',
                                  llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
                                  layer_conf=2,
                                  target_dim_list=[1024],
                                  dim_reduce_type='gru',
                                  max_learning_rate_list=[1e-4],
                                  criterion="BCEWithLogitsLoss").get_generated_configs()[base_model_fold]

config_factory = DetectionModelConfigFactory(base_model_config=base_model_config,
                                             transfer_learning_model_config=DetectionModelLayerConfig.NEW_PROJECTION,
                                             num_classes=1,
                                             exp_config='exp_detection',
                                             dataset_version='v1',
                                             dataset_name='solidity_detect_1',
                                             max_learning_rate=1e-3,
                                             criterion="BCEWithLogitsLoss").get_generated_configs()

configs = config_factory.get_generated_configs()

print(f"Number of configurations: {len(configs)}")

print(run_training_job(configs[0]))
print("Finished the jobs!")
