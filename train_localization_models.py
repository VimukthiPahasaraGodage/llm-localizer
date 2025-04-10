from components.llm_utils import LLMModels
from components.localization_model import Driver, ConfigFactory


def run_training_job(config):
    # star the training job by giving configuration object as input
    # training device is set as 'cuda:0' in the config
    config.device = 'cuda:0'
    config.train_epochs = 50
    try:
        Driver(config)
        return f"Finished the execution of the configuration: {config.exp_config}"
    except Exception as e:
        return f"Exception occurred running the configuration: {config.exp_config} Error: {e}"


# # Original model(Defects4j)
# config_factory = ConfigFactory(exp_config='exp1',
#                                dataset_version='v2',
#                                dataset_name='defects4j',
#                                llm_models_list=[LLMModels.CODEGEN_350M_MULTI, LLMModels.CODEGEN_6B_MULTI, LLMModels.CODEGEN_16B_MULTI],
#                                layer_conf=1,
#                                target_dim_list=[256, 512, 1024],
#                                dim_reduce_type='linear',
#                                max_learning_rate_list=[1e-4, 7e-5, 5e-5],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# # Dimensional reduction ablation test
# config_factory = ConfigFactory(exp_config='exp2',
#                                dataset_version='v2',
#                                dataset_name='defects4j',
#                                llm_models_list=[LLMModels.CODEGEN_16B_MULTI],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# # LLM ablation test
# config_factory = ConfigFactory(exp_config='exp3',
#                                dataset_version='v2',
#                                dataset_name='defects4j',
#                                llm_models_list=[LLMModels.QWEN_QWQ_32B],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='linear',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# # LLM ablation test
# config_factory = ConfigFactory(exp_config='exp4',
#                                dataset_version='v2',
#                                dataset_name='defects4j',
#                                llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='linear',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# # CustomCosineLoss ablation test
# config_factory = ConfigFactory(exp_config='exp5',
#                                dataset_version='v2',
#                                dataset_name='defects4j',
#                                llm_models_list=[LLMModels.CODEGEN_16B_MULTI],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='linear',
#                                max_learning_rate_list=[1e-4],
#                                criterion="CustomCosine")
# configs = config_factory.get_generated_configs()

# # Solidity model
# config_factory = ConfigFactory(exp_config='exp6',
#                                dataset_version='v2',
#                                dataset_name='solidity',
#                                llm_models_list=[LLMModels.QWEN_QWQ_32B],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# config_factory = ConfigFactory(exp_config='exp7',
#                                dataset_version='v2',
#                                dataset_name='solidity',
#                                llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_LLAMA_8B],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# config_factory = ConfigFactory(exp_config='exp8',
#                                dataset_version='v2',
#                                dataset_name='solidity',
#                                llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# config_factory = ConfigFactory(exp_config='exp9',
#                                dataset_version='v2',
#                                dataset_name='solidity',
#                                llm_models_list=[LLMModels.CODEGEN_16B_MULTI],
#                                layer_conf=1,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()
#
# config_factory = ConfigFactory(exp_config='exp10',
#                                dataset_version='v2',
#                                dataset_name='solidity',
#                                llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
#                                layer_conf=2,
#                                target_dim_list=[1024],
#                                dim_reduce_type='gru',
#                                max_learning_rate_list=[1e-4],
#                                criterion="BCEWithLogitsLoss")
# configs = config_factory.get_generated_configs()


# No instructions in the prompt
config_factory = ConfigFactory(exp_config='exp11',
                               dataset_version='v2',
                               dataset_name='solidity',
                               llm_models_list=[LLMModels.DEEPSEEK_R1_DISTILL_QWEN_14B],
                               layer_conf=2,
                               target_dim_list=[1024],
                               dim_reduce_type='gru',
                               max_learning_rate_list=[1e-4],
                               criterion="BCEWithLogitsLoss")
configs = config_factory.get_generated_configs()

print(f"Number of configurations: {len(configs)}")

print(run_training_job(configs[0]))
print("Finished all jobs!")
