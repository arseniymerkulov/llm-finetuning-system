import pytest


from .lib import TestProcess


# @pytest.mark.meta_dataset_lr
@pytest.mark.tiny
def test():
    test_name = 'meta-dataset-lr'
    scenario = [1e-2, 1e-3, 1e-4, 1e-5]

    for lr in scenario:
        test_process = TestProcess(test_name, configuration={
            'project': test_name,
            'pipeline_setup': 'FULL',
            'task': 'CLASSIFICATION',
            'dataset_alias': 'jp797498e/twitter-entity-sentiment-analysis',
            'dataset_file': 'twitter_training.csv',
            'dataset_table_columns': ['im getting on borderlands and i will murder you all ,', 'Positive'],
            'dataset_partition': 0,
            'dataset_need_balance': True,
            'model_alias': 'prajjwal1/bert-tiny',
            'learning_rate': lr
        })

        test_process.start_run()
        test_process.update_configuration('learning_rate', forced=True)
        test_process.execute_pipeline()
