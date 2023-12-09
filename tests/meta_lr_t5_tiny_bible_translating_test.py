import pytest


from .lib import TestProcess


@pytest.mark.meta_dataset_lr
@pytest.mark.tiny
def test():
    test_name = 'meta-dataset-lr'
    scenario = [1e-3, 1e-4, 1e-5]

    for lr in scenario:
        test_process = TestProcess(test_name, configuration={
            'project': test_name,
            'pipeline_setup': 'EVALUATION',
            'task': 'SEQ_2_SEQ_LM',
            'dataset_alias': 'rexhaif/rus-eng-bible',
            'dataset_file': 'sentences (1).csv',
            'dataset_table_columns': ['english', 'russian'],
            'dataset_partition': 100,
            'model_alias': 'google/t5-efficient-tiny',
            'learning_rate': lr
        })

        test_process.start_run()
        test_process.update_configuration('learning_rate', forced=True)
        test_process.execute_pipeline()
