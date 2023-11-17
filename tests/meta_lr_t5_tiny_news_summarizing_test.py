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
            'pipeline_setup': 'FULL',
            'task': 'SEQ_2_SEQ_LM',
            'dataset_alias': 'sunnysai12345/news-summary',
            'dataset_file': 'news_summary.csv',
            'dataset_table_columns': ['ctext', 'text'],
            'dataset_partition': 6000,
            'model_alias': 'google/t5-efficient-tiny',
            'learning_rate': lr
        })

        test_process.start_run()
        test_process.update_configuration('learning_rate', forced=True)
        test_process.execute_pipeline()
