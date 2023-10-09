import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'causal-lm-finetuning-gpt2-medium-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'CAUSAL_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_file': 'news_summary.csv',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 10000,
        'model_alias': 'gpt2-medium'
    })

    test_process.start_run()
    test_process.update_configuration('project')
    test_process.update_configuration('pipeline_setup')
    test_process.update_configuration('task')
    test_process.update_configuration('dataset_alias')
    test_process.update_configuration('dataset_file')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.update_configuration('model_alias')
