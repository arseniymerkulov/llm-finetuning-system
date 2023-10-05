import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'causal-lm-finetuning-gpt2-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'task': 'CAUSAL_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 1000,
        'model_alias': 'sshleifer/tiny-gpt2'
    })

    TestProcess.start_run()
    test_process.update_configuration('project')
    test_process.update_configuration('task')
    test_process.update_configuration('dataset_alias')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.update_configuration('model_alias')
