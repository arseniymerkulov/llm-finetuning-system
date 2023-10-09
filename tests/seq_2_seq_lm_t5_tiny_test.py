import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'seq-2-seq-lm-t5-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'SEQ_2_SEQ_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 10000,
        'model_alias': 'google/t5-efficient-tiny'
    })

    test_process.start_run()
    test_process.update_configuration('project')
    test_process.update_configuration('task')
    test_process.update_configuration('dataset_alias')
    test_process.update_configuration('dataset_table_columns')
    test_process.update_configuration('dataset_partition')
    test_process.update_configuration('model_alias')
