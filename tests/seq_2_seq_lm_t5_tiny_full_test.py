import pytest


from .lib import TestProcess


@pytest.mark.full
@pytest.mark.tiny
def test():
    test_name = 'seq-2-seq-lm-t5-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'SEQ_2_SEQ_LM',
        'dataset_alias': 'sunnysai12345/news-summary',
        'dataset_file': 'news_summary.csv',
        'dataset_table_columns': ['text', 'ctext'],
        'dataset_partition': 1000,
        'model_alias': 'google/t5-efficient-tiny'
    })

    test_process.run_pipeline()
