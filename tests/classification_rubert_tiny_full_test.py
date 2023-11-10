import pytest


from .lib import TestProcess


@pytest.mark.full
@pytest.mark.tiny
def test():
    test_name = 'classification-rubert-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'rmisra/imdb-spoiler-dataset',
        'dataset_file': 'IMDB_reviews.json',
        'dataset_table_columns': ['review_text', 'is_spoiler'],
        'dataset_partition': 1000,
        'dataset_need_balance': True,
        'model_alias': 'cointegrated/rubert-tiny2'
    })

    test_process.start_run()
    test_process.execute_pipeline()
