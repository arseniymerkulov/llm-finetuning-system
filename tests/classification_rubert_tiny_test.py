import pytest


from lib import TestProcess


@pytest.mark.full
def test():
    test_name = 'classification-rubert-tiny-test'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'EVALUATION',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'rmisra/imdb-spoiler-dataset',
        'dataset_file': 'IMDB_reviews.json',
        'dataset_table_columns': ['review_text', 'is_spoiler'],
        'dataset_partition': 1000,
        'model_alias': 'cointegrated/rubert-tiny2'
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
