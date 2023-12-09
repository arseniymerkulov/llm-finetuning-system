import pytest


from .lib import TestProcess


@pytest.mark.meta_classification
@pytest.mark.tiny
def test():
    # todo: do i need to separate tests not only by task, but also by model/dataset?
    #       charts lose meaning when contain information about several datasets
    test_name = 'meta-lr-bert-classification'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'team-ai/spam-text-message-classification',
        'dataset_file': 'SPAM text message 20170820 - Data.csv',
        'dataset_table_columns': ['Message', 'Category'],
        'dataset_partition': 0,
        'dataset_need_balance': True,
        'model_alias': 'prajjwal1/bert-tiny'
    })

    test_process.selection({
        'additional_learning_rate': [1e-3, 1e-4]
    })
