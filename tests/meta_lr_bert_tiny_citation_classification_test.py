import pytest


from .lib import TestProcess


@pytest.mark.meta_classification
@pytest.mark.tiny
def test():
    test_name = 'meta-lr-bert-classification'
    test_process = TestProcess(test_name, configuration={
        'project': test_name,
        'pipeline_setup': 'FULL',
        'task': 'CLASSIFICATION',
        'dataset_alias': 'shivamb/legal-citation-text-classification',
        'dataset_file': 'legal_text_classification.csv',
        'dataset_table_columns': ['case_text', 'case_outcome'],
        'dataset_partition': 0,
        'dataset_need_balance': True,
        'model_alias': 'prajjwal1/bert-tiny'
    })

    test_process.selection({
        'additional_learning_rate': [1e-3, 1e-4],
        'epochs': [5, 10, 20]
    })
