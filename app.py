from flask import Flask, request, jsonify, make_response
from enum import Enum
import logging


from core.configuration.configuration import Configuration
from core.configuration.hyperparams import PipelineSetup
from core.main_process.main_process import MainProcess


app = Flask(__name__)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@app.route('/api/start', methods=['POST'])
def start_main_process():
    try:
        config = Configuration.get_instance()

        assert 'pipeline_setup' in request.json.keys(), 'request missing "pipeline_setup" field'
        value = request.json['pipeline_setup']

        assert isinstance(value, str), 'request field "pipeline_setup" is not <str> type'
        assert value.upper() in PipelineSetup.__members__, \
            f'invalid value "{value.upper()}" for enum field "PipelineSetup"'

        value = PipelineSetup[value.upper()]

        config.configure('pipeline_setup', value)

        logger.info('starting new thread for main process...')
        thread = MainProcess(value)
        thread.daemon = True
        thread.start()

        return make_response(jsonify({
            'success': True,
            'message': f'finetuning process started with pipeline setup "{value.name}"'
        }), 200)

    except AssertionError as e:
        return make_response(jsonify({
            'success': False,
            'error': str(e)
        }), 400)


@app.route('/api/update', methods=['POST'])
def update_configuration():
    try:
        config = Configuration.get_instance()

        assert 'field' in request.json.keys(), 'request missing "field" field'
        assert 'value' in request.json.keys(), 'request missing "value" field'

        field = request.json['field']
        value = request.json['value']

        assert isinstance(field, str), f'{request.json}: request field "field" is not <str> type'
        assert hasattr(config, field), f'{request.json}: there is no such field "{field}" in the configuration'
        assert getattr(config, field) is not None, f'{request.json}: configuring NoneType fields from API is forbidden'
        desired_class = getattr(config, field).__class__

        if issubclass(desired_class, Enum):
            assert isinstance(value, str), f'{request.json}: request field "value" is not <str> type'
            assert value.upper() in desired_class.__members__, \
                f'{request.json}: invalid "value" for enum field "{field}"'

            value = desired_class[value.upper()]

        else:
            assert isinstance(value, desired_class), \
                f'{request.json}: request field "value" is not {desired_class} type'

        forced = False
        if 'forced' in request.json.keys():
            forced = request.json['forced']
            assert isinstance(forced, bool), f'{request.json}: request field "forced" is not <bool> type'

        config.configure(field, value, forced)

        return make_response(jsonify({
            'success': True,
            'message': 'configuration updated'
        }), 200)

    except AssertionError as e:
        return make_response(jsonify({
            'success': False,
            'error': str(e)
        }), 400)


@app.route('/api/approve', methods=['POST'])
def approve_stage():
    config = Configuration.get_instance()
    config.configure_status('approved', True)
    return make_response(jsonify({
            'success': True,
            'message': 'stage approved'
        }), 200)


@app.route('/api/status', methods=['GET'])
def get_status():
    config = Configuration.get_instance()
    logger.info(f'current status: {config.status}')
    return make_response(jsonify(config.status), 200 if config.status['success'] else 500)
