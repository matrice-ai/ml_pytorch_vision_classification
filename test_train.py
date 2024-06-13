import pytest
import os
import sys
import yaml
import shutil
from unittest.mock import patch

# Add the path to the directory containing train.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'path_to_train_py_directory')))

import train

@pytest.fixture

def setup_env():
    """Set up any necessary environment variables or configurations."""
    os.makedirs('workspace', exist_ok=True)
    # Load test configuration
    with open('test_config.yaml', 'r') as config_file:
        config = yaml.safe_load(config_file)
        for key, value in config['hyperparameters'].items():
            os.environ[key.upper()] = str(value)
        global action_id
        action_id = config['action_id']
        os.environ['TRAIN_DATA_PATH'] = config['data_paths']['train']
        os.environ['VAL_DATA_PATH'] = config['data_paths']['val']
        os.environ['TEST_DATA_PATH'] = config['data_paths']['test']
        global model_config
        model_config = config['hyperparameters']
    yield
    # Cleanup after test
    shutil.rmtree('workspace')

def test_train_model(setup_env):
    """Test if the model can be trained without errors."""
    if action_id == 'test_action_id':
        pytest.fail('Invalid action ID')

    with patch.object(train.Session, 'get_job_params', return_value=model_config):
        with patch.object(train.ActionTracker, 'add_index_to_category') as mock_add_index_to_category, \
             patch.object(train.ActionTracker, 'update_status') as mock_update_status, \
             patch.object(train.ActionTracker, 'log_epoch_results') as mock_log_epoch_results, \
             patch.object(train.ActionTracker, 'upload_checkpoint') as mock_upload_checkpoint, \
             patch('builtins.print'):  # Suppress print statements
            try:
                train.main(action_id)
            except Exception as e:
                pytest.fail(f"Training failed: {e}")    

    # Verify if the status updates were called appropriately
    mock_update_status.assert_any_call('MDL_TRN_ACK', 'OK', 'Model Training has acknowledged')
    mock_update_status.assert_any_call('MDL_TRN_DTL', 'OK', 'Training Dataset is loaded')
    mock_update_status.assert_any_call('MDL_TRN_STRT', 'OK', 'Model Training has started')
    mock_update_status.assert_any_call('MDL_TRN_CMPL', 'SUCCESS', 'Model Training is completed')

    # Verify if the model checkpoints were attempted to be uploaded
    mock_upload_checkpoint.assert_called()
