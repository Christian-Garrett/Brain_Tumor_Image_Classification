from pathlib import Path
import sys

module_path = Path(__file__).parents[1]
sys.path.append(str(module_path))

from ML_Pipeline import DataPipeline


dp_object = DataPipeline('Clean_Set_Updated')

dp_object.load_input_data()
dp_object.perform_EDA()
dp_object.perform_data_preprocessing()
dp_object.perform_data_wrangling()
dp_object.run_convolutional_model('Model_122324') 
dp_object.run_capsule_model('Cap_Model_010325')  
