from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessing
from src.model_training import ModelTraining
from src.feature_store import RedisFeatureStore
from config.paths_config import *
from config.database_config import *


if __name__ == "__main__":
    
        data_ingestion = DataIngestion(DB_CONFIG,RAW_DIR)
        data_ingestion.run()
        feature_store = RedisFeatureStore()
        datapreprocessing = DataProcessing(TRAIN_PATH,TEST_PATH,feature_store)
        datapreprocessing.run()
        #print(datapreprocessing.retrive_feature_redis_store(entity_id=332))
        
        model_trianer = ModelTraining(feature_store)
        model_trianer.run()
