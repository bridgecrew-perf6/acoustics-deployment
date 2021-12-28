import mlflow
import mlflow.sagemaker as mfs
import boto3

MLFLOW_TRACKING_URI="http://159.138.93.72"
MODEL_NAME = "issue-detection"
SAGEMAKER_MODEL_BUCKET = "carro-sagemaker-acoustics"
SAGEMAKER_EXECUTION_ROLE = "arn:aws:iam::302145289873:role/service-role/AmazonSageMaker-ExecutionRole-20211223T112159"
SAGEMAKER_PREFIX = "acoustics-issuedetection"


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow_client = mlflow.tracking.MlflowClient()
    version = mlflow_client.get_latest_versions(MODEL_NAME, ["production"])[0].version
    sm_model_name = f"{SAGEMAKER_PREFIX}-v{version}"
    

    # push model to sagemaker
    mfs.push_model_to_sagemaker(
            model_name=sm_model_name,
            model_uri=f"models:/{MODEL_NAME}/production",
            image_url="302145289873.dkr.ecr.ap-southeast-1.amazonaws.com/carro-acoustics-issuedetection:latest",
            execution_role_arn=SAGEMAKER_EXECUTION_ROLE,
            region_name="ap-southeast-1",
            bucket=SAGEMAKER_MODEL_BUCKET
            )


    boto_client = boto3.client("sagemaker", region_name="ap-southeast-1")
    # create endpoint config
    production_variant = {
            "InitialInstanceCount": 1,
            "InstanceType": "ml.m4.xlarge",
            "ModelName": sm_model_name,
            "VariantName": sm_model_name
            }

    boto_client.create_endpoint_config(
            EndpointConfigName=sm_model_name,
            ProductionVariants=[production_variant]
            )

    deployment_config={ 
          "BlueGreenUpdatePolicy": { 
                 "MaximumExecutionTimeoutInSeconds": 600,
                 "TerminationWaitInSeconds": 0,
                 "TrafficRoutingConfiguration": { 
                    "Type": "ALL_AT_ONCE",
                    "WaitIntervalInSeconds": 0
                 }
              }
           }

    # create endpoint
    boto_client.create_endpoint(
            EndpointName=f"carro-{SAGEMAKER_PREFIX}",
            EndpointConfigName=sm_model_name,
            DeploymentConfig=deployment_config
            )
    
if __name__ == "__main__":
    main()
